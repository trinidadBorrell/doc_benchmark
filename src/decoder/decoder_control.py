#!/usr/bin/env python3
"""
Script to decode original vs reconstructed EEG data from .fif files for control subjects.
This version does not filter by subject type (patients vs controls) but allows optional
filtering by session types (trial types like LSGS, LSGD, LDGD, LDGS).

Original files (-original.fif) are labeled as class 0.
Reconstructed files (-recon.fif) are labeled as class 1.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
from datetime import datetime
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy import stats

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

# Event ID mapping for trial types
EVENT_ID_MAPPING = {
    10: 'HSTD',   # Control
    20: 'HDVT',   # Control 
    30: 'LSGS',   # Local Standard Global Standard 
    40: 'LSGD',   # Local Standard Global Deviant
    50: 'LDGD',   # Local Deviant Global Deviant
    60: 'LDGS'    # Local Deviant Global Standard
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode original vs reconstructed EEG data from .fif files (control version)"
    )
    parser.add_argument(
        "--main_path", 
        type=str, 
        required=True,
        help="Main path containing subject directories (e.g., /data/eeg_study)"
    )
    parser.add_argument(
        "--subjects", 
        type=str, 
        nargs="+",
        help="List of subject IDs (e.g., 01 02 03). If not provided, will auto-detect."
    )
    parser.add_argument(
        "--sessions", 
        type=str, 
        nargs="+",
        help="List of session IDs (e.g., 01 02). If not provided, will auto-detect."
    )
    parser.add_argument(
        "--filter_trial_types",
        type=str,
        nargs="+",
        choices=['HSTD', 'HDVT', 'LSGS', 'LSGD', 'LDGD', 'LDGS'],
        help="Optional: Filter to include only specific trial types (e.g., LSGS LSGD LDGD LDGS)"
    )
    parser.add_argument(
        "--cv", 
        type=int, 
        default=10,
        help="Number of cross-validation folds (default: 10)"
    )
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=None,
        help="Number of parallel jobs (-1 for all cores, default: None)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./doc_benchmark/results/DECODER_CONTROL_LocalGlobal",
        help="Directory to save results (default: ./doc_benchmark/results/DECODER_CONTROL)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run aggregation on existing results (skip subject processing)"
    )
    
    return parser.parse_args()


def find_subjects_sessions(main_path):
    """Auto-detect subjects and sessions from directory structure."""
    subjects = []
    sessions = []
    
    main_path = Path(main_path)
    if not main_path.exists():
        raise ValueError(f"Main path does not exist: {main_path}")
    
    # Find all sub-* directories
    for sub_dir in main_path.glob("sub-*"):
        if sub_dir.is_dir():
            sub_id = sub_dir.name.replace("sub-", "")
            subjects.append(sub_id)
            
            # Find all sess-* directories within this subject
            for sess_dir in sub_dir.glob("ses-*"):
                if sess_dir.is_dir():
                    sess_id = sess_dir.name.replace("ses-", "")
                    if sess_id not in sessions:
                        sessions.append(sess_id)
    
    subjects = sorted(list(set(subjects)))
    sessions = sorted(list(set(sessions)))
    
    return subjects, sessions


def load_epochs_single_subject_session(main_path, subject_id, session_id, 
                                       filter_trial_types=None, verbose=False):
    """
    Load epochs from .fif files for a single subject and session.
    
    Parameters:
        main_path: str, path to main data directory
        subject_id: str, subject identifier
        session_id: str, session identifier
        filter_trial_types: list of str or None, trial types to include (e.g., ['LSGS', 'LSGD'])
        verbose: bool, print verbose output
    
    Returns:
        X: epochs data (n_epochs, n_channels, n_timepoints)
        y: labels (n_epochs,) - 0 for original, 1 for reconstructed
        events: event IDs (n_epochs,)
        times: time points array
    """
    main_path = Path(main_path)
    sub_ses_dir = main_path / f"sub-{subject_id}" / f"ses-{session_id}"
    
    if not sub_ses_dir.exists():
        if verbose:
            print(f"Directory not found: {sub_ses_dir}")
        return None, None, None, None
    
    # Find original and reconstructed files
    original_files = list(sub_ses_dir.glob("*_epo_original.fif"))
    recon_files = list(sub_ses_dir.glob("*_epo_recon.fif"))
    
    if verbose:
        print(f"Subject {subject_id}, Session {session_id}:")
        print(f"  Original files: {len(original_files)}")
        print(f"  Reconstructed files: {len(recon_files)}")
        if filter_trial_types:
            print(f"  Filtering to trial types: {filter_trial_types}")
    
    # Get event IDs for filtering if specified
    filter_event_ids = None
    if filter_trial_types:
        filter_event_ids = [event_id for event_id, name in EVENT_ID_MAPPING.items() 
                           if name in filter_trial_types]
    
    all_epochs = []
    all_labels = []
    all_events = []
    times = None
    
    # Process original files (label 0)
    for fif_file in original_files:
        try:
            epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
            data = epochs.get_data().astype(np.float32)
            events = epochs.events[:, 2]
            
            # Apply trial type filtering if specified
            if filter_event_ids is not None:
                mask = np.isin(events, filter_event_ids)
                data = data[mask]
                events = events[mask]
            
            if len(data) > 0:
                n_epochs = data.shape[0]
                all_epochs.append(data)
                all_labels.extend([0] * n_epochs)
                all_events.extend(events)
                
                if times is None:
                    times = epochs.times
                
                if verbose:
                    print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs")
                    
        except Exception as e:
            print(f"    Error loading {fif_file}: {e}")
    
    # Process reconstructed files (label 1)
    for fif_file in recon_files:
        try:
            epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
            data = epochs.get_data().astype(np.float32)
            events = epochs.events[:, 2]
            
            # Apply trial type filtering if specified
            if filter_event_ids is not None:
                mask = np.isin(events, filter_event_ids)
                data = data[mask]
                events = events[mask]
            
            if len(data) > 0:
                n_epochs = data.shape[0]
                all_epochs.append(data)
                all_labels.extend([1] * n_epochs)
                all_events.extend(events)
                
                if times is None:
                    times = epochs.times
                
                if verbose:
                    print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs")
                    
        except Exception as e:
            print(f"    Error loading {fif_file}: {e}")
    
    if not all_epochs:
        return None, None, None, None
    
    # Concatenate all data
    X = np.concatenate(all_epochs, axis=0)
    y = np.array(all_labels)
    events = np.array(all_events)
    
    if verbose:
        print(f"  Total epochs: {X.shape[0]}")
        print(f"  Original epochs: {np.sum(y == 0)}")
        print(f"  Reconstructed epochs: {np.sum(y == 1)}")
        print(f"  Event types: {np.unique(events)}")
    
    return X, y, events, times


def decode_single_subject_session(X, y, events, times, cv=10, n_jobs=None, verbose=False):
    """
    Perform decoding analysis for a single subject/session.
    
    Parameters:
        X: array, epochs data (n_epochs, n_channels, n_timepoints)
        y: array, labels (0=original, 1=reconstructed)
        events: array, event IDs
        times: array, time points
        cv: int, number of cross-validation folds
        n_jobs: int or None, number of parallel jobs
        verbose: bool, print verbose output
    
    Returns:
        results: dict containing analysis results
    """
    if X is None or len(np.unique(y)) < 2:
        return None
    
    if X.shape[0] < cv:
        print(f"  WARNING: Too few trials ({X.shape[0]}) for {cv}-fold CV")
        return None
    
    results = {}
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Overall classification
    print("  Performing overall classification...")
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
    try:
        scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
        if scores.ndim == 2:
            mean_scores_time = np.mean(scores, axis=0)
        else:
            mean_scores_time = scores
        
        results['overall'] = {
            'scores': scores,
            'mean_scores_time': mean_scores_time,
            'mean_auc': np.mean(mean_scores_time),
            'std_auc': np.std(mean_scores_time)
        }
        print(f"    Overall AUC: {results['overall']['mean_auc']:.3f}")
    except Exception as e:
        print(f"    Error in overall classification: {e}")
        return None
    
    # Trial-type-specific classification
    print("  Performing trial-type classification...")
    target_trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    target_event_ids = [30, 40, 50, 60]
    
    results['trial_types'] = {}
    for trial_type, event_id in zip(target_trial_types, target_event_ids):
        trial_mask = events == event_id
        if np.sum(trial_mask) == 0:
            continue
        
        X_trial = X[trial_mask]
        y_trial = y[trial_mask]
        
        if X_trial.shape[0] < cv or len(np.unique(y_trial)) < 2:
            continue
        
        try:
            time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
            scores = cross_val_multiscore(time_decod, X_trial, y_trial, cv=cv, n_jobs=1)
            
            if scores.ndim == 2:
                mean_scores_time = np.mean(scores, axis=0)
            else:
                mean_scores_time = scores
            
            results['trial_types'][trial_type] = {
                'scores': scores,
                'mean_scores_time': mean_scores_time,
                'mean_auc': np.mean(mean_scores_time),
                'std_auc': np.std(mean_scores_time),
                'n_trials': X_trial.shape[0]
            }
            print(f"    {trial_type} AUC: {results['trial_types'][trial_type]['mean_auc']:.3f}")
        except Exception as e:
            print(f"    Error in {trial_type} classification: {e}")
    
    return results


def save_single_subject_session_results(results, times, subject_id, session_id, output_dir):
    """Save results for a single subject/session."""
    # Create output directory
    sub_ses_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_id}"
    plots_dir = sub_ses_dir / "plots"
    data_dir = sub_ses_dir / "data"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results data
    with open(data_dir / "decoding_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save times array
    np.save(data_dir / "times.npy", times)
    
    # Save JSON summary
    json_results = {
        "subject_id": subject_id,
        "session_id": session_id,
        "analysis_date": datetime.now().isoformat(),
        "overall": {
            "mean_auc": float(results['overall']['mean_auc']),
            "std_auc": float(results['overall']['std_auc'])
        } if 'overall' in results else None,
        "trial_types": {}
    }
    
    if 'trial_types' in results:
        for trial_type, trial_data in results['trial_types'].items():
            json_results["trial_types"][trial_type] = {
                "mean_auc": float(trial_data['mean_auc']),
                "std_auc": float(trial_data['std_auc']),
                "n_trials": int(trial_data['n_trials'])
            }
    
    with open(data_dir / "decoding_summary.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create plots
    create_single_subject_plots(results, times, plots_dir, subject_id, session_id)
    
    print(f"  Results saved to: {sub_ses_dir}")
    return sub_ses_dir


def add_stimulus_lines(ax, times):
    """Add vertical lines at stimulus times to a plot."""
    stimulus_times = [0.0, 0.15, 0.3, 0.45, 0.6]
    for i, stim_time in enumerate(stimulus_times):
        if times[0] <= stim_time <= times[-1]:
            label = 'Stimuli' if i == 0 else None
            ax.axvline(stim_time, color='darkgray', linestyle='--', alpha=1, linewidth=1.5, label=label)


def create_single_subject_plots(results, times, plots_dir, subject_id, session_id):
    """Create plots for single subject/session results."""
    
    # Overall classification plot
    if 'overall' in results:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        mean_scores = results['overall']['mean_scores_time']
        ax.plot(times, mean_scores, 'b-', linewidth=2, label='AUC')
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
        add_stimulus_lines(ax, times)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AUC")
        ax.set_title(f"Overall Classification - sub-{subject_id} ses-{session_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        
        plt.tight_layout()
        plt.savefig(plots_dir / "overall_classification.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Trial type plots
    if 'trial_types' in results and results['trial_types']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, trial_type in enumerate(trial_types):
            if trial_type in results['trial_types']:
                mean_scores = results['trial_types'][trial_type]['mean_scores_time']
                axes[i].plot(times, mean_scores, color=colors[i], linewidth=2)
                axes[i].axhline(0.5, color="k", linestyle="--", alpha=0.7)
                add_stimulus_lines(axes[i], times)
                axes[i].set_title(f"{trial_type}")
                axes[i].set_xlabel("Time (s)")
                axes[i].set_ylabel("AUC")
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim([0.4, 1.0])
            else:
                axes[i].text(0.5, 0.5, f'No data for {trial_type}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{trial_type}")
        
        plt.suptitle(f"Trial Type Classification - sub-{subject_id} ses-{session_id}")
        plt.tight_layout()
        plt.savefig(plots_dir / "trial_type_classification.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_permutation_test(all_scores, n_permutations=1000, seed=42):
    """Run permutation test to assess significance of decoding results."""
    np.random.seed(seed)
    
    # Observed mean AUC
    observed_mean = np.mean(all_scores)
    
    # One-sample t-test against chance (0.5)
    mean_auc_per_subject = np.mean(all_scores, axis=1)
    t_statistic, p_value_ttest = stats.ttest_1samp(mean_auc_per_subject, 0.5, alternative='greater')
    
    # Permutation test
    null_means = []
    centered_scores = all_scores - mean_auc_per_subject[:, np.newaxis] + 0.5
    
    for _ in range(n_permutations):
        indices = np.random.randint(0, all_scores.shape[0], size=all_scores.shape[0])
        permuted_scores = centered_scores[indices, :]
        null_means.append(np.mean(permuted_scores))
    
    null_means = np.array(null_means)
    p_value_perm = np.mean(null_means >= observed_mean)
    
    # Effect size
    null_mean = np.mean(null_means)
    null_std = np.std(null_means)
    cohens_d = (observed_mean - 0.5) / np.std(mean_auc_per_subject) if np.std(mean_auc_per_subject) > 0 else 0
    
    return {
        'observed_mean': float(observed_mean),
        'p_value_permutation': float(p_value_perm),
        'p_value_ttest': float(p_value_ttest),
        't_statistic': float(t_statistic),
        'cohens_d': float(cohens_d),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'n_permutations': int(n_permutations),
        'n_subjects': int(all_scores.shape[0]),
        'significant_permutation': bool(p_value_perm < 0.05),
        'significant_ttest': bool(p_value_ttest < 0.05)
    }


def aggregate_all_results(output_dir):
    """Aggregate results from all individual subject/session analyses."""
    print("\nAggregating results from all subjects and sessions...")
    
    output_path = Path(output_dir)
    
    # Find all individual results
    all_results = []
    subjects_sessions = []
    times = None
    
    for sub_dir in output_path.glob("sub-*"):
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name.replace("sub-", "")
        
        for ses_dir in sub_dir.glob("ses-*"):
            if not ses_dir.is_dir():
                continue
            session_id = ses_dir.name.replace("ses-", "")
            
            # Load results
            results_path = ses_dir / "data" / "decoding_results.pkl"
            times_path = ses_dir / "data" / "times.npy"
            
            if results_path.exists():
                try:
                    with open(results_path, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Load times
                    if times is None and times_path.exists():
                        times = np.load(times_path)
                    
                    all_results.append(results)
                    subjects_sessions.append((subject_id, session_id))
                    print(f"  Loaded: sub-{subject_id} ses-{session_id}")
                    
                except Exception as e:
                    print(f"  Error loading sub-{subject_id} ses-{session_id}: {e}")
    
    if not all_results:
        print("No individual results found for aggregation!")
        return None, None, []
    
    print(f"Found {len(all_results)} individual results to aggregate")
    
    # Initialize aggregated results
    aggregated = {
        'overall': {
            'all_mean_aucs': [], 
            'all_mean_scores_time': []
        },
        'trial_types': {}
    }
    
    # Aggregate overall results
    for results in all_results:
        if 'overall' in results:
            aggregated['overall']['all_mean_aucs'].append(results['overall']['mean_auc'])
            aggregated['overall']['all_mean_scores_time'].append(results['overall']['mean_scores_time'])
    
    # Calculate overall statistics
    if aggregated['overall']['all_mean_aucs']:
        aggregated['overall']['mean_auc'] = np.mean(aggregated['overall']['all_mean_aucs'])
        aggregated['overall']['std_auc'] = np.std(aggregated['overall']['all_mean_aucs'])
        
        # Point-by-point statistics
        all_timeseries = np.array(aggregated['overall']['all_mean_scores_time'])
        print(f'All timeseries shape: {all_timeseries.shape}')
        
        # Save timeseries data
        timeseries_dir = output_path / "all_auc_timeseries"
        timeseries_dir.mkdir(exist_ok=True)
        np.save(timeseries_dir / 'all_mean_scores_time.npy', all_timeseries)
        
        aggregated['overall']['mean_scores_time'] = np.mean(all_timeseries, axis=0)
        aggregated['overall']['std_scores_time'] = np.std(aggregated['overall']['all_mean_scores_time'], axis=0)
        aggregated['overall']['sem_scores_time'] = aggregated['overall']['std_scores_time'] / np.sqrt(len(all_timeseries))
    
    # Aggregate trial type results
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    for trial_type in trial_types:
        aggregated['trial_types'][trial_type] = {
            'all_mean_aucs': [],
            'all_mean_scores_time': []
        }
        
        for results in all_results:
            if 'trial_types' in results and trial_type in results['trial_types']:
                aggregated['trial_types'][trial_type]['all_mean_aucs'].append(
                    results['trial_types'][trial_type]['mean_auc']
                )
                aggregated['trial_types'][trial_type]['all_mean_scores_time'].append(
                    results['trial_types'][trial_type]['mean_scores_time']
                )
        
        # Save trial type timeseries data
        if aggregated['trial_types'][trial_type]['all_mean_scores_time']:
            np.save(timeseries_dir / f'all_mean_scores_time_{trial_type}.npy', 
                   aggregated['trial_types'][trial_type]['all_mean_scores_time'])
        
        # Calculate statistics for this trial type
        if aggregated['trial_types'][trial_type]['all_mean_aucs']:
            aggregated['trial_types'][trial_type]['mean_auc'] = np.mean(
                aggregated['trial_types'][trial_type]['all_mean_aucs']
            )
            aggregated['trial_types'][trial_type]['std_auc'] = np.std(
                aggregated['trial_types'][trial_type]['all_mean_aucs']
            )
            
            # Point-by-point statistics
            all_timeseries = np.array(aggregated['trial_types'][trial_type]['all_mean_scores_time'])
            aggregated['trial_types'][trial_type]['mean_scores_time'] = np.mean(all_timeseries, axis=0)
            aggregated['trial_types'][trial_type]['std_scores_time'] = np.std(
                aggregated['trial_types'][trial_type]['all_mean_scores_time'], axis=0
            )
            aggregated['trial_types'][trial_type]['sem_scores_time'] = (
                aggregated['trial_types'][trial_type]['std_scores_time'] / np.sqrt(len(all_timeseries))
            )
            aggregated['trial_types'][trial_type]['n_subjects_sessions'] = len(all_timeseries)
            
            # Run permutation test
            print(f"  Running permutation test for {trial_type}...")
            perm_result = run_permutation_test(all_timeseries, n_permutations=1000, seed=42)
            aggregated['trial_types'][trial_type]['permutation_test'] = perm_result
            print(f"    Observed mean AUC: {perm_result['observed_mean']:.4f}")
            print(f"    p-value (permutation): {perm_result['p_value_permutation']:.4f}")
            print(f"    p-value (t-test): {perm_result['p_value_ttest']:.4f}")
            print(f"    Significant: {perm_result['significant_ttest']}")
    
    return aggregated, times, subjects_sessions


def save_aggregated_results(aggregated_results, times, subjects_sessions, output_dir):
    """Save aggregated results to decoding-control-global-{date} directory."""
    
    # Create global results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_dir = Path(output_dir).parent / f"decoding-control-global-{timestamp}"
    plots_dir = global_dir / "plots"
    data_dir = global_dir / "data"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving aggregated results to: {global_dir}")
    
    # Save raw aggregated results
    with open(data_dir / "aggregated_results.pkl", 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    # Save times array
    np.save(data_dir / "times.npy", times)
    
    # Save JSON summary
    json_results = {
        "analysis_date": datetime.now().isoformat(),
        "n_subjects_sessions": len(subjects_sessions),
        "subjects_sessions": [{"subject_id": sub, "session_id": ses} for sub, ses in subjects_sessions],
        "overall_classification": {},
        "trial_type_classification": {}
    }
    
    # Add overall results
    if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
        json_results["overall_classification"] = {
            "mean_auc": float(aggregated_results['overall']['mean_auc']),
            "std_auc": float(aggregated_results['overall']['std_auc'])
        }
    
    # Add trial type results
    for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
        if (trial_type in aggregated_results['trial_types'] and 
            'mean_auc' in aggregated_results['trial_types'][trial_type]):
            trial_data = aggregated_results['trial_types'][trial_type]
            json_results["trial_type_classification"][trial_type] = {
                "mean_auc": float(trial_data['mean_auc']),
                "std_auc": float(trial_data['std_auc']),
                "n_subjects_sessions": int(trial_data['n_subjects_sessions'])
            }
            
            # Add permutation test results if available
            if 'permutation_test' in trial_data:
                perm = trial_data['permutation_test']
                json_results["trial_type_classification"][trial_type]['permutation_test'] = perm
    
    # Save JSON
    with open(data_dir / "aggregated_summary.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create aggregated plots
    create_aggregated_plots(aggregated_results, times, plots_dir)
    
    print(f"Aggregated results saved to: {global_dir}")
    return global_dir


def create_aggregated_plots(aggregated_results, times, plots_dir):
    """Create plots for aggregated results."""
    
    # Overall classification plot
    if 'overall' in aggregated_results and 'mean_scores_time' in aggregated_results['overall']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        mean_scores = aggregated_results['overall']['mean_scores_time']
        std_scores = aggregated_results['overall']['std_scores_time']
        n_sessions = len(aggregated_results['overall']['all_mean_aucs'])
        
        # Plot mean with SD error band
        ax.plot(times, mean_scores, 'b-', linewidth=3, label=f'Mean AUC (n={n_sessions} sessions)')
        ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                       alpha=0.3, color='blue', label='±1 SD')
        
        # Reference lines
        ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
        add_stimulus_lines(ax, times)
        
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel("AUC", fontsize=16)
        ax.set_title("Aggregated Overall Classification: Original vs Reconstructed EEG", fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        
        # Add peak info
        peak_idx = np.argmax(mean_scores)
        peak_auc = mean_scores[peak_idx]
        peak_time = times[peak_idx]
        ax.text(0.02, 0.98, f'Peak: AUC={peak_auc:.3f} at t={peak_time:.3f}s', 
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plots_dir / "aggregated_overall_classification.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Trial type plots
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, trial_type in enumerate(trial_types):
        if (trial_type in aggregated_results['trial_types'] and 
            'mean_scores_time' in aggregated_results['trial_types'][trial_type]):
            
            data = aggregated_results['trial_types'][trial_type]
            mean_scores = data['mean_scores_time']
            std_scores = data['std_scores_time']
            mean_auc = data['mean_auc']
            std_auc = data['std_auc']
            
            # Plot line with error band
            ax.plot(times, mean_scores, color=colors[i], linewidth=2.5, 
                   label=f'{trial_type}: {mean_auc:.3f} ± {std_auc:.3f}')
            ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                           alpha=0.1, color=colors[i])
    
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5, label="Chance")
    add_stimulus_lines(ax, times)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("AUC", fontsize=18)
    ax.set_title("Decoding per Trial Type", fontsize=20)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_trial_type_classification.png", dpi=300, bbox_inches='tight')
    plt.close()


def is_decoder_result_complete(output_dir, subject_id, session_id):
    """Check if decoder results already exist for a subject/session."""
    output_path = Path(output_dir)
    sub_ses_dir = output_path / f"sub-{subject_id}" / f"ses-{session_id}"
    
    # Check for required files
    required_files = [
        sub_ses_dir / "data" / "decoding_results.pkl",
        sub_ses_dir / "data" / "decoding_summary.json",
        sub_ses_dir / "data" / "times.npy"
    ]
    
    # Check if all required files exist
    return all(f.exists() for f in required_files)


def main():
    """Main function."""
    args = parse_arguments()
    
    print("EEG Original vs Reconstructed Decoder - Control Version")
    print("=" * 60)
    
    # If aggregate-only mode, skip subject processing
    if args.aggregate_only:
        print("AGGREGATE-ONLY MODE: Aggregating existing results...")
        print(f"Looking for results in: {args.output_dir}")
        
        aggregated_results, times, subjects_sessions = aggregate_all_results(args.output_dir)
        
        if aggregated_results is None:
            print("ERROR: Failed to aggregate results! No completed subjects found.")
            return
        
        # Save aggregated results
        global_dir = save_aggregated_results(aggregated_results, times, subjects_sessions, args.output_dir)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATION COMPLETED!")
        print(f"Individual results source: {args.output_dir}")
        print(f"Aggregated results: {global_dir}")
        print(f"{'='*60}")
        
        return
    
    # Auto-detect subjects and sessions if not provided
    if args.subjects is None or args.sessions is None:
        print(f"Auto-detecting subjects and sessions from: {args.main_path}")
        detected_subjects, detected_sessions = find_subjects_sessions(args.main_path)
        
        subjects = args.subjects if args.subjects is not None else detected_subjects
        sessions = args.sessions if args.sessions is not None else detected_sessions
        
        print(f"Detected subjects: {subjects}")
        print(f"Detected sessions: {sessions}")
    else:
        subjects = args.subjects
        sessions = args.sessions
    
    if not subjects or not sessions:
        raise ValueError("No subjects or sessions found/specified!")
    
    print(f"\nProcessing {len(subjects)} subjects × {len(sessions)} sessions")
    if args.filter_trial_types:
        print(f"Filtering to trial types: {args.filter_trial_types}")
    
    # Sequential processing
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for subject_id in subjects:
        for session_id in sessions:
            print(f"\n{'='*50}")
            print(f"Processing sub-{subject_id} ses-{session_id}")
            print(f"{'='*50}")
            
            # Check if results already exist
            if is_decoder_result_complete(args.output_dir, subject_id, session_id):
                print(f"  ⏭️  Skipping - results already exist")
                skipped_count += 1
                processed_count += 1
                continue
            
            # Load data
            X, y, events, times = load_epochs_single_subject_session(
                args.main_path, subject_id, session_id, 
                filter_trial_types=args.filter_trial_types,
                verbose=args.verbose
            )
            
            if X is None:
                print(f"  No data found")
                failed_count += 1
                continue
            
            # Perform decoding
            results = decode_single_subject_session(
                X, y, events, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
            )
            
            if results is None:
                print(f"  Decoding failed")
                failed_count += 1
                continue
            
            # Save results
            save_single_subject_session_results(
                results, times, subject_id, session_id, args.output_dir
            )
            
            processed_count += 1
            print(f"  ✓ Successfully processed")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {processed_count - skipped_count}")
    print(f"Skipped (already complete): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total complete: {processed_count}")
    print(f"{'='*60}")
    
    if processed_count == 0:
        print("ERROR: No subject-sessions were successfully processed!")
        return
    
    # Aggregate all results
    print(f"\nAggregating results from {processed_count} subject-sessions...")
    aggregated_results, times, subjects_sessions = aggregate_all_results(args.output_dir)
    
    if aggregated_results is None:
        print("ERROR: Failed to aggregate results!")
        return
    
    # Save aggregated results
    global_dir = save_aggregated_results(aggregated_results, times, subjects_sessions, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETED!")
    print(f"Individual results: {args.output_dir}")
    print(f"Aggregated results: {global_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
