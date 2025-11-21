#!/usr/bin/env python3
"""
Unified script to decode original vs reconstructed EEG data from .fif files.
Original files (-original.fif) are labeled as class 0.
Reconstructed files (-recon.fif) are labeled as class 1.

Supports two modes:
- 'patient' mode: Overall decoding + optional trial-type decomposition with patient state grouping
- 'control' mode: Overall decoding + optional trial-type decomposition (disabled for resting state data)

Examples:

# Patient mode (default)
python decoder.py --main_path /data/patients --patient-labels labels.csv

# Control mode  
python decoder.py --mode control --main_path /data/controls

# Control mode with filtering (task-based data)
python decoder.py --mode control --filter_trial_types LSGS LSGD LDGD LDGS

# Control mode with resting state (no trial decomposition)
python decoder.py --mode control --main_path /data/controls_rest
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from datetime import datetime
import pickle
import pandas as pd
from scipy import stats

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

# Event ID mapping for trial types
event_id_mapping = {
    # Based on nice/algorithms/information_theory/tests/test_komplexity.py mapping
    # Main experimental conditions (Local/Standard, Global/Standard, etc.)
    10: 'HSTD',   # Control
    20: 'HDVT',   # Control 
    30: 'LSGS',    # Local Standard Global Standard 
    40: 'LSGD',    # Local Standard Global Deviant
    50: 'LDGD',    # Local Deviant Global Deviant
    60: 'LDGS'     # Local Deviant Global Standard
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode original vs reconstructed EEG data from .fif files"
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
        "--cv", 
        type=int, 
        default=10,
        help="Number of cross-validation folds (default: 10). As in JRK et al 2013"
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
        default="./doc_benchmark/results/DECODER",
        help="Directory to save results (default: ./doc_benchmark/results/DECODER)"
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
    parser.add_argument(
        "--patient-labels",
        type=str,
        default=None,
        help="Path to patient labels CSV file for subject-type aggregation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['patient', 'control'],
        default='patient',
        help="Analysis mode: 'patient' for patient data with state grouping, 'control' for simplified control analysis"
    )
    parser.add_argument(
        "--filter_trial_types",
        type=str,
        nargs="+",
        choices=['HSTD', 'HDVT', 'LSGS', 'LSGD', 'LDGD', 'LDGS'],
        help="Optional: Filter to include only specific trial types (typically used in control mode)"
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

def load_epochs_single_subject_session(main_path, subject_id, session_id, filter_trial_types=None, verbose=False):
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
        filter_event_ids = [event_id for event_id, name in event_id_mapping.items() 
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


def decode_single_subject_session(X, y, events, times, cv=10, n_jobs=None, mode='patient', verbose=False):
    """
    Perform all decoding analyses for a single subject/session.
    
    Parameters:
        X: array, epochs data (n_epochs, n_channels, n_timepoints)
        y: array, labels (0=original, 1=reconstructed)
        events: array, event IDs
        times: array, time points
        cv: int, number of cross-validation folds
        n_jobs: int or None, number of parallel jobs
        mode: str, 'patient' or 'control' - determines whether to run local/global analysis
        verbose: bool, print verbose output
    
    Returns:
        results: dict containing all analysis results
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
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=True)
    try:
        scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs, verbose=1)
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
    
    # Trial-type-specific classification (optional - depends on data type)
    # Check if data has task-specific events (local/global paradigm) or is resting state
    task_event_ids = [30, 40, 50, 60]  # LSGS, LSGD, LDGD, LDGS
    has_task_events = np.any(np.isin(events, task_event_ids))
    
    results['trial_types'] = {}
    results['has_task_events'] = has_task_events
    
    if has_task_events:
        print("  Performing trial-type classification...")
        target_trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
        target_event_ids = [30, 40, 50, 60]
        
        for trial_type, event_id in zip(target_trial_types, target_event_ids):
            trial_mask = events == event_id
            if np.sum(trial_mask) == 0:
                continue
            
            X_trial = X[trial_mask]
            y_trial = y[trial_mask]
            
            if X_trial.shape[0] < cv or len(np.unique(y_trial)) < 2:
                continue
            
            try:
                time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=True)
                scores = cross_val_multiscore(time_decod, X_trial, y_trial, cv=cv, n_jobs=n_jobs, verbose=1)
                
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
    else:
        print("  Skipping trial-type classification (resting state data detected)")
    
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
    results_path = data_dir / "decoding_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save JSON summary
    json_results = {
        "subject_id": subject_id,
        "session_id": session_id,
        "analysis_date": datetime.now().isoformat(),
        "has_task_events": bool(results.get('has_task_events', False)),
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
    
    json_path = data_dir / "decoding_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create plots
    create_single_subject_plots(results, times, plots_dir, subject_id, session_id)
    
    print(f"  Results saved to: {sub_ses_dir}")
    return sub_ses_dir

def add_stimulus_lines(ax, times):
    """Add vertical lines at stimulus times to a plot."""
    stimulus_times = [0.0, 0.15, 0.3, 0.45, 0.6]
    for i, stim_time in enumerate(stimulus_times):
        if times[0] <= stim_time <= times[-1]:  # Only plot if within time range
            label = 'Stimuli' if i == 0 else None
            ax.axvline(stim_time, color='darkgray', linestyle='--', alpha=1, linewidth=1.5, label=label)

def run_wilcoxon_per_timepoint(all_scores, alternative='greater'):
    """
    Run Wilcoxon signed-rank test per timepoint against chance (0.5).
    
    For each timepoint, tests whether the distribution of AUC values across subjects
    is significantly different from 0.5 (chance level).
    
    Parameters:
        all_scores: array of shape (n_subjects, n_timepoints) containing AUC scores
        alternative: str, 'greater', 'less', or 'two-sided'
        
    Returns:
        dict containing:
            - p_values: array of p-values per timepoint (n_timepoints,)
            - test_statistics: array of test statistics per timepoint
            - significant_mask: boolean array indicating p < 0.05 per timepoint
            - n_subjects: number of subjects
            - n_timepoints: number of timepoints
    """
    n_subjects, n_timepoints = all_scores.shape
    p_values = np.zeros(n_timepoints)
    test_statistics = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Get AUC values for this timepoint across all subjects
        auc_at_t = all_scores[:, t]
        
        # Wilcoxon signed-rank test against 0.5
        # Tests if the median of (auc_at_t - 0.5) is significantly different from 0
        try:
            statistic, p_value = stats.wilcoxon(auc_at_t - 0.5, alternative=alternative)
            p_values[t] = p_value
            test_statistics[t] = statistic
        except Exception as e:
            # If test fails (e.g., all values equal), set p-value to 1
            p_values[t] = 1.0
            test_statistics[t] = np.nan
    
    return {
        'p_values': p_values,
        'test_statistics': test_statistics,
        'significant_mask': p_values < 0.05,
        'n_subjects': int(n_subjects),
        'n_timepoints': int(n_timepoints)
    }

def run_permutation_test(all_scores, n_permutations=1000, seed=42):
    """
    Run permutation test to assess significance of decoding results.
    
    Tests whether mean AUC across subjects is significantly different from chance (0.5).
    Uses a permutation approach where we randomly sample from subjects with replacement
    and compute the null distribution of mean AUC values.
    
    Parameters:
        all_scores: array of shape (n_subjects_sessions, n_timepoints) containing AUC scores
        n_permutations: number of permutations to run
        seed: random seed for reproducibility
        
    Returns:
        dict containing:
            - observed_mean: observed mean AUC across all subjects/sessions
            - p_value_permutation: permutation test p-value (proportion null >= observed)
            - p_value_ttest: one-sample t-test p-value against 0.5
            - t_statistic: t-statistic from t-test
            - null_mean: mean of null distribution
            - null_std: std of null distribution
    """
    np.random.seed(seed)
    
    # Observed mean AUC (across all subjects/sessions and timepoints)
    observed_mean = np.mean(all_scores)
    
    # Method 1: Parametric test - one-sample t-test against chance (0.5)
    # This tests if the mean AUC per subject is significantly different from 0.5
    mean_auc_per_subject = np.mean(all_scores, axis=1)  # Average across timepoints for each subject
    t_statistic, p_value_ttest = stats.ttest_1samp(mean_auc_per_subject, 0.5, alternative='greater')
    
    # Method 2: Permutation test - bootstrap resampling under null hypothesis
    # Under H0: each subject's AUC is drawn from a distribution centered at 0.5
    # We create null by centering each subject at 0.5, then resampling
    null_means = []
    centered_scores = all_scores - mean_auc_per_subject[:, np.newaxis] + 0.5
    
    for perm_idx in range(n_permutations):
        # Randomly sample subjects with replacement
        indices = np.random.randint(0, all_scores.shape[0], size=all_scores.shape[0])
        permuted_scores = centered_scores[indices, :]
        null_means.append(np.mean(permuted_scores))
    
    null_means = np.array(null_means)
    
    # Calculate p-value: proportion of null values >= observed
    # np.mean on boolean array gives proportion of True values
    p_value_perm = np.mean(null_means >= observed_mean)
    
    # Calculate effect size (Cohen's d)
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

def create_single_subject_plots(results, times, plots_dir, subject_id, session_id):
    """Create plots for single subject/session results."""
    
    # Overall classification plot
    if 'overall' in results:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        mean_scores = results['overall']['mean_scores_time']
        ax.plot(times, mean_scores, 'b-', linewidth=2, label='AUC')
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
    
    # Trial type plots (only if task events are present)
    if 'trial_types' in results and results['trial_types'] and results.get('has_task_events', False):
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

def aggregate_all_results(output_dir, patient_labels_path=None, mode='patient'):
    """
    Aggregate results from all individual subject/session analyses.
    
    Parameters:
        output_dir: Path to output directory
        patient_labels_path: Optional path to patient labels CSV for subject-type aggregation
        mode: str, 'patient' or 'control' - determines aggregation strategy
    
    Returns:
        aggregated_results: dict containing aggregated results
        times: time points array
        subjects_sessions: list of (subject_id, session_id) tuples processed
    """
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
            if results_path.exists():
                try:
                    with open(results_path, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Load times from the first valid result
                    if times is None:
                        # Try to get times from a saved numpy file or reconstruct
                        times_path = ses_dir / "data" / "times.npy"
                        if times_path.exists():
                            times = np.load(times_path)
                        else:
                            # Reconstruct times from result shape - this is a fallback
                            if 'overall' in results and 'mean_scores_time' in results['overall']:
                                n_times = len(results['overall']['mean_scores_time'])
                                # Standard EEG epoch times (adjust as needed)
                                times = np.linspace(-0.2, 0.8, n_times)  # -200ms to 800ms
                    
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
            'all_mean_scores_time' : [],
            'all_mean_scores_time_VS': [],
            'all_mean_scores_time_MCS': [],
            'all_mean_scores_time_dict': {},
            'MCS': {},
            'VS': {}
        },
        'trial_types': {},
        'has_task_events': False  # Will be updated based on data
    }
    
    # Load patient labels to filter by subject type (only in patient mode)
    subject_state_map = {}
    if mode == 'patient' and patient_labels_path and Path(patient_labels_path).exists():
        try:
            df = pd.read_csv(patient_labels_path, dtype={'session': str})
            for _, row in df.iterrows():
                # Ensure session is zero-padded to 2 digits
                session_str = str(row['session']).zfill(2)
                key = f"{row['subject']}_{session_str}"
                subject_state_map[key] = row['state']
        except Exception as e:
            print(f"Warning: Could not load patient labels for filtering: {e}")
    
    keys_VS = []
    keys_MCS = []
    # Aggregate overall results
    for i, results in enumerate(all_results):
        if 'overall' in results:
            aggregated['overall']['all_mean_aucs'].append(results['overall']['mean_auc'])
            aggregated['overall']['all_mean_scores_time'].append(results['overall']['mean_scores_time'])

            subject_id, session_id = subjects_sessions[i]
            key = f"{subject_id}_{session_id}"
            aggregated['overall']['all_mean_scores_time_dict'][key] = results['overall']['mean_scores_time']
            
            if subject_state_map:
                subject_id, session_id = subjects_sessions[i]
                key = f"{subject_id}_{session_id}"
                state = subject_state_map.get(key, 'UNKNOWN')


    
                if state in ['VS', 'UWS']:
                    aggregated['overall']['all_mean_scores_time_VS'].append(results['overall']['mean_scores_time'])
                    aggregated['overall']['VS'][key] = results['overall']['mean_scores_time']
                    keys_VS.append(key)
                elif state in ['MCS', 'MCS-', 'MCS+']:
                    aggregated['overall']['all_mean_scores_time_MCS'].append(results['overall']['mean_scores_time'])
                    aggregated['overall']['MCS'][key] = results['overall']['mean_scores_time']
                    keys_MCS.append(key)
                
    
    # Calculate overall statistics
    if aggregated['overall']['all_mean_aucs']:
        aggregated['overall']['mean_auc'] = np.mean(aggregated['overall']['all_mean_aucs'])
        aggregated['overall']['std_auc'] = np.std(aggregated['overall']['all_mean_aucs'])
        
        # Point-by-point statistics across subjects/sessions
        all_timeseries = np.array(aggregated['overall']['all_mean_scores_time'])  # (n_subjects, n_timepoints)
        print(f'All timeseries shape: {all_timeseries.shape}')
        
        # Save timeseries data to output directory
        timeseries_dir = output_path / "all_auc_timeseries"
        timeseries_dir.mkdir(exist_ok=True)
        np.save(timeseries_dir / 'all_mean_scores_time.npy', all_timeseries)
        np.save(timeseries_dir / 'keys_VS.npy', np.array(keys_VS))
        np.save(timeseries_dir / 'keys_MCS.npy', np.array(keys_MCS))
        
        aggregated['overall']['mean_scores_time'] = np.mean(all_timeseries, axis=0)  # (n_timepoints,)
        # Calculate std per timepoint (axis=0 gives std across subjects at each time)
        aggregated['overall']['std_scores_time'] = np.std(aggregated['overall']['all_mean_scores_time'], axis=0)
        aggregated['overall']['sem_scores_time'] = aggregated['overall']['std_scores_time'] / np.sqrt(len(all_timeseries))
        
        # Calculate statistics for VS and MCS subgroups
        if aggregated['overall']['all_mean_scores_time_VS']:
            all_timeseries_VS = np.array(aggregated['overall']['all_mean_scores_time_VS'])
            print(f'VS timeseries shape: {all_timeseries_VS.shape}')
            aggregated['overall']['mean_scores_time_VS'] = np.mean(all_timeseries_VS, axis=0)
            aggregated['overall']['std_scores_time_VS'] = np.std(aggregated['overall']['all_mean_scores_time_VS'], axis=0)
            aggregated['overall']['n_subjects_VS'] = len(all_timeseries_VS)
        
        if aggregated['overall']['all_mean_scores_time_MCS']:
            all_timeseries_MCS = np.array(aggregated['overall']['all_mean_scores_time_MCS'])
            print(f'MCS timeseries shape: {all_timeseries_MCS.shape}')
            aggregated['overall']['mean_scores_time_MCS'] = np.mean(all_timeseries_MCS, axis=0)
            aggregated['overall']['std_scores_time_MCS'] = np.std(aggregated['overall']['all_mean_scores_time_MCS'], axis=0)
            aggregated['overall']['n_subjects_MCS'] = len(all_timeseries_MCS)
    
    # Aggregate trial type results
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    for trial_type in trial_types:
        aggregated['trial_types'][trial_type] = {
            'all_mean_aucs': [],
            'all_mean_scores_time': [],
            'all_mean_scores_time_VS': [],
            'all_mean_scores_time_MCS': []
        }
        
        for i, results in enumerate(all_results):
            if 'trial_types' in results and trial_type in results['trial_types']:
                aggregated['trial_types'][trial_type]['all_mean_aucs'].append(
                    results['trial_types'][trial_type]['mean_auc']
                )
                aggregated['trial_types'][trial_type]['all_mean_scores_time'].append(
                    results['trial_types'][trial_type]['mean_scores_time']
                )
                
                if subject_state_map:
                    subject_id, session_id = subjects_sessions[i]
                    key = f"{subject_id}_{session_id}"
                    state = subject_state_map.get(key, 'UNKNOWN')
                    
                    if state in ['VS', 'UWS']:
                        aggregated['trial_types'][trial_type]['all_mean_scores_time_VS'].append(
                            results['trial_types'][trial_type]['mean_scores_time']
                        )
                    elif state in ['MCS', 'MCS-', 'MCS+']:
                        aggregated['trial_types'][trial_type]['all_mean_scores_time_MCS'].append(
                            results['trial_types'][trial_type]['mean_scores_time']
                        )
        
        # Save trial type timeseries data
        if aggregated['trial_types'][trial_type]['all_mean_scores_time']:
            timeseries_dir = output_path / "all_auc_timeseries"
            timeseries_dir.mkdir(exist_ok=True)
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
            # Calculate std per timepoint (axis=0 gives std across subjects at each time)
            aggregated['trial_types'][trial_type]['std_scores_time'] = np.std(aggregated['trial_types'][trial_type]['all_mean_scores_time'], axis=0)
            aggregated['trial_types'][trial_type]['sem_scores_time'] = aggregated['trial_types'][trial_type]['std_scores_time'] / np.sqrt(len(all_timeseries))
            aggregated['trial_types'][trial_type]['n_subjects_sessions'] = len(all_timeseries)
            
            # Calculate statistics for VS and MCS subgroups
            if aggregated['trial_types'][trial_type]['all_mean_scores_time_VS']:
                all_timeseries_VS = np.array(aggregated['trial_types'][trial_type]['all_mean_scores_time_VS'])
                print(f'  {trial_type} VS timeseries shape: {all_timeseries_VS.shape}')
                aggregated['trial_types'][trial_type]['mean_scores_time_VS'] = np.mean(all_timeseries_VS, axis=0)
                aggregated['trial_types'][trial_type]['std_scores_time_VS'] = np.std(all_timeseries_VS, axis=0)
                aggregated['trial_types'][trial_type]['n_subjects_VS'] = len(all_timeseries_VS)
            
            if aggregated['trial_types'][trial_type]['all_mean_scores_time_MCS']:
                all_timeseries_MCS = np.array(aggregated['trial_types'][trial_type]['all_mean_scores_time_MCS'])
                print(f'  {trial_type} MCS timeseries shape: {all_timeseries_MCS.shape}')
                aggregated['trial_types'][trial_type]['mean_scores_time_MCS'] = np.mean(all_timeseries_MCS, axis=0)
                aggregated['trial_types'][trial_type]['std_scores_time_MCS'] = np.std(all_timeseries_MCS, axis=0)
                aggregated['trial_types'][trial_type]['n_subjects_MCS'] = len(all_timeseries_MCS)
            
            # Run permutation test
            print(f"  Running permutation test for {trial_type}...")
            perm_result = run_permutation_test(all_timeseries, n_permutations=1000, seed=42)
            aggregated['trial_types'][trial_type]['permutation_test'] = perm_result
            print(f"    Observed mean AUC: {perm_result['observed_mean']:.4f}")
            print(f"    p-value (permutation): {perm_result['p_value_permutation']:.4f}")
            print(f"    p-value (t-test): {perm_result['p_value_ttest']:.4f}")
            print(f"    t-statistic: {perm_result['t_statistic']:.4f}")
            print(f"    Cohen's d: {perm_result['cohens_d']:.4f}")
            print(f"    Significant: {perm_result['significant_ttest']}")
    
    # Check if any subject has task events
    for i, results in enumerate(all_results):
        if results.get('has_task_events', False):
            aggregated['has_task_events'] = True
            break
    
    # Aggregate by subject type if patient labels provided (only in patient mode)
    aggregated['by_subject_type'] = {}
    if mode == 'patient' and patient_labels_path and Path(patient_labels_path).exists():
        try:
            df = pd.read_csv(patient_labels_path, dtype={'session': str})
            # Create mapping of subject to state with grouping
            subject_state_map = {}
            for _, row in df.iterrows():
                # Ensure session is zero-padded to 2 digits
                session_str = str(row['session']).zfill(2)
                key = f"{row['subject']}_{session_str}"
                state = row['state']
                
                # Group states: VS+UWS -> UWS, MCS+/MCS-/MCS -> MCS, COMA -> COMA
                if state in ['VS', 'UWS']:
                    grouped_state = 'UWS'
                elif state in ['MCS', 'MCS-', 'MCS+']:
                    grouped_state = 'MCS'
                elif state == 'COMA':
                    grouped_state = 'COMA'
                else:
                    grouped_state = state  # Keep others as-is (e.g., CONTROL, EMCS)
                
                subject_state_map[key] = grouped_state
            
            # Group results by subject type
            type_results = {}
            for i, (subject_id, session_id) in enumerate(subjects_sessions):
                key = f"{subject_id}_{session_id}"
                state = subject_state_map.get(key, 'UNKNOWN')
                
                if state not in type_results:
                    type_results[state] = []
                type_results[state].append(all_results[i])
            
            # Debug: Print distribution
            print(f"  Subject-type distribution:")
            for state, results in sorted(type_results.items()):
                print(f"{state}: {len(results)} subject-sessions")
            
            # Aggregate each subject type - focus on COMA, UWS, MCS
            target_groups = ['COMA', 'UWS', 'MCS']
            for state in target_groups:
                if state not in type_results:
                    print(f"    Skipping {state}: not found in results")
                    continue
                    
                state_res_list = type_results[state]
                if len(state_res_list) < 2:  # Skip if too few subjects
                    print(f"    Skipping {state}: only {len(state_res_list)} subject-sessions (need >=2)")
                    continue
                    
                aggregated['by_subject_type'][state] = {
                    'overall': {'all_mean_aucs': [], 'all_mean_scores_time': []}
                }
                
                for res in state_res_list:
                    if 'overall' in res:
                        aggregated['by_subject_type'][state]['overall']['all_mean_aucs'].append(res['overall']['mean_auc'])
                        aggregated['by_subject_type'][state]['overall']['all_mean_scores_time'].append(res['overall']['mean_scores_time'])
                
                # Calculate statistics
                if aggregated['by_subject_type'][state]['overall']['all_mean_aucs']:
                    all_aucs = aggregated['by_subject_type'][state]['overall']['all_mean_aucs']
                    all_ts = np.array(aggregated['by_subject_type'][state]['overall']['all_mean_scores_time'])
                    
                    aggregated['by_subject_type'][state]['overall']['mean_auc'] = np.mean(all_aucs)
                    aggregated['by_subject_type'][state]['overall']['std_auc'] = np.std(all_aucs)
                    aggregated['by_subject_type'][state]['overall']['mean_scores_time'] = np.mean(all_ts, axis=0)
                    aggregated['by_subject_type'][state]['overall']['std_scores_time'] = np.std(all_ts, axis=0)  # SD per time point
                    aggregated['by_subject_type'][state]['overall']['n_subjects'] = len(all_aucs)
                    
            print(f"  Aggregated by subject type: {list(aggregated['by_subject_type'].keys())}")
        except Exception as e:
            print(f"Warning: Could not aggregate by subject type: {e}")
    
    return aggregated, times, subjects_sessions

def save_aggregated_results(aggregated_results, times, subjects_sessions, output_dir, mode='patient'):
    """Save aggregated results to decoding-global-{date} directory."""
    
    # Create global results directory with timestamp
    # Save in the same parent directory as output_dir (e.g., if output_dir is .../DECODER, save in .../DECODER/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    
    if mode == 'control':
        global_dir = output_path / f"decoding-control-global-{timestamp}"
    else:
        global_dir = output_path / f"decoding-global-{timestamp}"
    plots_dir = global_dir / "plots"
    data_dir = global_dir / "data"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving aggregated results to: {global_dir}")
        
    # Save timeseries data as separate numpy files
    if 'overall' in aggregated_results:
        if 'all_mean_scores_time_dict' in aggregated_results['overall']:
            np.save(data_dir / "all_mean_scores_time_dict.npy", 
                   np.array(aggregated_results['overall']['all_mean_scores_time_dict']))
        
        if 'all_mean_scores_time_VS' in aggregated_results['overall'] and aggregated_results['overall']['all_mean_scores_time_VS']:
            np.save(data_dir / "all_mean_scores_time_VS.npy", 
                   np.array(aggregated_results['overall']['all_mean_scores_time_VS']))
            print(f"Saved VS timeseries: {len(aggregated_results['overall']['all_mean_scores_time_VS'])} subjects")
        
        if 'all_mean_scores_time_MCS' in aggregated_results['overall'] and aggregated_results['overall']['all_mean_scores_time_MCS']:
            np.save(data_dir / "all_mean_scores_time_MCS.npy", 
                   np.array(aggregated_results['overall']['all_mean_scores_time_MCS']))
            print(f"Saved MCS timeseries: {len(aggregated_results['overall']['all_mean_scores_time_MCS'])} subjects")
    
        if 'VS' in aggregated_results['overall'] and aggregated_results['overall']['VS']:
            np.save(data_dir / "VS.npy", 
                   np.array(aggregated_results['overall']['VS']))
            print(f"Saved VS timeseries: {len(aggregated_results['overall']['VS'])} subjects")
        
        if 'MCS' in aggregated_results['overall'] and aggregated_results['overall']['MCS']:
            np.save(data_dir / "MCS.npy", 
                   np.array(aggregated_results['overall']['MCS']))
            print(f"Saved MCS timeseries: {len(aggregated_results['overall']['MCS'])} subjects")
            
    # Save per-subject-type per-trial-type timeseries data (8 combinations)
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    for trial_type in trial_types:
        if trial_type in aggregated_results['trial_types']:
            trial_data = aggregated_results['trial_types'][trial_type]
            
            # Save VS + trial_type
            if 'all_mean_scores_time_VS' in trial_data and trial_data['all_mean_scores_time_VS']:
                np.save(data_dir / f"all_mean_scores_time_VS_{trial_type}.npy", 
                       np.array(trial_data['all_mean_scores_time_VS']))
                print(f"Saved VS_{trial_type} timeseries: {len(trial_data['all_mean_scores_time_VS'])} subjects")
            
            # Save MCS + trial_type
            if 'all_mean_scores_time_MCS' in trial_data and trial_data['all_mean_scores_time_MCS']:
                np.save(data_dir / f"all_mean_scores_time_MCS_{trial_type}.npy", 
                       np.array(trial_data['all_mean_scores_time_MCS']))
                print(f"Saved MCS_{trial_type} timeseries: {len(trial_data['all_mean_scores_time_MCS'])} subjects")
    
    # Save raw aggregated results
    with open(data_dir / "aggregated_results.pkl", 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    # Save times array for analysis scripts
    np.save(data_dir / "times.npy", times)
    
    # Save permutation test results separately
    permutation_results = {}
    for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
        if (trial_type in aggregated_results['trial_types'] and 
            'permutation_test' in aggregated_results['trial_types'][trial_type]):
            perm_test = aggregated_results['trial_types'][trial_type]['permutation_test']
            # Save without the full null distribution to keep file size manageable
            permutation_results[trial_type] = {
                'observed_mean': float(perm_test['observed_mean']),
                'p_value_permutation': float(perm_test['p_value_permutation']),
                'p_value_ttest': float(perm_test['p_value_ttest']),
                't_statistic': float(perm_test['t_statistic']),
                'cohens_d': float(perm_test['cohens_d']),
                'null_mean': float(perm_test['null_mean']),
                'null_std': float(perm_test['null_std']),
                'n_permutations': int(perm_test['n_permutations']),
                'n_subjects': int(perm_test['n_subjects']),
                'significant_permutation': bool(perm_test['significant_permutation']),
                'significant_ttest': bool(perm_test['significant_ttest'])
            }
    
    # Save permutation results as JSON
    with open(data_dir / "permutation_test_results.json", 'w') as f:
        json.dump(permutation_results, f, indent=2)
    
    print(f"Permutation test results saved to: {data_dir / 'permutation_test_results.json'}")
    
    # Save comprehensive JSON summary
    json_results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "aggregated_sequential_decoding",
        "mode": mode,
        "n_subjects_sessions": len(subjects_sessions),
        "subjects_sessions": [{"subject_id": sub, "session_id": ses} for sub, ses in subjects_sessions],
        "model_info": {
            "classifier": "LogisticRegression",
            "solver": "liblinear", 
            "preprocessing": "StandardScaler",
            "approach": "sequential_subject_session_then_aggregated",
            "scoring": "roc_auc"
        },
        "overall_classification": {},
        "trial_type_classification": {},
        "has_task_events": aggregated_results.get('has_task_events', False)
    }
    
    # Add overall results
    if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
        json_results["overall_classification"] = {
            "mean_auc": float(aggregated_results['overall']['mean_auc']),
            "std_auc": float(aggregated_results['overall']['std_auc']),
            "description": "Mean and std calculated across all subjects and sessions",
            "individual_mean_aucs": [float(x) for x in aggregated_results['overall']['all_mean_aucs']]
        }
        
        # Add VS subgroup if available
        if 'n_subjects_VS' in aggregated_results['overall']:
            json_results["overall_classification"]["VS_subgroup"] = {
                "n_subjects": int(aggregated_results['overall']['n_subjects_VS']),
                "description": "VS and UWS subjects"
            }
        
        # Add MCS subgroup if available
        if 'n_subjects_MCS' in aggregated_results['overall']:
            json_results["overall_classification"]["MCS_subgroup"] = {
                "n_subjects": int(aggregated_results['overall']['n_subjects_MCS']),
                "description": "MCS, MCS+, and MCS- subjects"
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
                json_results["trial_type_classification"][trial_type]['permutation_test'] = {
                    "observed_mean": float(perm['observed_mean']),
                    "p_value_permutation": float(perm['p_value_permutation']),
                    "p_value_ttest": float(perm['p_value_ttest']),
                    "t_statistic": float(perm['t_statistic']),
                    "cohens_d": float(perm['cohens_d']),
                    "n_subjects": int(perm['n_subjects']),
                    "significant_permutation": bool(perm['significant_permutation']),
                    "significant_ttest": bool(perm['significant_ttest'])
                }
    
    # Save JSON
    with open(data_dir / "aggregated_summary.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create aggregated plots
    create_aggregated_plots(aggregated_results, times, plots_dir)
    
    # Save detailed text summary
    save_text_summary(aggregated_results, subjects_sessions, data_dir)
    
    print(f"Aggregated results saved to: {global_dir}")
    return global_dir

def create_aggregated_plots(aggregated_results, times, plots_dir):
    """Create plots for aggregated results."""
    
    # Overall classification plot
    if 'overall' in aggregated_results and 'mean_scores_time' in aggregated_results['overall']:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        mean_scores = aggregated_results['overall']['mean_scores_time']
        std_scores = aggregated_results['overall']['std_scores_time']
        sem_scores = aggregated_results['overall']['sem_scores_time']
        n_sessions = len(aggregated_results['overall']['all_mean_aucs'])
        
        # Plot mean with SD error band (variability per time point)
        ax.plot(times, mean_scores, 'b-', linewidth=3, label=f'Mean AUC (n={n_sessions} subjects)')
        ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                       alpha=0.3, color='blue', label=r'$\sigma$')
        
        # Reference lines
        add_stimulus_lines(ax, times)
    
        # Formatting
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylabel("AUC", fontsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
    
        # Add peak info
        peak_idx = np.argmax(mean_scores)
        peak_auc = mean_scores[peak_idx]
        peak_time = times[peak_idx]
        ax.text(0.02, 0.98, f'Peak: AUC={peak_auc:.3f} at t={peak_time:.3f}s', 
                transform=ax.transAxes, fontsize=20, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plots_dir / "aggregated_overall_classification.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Overall classification plot - Small version (no legend, larger labels)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    mean_scores = aggregated_results['overall']['mean_scores_time']
    std_scores = aggregated_results['overall']['std_scores_time']
    n_sessions = len(aggregated_results['overall']['all_mean_aucs'])
    
    # Remove last 10 points from the plot
    times_plot = times[:-20]
    mean_scores_plot = mean_scores[:-20]
    std_scores_plot = std_scores[:-20]
    
    # Plot mean with SD error band (variability per time point)
    ax.plot(times_plot, mean_scores_plot, 'b-', linewidth=3)
    ax.fill_between(times_plot, mean_scores_plot - std_scores_plot, mean_scores_plot + std_scores_plot,
                   alpha=0.3, color='blue')
    
    # Reference lines
    add_stimulus_lines(ax, times_plot)
    
    # Formatting
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("AUC-ROC", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Add peak info with larger font (using the reduced data)
    peak_idx = np.argmax(mean_scores_plot)
    peak_auc = mean_scores_plot[peak_idx]
    peak_time = times_plot[peak_idx]
    ax.text(0.02, 0.98, f'Peak: AUC={peak_auc:.3f} at t={peak_time:.3f}s', 
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_overall_classification_small.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Trial type plots - Single plot with 4 lines
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
            
            # Plot line with sigma fill
            ax.plot(times, mean_scores, color=colors[i], linewidth=2.5, 
                   label=f'{trial_type}: {mean_auc:.3f} ± {std_auc:.3f}')
            ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                           alpha=0.1, color=colors[i])
    
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("AUC", fontsize=24)
    ax.set_title("Decoding per Trial Type", fontsize=26)
    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_trial_type_classification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Trial type plots - Small version (no title, larger labels, simple legend)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    for i, trial_type in enumerate(trial_types):
        if (trial_type in aggregated_results['trial_types'] and 
            'mean_scores_time' in aggregated_results['trial_types'][trial_type]):
            
            data = aggregated_results['trial_types'][trial_type]
            mean_scores = data['mean_scores_time']
            std_scores = data['std_scores_time']
            
            # Plot line with sigma fill, just trial name as label
            ax.plot(times, mean_scores, color=colors[i], linewidth=2.5, label=trial_type)
            ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                           alpha=0.1, color=colors[i])
    
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("AUC-ROC", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_trial_type_classification_small.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Subject-type aggregated plots (COMA, UWS, MCS)
    if 'by_subject_type' in aggregated_results and aggregated_results['by_subject_type']:
        print("Creating subject-type aggregated plots...")
        
        # Order: COMA, UWS, MCS
        target_types = ['COMA', 'UWS', 'MCS']
        available_types = [st for st in target_types 
                          if st in aggregated_results['by_subject_type'] and
                             'overall' in aggregated_results['by_subject_type'][st] and 
                             'mean_scores_time' in aggregated_results['by_subject_type'][st]['overall']]
        
        if available_types:
            # Create 1 row, 3 columns (one per type)
            n_types = len(available_types)
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            
            colors_map = {'COMA': 'darkred', 'UWS': 'red', 'MCS': 'orange'}
            
            for i, subject_type in enumerate(available_types):
                ax = axes[i]
                data = aggregated_results['by_subject_type'][subject_type]['overall']
                
                mean_scores = data['mean_scores_time']
                std_scores = data['std_scores_time']  # SD per time point
                n_subjects = data['n_subjects']
                mean_auc = data['mean_auc']
                std_auc = data['std_auc']
                
                color = colors_map.get(subject_type, 'gray')
                
                # Plot with only SD (no SEM)
                ax.plot(times, mean_scores, color=color, linewidth=2, label=f'Mean AUC (n={n_subjects})')
                ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                               alpha=0.3, color=color, label=r'$\sigma$')
                
                ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, label="Chance")
                add_stimulus_lines(ax, times)
                
                ax.set_xlabel("Time (s)", fontsize=20)
                ax.set_ylabel("AUC", fontsize=20)
                ax.set_title(f"{subject_type}\nMean AUC: {mean_auc:.3f} ± {std_auc:.3f} (n={n_subjects} subjects)", 
                           fontsize=22)
                ax.legend(fontsize=16, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0.4, 1.0])
            
            # Hide unused subplots if less than 3
            for j in range(len(available_types), 3):
                axes[j].axis('off')
            
            plt.suptitle("Decoder Performance by Subject Type (Original vs Reconstructed EEG)\n" +
                        "Groups: COMA | UWS (VS+UWS) | MCS (MCS-/MCS/MCS+)", 
                        fontsize=22)
            plt.tight_layout()
            plt.savefig(plots_dir / "aggregated_by_subject_type.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Created subject-type plot with {len(available_types)} types: {', '.join(available_types)}")

def is_decoder_result_complete(output_dir, subject_id, session_id):
    """
    Check if decoder results already exist for a subject/session.
    
    Parameters:
        output_dir: str or Path, output directory
        subject_id: str, subject identifier
        session_id: str, session identifier
        
    Returns:
        bool: True if complete results exist, False otherwise
    """
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

def save_text_summary(aggregated_results, subjects_sessions, data_dir):
    """Save a detailed text summary of aggregated results."""
    
    summary_path = data_dir / "aggregated_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("AGGREGATED DECODING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Subject-Sessions: {len(subjects_sessions)}\n\n")
        
        f.write("Subjects and Sessions Analyzed:\n")
        f.write("-" * 30 + "\n")
        for sub, ses in subjects_sessions:
            f.write(f"  sub-{sub} ses-{ses}\n")
        f.write("\n")
        
        # Overall results
        if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
            f.write("OVERALL CLASSIFICATION RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean AUC (across sessions): {aggregated_results['overall']['mean_auc']:.6f}\n")
            f.write(f"Std AUC (across sessions): {aggregated_results['overall']['std_auc']:.6f}\n\n")
        
        # Trial type results
        f.write("TRIAL TYPE CLASSIFICATION RESULTS\n")
        f.write("-" * 30 + "\n")
        for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
            if (trial_type in aggregated_results['trial_types'] and 
                'mean_auc' in aggregated_results['trial_types'][trial_type]):
                data = aggregated_results['trial_types'][trial_type]
                f.write(f"{trial_type}:\n")
                f.write(f"  Mean AUC: {data['mean_auc']:.6f}\n")
                f.write(f"  Std AUC: {data['std_auc']:.6f}\n")
                f.write(f"  N Sessions: {data['n_subjects_sessions']}\n\n")

def main():
    """Main function - Sequential processing approach."""
    args = parse_arguments()
    
    # Adjust output directory based on mode if using default
    if args.output_dir == "./doc_benchmark/results/DECODER" and args.mode == 'control':
        args.output_dir = "./doc_benchmark/results/DECODER_CONTROL"
    
    mode_label = "Control" if args.mode == 'control' else "Patient"
    print(f"EEG Original vs Reconstructed Decoder - {mode_label} Mode")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    if args.filter_trial_types:
        print(f"Trial type filtering: {args.filter_trial_types}")
    if args.mode == 'patient' and args.patient_labels:
        print(f"Patient labels: {args.patient_labels}")
    
    # If aggregate-only mode, skip subject processing and go straight to aggregation
    if args.aggregate_only:
        print("AGGREGATE-ONLY MODE: Aggregating existing results...")
        print(f"Looking for results in: {args.output_dir}")
        
        aggregated_results, times, subjects_sessions = aggregate_all_results(args.output_dir, args.patient_labels, args.mode)
        
        if aggregated_results is None:
            print("ERROR: Failed to aggregate results! No completed subjects found.")
            return
        
        # Save aggregated results to global directory
        global_dir = save_aggregated_results(aggregated_results, times, subjects_sessions, args.output_dir, args.mode)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATION COMPLETED!")
        print(f"Individual results source: {args.output_dir}")
        print(f"Aggregated results: {global_dir}")
        print(f"{'='*60}")
        
        # Print summary of aggregated results
        if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
            print(f"\nAGGREGATED SUMMARY:")
            print(f"  Overall AUC: {aggregated_results['overall']['mean_auc']:.3f} ± {aggregated_results['overall']['std_auc']:.3f}")
            
            if aggregated_results['trial_types']:
                print(f"  Trial Type AUCs:")
                for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
                    if trial_type in aggregated_results['trial_types'] and 'mean_auc' in aggregated_results['trial_types'][trial_type]:
                        data = aggregated_results['trial_types'][trial_type]
                        print(f"    {trial_type}: {data['mean_auc']:.3f} ± {data['std_auc']:.3f} (n={data['n_subjects_sessions']})")
            
            print(f"\nTotal subjects-sessions analyzed: {len(subjects_sessions)}")
            print(f"Subject-session combinations: {[f'sub-{s} ses-{ses}' for s, ses in subjects_sessions]}")
        
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
    
    print(f"\nProcessing {len(subjects)} subjects × {len(sessions)} sessions = {len(subjects) * len(sessions)} total combinations")
    
    # Sequential processing: decode each subject-session combination
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
                print(f"  ⏭️  Skipping sub-{subject_id} ses-{session_id} - results already exist")
                skipped_count += 1
                processed_count += 1  # Count as processed for aggregation
                continue
            
            # Load data for this specific subject-session
            X, y, events, times = load_epochs_single_subject_session(
                args.main_path, subject_id, session_id, 
                filter_trial_types=args.filter_trial_types,
                verbose=args.verbose
            )
            
            if X is None:
                print(f"  No data found for sub-{subject_id} ses-{session_id}")
                failed_count += 1
                continue
            
            # Perform all decoding analyses for this subject-session
            results = decode_single_subject_session(
                X, y, events, times, cv=args.cv, n_jobs=args.n_jobs, 
                mode=args.mode, verbose=args.verbose
            )
            
            if results is None:
                print(f"  Decoding failed for sub-{subject_id} ses-{session_id}")
                failed_count += 1
                continue
            
            # Save times array for this subject-session (needed for aggregation)
            sub_ses_dir = Path(args.output_dir) / f"sub-{subject_id}" / f"ses-{session_id}" / "data"
            sub_ses_dir.mkdir(parents=True, exist_ok=True)
            np.save(sub_ses_dir / "times.npy", times)
            
            # Save individual results
            save_single_subject_session_results(
                results, times, subject_id, session_id, args.output_dir
            )
            
            processed_count += 1
            print(f"  ✓ Successfully processed sub-{subject_id} ses-{session_id}")
    
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL PROCESSING COMPLETE")
    print(f"Successfully processed: {processed_count - skipped_count}")
    print(f"Skipped (already complete): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total complete: {processed_count}")
    print(f"{'='*60}")
    
    if processed_count == 0:
        print("ERROR: No subject-sessions were successfully processed!")
        return
    
    # Aggregate all individual results
    print(f"\nAggregating results from {processed_count} subject-sessions...")
    aggregated_results, times, subjects_sessions = aggregate_all_results(args.output_dir, args.patient_labels, args.mode)
    
    if aggregated_results is None:
        print("ERROR: Failed to aggregate results!")
        return
    
    # Save aggregated results to global directory
    global_dir = save_aggregated_results(aggregated_results, times, subjects_sessions, args.output_dir, args.mode)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETED!")
    print(f"Individual results: {args.output_dir}")
    print(f"Aggregated results: {global_dir}")
    print(f"{'='*60}")
    
    # Print summary of aggregated results
    if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
        print(f"\nAGGREGATED SUMMARY:")
        print(f"  Overall AUC: {aggregated_results['overall']['mean_auc']:.3f} ± {aggregated_results['overall']['std_auc']:.3f}")
        
        if aggregated_results['trial_types']:
            print(f"  Trial Type AUCs:")
            for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
                if trial_type in aggregated_results['trial_types'] and 'mean_auc' in aggregated_results['trial_types'][trial_type]:
                    data = aggregated_results['trial_types'][trial_type]
                    print(f"    {trial_type}: {data['mean_auc']:.3f} ± {data['std_auc']:.3f} (n={data['n_subjects_sessions']})")
        
        print(f"\nTotal subjects-sessions analyzed: {len(subjects_sessions)}")
        print(f"Subject-session combinations: {[f'sub-{s} ses-{ses}' for s, ses in subjects_sessions]}")

if __name__ == "__main__":
    main()