#!/usr/bin/env python3
"""
Script to decode original vs reconstructed EEG data from .fif files.
Original files (-original.fif) are labeled as class 0.
Reconstructed files (-recon.fif) are labeled as class 1.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from datetime import datetime
import pickle

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
    30: 'LSGS',    # Local Standard Global Standard (highest count)
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
        help="Main path containing subject directories (e.g., /data/eeg_study)",
        default="/Users/trinidad.borrell/Documents/Work/PhD/data/TOTEM/zero_shot_data/fifdata/fifdata"
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

def load_epochs_single_subject_session(main_path, subject_id, session_id, verbose=False):
    """
    Load epochs from .fif files for a single subject and session.
    
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
            
            n_epochs = data.shape[0]
            all_epochs.append(data)
            all_labels.extend([0] * n_epochs)
            all_events.extend(events)
            
            if times is None:
                times = epochs.times
            
            if verbose:
                print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs, {data.shape[2]} timepoints")
                
        except Exception as e:
            print(f"    Error loading {fif_file}: {e}")
    
    # Process reconstructed files (label 1)
    for fif_file in recon_files:
        try:
            epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
            data = epochs.get_data().astype(np.float32)
            events = epochs.events[:, 2]
            
            n_epochs = data.shape[0]
            all_epochs.append(data)
            all_labels.extend([1] * n_epochs)
            all_events.extend(events)
            
            if times is None:
                times = epochs.times
            
            if verbose:
                print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs, {data.shape[2]} timepoints")
                
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
    Perform all decoding analyses for a single subject/session.
    
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
    
    # Local/Global effects
    print("  Performing local/global analysis...")
    local_standard_events = [30, 40]  # LSGS, LSGD
    local_deviant_events = [50, 60]   # LDGD, LDGS
    global_standard_events = [30, 60] # LSGS, LDGS
    global_deviant_events = [40, 50]  # LSGD, LDGD
    
    results['local_global'] = {}
    
    # Filter for relevant events
    relevant_mask = np.isin(events, [30, 40, 50, 60])
    X_relevant = X[relevant_mask]
    y_relevant = y[relevant_mask]
    events_relevant = events[relevant_mask]
    
    # For each data type (original=0, reconstructed=1)
    for data_type in [0, 1]:
        data_name = 'original' if data_type == 0 else 'reconstructed'
        data_mask = y_relevant == data_type
        X_data = X_relevant[data_mask]
        events_data = events_relevant[data_mask]
        
        if X_data.shape[0] == 0:
            continue
        
        # Local effect analysis
        local_mask = np.isin(events_data, local_standard_events + local_deviant_events)
        if np.sum(local_mask) > 0:
            X_local = X_data[local_mask]
            events_local = events_data[local_mask]
            y_local = np.isin(events_local, local_deviant_events).astype(int)
            
            if len(np.unique(y_local)) == 2 and X_local.shape[0] >= cv:
                try:
                    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
                    scores_local = cross_val_multiscore(time_decod, X_local, y_local, cv=cv, n_jobs=1)
                    
                    if scores_local.ndim == 2:
                        mean_scores_time = np.mean(scores_local, axis=0)
                    else:
                        mean_scores_time = scores_local
                    
                    results['local_global'][f'{data_name}_local'] = {
                        'scores': scores_local,
                        'mean_scores_time': mean_scores_time,
                        'mean_auc': np.mean(mean_scores_time),
                        'std_auc': np.std(mean_scores_time)
                    }
                    print(f"    {data_name} local AUC: {results['local_global'][f'{data_name}_local']['mean_auc']:.3f}")
                except Exception as e:
                    print(f"    Error in {data_name} local analysis: {e}")
        
        # Global effect analysis
        global_mask = np.isin(events_data, global_standard_events + global_deviant_events)
        if np.sum(global_mask) > 0:
            X_global = X_data[global_mask]
            events_global = events_data[global_mask]
            y_global = np.isin(events_global, global_standard_events).astype(int)
            
            if len(np.unique(y_global)) == 2 and X_global.shape[0] >= cv:
                try:
                    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
                    scores_global = cross_val_multiscore(time_decod, X_global, y_global, cv=cv, n_jobs=1)
                    
                    if scores_global.ndim == 2:
                        mean_scores_time = np.mean(scores_global, axis=0)
                    else:
                        mean_scores_time = scores_global
                    
                    results['local_global'][f'{data_name}_global'] = {
                        'scores': scores_global,
                        'mean_scores_time': mean_scores_time,
                        'mean_auc': np.mean(mean_scores_time),
                        'std_auc': np.std(mean_scores_time)
                    }
                    print(f"    {data_name} global AUC: {results['local_global'][f'{data_name}_global']['mean_auc']:.3f}")
                except Exception as e:
                    print(f"    Error in {data_name} global analysis: {e}")
    
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
        "overall": {
            "mean_auc": float(results['overall']['mean_auc']),
            "std_auc": float(results['overall']['std_auc'])
        } if 'overall' in results else None,
        "trial_types": {},
        "local_global": {}
    }
    
    if 'trial_types' in results:
        for trial_type, trial_data in results['trial_types'].items():
            json_results["trial_types"][trial_type] = {
                "mean_auc": float(trial_data['mean_auc']),
                "std_auc": float(trial_data['std_auc']),
                "n_trials": int(trial_data['n_trials'])
            }
    
    if 'local_global' in results:
        for effect_name, effect_data in results['local_global'].items():
            json_results["local_global"][effect_name] = {
                "mean_auc": float(effect_data['mean_auc']),
                "std_auc": float(effect_data['std_auc'])
            }
    
    json_path = data_dir / "decoding_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create plots
    create_single_subject_plots(results, times, plots_dir, subject_id, session_id)
    
    print(f"  Results saved to: {sub_ses_dir}")
    return sub_ses_dir

def create_single_subject_plots(results, times, plots_dir, subject_id, session_id):
    """Create plots for single subject/session results."""
    
    # Overall classification plot
    if 'overall' in results:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        mean_scores = results['overall']['mean_scores_time']
        ax.plot(times, mean_scores, 'b-', linewidth=2, label='AUC')
        ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
        ax.axvline(0, color="red", linestyle=":", label="Stimulus onset", alpha=0.5)
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
    
    # Local/Global effects plots
    if 'local_global' in results and results['local_global']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        analyses = [
            ('original_local', 'Original - Local Effect', 0, 0),
            ('original_global', 'Original - Global Effect', 0, 1),
            ('reconstructed_local', 'Reconstructed - Local Effect', 1, 0),
            ('reconstructed_global', 'Reconstructed - Global Effect', 1, 1)
        ]
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (key, title, row, col) in enumerate(analyses):
            ax = axes[row, col]
            if key in results['local_global']:
                mean_scores = results['local_global'][key]['mean_scores_time']
                ax.plot(times, mean_scores, color=colors[i], linewidth=2)
                ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
                ax.set_title(title)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("AUC")
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0.4, 1.0])
            else:
                ax.text(0.5, 0.5, f'No data for\n{title}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.suptitle(f"Local/Global Effects - sub-{subject_id} ses-{session_id}")
        plt.tight_layout()
        plt.savefig(plots_dir / "local_global_effects.png", dpi=300, bbox_inches='tight')
        plt.close()

def aggregate_all_results(output_dir):
    """
    Aggregate results from all individual subject/session analyses.
    
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
        'overall': {'all_mean_aucs': [], 'all_mean_scores_time': []},
        'trial_types': {},
        'local_global': {}
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
        
        # Point-by-point statistics across subjects/sessions
        all_timeseries = np.array(aggregated['overall']['all_mean_scores_time'])  # (n_subjects_sessions, n_timepoints)
        aggregated['overall']['mean_scores_time'] = np.mean(all_timeseries, axis=0)  # (n_timepoints,)
        aggregated['overall']['std_scores_time'] = np.std(all_timeseries, axis=0)    # (n_timepoints,)
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
            aggregated['trial_types'][trial_type]['std_scores_time'] = np.std(all_timeseries, axis=0)
            aggregated['trial_types'][trial_type]['sem_scores_time'] = aggregated['trial_types'][trial_type]['std_scores_time'] / np.sqrt(len(all_timeseries))
            aggregated['trial_types'][trial_type]['n_subjects_sessions'] = len(all_timeseries)
    
    # Aggregate local/global effects
    effect_types = ['original_local', 'original_global', 'reconstructed_local', 'reconstructed_global']
    for effect_type in effect_types:
        aggregated['local_global'][effect_type] = {
            'all_mean_aucs': [],
            'all_mean_scores_time': []
        }
        
        for results in all_results:
            if 'local_global' in results and effect_type in results['local_global']:
                aggregated['local_global'][effect_type]['all_mean_aucs'].append(
                    results['local_global'][effect_type]['mean_auc']
                )
                aggregated['local_global'][effect_type]['all_mean_scores_time'].append(
                    results['local_global'][effect_type]['mean_scores_time']
                )
        
        # Calculate statistics for this effect type
        if aggregated['local_global'][effect_type]['all_mean_aucs']:
            aggregated['local_global'][effect_type]['mean_auc'] = np.mean(
                aggregated['local_global'][effect_type]['all_mean_aucs']
            )
            aggregated['local_global'][effect_type]['std_auc'] = np.std(
                aggregated['local_global'][effect_type]['all_mean_aucs']
            )
            
            # Point-by-point statistics
            all_timeseries = np.array(aggregated['local_global'][effect_type]['all_mean_scores_time'])
            aggregated['local_global'][effect_type]['mean_scores_time'] = np.mean(all_timeseries, axis=0)
            aggregated['local_global'][effect_type]['std_scores_time'] = np.std(all_timeseries, axis=0)
            aggregated['local_global'][effect_type]['sem_scores_time'] = aggregated['local_global'][effect_type]['std_scores_time'] / np.sqrt(len(all_timeseries))
            aggregated['local_global'][effect_type]['n_subjects_sessions'] = len(all_timeseries)
    
    return aggregated, times, subjects_sessions

def save_aggregated_results(aggregated_results, times, subjects_sessions, output_dir):
    """Save aggregated results to decoding-global-{date} directory."""
    
    # Create global results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_dir = Path(output_dir).parent / f"decoding-global-{timestamp}"
    plots_dir = global_dir / "plots"
    data_dir = global_dir / "data"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving aggregated results to: {global_dir}")
    
    # Save times array
    np.save(data_dir / "times.npy", times)
    
    # Save raw aggregated results
    with open(data_dir / "aggregated_results.pkl", 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    # Save comprehensive JSON summary
    json_results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "aggregated_sequential_decoding",
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
        "local_global_effects": {}
    }
    
    # Add overall results
    if 'overall' in aggregated_results and 'mean_auc' in aggregated_results['overall']:
        json_results["overall_classification"] = {
            "mean_auc": float(aggregated_results['overall']['mean_auc']),
            "std_auc": float(aggregated_results['overall']['std_auc']),
            "description": "Mean and std calculated across all subjects and sessions",
            "individual_mean_aucs": [float(x) for x in aggregated_results['overall']['all_mean_aucs']]
        }
    
    # Add trial type results
    for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
        if (trial_type in aggregated_results['trial_types'] and 
            'mean_auc' in aggregated_results['trial_types'][trial_type]):
            json_results["trial_type_classification"][trial_type] = {
                "mean_auc": float(aggregated_results['trial_types'][trial_type]['mean_auc']),
                "std_auc": float(aggregated_results['trial_types'][trial_type]['std_auc']),
                "n_subjects_sessions": int(aggregated_results['trial_types'][trial_type]['n_subjects_sessions'])
            }
    
    # Add local/global effects
    for effect_type in ['original_local', 'original_global', 'reconstructed_local', 'reconstructed_global']:
        if (effect_type in aggregated_results['local_global'] and 
            'mean_auc' in aggregated_results['local_global'][effect_type]):
            json_results["local_global_effects"][effect_type] = {
                "mean_auc": float(aggregated_results['local_global'][effect_type]['mean_auc']),
                "std_auc": float(aggregated_results['local_global'][effect_type]['std_auc']),
                "n_subjects_sessions": int(aggregated_results['local_global'][effect_type]['n_subjects_sessions'])
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
        
        # Plot mean with error bands
        ax.plot(times, mean_scores, 'b-', linewidth=3, label=f'Mean AUC (n={n_sessions} sessions)')
        ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                       alpha=0.3, color='blue', label='±1 SD')
        ax.fill_between(times, mean_scores - sem_scores, mean_scores + sem_scores,
                       alpha=0.5, color='blue', label='±1 SEM')
        
        # Reference lines
        ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
        ax.axvline(0, color="red", linestyle=":", label="Stimulus onset", alpha=0.5)
    
    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
    ax.set_ylabel("AUC", fontsize=14, fontweight='bold')
    ax.set_title("Aggregated Overall Classification: Original vs Reconstructed EEG\n" +
                f"Mean ± SD/SEM across {n_sessions} subject-sessions", fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Add peak info
    peak_idx = np.argmax(mean_scores)
    peak_auc = mean_scores[peak_idx]
    peak_time = times[peak_idx]
    ax.text(0.02, 0.98, f'Peak: AUC={peak_auc:.3f} at t={peak_time:.3f}s', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_overall_classification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Trial type plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, trial_type in enumerate(trial_types):
        ax = axes[i]
        if (trial_type in aggregated_results['trial_types'] and 
            'mean_scores_time' in aggregated_results['trial_types'][trial_type]):
            
            data = aggregated_results['trial_types'][trial_type]
            mean_scores = data['mean_scores_time']
            std_scores = data['std_scores_time']
            n_sessions = data['n_subjects_sessions']
            
            ax.plot(times, mean_scores, color=colors[i], linewidth=2, label=f'{trial_type} (n={n_sessions})')
            ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                           alpha=0.3, color=colors[i], label='±1 SD')
            ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
            ax.set_title(f"{trial_type}\nMean AUC: {data['mean_auc']:.3f} ± {data['std_auc']:.3f}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("AUC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.4, 1.0])
        else:
            ax.text(0.5, 0.5, f'No data for {trial_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{trial_type}")
    
    plt.suptitle("Aggregated Trial Type Classification", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_trial_type_classification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Local/Global effects plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    analyses = [
        ('original_local', 'Original - Local Effect', 0, 0),
        ('original_global', 'Original - Global Effect', 0, 1),
        ('reconstructed_local', 'Reconstructed - Local Effect', 1, 0),
        ('reconstructed_global', 'Reconstructed - Global Effect', 1, 1)
    ]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (key, title, row, col) in enumerate(analyses):
        ax = axes[row, col]
        if (key in aggregated_results['local_global'] and 
            'mean_scores_time' in aggregated_results['local_global'][key]):
            
            data = aggregated_results['local_global'][key]
            mean_scores = data['mean_scores_time']
            std_scores = data['std_scores_time']
            n_sessions = data['n_subjects_sessions']
            
            ax.plot(times, mean_scores, color=colors[i], linewidth=2, label=f'Mean AUC (n={n_sessions})')
            ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                           alpha=0.3, color=colors[i], label='±1 SD')
            ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
            ax.set_title(f"{title}\nMean AUC: {data['mean_auc']:.3f} ± {data['std_auc']:.3f}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("AUC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.4, 1.0])
        else:
            ax.text(0.5, 0.5, f'No data for\n{title}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.suptitle("Aggregated Local/Global Effects", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregated_local_global_effects.png", dpi=300, bbox_inches='tight')
    plt.close()

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
        
        # Local/Global effects
        f.write("LOCAL/GLOBAL EFFECTS RESULTS\n")
        f.write("-" * 30 + "\n")
        for effect_type in ['original_local', 'original_global', 'reconstructed_local', 'reconstructed_global']:
            if (effect_type in aggregated_results['local_global'] and 
                'mean_auc' in aggregated_results['local_global'][effect_type]):
                data = aggregated_results['local_global'][effect_type]
                f.write(f"{effect_type}:\n")
                f.write(f"  Mean AUC: {data['mean_auc']:.6f}\n")
                f.write(f"  Std AUC: {data['std_auc']:.6f}\n")
                f.write(f"  N Sessions: {data['n_subjects_sessions']}\n\n")

def main():
    """Main function - Sequential processing approach."""
    args = parse_arguments()
    
    print("EEG Original vs Reconstructed Decoder - Sequential Processing")
    print("=" * 60)
    
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
                args.main_path, subject_id, session_id, args.verbose
            )
            
            if X is None:
                print(f"  No data found for sub-{subject_id} ses-{session_id}")
                failed_count += 1
                continue
            
            # Perform all decoding analyses for this subject-session
            results = decode_single_subject_session(
                X, y, events, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
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
    aggregated_results, times, subjects_sessions = aggregate_all_results(args.output_dir)
    
    if aggregated_results is None:
        print("ERROR: Failed to aggregate results!")
        return
    
    # Save aggregated results to global directory
    global_dir = save_aggregated_results(aggregated_results, times, subjects_sessions, args.output_dir)
    
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
        
        if aggregated_results['local_global']:
            print(f"  Local/Global Effects:")
            for effect in ['original_local', 'original_global', 'reconstructed_local', 'reconstructed_global']:
                if effect in aggregated_results['local_global'] and 'mean_auc' in aggregated_results['local_global'][effect]:
                    data = aggregated_results['local_global'][effect]
                    print(f"    {effect}: {data['mean_auc']:.3f} ± {data['std_auc']:.3f} (n={data['n_subjects_sessions']})")
        
        print(f"\nTotal subjects-sessions analyzed: {len(subjects_sessions)}")
        print(f"Subject-session combinations: {[f'sub-{s} ses-{ses}' for s, ses in subjects_sessions]}")

if __name__ == "__main__":
    main()