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

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
        default=3,
        help="Number of cross-validation folds (default: 3)"
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

def load_epochs_from_files(main_path, subjects, sessions, verbose=False):
    """
    Load and concatenate epochs from .fif files per subject.
    
    Returns:
        subject_data: dict with subject IDs as keys and (X, y, events) tuples as values
        times: time points array
    """
    subject_data = {}
    times = None
    
    main_path = Path(main_path)
    
    for sub in subjects:
        subject_epochs = []
        subject_labels = []
        subject_events = []
        
        for ses in sessions:
            # Construct the directory path
            sub_ses_dir = main_path / f"sub-{sub}" / f"ses-{ses}"
            
            if not sub_ses_dir.exists():
                if verbose:
                    print(f"Directory not found: {sub_ses_dir}")
                continue
            
            # Find original and reconstructed files
            original_files = list(sub_ses_dir.glob("*_epo_original.fif"))
            recon_files = list(sub_ses_dir.glob("*_epo_recon.fif"))
            
            if verbose:
                print(f"Subject {sub}, Session {ses}:")
                print(f"  Original files: {len(original_files)}")
                print(f"  Reconstructed files: {len(recon_files)}")
            
            # Process original files (label 0)
            for fif_file in original_files:
                try:
                    epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
                    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
                    events = epochs.events[:, 2]  # Extract event IDs
                    
                    n_epochs, n_channels, n_times = data.shape
                    
                    subject_epochs.append(data)
                    subject_labels.extend([0] * n_epochs)  # Each epoch gets label 0
                    subject_events.extend(events)  # Store event IDs for each epoch
                    
                    if times is None:
                        times = epochs.times
                    
                    if verbose:
                        print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs, {data.shape[2]} timepoints")
                        print(f"    Event types: {np.unique(events)}")
                        
                except Exception as e:
                    print(f"    Error loading {fif_file}: {e}")
            
            # Process reconstructed files (label 1)
            for fif_file in recon_files:
                try:
                    epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
                    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
                    events = epochs.events[:, 2]  # Extract event IDs

                    subject_epochs.append(data)
                    subject_labels.extend([1] * (n_epochs))  # Each epoch gets label 1
                    subject_events.extend(events)  # Store event IDs for each epoch
                    
                    if times is None:
                        times = epochs.times
                    
                    if verbose:
                        print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs, {data.shape[2]} timepoints")
                        print(f"    Event types: {np.unique(events)}")
                        
                except Exception as e:
                    print(f"    Error loading {fif_file}: {e}")
        
        # Store data for this subject
        if subject_epochs:
            subject_data[sub] = (subject_epochs, subject_labels, subject_events)
            if verbose:
                print(f"Subject {sub} summary:")
                print(f"  Total epochs: {len(subject_labels)}")
                print(f"  Original epochs (label 0): {np.sum(np.array(subject_labels) == 0)}")
                print(f"  Reconstructed epochs (label 1): {np.sum(np.array(subject_labels) == 1)}")
                print(f"  Event types: {np.unique(subject_events)}")
    
    if not subject_data:
        raise ValueError("No epochs were loaded. Check your file paths and naming convention.")
    
    print("\nOverall data summary:")
    first_subject_data = list(subject_data.values())[0]
    print(f"  Total timepoints: {np.shape(first_subject_data[0][0])[2]}")
    
    return subject_data, times

def perform_decoding(subject_data, times, cv=3, n_jobs=None, verbose=False):
    """Perform classification with subject-wise data where each epoch is a sample."""
    print(f"\nPerforming classification with {cv}-fold cross-validation...")
    
    # Create the classifier pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Debug: Check the structure of the first subject's data
    first_subject = list(subject_data.keys())[0]
    X_first, y_first, events_first = subject_data[first_subject]
    
    print(f"Debug info for subject {first_subject}:")
    print(f"  X_first type: {type(X_first)}, length: {len(X_first) if hasattr(X_first, '__len__') else 'no length'}")
    print(f"  y_first type: {type(y_first)}, length: {len(y_first) if hasattr(y_first, '__len__') else 'no length'}")
    print(f"  events_first type: {type(events_first)}, length: {len(events_first) if hasattr(events_first, '__len__') else 'no length'}")
    
    if len(X_first) > 0:
        print(f"  X_first[0] shape: {np.shape(X_first[0])}")
    if len(y_first) > 0:
        print(f"  y_first[0] shape: {np.shape(y_first[0])}")
        print(f"  y_first[0] type: {type(y_first[0])}")
    
    # Extract and concatenate X, y, and events data across all subjects
    X_list = []
    y_list = []
    events_list = []
    
    for subject_id, (X_subject, y_subject, events_subject) in subject_data.items():
        print(f"\nProcessing subject {subject_id}:")
        print(f"  X_subject length: {len(X_subject)}") # (trials, channels, timepoints)
        print(f"  y_subject length: {len(y_subject)}") # (trials,)
        print(f"  events_subject length: {len(events_subject)}") # (trials,)

        # Handle X data
        if len(X_subject) == 1:
            # If there's only one array, no need to concatenate
            X_subject_concat = X_subject[0]
        else:
            # Concatenate across the list elements for this subject
            X_subject_concat = np.concatenate(X_subject, axis=0)
        
        # Handle y data - check if elements are scalars or arrays
        if len(y_subject) == 1:
            # If there's only one element
            print('There is only one element')
            if np.ndim(y_subject[0]) == 0:
                print('It is a scalar')
                # It's a scalar, convert to array
                y_subject_concat = np.array([y_subject[0]])
            else:
                print('It is an array')
                # It's already an array
                y_subject_concat = y_subject[0]
        else:
            # Multiple elements - check if they're scalars or arrays
            if all(np.ndim(y) == 0 for y in y_subject):
                # All are scalars, convert to array
                y_subject_concat = np.array(y_subject)
            else:
                # They're arrays, concatenate them
                y_subject_concat = np.concatenate(y_subject, axis=0)
        
        # Handle events data
        events_subject_concat = np.array(events_subject)
        
        print(f"  Final shapes - X: {X_subject_concat.shape}, y: {y_subject_concat.shape}, events: {events_subject_concat.shape}")
                #X is (n_trials, n_channels, n_times)
                #y is (n_trials,)
                #events is (n_trials,)

        X_list.append(X_subject_concat)
        y_list.append(y_subject_concat)
        events_list.append(events_subject_concat)
    
    # Concatenate across all subjects
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    events = np.concatenate(events_list, axis=0)
    
    print("\nFinal concatenated shapes:")
    print(f"  Total trials for decoding: {X.shape[0]}")
    print(f"  Number of channels: {X.shape[1]}")
    print(f"  Number of timepoints: {X.shape[2]}")
    print(f"  Number of labels: {y.shape[0]}")
    print(f"  Number of events: {events.shape[0]}")
    print(f"  Number of subjects: {len(subject_data)}")
    print(f"  Event types present: {np.unique(events)}")
    
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs)
    
    # Calculate mean and std correctly
    # scores shape: (n_cv_folds, n_timepoints)
    if scores.ndim == 2:
        mean_scores_time = np.mean(scores, axis=0)  # Mean across CV folds for each timepoint
        std_scores_time = np.std(scores, axis=0)    # Std across CV folds for each timepoint
        overall_mean = np.mean(mean_scores_time)    # Overall mean across time
        overall_std = np.std(mean_scores_time)      # Std across time points
    else:
        # Fallback for 1D scores
        mean_scores_time = scores
        std_scores_time = np.zeros_like(scores)
        overall_mean = np.mean(scores)
        overall_std = np.std(scores)
    
    return scores, overall_mean, overall_std, X, y, events

def perform_trial_type_decoding(subject_data, times, cv=3, n_jobs=None, verbose=False):
    """Perform classification for each trial type separately."""
    print(f"\nPerforming trial-type-specific classification with {cv}-fold cross-validation...")
    
    # Create the classifier pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Extract and concatenate X, y, and events data across all subjects
    X_list = []
    y_list = []
    events_list = []
    
    for subject_id, (X_subject, y_subject, events_subject) in subject_data.items():
        # Handle X data
        if len(X_subject) == 1:
            X_subject_concat = X_subject[0]
        else:
            X_subject_concat = np.concatenate(X_subject, axis=0)
        
        # Handle y data
        if len(y_subject) == 1:
            if np.ndim(y_subject[0]) == 0:
                y_subject_concat = np.array([y_subject[0]])
            else:
                y_subject_concat = y_subject[0]
        else:
            if all(np.ndim(y) == 0 for y in y_subject):
                y_subject_concat = np.array(y_subject)
            else:
                y_subject_concat = np.concatenate(y_subject, axis=0)
        
        # Handle events data
        events_subject_concat = np.array(events_subject)
        
        X_list.append(X_subject_concat)
        y_list.append(y_subject_concat)
        events_list.append(events_subject_concat)
    
    # Concatenate across all subjects
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    events_all = np.concatenate(events_list, axis=0)
    
    print(f"\nTotal data: {X_all.shape[0]} trials, {X_all.shape[1]} channels, {X_all.shape[2]} timepoints")
    print(f"Event types present: {np.unique(events_all)}")
    
    # Target trial types for analysis (excluding controls)
    target_trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    target_event_ids = [30, 40, 50, 60]  # Corresponding event IDs
    
    trial_scores_dict = {}
    
    for trial_type, event_id in zip(target_trial_types, target_event_ids):
        # Filter data for this trial type
        trial_mask = events_all == event_id
        
        if np.sum(trial_mask) == 0:
            print(f"\nNo trials found for {trial_type} (event_id {event_id})")
            continue
            
        X_trial = X_all[trial_mask]
        y_trial = y_all[trial_mask]
        
        print(f"\nProcessing {trial_type} (event_id {event_id}):")
        print(f"  Total trials: {X_trial.shape[0]}")
        print(f"  Original trials: {np.sum(y_trial == 0)}")
        print(f"  Reconstructed trials: {np.sum(y_trial == 1)}")
        
        # Check if we have enough trials and both classes
        if X_trial.shape[0] < cv or len(np.unique(y_trial)) < 2:
            print(f"  Insufficient data for cross-validation (need at least {cv} trials and both classes)")
            continue
        
        # Perform classification for this trial type
        time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring="roc_auc", verbose=verbose)
        try:
            scores = cross_val_multiscore(time_decod, X_trial, y_trial, cv=cv, n_jobs=n_jobs)
            trial_scores_dict[trial_type] = scores
            
            # Calculate mean and std correctly
            if scores.ndim == 2:
                mean_scores_time = np.mean(scores, axis=0)  # Mean across CV folds for each timepoint
                overall_mean = np.mean(mean_scores_time)    # Overall mean across time
                overall_std = np.std(mean_scores_time)      # Std across time points
            else:
                overall_mean = np.mean(scores)
                overall_std = np.std(scores)
            
            print(f"  Mean AUC: {overall_mean:.3f} ± {overall_std:.3f}")
            
        except Exception as e:
            print(f"  Error in classification: {e}")
    
    return trial_scores_dict

def perform_local_global_analysis(subject_data, times, cv=3, n_jobs=None, verbose=False):
    """Perform local vs global effect analysis for original and reconstructed data separately."""
    print(f"\nPerforming local/global effect analysis with {cv}-fold cross-validation...")
    
    # Create the classifier pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Extract and concatenate all data
    X_list = []
    y_list = []
    events_list = []
    
    for subject_id, (X_subject, y_subject, events_subject) in subject_data.items():
        # Handle X data
        if len(X_subject) == 1:
            X_subject_concat = X_subject[0]
        else:
            X_subject_concat = np.concatenate(X_subject, axis=0)
        
        # Handle y data
        if len(y_subject) == 1:
            if np.ndim(y_subject[0]) == 0:
                y_subject_concat = np.array([y_subject[0]])
            else:
                y_subject_concat = y_subject[0]
        else:
            if all(np.ndim(y) == 0 for y in y_subject):
                y_subject_concat = np.array(y_subject)
            else:
                y_subject_concat = np.concatenate(y_subject, axis=0)
        
        # Handle events data
        events_subject_concat = np.array(events_subject)
        
        X_list.append(X_subject_concat)
        y_list.append(y_subject_concat)
        events_list.append(events_subject_concat)
    
    # Concatenate across all subjects
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    events_all = np.concatenate(events_list, axis=0)
    
    # Define event mappings for analysis
    # LSGS=30, LSGD=40, LDGD=50, LDGS=60
    local_standard_events = [30, 40]  # LSGS, LSGD
    local_deviant_events = [50, 60]   # LDGD, LDGS
    global_standard_events = [30, 60] # LSGS, LDGS
    global_deviant_events = [40, 50]  # LSGD, LDGD
    
    results = {}
    
    # Filter for relevant events (exclude controls 10, 20)
    relevant_mask = np.isin(events_all, [30, 40, 50, 60])
    X_relevant = X_all[relevant_mask]
    y_relevant = y_all[relevant_mask]
    events_relevant = events_all[relevant_mask]
    
    print(f"Total relevant trials: {X_relevant.shape[0]}")
    
    # For each data type (original=0, reconstructed=1)
    for data_type in [0, 1]:
        data_name = 'original' if data_type == 0 else 'reconstructed'
        print(f"\nAnalyzing {data_name} data...")
        
        # Filter by data type
        data_mask = y_relevant == data_type
        X_data = X_relevant[data_mask]
        events_data = events_relevant[data_mask]
        
        print(f"  {data_name.capitalize()} trials: {X_data.shape[0]}")
        
        if X_data.shape[0] == 0:
            print(f"  No {data_name} data found")
            continue
            
        # Local effect analysis: Local Standard vs Local Deviant
        local_mask = np.isin(events_data, local_standard_events + local_deviant_events)
        if np.sum(local_mask) > 0:
            X_local = X_data[local_mask]
            events_local = events_data[local_mask]
            
            # Create binary labels: 0=Local Standard, 1=Local Deviant
            y_local = np.isin(events_local, local_deviant_events).astype(int)
            
            print(f"  Local effect - Local Standard: {np.sum(y_local == 0)}, Local Deviant: {np.sum(y_local == 1)}")
            
            if len(np.unique(y_local)) == 2 and X_local.shape[0] >= cv:
                time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring="roc_auc", verbose=False)
                try:
                    scores_local = cross_val_multiscore(time_decod, X_local, y_local, cv=cv, n_jobs=n_jobs)
                    results[f'{data_name}_local'] = scores_local
                except Exception as e:
                    print(f"    Error in local analysis: {e}")
        
        # Global effect analysis: Global Standard vs Global Deviant
        global_mask = np.isin(events_data, global_standard_events + global_deviant_events)
        if np.sum(global_mask) > 0:
            X_global = X_data[global_mask]
            events_global = events_data[global_mask]
            
            # Create binary labels: 0=Global Deviant, 1=Global Standard
            y_global = np.isin(events_global, global_standard_events).astype(int)
            
            print(f"  Global effect - Global Deviant: {np.sum(y_global == 0)}, Global Standard: {np.sum(y_global == 1)}")
            
            if len(np.unique(y_global)) == 2 and X_global.shape[0] >= cv:
                time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring="roc_auc", verbose=False)
                try:
                    scores_global = cross_val_multiscore(time_decod, X_global, y_global, cv=cv, n_jobs=n_jobs)
                    results[f'{data_name}_global'] = scores_global
                except Exception as e:
                    print(f"    Error in global analysis: {e}")
    
    return results


def plot_overall_results(mean_score, std_score, output_dir, title_suffix=""):
    """Plot the overall classification results."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure mean_score and std_score are scalars
    if hasattr(mean_score, '__len__') and len(mean_score) > 1:
        mean_score = np.mean(mean_score)
    elif hasattr(mean_score, 'item'):
        mean_score = mean_score.item()
    
    if hasattr(std_score, '__len__') and len(std_score) > 1:
        std_score = np.std(std_score)
    elif hasattr(std_score, 'item'):
        std_score = std_score.item()
    
    # Create figure for overall classification performance
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.bar(['Original vs Reconstructed'], [mean_score], 
           yerr=[std_score], capsize=10, color='blue', alpha=0.7)
    
    # Add chance level line
    ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
    
    # Add value text on the bar
    ax.text(0, mean_score + std_score + 0.01, f'{mean_score:.3f} ± {std_score:.3f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Formatting for overall plot
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"Overall Classification: Original vs Reconstructed EEG Data{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.4, max(0.6, mean_score + std_score + 0.1)])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / f"overall_classification_results{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Overall plot saved to: {plot_path}")
    
    plt.show()
    return fig, ax

def plot_time_resolved_results(scores, times, output_dir, title_suffix=""):
    """Plot the time-resolved classification results."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure for time-resolved classification performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    if scores.ndim == 2:  # Shape: (n_folds, n_timepoints)
        mean_scores_time = np.mean(scores, axis=0)
        std_scores_time = np.std(scores, axis=0)
    else:  # Shape: (n_timepoints,)
        mean_scores_time = scores
        std_scores_time = np.zeros_like(scores)
    
    # Plot time-resolved AUC
    ax.plot(times, mean_scores_time, 'b-', linewidth=2, label='Mean AUC')
    ax.fill_between(times, 
                     mean_scores_time - std_scores_time,
                     mean_scores_time + std_scores_time,
                     alpha=0.3, color='blue', label='±1 std')
    
    # Add chance level line
    ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
    
    # Formatting for time-resolved plot
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"Time-resolved Classification: Original vs Reconstructed EEG Data{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / f"time_resolved_classification_results{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Time-resolved plot saved to: {plot_path}")
    
    plt.show()
    return fig, ax

def plot_trial_type_results(trial_scores_dict, times, output_dir):
    """Plot 2x2 subplots for trial-type-specific AUC across time."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define trial types we want to plot (excluding controls)
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    colors = ['blue', 'red', 'green', 'orange']
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, trial_type in enumerate(trial_types):
        if trial_type in trial_scores_dict:
            scores = trial_scores_dict[trial_type]
            
            if scores.ndim == 2:  # Shape: (n_folds, n_timepoints)
                mean_scores = np.mean(scores, axis=0)
                std_scores = np.std(scores, axis=0)
            else:  # Shape: (n_timepoints,)
                mean_scores = scores
                std_scores = np.zeros_like(scores)
            
            # Plot time-resolved AUC
            axes[i].plot(times, mean_scores, color=colors[i], linewidth=2, label=f'Mean AUC ({trial_type})')
            axes[i].fill_between(times, 
                               mean_scores - std_scores,
                               mean_scores + std_scores,
                               alpha=0.3, color=colors[i], label='±1 std')
            
            # Add chance level line
            axes[i].axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
            
            # Formatting
            axes[i].set_xlabel("Time (s)", fontsize=10)
            axes[i].set_ylabel("AUC", fontsize=10)
            axes[i].set_title(f"Classification AUC: {trial_type}", fontsize=12)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([0.4, 1.0])
        else:
            # If trial type not found, show empty plot with message
            axes[i].text(0.5, 0.5, f'No data for {trial_type}', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f"Classification AUC: {trial_type}", fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / "trial_type_classification_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Trial type plot saved to: {plot_path}")
    
    plt.show()
    return fig, axes

def plot_local_global_effects(local_global_results, times, output_dir):
    """Plot 2x2 subplots for local/global effects in original and reconstructed data."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define the analyses
    analyses = [
        ('original_local', 'Original Data - Local Effect\n(Local Standard vs Local Deviant)', 0, 0),
        ('original_global', 'Original Data - Global Effect\n(Global Standard vs Global Deviant)', 1, 0),
        ('reconstructed_local', 'Reconstructed Data - Local Effect\n(Local Standard vs Local Deviant)', 0, 1),
        ('reconstructed_global', 'Reconstructed Data - Global Effect\n(Global Standard vs Global Deviant)', 1, 1)
    ]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (key, title, row, col) in enumerate(analyses):
        ax = axes[row, col]
        
        if key in local_global_results:
            scores = local_global_results[key]
            
            if scores.ndim == 2:  # Shape: (n_folds, n_timepoints)
                mean_scores = np.mean(scores, axis=0)
                std_scores = np.std(scores, axis=0)
            else:  # Shape: (n_timepoints,)
                mean_scores = scores
                std_scores = np.zeros_like(scores)
            
            # Plot time-resolved AUC
            ax.plot(times, mean_scores, color=colors[i], linewidth=2, label='Mean AUC')
            ax.fill_between(times, 
                           mean_scores - std_scores,
                           mean_scores + std_scores,
                           alpha=0.3, color=colors[i], label='±1 std')
            
            # Add chance level line
            ax.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
            
            # Calculate overall performance for title
            overall_mean = np.mean(mean_scores)
            ax.set_title(f"{title}\nMean AUC: {overall_mean:.3f}", fontsize=10)
            
        else:
            # If analysis not found, show empty plot with message
            ax.text(0.5, 0.5, f'No data for\n{title}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=10)
        
        # Formatting
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("AUC", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / "local_global_effects_classification.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Local/Global effects plot saved to: {plot_path}")
    
    plt.show()
    return fig, axes

def save_results(mean_score, std_score, scores, output_dir):
    """Save numerical results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save classification results
    results_path = Path(output_dir) / "classification_results.txt"
    with open(results_path, 'w') as f:
        f.write("Classification Results\n")
        f.write("=====================\n")
        f.write(f"Mean AUC: {mean_score:.6f}\n")
        f.write(f"Std AUC: {std_score:.6f}\n")
        f.write(f"Individual fold scores: {scores}\n")
    print(f"Results saved to: {results_path}")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("EEG Original vs Reconstructed Decoder")
    print("=" * 40)
    
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
    
    # Load epochs
    subject_data, times = load_epochs_from_files(args.main_path, subjects, sessions, args.verbose)
    
    # Check class balance across all subjects
    all_y = np.concatenate([y for X, y, events in subject_data.values()])
    all_events = np.concatenate([events for X, y, events in subject_data.values()])
    class_counts = np.bincount(all_y)
    print("\nOverall class distribution:")
    print(f"  Original (0): {class_counts[0]} ({class_counts[0]/len(all_y)*100:.1f}%)")
    print(f"  Reconstructed (1): {class_counts[1]} ({class_counts[1]/len(all_y)*100:.1f}%)")
    
    # Print event distribution
    unique_events, event_counts = np.unique(all_events, return_counts=True)
    print("\nEvent distribution:")
    for event_id, count in zip(unique_events, event_counts):
        event_name = event_id_mapping.get(event_id, f"Unknown_{event_id}")
        print(f"  {event_name} ({event_id}): {count} trials ({count/len(all_events)*100:.1f}%)")
    
    # Perform overall classification
    all_scores, mean_score, std_score, X, y, events = perform_decoding(
        subject_data, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
    )
    
    # Perform trial-type-specific classification
    trial_scores_dict = perform_trial_type_decoding(
        subject_data, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
    )
    
    # Perform local/global effect analysis
    local_global_results = perform_local_global_analysis(
        subject_data, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
    )
    
    # Plot and save results
    plot_overall_results(mean_score, std_score, args.output_dir)
    plot_time_resolved_results(all_scores, times, args.output_dir)
    plot_trial_type_results(trial_scores_dict, times, args.output_dir)
    plot_local_global_effects(local_global_results, times, args.output_dir)
    save_results(mean_score, std_score, all_scores, args.output_dir)
    
    print(f"\nAnalysis completed! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()