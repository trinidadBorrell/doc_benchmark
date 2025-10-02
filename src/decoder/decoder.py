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
                    data = epochs.get_data(copy=False).astype(np.float32)  # Use float32 to save memory
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
                    data = epochs.get_data(copy=False).astype(np.float32)  # Use float32 to save memory
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
    """
    Perform within-subject classification (one decoder per subject).
    Returns AUC time series for each subject.
    """
    print(f"\nPerforming WITHIN-SUBJECT classification with {cv}-fold cross-validation...")
    print(f"Training one decoder per subject, then averaging across subjects")
    
    # Create the classifier pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Store results for each subject
    subject_scores = []
    subject_ids = []
    all_X = []
    all_y = []
    all_events = []
    
    # Process each subject independently
    for subject_id, (X_subject, y_subject, events_subject) in subject_data.items():
        print(f"\nProcessing subject {subject_id}:")
        print(f"  X_subject length: {len(X_subject)}")
        print(f"  y_subject length: {len(y_subject)}")
        print(f"  events_subject length: {len(events_subject)}")

        # Handle X data - concatenate sessions for this subject
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
        
        print(f"  Shape - X: {X_subject_concat.shape}, y: {y_subject_concat.shape}, events: {events_subject_concat.shape}")
        print(f"  Original trials: {np.sum(y_subject_concat == 0)}, Reconstructed trials: {np.sum(y_subject_concat == 1)}")
        
        # Check if we have enough data for this subject
        if len(np.unique(y_subject_concat)) < 2:
            print(f"  WARNING: Subject {subject_id} has only one class, skipping...")
            continue
        
        if X_subject_concat.shape[0] < cv:
            print(f"  WARNING: Subject {subject_id} has fewer trials ({X_subject_concat.shape[0]}) than CV folds ({cv}), skipping...")
            continue
        
        # Train decoder for this subject
        time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
        try:
            scores = cross_val_multiscore(time_decod, X_subject_concat, y_subject_concat, cv=cv, n_jobs=1)
            # scores shape: (n_cv_folds, n_timepoints)
            
            # Average across CV folds to get this subject's AUC time series
            if scores.ndim == 2:
                subject_auc_timeseries = np.mean(scores, axis=0)  # (n_timepoints,)
            else:
                subject_auc_timeseries = scores
            
            subject_scores.append(subject_auc_timeseries)
            subject_ids.append(subject_id)
            
            overall_auc = np.mean(subject_auc_timeseries)
            print(f"  ✓ Subject {subject_id} mean AUC: {overall_auc:.3f}")
            
            # Store data for overall statistics
            all_X.append(X_subject_concat)
            all_y.append(y_subject_concat)
            all_events.append(events_subject_concat)
            
        except Exception as e:
            print(f"  ERROR: Failed to decode subject {subject_id}: {e}")
            continue
    
    # Convert to array: (n_subjects, n_timepoints)
    subject_scores = np.array(subject_scores)
    
    print(f"\n{'='*60}")
    print(f"Successfully decoded {len(subject_ids)} subjects")
    print(f"Subject IDs: {subject_ids}")
    print(f"Scores shape: {subject_scores.shape}")
    
    # Calculate mean and std ACROSS SUBJECTS for each timepoint
    mean_auc_time = np.mean(subject_scores, axis=0)  # (n_timepoints,)
    std_auc_time = np.std(subject_scores, axis=0)    # (n_timepoints,)
    
    # Overall statistics
    overall_mean = np.mean(mean_auc_time)
    overall_std = np.std(mean_auc_time)
    
    print(f"\nOverall Results (across subjects):")
    print(f"  Mean AUC: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"  Peak AUC: {np.max(mean_auc_time):.3f} at time {times[np.argmax(mean_auc_time)]:.3f}s")
    print(f"{'='*60}\n")
    
    # Concatenate all data for return (for compatibility with existing code)
    X_all = np.concatenate(all_X, axis=0) if all_X else None
    y_all = np.concatenate(all_y, axis=0) if all_y else None
    events_all = np.concatenate(all_events, axis=0) if all_events else None
    
    # Return: subject_scores (n_subjects, n_timepoints), mean, std, and concatenated data
    return subject_scores, overall_mean, overall_std, X_all, y_all, events_all

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
    """
    Plot the time-resolved classification results.
    
    Parameters:
    -----------
    scores : array
        Shape (n_subjects, n_timepoints) - AUC time series for each subject
    times : array
        Time points
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure for time-resolved classification performance
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    if scores.ndim == 2:  # Shape: (n_subjects, n_timepoints)
        n_subjects = scores.shape[0]
        
        # Calculate mean and std ACROSS SUBJECTS
        mean_scores_time = np.mean(scores, axis=0)  # (n_timepoints,)
        std_scores_time = np.std(scores, axis=0)    # (n_timepoints,)
        sem_scores_time = std_scores_time / np.sqrt(n_subjects)  # Standard error
        
        # Plot individual subjects (light lines)
        for i in range(n_subjects):
            ax.plot(times, scores[i, :], 'gray', linewidth=0.5, alpha=0.3)
        
        # Plot mean AUC across subjects (bold line)
        ax.plot(times, mean_scores_time, 'b-', linewidth=3, label=f'Mean AUC (n={n_subjects} subjects)', zorder=10)
        
        # Fill std band
        ax.fill_between(times, 
                         mean_scores_time - std_scores_time,
                         mean_scores_time + std_scores_time,
                         alpha=0.3, color='blue', label='±1 SD (across subjects)', zorder=5)
        
    else:  # Shape: (n_timepoints,) - fallback for old format
        mean_scores_time = scores
        std_scores_time = np.zeros_like(scores)
        ax.plot(times, mean_scores_time, 'b-', linewidth=2, label='Mean AUC')
    
    # Add chance level line
    ax.axhline(0.5, color="k", linestyle="--", label="Chance level", alpha=0.7, linewidth=2)
    
    # Add vertical line at t=0 (stimulus onset)
    ax.axvline(0, color="red", linestyle=":", label="Stimulus onset", alpha=0.5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
    ax.set_ylabel("AUC (Area Under ROC Curve)", fontsize=14, fontweight='bold')
    ax.set_title(f"Within-Subject Decoding: Original vs Reconstructed EEG{title_suffix}\nMean ± SD across subjects", 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.40, 1.0])
    
    # Add text with peak AUC
    peak_idx = np.argmax(mean_scores_time)
    peak_auc = mean_scores_time[peak_idx]
    peak_time = times[peak_idx]
    ax.text(0.02, 0.98, f'Peak: AUC={peak_auc:.3f} at t={peak_time:.3f}s', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / f"time_resolved_within_subject_classification{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Time-resolved plot saved to: {plot_path}")
    
    plt.close(fig)
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

def save_results(mean_score, std_score, scores, output_dir, trial_scores_dict=None, 
                local_global_results=None, n_subjects=0, model_info=None, subject_ids=None):
    """Save numerical results to text and JSON files."""
    import json
    from datetime import datetime
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save classification results as text
    results_path = Path(output_dir) / "classification_results.txt"
    with open(results_path, 'w') as f:
        f.write("Within-Subject Classification Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Type: Within-subject decoding\n")
        f.write(f"Number of subjects: {n_subjects}\n")
        f.write(f"Mean AUC (across subjects): {mean_score:.6f}\n")
        f.write(f"Std AUC (across subjects): {std_score:.6f}\n")
        f.write(f"\nSubject-wise results shape: {scores.shape}\n")
        if subject_ids:
            f.write(f"Subject IDs: {subject_ids}\n")
    print(f"Results saved to: {results_path}")
    
    # Prepare comprehensive JSON results
    json_results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "within_subject",
        "n_subjects": n_subjects,
        "subject_ids": subject_ids if subject_ids else [],
        "model_info": model_info if model_info else {
            "classifier": "LogisticRegression",
            "solver": "liblinear",
            "preprocessing": "StandardScaler",
            "approach": "within_subject_sliding_estimator",
            "scoring": "roc_auc"
        },
        "overall_classification": {
            "mean_auc": float(mean_score),
            "std_auc": float(std_score),
            "description": "Mean and std calculated across subjects",
            "per_subject_auc_timeseries": scores.tolist() if scores is not None else None
        }
    }
    
    # Add trial-type-specific results
    if trial_scores_dict:
        json_results["trial_type_classification"] = {}
        for trial_type, trial_scores in trial_scores_dict.items():
            if trial_scores.ndim == 2:
                mean_scores = np.mean(trial_scores, axis=0)
                overall_mean = np.mean(mean_scores)
                overall_std = np.std(mean_scores)
            else:
                mean_scores = trial_scores
                overall_mean = np.mean(trial_scores)
                overall_std = np.std(trial_scores)
            
            json_results["trial_type_classification"][trial_type] = {
                "mean_auc": float(overall_mean),
                "std_auc": float(overall_std),
                "auc_time_series": trial_scores.tolist()
            }
    
    # Add local/global effect results
    if local_global_results:
        json_results["local_global_effects"] = {}
        for effect_name, effect_scores in local_global_results.items():
            if effect_scores.ndim == 2:
                mean_scores = np.mean(effect_scores, axis=0)
                overall_mean = np.mean(mean_scores)
                overall_std = np.std(mean_scores)
            else:
                mean_scores = effect_scores
                overall_mean = np.mean(effect_scores)
                overall_std = np.std(effect_scores)
            
            json_results["local_global_effects"][effect_name] = {
                "mean_auc": float(overall_mean),
                "std_auc": float(overall_std),
                "auc_time_series": effect_scores.tolist()
            }
    
    # Save JSON results
    json_path = Path(output_dir) / "decoder_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")

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
    
    # Save comprehensive results including JSON
    save_results(
        mean_score, std_score, all_scores, args.output_dir,
        trial_scores_dict=trial_scores_dict,
        local_global_results=local_global_results,
        n_subjects=all_scores.shape[0] if all_scores.ndim == 2 else len(subjects),
        subject_ids=list(subject_data.keys())
    )
    
    print(f"\nAnalysis completed! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()