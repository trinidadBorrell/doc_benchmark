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
        default="./results",
        help="Directory to save results (default: ./results)"
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
        subject_data: dict with subject IDs as keys and (X, y) tuples as values
        times: time points array
    """
    subject_data = {}
    times = None
    
    main_path = Path(main_path)
    
    for sub in subjects:
        subject_epochs = []
        subject_labels = []
        
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
                    
                    n_epochs, n_channels, n_times = data.shape
                    
                    subject_epochs.append(data)
                    subject_labels.extend([0] * n_epochs)  # Each timepoint gets label 0
                    
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
                    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

                    subject_epochs.append(data)
                    subject_labels.extend([1] * (n_epochs))  # Each timepoint gets label 1
                    
                    if times is None:
                        times = epochs.times
                    
                    if verbose:
                        print(f"    Loaded {fif_file.name}: {data.shape[0]} epochs, {data.shape[2]} timepoints")
                        
                except Exception as e:
                    print(f"    Error loading {fif_file}: {e}")
        
        # Concatenate epochs for this subject --> These are already concatenated
        #    X_subject = np.concatenate(subject_epochs, axis=0)  # Shape: (epochs, n_channels)
        #    y_subject = np.array(subject_labels)  # Shape: (total_timepoints,)

        if subject_epochs:
               subject_data[sub] = (subject_epochs, subject_labels)
        if verbose:
                print(f"Subject {sub} summary:")
                print(f"  Total timepoints: {subject_epochs[sub].shape[0]}")
                print(f"  Channels: {subject_epochs[sub].shape[1]}")
                print(f"  Original timepoints (label 0): {np.sum(subject_labels[sub] == 0)}")
                print(f"  Reconstructed timepoints (label 1): {np.sum(subject_labels[sub] == 1)}")
    
    if not subject_data:
        raise ValueError("No epochs were loaded. Check your file paths and naming convention.")
    
    print("\nOverall data summary:")
    print(f"  Total timepoints: {np.shape(list(list(subject_data.values())[0])[0][0])[2]}")
    
    return subject_data, times

def perform_decoding(subject_data, times, cv=3, n_jobs=None, verbose=False):
    """Perform classification with subject-wise data where each timepoint is a sample."""
    print(f"\nPerforming classification with {cv}-fold cross-validation...")
    
    # Create the classifier pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    
    # Debug: Check the structure of the first subject's data
    first_subject = list(subject_data.keys())[0]
    X_first, y_first = subject_data[first_subject]
    
    print(f"Debug info for subject {first_subject}:")
    print(f"  X_first type: {type(X_first)}, length: {len(X_first) if hasattr(X_first, '__len__') else 'no length'}")
    print(f"  y_first type: {type(y_first)}, length: {len(y_first) if hasattr(y_first, '__len__') else 'no length'}")
    
    if len(X_first) > 0:
        print(f"  X_first[0] shape: {np.shape(X_first[0])}")
    if len(y_first) > 0:
        print(f"  y_first[0] shape: {np.shape(y_first[0])}")
        print(f"  y_first[0] type: {type(y_first[0])}")
    
    # Extract and concatenate X and y data across all subjects
    X_list = []
    y_list = []
    
    for subject_id, (X_subject, y_subject) in subject_data.items():
        print(f"\nProcessing subject {subject_id}:")
        print(f"  X_subject length: {len(X_subject)}")
        print(f"  y_subject length: {len(y_subject)}")
        
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
        
        print(f"  Final shapes - X: {X_subject_concat.shape}, y: {y_subject_concat.shape}")
        
        X_list.append(X_subject_concat)
        y_list.append(y_subject_concat)
    
    # Concatenate across all subjects
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(" Final concatenated shapes:")
    print(f"  Total trials for decoding: {X.shape[0]}")
    print(f"  Number of channels: {X.shape[1]}")
    print(f"  Number of timepoints: {X.shape[2]}")
    print(f"  Number of labels: {y.shape[0]}")
    print(f"  Number of subjects: {len(subject_data)}")
    
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs)
    
    # Calculate mean and std
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    return scores, mean_score, std_score


def plot_results(mean_score, std_score, scores, times, output_dir):
    """Plot the classification results."""
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
    
    # Create subplots for both overall and time-resolved results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Overall classification performance
    ax1.bar(['Original vs Reconstructed'], [mean_score], 
           yerr=[std_score], capsize=10, color='blue', alpha=0.7)
    
    # Add chance level line
    ax1.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
    
    # Add value text on the bar
    ax1.text(0, mean_score + std_score + 0.01, f'{mean_score:.3f} ± {std_score:.3f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Formatting for overall plot
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.set_title("Overall Classification: Original vs Reconstructed EEG Data", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.4, max(0.6, mean_score + std_score + 0.1)])
    
    # Plot 2: Time-resolved classification performance
    if scores.ndim == 2:  # Shape: (n_folds, n_timepoints)
        mean_scores_time = np.mean(scores, axis=0)
        std_scores_time = np.std(scores, axis=0)
    else:  # Shape: (n_timepoints,)
        mean_scores_time = scores
        std_scores_time = np.zeros_like(scores)
    
    # Plot time-resolved AUC
    ax2.plot(times, mean_scores_time, 'b-', linewidth=2, label='Mean AUC')
    ax2.fill_between(times, 
                     mean_scores_time - std_scores_time,
                     mean_scores_time + std_scores_time,
                     alpha=0.3, color='blue', label='±1 std')
    
    # Add chance level line
    ax2.axhline(0.5, color="k", linestyle="--", label="Chance", alpha=0.7)
    
    # Formatting for time-resolved plot
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_title("Time-resolved Classification: Original vs Reconstructed EEG Data", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / "classification_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

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
    all_y = np.concatenate([y for X, y in subject_data.values()])
    class_counts = np.bincount(all_y)
    print("\nOverall class distribution:")
    print(f"  Original (0): {class_counts[0]} ({class_counts[0]/len(all_y)*100:.1f}%)")
    print(f"  Reconstructed (1): {class_counts[1]} ({class_counts[1]/len(all_y)*100:.1f}%)")
    
    # Perform classification
    all_scores, mean_score, std_score = perform_decoding(
        subject_data, times, cv=args.cv, n_jobs=args.n_jobs, verbose=args.verbose
    )
    
    # Plot and save results
    plot_results(mean_score, std_score, all_scores, times, args.output_dir)
    save_results(mean_score, std_score, all_scores, args.output_dir)
    
    print(f"\nAnalysis completed! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()