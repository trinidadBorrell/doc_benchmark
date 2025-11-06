#Pick 6 subjects (choosing, all, and random)

#Mean and std by epoch and sensor --> heatmaps

#Time frequency decomposition

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict
from pathlib import Path
import mne
import warnings
from scipy.stats import chi2
warnings.filterwarnings('ignore')

COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        # colour‑consistent theme
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"


def find_subjects_and_sessions(base_dir: str) -> List[Dict[str, str]]:
    """
    Find all subjects and sessions in the data directory.
    
    Args:
        base_dir: Base directory containing subject data
        
    Returns:
        List of dictionaries with 'subject_id', 'session', and 'path' keys
    """
    subjects_sessions = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Find all subject directories (sub-*)
    for sub_dir in sorted(base_path.glob('sub-*')):
        if sub_dir.is_dir():
            subject_id = sub_dir.name.split('-')[1]  # Extract ID from sub-{id}
            
            # Find all session directories (ses-*)
            for ses_dir in sorted(sub_dir.glob('ses-*')):
                if ses_dir.is_dir():
                    session = ses_dir.name.split('-')[1]  # Extract num from ses-{num}
                    subjects_sessions.append({
                        'subject_id': subject_id,
                        'session': session,
                        'path': str(ses_dir)
                    })
    
    return subjects_sessions


def load_epoch_data(subject_id: str, session: str, session_path: str, epoch_type: str = 'original') -> mne.Epochs:
    """
    Load epoch data for a specific subject and session.
    
    Args:
        subject_id: Subject ID (e.g., 'AA078')
        session: Session number (e.g., '01')
        session_path: Path to the session directory
        epoch_type: Type of epochs to load ('original' or 'recon')
        
    Returns:
        MNE Epochs object
    """
    # Construct filename: sub-{id}_ses-{num}_task-lg_acq-01_epo_{type}.fif
    filename = f"sub-{subject_id}_ses-{session}_task-lg_acq-01_epo_{epoch_type}.fif"
    filepath = op.join(session_path, filename)
    
    if not op.exists(filepath):
        raise FileNotFoundError(f"Epoch file not found: {filepath}")
    
    print(f"Loading {epoch_type} epochs: {filename}")
    epochs = mne.read_epochs(filepath, preload=True, verbose=False)
    
    return epochs


def load_subject_session_data(subject_id: str, session: str, session_path: str) -> Tuple[mne.Epochs, mne.Epochs]:
    """
    Load both original and reconstructed epoch data for a subject/session.
    
    Args:
        subject_id: Subject ID
        session: Session number
        session_path: Path to the session directory
        
    Returns:
        Tuple of (original_epochs, reconstructed_epochs)
    """
    original_epochs = load_epoch_data(subject_id, session, session_path, 'original')
    recon_epochs = load_epoch_data(subject_id, session, session_path, 'recon')
    
    return original_epochs, recon_epochs


class QualitativeAnalysis:
    def __init__(self, output_dir: str):
        """
        Initialize QualitativeAnalysis.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_analysis(self, fif_dir: str, num_subjects: int = 6):
        """
        Run the complete qualitative analysis.
        
        Args:
            fif_dir: Directory containing FIF data
            num_subjects: Number of subjects to analyze
        """
        print(f"Starting qualitative analysis on {num_subjects} subjects...")
        print(f"Data directory: {fif_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all available subjects and sessions
        subjects_sessions = find_subjects_and_sessions(fif_dir)
        print(f"Found {len(subjects_sessions)} subject-session combinations")
        
        if len(subjects_sessions) == 0:
            raise ValueError("No subjects/sessions found in the specified directory")
        
        # Select subjects to analyze (limit to num_subjects)
        selected = subjects_sessions[:num_subjects]
        print(f"\nAnalyzing {len(selected)} subject-session(s):")
        
        # Load and analyze data for each subject-session
        for item in selected:
            subject_id = item['subject_id']
            session = item['session']
            session_path = item['path']
            
            print(f"\n--- Subject: {subject_id}, Session: {session} ---")
            
            try:
                # Load both original and reconstructed epochs
                original_epochs, recon_epochs = load_subject_session_data(
                    subject_id, session, session_path
                )
                
                print(f"  Original epochs: {len(original_epochs)} epochs, {original_epochs.info['nchan']} channels")
                print(f"  Reconstructed epochs: {len(recon_epochs)} epochs, {recon_epochs.info['nchan']} channels")
                
                # Run analyses
                self.first_second_momentum(original_epochs, recon_epochs, subject_id, session)
                self.time_frequency_decomposition(original_epochs, recon_epochs, subject_id, session)
                
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue
            except Exception as e:
                print(f"  Error processing subject {subject_id}, session {session}: {e}")
                continue
        
        print("\nQualitative analysis complete!")
    
    def first_second_momentum(self, original_epochs: mne.Epochs, recon_epochs: mne.Epochs,
                              subject_id: str, session: str):
        """
        Calculate and visualize mean and std by epoch and sensor using heatmaps.
        
        Args:
            original_epochs: Original epochs data
            recon_epochs: Reconstructed epochs data
            subject_id: Subject ID
            session: Session number
        """
        print("  Computing first and second momentum (mean and std)...")
        
        # Get data arrays (epochs x channels x times)
        orig_data = original_epochs.get_data()
        recon_data = recon_epochs.get_data()
        
        # Compute mean and std across time for each epoch and channel
        orig_mean = np.mean(orig_data, axis=2)  
        orig_std = np.std(orig_data, axis=2)
        recon_mean = np.mean(recon_data, axis=2)
        recon_std = np.std(recon_data, axis=2)
        
        # Create heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sns.heatmap(orig_mean.T, ax=axes[0, 0], cmap='RdBu_r', center=0)
        axes[0, 0].set_title(f'Original Mean - sub-{subject_id}_ses-{session}')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Channels')
        
        sns.heatmap(orig_std.T, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Original Std - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Channels')
        
        sns.heatmap(recon_mean.T, ax=axes[1, 0], cmap='RdBu_r', center=0)
        axes[1, 0].set_title(f'Reconstructed Mean - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Channels')
        
        sns.heatmap(recon_std.T, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Std - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Channels')
        
        plt.tight_layout()
        os.makedirs(os.path.join(self.output_dir, f'sub-{subject_id}'), exist_ok=True)
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'momentum_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved momentum heatmaps to: {output_file}")
        
        # Additional plots: Time vs Epochs and Time vs Channels
        # Plot 1: Time vs Epochs (mean across channels)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original: mean and std across channels for each epoch and time point
        orig_time_epoch_mean = np.mean(orig_data, axis=1)  # Shape: (epochs, times)
        orig_time_epoch_std = np.std(orig_data, axis=1)
        recon_time_epoch_mean = np.mean(recon_data, axis=1)
        recon_time_epoch_std = np.std(recon_data, axis=1)
        
        times = original_epochs.times
        
        sns.heatmap(orig_time_epoch_mean, ax=axes[0, 0], cmap='RdBu_r', center=0)
        axes[0, 0].set_title(f'Original Mean (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[0, 0].set_xlabel('Time (samples)')
        axes[0, 0].set_ylabel('Epochs')
        
        sns.heatmap(orig_time_epoch_std, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Original Std (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Time (samples)')
        axes[0, 1].set_ylabel('Epochs')
        
        sns.heatmap(recon_time_epoch_mean, ax=axes[1, 0], cmap='RdBu_r', center=0)
        axes[1, 0].set_title(f'Reconstructed Mean (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Time (samples)')
        axes[1, 0].set_ylabel('Epochs')
        
        sns.heatmap(recon_time_epoch_std, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Std (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Time (samples)')
        axes[1, 1].set_ylabel('Epochs')
        
        plt.tight_layout()
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'time_epochs_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved time×epochs heatmaps to: {output_file}")
        
        # Plot 2: Time vs Channels (mean across epochs)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean and std across epochs for each channel and time point
        orig_time_channel_mean = np.mean(orig_data, axis=0)  # Shape: (channels, times)
        orig_time_channel_std = np.std(orig_data, axis=0)
        recon_time_channel_mean = np.mean(recon_data, axis=0)
        recon_time_channel_std = np.std(recon_data, axis=0)
        
        sns.heatmap(orig_time_channel_mean, ax=axes[0, 0], cmap='RdBu_r', center=0)
        axes[0, 0].set_title(f'Original Mean (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[0, 0].set_xlabel('Time (samples)')
        axes[0, 0].set_ylabel('Channels')
        
        sns.heatmap(orig_time_channel_std, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Original Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Time (samples)')
        axes[0, 1].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_mean, ax=axes[1, 0], cmap='RdBu_r', center=0)
        axes[1, 0].set_title(f'Reconstructed Mean (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Time (samples)')
        axes[1, 0].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_std, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Time (samples)')
        axes[1, 1].set_ylabel('Channels')
        
        plt.tight_layout()
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'time_channels_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved time×channels heatmaps to: {output_file}")

    def time_frequency_decomposition(self, original_epochs: mne.Epochs, recon_epochs: mne.Epochs,
                                     subject_id: str, session: str):
        """
        Perform time-frequency decomposition on the data.
        
        Args:
            original_epochs: Original epochs data
            recon_epochs: Reconstructed epochs data
            subject_id: Subject ID
            session: Session number
        """
        print("  Computing time-frequency decomposition...")
        
        # Define frequencies of interest (e.g., 1-40 Hz)
        freqs = np.arange(1, 41, 1)
        n_cycles = freqs / 2.0  # Different number of cycles per frequency
        
        # Compute power for a subset of channels (to save time)
        # Pick first channel for demonstration
        picks = [0]
        
        # Compute power for original epochs using new API
        orig_power = original_epochs.compute_tfr(
            method='morlet', freqs=freqs, n_cycles=n_cycles, 
            picks=picks, return_itc=False, average=True, verbose=False
        )
        
        # Compute power for reconstructed epochs using new API
        recon_power = recon_epochs.compute_tfr(
            method='morlet', freqs=freqs, n_cycles=n_cycles,
            picks=picks, return_itc=False, average=True, verbose=False
        )
        
        # Plot time-frequency representations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        orig_power.plot(picks=0, axes=axes[0], show=False, colorbar=True)
        axes[0].set_title(f'Original TFR - sub-{subject_id}_ses-{session}')
        
        recon_power.plot(picks=0, axes=axes[1], show=False, colorbar=True)
        axes[1].set_title(f'Reconstructed TFR - sub-{subject_id}_ses-{session}')
        
        plt.tight_layout()
        output_file = op.join(self.output_dir,f'sub-{subject_id}', f'tfr_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved TFR plots to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Qualitative Analysis of EEG Data')
    parser.add_argument('--fif-dir', required=True, 
                        help='Directory containing FIF data (e.g., /data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata)')
    parser.add_argument('--num-subjects', type=int, default=6,
                        help='Number of subjects to analyze (default: 6)')
    parser.add_argument('--output-dir', default='./results/qualitative_analysis/',
                        help='Directory to save analysis results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run analysis
    analysis = QualitativeAnalysis(output_dir=args.output_dir)
    analysis.run_analysis(fif_dir=args.fif_dir, num_subjects=args.num_subjects)


if __name__ == '__main__':
    main()
