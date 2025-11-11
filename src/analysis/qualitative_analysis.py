#Pick 6 subjects (choosing, all, and random)

#Mean and std by epoch and sensor --> heatmaps

#Time frequency decomposition

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
import mne
import warnings
from scipy.stats import chi2, pearsonr
from sklearn.feature_selection import mutual_info_regression
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
        
    def run_analysis(self, fif_dir: str, num_subjects: int = 6, only_correlation_grid: bool = False):
        """
        Run the complete qualitative analysis.
        
        Args:
            fif_dir: Directory containing FIF data
            num_subjects: Number of subjects to analyze
            only_correlation_grid: If True, only generate the 3x3 correlation heatmaps grid
        """
        print(f"Starting qualitative analysis on {num_subjects} subjects...")
        print(f"Data directory: {fif_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all available subjects and sessions
        subjects_sessions = find_subjects_and_sessions(fif_dir)
        print(f"Found {len(subjects_sessions)} subject-session combinations")
        
        if len(subjects_sessions) == 0:
            raise ValueError("No subjects/sessions found in the specified directory")
        
        # If only generating correlation grid, skip individual analyses
        if only_correlation_grid:
            print("\n--- Generating correlation heatmaps grid only ---")
            try:
                self.correlation_heatmaps_grid(fif_dir)
            except Exception as e:
                print(f"  Error generating correlation heatmaps grid: {e}")
            print("\nQualitative analysis complete!")
            return
        
        # Select subjects to analyze (limit to num_subjects)
        # selected = subjects_sessions[:num_subjects]
        selected = random.sample(subjects_sessions, num_subjects)

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
                self.pixel_corr(original_epochs, recon_epochs, subject_id, session)
                
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue
            except Exception as e:
                print(f"  Error processing subject {subject_id}, session {session}: {e}")
                continue
        
        # Generate 3x3 correlation heatmaps grid for 9 random subjects
        print("\n--- Generating correlation heatmaps grid ---")
        try:
            self.correlation_heatmaps_grid(fif_dir)
        except Exception as e:
            print(f"  Error generating correlation heatmaps grid: {e}")
        
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
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Epochs')
        
        sns.heatmap(orig_time_epoch_std, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Original Std (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Epochs')
        
        sns.heatmap(recon_time_epoch_mean, ax=axes[1, 0], cmap='RdBu_r', center=0)
        axes[1, 0].set_title(f'Reconstructed Mean (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Epochs')
        
        sns.heatmap(recon_time_epoch_std, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Std (Time × Epochs) - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Time')
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
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Channels')
        
        sns.heatmap(orig_time_channel_std, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Original Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_mean, ax=axes[1, 0], cmap='RdBu_r', center=0)
        axes[1, 0].set_title(f'Reconstructed Mean (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_std, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Channels')
        
        plt.tight_layout()
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'time_channels_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved time×channels heatmaps to: {output_file}")
        
        # Plot 3: Time vs Channels with shared colormaps per column
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Compute shared colormap limits for mean column
        mean_vmin = min(orig_time_channel_mean.min(), recon_time_channel_mean.min())
        mean_vmax = max(orig_time_channel_mean.max(), recon_time_channel_mean.max())
        
        # Compute shared colormap limits for std column
        std_vmin = min(orig_time_channel_std.min(), recon_time_channel_std.min())
        std_vmax = max(orig_time_channel_std.max(), recon_time_channel_std.max())
        
        sns.heatmap(orig_time_channel_mean, ax=axes[0, 0], cmap='RdBu_r', 
                    center=0, vmin=mean_vmin, vmax=mean_vmax)
        axes[0, 0].set_title(f'Original Mean (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Channels')
        
        sns.heatmap(orig_time_channel_std, ax=axes[0, 1], cmap='viridis',
                    vmin=std_vmin, vmax=std_vmax)
        axes[0, 1].set_title(f'Original Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_mean, ax=axes[1, 0], cmap='RdBu_r', 
                    center=0, vmin=mean_vmin, vmax=mean_vmax)
        axes[1, 0].set_title(f'Reconstructed Mean (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Channels')
        
        sns.heatmap(recon_time_channel_std, ax=axes[1, 1], cmap='viridis',
                    vmin=std_vmin, vmax=std_vmax)
        axes[1, 1].set_title(f'Reconstructed Std (Time × Channels) - sub-{subject_id}_ses-{session}')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Channels')
        
        plt.tight_layout()
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'time_channels_shared_cmap_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved time×channels heatmaps (shared colormap) to: {output_file}")
        
        # Plot 4: Correlation and Mutual Information over time
        # Compute mean across channels for each time point
        # orig_time_channel_mean and recon_time_channel_mean are already (channels, times)
        # We need mean across channels: (times,)
        orig_mean_across_channels = np.mean(orig_time_channel_mean, axis=0)  # Shape: (times,)
        recon_mean_across_channels = np.mean(recon_time_channel_mean, axis=0)  # Shape: (times,)
        
        # Compute Pearson correlation for each time point
        correlations = []
        for t in range(len(times)):
            # Get all channel values at time t across epochs
            orig_vals = orig_data[:, :, t].flatten()  # All channel values at time t
            recon_vals = recon_data[:, :, t].flatten()
            corr, _ = pearsonr(orig_vals, recon_vals)
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Compute Mutual Information for each time point
        mutual_info = []
        for t in range(len(times)):
            orig_vals = orig_data[:, :, t].flatten().reshape(-1, 1)
            recon_vals = recon_data[:, :, t].flatten()
            mi = mutual_info_regression(orig_vals, recon_vals, random_state=42)[0]
            mutual_info.append(mi)
        
        mutual_info = np.array(mutual_info)
        
        # Create plot with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Subplot 1: Pearson Correlation over time
        axes[0].plot(times, correlations, linewidth=2, color='steelblue')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].set_title(f'Pearson Correlation between Original and Reconstructed - sub-{subject_id}_ses-{session}')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Subplot 2: Mutual Information over time
        axes[1].plot(times, mutual_info, linewidth=2, color='darkorange')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mutual Information')
        axes[1].set_title(f'Mutual Information between Original and Reconstructed - sub-{subject_id}_ses-{session}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = op.join(self.output_dir, f'sub-{subject_id}', f'correlation_mi_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved correlation and MI plots to: {output_file}")

    def time_frequency_decomposition(self, original_epochs: mne.Epochs, recon_epochs: mne.Epochs,
                                     subject_id: str, session: str):
        """
        Perform time-frequency decomposition on the data using Welch Method.
        
        Args:
            original_epochs: Original epochs data
            recon_epochs: Reconstructed epochs data
            subject_id: Subject ID
            session: Session number
        """
        print("  Computing time-frequency decomposition...")
        
        # Define frequencies of interest (e.g., 1-40 Hz)
        freqs = np.arange(0.5, 45, 1)
        n_cycles = freqs / 2.0  # Different number of cycles per frequency
        
        # Compute power for a subset of channels (to save time)
        # Pick first channel for demonstration
                
        # Compute power for original epochs using new API
        orig_power = original_epochs.compute_psd(method='welch', fmin = 0.4, fmax = 45)
        
        # Compute power for reconstructed epochs using new API
        recon_power = recon_epochs.compute_psd(method='welch', fmin = 0.4, fmax = 45)
        
        # Plot time-frequency representations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        orig_power.plot(axes=axes[0], show=False)
        axes[0].set_title(f'Original PSD (Welch) - sub-{subject_id}_ses-{session}')
        
        recon_power.plot(axes=axes[1], show=False)
        axes[1].set_title(f'Reconstructed PSD (Welch) - sub-{subject_id}_ses-{session}')
        
        # Get y-axis limits from both subplots and set them to be the same
        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        
        # Compute shared y-axis limits (min of both mins, max of both maxes)
        shared_ymin = min(ylim0[0], ylim1[0])
        shared_ymax = max(ylim0[1], ylim1[1])
        
        # Apply shared limits to both subplots
        axes[0].set_ylim([shared_ymin, shared_ymax])
        axes[1].set_ylim([shared_ymin, shared_ymax])

        plt.tight_layout()
        output_file = op.join(self.output_dir,f'sub-{subject_id}', f'psd_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved PSD plots to: {output_file}")
    
    def pixel_corr(self, original_epochs: mne.Epochs, recon_epochs: mne.Epochs,
                              subject_id: str, session: str):
        
        orig_data = original_epochs.get_data()
        recon_data = recon_epochs.get_data()
        
        correlations = np.zeros((orig_data.shape[1], orig_data.shape[2]))

        for i in range(orig_data.shape[1]):
            for j in range(orig_data.shape[2]):
                correlations[i, j] = np.corrcoef(orig_data[:, i, j], recon_data[:, i, j])[0, 1]

        fig, axes = plt.subplots(1, 1, figsize=(14, 12))
        
        sns.heatmap(correlations.T, ax=axes[0, 0], cmap='RdBu_r', center=0)
        axes[0, 0].set_title(f'Correlation across epochs - sub-{subject_id}_ses-{session}')
        axes[0, 0].set_xlabel('Times')
        axes[0, 0].set_ylabel('Channels')

        plt.tight_layout()
        output_file = op.join(self.output_dir,f'sub-{subject_id}', f'corr_sub-{subject_id}_ses-{session}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved correlations plots to: {output_file}")
    
    def correlation_heatmaps_grid(self, fif_dir: str):
        """
        Pick 9 random subjects, compute Pearson correlation between original and
        reconstructed signals for each epoch and channel (concatenating time dimension),
        and plot the results as 9 heatmaps in a 3x3 grid.
        
        Args:
            fif_dir: Directory containing FIF data
        """
        print("Computing correlation heatmaps for 9 random subjects...")
        
        # Find all available subjects and sessions
        subjects_sessions = find_subjects_and_sessions(fif_dir)
        
        if len(subjects_sessions) < 9:
            print(f"Warning: Only {len(subjects_sessions)} subject-sessions found, using all available.")
            selected = subjects_sessions
        else:
            # Randomly select 9 subjects
            selected = random.sample(subjects_sessions, 9)
        
        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        # Compute correlation for each selected subject
        for idx, item in enumerate(selected):
            subject_id = item['subject_id']
            session = item['session']
            session_path = item['path']
            
            print(f"  Processing subject {idx + 1}/9: sub-{subject_id}_ses-{session}")
            
            try:
                # Load both original and reconstructed epochs
                original_epochs, recon_epochs = load_subject_session_data(
                    subject_id, session, session_path
                )
                
                # Get data arrays (epochs x channels x times)
                orig_data = original_epochs.get_data()
                recon_data = recon_epochs.get_data()
                
                n_epochs = orig_data.shape[0]
                n_channels = orig_data.shape[1]
                n_times = orig_data.shape[2]
                
                # Compute correlation for each (channel, time) pair across epochs
                correlations = np.zeros((n_channels, n_times))
                
                for channel_idx in range(n_channels):
                    for time_idx in range(n_times):
                        # Get all epoch values for this channel and time point
                        orig_epochs_values = orig_data[:, channel_idx, time_idx]
                        recon_epochs_values = recon_data[:, channel_idx, time_idx]
                        
                        # Compute Pearson correlation across epochs
                        corr, _ = pearsonr(orig_epochs_values, recon_epochs_values)
                        correlations[channel_idx, time_idx] = corr
                
                # Plot heatmap in the grid
                sns.heatmap(correlations, ax=axes[idx], cmap='RdBu_r', 
                           center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
                axes[idx].set_title(f'sub-{subject_id}_ses-{session}')
                axes[idx].set_xlabel('Time')
                axes[idx].set_ylabel('Channels')
                
            except Exception as e:
                print(f"  Error processing subject {subject_id}, session {session}: {e}")
                # Plot empty heatmap with error message
                axes[idx].text(0.5, 0.5, f'Error loading\nsub-{subject_id}_ses-{session}',
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
        
        # If fewer than 9 subjects, hide extra subplots
        for idx in range(len(selected), 9):
            axes[idx].axis('off')
        
        plt.suptitle('Pearson Correlation between Original and Reconstructed Signals\n(Epoch × Channel)', 
                    fontsize=20, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save the figure
        output_file = op.join(self.output_dir, 'correlation_heatmaps_grid_9subjects.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation heatmaps grid to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Qualitative Analysis of EEG Data')
    parser.add_argument('--fif-dir', required=True, 
                        help='Directory containing FIF data (e.g., /data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata)')
    parser.add_argument('--num-subjects', type=int, default=6,
                        help='Number of subjects to analyze (default: 6)')
    parser.add_argument('--output-dir', default='./results/qualitative_analysis/',
                        help='Directory to save analysis results')
    parser.add_argument('--only-correlation-grid', action='store_true',
                        help='Only generate the 3x3 correlation heatmaps grid (skips other analyses)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run analysis
    analysis = QualitativeAnalysis(output_dir=args.output_dir)
    analysis.run_analysis(fif_dir=args.fif_dir, num_subjects=args.num_subjects, 
                         only_correlation_grid=args.only_correlation_grid)


if __name__ == '__main__':
    main()
