#!/usr/bin/env python3
"""
Dummy Model for EEG Data Reconstruction

This module implements a simple dummy model that:
1. Reads original .fif files
2. For each epoch and channel, computes the mean value across time
3. Creates "reconstructed" data by replacing time series with constant mean values
4. Computes markers on-the-fly for both original and reconstructed data
5. Implements a decoder between original and reconstructed data

The dummy model serves as a baseline for comparison with more sophisticated
reconstruction methods in the DOC benchmark pipeline.

Usage:
    python dummy_model.py --input /path/to/original.fif --output /path/to/output_dir
    python dummy_model.py --subject AD023 --session ses-01 --data-dir /path/to/data --output /path/to/output

Authors: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import sys
from datetime import datetime
import pickle
import pandas as pd
from scipy import stats
import logging

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Add NICE library to path
sys.path.append('/data/project/eeg_foundation/src/nice')
from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator,
                          ContingentNegativeVariation,
                          TimeLockedTopography,
                          TimeLockedContrast)

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


class DummyModel:
    """
    Dummy model that reconstructs EEG data by replacing time series with mean values.
    
    This model serves as a baseline for comparison with more sophisticated
    reconstruction methods. It processes each epoch and channel independently,
    computing the temporal mean and creating a constant signal.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the dummy model.
        
        Parameters
        ----------
        verbose : bool
            Whether to print detailed information during processing
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def load_epochs(self, fif_path):
        """
        Load epochs from a .fif file.
        
        Parameters
        ----------
        fif_path : str or Path
            Path to the .fif file
            
        Returns
        -------
        epochs : mne.Epochs
            Loaded epochs object
        """
        fif_path = Path(fif_path)
        if not fif_path.exists():
            raise FileNotFoundError(f"File not found: {fif_path}")
            
        self.logger.info(f"Loading epochs from {fif_path}")
        
        # Load the epochs
        epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
        
        # Apply event ID mapping if needed
        epochs = self._apply_event_id_mapping(epochs)
        
        self.logger.info(f"Loaded {len(epochs)} epochs with {len(epochs.ch_names)} channels")
        self.logger.info(f"Time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} seconds")
        self.logger.info(f"Sampling frequency: {epochs.info['sfreq']:.1f} Hz")
        
        return epochs
    
    def _apply_event_id_mapping(self, epochs):
        """Apply event ID mapping to epochs if needed."""
        # Check if epochs already have string event names
        if all(isinstance(key, str) for key in epochs.event_id.keys()):
            self.logger.info("Epochs already have string event names")
            return epochs
            
        # Apply mapping
        self.logger.info("Applying event ID mapping")
        new_event_id = {}
        events = epochs.events.copy()
        
        for old_id, new_name in EVENT_ID_MAPPING.items():
            if old_id in epochs.event_id.values():
                # Find the key for this value
                old_key = [k for k, v in epochs.event_id.items() if v == old_id][0]
                new_event_id[new_name] = old_id
                # Update events array
                events[events[:, 2] == old_id, 2] = old_id
                
        if new_event_id:
            epochs.event_id = new_event_id
            epochs.events = events
            self.logger.info(f"Applied mapping: {new_event_id}")
        
        return epochs
    
    def reconstruct_epochs(self, epochs):
        """
        Reconstruct epochs by replacing time series with mean values.
        
        For each epoch and channel, computes the mean value across time
        and creates a constant signal with that mean value.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Original epochs to reconstruct
            
        Returns
        -------
        reconstructed_epochs : mne.Epochs
            Reconstructed epochs with constant mean values
        """
        self.logger.info("Reconstructing epochs using temporal mean...")
        
        # Get the data (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        self.logger.info(f"Processing {n_epochs} epochs, {n_channels} channels, {n_times} time points")
        
        # Compute mean across time for each epoch and channel
        mean_values = np.mean(data, axis=2, keepdims=True)  # Shape: (n_epochs, n_channels, 1)
        
        # Create reconstructed data by repeating mean values across time
        reconstructed_data = np.repeat(mean_values, n_times, axis=2)
        
        # Create new epochs object with reconstructed data
        reconstructed_epochs = epochs.copy()
        reconstructed_epochs._data = reconstructed_data
        
        # Compute reconstruction statistics
        original_std = np.std(data)
        reconstructed_std = np.std(reconstructed_data)
        correlation = np.corrcoef(data.flatten(), reconstructed_data.flatten())[0, 1]
        
        self.logger.info(f"Reconstruction statistics:")
        self.logger.info(f"  Original data std: {original_std:.6f}")
        self.logger.info(f"  Dummy data std: {reconstructed_std:.6f}")
        self.logger.info(f"  Correlation: {correlation:.6f}")
        
        return reconstructed_epochs
    
    def save_reconstructed_epochs(self, epochs, output_path):
        """
        Save reconstructed epochs to a .fif file.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Reconstructed epochs to save
        output_path : str or Path
            Path where to save the reconstructed epochs
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving reconstructed epochs to {output_path}")
        epochs.save(str(output_path), overwrite=True, verbose=False)
    
    def compute_markers(self, epochs, output_path, skip_smi=True):
        """
        Compute markers for epochs using the NICE library.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to compute markers for
        output_path : str or Path
            Path to save the markers HDF5 file
        skip_smi : bool
            Whether to skip SymbolicMutualInformation (computationally expensive)
            
        Returns
        -------
        markers : nice.Markers
            Computed markers object
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Computing markers for {len(epochs)} epochs...")
        
        # Define markers to compute
        markers_list = [
            PowerSpectralDensity(
                estimator=PowerSpectralDensityEstimator(
                    method='welch', n_fft=None, n_overlap=0, n_per_seg=None,
                    tmin=None, tmax=None, fmin=1., fmax=45., n_jobs=1, verbose=False),
                frequency_bands={"alpha": [8, 12], "beta": [13, 25], "gamma": [30, 45]}
            ),
            PowerSpectralDensitySummary(
                estimator=PowerSpectralDensityEstimator(
                    method='welch', n_fft=None, n_overlap=0, n_per_seg=None,
                    tmin=None, tmax=None, fmin=1., fmax=45., n_jobs=1, verbose=False),
                frequency_bands={"alpha": [8, 12], "beta": [13, 25], "gamma": [30, 45]}
            ),
            KolmogorovComplexity(),
            PermutationEntropy(tmin=None, tmax=None)
        ]
        
        # Add SymbolicMutualInformation if not skipped
        if not skip_smi:
            self.logger.info("Including SymbolicMutualInformation (this may take a while...)")
            markers_list.append(SymbolicMutualInformation(
                tmin=None, tmax=None, method='weighted', backend='c'
            ))
        else:
            self.logger.info("Skipping SymbolicMutualInformation for faster computation")
        
        # Add time-locked markers if we have proper event names
        if all(isinstance(key, str) for key in epochs.event_id.keys()):
            self.logger.info("Adding time-locked markers")
            markers_list.extend([
                ContingentNegativeVariation(tmin=None, tmax=None),
                TimeLockedTopography(tmin=None, tmax=None),
                TimeLockedContrast(tmin=None, tmax=None)
            ])
        
        # Create markers object and fit
        markers = Markers(markers_list)
        
        try:
            markers.fit(epochs)
            self.logger.info("✓ Markers computation completed successfully")
            
            # Save markers
            markers.save(str(output_path))
            self.logger.info(f"✓ Markers saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"✗ Markers computation failed: {e}")
            raise
        
        return markers
    
    def compute_features(self, markers_path, output_scalars_path, output_topos_path):
        """
        Compute features from markers file.
        
        Parameters
        ----------
        markers_path : str or Path
            Path to the markers HDF5 file
        output_scalars_path : str or Path
            Path to save scalar features
        output_topos_path : str or Path
            Path to save topographic features
        """
        markers_path = Path(markers_path)
        output_scalars_path = Path(output_scalars_path)
        output_topos_path = Path(output_topos_path)
        
        # Create output directories
        output_scalars_path.parent.mkdir(parents=True, exist_ok=True)
        output_topos_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Computing features from {markers_path}")
        
        # Load markers
        markers = Markers.load(str(markers_path))
        
        # Extract features
        scalars = markers.as_data_frame()
        topos = markers.as_data_frame(reduction=None)
        
        # Convert to numpy arrays
        scalars_array = scalars.values
        topos_array = topos.values
        
        # Save features
        np.save(str(output_scalars_path), scalars_array)
        np.save(str(output_topos_path), topos_array)
        
        self.logger.info(f"✓ Scalar features saved to {output_scalars_path}")
        self.logger.info(f"✓ Topographic features saved to {output_topos_path}")
        self.logger.info(f"  Scalars shape: {scalars_array.shape}")
        self.logger.info(f"  Topos shape: {topos_array.shape}")
    
    def decode_original_vs_reconstructed(self, original_epochs, reconstructed_epochs, 
                                       output_dir, cv=5, n_jobs=1):
        """
        Decode between original and reconstructed EEG data.
        
        Parameters
        ----------
        original_epochs : mne.Epochs
            Original epochs (class 0)
        reconstructed_epochs : mne.Epochs
            Reconstructed epochs (class 1)
        output_dir : str or Path
            Directory to save decoding results
        cv : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        results : dict
            Decoding results including scores and times
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Setting up decoding analysis...")
        
        # Combine data and create labels
        original_data = original_epochs.get_data()
        reconstructed_data = reconstructed_epochs.get_data()
        
        # Ensure same number of epochs
        min_epochs = min(len(original_data), len(reconstructed_data))
        original_data = original_data[:min_epochs]
        reconstructed_data = reconstructed_data[:min_epochs]
        
        # Combine data
        X = np.concatenate([original_data, reconstructed_data], axis=0)
        y = np.concatenate([np.zeros(min_epochs), np.ones(min_epochs)])
        
        self.logger.info(f"Decoding setup:")
        self.logger.info(f"  Total epochs: {len(X)}")
        self.logger.info(f"  Original epochs (class 0): {min_epochs}")
        self.logger.info(f"  Reconstructed epochs (class 1): {min_epochs}")
        self.logger.info(f"  Channels: {X.shape[1]}")
        self.logger.info(f"  Time points: {X.shape[2]}")
        
        # Create classifier pipeline
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear', random_state=42)
        )
        
        # Create sliding estimator
        sliding_estimator = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc', verbose=False)
        
        # Perform cross-validation
        self.logger.info(f"Running {cv}-fold cross-validation...")
        scores = cross_val_multiscore(sliding_estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=False)
        
        # Get time points
        times = original_epochs.times
        
        # Compute statistics
        mean_scores = np.mean(scores, axis=0)
        std_scores = np.std(scores, axis=0)
        
        # Find peak decoding time
        peak_idx = np.argmax(mean_scores)
        peak_time = times[peak_idx]
        peak_score = mean_scores[peak_idx]
        
        self.logger.info(f"Decoding results:")
        self.logger.info(f"  Peak AUC: {peak_score:.3f} ± {std_scores[peak_idx]:.3f}")
        self.logger.info(f"  Peak time: {peak_time:.3f} seconds")
        self.logger.info(f"  Mean AUC: {np.mean(mean_scores):.3f}")
        
        # Prepare results
        results = {
            'scores': scores,
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'times': times,
            'peak_score': peak_score,
            'peak_time': peak_time,
            'cv_folds': cv,
            'n_epochs_per_class': min_epochs,
            'n_channels': X.shape[1],
            'n_timepoints': X.shape[2]
        }
        
        # Save results
        results_file = output_dir / "decoding_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON
        summary = {
            'peak_auc': float(peak_score),
            'peak_time': float(peak_time),
            'mean_auc': float(np.mean(mean_scores)),
            'cv_folds': cv,
            'n_epochs_per_class': min_epochs,
            'n_channels': int(X.shape[1]),
            'n_timepoints': int(X.shape[2]),
            'processing_date': datetime.now().isoformat()
        }
        
        summary_file = output_dir / "decoding_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save times array
        times_file = output_dir / "times.npy"
        np.save(str(times_file), times)
        
        # Create and save plot
        self._plot_decoding_results(results, output_dir)
        
        self.logger.info(f"✓ Decoding results saved to {output_dir}")
        
        return results
    
    def _plot_decoding_results(self, results, output_dir):
        """Plot decoding results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = results['times']
        mean_scores = results['mean_scores']
        std_scores = results['std_scores']
        
        # Plot mean scores with error bars
        ax.plot(times, mean_scores, 'b-', linewidth=2, label='Mean AUC')
        ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores, 
                       alpha=0.3, color='blue', label='±1 SD')
        
        # Add chance level
        ax.axhline(0.5, color='k', linestyle='--', alpha=0.7, label='Chance level')
        
        # Mark peak
        peak_idx = np.argmax(mean_scores)
        ax.plot(times[peak_idx], mean_scores[peak_idx], 'ro', markersize=8, 
               label=f'Peak: {mean_scores[peak_idx]:.3f}')
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AUC Score')
        ax.set_title('Decoding Original vs Reconstructed EEG Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / "decoding_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Decoding plot saved to {plot_file}")
    
    def compute_features_from_markers(self, markers):
        """
        Compute features directly from markers object (without saving to file).
        
        Parameters
        ----------
        markers : nice.Markers
            Computed markers object
            
        Returns
        -------
        tuple
            (scalars_array, topos_array) - numpy arrays of features
        """
        # Extract features
        scalars = markers.as_data_frame()
        topos = markers.as_data_frame(reduction=None)
        
        # Convert to numpy arrays
        scalars_array = scalars.values
        topos_array = topos.values
        
        self.logger.info(f"  Extracted features - Scalars: {scalars_array.shape}, Topos: {topos_array.shape}")
        
        return scalars_array, topos_array

    def process_subject(self, subject_id, session, data_dir, output_dir, 
                       skip_smi=True, cv=5, n_jobs=1):
        """
        Process a complete subject through the dummy model pipeline ON-THE-FLY.
        
        This method computes everything in memory without saving heavy .fif files.
        Only the final results (markers, features, decoder results) are saved.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'AD023')
        session : str
            Session identifier (e.g., 'ses-01')
        data_dir : str or Path
            Directory containing the subject data
        output_dir : str or Path
            Directory to save all outputs
        skip_smi : bool
            Whether to skip SymbolicMutualInformation computation
        cv : int
            Number of cross-validation folds for decoding
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        success : bool
            Whether processing completed successfully
        """
        try:
            data_dir = Path(data_dir)
            output_dir = Path(output_dir)
            
            # Find input file
            subject_dir = data_dir / f"sub-{subject_id}" / session
            original_file = next(subject_dir.glob("*_original.fif"), None)
            
            if not original_file:
                raise FileNotFoundError(f"No original .fif file found in {subject_dir}")
            
            self.logger.info(f"Processing subject {subject_id}, session {session} (ON-THE-FLY)")
            self.logger.info(f"Input file: {original_file}")
            self.logger.info("Note: No heavy .fif files will be saved - computing everything in memory")
            
            # Create output directories
            subject_output_dir = output_dir / f"sub-{subject_id}" / session
            markers_dir = subject_output_dir / "markers_variable"
            features_dir = subject_output_dir / "features_variable"
            decoder_dir = subject_output_dir / "decoder"
            
            for dir_path in [markers_dir, features_dir, decoder_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Load original epochs
            self.logger.info("Step 1: Loading original epochs...")
            original_epochs = self.load_epochs(original_file)
            
            # Step 2: Create reconstructed epochs (in memory only)
            self.logger.info("Step 2: Creating dummy reconstructed epochs (in memory)...")
            reconstructed_epochs = self.reconstruct_epochs(original_epochs)
            
            # Step 3: Compute markers for original data
            self.logger.info("Step 3: Computing markers for original data...")
            original_markers = self.compute_markers(
                original_epochs, 
                markers_dir / "markers_original.hdf5",
                skip_smi=skip_smi
            )
            
            # Step 4: Compute markers for reconstructed data
            self.logger.info("Step 4: Computing markers for dummy reconstructed data...")
            reconstructed_markers = self.compute_markers(
                reconstructed_epochs,
                markers_dir / "markers_reconstructed.hdf5", 
                skip_smi=skip_smi
            )
            
            # Step 5: Compute features directly from markers (on-the-fly)
            self.logger.info("Step 5: Computing features from markers (on-the-fly)...")
            
            # Extract features from original markers
            self.logger.info("  Extracting features from original markers...")
            original_scalars, original_topos = self.compute_features_from_markers(original_markers)
            
            # Extract features from reconstructed markers  
            self.logger.info("  Extracting features from dummy markers...")
            reconstructed_scalars, reconstructed_topos = self.compute_features_from_markers(reconstructed_markers)
            
            # Save features to files
            np.save(str(features_dir / "scalars_original.npy"), original_scalars)
            np.save(str(features_dir / "topos_original.npy"), original_topos)
            np.save(str(features_dir / "scalars_reconstructed.npy"), reconstructed_scalars)
            np.save(str(features_dir / "topos_reconstructed.npy"), reconstructed_topos)
            
            self.logger.info("  ✓ Features saved to disk")
            
            # Step 6: Decode original vs reconstructed (on-the-fly)
            self.logger.info("Step 6: Running decoding analysis (on-the-fly)...")
            decoding_results = self.decode_original_vs_reconstructed(
                original_epochs, reconstructed_epochs,
                decoder_dir, cv=cv, n_jobs=n_jobs
            )
            
            # Clean up memory (epochs objects can be large)
            del original_epochs, reconstructed_epochs
            
            self.logger.info(f"✓ Successfully processed {subject_id}/{session} (ON-THE-FLY)")
            self.logger.info("✓ No heavy .fif files were saved - only results are stored")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {subject_id}/{session}: {e}")
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dummy Model for EEG Data Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single file
    python dummy_model.py --input /path/to/original.fif --output /path/to/output_dir
    
    # Process subject/session
    python dummy_model.py --subject AD023 --session ses-01 --data-dir /path/to/data --output /path/to/output
    
    # Process with custom parameters
    python dummy_model.py --subject AD023 --session ses-01 --data-dir /path/to/data --output /path/to/output --cv 10 --n-jobs 4
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", 
        type=str,
        help="Path to input .fif file"
    )
    group.add_argument(
        "--subject",
        type=str, 
        help="Subject ID (use with --session and --data-dir)"
    )
    
    parser.add_argument(
        "--session",
        type=str,
        help="Session ID (required with --subject)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory (required with --subject)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    
    # Processing options
    parser.add_argument(
        "--skip-smi",
        action="store_true",
        default=True,
        help="Skip SymbolicMutualInformation computation (default: True)"
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds for decoding (default: 5)"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create dummy model
    model = DummyModel(verbose=args.verbose)
    
    try:
        if args.input:
            # Single file processing
            model.logger.info("Single file processing mode")
            
            # Load and process single file
            original_epochs = model.load_epochs(args.input)
            reconstructed_epochs = model.reconstruct_epochs(original_epochs)
            
            # Create output structure
            output_dir = Path(args.output)
            markers_dir = output_dir / "markers"
            features_dir = output_dir / "features"
            decoder_dir = output_dir / "decoder"
            
            for dir_path in [markers_dir, features_dir, decoder_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save reconstructed data
            recon_file = output_dir / "reconstructed.fif"
            model.save_reconstructed_epochs(reconstructed_epochs, recon_file)
            
            # Compute markers and features
            model.compute_markers(original_epochs, markers_dir / "markers_original.hdf5", 
                                skip_smi=args.skip_smi)
            model.compute_markers(reconstructed_epochs, markers_dir / "markers_reconstructed.hdf5",
                                skip_smi=args.skip_smi)
            
            model.compute_features(markers_dir / "markers_original.hdf5",
                                 features_dir / "scalars_original.npy",
                                 features_dir / "topos_original.npy")
            model.compute_features(markers_dir / "markers_reconstructed.hdf5",
                                 features_dir / "scalars_reconstructed.npy", 
                                 features_dir / "topos_reconstructed.npy")
            
            # Decode
            model.decode_original_vs_reconstructed(original_epochs, reconstructed_epochs,
                                                 decoder_dir, cv=args.cv, n_jobs=args.n_jobs)
            
        else:
            # Subject/session processing
            if not args.session or not args.data_dir:
                raise ValueError("--session and --data-dir are required when using --subject")
            
            model.logger.info("Subject/session processing mode")
            success = model.process_subject(
                args.subject, args.session, args.data_dir, args.output,
                skip_smi=args.skip_smi, cv=args.cv, n_jobs=args.n_jobs
            )
            
            if not success:
                sys.exit(1)
        
        model.logger.info("✓ Processing completed successfully")
        
    except Exception as e:
        model.logger.error(f"✗ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
