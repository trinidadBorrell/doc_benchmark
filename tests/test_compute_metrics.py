#!/usr/bin/env python3
"""
Test script for compute_metrics.py
Runs the general metrics computation with detailed debugging.

This tests the GENERAL_METRICS phase of the pipeline.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mne

# Configuration - EDIT THESE PATHS
RECON_DATA_PATH = "/data/project/eeg_foundation/data/LaBram/results_DoC/recon_data_inference"
ORIGINAL_DATA_PATH = "/data/project/eeg_foundation/data/data_250Hz_EGI256/nice_epochs_from_cohen_2/nice_epochs/nice_epochs2"
OUTPUT_DIR = "/data/project/eeg_foundation/src/doc_benchmark/results/LaBram/doc_patients/test_metrics"

def print_header(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)

def print_section(msg):
    print("\n" + "-" * 50)
    print(msg)
    print("-" * 50)

def find_matching_files():
    """Find matching original and reconstructed files."""
    recon_path = Path(RECON_DATA_PATH)
    orig_path = Path(ORIGINAL_DATA_PATH)
    
    print_header("STEP 1: Finding data files")
    
    # Find all reconstructed files
    print(f"\nSearching for reconstructed files in: {recon_path}")
    recon_files = list(recon_path.glob("**/sub-*_ses-*_vqnsp_reconstructed_epo.fif"))
    print(f"Found {len(recon_files)} reconstructed files")
    
    if recon_files:
        print("\nFirst 5 reconstructed files:")
        for f in recon_files[:5]:
            print(f"  {f}")
    
    # Find all original files
    print(f"\nSearching for original files in: {orig_path}")
    orig_files = list(orig_path.glob("**/sub-*/ses-*/eeg/*_epo.fif"))
    print(f"Found {len(orig_files)} original files")
    
    if orig_files:
        print("\nFirst 5 original files:")
        for f in orig_files[:5]:
            print(f"  {f}")
    
    # Try to match files
    print_section("Matching original and reconstructed files")
    
    matches = []
    for recon_file in recon_files:
        # Extract subject and session from recon filename
        # Pattern: sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif
        name = recon_file.name
        parts = name.split("_")
        if len(parts) >= 2:
            subject_id = parts[0].replace("sub-", "")
            session_id = parts[1].replace("ses-", "")
            
            # Look for matching original file
            orig_session_dir = orig_path / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg"
            if orig_session_dir.exists():
                orig_candidates = list(orig_session_dir.glob(f"sub-{subject_id}_ses-{session_id}_task-*_epo.fif"))
                if orig_candidates:
                    matches.append({
                        'subject': subject_id,
                        'session': session_id,
                        'recon': recon_file,
                        'original': orig_candidates[0]
                    })
    
    print(f"\nFound {len(matches)} matching pairs")
    if matches:
        print("\nFirst 3 matches:")
        for m in matches[:3]:
            print(f"  Subject: {m['subject']}, Session: {m['session']}")
            print(f"    Original: {m['original']}")
            print(f"    Recon:    {m['recon']}")
    
    return matches

def load_and_compare_epochs(match):
    """Load and compare a single pair of epochs files."""
    print_section(f"Loading epochs for {match['subject']}/ses-{match['session']}")
    
    try:
        # Load original
        print(f"\nLoading original: {match['original']}")
        orig_epochs = mne.read_epochs(match['original'], preload=True, verbose=False)
        print(f"  Shape: {orig_epochs.get_data().shape}")
        print(f"  N epochs: {len(orig_epochs)}")
        print(f"  N channels: {len(orig_epochs.ch_names)}")
        print(f"  Sfreq: {orig_epochs.info['sfreq']}")
        
        # Load reconstructed
        print(f"\nLoading reconstructed: {match['recon']}")
        recon_epochs = mne.read_epochs(match['recon'], preload=True, verbose=False)
        print(f"  Shape: {recon_epochs.get_data().shape}")
        print(f"  N epochs: {len(recon_epochs)}")
        print(f"  N channels: {len(recon_epochs.ch_names)}")
        print(f"  Sfreq: {recon_epochs.info['sfreq']}")
        
        # Get data arrays
        orig_data = orig_epochs.get_data()
        recon_data = recon_epochs.get_data()
        
        # Check if shapes match
        print_section("Comparing data")
        if orig_data.shape != recon_data.shape:
            print(f"WARNING: Shape mismatch!")
            print(f"  Original: {orig_data.shape}")
            print(f"  Recon:    {recon_data.shape}")
            
            # Try to align by taking minimum
            min_epochs = min(orig_data.shape[0], recon_data.shape[0])
            min_channels = min(orig_data.shape[1], recon_data.shape[1])
            min_times = min(orig_data.shape[2], recon_data.shape[2])
            
            orig_data = orig_data[:min_epochs, :min_channels, :min_times]
            recon_data = recon_data[:min_epochs, :min_channels, :min_times]
            print(f"  Aligned to: {orig_data.shape}")
        
        # Compute metrics
        print("\nComputing metrics:")
        
        # MAPE (Mean Absolute Percentage Error)
        epsilon = 1e-10
        mape = np.mean(np.abs((orig_data - recon_data) / (np.abs(orig_data) + epsilon))) * 100
        print(f"  MAPE: {mape:.4f}%")
        
        # Correlation (per channel, averaged)
        correlations = []
        for ch in range(orig_data.shape[1]):
            orig_flat = orig_data[:, ch, :].flatten()
            recon_flat = recon_data[:, ch, :].flatten()
            corr = np.corrcoef(orig_flat, recon_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        mean_corr = np.mean(correlations) if correlations else 0
        print(f"  Mean Correlation: {mean_corr:.4f}")
        
        # MSE
        mse = np.mean((orig_data - recon_data) ** 2)
        print(f"  MSE: {mse:.6f}")
        
        # RMSE
        rmse = np.sqrt(mse)
        print(f"  RMSE: {rmse:.6f}")
        
        return {
            'subject': match['subject'],
            'session': match['session'],
            'mape': mape,
            'correlation': mean_corr,
            'mse': mse,
            'rmse': rmse,
            'n_epochs': orig_data.shape[0],
            'n_channels': orig_data.shape[1]
        }
        
    except Exception as e:
        print(f"ERROR loading epochs: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print_header("TEST: COMPUTE METRICS")
    print(f"Reconstructed data: {RECON_DATA_PATH}")
    print(f"Original data: {ORIGINAL_DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # Check paths exist
    if not Path(RECON_DATA_PATH).exists():
        print(f"ERROR: Reconstructed data path does not exist: {RECON_DATA_PATH}")
        return
    
    if not Path(ORIGINAL_DATA_PATH).exists():
        print(f"ERROR: Original data path does not exist: {ORIGINAL_DATA_PATH}")
        return
    
    # Find matching files
    matches = find_matching_files()
    
    if not matches:
        print("\nERROR: No matching file pairs found!")
        return
    
    # Process first few matches for testing
    print_header("STEP 2: Computing metrics for first 3 subjects")
    
    results = []
    for match in matches[:3]:
        result = load_and_compare_epochs(match)
        if result:
            results.append(result)
    
    # Summary
    print_header("SUMMARY")
    if results:
        print(f"\nProcessed {len(results)} subjects successfully")
        print("\nResults:")
        for r in results:
            print(f"  {r['subject']}/ses-{r['session']}: MAPE={r['mape']:.2f}%, Corr={r['correlation']:.4f}")
        
        # Aggregate
        mean_mape = np.mean([r['mape'] for r in results])
        mean_corr = np.mean([r['correlation'] for r in results])
        print(f"\nAggregate:")
        print(f"  Mean MAPE: {mean_mape:.2f}%")
        print(f"  Mean Correlation: {mean_corr:.4f}")
    else:
        print("No results computed!")

if __name__ == "__main__":
    main()
