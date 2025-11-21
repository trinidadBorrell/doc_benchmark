#!/usr/bin/env python3
"""
Visualization functions for decoder analysis with statistical significance highlighting.

This module contains functions to create publication-quality plots for decoder results,
with support for highlighting statistically significant timepoints from Wilcoxon tests.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


def add_stimulus_lines(ax, times):
    """Add vertical lines at stimulus times to a plot."""
    stimulus_times = [0.0, 0.15, 0.3, 0.45, 0.6]
    for i, stim_time in enumerate(stimulus_times):
        if times[0] <= stim_time <= times[-1]:
            label = 'Stimuli' if i == 0 else None
            ax.axvline(stim_time, color='darkgray', linestyle='--', alpha=1, linewidth=1.5, label=label)


def plot_global_with_pvalues(data_dir, output_dir, mode='DOC', analysis_type='overall'):
    """
    Create global plot with p-value highlighting.
    
    Parameters:
        data_dir: Path to directory containing aggregated results
        output_dir: Path to save output plots
        mode: 'DOC' or 'control'
        analysis_type: 'overall' or trial type name ('LSGS', 'LSGD', 'LDGD', 'LDGS')
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated results
    with open(data_dir / "aggregated_results.pkl", 'rb') as f:
        aggregated_results = pickle.load(f)
    
    # Load times (try multiple sources)
    times = None
    
    # First try to load times.npy from data directory
    times_file = data_dir / "times.npy"
    if times_file.exists():
        times = np.load(times_file)
    elif (data_dir.parent / "all_auc_timeseries" / "all_mean_scores_time.npy").exists():
        # Fallback: reconstruct times from data shape
        all_scores = np.load(data_dir.parent / "all_auc_timeseries" / "all_mean_scores_time.npy")
        n_times = all_scores.shape[1]
        times = np.linspace(-0.2, 1.34, n_times)
    
    # Load Wilcoxon results
    if analysis_type == 'overall':
        wilcoxon_file = data_dir / "overall_wilcoxon_per_timepoint.csv"
    else:
        wilcoxon_file = data_dir / f"{analysis_type}_wilcoxon_per_timepoint.csv"
    
    if not wilcoxon_file.exists():
        print(f"Warning: Wilcoxon results not found at {wilcoxon_file}")
        wilcoxon_df = None
    else:
        wilcoxon_df = pd.read_csv(wilcoxon_file)
        if times is None:
            times = wilcoxon_df['time_seconds'].values
    
    # Get data
    if analysis_type == 'overall':
        data = aggregated_results['overall']
        title_text = f"{mode} Overall Decoding: Original vs Reconstructed"
        filename_prefix = f"{mode.lower()}_overall"
    else:
        if analysis_type not in aggregated_results['trial_types']:
            print(f"Warning: {analysis_type} not found in results")
            return
        data = aggregated_results['trial_types'][analysis_type]
        title_text = f"{mode} {analysis_type} Decoding: Original vs Reconstructed"
        filename_prefix = f"{mode.lower()}_{analysis_type}"
    
    mean_scores = data['mean_scores_time']
    std_scores = data['std_scores_time']
    n_subjects = len(data['all_mean_scores_time'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Plot mean line
    ax.plot(times, mean_scores, 'b-', linewidth=3, label=f'Mean AUC (n={n_subjects})')
    
    # Fill between for std
    ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                     alpha=0.3, color='blue', label='±1 SD')
    
    # Highlight significant timepoints with yellow background
    if wilcoxon_df is not None and 'significant' in wilcoxon_df.columns:
        significant_mask = wilcoxon_df['significant'].values
        
        # Find consecutive significant regions
        sig_regions = []
        in_region = False
        start_idx = None
        
        for i, is_sig in enumerate(significant_mask):
            if is_sig and not in_region:
                start_idx = i
                in_region = True
            elif not is_sig and in_region:
                sig_regions.append((start_idx, i - 1))
                in_region = False
        
        # Close last region if still open
        if in_region:
            sig_regions.append((start_idx, len(significant_mask) - 1))
        
        # Highlight regions
        for start_idx, end_idx in sig_regions:
            ax.axvspan(times[start_idx], times[end_idx], 
                      alpha=0.5, color='yellow', 
                      label='p < 0.01' if start_idx == sig_regions[0][0] else None)
    
    # Reference lines
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5, label="Chance")
    add_stimulus_lines(ax, times)
    
    # Formatting
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("AUC-ROC", fontsize=20)
    ax.set_title(title_text, fontsize=22, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    ax.tick_params(axis='both', labelsize=16)
    
    # Add text box with statistics
    info_text = f'n={n_subjects} subjects\n'
    info_text += f'Mean AUC: {np.mean(mean_scores):.3f}\n'
    if wilcoxon_df is not None:
        n_sig = np.sum(wilcoxon_df['significant'].values)
        n_total = len(wilcoxon_df)
        info_text += f'Sig. timepoints: {n_sig}/{n_total}'
    
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename_prefix}_global_with_pvalues.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / f'{filename_prefix}_global_with_pvalues.png'}")


def plot_all_global_analyses(results_dir, mode='DOC'):
    """
    Create all global plots for a given analysis results directory.
    
    Parameters:
        results_dir: Path to the decoding-global-XXXXXXXX directory
        mode: 'DOC' or 'control'
    """
    results_dir = Path(results_dir)
    data_dir = results_dir / "data"
    output_base = Path("/data/project/eeg_foundation/src/doc_benchmark/results/new_results/DECODER/global_analysis")
    
    # Create output directory with analysis name
    output_dir = output_base / f"{mode}_{results_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating global plots for {mode} analysis")
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load aggregated results to check what's available
    with open(data_dir / "aggregated_results.pkl", 'rb') as f:
        aggregated_results = pickle.load(f)
    
    # Plot overall
    print("Creating overall global plot...")
    plot_global_with_pvalues(data_dir, output_dir, mode=mode, analysis_type='overall')
    
    # Plot trial types (if task events are present)
    if aggregated_results.get('has_task_events', False):
        print("\nCreating trial-type global plots...")
        trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
        for trial_type in trial_types:
            if trial_type in aggregated_results['trial_types']:
                print(f"  - {trial_type}")
                plot_global_with_pvalues(data_dir, output_dir, mode=mode, analysis_type=trial_type)
    else:
        print("\nSkipping trial-type plots (resting state data)")
    
    # Copy statistical CSVs to output directory
    print("\nCopying statistical results...")
    csv_files = list(data_dir.glob("*wilcoxon_per_timepoint.csv"))
    for csv_file in csv_files:
        import shutil
        shutil.copy(csv_file, output_dir / csv_file.name)
        print(f"  Copied: {csv_file.name}")
    
    # Create summary text file
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"DECODER GLOBAL ANALYSIS SUMMARY\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Results directory: {results_dir}\n")
        f.write(f"Analysis date: {results_dir.name.split('-')[-1]}\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"-"*30 + "\n")
        if 'overall' in aggregated_results:
            f.write(f"  Mean AUC: {aggregated_results['overall']['mean_auc']:.4f}\n")
            f.write(f"  Std AUC: {aggregated_results['overall']['std_auc']:.4f}\n")
            f.write(f"  N subjects: {len(aggregated_results['overall']['all_mean_scores_time'])}\n")
            
            # Load and report Wilcoxon results
            wilcoxon_file = data_dir / "overall_wilcoxon_per_timepoint.csv"
            if wilcoxon_file.exists():
                wilcoxon_df = pd.read_csv(wilcoxon_file)
                n_sig = np.sum(wilcoxon_df['significant'])
                n_total = len(wilcoxon_df)
                f.write(f"  Significant timepoints: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)\n")
        
        f.write(f"\n")
        
        if aggregated_results.get('has_task_events', False):
            f.write(f"Trial Type Statistics:\n")
            f.write(f"-"*30 + "\n")
            for trial_type in ['LSGS', 'LSGD', 'LDGD', 'LDGS']:
                if trial_type in aggregated_results['trial_types']:
                    data = aggregated_results['trial_types'][trial_type]
                    f.write(f"\n{trial_type}:\n")
                    f.write(f"  Mean AUC: {data['mean_auc']:.4f}\n")
                    f.write(f"  Std AUC: {data['std_auc']:.4f}\n")
                    f.write(f"  N subjects: {data['n_subjects_sessions']}\n")
                    
                    # Load and report Wilcoxon results
                    wilcoxon_file = data_dir / f"{trial_type}_wilcoxon_per_timepoint.csv"
                    if wilcoxon_file.exists():
                        wilcoxon_df = pd.read_csv(wilcoxon_file)
                        n_sig = np.sum(wilcoxon_df['significant'])
                        n_total = len(wilcoxon_df)
                        f.write(f"  Significant timepoints: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"\n{'='*60}")
    print(f"COMPLETED: All plots and statistics saved to {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def main():
    """
    Example usage of the visualization functions.
    
    Call this script after running decoder.py to generate publication-ready plots
    with statistical significance highlighting.
    
    Example:
        python viz.py --results-dir /path/to/decoding-global-20231115_120000 --mode DOC
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create global plots with p-value highlighting')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to the decoding-global-XXXXXXXX directory')
    parser.add_argument('--mode', type=str, default='DOC', choices=['DOC', 'control'],
                       help='Analysis mode (DOC or control)')
    
    args = parser.parse_args()
    
    plot_all_global_analyses(args.results_dir, mode=args.mode)


if __name__ == "__main__":
    main()
