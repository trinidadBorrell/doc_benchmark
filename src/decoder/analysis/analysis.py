#!/usr/bin/env python3
"""
Statistical analysis for decoder results.

This module loads aggregated decoder results and performs per-timepoint
Wilcoxon signed-rank tests against chance level (0.5 AUC), saving results to CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from scipy import stats
import argparse


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
        'significant_mask': p_values < 0.01,
        'n_subjects': int(n_subjects),
        'n_timepoints': int(n_timepoints)
    }


def analyze_decoder_results(results_dir, output_dir=None):
    """
    Load decoder results and perform statistical analysis.
    
    Parameters:
        results_dir: Path to the decoding-global-XXXXXXXX directory
        output_dir: Optional custom output directory (defaults to results_dir/data)
        
    Returns:
        dict containing analysis results and paths to saved files
    """
    results_dir = Path(results_dir)
    data_dir = results_dir / "data"
    
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS OF DECODER RESULTS")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load aggregated results
    aggregated_file = data_dir / "aggregated_results.pkl"
    if not aggregated_file.exists():
        raise FileNotFoundError(f"Aggregated results not found: {aggregated_file}")
    
    with open(aggregated_file, 'rb') as f:
        aggregated_results = pickle.load(f)
    
    # Load times - try multiple sources
    times = None
    
    # First try to load times.npy from data directory
    times_file = data_dir / "times.npy"
    if times_file.exists():
        times = np.load(times_file)
    else:
        # Try parent directory's all_auc_timeseries folder
        timeseries_dir = results_dir.parent / "all_auc_timeseries"
        if (timeseries_dir / "all_mean_scores_time.npy").exists():
            all_scores = np.load(timeseries_dir / "all_mean_scores_time.npy")
            n_times = all_scores.shape[1]
            # Standard time range for EEG epochs
            times = np.linspace(-0.2, 1.34, n_times)
    
    analysis_results = {
        'overall': None,
        'trial_types': {},
        'files_created': []
    }
    
    # Analyze overall classification
    if 'overall' in aggregated_results and 'all_mean_scores_time' in aggregated_results['overall']:
        print("Analyzing overall classification...")
        all_scores = np.array(aggregated_results['overall']['all_mean_scores_time'])
        
        print(f"  Data shape: {all_scores.shape} (n_subjects x n_timepoints)")
        
        # Run Wilcoxon test
        wilcoxon_result = run_wilcoxon_per_timepoint(all_scores, alternative='greater')
        analysis_results['overall'] = wilcoxon_result
        
        n_sig = np.sum(wilcoxon_result['significant_mask'])
        n_total = wilcoxon_result['n_timepoints']
        print(f"  Significant timepoints: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")
        
        # If times still not available, reconstruct from data shape
        if times is None:
            n_times = all_scores.shape[1]
            times = np.linspace(-0.2, 1.34, n_times)
        
        # Save to CSV
        csv_file = output_dir / "overall_wilcoxon_per_timepoint.csv"
        wilcoxon_df = pd.DataFrame({
            'timepoint_index': np.arange(len(wilcoxon_result['p_values'])),
            'time_seconds': times if times is not None else np.arange(len(wilcoxon_result['p_values'])),
            'p_value': wilcoxon_result['p_values'],
            'test_statistic': wilcoxon_result['test_statistics'],
            'significant': wilcoxon_result['significant_mask']
        })
        wilcoxon_df.to_csv(csv_file, index=False)
        analysis_results['files_created'].append(str(csv_file))
        print(f"  Saved: {csv_file}")
    
    # Analyze trial types (if present)
    has_task_events = aggregated_results.get('has_task_events', False)
    
    if has_task_events and 'trial_types' in aggregated_results:
        print("\nAnalyzing trial-type classifications...")
        trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
        
        for trial_type in trial_types:
            if trial_type not in aggregated_results['trial_types']:
                continue
            
            trial_data = aggregated_results['trial_types'][trial_type]
            if 'all_mean_scores_time' not in trial_data:
                continue
            
            print(f"\n  {trial_type}:")
            all_scores = np.array(trial_data['all_mean_scores_time'])
            print(f"    Data shape: {all_scores.shape}")
            
            # Run Wilcoxon test
            wilcoxon_result = run_wilcoxon_per_timepoint(all_scores, alternative='greater')
            analysis_results['trial_types'][trial_type] = wilcoxon_result
            
            n_sig = np.sum(wilcoxon_result['significant_mask'])
            n_total = wilcoxon_result['n_timepoints']
            print(f"    Significant timepoints: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")
            
            # Save to CSV
            csv_file = output_dir / f"{trial_type}_wilcoxon_per_timepoint.csv"
            wilcoxon_df = pd.DataFrame({
                'timepoint_index': np.arange(len(wilcoxon_result['p_values'])),
                'time_seconds': times if times is not None else np.arange(len(wilcoxon_result['p_values'])),
                'p_value': wilcoxon_result['p_values'],
                'test_statistic': wilcoxon_result['test_statistics'],
                'significant': wilcoxon_result['significant_mask']
            })
            wilcoxon_df.to_csv(csv_file, index=False)
            analysis_results['files_created'].append(str(csv_file))
            print(f"    Saved: {csv_file}")
    else:
        print("\nSkipping trial-type analysis (resting state data or no task events)")
    
    # Save summary
    summary_file = output_dir / "statistical_analysis_summary.json"
    summary = {
        'analysis_type': 'wilcoxon_per_timepoint',
        'alternative_hypothesis': 'greater',
        'significance_threshold': 0.01,
        'has_task_events': has_task_events,
        'overall': {
            'n_subjects': analysis_results['overall']['n_subjects'] if analysis_results['overall'] else 0,
            'n_timepoints': analysis_results['overall']['n_timepoints'] if analysis_results['overall'] else 0,
            'n_significant': int(np.sum(analysis_results['overall']['significant_mask'])) if analysis_results['overall'] else 0
        },
        'trial_types': {}
    }
    
    for trial_type, result in analysis_results['trial_types'].items():
        summary['trial_types'][trial_type] = {
            'n_subjects': result['n_subjects'],
            'n_timepoints': result['n_timepoints'],
            'n_significant': int(np.sum(result['significant_mask']))
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    analysis_results['files_created'].append(str(summary_file))
    print(f"\nSummary saved: {summary_file}")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"Files created: {len(analysis_results['files_created'])}")
    for file in analysis_results['files_created']:
        print(f"  - {Path(file).name}")
    print(f"{'='*60}\n")
    
    return analysis_results


def main():
    """
    Run statistical analysis on decoder results.
    
    Example:
        python analysis.py --results-dir /path/to/decoding-global-20231115_120000
    """
    parser = argparse.ArgumentParser(
        description='Perform statistical analysis on decoder results'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to the decoding-global-XXXXXXXX directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Optional custom output directory (defaults to results-dir/data)')
    
    args = parser.parse_args()
    
    analyze_decoder_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
