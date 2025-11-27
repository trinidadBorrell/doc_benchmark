#!/usr/bin/env python3
"""
Enhanced plots for OHBM presentation - 4x1 layout with blue tones.
Creates improved version of aggregated_by_subject_type.png with:
- 4 rows, 1 column layout (EMCS, MCS, UWS, COMA)
- Different blue tones for each group
- Filtering by diagnostic_crs_final column
- Shared x-axis with clean labels
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
import json
from datetime import datetime

COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": False,
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
    }
)


def add_stimulus_lines(ax, times):
    """Add vertical lines at stimulus times to a plot."""
    stimulus_times = [0.0, 0.15, 0.3, 0.45, 0.6]
    for i, stim_time in enumerate(stimulus_times):
        if times[0] <= stim_time <= times[-1]:  # Only plot if within time range
            #label = 'Stimuli' if i == 0 else None
            ax.axvline(stim_time, color='darkgray', linestyle='--', alpha=1, linewidth=1.5)

def load_decoder_results(output_dir):
    """
    Load all decoder results from individual subject/session analyses.
    
    Parameters:
        output_dir: Path to output directory containing decoder results
        
    Returns:
        all_results: list of result dictionaries
        subjects_sessions: list of (subject_id, session_id) tuples
        times: time points array
    """
    print("Loading decoder results...")
    
    output_path = Path(output_dir)
    all_results = []
    subjects_sessions = []
    times = None
    
    # Find all individual results
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
                        if 'overall' in results and 'mean_scores_time' in results['overall']:
                            n_times = len(results['overall']['mean_scores_time'])
                            # Standard EEG epoch times
                            times = np.linspace(-0.2, 0.8, n_times)  # -200ms to 800ms
                    
                    all_results.append(results)
                    subjects_sessions.append((subject_id, session_id))
                    print(f"  Loaded: sub-{subject_id} ses-{session_id}")
                    
                except Exception as e:
                    print(f"  Error loading sub-{subject_id} ses-{session_id}: {e}")
    
    if not all_results:
        print("No decoder results found!")
        return None, None, None
    
    print(f"Found {len(all_results)} decoder results")
    return all_results, subjects_sessions, times

def group_by_diagnostic_group(subjects_sessions, patient_labels_path):
    """
    Group subjects by diagnostic_crs_final values.
    
    Parameters:
        subjects_sessions: list of (subject_id, session_id) tuples
        patient_labels_path: path to patient labels CSV file
        
    Returns:
        subject_group_map: dictionary mapping key -> diagnostic group
    """
    if not patient_labels_path or not Path(patient_labels_path).exists():
        print("Warning: Patient labels file not found")
        return {}
    
    print("Loading patient labels...")
    subject_group_map = {}
    
    try:
        df = pd.read_csv(patient_labels_path, dtype={'session': str})
        for _, row in df.iterrows():
            # Ensure session is zero-padded to 2 digits
            session_str = str(row['session']).zfill(2)
            key = f"{row['subject']}_{session_str}"
            diagnostic = row['diagnostic_crs_final']
            
            # Skip rows with missing diagnostic values
            if pd.isna(diagnostic) or diagnostic == 'n/a' or diagnostic == '':
                continue
            
            # Group into 4 categories
            if diagnostic == 'EMCS':
                group = 'EMCS'
            elif diagnostic in ['MCS', 'MCS-', 'MCS+']:
                group = 'MCS'
            elif diagnostic in ['VS', 'UWS']:
                group = 'UWS'
            elif diagnostic == 'COMA':
                group = 'COMA'
            else:
                group = diagnostic  # Keep others as-is
            
            subject_group_map[key] = group
            
    except Exception as e:
        print(f"Warning: Could not load patient labels: {e}")
        return {}
    
    # Print distribution
    group_counts = {}
    for group in subject_group_map.values():
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print(f"  Diagnostic group distribution:")
    for group, count in sorted(group_counts.items()):
        print(f"    {group}: {count} subject-sessions")
    
    return subject_group_map

def create_enhanced_ohbm_plot(all_results, subjects_sessions, times, subject_group_map, output_dir):
    """
    Create enhanced 4x1 plot with blue tones for OHBM presentation.
    
    Parameters:
        all_results: list of decoder result dictionaries
        subjects_sessions: list of (subject_id, session_id) tuples
        times: time points array
        subject_group_map: dictionary mapping key -> diagnostic group
        output_dir: directory to save the plot
    """
    print("Creating enhanced OHBM plot...")
    
    # Define target groups and blue color palette using matplotlib Blues colormap
    target_groups = ['EMCS', 'MCS', 'UWS', 'COMA']
    import matplotlib.cm as cm
    blues_cmap = cm.get_cmap('Blues')
    blue_palette = {
        'EMCS': blues_cmap(0.95),   # Very dark blue (90% of colormap)
        'MCS': blues_cmap(0.8),    # Dark blue (70% of colormap)
        'UWS': blues_cmap(0.65),    # Medium blue (50% of colormap)
        'COMA': blues_cmap(0.5)    # Light blue (30% of colormap)
    }
    
    # Group results by diagnostic category
    grouped_results = {}
    for group in target_groups:
        grouped_results[group] = []
    
    for i, (subject_id, session_id) in enumerate(subjects_sessions):
        key = f"{subject_id}_{session_id}"
        group = subject_group_map.get(key, 'UNKNOWN')
        
        if group in target_groups and 'overall' in all_results[i]:
            grouped_results[group].append(all_results[i]['overall'])
    
    # Calculate statistics for each group
    group_stats = {}
    for group in target_groups:
        if len(grouped_results[group]) >= 2:  # Need at least 2 subjects
            all_scores = np.array([res['mean_scores_time'] for res in grouped_results[group]])
            all_aucs = [res['mean_auc'] for res in grouped_results[group]]
            
            # Select all elements except the last 20 time points
            all_scores_trimmed = all_scores[:, :-10] if all_scores.shape[1] > 10 else all_scores
            
            group_stats[group] = {
                'mean_scores_time': np.mean(all_scores_trimmed, axis=0),
                'std_scores_time': np.std(all_scores_trimmed, axis=0),
                'mean_auc': np.mean(all_aucs),
                'std_auc': np.std(all_aucs),
                'n_subjects': len(grouped_results[group])
            }
            print(f"  {group}: {len(grouped_results[group])} subjects, "
                  f"Mean AUC = {group_stats[group]['mean_auc']:.3f} ± {group_stats[group]['std_auc']:.3f}")
        else:
            print(f"  {group}: insufficient data ({len(grouped_results[group])} subjects)")
    
    # Create 4x1 subplot
    available_groups = [g for g in target_groups if g in group_stats]
    
    if not available_groups:
        print("No groups with sufficient data for plotting!")
        return
    
    # Trim times array to match trimmed data (exclude last 20 time points)
    times_trimmed = times[:-10] if len(times) > 10 else times
    
    fig, axes = plt.subplots(len(available_groups), 1, figsize=(8, 16), sharex=True)
    
    if len(available_groups) == 1:
        axes = [axes]  # Make it iterable
    
    for i, group in enumerate(available_groups):
        ax = axes[i]
        stats = group_stats[group]
        
        # Plot mean line with SD shading
        ax.plot(times_trimmed, stats['mean_scores_time'], 
               color=blue_palette[group], linewidth=2.5, 
               label=f'{group} (n={stats["n_subjects"]})')
        
        ax.fill_between(times_trimmed, 
                       stats['mean_scores_time'] - stats['std_scores_time'],
                       stats['mean_scores_time'] + stats['std_scores_time'],
                       alpha=0.3, color=blue_palette[group])
        
        # Add chance line and stimulus markers
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
        add_stimulus_lines(ax, times_trimmed)
        
        # Labels and formatting
        ax.set_ylabel("AUC", fontsize=22)
      #  ax.set_title(f"{group} - Mean AUC: {stats['mean_auc']:.3f} ± {stats['std_auc']:.3f} (n={stats['n_subjects']})", 
      #              fontsize=18, fontweight='bold')
        ax.legend(fontsize=22, loc='upper right')
        ax.set_ylim([0.4, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Only show x-axis label on bottom subplot
        if i == len(available_groups) - 1:
            ax.set_xlabel("Time (s)", fontsize=22)
    
    #plt.suptitle("Decoder Performance by Diagnostic Group (Original vs Reconstructed EEG)\n" +
    #            "Enhanced OHBM Presentation - Blue Tone Gradient", 
    #            fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "DECODER_by_diagnostic_group.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced OHBM plot saved to: {plot_path}")
    return plot_path

def create_trial_type_grid_plot(all_results, subjects_sessions, times, subject_group_map, output_dir):
    """
    Create enhanced 4x4 grid plot with trial types as columns and diagnostic groups as rows.
    
    Parameters:
        all_results: list of decoder result dictionaries
        subjects_sessions: list of (subject_id, session_id) tuples
        times: time points array
        subject_group_map: dictionary mapping key -> diagnostic group
        output_dir: directory to save the plot
    """
    print("Creating enhanced OHBM trial type grid plot...")
    
    # Define target groups and trial types
    target_groups = ['EMCS', 'MCS', 'UWS', 'COMA']
    trial_types = ['LSGS', 'LSGD', 'LDGD', 'LDGS']
    col_cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
    
    # Group results by diagnostic category AND trial type
    grouped_results = {}
    for group in target_groups:
        grouped_results[group] = {}
        for trial_type in trial_types:
            grouped_results[group][trial_type] = []
    
    for i, (subject_id, session_id) in enumerate(subjects_sessions):
        key = f"{subject_id}_{session_id}"
        group = subject_group_map.get(key, 'UNKNOWN')
        
        if group in target_groups and 'trial_types' in all_results[i]:
            for trial_type in trial_types:
                if trial_type in all_results[i]['trial_types']:
                    grouped_results[group][trial_type].append(all_results[i]['trial_types'][trial_type])
    
    # Calculate statistics for each group x trial type combination
    group_stats = {}
    for group in target_groups:
        group_stats[group] = {}
        for trial_type in trial_types:
            if len(grouped_results[group][trial_type]) >= 1:  # Need at least 1 subject
                all_scores = np.array([res['mean_scores_time'] for res in grouped_results[group][trial_type]])
                all_aucs = [res['mean_auc'] for res in grouped_results[group][trial_type]]
                
                # Select all elements except the last 10 time points
                all_scores_trimmed = all_scores[:, :-10] if all_scores.shape[1] > 10 else all_scores
                
                group_stats[group][trial_type] = {
                    'mean_scores_time': np.mean(all_scores_trimmed, axis=0),
                    'std_scores_time': np.std(all_scores_trimmed, axis=0),
                    'mean_auc': np.mean(all_aucs),
                    'std_auc': np.std(all_aucs),
                    'n_subjects': len(grouped_results[group][trial_type])
                }
                print(f"  {group}-{trial_type}: {len(grouped_results[group][trial_type])} subjects, "
                      f"Mean AUC = {group_stats[group][trial_type]['mean_auc']:.3f} ± {group_stats[group][trial_type]['std_auc']:.3f}")
            else:
                print(f"  {group}-{trial_type}: insufficient data ({len(grouped_results[group][trial_type])} subjects)")
                group_stats[group][trial_type] = None
    
    # Create 4x4 subplot
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    
    # Trim times array to match trimmed data (exclude last 10 time points)
    times_trimmed = times[:-10] if len(times) > 10 else times
    
    # Get colormaps
    from matplotlib import cm
    colormaps = [cm.get_cmap(cmap_name) for cmap_name in col_cmaps]
    
    # Plot each cell
    for row_idx, group in enumerate(target_groups):
        for col_idx, trial_type in enumerate(trial_types):
            ax = axes[row_idx, col_idx]
            stats = group_stats[group][trial_type]
            
            if stats is not None:
                # Get color from colormap (darker for higher consciousness level)
                color = colormaps[col_idx](0.95 - row_idx * 0.15)
                
                # Plot mean line with SD shading
                ax.plot(times_trimmed, stats['mean_scores_time'], 
                       color=color, linewidth=2.5)
                
                ax.fill_between(times_trimmed, 
                               stats['mean_scores_time'] - stats['std_scores_time'],
                               stats['mean_scores_time'] + stats['std_scores_time'],
                               alpha=0.3, color=color)
                
                # Add chance line
                ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
                
                # Add stimulus markers
                add_stimulus_lines(ax, times_trimmed)
                
                # Add text annotation with subject count
                ax.text(0.02, 0.98, f"n={stats['n_subjects']}", 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No data available
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, color='gray')
                ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
            
            # Set labels and formatting
            if col_idx == 0:
                ax.set_ylabel(group, fontsize=16, fontweight='bold')
            if row_idx == 3:  # Bottom row
                ax.set_xlabel("Time (s)", fontsize=14)
            
            ax.set_ylim([0.4, 1.0])
            ax.grid(True, alpha=0.3)
            
            # Add column titles (trial types)
            if row_idx == 0:
                ax.set_title(trial_type, fontsize=16, fontweight='bold')
    
    plt.suptitle("Decoder Performance by Diagnostic Group and Trial Type\n" +
                "Rows: EMCS → MCS → UWS → COMA | Columns: LSGS → LSGD → LDGD → LDGS", 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "DECODER_by_diagnostic_group_trial_type.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced OHBM trial type grid plot saved to: {plot_path}")
    return plot_path

def create_aggregated_overall_comparison_plot(output_dir):
    """
    Create aggregated overall classification comparison plot from three different datasets.
    
    Parameters:
        output_dir: directory to save the plot
    """
    print("Creating aggregated overall comparison plot...")
    
    # Define the three datasets
    datasets = [
        {
            'name': 'Control LG',
            'path': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/control_lg/new_results/DECODER',
            'color': 'red'
        },
        {
            'name': 'DoC', 
            'path': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/DECODER/decoder_results_2025-10-03_153133',
            'color': 'blue'
        },
        {
            'name': 'Control RS',
            'path': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/new_results/DECODER', 
            'color': 'green'
        }
    ]
    
    # Create 3x1 subplot
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    
    for idx, dataset in enumerate(datasets):
        print(f"  Loading {dataset['name']} data...")
        
        # Load results for this dataset
        all_results, subjects_sessions, times = load_decoder_results(dataset['path'])
        
        if all_results is None:
            print(f"    No results found for {dataset['name']}")
            axes[idx].text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=14, color='gray')
            axes[idx].set_ylabel("AUC", fontsize=16)
            continue
        
        # Extract overall results and calculate statistics
        overall_scores = []
        for result in all_results:
            if 'overall' in result:
                overall_scores.append(result['overall']['mean_scores_time'])
        
        if not overall_scores:
            print(f"    No overall results found for {dataset['name']}")
            axes[idx].text(0.5, 0.5, 'No Overall Data', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=14, color='gray')
            axes[idx].set_ylabel("AUC", fontsize=16)
            continue
        
        # Calculate aggregated statistics
        all_scores_array = np.array(overall_scores)
        
        # Select all elements except the last 10 time points
        all_scores_trimmed = all_scores_array[:, :-10] if all_scores_array.shape[1] > 10 else all_scores_array
        times_trimmed = times[:-10] if len(times) > 10 else times
        
        mean_scores = np.mean(all_scores_trimmed, axis=0)
        std_scores = np.std(all_scores_trimmed, axis=0)
        
        # Plot mean line with SD shading
        ax = axes[idx]
        if dataset['name'] == 'DoC':
            label_text = f'{dataset["name"]} (n=161)'
        else:
            label_text = f'{dataset["name"]} (n={len(overall_scores)})'
        ax.plot(times_trimmed, mean_scores, color=dataset['color'], linewidth=2.5, 
               label=label_text)
        
        ax.fill_between(times_trimmed, 
                       mean_scores - std_scores, mean_scores + std_scores,
                       alpha=0.3, color=dataset['color'])
        
        # Add chance line and stimulus markers
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
        add_stimulus_lines(ax, times_trimmed)
        
        # Labels and formatting
        ax.set_ylabel("AUC", fontsize=22)
        ax.legend(fontsize=22, loc='upper right')
        ax.set_ylim([0.4, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3)
        
        # Only show x-axis label on bottom subplot
        if idx == 2:
            ax.set_xlabel("Time (s)", fontsize=22)
        
        print(f"    {dataset['name']}: {len(overall_scores)} subjects processed")
    
   # plt.suptitle("Aggregated Overall Classification Comparison\n" +
   #             "Control LG (Red) | DoC Patients (Blue) | Control Resting State (Green)", 
   #             fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "DECODER_aggregated_overall.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregated overall comparison plot saved to: {plot_path}")
    return plot_path

def main():
    """Main function to create enhanced OHBM plots."""
    
    # Configuration - adjust these paths as needed
    decoder_results_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/DECODER/decoder_results_2025-10-03_153133"
    patient_labels_path = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    output_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/OHBM"
    
    print("=" * 60)
    print("Creating Enhanced OHBM Plots")
    print("=" * 60)
    
    # Load decoder results
    all_results, subjects_sessions, times = load_decoder_results(decoder_results_dir)
    
    if all_results is None:
        print("Failed to load decoder results. Exiting.")
        return
    
    # Load patient labels and create grouping
    subject_group_map = group_by_diagnostic_group(subjects_sessions, patient_labels_path)
    
    if not subject_group_map:
        print("No patient label data available. Exiting.")
        return
    
    # Create enhanced plot (original)
    plot_path = create_enhanced_ohbm_plot(
        all_results, subjects_sessions, times, subject_group_map, output_dir
    )
    
    # Create trial type grid plot (new)
    trial_type_plot_path = create_trial_type_grid_plot(
        all_results, subjects_sessions, times, subject_group_map, output_dir
    )
    
    # Create aggregated overall comparison plot (new)
    comparison_plot_path = create_aggregated_overall_comparison_plot(output_dir)
    
    if plot_path and trial_type_plot_path and comparison_plot_path:
        print("\n" + "=" * 60)
        print("Enhanced OHBM plots creation completed successfully!")
        print(f"Original plot: {plot_path}")
        print(f"Trial type grid plot: {trial_type_plot_path}")
        print(f"Aggregated overall comparison plot: {comparison_plot_path}")
        print("=" * 60)
    else:
        print("\nFailed to create enhanced plots.")

if __name__ == "__main__":
    main()
