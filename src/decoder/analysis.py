import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from pathlib import Path
import os 
import mne
import argparse
import pandas as pd
from scipy.stats import trim_mean



# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

class PeakDetector:
    def __init__(self, data_dict: dict, output_dir: str, plot_stim = True):
        self.data = np.array(list(data_dict.values()))
        self.ids = np.array(list(data_dict.keys()))
        self.output_dir = output_dir
        self.plot_stim = plot_stim
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def peak_detection(self) -> dict:
        data_peak = []
        id_peak = []

        for i in range(len(self.data)):
            if find_peaks(self.data[i] - self.data.mean(axis=0), prominence=0.2)[0].size > 0:
                data_peak.append(self.data[i])
                id_peak.append(self.ids[i])
        
        for j in range(len(data_peak)):
            plt.plot(np.linspace(-0.2, 1.34, data_peak[j].shape[0]), data_peak[j])
            if self.plot_stim:
                plt.vlines(0, 0.45, 1, color = 'black', lw = 0.7, ls = '--')
                plt.vlines(0.15, 0.45, 1, color = 'black', lw = 0.7, ls = '--')
                plt.vlines(0.3, 0.45, 1, color = 'black', lw = 0.7, ls = '--')
                plt.vlines(0.45, 0.45, 1, color = 'black', lw = 0.7, ls = '--')
                plt.vlines(0.6, 0.45, 1, color = 'black', lw = 0.7, ls = '--')
        
        # Ensure output directory exists before saving
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{self.output_dir}/peak_detection.png')
        plt.show()

        # Convert numpy arrays to lists for JSON serialization
        result = dict(zip(id_peak, data_peak))
        result_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        
        with open(f'{self.output_dir}/peak_detection.json', 'w') as f:
            json.dump(result_json, f, indent=2)

        return result

    def _find_subjects_sessions(self, main_path: str, ids_sess: list):
        ids = [i.split("_")[0] for i in ids_sess]
        sessions = [i.split("_")[1] for i in ids_sess]

        parent_folder = Path(main_path)
        found_paths = []

        for i, id in enumerate(ids):
            folder_path = parent_folder / f"sub-{id}"
            if folder_path.exists() and folder_path.is_dir():
                sub_sessions = folder_path / f'ses-{sessions[i]}'
                if sub_sessions.exists() and sub_sessions.is_dir():
                    found_paths.append(str(sub_sessions))  
        
        return found_paths

    
    def plot_mean_epochs_sensors_mne(self, paths: list, peak_data: dict, plot_std: bool = False, metric: str = 'mean') -> dict:
        res = {}
        
        for path in paths:
            # Extract subject and session info from path
            sub_info = path.split('/')[-2]  # sub-XXX
            sess_info = path.split('/')[-1]  # ses-YYY
            
            # Get the ID format used in peak_data (e.g., "XXX_YYY")
            sub_id = sub_info.replace('sub-', '')
            sess_id = sess_info.replace('ses-', '')
            peak_id = f"{sub_id}_{sess_id}"
            
            # Get peak data for this subject
            if peak_id not in peak_data:
                print(f"Warning: {peak_id} not found in peak_data, skipping...")
                continue
                
            peak_signal = peak_data[peak_id]
            
            # Read original and reconstructed epochs
            dir_list = [f'{path}/{f}' for f in os.listdir(path)]
            print(f"Processing {path}")
            print(dir_list)
            
            # Create 2-column subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left column: Original and Reconstructed signals with std bands
            for dir_path in dir_list:
                epochs = mne.read_epochs(dir_path)
                print(f"Reading: {dir_path}")
                
                # Get raw data
                epochs_data = epochs.get_data()
                print(f"Shape: {epochs_data.shape}")
                
                # Calculate mean across epochs and sensors
                if metric == 'mean':
                    # Calculate mean and std across epochs and sensors
                    mean_timeseries = np.mean(epochs_data, axis=(0, 1))  # Mean across epochs and sensors

                elif metric == 'trim_mean':
                    #trim_mean 80
                    mean_timeseries = trim_mean(epochs_data, 0.1, axis = 0)
                    print('shape', mean_timeseries.shape)
                    mean_timeseries = trim_mean(mean_timeseries, 0.1, axis = 0)
                elif metric == 'median':
                    mean_timeseries = np.median(epochs_data, axis = (0,1))
                else:
                    print('Did not define a correct metric')    

                # Calculate std across epochs and sensors (point by point through time)
                std_timeseries = np.std(epochs_data, axis=(0, 1))  # Std across epochs and sensors

                print(f"Mean/Median/Trim_Mean shape: {mean_timeseries.shape}, Std shape: {std_timeseries.shape}")
                
                # Create time axis
                time_axis = np.linspace(-200, 1340, mean_timeseries.shape[0])
                
                if 'original' in dir_path:
                    label = 'Original'
                    color = 'C0'
                    res[(sub_info, sess_info, 'original')] = mean_timeseries.tolist()
                    
                    # Plot mean line
                    ax1.plot(time_axis, mean_timeseries, label=label, linewidth=1.5, color=color)
                    
                    # Plot std band (optional)
                    if plot_std:
                        ax1.fill_between(time_axis, 
                                        mean_timeseries - std_timeseries, 
                                        mean_timeseries + std_timeseries,
                                        alpha=0.2, color=color, label=f'{label} ±std')
                else:
                    label = 'Reconstructed'
                    color = 'C1'
                    res[(sub_info, sess_info, 'reverted')] = mean_timeseries.tolist()
                    
                    # Plot mean line
                    ax1.plot(time_axis, mean_timeseries, label=label, linewidth=1.5, 
                            linestyle='--', color=color)
                    
                    # Plot std band (optional)
                    if plot_std:
                        ax1.fill_between(time_axis, 
                                        mean_timeseries - std_timeseries, 
                                        mean_timeseries + std_timeseries,
                                        alpha=0.2, color=color, label=f'{label} ±std')
            
            ax1.set_title(f'{metric} Epochs: {sub_info} {sess_info}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude')
            ax1.legend(loc='best', fontsize='small')
            ax1.grid(True, alpha=0.3)
            
            # Right column: Peak detection visualization for this subject
            ax2.plot(np.linspace(-0.2, 1.34, len(peak_signal)), peak_signal, 
                    linewidth=1.5, color='C2')
            if self.plot_stim:
                for stim_time in [0, 0.15, 0.3, 0.45, 0.6]:
                    ax2.axvline(stim_time, ymin=0.45, ymax=1, color='black', 
                              linewidth=0.7, linestyle='--', alpha=0.7)
            
            ax2.set_title(f'Peak Detection: {sub_info} {sess_info}', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{sub_info}_{sess_info}_combined_analysis.png', dpi=150)
            plt.close()
            print(f"Saved: {sub_info}_{sess_info}_combined_analysis.png")

        # Save results to JSON (convert tuples to strings for JSON compatibility)
        res_json = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in res.items()}
        with open(f'{self.output_dir}/subjects_results.json', 'w') as f:
            json.dump(res_json, f, indent=2)

        return res

    def plot_dual_yaxis_combined(self, paths: list, peak_data: dict, plot_std: bool = False, metric: str = 'mean') -> dict:
        """Create a dual y-axis plot combining mean epochs and peak detection in a single subplot.
        
        This plot combines the two subplots from combined_analysis into one:
        - Left y-axis: Mean across Epochs and Sensors (original & reconstructed)
        - Right y-axis: Peak Detection
        - Shared x-axis: Time
        
        Args:
            paths: List of paths to subject/session directories
            peak_data: Dictionary containing peak detection data
            plot_std: Whether to plot standard deviation bands
            
        Returns:
            Dictionary containing results
        """
        res = {}
        
        for path in paths:
            # Extract subject and session info from path
            sub_info = path.split('/')[-2]  # sub-XXX
            sess_info = path.split('/')[-1]  # ses-YYY
            
            # Get the ID format used in peak_data (e.g., "XXX_YYY")
            sub_id = sub_info.replace('sub-', '')
            sess_id = sess_info.replace('ses-', '')
            peak_id = f"{sub_id}_{sess_id}"
            
            # Get peak data for this subject
            if peak_id not in peak_data:
                print(f"Warning: {peak_id} not found in peak_data, skipping...")
                continue
                
            peak_signal = peak_data[peak_id]
            
            # Read original and reconstructed epochs
            dir_list = [f'{path}/{f}' for f in os.listdir(path)]
            print(f"Processing {path} for dual y-axis plot")
            
            # Create single subplot with dual y-axes
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            
            # Left y-axis: Mean Epochs (Original and Reconstructed)
            mean_timeseries_list = []
            for dir_path in dir_list:
                epochs = mne.read_epochs(dir_path)
                
                # Get raw data
                epochs_data = epochs.get_data()
                
                if metric == 'mean':
                    # Calculate mean and std across epochs and sensors
                    mean_timeseries = np.mean(epochs_data, axis=(0, 1))  # Mean across epochs and sensors

                elif metric == 'trim_mean':
                    #trim_mean 80
                    mean_timeseries = trim_mean(epochs_data, 0.1, axis = 0)
                    print('shape', mean_timeseries.shape)
                    mean_timeseries = trim_mean(mean_timeseries, 0.1, axis = 0)

                elif metric == 'median':
                    mean_timeseries = np.median(epochs_data, axis = (0,1))
                else:
                    print('Did not define a correct metric')

                std_timeseries = np.std(epochs_data, axis=(0, 1))

                # Create time axis in seconds (matching peak detection scale)
                time_axis = np.linspace(-0.2, 1.34, mean_timeseries.shape[0])
                
                if 'original' in dir_path:
                    label = 'Original'
                    color = 'C0'
                    res[(sub_info, sess_info, 'original')] = mean_timeseries.tolist()
                    
                    # Plot mean line on left y-axis
                    ax1.plot(time_axis, mean_timeseries, label=label, linewidth=2, color=color)
                    
                    # Plot std band (optional)
                    if plot_std:
                        ax1.fill_between(time_axis, 
                                        mean_timeseries - std_timeseries, 
                                        mean_timeseries + std_timeseries,
                                        alpha=0.2, color=color)
                else:
                    label = 'Reconstructed'
                    color = 'C1'
                    res[(sub_info, sess_info, 'reverted')] = mean_timeseries.tolist()
                    
                    # Plot mean line on left y-axis
                    ax1.plot(time_axis, mean_timeseries, label=label, linewidth=2, 
                            linestyle='--', color=color)
                    
                    # Plot std band (optional)
                    if plot_std:
                        ax1.fill_between(time_axis, 
                                        mean_timeseries - std_timeseries, 
                                        mean_timeseries + std_timeseries,
                                        alpha=0.2, color=color)
            
            # Configure left y-axis
            ax1.set_xlabel('Time (s)', fontsize=12)
            if metric == 'trim_mean':
                ax1.set_ylabel('Trim mean across Epochs and Sensors', fontsize=12, color='black')
            elif metric == 'median':
                ax1.set_ylabel('Median across Epochs and Sensors', fontsize=12, color='black')
            elif metric == 'mean':
                ax1.set_ylabel('Mean across Epochs and Sensors', fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            
            # Create right y-axis for peak detection
            ax2 = ax1.twinx()
            
            # Plot peak detection on right y-axis
            peak_time_axis = np.linspace(-0.2, 1.34, len(peak_signal))
            ax2.plot(peak_time_axis, peak_signal, linewidth=2, color='C2', 
                    label='Peak Detection', alpha=0.8)
            
            # Add stimulus lines
            if self.plot_stim:
                for stim_time in [0, 0.15, 0.3, 0.45, 0.6]:
                    ax1.axvline(stim_time, color='black', linewidth=0.7, 
                              linestyle='--', alpha=0.5)
            
            # Configure right y-axis
            ax2.set_ylabel('Peak Detection', fontsize=12, color='C2')
            ax2.tick_params(axis='y', labelcolor='C2')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize='small')
            
            # Set title
            ax1.set_title(f'Combined Analysis (Dual Y-Axis): {sub_info} {sess_info}', 
                         fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{sub_info}_{sess_info}_dual_yaxis_combined.png', dpi=150)
            plt.close()
            print(f"Saved: {sub_info}_{sess_info}_dual_yaxis_combined.png")
        
        return res

    def plot_metadata_boxplots(self, peak_data: dict, metadata_path: str = '/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv'):
        """Create boxplots showing clinical outcomes for subjects with pronounced peaks.
        
        Args:
            peak_data: Dictionary of subjects with peaks (from peak_detection)
            metadata_path: Path to the metadata CSV file
        """
        # Load metadata
        metadata = pd.read_csv(metadata_path)
        
        # Get subject IDs with peaks (format: "XXXXX_YY")
        peak_ids = list(peak_data.keys())
        peak_subjects = [pid.split('_')[0] for pid in peak_ids]
        peak_sessions = [pid.split('_')[1] for pid in peak_ids]
        
        # Filter metadata to only include subjects/sessions with peaks
        metadata_filtered = []
        for subj, sess in zip(peak_subjects, peak_sessions):
            # Match subject (without 'AA' prefix in metadata vs with it in peak_ids)
            match = metadata[(metadata['subject'] == subj) & (metadata['session'] == int(sess))]
            if not match.empty:
                metadata_filtered.append(match)
        
        if not metadata_filtered:
            print("Warning: No matching subjects found in metadata")
            return
        
        df = pd.concat(metadata_filtered, ignore_index=True)
        print(f"Found {len(df)} subjects with peaks in metadata")
        
        # Merge categories
        def merge_categories(value):
            if pd.isna(value):
                return 'n/a'
            value_str = str(value).upper()
            # Merge MCS variants
            if 'MCS' in value_str:
                return 'MCS'
            # Merge UWS/VS
            if value_str in ['UWS', 'VS']:
                return 'VS'
            return value_str
        
        # Apply merging to relevant columns
        columns_to_plot = ['state', 'cs_1y', 'cs_2y', 'cs_6m', 'diagnostic_crs_final']
        for col in columns_to_plot:
            if col in df.columns:
                df[f'{col}_merged'] = df[col].apply(merge_categories)
        
        # Create figure with 5 subplots
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Plot each variable as a boxplot
        for idx, col in enumerate(columns_to_plot):
            merged_col = f'{col}_merged'
            if merged_col not in df.columns:
                axes[idx].text(0.5, 0.5, f'{col}\nNot available', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_title(col.replace('_', ' ').title())
                continue
            
            # Get category counts
            category_data = df[merged_col].value_counts()
            categories = category_data.index.tolist()
            counts = category_data.values.tolist()
            
            # Create boxplot-style visualization (bar plot with counts)
            axes[idx].bar(range(len(categories)), counts, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_xticks(range(len(categories)))
            axes[idx].set_xticklabels(categories, rotation=45, ha='right')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(col.replace('_', ' ').title(), fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                axes[idx].text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metadata_clinical_outcomes_peaks.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: metadata_clinical_outcomes_peaks.png")
        
        # Save filtered metadata
        df.to_csv(f'{self.output_dir}/subjects_with_peaks_metadata.csv', index=False)
        print(f"Saved: subjects_with_peaks_metadata.csv")

    '''
    def plot_mean_epochs_sensors_pydata(self) -> dict:
        res = {}

        for path in paths:
            dir_list = [f'{path}/original.npy', f'{path}/reverted.npy']
            print(dir_list)
            for dir_path in dir_list:
                sess_info = dir_path.split('/')[-2]
                sub_info = dir_path.split('/')[-3]

                epochs = np.load(dir_path)
                print(dir_path)
                print(epochs.shape)

                epochs = np.mean(epochs, axis=2) 
                print(epochs.shape)
                epochs = np.mean(epochs, axis=0)
                print(epochs.shape)            
                
                if 'original' in dir_path:
                    label = 'Original'
                    res[(sub_info, sess_info, 'original')] = epochs
                else:
                    label = 'Reconstructed'
                    res[(sub_info, sess_info, 'reverted')] = epochs

                
                plt.title(f'Epochs for subject {sub_info} session {sess_info}')
                plt.plot(np.linspace(-200, 1340, epochs.shape[0]), epochs, label=label)
            
            plt.legend()
            plt.show()
    '''
        
    def run_all_analysis_save_results(self, main_path: str, metadata_path: str = '/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv'):
        """Orchestrates the full analysis pipeline.
        
        Args:
            main_path: Path to the main data directory containing subject folders
            metadata_path: Path to the metadata CSV file
        """
        print("=" * 60)
        print("Starting full analysis pipeline")
        print("=" * 60)
        
        # Step 1: Run peak detection to identify subjects with large peaks
        print("\n[1/4] Running peak detection...")
        peak_data = self.peak_detection()
        print(f"Found {len(peak_data)} subjects with significant peaks")
        print(f"Subject IDs: {list(peak_data.keys())}")
        
        # Step 2: Find subject session paths for those with peaks
        print("\n[2/4] Finding subject session paths...")
        ids_sess = list(peak_data.keys())
        paths = self._find_subjects_sessions(main_path, ids_sess)
        print(f"Found {len(paths)} paths:")
        for p in paths:
            print(f"  - {p}")
        
        # Step 3: Plot mean epochs with peak detection side-by-side
        print("\n[3/5] Generating combined plots (epochs + peaks)...")
        results = self.plot_mean_epochs_sensors_mne(paths, peak_data)
        
        # Step 4: Generate dual y-axis plots (new)
        print("\n[4/5] Generating dual y-axis combined plots...")
        self.plot_dual_yaxis_combined(paths, peak_data)
        
        # Step 5: Generate metadata boxplots for clinical outcomes
        print("\n[5/5] Generating metadata boxplots for clinical outcomes...")
        self.plot_metadata_boxplots(peak_data, metadata_path)
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return results
            
        

def main():
    parser = argparse.ArgumentParser(description='Global analysis across subjects')
    parser.add_argument('--decoder-global-data', 
                       default='/data/project/eeg_foundation/src/doc_benchmark/results/DECODER/decoding-global-20251029_082136/data/',
                       help='Path to decoder global analysis')
    parser.add_argument('--main-data-path', 
                       default='/data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata',
                       help='Path to the main data directory containing subject folders (e.g., sub-XXX/ses-YYY)')
    parser.add_argument('--output-dir', 
                       required=True,
                       help='Output directory for analysis')
    parser.add_argument('--metadata', 
                       default='/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv',
                       help='Path to the metadata CSV file')
    parser.add_argument('--run-analysis', 
                       action='store_true',
                       help='Run all the analysis')
    parser.add_argument('--subject-type', 
                       default=None, 
                       help='VS or MCS. If not specified, it will consider all subjects.')

    args = parser.parse_args()

    # Load data based on subject type
    if args.subject_type == 'MCS':
        data = np.load(f'{args.decoder_global_data}/MCS.npy', allow_pickle=True).item()
    elif args.subject_type == 'VS':
        data = np.load(f'{args.decoder_global_data}/VS.npy', allow_pickle=True).item()
    else:
        data = np.load(f'{args.decoder_global_data}/all_mean_scores_time_dict.npy', allow_pickle=True).item()

    # Initialize PeakDetector
    peak_det = PeakDetector(data, args.output_dir)

    # Run the full analysis pipeline
    if args.run_analysis:
        peak_det.run_all_analysis_save_results(args.main_data_path, args.metadata)

if __name__ == '__main__':
    main()