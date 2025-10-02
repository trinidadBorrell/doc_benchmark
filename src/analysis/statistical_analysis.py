"""Statistical analysis across multiple subjects.

This script performs statistical tests on EEG markers across subjects,
including permutation-based cluster tests and other statistical comparisons.

Authors: Denis A. Engemann, Federico Raimondo, Trinidad Borrell
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import argparse
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Try to import MNE for cluster permutation tests
try:
    import mne
    from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
    HAS_MNE = True
    mne.set_log_level('WARNING')
except ImportError:
    HAS_MNE = False
    print("Warning: MNE-Python not available. Cluster tests will be skipped.")

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StatisticalAnalyzer:
    """Performs statistical analysis on EEG markers across subjects."""
    
    def __init__(self, results_dir, output_dir, patient_labels_file):
        """
        Initialize the statistical analyzer.
        
        Parameters
        ----------
        results_dir : str
            Path to results directory containing subject data
        output_dir : str
            Output directory for statistical results
        patient_labels_file : str
            Path to CSV file with patient labels
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        
        # Create output directories
        self.plots_dir = op.join(output_dir, 'plots')
        self.data_dir = op.join(output_dir, 'data')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load patient labels
        self.patient_labels = self._load_patient_labels()
        
    def _load_patient_labels(self):
        """Load patient labels from CSV file."""
        print(f"Loading patient labels from: {self.patient_labels_file}")
        
        if not op.exists(self.patient_labels_file):
            raise FileNotFoundError(f"Patient labels file not found: {self.patient_labels_file}")
        
        df = pd.read_csv(self.patient_labels_file)
        print(f"Loaded labels for {len(df)} subjects")
        return df
    
    def load_topographic_data(self):
        """
        Load topographic data for all subjects.
        
        Returns
        -------
        dict
            Dictionary with 'original' and 'reconstructed' topographic data
        """
        print("\n📊 Loading topographic data...")
        
        topos_orig_list = []
        topos_recon_list = []
        subject_ids = []
        
        # Find all subject directories
        subject_dirs = sorted(glob.glob(op.join(self.results_dir, 'sub-*')))
        
        for subject_dir in subject_dirs:
            subject_id = op.basename(subject_dir).replace('sub-', '')
            
            # Find session directories
            session_dirs = sorted(glob.glob(op.join(subject_dir, 'ses-*')))
            
            for session_dir in session_dirs:
                features_dir = op.join(session_dir, 'features_variable')
                
                topo_orig_file = op.join(features_dir, 'topos_original.npy')
                topo_recon_file = op.join(features_dir, 'topos_reconstructed.npy')
                
                if op.exists(topo_orig_file) and op.exists(topo_recon_file):
                    topos_orig = np.load(topo_orig_file)
                    topos_recon = np.load(topo_recon_file)
                    
                    topos_orig_list.append(topos_orig)
                    topos_recon_list.append(topos_recon)
                    subject_ids.append(subject_id)
                    
                    print(f"  ✓ Loaded {subject_id}: {topos_orig.shape}")
        
        print(f"\n✅ Loaded data for {len(subject_ids)} subjects")
        
        return {
            'original': np.array(topos_orig_list),
            'reconstructed': np.array(topos_recon_list),
            'subject_ids': subject_ids
        }
    
    def permutation_cluster_test_topos(self, topos_orig, topos_recon, n_permutations=1000):
        """
        Perform permutation-based cluster test on topographic data.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        n_permutations : int
            Number of permutations
            
        Returns
        -------
        dict
            Results including T-values, clusters, and p-values
        """
        print("\n🔬 Performing permutation-based cluster test...")
        
        if not HAS_MNE:
            print("⚠️  MNE not available, skipping cluster test")
            return None
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        print(f"  Data shape: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        
        # Compute differences
        differences = topos_orig - topos_recon
        
        # Perform cluster test for each marker separately
        results = {
            'n_permutations': n_permutations,
            'n_subjects': n_subjects,
            'n_markers': n_markers,
            'n_channels': n_channels,
            'markers': []
        }
        
        for marker_idx in range(n_markers):
            print(f"  Testing marker {marker_idx + 1}/{n_markers}...", end='\r')
            
            # Get data for this marker: (n_subjects, n_channels)
            X = [differences[:, marker_idx, :]]
            
            try:
                # Perform one-sample t-test against zero with cluster correction
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                    X, n_permutations=n_permutations, tail=0, n_jobs=-1,
                    out_type='mask', verbose=False
                )
                
                marker_results = {
                    'marker_idx': int(marker_idx),
                    'T_obs': T_obs.tolist(),
                    'n_clusters': len(clusters),
                    'cluster_p_values': cluster_p_values.tolist() if len(cluster_p_values) > 0 else [],
                    'significant_clusters': int(np.sum(np.array(cluster_p_values) < 0.05)) if len(cluster_p_values) > 0 else 0
                }
                
                results['markers'].append(marker_results)
                
            except Exception as e:
                print(f"\n⚠️  Error testing marker {marker_idx}: {e}")
                results['markers'].append({
                    'marker_idx': int(marker_idx),
                    'error': str(e)
                })
        
        print(f"\n✅ Cluster test completed for {n_markers} markers")
        
        return results
    
    def compute_ssmi(self, topos_orig, topos_recon):
        """
        Compute Sum of Squared Mean Intensity (SSMI) or similar statistic.
        This computes the squared difference between original and reconstructed data.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
            
        Returns
        -------
        dict
            SSMI results per marker and channel
        """
        print("\n📈 Computing SSMI (Sum of Squared Mean Intensity)...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        
        # Compute mean across subjects
        mean_orig = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        mean_recon = np.mean(topos_recon, axis=0)  # (n_markers, n_channels)
        
        # Compute squared differences
        squared_diff = (mean_orig - mean_recon) ** 2
        
        # Sum across channels for each marker
        ssmi_per_marker = np.sum(squared_diff, axis=1)
        
        # Sum across markers for each channel
        ssmi_per_channel = np.sum(squared_diff, axis=0)
        
        # Overall SSMI
        ssmi_total = np.sum(squared_diff)
        
        results = {
            'ssmi_total': float(ssmi_total),
            'ssmi_per_marker': ssmi_per_marker.tolist(),
            'ssmi_per_channel': ssmi_per_channel.tolist(),
            'mean_ssmi_per_marker': float(np.mean(ssmi_per_marker)),
            'std_ssmi_per_marker': float(np.std(ssmi_per_marker)),
            'mean_ssmi_per_channel': float(np.mean(ssmi_per_channel)),
            'std_ssmi_per_channel': float(np.std(ssmi_per_channel)),
            'n_subjects': int(n_subjects),
            'n_markers': int(n_markers),
            'n_channels': int(n_channels)
        }
        
        print(f"  Total SSMI: {ssmi_total:.6f}")
        print(f"  Mean SSMI per marker: {np.mean(ssmi_per_marker):.6f} ± {np.std(ssmi_per_marker):.6f}")
        
        return results
    
    def plot_cluster_test_results(self, cluster_results):
        """Plot cluster test results."""
        if cluster_results is None:
            return
        
        print("\n📊 Plotting cluster test results...")
        
        # Extract significant clusters per marker
        marker_indices = []
        n_significant = []
        
        for marker in cluster_results['markers']:
            if 'error' not in marker:
                marker_indices.append(marker['marker_idx'])
                n_significant.append(marker['significant_clusters'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(marker_indices, n_significant, alpha=0.7, color='steelblue')
        ax.set_xlabel('Marker Index')
        ax.set_ylabel('Number of Significant Clusters (p < 0.05)')
        ax.set_title(f'Permutation Cluster Test Results\n({cluster_results["n_permutations"]} permutations)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = op.join(self.plots_dir, 'cluster_test_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved cluster test plot: {plot_path}")
    
    def plot_ssmi_results(self, ssmi_results):
        """Plot SSMI results."""
        print("\n📊 Plotting SSMI results...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot SSMI per marker
        marker_indices = np.arange(len(ssmi_results['ssmi_per_marker']))
        ax1.bar(marker_indices, ssmi_results['ssmi_per_marker'], alpha=0.7, color='coral')
        ax1.axhline(y=ssmi_results['mean_ssmi_per_marker'], color='red', linestyle='--', 
                   label=f'Mean: {ssmi_results["mean_ssmi_per_marker"]:.6f}')
        ax1.set_xlabel('Marker Index')
        ax1.set_ylabel('SSMI')
        ax1.set_title('SSMI per Marker')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot SSMI per channel
        channel_indices = np.arange(len(ssmi_results['ssmi_per_channel']))
        ax2.bar(channel_indices, ssmi_results['ssmi_per_channel'], alpha=0.7, color='mediumseagreen')
        ax2.axhline(y=ssmi_results['mean_ssmi_per_channel'], color='green', linestyle='--',
                   label=f'Mean: {ssmi_results["mean_ssmi_per_channel"]:.6f}')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('SSMI')
        ax2.set_title('SSMI per Channel')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = op.join(self.plots_dir, 'ssmi_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved SSMI plot: {plot_path}")
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Load data
        data = self.load_topographic_data()
        topos_orig = data['original']
        topos_recon = data['reconstructed']
        subject_ids = data['subject_ids']
        
        # Run cluster permutation test
        cluster_results = self.permutation_cluster_test_topos(topos_orig, topos_recon, n_permutations=1000)
        
        # Compute SSMI
        ssmi_results = self.compute_ssmi(topos_orig, topos_recon)
        
        # Plot results
        if cluster_results:
            self.plot_cluster_test_results(cluster_results)
        
        self.plot_ssmi_results(ssmi_results)
        
        # Save results as JSON
        results_json = {
            'analysis_date': datetime.now().isoformat(),
            'n_subjects': len(subject_ids),
            'subject_ids': subject_ids,
            'cluster_test': cluster_results,
            'ssmi': ssmi_results
        }
        
        json_path = op.join(self.data_dir, 'statistical_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✅ Saved results to: {json_path}")
        
        # Save SSMI as CSV
        ssmi_df = pd.DataFrame({
            'marker_idx': np.arange(len(ssmi_results['ssmi_per_marker'])),
            'ssmi': ssmi_results['ssmi_per_marker']
        })
        
        csv_path = op.join(self.data_dir, 'ssmi_per_marker.csv')
        ssmi_df.to_csv(csv_path, index=False)
        
        print(f"✅ Saved SSMI CSV to: {csv_path}")
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Statistical analysis across multiple subjects')
    parser.add_argument('--results-dir', required=True, help='Path to results directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for statistical results')
    parser.add_argument('--patient-labels', required=True, help='Path to patient labels CSV file')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = StatisticalAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        patient_labels_file=args.patient_labels
    )
    
    analyzer.run_analysis()


if __name__ == '__main__':
    main()

