"""Enhanced marker comparison with statistical analysis and global results.

This script provides comprehensive comparison between original and reconstructed EEG data,
including statistical tests, detailed visualizations, and global analysis across subjects.

Authors: Denis A. Engemann, Federico Raimondo, Trinidad Borrell
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from datetime import datetime
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Set plotting style - apply seaborn first, then override with custom settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "legend.fontsize": 14,
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
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


class MarkerNameMapper:
    """Maps marker indices to human-readable names.
        
    IMPORTANT: Both scalar and topographic features come from the SAME underlying 
    markers, just reduced differently:
    - Scalars: averaged across epochs AND channels → 1 value per marker
    - Topos: averaged across epochs ONLY → 1 value per marker per channel
    
    Therefore, they should have the SAME names since they represent the same markers.
    """
    
    def __init__(self):
        
        self.marker_names = [
            'PowerSpectralDensity_delta',
            'PowerSpectralDensity_deltan',
            'PowerSpectralDensity_theta',
            'PowerSpectralDensity_thetan',
            'PowerSpectralDensity_alpha',
            'PowerSpectralDensity_alphan',
            'PowerSpectralDensity_beta',
            'PowerSpectralDensity_betan',
            'PowerSpectralDensity_gamma',
            'PowerSpectralDensity_gamman',
            'PowerSpectralDensity_summary_se',
            'PowerSpectralDensitySummary_summary_msf',
            'PowerSpectralDensitySummary_summary_sef90',
            'PowerSpectralDensitySummary_summary_sef95',
            'PermutationEntropy_default',
            'SymbolicMutualInformation_weighted',
            'KolmogorovComplexity_default',
            'ContingentNegativeVariation_default',
            'TimeLockedContrast_mmn',
            'TimeLockedContrast_p3a',
            'TimeLockedContrast_p3b'
        ]
        
        # Both scalar and topo use the same names since they're the same markers
        self.scalar_names = self.marker_names.copy()
        self.topo_names = self.marker_names.copy()


    def get_scalar_name(self, idx):
        """Get scalar marker name by index."""
        if idx < len(self.scalar_names):
            return self.scalar_names[idx]
        return f'Scalar_Marker_{idx}'
    
    def get_topo_name(self, idx):
        """Get topographic marker name by index.""" 
        if idx < len(self.topo_names):
            return self.topo_names[idx]
        return f'Topo_Marker_{idx}'


class StatisticalAnalyzer:
    """Statistical analysis utilities."""
    
    @staticmethod
    def paired_ttest(data1, data2):
        """Paired t-test with effect size."""
        statistic, p_value = ttest_rel(data1, data2)
        effect_size = np.mean(data1 - data2) / np.std(data1 - data2)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'test': 'paired_ttest'
        }
    
    @staticmethod
    def wilcoxon_test(data1, data2):
        """Wilcoxon signed-rank test."""
        try:
            statistic, p_value = wilcoxon(data1, data2)
            return {
                'statistic': statistic,
                'p_value': p_value,
                'test': 'wilcoxon'
            }
        except ValueError:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'test': 'wilcoxon_failed'
            }


class ScalarAnalyzer:
    """Analysis of scalar features."""
    
    def __init__(self, scalars_orig, scalars_recon, output_dir, subject_id):
        self.scalars_orig = scalars_orig
        self.scalars_recon = scalars_recon
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.mapper = MarkerNameMapper()
        self.stats_analyzer = StatisticalAnalyzer()
        
        # Create subdirectories
        self.metrics_dir = op.join(output_dir, 'scalars', 'metrics')
        self.plots_dir = op.join(output_dir, 'scalars', 'plots') 
        self.tests_dir = op.join(output_dir, 'scalars', 'statistical_tests')
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.tests_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def compute_metrics(self):
        """Compute scalar metrics."""
        print("  📊 Computing scalar metrics...")
        print(f"     🔢 Processing {len(self.scalars_orig)} scalar markers")
        
        # Basic statistics
        print(f"     📈 Original scalar stats: min={np.min(self.scalars_orig):.3f}, "
              f"max={np.max(self.scalars_orig):.3f}, mean={np.mean(self.scalars_orig):.3f}")
        print(f"     📈 Reconstructed scalar stats: min={np.min(self.scalars_recon):.3f}, "
              f"max={np.max(self.scalars_recon):.3f}, mean={np.mean(self.scalars_recon):.3f}")
        
        metrics = {}
        
        # Overall metrics
        correlation = np.corrcoef(self.scalars_orig, self.scalars_recon)[0, 1]
        cosine_sim = np.dot(self.scalars_orig, self.scalars_recon) / (
            np.linalg.norm(self.scalars_orig) * np.linalg.norm(self.scalars_recon))
        
        # Calculate basic metrics
        mse = np.mean((self.scalars_orig - self.scalars_recon) ** 2)
        mae = np.mean(np.abs(self.scalars_orig - self.scalars_recon))
        
        print("     ✅ Overall scalar metrics computed:")
        print(f"        Correlation: {correlation:.4f}")
        print(f"        Cosine similarity: {cosine_sim:.4f}")
        print(f"        MSE: {mse:.6f}")
        print(f"        MAE: {mae:.6f}")
        
        metrics['overall'] = {
            'correlation': float(correlation),
            'cosine_similarity': float(cosine_sim),
            'mse': float(mse),
            'mae': float(mae),
            'mean_relative_difference': float(np.mean(np.abs(self.scalars_orig - self.scalars_recon) / 
                                                    (np.abs(self.scalars_orig) + 1e-8)))
        }
        
        # Per-marker metrics
        n_markers = len(self.scalars_orig)
        marker_metrics = {}
        
        print(f"     🔍 Computing per-marker metrics for {n_markers} markers:")
        for i in range(n_markers):
            marker_name = self.mapper.get_scalar_name(i)
            orig_val = self.scalars_orig[i]
            recon_val = self.scalars_recon[i]
            
            abs_diff = abs(orig_val - recon_val)
            rel_diff = abs_diff / (abs(orig_val) + 1e-8)
            sq_error = (orig_val - recon_val) ** 2
            
            # Calculate normalized errors for this marker
            norm_sq_error = sq_error / (orig_val ** 2 + 1e-8)
            norm_abs_error = abs_diff / (abs(orig_val) + 1e-8)  # Same as relative difference
            
            marker_metrics[marker_name] = {
                'original_value': float(orig_val),
                'reconstructed_value': float(recon_val),
                'absolute_difference': float(abs_diff),
                'relative_difference': float(rel_diff),
                'squared_error': float(sq_error),
                'norm_sq_error': float(norm_sq_error),
                'norm_abs_error': float(norm_abs_error)
            }
            
           
        metrics['per_marker'] = marker_metrics
        print(f"     ✅ Per-marker metrics computed for all {n_markers} markers")
        
        # Save as CSV for detailed analysis
        df = pd.DataFrame(marker_metrics).T
        df.to_csv(op.join(self.metrics_dir, 'scalar_metrics.csv'))
        
        return metrics
    
    def create_plots(self, metrics):
        """Create scalar plots."""
        print("  Creating scalar plots...")
        
        # 1. Correlation scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Scalar Features Analysis - Subject {self.subject_id}', fontsize=14)
        
        # Scatter plot
        axes[0, 0].scatter(self.scalars_orig, self.scalars_recon, alpha=0.7)
        axes[0, 0].plot([self.scalars_orig.min(), self.scalars_orig.max()], 
                        [self.scalars_orig.min(), self.scalars_orig.max()], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Original Features')
        axes[0, 0].set_ylabel('Reconstructed Features')
        axes[0, 0].set_title(f'Correlation: {metrics["overall"]["correlation"]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Difference histogram
        diff = self.scalars_orig - self.scalars_recon
        axes[0, 1].hist(diff, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Difference (Original - Reconstructed)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'MAE: {metrics["overall"]["mae"]:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-marker comparison
        n_markers = min(15, len(self.scalars_orig))
        marker_names = [self.mapper.get_scalar_name(i) for i in range(n_markers)]
        x_pos = np.arange(n_markers)
        
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, self.scalars_orig[:n_markers], width, 
                       label='Original', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, self.scalars_recon[:n_markers], width, 
                       label='Reconstructed', alpha=0.7)
        axes[1, 0].set_xlabel('Markers')
        axes[1, 0].set_ylabel('Feature Values')
        axes[1, 0].set_title('Per-Marker Comparison (First 15)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Relative differences
        rel_diff = np.abs(self.scalars_orig - self.scalars_recon) / (np.abs(self.scalars_orig) + 1e-8)
        axes[1, 1].plot(rel_diff, 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Marker Index')
        axes[1, 1].set_ylabel('Relative Difference')
        axes[1, 1].set_title(f'Mean Rel. Diff: {metrics["overall"]["mean_relative_difference"]:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'scalar_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def statistical_tests(self):
        """Perform statistical tests on scalar data."""
        print("  Performing scalar statistical tests...")
        
        # For scalars, we test if the difference between original and reconstructed is significantly different from 0
        differences = self.scalars_orig - self.scalars_recon
        
        # Test if mean difference is significantly different from 0
        ttest_result = self.stats_analyzer.paired_ttest(self.scalars_orig, self.scalars_recon)
        
        # One-sample t-test on differences (testing if mean difference = 0)
        one_sample_stat, one_sample_p = stats.ttest_1samp(differences, 0)
        
        # Wilcoxon test
        wilcoxon_result = self.stats_analyzer.wilcoxon_test(self.scalars_orig, self.scalars_recon)
        
        results = {
            'paired_ttest': ttest_result,
            'one_sample_ttest_on_differences': {
                'statistic': float(one_sample_stat),
                'p_value': float(one_sample_p),
                'test': 'one_sample_ttest'
            },
            'wilcoxon': wilcoxon_result,
            'summary_stats': {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences)),
                'median_difference': float(np.median(differences))
            }
        }
        

        
        # Save results
        with open(op.join(self.tests_dir, 'scalar_statistical_tests.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


# Simplified TopographicAnalyzer for now (can be expanded later)
class TopographicAnalyzer:
    """Analysis of topographic features."""
    
    def __init__(self, topos_orig, topos_recon, output_dir, subject_id):
        self.topos_orig = topos_orig
        self.topos_recon = topos_recon
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.mapper = MarkerNameMapper()
        self.stats_analyzer = StatisticalAnalyzer()
        
        # Create subdirectories
        self.metrics_dir = op.join(output_dir, 'topography', 'metrics')
        self.plots_dir = op.join(output_dir, 'topography', 'plots')
        self.tests_dir = op.join(output_dir, 'topography', 'statistical_tests')
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.tests_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def compute_metrics(self):
        """Compute topographic metrics."""
        print("  🗺️  Computing topographic metrics...")
        print(f"     🔢 Processing {self.topos_orig.shape[0]} markers × {self.topos_orig.shape[1]} channels")
        
        # Basic statistics
        print(f"     📈 Original topo stats: min={np.min(self.topos_orig):.3f}, "
              f"max={np.max(self.topos_orig):.3f}, mean={np.mean(self.topos_orig):.3f}")
        print(f"     📈 Reconstructed topo stats: min={np.min(self.topos_recon):.3f}, "
              f"max={np.max(self.topos_recon):.3f}, mean={np.mean(self.topos_recon):.3f}")
        
        metrics = {}
        
        # Overall metrics
        overall_corr = np.corrcoef(self.topos_orig.flatten(), self.topos_recon.flatten())[0, 1]
        cosine_sim = np.dot(self.topos_orig.flatten(), self.topos_recon.flatten()) / (
            np.linalg.norm(self.topos_orig.flatten()) * np.linalg.norm(self.topos_recon.flatten()))
        
        # Calculate basic metrics for topographic data
        mse = np.mean((self.topos_orig - self.topos_recon) ** 2)
        mae = np.mean(np.abs(self.topos_orig - self.topos_recon))
        
        # Calculate normalized metrics for overall
        var_orig = np.var(self.topos_orig, ddof=1)
        nmse = mse / (var_orig + 1e-8)
        
        rmse = np.sqrt(mse)
        std_orig = np.std(self.topos_orig, ddof=1)
        nrmse = rmse / (std_orig + 1e-8)
        
        print("     ✅ Overall topographic metrics computed:")
        print(f"        Correlation: {overall_corr:.4f}")
        print(f"        Cosine similarity: {cosine_sim:.4f}")
        print(f"        MSE: {mse:.6f}")
        print(f"        MAE: {mae:.6f}")
        print(f"        NMSE: {nmse:.6f}")
        print(f"        NRMSE: {nrmse:.6f}")
        
        metrics['overall'] = {
            'correlation': float(overall_corr),
            'cosine_similarity': float(cosine_sim),
            'mse': float(mse),
            'mae': float(mae),
            'nmse': float(nmse),
            'rmse': float(rmse),
            'nrmse': float(nrmse)
        }
        
        # Per-marker topographic metrics
        n_markers, n_channels = self.topos_orig.shape
        per_marker_metrics = {}
        
        print("     🔍 Computing per-marker topographic metrics:")
        high_error_count = 0
        for i in range(n_markers):
            marker_name = self.mapper.get_topo_name(i)
            orig_topo = self.topos_orig[i, :]
            recon_topo = self.topos_recon[i, :]
            
            # Compute per-marker correlation and all metrics including normalized
            marker_corr = np.corrcoef(orig_topo, recon_topo)[0, 1] if np.std(orig_topo) > 1e-8 and np.std(recon_topo) > 1e-8 else 0
            marker_mse = np.mean((orig_topo - recon_topo) ** 2)
            marker_mae = np.mean(np.abs(orig_topo - recon_topo))
            
            # Calculate normalized metrics for this marker
            marker_var_orig = np.var(orig_topo)
            marker_nmse = marker_mse / (marker_var_orig + 1e-8)
            
            marker_rmse = np.sqrt(marker_mse)
            marker_std_orig = np.std(orig_topo)
            marker_nrmse = marker_rmse / (marker_std_orig + 1e-8)
            
            per_marker_metrics[marker_name] = {
                'correlation': float(marker_corr),
                'mse': float(marker_mse),
                'mae': float(marker_mae),
                'nmse': float(marker_nmse),
                'rmse': float(marker_rmse),
                'nrmse': float(marker_nrmse),
                'topo_original': orig_topo.tolist(),
                'topo_reconstructed': recon_topo.tolist()
            }
            
            # Log problematic markers
            if marker_mse > mse * 2 or marker_corr < 0.5:  # High error threshold
                high_error_count += 1
                print(f"        ⚠️  {i:2d}. {marker_name}: corr={marker_corr:.4f}, mse={marker_mse:.6f}")
            elif i < 3 or i >= n_markers - 3:  # First/last few
                print(f"        {i:2d}. {marker_name}: corr={marker_corr:.4f}, mse={marker_mse:.6f}")
        
        metrics['per_marker'] = per_marker_metrics
        print(f"     ✅ Per-marker metrics computed for all {n_markers} markers")
        if high_error_count > 0:
            print(f"     ⚠️  Found {high_error_count} markers with high reconstruction error!")
        
        # Save metrics
        with open(op.join(self.metrics_dir, 'topographic_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def create_plots(self, metrics):
        """Create topographic plots."""
        print("  Creating topographic plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Topographic Analysis - Subject {self.subject_id}', fontsize=14)
        
        # Scatter plot
        axes[0].scatter(self.topos_orig.flatten(), self.topos_recon.flatten(), alpha=0.1)
        axes[0].plot([self.topos_orig.min(), self.topos_orig.max()], 
                     [self.topos_orig.min(), self.topos_orig.max()], 'r--', alpha=0.8)
        axes[0].set_xlabel('Original Topography Values')
        axes[0].set_ylabel('Reconstructed Topography Values')
        axes[0].set_title(f'Correlation: {metrics["overall"]["correlation"]:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Difference histogram
        all_diffs = (self.topos_orig - self.topos_recon).flatten()
        axes[1].hist(all_diffs, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[1].set_xlabel('Difference (Original - Reconstructed)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'NMSE: {metrics["overall"]["nmse"]:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topographic_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def statistical_tests(self):
        """Perform statistical tests on topographic data."""
        print("  Performing topographic statistical tests...")
        
        # Overall tests (same as scalar tests but on flattened topographic data)
        orig_flat = self.topos_orig.flatten()
        recon_flat = self.topos_recon.flatten()
        differences = orig_flat - recon_flat
        
        # Test if mean difference is significantly different from 0
        ttest_result = self.stats_analyzer.paired_ttest(orig_flat, recon_flat)
        
        # One-sample t-test on differences
        one_sample_stat, one_sample_p = stats.ttest_1samp(differences, 0)
        
        # Wilcoxon test
        wilcoxon_result = self.stats_analyzer.wilcoxon_test(orig_flat, recon_flat)
        
        results = {
            'paired_ttest': ttest_result,
            'one_sample_ttest_on_differences': {
                'statistic': float(one_sample_stat),
                'p_value': float(one_sample_p),
                'test': 'one_sample_ttest'
            },
            'wilcoxon': wilcoxon_result,
            'summary_stats': {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences)),
                'median_difference': float(np.median(differences))
            }
        }
        
        # Per-marker tests (each marker has an array of values across sensors)
        n_markers = self.topos_orig.shape[0]
        per_marker_results = {}
        
        for i in range(n_markers):
            marker_name = self.mapper.get_topo_name(i)
            orig_marker = self.topos_orig[i, :]
            recon_marker = self.topos_recon[i, :]
            marker_diff = orig_marker - recon_marker
            
            # Calculate NMSE and NRMSE for this marker for the tests
            marker_mse = np.mean(marker_diff ** 2)
            marker_var_orig = np.var(orig_marker)
            marker_nmse = marker_mse / (marker_var_orig + 1e-8)
            
            marker_rmse = np.sqrt(marker_mse)
            marker_std_orig = np.std(orig_marker)
            marker_nrmse = marker_rmse / (marker_std_orig + 1e-8)
            
            # Paired t-test for this marker across sensors
            marker_ttest = self.stats_analyzer.paired_ttest(orig_marker, recon_marker)
            
            # One-sample t-test on differences
            marker_one_sample_stat, marker_one_sample_p = stats.ttest_1samp(marker_diff, 0)
            
            # Wilcoxon test for this marker
            marker_wilcoxon = self.stats_analyzer.wilcoxon_test(orig_marker, recon_marker)
            
            per_marker_results[marker_name] = {
                'paired_ttest': marker_ttest,
                'one_sample_ttest_on_differences': {
                    'statistic': float(marker_one_sample_stat),
                    'p_value': float(marker_one_sample_p),
                    'test': 'one_sample_ttest'
                },
                'wilcoxon': marker_wilcoxon,
                'summary_stats': {
                    'mean_difference': float(np.mean(marker_diff)),
                    'std_difference': float(np.std(marker_diff)),
                    'median_difference': float(np.median(marker_diff))
                },
                'metrics': {
                    'mse': float(marker_mse),
                    'nmse': float(marker_nmse),
                    'rmse': float(marker_rmse),
                    'nrmse': float(marker_nrmse)
                }
            }
        
        results['per_marker_tests'] = per_marker_results
        
        # Save results
        with open(op.join(self.tests_dir, 'topographic_statistical_tests.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-marker results as CSV
        per_marker_csv_data = []
        for marker_name, marker_results in per_marker_results.items():
            row = {
                'marker': marker_name,
                'paired_ttest_statistic': marker_results['paired_ttest']['statistic'],
                'paired_ttest_pvalue': marker_results['paired_ttest']['p_value'],
                'paired_ttest_effect_size': marker_results['paired_ttest']['effect_size'],
                'one_sample_ttest_statistic': marker_results['one_sample_ttest_on_differences']['statistic'],
                'one_sample_ttest_pvalue': marker_results['one_sample_ttest_on_differences']['p_value'],
                'wilcoxon_statistic': marker_results['wilcoxon']['statistic'],
                'wilcoxon_pvalue': marker_results['wilcoxon']['p_value'],
                'mean_difference': marker_results['summary_stats']['mean_difference'],
                'std_difference': marker_results['summary_stats']['std_difference'],
                'median_difference': marker_results['summary_stats']['median_difference'],
                'mse': marker_results['metrics']['mse'],
                'nmse': marker_results['metrics']['nmse'],
                'rmse': marker_results['metrics']['rmse'],
                'nrmse': marker_results['metrics']['nrmse']
            }
            per_marker_csv_data.append(row)
        
        df = pd.DataFrame(per_marker_csv_data)
        df.to_csv(op.join(self.tests_dir, 'topographic_statistical_tests.csv'), index=False)
        
        return results


def load_data(subject_dir):
    """Load data from subject directory."""
    print(f"\n📁 Loading data from: {subject_dir}")
    
    # Load feature data
    features_dir = op.join(subject_dir, 'features_variable')
    print(f"   Features directory: {features_dir}")
    
    # Check if all required files exist
    required_files = [
        'scalars_original.npy', 'scalars_reconstructed.npy',
        'topos_original.npy', 'topos_reconstructed.npy'
    ]
    
    for file_name in required_files:
        file_path = op.join(features_dir, file_name)
        if not op.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        print(f"   ✓ Found: {file_name}")
    
    # Load data with detailed logging
    scalars_orig = np.load(op.join(features_dir, 'scalars_original.npy'))
    scalars_recon = np.load(op.join(features_dir, 'scalars_reconstructed.npy'))
    topos_orig = np.load(op.join(features_dir, 'topos_original.npy'))
    topos_recon = np.load(op.join(features_dir, 'topos_reconstructed.npy'))
    
    # Detailed data shape logging
    print("\n📊 Data shapes loaded:")
    print(f"   Scalars original:     {scalars_orig.shape} (should be (n_markers,))")
    print(f"   Scalars reconstructed: {scalars_recon.shape} (should be (n_markers,))")
    print(f"   Topos original:       {topos_orig.shape} (should be (n_markers, n_channels))")
    print(f"   Topos reconstructed:  {topos_recon.shape} (should be (n_markers, n_channels))")
    
    # Validate shapes
    if scalars_orig.shape != scalars_recon.shape:
        raise ValueError(f"Scalar shape mismatch: orig {scalars_orig.shape} vs recon {scalars_recon.shape}")
    if topos_orig.shape != topos_recon.shape:
        raise ValueError(f"Topo shape mismatch: orig {topos_orig.shape} vs recon {topos_recon.shape}")
    if len(scalars_orig) != topos_orig.shape[0]:
        raise ValueError(f"Marker count mismatch: scalars {len(scalars_orig)} vs topos {topos_orig.shape[0]}")
    
    print("   ✓ All shapes are consistent!")
    print(f"   📈 Found {len(scalars_orig)} markers, {topos_orig.shape[1]} channels")
    
    # Load training results if available
    training_results = None
    training_file = op.join(subject_dir, 'train_extratrees', 'training_results.json')
    if op.exists(training_file):
        print(f"   ✓ Found training results: {training_file}")
        with open(training_file, 'r') as f:
            training_results = json.load(f)
    else:
        print(f"   ⚠️  No training results found at: {training_file}")
    
    return {
        'scalars_original': scalars_orig,
        'scalars_reconstructed': scalars_recon,
        'topos_original': topos_orig,
        'topos_reconstructed': topos_recon,
        'training_results': training_results
    }


def analyze_subject(subject_dir, output_dir, subject_id):
    """Analyze single subject."""
    print(f"\n=== Analyzing Subject: {subject_id} ===")
    
    # Load data
    data = load_data(subject_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Scalar analysis
    print("\n--- Scalar Analysis ---")
    scalar_analyzer = ScalarAnalyzer(
        data['scalars_original'], 
        data['scalars_reconstructed'],
        output_dir, 
        subject_id
    )
    scalar_metrics = scalar_analyzer.compute_metrics()
    scalar_analyzer.create_plots(scalar_metrics)
    scalar_stats = scalar_analyzer.statistical_tests()
    
    # Topographic analysis
    print("\n--- Topographic Analysis ---")
    topo_analyzer = TopographicAnalyzer(
        data['topos_original'],
        data['topos_reconstructed'],
        output_dir,
        subject_id
    )
    topo_metrics = topo_analyzer.compute_metrics()
    topo_analyzer.create_plots(topo_metrics)
    topo_stats = topo_analyzer.statistical_tests()
    
    # Combined summary
    summary = {
        'subject_id': subject_id,
        'analysis_date': datetime.now().isoformat(),
        'scalar_metrics': scalar_metrics,
        'topographic_metrics': topo_metrics,
        'scalar_statistical_tests': scalar_stats,
        'topographic_statistical_tests': topo_stats,
        'training_results': data['training_results']
    }
    
    # Save summary
    with open(op.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Analysis complete for subject {subject_id}")
    print(f"Results saved to: {output_dir}")
    
    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced marker comparison analysis')
    parser.add_argument('subject_dir', help='Subject directory containing features')
    parser.add_argument('--output', '-o', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Determine subject ID and output directory
    subject_id = op.basename(args.subject_dir.rstrip('/'))
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]  # Remove 'sub-' prefix
    
    if args.output:
        output_dir = args.output
    else:
        output_dir = op.join(args.subject_dir, 'compare_markers')
    
    # Run analysis
    summary = analyze_subject(args.subject_dir, output_dir, subject_id)
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Subject: {subject_id}")
    print(f"Scalar Correlation: {summary['scalar_metrics']['overall']['correlation']:.4f}")
    print(f"Scalar MSE: {summary['scalar_metrics']['overall']['mse']:.4f}")
    print(f"Scalar MAE: {summary['scalar_metrics']['overall']['mae']:.4f}")
    print(f"Topo Correlation: {summary['topographic_metrics']['overall']['correlation']:.4f}")
    print(f"Topo MSE: {summary['topographic_metrics']['overall']['mse']:.4f}")
    print(f"Topo MAE: {summary['topographic_metrics']['overall']['mae']:.4f}")
    print(f"Topo NMSE: {summary['topographic_metrics']['overall']['nmse']:.4f}")
    print(f"Topo NRMSE: {summary['topographic_metrics']['overall']['nrmse']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()