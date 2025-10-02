"""Individual analysis without statistical tests and with Global Field Power analysis.

This script provides comprehensive analysis of EEG data including scalar and topographic features,
plus Global Field Power analysis for different event types.

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
import mne
import warnings
from scipy.stats import chi2
warnings.filterwarnings('ignore')

# Import plot_gfp from nice-extensions
try:
    from nice_ext.viz.evokeds import plot_gfp
    HAS_NICE_EXT = True
except ImportError:
    # Try adding nice-extensions to path
    import sys
    nice_ext_path = '/Users/trinidad.borrell/Documents/Work/PhD/Proyects/nice/nice-extensions'
    if nice_ext_path not in sys.path:
        sys.path.insert(0, nice_ext_path)
    try:
        from nice_ext.viz.evokeds import plot_gfp
        HAS_NICE_EXT = True
    except ImportError:
        print("Warning: nice-extensions not found. Using basic GFP plotting.")
        HAS_NICE_EXT = False

# Define local compute_gfp function for fallback
def compute_gfp_local(x, alpha=0.05):
    """Compute GFP with confidence intervals - simplified version."""
    from scipy.stats import chi2
    
    # Estimate degrees of freedom
    try:
        df = mne.rank.estimate_rank(x * 1e12, norm=False)
    except:
        df = x.shape[0] - 1
    
    std = x.std(axis=0, ddof=1)
    
    # Compute confidence intervals using chi-squared distribution
    ci_lower = np.sqrt(df * std ** 2 / chi2.ppf(alpha / 2, df))
    ci_upper = np.sqrt(df * std ** 2 / chi2.ppf(1 - (alpha / 2), df))
    
    return std, ci_lower, ci_upper

def plot_gfp_local(epochs, conditions=None, colors=None, linestyles=None,
                   shift_time=0, labels=None, ax=None, fig_kwargs=None):
    """Local implementation of plot_gfp function."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    else:
        fig = None
        
    if conditions is None:
        conditions = list(epochs.event_id.keys())
    if colors is None:
        colors = [None for x in conditions]
    if linestyles is None:
        linestyles = ['-' for x in conditions]
    if labels is None:
        labels = [None for x in conditions]
        
    this_times = (epochs.times + shift_time) * 1e3
    
    for condition, color, ls, label in zip(conditions, colors, linestyles, labels):
        if label is None:
            label = '{}'.format(condition)
            
        if condition not in epochs.event_id:
            print(f"Warning: Condition '{condition}' not found in epochs")
            continue
            
        # Get data for this condition
        data = epochs[condition].get_data()
        
        # Average across epochs first
        data = np.mean(data, axis=0)
        
        # Compute Global Field Power and confidence intervals
        gfp, ci1, ci2 = compute_gfp_local(data)
        
        # Plot the GFP
        lines = ax.plot(this_times, gfp * 1e6, color=color, linestyle=ls, label=label)
        
        # Add confidence interval
        ax.fill_between(this_times, y1=ci1 * 1e6, y2=ci2 * 1e6,
                       color=lines[0].get_color(), alpha=0.5)
    
    # Add stimulus onset line
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.axvline(x=150, color='black', linestyle='--', alpha=0.7)
    ax.axvline(x=300, color='black', linestyle='--', alpha=0.7)
    ax.axvline(x=450, color='black', linestyle='--', alpha=0.7)
    ax.axvline(x=600, color='black', linestyle='--', alpha=0.7)
    
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r'Global Field Power ($\mu{V}$)')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig

# Set plotting style - apply seaborn first, then override with custom settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
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
            'TimeLockedTopography_p1',
            'TimeLockedTopography_p3a',
            'TimeLockedTopography_p3b',
            'TimeLockedContrast_LSGS-LDGD',
            'TimeLockedContrast_LSGD-LDGS',
            'TimeLockedContrast_LD-LS',
            'TimeLockedContrast_mmn',
            'TimeLockedContrast_p3a',
            'TimeLockedContrast_GD-GS',
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


class ScalarAnalyzer:
    """Analysis of scalar features."""
    
    def __init__(self, scalars_orig, scalars_recon, output_dir, subject_id):
        self.scalars_orig = scalars_orig
        self.scalars_recon = scalars_recon
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.mapper = MarkerNameMapper()
        
        # Create subdirectories
        self.metrics_dir = op.join(output_dir, 'scalars', 'metrics')
        self.plots_dir = op.join(output_dir, 'scalars', 'plots') 
        
        for dir_path in [self.metrics_dir, self.plots_dir]:
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


# Simplified TopographicAnalyzer for now (can be expanded later)
class TopographicAnalyzer:
    """Analysis of topographic features."""
    
    def __init__(self, topos_orig, topos_recon, output_dir, subject_id):
        self.topos_orig = topos_orig
        self.topos_recon = topos_recon
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.mapper = MarkerNameMapper()
        
        # Create subdirectories
        self.metrics_dir = op.join(output_dir, 'topography', 'metrics')
        self.plots_dir = op.join(output_dir, 'topography', 'plots')
        
        for dir_path in [self.metrics_dir, self.plots_dir]:
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


class GlobalFieldPower:
    """Global Field Power analysis for different event types."""
    
    def __init__(self, output_dir, subject_id):
        self.output_dir = output_dir
        self.subject_id = subject_id
        
        # Create subdirectories
        self.plots_dir = op.join(output_dir, 'global_field_power', 'plots')
        self.metrics_dir = op.join(output_dir, 'global_field_power', 'metrics')
        
        for dir_path in [self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_event_types(self, fif_dir):
        """Analyze Global Field Power for different event types."""
        print("  🌍 Computing Global Field Power analysis...")
        
        # Event types to analyze
        event_types = ['LSGS', 'LSGD', 'LDGS', 'LDGD']
        
        # Load both original and reconstructed EEG data from .fif files
        epochs_orig, epochs_recon = self._load_epochs_data(fif_dir)
        
        if epochs_orig is None:
            print("     ❌ No original EEG data found. Cannot perform Global Field Power analysis.")
            return
        
        if epochs_recon is None:
            print("     ⚠️  No reconstructed EEG data found. Will only analyze original data.")
            epochs_recon = None
        
        print("     📊 Creating Global Field Power plots for event types:")
        for event_type in event_types:
            print(f"        - {event_type}")
        
        # Create time-series plot for each event type
        self._create_time_series_plots(event_types, epochs_orig, epochs_recon)
        
        print("     ✅ Global Field Power analysis completed")
    
    def _load_epochs_data(self, data_dir):
        """Load EEG epochs data from .fif files."""
        print("     📁 Loading EEG data from .fif files...")
        
        # Look for original and reconstructed .fif files
        orig_files = []
        recon_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.fif') and ('epo' in file or 'epochs' in file):
                    file_path = op.join(root, file)
                    if 'recon' in file.lower() or 'reconstructed' in file.lower():
                        recon_files.append(file_path)
                    else:
                        orig_files.append(file_path)
        
        print(f"     📊 Found {len(orig_files)} original .fif files")
        print(f"     📊 Found {len(recon_files)} reconstructed .fif files")
        
        # Load original epochs
        epochs_orig = None
        if orig_files:
            epochs_file = orig_files[0]
            print(f"     📂 Loading original epochs from: {op.basename(epochs_file)}")
            try:
                epochs_orig = mne.read_epochs(epochs_file, preload=True)
                print(f"     ✅ Loaded original epochs: {len(epochs_orig)} epochs, {len(epochs_orig.ch_names)} channels")
                print(f"     📊 Event types: {list(epochs_orig.event_id.keys())}")
            except Exception as e:
                print(f"     ❌ Error loading original epochs: {e}")
        
        # Load reconstructed epochs
        epochs_recon = None
        if recon_files:
            epochs_file = recon_files[0]
            print(f"     📂 Loading reconstructed epochs from: {op.basename(epochs_file)}")
            try:
                epochs_recon = mne.read_epochs(epochs_file, preload=True)
                print(f"     ✅ Loaded reconstructed epochs: {len(epochs_recon)} epochs, {len(epochs_recon.ch_names)} channels")
                print(f"     📊 Event types: {list(epochs_recon.event_id.keys())}")
            except Exception as e:
                print(f"     ❌ Error loading reconstructed epochs: {e}")
        
        return epochs_orig, epochs_recon
    
    def _compute_global_field_power(self, epochs, event_type):
        """Compute Global Field Power for a specific event type."""
        if event_type not in epochs.event_id:
            print(f"     ⚠️  Event type {event_type} not found in epochs")
            return None, None
        
        # Get epochs for this event type
        event_epochs = epochs[event_type]
        
        if len(event_epochs) == 0:
            print(f"     ⚠️  No epochs found for event type {event_type}")
            return None, None
        
        # Compute Global Field Power (standard deviation across channels)
        gfp = np.std(event_epochs.get_data(), axis=1)  # Shape: (n_epochs, n_times)
        
        # Compute mean and std across epochs
        gfp_mean = np.mean(gfp, axis=0)
        gfp_std = np.std(gfp, axis=0)
        
        return gfp_mean, gfp_std
    
    def _create_time_series_plots(self, event_types, epochs_orig, epochs_recon=None):
        """Create time-series plots for Global Field Power using plot_gfp function."""
        print("     📈 Creating time-series plots...")
        
        if not HAS_NICE_EXT:
            print("     ⚠️  Using local plot_gfp implementation")
        
        # Create individual plots for each event type
        for event_type in event_types:
            if event_type not in epochs_orig.event_id:
                print(f"     ⚠️  Event type '{event_type}' not found in original epochs")
                continue
                
            # Create figure for this event type - single subplot with both curves
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            fig.suptitle(f'Global Field Power - {event_type} - Subject {self.subject_id}', fontsize=14)
            
            # Plot original data
            if HAS_NICE_EXT:
                plot_gfp(epochs_orig, conditions=[event_type], colors=['blue'], 
                        labels=[f'Original'], ax=ax, 
                        fig_kwargs={}, sns_kwargs={})
            else:
                plot_gfp_local(epochs_orig, conditions=[event_type], colors=['blue'],
                              labels=[f'Original'], ax=ax, fig_kwargs={})
            
            # Plot reconstructed data on the same axis if available
            if epochs_recon is not None and event_type in epochs_recon.event_id:
                if HAS_NICE_EXT:
                    plot_gfp(epochs_recon, conditions=[event_type], colors=['red'],
                            labels=[f'Reconstructed'], ax=ax,
                            fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_recon, conditions=[event_type], colors=['red'],
                                  labels=[f'Reconstructed'], ax=ax, fig_kwargs={})
            
            ax.set_title(f'{event_type} - Original vs Reconstructed')
            
            plt.tight_layout()
            plt.savefig(op.join(self.plots_dir, f'global_field_power_{event_type}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined plot showing all event types
        self._create_combined_plot(event_types, epochs_orig, epochs_recon)
    
    def _create_combined_plot(self, event_types, epochs_orig, epochs_recon=None):
        """Create a combined plot showing all event types using plot_gfp function."""
        print("     📈 Creating combined plot...")
        
        if not HAS_NICE_EXT:
            print("     ⚠️  Using local plot_gfp implementation")
        
        # Filter event types that exist in the data
        available_events_orig = [et for et in event_types if et in epochs_orig.event_id]
        
        if not available_events_orig:
            print("     ⚠️  No valid event types found in original epochs")
            return
        
        # Create subplots for original and reconstructed data
        if epochs_recon is not None:
            available_events_recon = [et for et in event_types if et in epochs_recon.event_id]
            if available_events_recon:
                fig, (ax_orig, ax_recon) = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f'Global Field Power - All Event Types - Subject {self.subject_id}', fontsize=16)
                
                # Define colors for each event type
                colors = ['blue', 'red', 'green', 'orange']
                colors_orig = [colors[i % len(colors)] for i in range(len(available_events_orig))]
                colors_recon = [colors[i % len(colors)] for i in range(len(available_events_recon))]
                
                # Plot original data with all event types
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                            labels=available_events_orig, ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                                  labels=available_events_orig, ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data')
                
                # Plot reconstructed data with all event types
                if HAS_NICE_EXT:
                    plot_gfp(epochs_recon, conditions=available_events_recon, colors=colors_recon,
                            labels=available_events_recon, ax=ax_recon, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_recon, conditions=available_events_recon, colors=colors_recon,
                                  labels=available_events_recon, ax=ax_recon, fig_kwargs={})
                ax_recon.set_title('Reconstructed Data')
                
                # Synchronize y-axis limits for comparison
                y_min = min(ax_orig.get_ylim()[0], ax_recon.get_ylim()[0])
                y_max = max(ax_orig.get_ylim()[1], ax_recon.get_ylim()[1])
                ax_orig.set_ylim(y_min, y_max)
                ax_recon.set_ylim(y_min, y_max)
            else:
                # Only original data available
                fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
                fig.suptitle(f'Global Field Power - All Event Types - Subject {self.subject_id}', fontsize=16)
                
                colors = ['blue', 'red', 'green', 'orange']
                colors_orig = [colors[i % len(colors)] for i in range(len(available_events_orig))]
                
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                            labels=available_events_orig, ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                                  labels=available_events_orig, ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data')
        else:
            # Only original data available
            fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'Global Field Power - All Event Types - Subject {self.subject_id}', fontsize=16)
            
            colors = ['blue', 'red', 'green', 'orange']
            colors_orig = [colors[i % len(colors)] for i in range(len(available_events_orig))]
            
            if HAS_NICE_EXT:
                plot_gfp(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                        labels=available_events_orig, ax=ax_orig, fig_kwargs={}, sns_kwargs={})
            else:
                plot_gfp_local(epochs_orig, conditions=available_events_orig, colors=colors_orig,
                              labels=available_events_orig, ax=ax_orig, fig_kwargs={})
            ax_orig.set_title('Original Data')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_combined.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional comparison plots
#        self._create_local_comparison_plot(event_types, epochs_orig, epochs_recon)
#        self._create_global_comparison_plot(event_types, epochs_orig, epochs_recon)
        self._create_local_effect_plot(epochs_orig, epochs_recon)
        self._create_global_effect_plot(epochs_orig, epochs_recon)
        self._create_difference_plot(epochs_orig, epochs_recon)
    
    def _create_local_effect_plot(self, epochs_orig, epochs_recon=None):
        """Create plot showing Local Standard vs Local Deviant effect for original and reconstructed data."""
        print("     📈 Creating Local Effect plot...")
        
        if not HAS_NICE_EXT:
            print("     ⚠️  Using local plot_gfp implementation")
        
        # Define the event groupings for local effect
        local_standard_events = ['LSGS', 'LSGD']  # Local Standard
        local_deviant_events = ['LDGS', 'LDGD']   # Local Deviant
        
        # Check which events are available in original data
        available_local_std_orig = [et for et in local_standard_events if et in epochs_orig.event_id]
        available_local_dev_orig = [et for et in local_deviant_events if et in epochs_orig.event_id]
        
        if not available_local_std_orig or not available_local_dev_orig:
            print(f"     ⚠️  Missing events for local effect analysis. Available: {list(epochs_orig.event_id.keys())}")
            return
        
        # Create subplots for original and reconstructed data
        if epochs_recon is not None:
            available_local_std_recon = [et for et in local_standard_events if et in epochs_recon.event_id]
            available_local_dev_recon = [et for et in local_deviant_events if et in epochs_recon.event_id]
            
            if available_local_std_recon and available_local_dev_recon:
                fig, (ax_orig, ax_recon) = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f'Local Effect Analysis - Subject {self.subject_id}', fontsize=16)
                
                # Plot original data - Local Standard vs Local Deviant
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                            labels=['Local Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                            labels=['Local Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                                  labels=['Local Standard'], ax=ax_orig, fig_kwargs={})
                    plot_gfp_local(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                                  labels=['Local Deviant'], ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data - Local Effect')
                
                # Plot reconstructed data - Local Standard vs Local Deviant
                if HAS_NICE_EXT:
                    plot_gfp(epochs_recon, conditions=available_local_std_recon, colors=['blue'],
                            labels=['Local Standard'], ax=ax_recon, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_recon, conditions=available_local_dev_recon, colors=['red'],
                            labels=['Local Deviant'], ax=ax_recon, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_recon, conditions=available_local_std_recon, colors=['blue'],
                                  labels=['Local Standard'], ax=ax_recon, fig_kwargs={})
                    plot_gfp_local(epochs_recon, conditions=available_local_dev_recon, colors=['red'],
                                  labels=['Local Deviant'], ax=ax_recon, fig_kwargs={})
                ax_recon.set_title('Reconstructed Data - Local Effect')
                
                # Synchronize y-axis limits for comparison
                y_min = min(ax_orig.get_ylim()[0], ax_recon.get_ylim()[0])
                y_max = max(ax_orig.get_ylim()[1], ax_recon.get_ylim()[1])
                ax_orig.set_ylim(y_min, y_max)
                ax_recon.set_ylim(y_min, y_max)
            else:
                # Only original data available
                fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
                fig.suptitle(f'Local Effect Analysis - Subject {self.subject_id}', fontsize=16)
                
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                            labels=['Local Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                            labels=['Local Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                                  labels=['Local Standard'], ax=ax_orig, fig_kwargs={})
                    plot_gfp_local(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                                  labels=['Local Deviant'], ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data - Local Effect')
        else:
            # Only original data available
            fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'Local Effect Analysis - Subject {self.subject_id}', fontsize=16)
            
            if HAS_NICE_EXT:
                plot_gfp(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                        labels=['Local Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                plot_gfp(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                        labels=['Local Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
            else:
                plot_gfp_local(epochs_orig, conditions=available_local_std_orig, colors=['blue'],
                              labels=['Local Standard'], ax=ax_orig, fig_kwargs={})
                plot_gfp_local(epochs_orig, conditions=available_local_dev_orig, colors=['red'],
                              labels=['Local Deviant'], ax=ax_orig, fig_kwargs={})
            ax_orig.set_title('Original Data - Local Effect')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_local_effect.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Local Effect plot saved")
    
    def _create_global_effect_plot(self, epochs_orig, epochs_recon=None):
        """Create plot showing Global Standard vs Global Deviant effect for original and reconstructed data."""
        print("     📈 Creating Global Effect plot...")
        
        if not HAS_NICE_EXT:
            print("     ⚠️  Using local plot_gfp implementation")
        
        # Define the event groupings for global effect
        global_standard_events = ['LDGS', 'LSGS']  # Global Standard
        global_deviant_events = ['LSGD', 'LDGD']   # Global Deviant
        
        # Check which events are available in original data
        available_global_std_orig = [et for et in global_standard_events if et in epochs_orig.event_id]
        available_global_dev_orig = [et for et in global_deviant_events if et in epochs_orig.event_id]
        
        if not available_global_std_orig or not available_global_dev_orig:
            print(f"     ⚠️  Missing events for global effect analysis. Available: {list(epochs_orig.event_id.keys())}")
            return
        
        # Create subplots for original and reconstructed data
        if epochs_recon is not None:
            available_global_std_recon = [et for et in global_standard_events if et in epochs_recon.event_id]
            available_global_dev_recon = [et for et in global_deviant_events if et in epochs_recon.event_id]
            
            if available_global_std_recon and available_global_dev_recon:
                fig, (ax_orig, ax_recon) = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f'Global Effect Analysis - Subject {self.subject_id}', fontsize=16)
                
                # Plot original data - Global Standard vs Global Deviant
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                            labels=['Global Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                            labels=['Global Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                                  labels=['Global Standard'], ax=ax_orig, fig_kwargs={})
                    plot_gfp_local(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                                  labels=['Global Deviant'], ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data - Global Effect')
                
                # Plot reconstructed data - Global Standard vs Global Deviant
                if HAS_NICE_EXT:
                    plot_gfp(epochs_recon, conditions=available_global_std_recon, colors=['green'],
                            labels=['Global Standard'], ax=ax_recon, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_recon, conditions=available_global_dev_recon, colors=['purple'],
                            labels=['Global Deviant'], ax=ax_recon, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_recon, conditions=available_global_std_recon, colors=['green'],
                                  labels=['Global Standard'], ax=ax_recon, fig_kwargs={})
                    plot_gfp_local(epochs_recon, conditions=available_global_dev_recon, colors=['purple'],
                                  labels=['Global Deviant'], ax=ax_recon, fig_kwargs={})
                ax_recon.set_title('Reconstructed Data - Global Effect')
                
                # Synchronize y-axis limits for comparison
                y_min = min(ax_orig.get_ylim()[0], ax_recon.get_ylim()[0])
                y_max = max(ax_orig.get_ylim()[1], ax_recon.get_ylim()[1])
                ax_orig.set_ylim(y_min, y_max)
                ax_recon.set_ylim(y_min, y_max)
            else:
                # Only original data available
                fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
                fig.suptitle(f'Global Effect Analysis - Subject {self.subject_id}', fontsize=16)
                
                if HAS_NICE_EXT:
                    plot_gfp(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                            labels=['Global Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                    plot_gfp(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                            labels=['Global Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                else:
                    plot_gfp_local(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                                  labels=['Global Standard'], ax=ax_orig, fig_kwargs={})
                    plot_gfp_local(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                                  labels=['Global Deviant'], ax=ax_orig, fig_kwargs={})
                ax_orig.set_title('Original Data - Global Effect')
        else:
            # Only original data available
            fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'Global Effect Analysis - Subject {self.subject_id}', fontsize=16)
            
            if HAS_NICE_EXT:
                plot_gfp(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                        labels=['Global Standard'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
                plot_gfp(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                        labels=['Global Deviant'], ax=ax_orig, fig_kwargs={}, sns_kwargs={})
            else:
                plot_gfp_local(epochs_orig, conditions=available_global_std_orig, colors=['green'],
                              labels=['Global Standard'], ax=ax_orig, fig_kwargs={})
                plot_gfp_local(epochs_orig, conditions=available_global_dev_orig, colors=['purple'],
                              labels=['Global Deviant'], ax=ax_orig, fig_kwargs={})
            ax_orig.set_title('Original Data - Global Effect')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_global_effect.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Global Effect plot saved")

    def _create_difference_plot(self, epochs_orig, epochs_recon=None):
        """Create plot showing the absolute difference between original and reconstructed data across time."""
        print("     📈 Creating Difference plot...")
        
        if epochs_recon is None:
            print("     ⚠️  No reconstructed data available for difference analysis")
            return
        
        # Individual event types
        individual_events = ['LDGD', 'LDGS', 'LSGD', 'LSGS']
        
        # Event groupings for effects
        local_standard_events = ['LSGS', 'LSGD']  # Local Standard
        local_deviant_events = ['LDGD', 'LDGS']   # Local Deviant
        global_standard_events = ['LDGS', 'LSGS']  # Global Standard
        global_deviant_events = ['LSGD', 'LDGD']   # Global Deviant
        
        # Check which events are available
        available_individual = [et for et in individual_events if et in epochs_orig.event_id and et in epochs_recon.event_id]
        
        if not available_individual:
            print(f"     ⚠️  No matching events between original and reconstructed data")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Absolute Difference: Original vs Reconstructed - Subject {self.subject_id}', fontsize=16)
        
        # Exclude the last time point to match the GFP computation
        times = epochs_orig.times[:-5] * 1000  # Convert to milliseconds
        colors = ['red', 'blue', 'orange', 'green']
        
        # Subplot 1: Individual events
        ax1.set_title('Individual Events')
        
        for i, event_type in enumerate(available_individual):
            # Compute GFP for original and reconstructed
            orig_gfp = self._compute_gfp(epochs_orig[event_type])
            recon_gfp = self._compute_gfp(epochs_recon[event_type])
            
            # Compute absolute difference
            diff_gfp = np.abs(orig_gfp - recon_gfp)
            
            ax1.plot(times, diff_gfp, label=event_type, color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('|GFP Original - GFP Reconstructed|')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax1.axvline(x=150, color='black', linestyle='--', alpha=0.7)
        ax1.axvline(x=300, color='black', linestyle='--', alpha=0.7)
        ax1.axvline(x=450, color='black', linestyle='--', alpha=0.7)
        ax1.axvline(x=600, color='black', linestyle='--', alpha=0.7)
        # Subplot 2: Effect comparisons
        ax2.set_title('Effect Comparisons')
        
        effect_data = []
        effect_labels = []
        effect_colors = ['purple', 'cyan', 'orange', 'pink']
        
        # Local Standard effect
        local_std_available = [et for et in local_standard_events if et in available_individual]
        if len(local_std_available) >= 1:
            # Average across available local standard events
            local_std_orig_gfp = np.mean([self._compute_gfp(epochs_orig[et]) for et in local_std_available], axis=0)
            local_std_recon_gfp = np.mean([self._compute_gfp(epochs_recon[et]) for et in local_std_available], axis=0)
            local_std_diff = np.abs(local_std_orig_gfp - local_std_recon_gfp)
            effect_data.append(local_std_diff)
            effect_labels.append('Local Standard')
        
        # Local Deviant effect
        local_dev_available = [et for et in local_deviant_events if et in available_individual]
        if len(local_dev_available) >= 1:
            local_dev_orig_gfp = np.mean([self._compute_gfp(epochs_orig[et]) for et in local_dev_available], axis=0)
            local_dev_recon_gfp = np.mean([self._compute_gfp(epochs_recon[et]) for et in local_dev_available], axis=0)
            local_dev_diff = np.abs(local_dev_orig_gfp - local_dev_recon_gfp)
            effect_data.append(local_dev_diff)
            effect_labels.append('Local Deviant')
        
        # Global Standard effect
        global_std_available = [et for et in global_standard_events if et in available_individual]
        if len(global_std_available) >= 1:
            global_std_orig_gfp = np.mean([self._compute_gfp(epochs_orig[et]) for et in global_std_available], axis=0)
            global_std_recon_gfp = np.mean([self._compute_gfp(epochs_recon[et]) for et in global_std_available], axis=0)
            global_std_diff = np.abs(global_std_orig_gfp - global_std_recon_gfp)
            effect_data.append(global_std_diff)
            effect_labels.append('Global Standard')
        
        # Global Deviant effect
        global_dev_available = [et for et in global_deviant_events if et in available_individual]
        if len(global_dev_available) >= 1:
            global_dev_orig_gfp = np.mean([self._compute_gfp(epochs_orig[et]) for et in global_dev_available], axis=0)
            global_dev_recon_gfp = np.mean([self._compute_gfp(epochs_recon[et]) for et in global_dev_available], axis=0)
            global_dev_diff = np.abs(global_dev_orig_gfp - global_dev_recon_gfp)
            effect_data.append(global_dev_diff)
            effect_labels.append('Global Deviant')
        
        # Plot effect comparisons
        for i, (diff_data, label) in enumerate(zip(effect_data, effect_labels)):
            ax2.plot(times, diff_data, label=label, color=effect_colors[i % len(effect_colors)], linewidth=2)
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('|GFP Original - GFP Reconstructed|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax2.axvline(x=150, color='black', linestyle='--', alpha=0.7)
        ax2.axvline(x=300, color='black', linestyle='--', alpha=0.7)
        ax2.axvline(x=450, color='black', linestyle='--', alpha=0.7)
        ax2.axvline(x=600, color='black', linestyle='--', alpha=0.7)
        
        # Synchronize y-axis limits for comparison
        y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_difference.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Difference plot saved")

    def _compute_gfp(self, epochs):
        """Compute Global Field Power for given epochs."""
        # Get the data (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        # Exclude the last time point
        data = data[:, :, :-5]
        # Compute GFP: standard deviation across channels for each time point, averaged across epochs
        gfp = np.std(data, axis=1)  # std across channels for each epoch and time point
        gfp_avg = np.mean(gfp, axis=0)  # average across epochs
        return gfp_avg

  
    def _create_time_series_plots_basic(self, event_types, epochs_orig, epochs_recon=None):
        """Basic time-series plotting fallback when nice-extensions is not available."""
        # Create a figure with subplots for each event type
        if epochs_recon is not None:
            fig, axes = plt.subplots(2, 4, figsize=(20, 12))
            fig.suptitle(f'Global Field Power Analysis - Subject {self.subject_id}', fontsize=16)
            orig_axes = axes[:, :2].flatten()
            recon_axes = axes[:, 2:].flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Global Field Power Analysis - Subject {self.subject_id}', fontsize=16)
            orig_axes = axes.flatten()
            recon_axes = None
        
        time_points = epochs_orig.times
        
        for i, event_type in enumerate(event_types):
            if i >= len(orig_axes):
                break
                
            ax_orig = orig_axes[i]
            gfp_orig_mean, gfp_orig_std = self._compute_global_field_power(epochs_orig, event_type)
            
            if gfp_orig_mean is not None:
                ax_orig.plot(time_points, gfp_orig_mean, linewidth=2, 
                           label=f'{event_type}', color='blue', alpha=0.8)
                ax_orig.fill_between(time_points, 
                                   gfp_orig_mean - gfp_orig_std,
                                   gfp_orig_mean + gfp_orig_std,
                                   alpha=0.2, color='blue')
                ax_orig.set_xlabel('Time (s)')
                ax_orig.set_ylabel('Global Field Power (μV)')
                ax_orig.set_title(f'Original - {event_type}')
                ax_orig.grid(True, alpha=0.3)
                ax_orig.legend()
                ax_orig.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            if epochs_recon is not None and recon_axes is not None and i < len(recon_axes):
                ax_recon = recon_axes[i]
                gfp_recon_mean, gfp_recon_std = self._compute_global_field_power(epochs_recon, event_type)
                
                if gfp_recon_mean is not None:
                    ax_recon.plot(time_points, gfp_recon_mean, linewidth=2, 
                               label=f'{event_type}', color='red', alpha=0.8)
                    ax_recon.fill_between(time_points, 
                                       gfp_recon_mean - gfp_recon_std,
                                       gfp_recon_mean + gfp_recon_std,
                                       alpha=0.2, color='red')
                    ax_recon.set_xlabel('Time (s)')
                    ax_recon.set_ylabel('Global Field Power (μV)')
                    ax_recon.set_title(f'Reconstructed - {event_type}')
                    ax_recon.grid(True, alpha=0.3)
                    ax_recon.legend()
                    ax_recon.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_time_series.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_combined_plot_basic(self, event_types, epochs_orig, epochs_recon=None):
        """Basic combined plotting fallback when nice-extensions is not available."""
        time_points = epochs_orig.times
        
        if epochs_recon is not None:
            fig, (ax_orig, ax_recon) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'Global Field Power - All Event Types - Subject {self.subject_id}', fontsize=16)
        else:
            fig, ax_orig = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'Global Field Power - All Event Types - Subject {self.subject_id}', fontsize=16)
            ax_recon = None
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, event_type in enumerate(event_types):
            gfp_orig_mean, gfp_orig_std = self._compute_global_field_power(epochs_orig, event_type)
            
            if gfp_orig_mean is not None:
                ax_orig.plot(time_points, gfp_orig_mean, linewidth=2, 
                           color=colors[i % len(colors)], label=f'{event_type}', alpha=0.8)
            
            if epochs_recon is not None and ax_recon is not None:
                gfp_recon_mean, gfp_recon_std = self._compute_global_field_power(epochs_recon, event_type)
                if gfp_recon_mean is not None:
                    ax_recon.plot(time_points, gfp_recon_mean, linewidth=2, 
                               color=colors[i % len(colors)], label=f'{event_type}', alpha=0.8)
        
        ax_orig.set_xlabel('Time (s)')
        ax_orig.set_ylabel('Global Field Power (μV)')
        ax_orig.set_title('Original Data')
        ax_orig.grid(True, alpha=0.3)
        ax_orig.legend()
        ax_orig.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        if epochs_recon is not None and ax_recon is not None:
            ax_recon.set_xlabel('Time (s)')
            ax_recon.set_ylabel('Global Field Power (μV)')
            ax_recon.set_title('Reconstructed Data')
            ax_recon.grid(True, alpha=0.3)
            ax_recon.legend()
            ax_recon.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_combined.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


class TimeSeriesErrorAnalyzer:
    """Analyzer for computing MSE and MAE per trial and sensor from time series data."""
    
    def __init__(self, output_dir, subject_id):
        self.output_dir = output_dir
        self.subject_id = subject_id
        
        # Create subdirectories
        self.plots_dir = op.join(output_dir, 'timeseries_error', 'plots')
        self.metrics_dir = op.join(output_dir, 'timeseries_error', 'metrics')
        
        for dir_path in [self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze(self, fif_dir):
        """Compute MSE and MAE per trial and sensor."""
        print("  📊 Computing time series MSE and MAE analysis...")
        
        # Load original and reconstructed epochs
        epochs_orig, epochs_recon = self._load_epochs_data(fif_dir)
        
        if epochs_orig is None or epochs_recon is None:
            print("     ❌ Could not load both original and reconstructed data. Skipping time series error analysis.")
            return None
        
        # Get data arrays: shape (n_trials, n_sensors, n_timepoints)
        data_orig = epochs_orig.get_data()
        data_recon = epochs_recon.get_data()
        
        n_trials, n_sensors, n_timepoints = data_orig.shape
        
        print(f"     📈 Data shape: {n_trials} trials × {n_sensors} sensors × {n_timepoints} timepoints")
        
        # Calculate MSE and MAE per trial and sensor
        # MSE: Mean across timepoints for each (trial, sensor) pair
        mse_per_trial_sensor = np.mean((data_orig - data_recon) ** 2, axis=2)  # Shape: (n_trials, n_sensors)
        mae_per_trial_sensor = np.mean(np.abs(data_orig - data_recon), axis=2)  # Shape: (n_trials, n_sensors)
        
        print(f"     ✅ Computed MSE and MAE matrices: {mse_per_trial_sensor.shape}")
        print(f"        MSE range: [{np.min(mse_per_trial_sensor):.6f}, {np.max(mse_per_trial_sensor):.6f}]")
        print(f"        MAE range: [{np.min(mae_per_trial_sensor):.6f}, {np.max(mae_per_trial_sensor):.6f}]")
        
        # Calculate overall statistics
        overall_mse = np.mean(mse_per_trial_sensor)
        overall_mae = np.mean(mae_per_trial_sensor)
        
        # Calculate per-sensor averages (across trials)
        mse_per_sensor = np.mean(mse_per_trial_sensor, axis=0)  # Shape: (n_sensors,)
        mae_per_sensor = np.mean(mae_per_trial_sensor, axis=0)  # Shape: (n_sensors,)
        
        # Calculate per-trial averages (across sensors)
        mse_per_trial = np.mean(mse_per_trial_sensor, axis=1)  # Shape: (n_trials,)
        mae_per_trial = np.mean(mae_per_trial_sensor, axis=1)  # Shape: (n_trials,)
        
        # Create heatmaps
        self._create_heatmaps(mse_per_trial_sensor, mae_per_trial_sensor, n_trials, n_sensors)
        
        # Prepare results dictionary
        results = {
            'overall': {
                'mse': float(overall_mse),
                'mae': float(overall_mae),
                'mse_std': float(np.std(mse_per_trial_sensor)),
                'mae_std': float(np.std(mae_per_trial_sensor))
            },
            'per_sensor': {
                'mse': mse_per_sensor.tolist(),
                'mae': mae_per_sensor.tolist()
            },
            'per_trial': {
                'mse': mse_per_trial.tolist(),
                'mae': mae_per_trial.tolist()
            },
            'dimensions': {
                'n_trials': int(n_trials),
                'n_sensors': int(n_sensors),
                'n_timepoints': int(n_timepoints)
            }
        }
        
        # Save results as JSON
        results_file = op.join(self.metrics_dir, 'timeseries_error_metrics.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"     💾 Saved metrics to: {results_file}")
        print(f"     📊 Overall MSE: {overall_mse:.6f} ± {np.std(mse_per_trial_sensor):.6f}")
        print(f"     📊 Overall MAE: {overall_mae:.6f} ± {np.std(mae_per_trial_sensor):.6f}")
        
        return results
    
    def _load_epochs_data(self, data_dir):
        """Load EEG epochs data from .fif files."""
        print("     📁 Loading EEG epochs for time series error analysis...")
        
        # Look for original and reconstructed .fif files
        orig_files = []
        recon_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.fif') and ('epo' in file or 'epochs' in file):
                    file_path = op.join(root, file)
                    if 'recon' in file.lower() or 'reconstructed' in file.lower():
                        recon_files.append(file_path)
                    elif 'original' in file.lower() or (not 'recon' in file.lower()):
                        orig_files.append(file_path)
        
        print(f"     📊 Found {len(orig_files)} original .fif files")
        print(f"     📊 Found {len(recon_files)} reconstructed .fif files")
        
        # Load original epochs
        epochs_orig = None
        if orig_files:
            try:
                epochs_orig = mne.read_epochs(orig_files[0], preload=True, verbose=False)
                print(f"     ✅ Loaded original epochs: {len(epochs_orig)} trials, {len(epochs_orig.ch_names)} sensors")
            except Exception as e:
                print(f"     ❌ Error loading original epochs: {e}")
                return None, None
        
        # Load reconstructed epochs
        epochs_recon = None
        if recon_files:
            try:
                epochs_recon = mne.read_epochs(recon_files[0], preload=True, verbose=False)
                print(f"     ✅ Loaded reconstructed epochs: {len(epochs_recon)} trials, {len(epochs_recon.ch_names)} sensors")
            except Exception as e:
                print(f"     ❌ Error loading reconstructed epochs: {e}")
                return epochs_orig, None
        
        # Validate that both have the same shape
        if epochs_orig is not None and epochs_recon is not None:
            if epochs_orig.get_data().shape != epochs_recon.get_data().shape:
                print(f"     ⚠️  Shape mismatch: orig {epochs_orig.get_data().shape} vs recon {epochs_recon.get_data().shape}")
                return None, None
        
        return epochs_orig, epochs_recon
    
    def _create_heatmaps(self, mse_matrix, mae_matrix, n_trials, n_sensors):
        """Create heatmaps for MSE and MAE."""
        print("     🎨 Creating MSE and MAE heatmaps...")
        
        # Create figure with 2 subplots side by side
        fig, ax = plt.subplots(1, 1, figsize=(20, max(8, n_trials * 0.05)))
        
        # MSE Heatmap
        im1 = ax.imshow(mse_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
        ax.set_title(f'MSE per Trial and Sensor\n{self.subject_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sensors', fontsize=12)
        ax.set_ylabel('Trials', fontsize=12)
        plt.colorbar(im1, ax=ax, label='MSE')

        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'heatmap_mse.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # MAE Heatmap
        im2 = ax.imshow(mae_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
        ax.set_title(f'MAE per Trial and Sensor\n{self.subject_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sensors', fontsize=12)
        ax.set_ylabel('Trials', fontsize=12)
        plt.colorbar(im2, ax=ax, label='MAE')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'heatmap_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Saved heatmaps to: {self.plots_dir}/heatmap_mse.png and {self.plots_dir}/heatmap_mae.png")


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


def analyze_subject(subject_dir, fif_dir, output_dir, subject_id):
    """Analyze single subject."""
    print(f"\n=== Analyzing Subject: {subject_id} ===")
    
    # Load data from subject directory (scalars/topos)
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
    
    # Global Field Power analysis
    print("\n--- Global Field Power Analysis ---")
    gfp_analyzer = GlobalFieldPower(output_dir, subject_id)
    gfp_analyzer.analyze_event_types(fif_dir)
    
    # Time Series Error Analysis (MSE/MAE per trial and sensor)
    print("\n--- Time Series Error Analysis ---")
    ts_error_analyzer = TimeSeriesErrorAnalyzer(output_dir, subject_id)
    ts_error_metrics = ts_error_analyzer.analyze(fif_dir)
    
    # Combined summary
    summary = {
        'subject_id': subject_id,
        'analysis_date': datetime.now().isoformat(),
        'scalar_metrics': scalar_metrics,
        'topographic_metrics': topo_metrics,
        'timeseries_error_metrics': ts_error_metrics,
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
    parser = argparse.ArgumentParser(description='Individual analysis without statistical tests')
    parser.add_argument('subject_dir', help='Subject directory containing scalars/topos features')
    parser.add_argument('fif_dir', help='Directory containing .fif files for Global Field Power analysis')
    parser.add_argument('--output', '-o', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Determine subject ID and output directory
    subject_id = op.basename(args.subject_dir.rstrip('/'))
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]  # Remove 'sub-' prefix
    
    if args.output:
        output_dir = args.output
    else:
        output_dir = op.join(args.subject_dir, 'individual_analysis')
    
    # Run analysis
    summary = analyze_subject(args.subject_dir, args.fif_dir, output_dir, subject_id)
    
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
