"""Global analysis across multiple subjects.

This script generates comprehensive global analysis plots and statistics
comparing original and reconstructed EEG data across multiple subjects.

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
import warnings
warnings.filterwarnings('ignore')

# Import statistical analysis with robust approach
HAS_STATISTICAL_ANALYSIS = False
StatisticalAnalyzer = None

def _import_statistical_analyzer():
    """Import StatisticalAnalyzer with multiple fallback approaches."""
    global HAS_STATISTICAL_ANALYSIS, StatisticalAnalyzer
    
    try:
        # First try relative import
        from .statistical_analysis import StatisticalAnalyzer
        HAS_STATISTICAL_ANALYSIS = True
        print("✅ StatisticalAnalyzer imported successfully (relative import)")
        return
    except ImportError:
        pass
    
    try:
        # Try absolute import from same directory
        import sys
        import os.path as op
        analysis_dir = op.dirname(op.abspath(__file__))
        if analysis_dir not in sys.path:
            sys.path.insert(0, analysis_dir)
        
        from statistical_analysis import StatisticalAnalyzer
        HAS_STATISTICAL_ANALYSIS = True
        print("✅ StatisticalAnalyzer imported successfully (absolute import)")
        return
    except ImportError:
        pass
    
    try:
        # Try importing with explicit module reloading
        import importlib.util
        import sys
        
        analysis_dir = op.dirname(op.abspath(__file__))
        stat_analysis_path = op.join(analysis_dir, 'statistical_analysis.py')
        
        if op.exists(stat_analysis_path):
            spec = importlib.util.spec_from_file_location("statistical_analysis", stat_analysis_path)
            statistical_analysis_module = importlib.util.module_from_spec(spec)
            sys.modules["statistical_analysis"] = statistical_analysis_module
            spec.loader.exec_module(statistical_analysis_module)
            
            StatisticalAnalyzer = statistical_analysis_module.StatisticalAnalyzer
            HAS_STATISTICAL_ANALYSIS = True
            print("✅ StatisticalAnalyzer imported successfully (dynamic import)")
            return
    except Exception:
        pass
    
    print("⚠️  Warning: StatisticalAnalyzer not available. Statistical tests will be skipped.")
    HAS_STATISTICAL_ANALYSIS = False

# Try to import the StatisticalAnalyzer
_import_statistical_analyzer()

# Try to import MNE for topographic plotting
try:
    import mne
    HAS_MNE = True
    mne.set_log_level('WARNING')
    
    # Import and register EGI montage tools
    import sys
    montage_tools_path = op.join(op.dirname(op.abspath(__file__)), '..', 'montage_tools')
    if montage_tools_path not in sys.path:
        sys.path.insert(0, montage_tools_path)
    
    from montage_tools import egi, montages
    egi.register()  # Register EGI equipment system
    print("✅ EGI montage tools registered successfully")
    HAS_EGI_TOOLS = True
except ImportError as e:
    HAS_MNE = False
    HAS_EGI_TOOLS = False
    print(f"Warning: MNE-Python or EGI tools not available. Topographic plots will be skipped. Error: {e}")

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

from scipy.stats import zscore
from scipy.stats import chi2
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# EGI-256 Configuration for Topoplots
# These definitions ensure proper electrode outlines for EGI 256-channel montage
# Note: Electrode numbers are 1-indexed in EGI naming, subtract 1 for Python 0-indexing
_egi256_outlines = {
    'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]) ,
    'ear2': np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]) ,
    'outer': np.array([9, 17, 24, 30, 31, 36, 45, 243, 240, 241, 242, 246, 250,
                       255, 90, 101, 110, 119, 132, 144, 164, 173, 186, 198,
                       207, 215, 228, 232, 236, 239, 238, 237, 233, 9]) ,
}

_egi256_rois = {
    'p3a': np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    'p3b': np.array([9, 45, 81, 100, 101, 110, 119, 128, 129, 132, 186]) - 1,
    'mmn': np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    'cnv': np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    'Fz': np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    'Cz': np.array([9, 45, 81, 132, 186]) - 1,
    'Pz': np.array([100, 101, 110, 119, 128, 129]) - 1,
    'scalp': np.arange(224),
    'nonscalp': np.arange(224, 256),
    'Fp1': np.array([26, 27, 32, 33, 34, 37, 38]) - 1,
    'Fp2': np.array([11, 12, 18, 19, 20, 25, 26]) - 1,
    'F3': np.array([30, 36, 40, 41, 42, 49, 50]) - 1,
    'F4': np.array([205, 206, 213, 214, 215, 223, 224]) - 1,
    'C3': np.array([51, 52, 58, 59, 60, 65, 66]) - 1,
    'C4': np.array([155, 164, 182, 183, 184, 195, 196]) - 1,
    'P3': np.array([76, 77, 85, 86, 87, 97, 98]) - 1,
    'P4': np.array([152, 153, 161, 162, 163, 171, 172]) - 1,
    'T5': np.array([83, 84, 85, 94, 95, 96, 104, 105, 106]) - 1,
    'T6': np.array([169, 170, 171, 177, 178, 179, 189, 190, 191]) - 1,
    'Oz': np.array([125, 136, 137, 138, 148]) - 1,
}

_egi_outlines = {
    256: _egi256_outlines
}

_egi_ch_names = {}
for i in [64, 65, 128, 129, 256, 257]:
    _egi_ch_names['{}'.format(i)] = ['E{}'.format(c) for c in range(1, i + 1)]

for i in [2, 4, 8, 16, 32, 64]:
    _egi_ch_names['{}a'.format(i)] = ['E{}'.format(c) for c in range(1, i + 1)]


def _prepare_egi256_sphere_and_outlines(evoked):
    """
    Prepare sphere and outlines for EGI-256 topographic plotting.
    
    Parameters
    ----------
    evoked : mne.Evoked
        Evoked object with EGI-256 montage set.
        
    Returns
    -------
    sphere : tuple or None
        Sphere definition (x, y, z, radius) for the head model.
    outlines : dict
        Outlines dictionary with proper boundaries and patch for plotting.
    """
    # Define sphere for EGI 256 (based on specific electrodes)
    _egi256_outlines = { 'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
        'ear2': np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]),
        'outer': np.array([9, 17, 24, 30, 31, 36, 45, 243, 240, 241, 242, 246, 250,
                       255, 90, 101, 110, 119, 132, 144, 164, 173, 186, 198,
                       207, 215, 228, 232, 236, 239, 238, 237, 233, 9]),
    }

    sphere_ch_names = ['E137', 'E26', 'E69', 'E202']
    ch_names = evoked.ch_names
    ch_idx = [ch_names.index(ch) for ch in sphere_ch_names if ch in ch_names]
    
    if len(ch_idx) == 4:
        pos_3d = np.stack([evoked.info['chs'][idx]['loc'][:3] for idx in ch_idx])
        radius = np.abs(pos_3d[[2, 3], 0]).mean()
        x = pos_3d[0, 0]
        y = pos_3d[-1, 1]
        z = pos_3d[:, -1].mean()
        sphere = (x, y, z, radius)
        print('Defined Esphere')
    #else:
    #    sphere = None
    
    # Get 2D positions for topomap
    _, pos, _, _, _, this_sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(evoked.info, 'eeg', sphere=sphere)
    
    # Build the outlines dictionary properly
    outlines = {}
    codes = []
    vertices = []
    for k, v in _egi256_outlines.items():
        t_verts = pos[v, :]
        outlines[k] = (t_verts[:, 0], t_verts[:, 1])
        t_codes = 2 * np.ones(v.shape[0])
        t_codes[0] = 1
        codes.append(t_codes)
        vertices.append(t_verts)
    
    vertices = np.concatenate(vertices, axis=0)
    codes = np.concatenate(codes, axis=0)
    
    # Add all required keys for MNE
    outlines['mask_pos'] = outlines['outer']
    outlines['clip_radius'] = clip_origin
    
    # Create path patch
    path = Path(vertices=vertices, codes=codes)
    
    def patch():
        return PathPatch(path, alpha=0.1)
    
    outlines['patch'] = patch
    
    return this_sphere, outlines


def _setup_montage_and_sphere(n_channels, topos_mean=None):
    """
    Set up MNE montage, info object, sphere, and outlines for topographic plotting.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    topos_mean : array, optional
        Mean topographic data (n_markers, n_channels) for creating evoked object.
        Required for 256-channel custom sphere/outlines.
        
    Returns
    -------
    info : mne.Info
        MNE info object with montage set
    sphere : tuple or str
        Sphere definition for plotting
    outlines : dict or str
        Outlines definition for plotting
    """
    if n_channels == 256:
        print('  Setting up EGI-256 montage with custom sphere and outlines')
        # Use standard GSN-HydroCel-256 montage
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage, on_missing='warn')
        
        if topos_mean is not None:
            # Create evoked object to calculate proper sphere and outlines
            evoked = mne.EvokedArray(topos_mean.T, info, tmin=0)
            sphere, outlines = _prepare_egi256_sphere_and_outlines(evoked)
            print(f'  ✓ Created EGI-256 montage with custom sphere: {sphere}')
        else:
            # Fallback if no data provided
            sphere = 'auto'
            outlines = 'head'
            print('  ⚠️  No data provided for custom sphere, using auto')
            
    elif n_channels == 128:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage)
        sphere = 'auto'
        outlines = 'head'
        print(f'  Created standard montage for 128 channels')
        
    elif n_channels == 64:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-64_1.0')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage)
        sphere = 'auto'
        outlines = 'head'
        print(f'  Created standard montage for 64 channels')
        
    else:
        # Create spherical layout as last resort
        print(f'  ⚠️  No standard montage for {n_channels} channels, creating generic layout')
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, 100, 'eeg')
        from mne.channels.layout import _auto_topomap_coords
        pos = _auto_topomap_coords(info, picks=None, sphere=None, ignore_overlap=True)
        montage_dict = dict(zip(ch_names, pos))
        montage = mne.channels.make_dig_montage(montage_dict, coord_frame='head')
        info.set_montage(montage)
        sphere = 'auto'
        outlines = 'head'
    
    return info, sphere, outlines


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
    
    FIXED VERSION: This now uses the ACTUAL marker order from NICE collection iteration.
    
    IMPORTANT: Both scalar and topographic features come from the SAME underlying 
    markers, just reduced differently:
    - Scalars: averaged across epochs AND channels → 1 value per marker
    - Topos: averaged across epochs ONLY → 1 value per marker per channel
    
    Therefore, they should have the SAME names since they represent the same markers.
    """
    
    def __init__(self):
        # IMPORTANT: These names must match EXACTLY the format used in analysis_summary.json
        # The format is: ClassName_comment (where comment comes from marker creation)
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
        if idx < len(self.scalar_names):
            return self.scalar_names[idx]
        return f'Scalar_Marker_{idx}'
    
    def get_topo_name(self, idx):
        if idx < len(self.topo_names):
            return self.topo_names[idx]
        return f'Topo_Marker_{idx}'


class OutlierDetector:
    """Methods for detecting and filtering outliers in data."""
    
    @staticmethod
    def detect_outliers_iqr(data, factor=1.5):
        """Detect outliers using the IQR method.
        
        Parameters
        ----------
        data : array-like
            Input data
        factor : float
            IQR factor for outlier detection (typically 1.5 or 3.0)
            
        Returns
        -------
        mask : boolean array
            True for outliers, False for normal values
        """
        data = np.asarray(data)
        if data.size == 0:
            return np.array([], dtype=bool)
            
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(data, threshold=3.0):
        """Detect outliers using z-score method.
        
        Parameters
        ----------
        data : array-like
            Input data
        threshold : float
            Z-score threshold for outlier detection
            
        Returns
        -------
        mask : boolean array
            True for outliers, False for normal values
        """
        data = np.asarray(data)
        if data.size == 0:
            return np.array([], dtype=bool)
            
        z_scores = np.abs(zscore(data, nan_policy='omit'))
        outliers = z_scores > threshold
        return outliers
    
    @staticmethod
    def filter_outliers_2d(matrix, method='iqr', factor=1.5, threshold=3.0):
        """Filter outliers from a 2D matrix for better visualization.
        
        Parameters
        ----------
        matrix : 2D array
            Input matrix
        method : str
            'iqr' or 'zscore'
        factor : float
            IQR factor (for IQR method)
        threshold : float
            Z-score threshold (for zscore method)
            
        Returns
        -------
        filtered_matrix : 2D array
            Matrix with outliers clipped to bounds
        outlier_info : dict
            Information about detected outliers
        """
        matrix = np.asarray(matrix)
        original_matrix = matrix.copy()
        
        # Flatten for outlier detection
        flat_data = matrix.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaNs
        
        if len(flat_data) == 0:
            return matrix, {'n_outliers': 0, 'outlier_percent': 0}
        
        # Detect outliers
        if method == 'iqr':
            outlier_mask = OutlierDetector.detect_outliers_iqr(flat_data, factor)
            # Get bounds
            q1 = np.percentile(flat_data, 25)
            q3 = np.percentile(flat_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
        else:  # zscore
            outlier_mask = OutlierDetector.detect_outliers_zscore(flat_data, threshold)
            # Get bounds based on std
            mean_val = np.nanmean(flat_data)
            std_val = np.nanstd(flat_data)
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
        
        # Count outliers
        n_outliers = np.sum(outlier_mask)
        outlier_percent = (n_outliers / len(flat_data)) * 100
        
        # Clip outliers to bounds
        filtered_matrix = matrix.copy()
        filtered_matrix = np.clip(filtered_matrix, lower_bound, upper_bound)
        
        outlier_info = {
            'n_outliers': int(n_outliers),
            'outlier_percent': float(outlier_percent),
            'total_values': len(flat_data),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'original_min': float(np.nanmin(matrix)),
            'original_max': float(np.nanmax(matrix)),
            'filtered_min': float(np.nanmin(filtered_matrix)),
            'filtered_max': float(np.nanmax(filtered_matrix))
        }
        
        return filtered_matrix, outlier_info
    
    @staticmethod
    def create_comparison_heatmaps(fig, axes, original_matrix, filtered_matrix, outlier_info, 
                                  title_base, xlabel, ylabel, xticklabels=None, yticklabels=None):
        """Create side-by-side comparison of original and outlier-filtered heatmaps.
        
        Parameters
        ----------
        fig : matplotlib figure
        axes : array of matplotlib axes (2 axes)
        original_matrix : 2D array
            Original data matrix
        filtered_matrix : 2D array  
            Outlier-filtered data matrix
        outlier_info : dict
            Information about outliers
        title_base : str
            Base title for the plots
        xlabel, ylabel : str
            Axis labels
        xticklabels, yticklabels : list, optional
            Tick labels
        """
        
        # Determine if data should be centered around 0
        orig_min, orig_max = np.nanmin(original_matrix), np.nanmax(original_matrix)
        filt_min, filt_max = np.nanmin(filtered_matrix), np.nanmax(filtered_matrix)
        
        # Check if data crosses zero significantly
        crosses_zero = (orig_min < -0.1 * abs(orig_max)) and (orig_max > 0.1 * abs(orig_min))
        
        if crosses_zero:
            # Use symmetric colormap around 0
            orig_vmax = max(abs(orig_min), abs(orig_max))
            filt_vmax = max(abs(filt_min), abs(filt_max))
            cmap = 'RdBu_r'
            orig_vmin, orig_vmax = -orig_vmax, orig_vmax
            filt_vmin, filt_vmax = -filt_vmax, filt_vmax
        else:
            # Use regular colormap
            cmap = 'viridis'
            orig_vmin, orig_vmax = orig_min, orig_max
            filt_vmin, filt_vmax = filt_min, filt_max
        
        # Original heatmap
        im1 = axes[0].imshow(original_matrix, aspect='auto', cmap=cmap, 
                            vmin=orig_vmin, vmax=orig_vmax)
        axes[0].set_title(f'{title_base} (Original)\nRange: [{orig_min:.3f}, {orig_max:.3f}]')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        
        # Filtered heatmap
        im2 = axes[1].imshow(filtered_matrix, aspect='auto', cmap=cmap,
                            vmin=filt_vmin, vmax=filt_vmax)
        axes[1].set_title(f'{title_base} (Outliers Filtered)\n'
                         f'{outlier_info["n_outliers"]} outliers ({outlier_info["outlier_percent"]:.1f}%) filtered')
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        
        # Set tick labels if provided
        if xticklabels is not None:
            for ax in axes:
                ax.set_xticks(range(len(xticklabels)))
                ax.set_xticklabels(xticklabels, rotation=45, ha='right')
        
        if yticklabels is not None:
            for ax in axes:
                ax.set_yticks(range(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[0])
        plt.colorbar(im2, ax=axes[1])
        
        return im1, im2


# Define local compute_gfp function for fallback
def compute_gfp_local(x, alpha=0.05):
    """Compute GFP with confidence intervals - simplified version."""
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


class GlobalFieldPowerGlobal:
    """Global Field Power analysis across all subjects."""
    
    def __init__(self, output_dir, subjects_data, results_dir, fif_data_dir=None):
        self.output_dir = output_dir
        self.subjects_data = subjects_data  # Dictionary with subject_id -> subject_info
        self.results_dir = results_dir
        self.fif_data_dir = fif_data_dir  # Directory containing raw .fif files
        
        # Create subdirectories
        self.plots_dir = op.join(output_dir, 'global_field_power', 'plots')
        self.metrics_dir = op.join(output_dir, 'global_field_power', 'metrics')
        
        for dir_path in [self.plots_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_all_subjects(self):
        """Analyze Global Field Power across all subjects."""
        print("  🌍 Computing Global Field Power analysis across all subjects...")
        
        if not HAS_MNE:
            print("     ⚠️  MNE-Python not available. Skipping GFP analysis.")
            return
        
        # Event types to analyze
        event_types = ['LSGS', 'LSGD', 'LDGS', 'LDGD']
        
        # Collect epochs from all subjects
        all_epochs_orig = []
        all_epochs_recon = []
        
        # Try to load cached GFP metrics first (much faster!)
        cached_gfp_data = self._load_cached_gfp_metrics()
        
        if cached_gfp_data:
            print(f"     🚀 Using cached GFP metrics from {len(cached_gfp_data)} subjects")
            print("     💡 This avoids expensive recomputation of individual GFP data")
            return self._analyze_gfp_from_cache(cached_gfp_data, event_types)
        
        # Fallback: compute from .fif files (legacy method)
        print("     ⚠️  No cached GFP metrics found. Computing from .fif files (slower)...")
        print("     💡 To speed up future runs, ensure individual analysis has been completed")
        
        for subject_id, subject_info in self.subjects_data.items():
            # [Previous .fif loading logic remains as fallback]
            # Reconstruct subject directory paths
            # subject_id format is like "sub-001_ses-01" or "AA069_ses-02"
            # Extract the actual subject and session parts
            if '_ses-' in subject_id:
                subj_part, sess_part = subject_id.split('_ses-')
                if not subj_part.startswith('sub-'):
                    subj_part = f'sub-{subj_part}'
                session_name = f'ses-{sess_part}'
            else:
                # Old structure without sessions
                if not subject_id.startswith('sub-'):
                    subj_part = f'sub-{subject_id}'
                else:
                    subj_part = subject_id
                session_name = None
            
            # Build possible directories to search for .fif files
            possible_dirs = []
            
            # If fif_data_dir is provided, use it as the primary location
            if self.fif_data_dir:
                if session_name:
                    fif_dir = op.join(self.fif_data_dir, subj_part, session_name)
                else:
                    fif_dir = op.join(self.fif_data_dir, subj_part)
                possible_dirs.append(fif_dir)
            
            # Fallback: try results directory structure
            if session_name:
                subject_dir = op.join(self.results_dir, subj_part, session_name)
            else:
                subject_dir = op.join(self.results_dir, subj_part)
            
            possible_dirs.extend([
                subject_dir,
                op.dirname(subject_dir),  # Go up one level
                op.join(op.dirname(op.dirname(subject_dir)), 'data'),  # data folder
            ])
            
            epochs_orig, epochs_recon = None, None
            for search_dir in possible_dirs:
                if op.exists(search_dir):
                    epochs_orig, epochs_recon = self._load_subject_epochs(search_dir, subject_id)
                    if epochs_orig is not None:
                        break
            
            if epochs_orig is not None:
                all_epochs_orig.append(epochs_orig)
                print(f"     ✅ Loaded epochs for subject {subject_id}")
            else:
                print(f"     ⚠️  Could not find .fif files for subject {subject_id}")
            
            if epochs_recon is not None:
                all_epochs_recon.append(epochs_recon)
        
        if len(all_epochs_orig) == 0:
            print("     ❌ No epochs loaded. Cannot perform GFP analysis.")
            return
        
        print(f"     📊 Loaded epochs from {len(all_epochs_orig)} subjects")
        
        # Use per-subject GFP analysis to avoid concatenation issues
        print("     🎯 Using per-subject GFP analysis (recommended approach)")
        print("     💡 This computes GFP for each subject individually, then aggregates statistically")
        return self._analyze_gfp_per_subject_alternative(all_epochs_orig, all_epochs_recon, event_types)
    
    def _analyze_gfp_per_subject_alternative(self, all_epochs_orig, all_epochs_recon, event_types):
        """
        GFP Difference Analysis: compute GFP difference (original - reconstructed) per subject,
        then aggregate statistically.
        
        This approach:
        1. Computes GFP for original and reconstructed data for each subject
        2. Computes GFP difference (original - reconstructed) for each subject  
        3. Aggregates differences: mean ± std across subjects for each time point
        4. Shows how reconstruction affects global brain activity strength
        """
        print("     🔄 Using per-subject GFP difference analysis")
        print("     💡 Computing: GFP_difference = GFP_original - GFP_reconstructed")
        print("     💡 Then: mean(differences) ± std(differences) across subjects")
        
        n_subjects_orig = len(all_epochs_orig)
        n_subjects_recon = len(all_epochs_recon)
        
        print(f"     📊 Processing {n_subjects_orig} subjects with original data")
        print(f"     📊 Processing {n_subjects_recon} subjects with reconstructed data")
        
        # Storage for per-subject GFP differences
        subject_gfp_differences = {}
        
        # Get times from first subject (should be consistent across subjects)
        times = all_epochs_orig[0].times
        
        # Process each subject individually - compute differences
        subjects_with_both_data = 0
        for i in range(min(n_subjects_orig, n_subjects_recon)):
            epochs_orig = all_epochs_orig[i]
            epochs_recon = all_epochs_recon[i] if i < len(all_epochs_recon) else None
            
            if epochs_recon is None:
                continue
                
            subject_id = f"subject_{i+1}"
            subject_gfp_differences[subject_id] = {}
            
            for event_type in event_types:
                # Check if event exists in both original and reconstructed data
                if (event_type in epochs_orig.event_id and 
                    event_type in epochs_recon.event_id):
                    
                    # Compute GFP for original data
                    orig_data = epochs_orig[event_type].get_data()  # (n_epochs, n_channels, n_times)
                    orig_gfp = self._compute_gfp_from_data(orig_data)  # (n_times,)
                    
                    # Compute GFP for reconstructed data  
                    recon_data = epochs_recon[event_type].get_data()
                    recon_gfp = self._compute_gfp_from_data(recon_data)  # (n_times,)
                    
                    # Compute GFP difference: original - reconstructed
                    gfp_difference = orig_gfp - recon_gfp  # (n_times,)
                    
                    subject_gfp_differences[subject_id][event_type] = gfp_difference
            
            subjects_with_both_data += 1
            
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"     📈 Processed {i + 1}/{min(n_subjects_orig, n_subjects_recon)} subjects")
        
        print(f"     ✅ Computed GFP differences for {subjects_with_both_data} subjects")
        
        if subjects_with_both_data == 0:
            print("     ❌ No subjects have both original and reconstructed data!")
            return
        
        # Now aggregate both individual GFP data AND differences
        print("     📊 Aggregating GFP data across subjects...")
        
        # Convert differences to include original GFP data too
        subject_gfp_complete = self._convert_to_complete_gfp_data(subject_gfp_differences, all_epochs_orig, all_epochs_recon, event_types)
        
        # Aggregate all data (original, reconstructed, and differences)
        aggregated_complete = self._aggregate_complete_gfp_data(subject_gfp_complete, event_types, times)
        
        # Create comprehensive plots
        print("     📈 Creating comprehensive Global Field Power plots...")
        self._create_comprehensive_gfp_plots(aggregated_complete, times, event_types)
        
        print("     ✅ Complete GFP analysis finished successfully")
    
    def _compute_gfp_from_data(self, data):
        """
        Compute Global Field Power from epoch data.
        
        DETAILED EXPLANATION OF GFP:
        ----------------------------
        Global Field Power (GFP) represents the "global strength" of electrical brain activity.
        
        Mathematical steps:
        1. Average across epochs: get typical response pattern for this condition
        2. For each time point: compute standard deviation across all EEG channels
        3. Result: single time series showing how "strong" the spatial brain pattern is
        
        Physical meaning:
        - HIGH GFP = Large voltage differences between channels → Strong spatial pattern
        - LOW GFP = Small voltage differences → Weak/diffuse activity  
        - PEAKS = Moments of synchronized, strong brain activity
        - VALLEYS = Moments of weak/desynchronized activity
        
        Why it's useful:
        - Reference-independent (doesn't depend on choice of reference electrode)
        - Summarizes complex multichannel data into interpretable time series
        - Shows temporal dynamics of global brain responses
        - Sensitive to reconstruction quality (differences reveal artifacts)
        
        Parameters:
        -----------
        data : array, shape (n_epochs, n_channels, n_times)
            EEG epoch data
            
        Returns:
        --------
        gfp : array, shape (n_times,)
            Global Field Power time series
            
        Mathematical formula:
        GFP(t) = std([V₁(t), V₂(t), ..., Vₙ(t)]) 
        where Vᵢ(t) is the voltage at channel i and time t
        """
        # Step 1: Average across epochs to get typical response pattern
        data_avg = np.mean(data, axis=0)  # (n_channels, n_times)
        
        # Step 2: Compute GFP = standard deviation across channels for each time point
        gfp = np.std(data_avg, axis=0)  # (n_times,)
        
        return gfp
    
    def _convert_to_complete_gfp_data(self, subject_gfp_differences, all_epochs_orig, all_epochs_recon, event_types):
        """
        Convert difference data to include original and reconstructed GFP data.
        
        This re-computes original and reconstructed GFP to provide complete data
        for comprehensive plotting.
        """
        complete_data = {
            'original': {},
            'reconstructed': {},
            'differences': subject_gfp_differences
        }
        
        # Re-compute original and reconstructed GFP for subjects with both data
        n_subjects = min(len(all_epochs_orig), len(all_epochs_recon))
        
        for i in range(n_subjects):
            epochs_orig = all_epochs_orig[i]
            epochs_recon = all_epochs_recon[i] if i < len(all_epochs_recon) else None
            
            if epochs_recon is None:
                continue
                
            subject_id = f"subject_{i+1}"
            complete_data['original'][subject_id] = {}
            complete_data['reconstructed'][subject_id] = {}
            
            for event_type in event_types:
                if (event_type in epochs_orig.event_id and 
                    event_type in epochs_recon.event_id):
                    
                    # Original GFP
                    orig_data = epochs_orig[event_type].get_data()
                    orig_gfp = self._compute_gfp_from_data(orig_data)
                    complete_data['original'][subject_id][event_type] = orig_gfp
                    
                    # Reconstructed GFP
                    recon_data = epochs_recon[event_type].get_data()
                    recon_gfp = self._compute_gfp_from_data(recon_data)
                    complete_data['reconstructed'][subject_id][event_type] = recon_gfp
        
        return complete_data
    
    def _aggregate_complete_gfp_data(self, subject_gfp_complete, event_types, times):
        """
        Aggregate complete GFP data with both empirical and chi-squared statistics.
        
        For GFP (which are standard deviations), we compute:
        1. Empirical statistics: mean ± std of GFP values across subjects
        2. Chi-squared confidence intervals: theoretical CI for standard deviations
        """
        from scipy.stats import chi2
        
        aggregated = {
            'times': times,
            'original': {},
            'reconstructed': {},
            'differences': {},
            'statistics_info': {}
        }
        
        # Process original, reconstructed, and differences
        for data_type in ['original', 'reconstructed', 'differences']:
            aggregated[data_type] = {}
            
            for event_type in event_types:
                # Collect data from all subjects for this event type
                all_subject_data = []
                
                for subject_id, subject_data in subject_gfp_complete[data_type].items():
                    if event_type in subject_data:
                        all_subject_data.append(subject_data[event_type])
                
                if all_subject_data:
                    # Convert to array: (n_subjects, n_times)
                    data_matrix = np.array(all_subject_data)
                    n_subjects = len(all_subject_data)
                    
                    # Empirical statistics
                    mean_data = np.mean(data_matrix, axis=0)
                    std_data = np.std(data_matrix, axis=0, ddof=1)  # Sample std
                 #   sem_data = std_data / np.sqrt(n_subjects)  # Standard error of mean
                    
                    # For original and reconstructed (which are standard deviations),
                    # also compute chi-squared confidence intervals
                    if data_type in ['original', 'reconstructed'] and n_subjects > 1:
                        # Chi-squared confidence intervals for standard deviations
                        alpha = 0.05  # 95% CI
                        df = n_subjects - 1
                        
                        # Compute chi-squared CI bounds for each time point
                        chi2_lower = chi2.ppf(alpha/2, df) 
                        chi2_upper = chi2.ppf(1-alpha/2, df)
                        
                        # Convert to confidence bounds for standard deviation
                        chi2_ci_lower = np.sqrt(df * std_data**2 / chi2_upper)
                        chi2_ci_upper = np.sqrt(df * std_data**2 / chi2_lower)
                    else:
                        chi2_ci_lower = chi2_ci_upper = None
                    
                    aggregated[data_type][event_type] = {
                        'mean': mean_data,
                        'std': std_data,
                 #       'sem': sem_data,
                        'n_subjects': n_subjects,
                        'chi2_ci_lower': chi2_ci_lower,
                        'chi2_ci_upper': chi2_ci_upper,
                        'raw_data_matrix': data_matrix  # Keep for advanced analysis
                    }
                    
                    ci_info = f" (Chi² CI available)" if chi2_ci_lower is not None else ""
                    print(f"     📊 {data_type.capitalize()} {event_type}: {n_subjects} subjects{ci_info}")
        
        # Store statistical approach info
        aggregated['statistics_info'] = {
            'empirical_approach': 'Standard empirical mean ± std across subjects',
            'chi2_approach': 'Chi-squared confidence intervals for standard deviations',
            'recommendation': 'Empirical is more robust for non-normal EEG data'
        }
        
        return aggregated
    
    def _aggregate_gfp_differences(self, subject_gfp_differences, event_types, times):
        """
        Aggregate per-subject GFP differences into population statistics.
        
        This computes the mean and standard deviation of GFP differences across subjects.
        
        Returns:
        --------
        aggregated_differences : dict
            Contains mean and std of GFP differences for each event type
        """
        aggregated = {
            'times': times,
            'event_data': {}
        }
        
        # For each event type, collect GFP differences from all subjects
        for event_type in event_types:
            all_subject_differences = []
            
            # Collect differences from all subjects for this event type
            for subject_id, subject_data in subject_gfp_differences.items():
                if event_type in subject_data:
                    all_subject_differences.append(subject_data[event_type])
            
            if all_subject_differences:
                # Convert to array: (n_subjects, n_times)
                differences_matrix = np.array(all_subject_differences)
                
                # Compute population statistics across subjects
                mean_difference = np.mean(differences_matrix, axis=0)  # (n_times,) - mean difference per time point
                std_difference = np.std(differences_matrix, axis=0)    # (n_times,) - std of differences per time point
                
                aggregated['event_data'][event_type] = {
                    'mean_difference': mean_difference,
                    'std_difference': std_difference,
                    'n_subjects': len(all_subject_differences),
                    'all_differences': differences_matrix  # Keep for additional analysis
                }
                
                print(f"     📊 {event_type}: {len(all_subject_differences)} subjects with differences")
        
        return aggregated
    
    def _create_gfp_difference_plots(self, aggregated_differences, times, event_types):
        """
        Create GFP difference plots showing mean ± std of (Original - Reconstructed) differences.
        
        This visualization shows:
        - How reconstruction affects global brain activity strength
        - Population-level patterns (mean across subjects)  
        - Individual variability (std across subjects)
        - Time points where reconstruction has biggest impact
        """
        times_ms = times * 1000  # Convert to milliseconds
        
        print(f"     📊 Creating difference plots for {len(aggregated_differences['event_data'])} event types")
        
        # 1. Individual event type difference plots
        for event_type in event_types:
            if event_type in aggregated_differences['event_data']:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                data = aggregated_differences['event_data'][event_type]
                mean_diff = data['mean_difference']
                std_diff = data['std_difference']
                n_subjects = data['n_subjects']
                
                # Plot mean difference
                ax.plot(times_ms, mean_diff * 1e6, 'purple', linewidth=3, 
                       label=f'Mean Difference (n={n_subjects})')
                
                # Plot ± 1 standard deviation band
                ax.fill_between(times_ms,
                               (mean_diff - std_diff) * 1e6,
                               (mean_diff + std_diff) * 1e6,
                               alpha=0.3, color='purple', label='±1 STD')
                
                # Add reference lines
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='No difference')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Stimulus onset')
                
                # Add stimulus timing markers (typical ERP experiment)
                stimulus_times = [150, 300, 450, 600]  # Common ERP component timings
                for stim_time in stimulus_times:
                    ax.axvline(x=stim_time, color='gray', linestyle=':', alpha=0.5)
                
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('GFP Difference (μV)\n(Original - Reconstructed)')
                ax.set_title(f'Global Field Power Difference - {event_type}\n'
                           f'Mean ± STD across {n_subjects} subjects\n'
                           f'Positive = Original > Reconstructed, Negative = Original < Reconstructed')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(op.join(self.plots_dir, f'gfp_difference_{event_type}_population.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"     ✅ Created {event_type} difference plot")
        
        # 2. Combined plot showing all event types differences
        if aggregated_differences['event_data']:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            fig.suptitle('Global Field Power Differences - All Event Types\n'
                        'Mean ± STD of (Original - Reconstructed) across subjects', fontsize=16)
            
            colors = ['purple', 'blue', 'green', 'orange', 'red']
            
            for i, event_type in enumerate(event_types):
                if event_type in aggregated_differences['event_data']:
                    data = aggregated_differences['event_data'][event_type]
                    mean_diff = data['mean_difference']
                    std_diff = data['std_difference']
                    n_subjects = data['n_subjects']
                    color = colors[i % len(colors)]
                    
                    # Plot mean difference
                    ax.plot(times_ms, mean_diff * 1e6, color=color, linewidth=2, 
                           label=f'{event_type} (n={n_subjects})')
                    
                    # Plot std band (lighter)
                    ax.fill_between(times_ms,
                                   (mean_diff - std_diff) * 1e6,
                                   (mean_diff + std_diff) * 1e6,
                                   alpha=0.2, color=color)
            
            # Add reference lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1, label='No difference')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Stimulus onset')
            
            # Add stimulus timing markers
            stimulus_times = [150, 300, 450, 600]
            for stim_time in stimulus_times:
                ax.axvline(x=stim_time, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('GFP Difference (μV)\n(Original - Reconstructed)')
            ax.set_title('Population-Level GFP Differences\n'
                        'How Reconstruction Affects Global Brain Activity Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op.join(self.plots_dir, 'gfp_differences_all_events_population.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     ✅ Created combined differences plot")
        
        # 3. Summary statistics plot
        self._create_gfp_difference_summary_plot(aggregated_differences, times_ms, event_types)
        
        print(f"     ✅ All GFP difference plots saved in {self.plots_dir}")
    
    def _create_gfp_difference_summary_plot(self, aggregated_differences, times_ms, event_types):
        """Create summary plot with GFP difference statistics."""
        
        if not aggregated_differences['event_data']:
            return
        
        # Create summary statistics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GFP Difference Summary Statistics Across All Events', fontsize=16)
        
        # Collect all data for summary
        all_mean_diffs = []
        all_std_diffs = []
        event_labels = []
        
        for event_type in event_types:
            if event_type in aggregated_differences['event_data']:
                data = aggregated_differences['event_data'][event_type]
                all_mean_diffs.append(data['mean_difference'])
                all_std_diffs.append(data['std_difference'])
                event_labels.append(event_type)
        
        if not all_mean_diffs:
            return
        
        all_mean_diffs = np.array(all_mean_diffs)  # (n_events, n_times)
        all_std_diffs = np.array(all_std_diffs)
        
        # Plot 1: Mean difference across all events
        axes[0,0].plot(times_ms, np.mean(all_mean_diffs, axis=0) * 1e6, 'red', linewidth=3)
        axes[0,0].fill_between(times_ms,
                              (np.mean(all_mean_diffs, axis=0) - np.std(all_mean_diffs, axis=0)) * 1e6,
                              (np.mean(all_mean_diffs, axis=0) + np.std(all_mean_diffs, axis=0)) * 1e6,
                              alpha=0.3, color='red')
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        axes[0,0].set_xlabel('Time (ms)')
        axes[0,0].set_ylabel('GFP Difference (μV)')
        axes[0,0].set_title('Grand Average Across All Events')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation patterns
        axes[0,1].plot(times_ms, np.mean(all_std_diffs, axis=0) * 1e6, 'orange', linewidth=2)
        axes[0,1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        axes[0,1].set_xlabel('Time (ms)')
        axes[0,1].set_ylabel('STD of GFP Differences (μV)')
        axes[0,1].set_title('Variability Across Subjects (Average STD)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Time-averaged statistics per event
        time_avg_means = np.mean(np.abs(all_mean_diffs), axis=1) * 1e6  # Average absolute difference
        time_avg_stds = np.mean(all_std_diffs, axis=1) * 1e6
        
        x_pos = np.arange(len(event_labels))
        axes[1,0].bar(x_pos, time_avg_means, color='skyblue', alpha=0.7)
        axes[1,0].set_xlabel('Event Types')
        axes[1,0].set_ylabel('Time-Averaged |GFP Difference| (μV)')
        axes[1,0].set_title('Average Reconstruction Effect per Event')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(event_labels)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Variability per event
        axes[1,1].bar(x_pos, time_avg_stds, color='lightcoral', alpha=0.7)
        axes[1,1].set_xlabel('Event Types')
        axes[1,1].set_ylabel('Time-Averaged STD (μV)')
        axes[1,1].set_title('Subject Variability per Event')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(event_labels)
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'gfp_difference_summary_statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Created summary statistics plot")
    
    def _create_comprehensive_gfp_plots(self, aggregated_complete, times, event_types):
        """
        Create comprehensive GFP plots showing:
        1. Original vs Reconstructed (with both empirical and chi-squared CIs)
        2. Differences (Original - Reconstructed)
        3. Combined overviews
        4. Statistical comparisons
        """
        times_ms = times * 1000  # Convert to milliseconds
        
        print(f"     📊 Creating comprehensive GFP plots for {len(event_types)} event types")
        
        # 1. ORIGINAL VS RECONSTRUCTED PLOTS (with statistical options)
        self._create_orig_vs_recon_plots(aggregated_complete, times_ms, event_types)
        
        # 2. DIFFERENCE PLOTS 
        self._create_difference_only_plots(aggregated_complete, times_ms, event_types)
        
        # 3. COMBINED OVERVIEW PLOTS
        self._create_combined_overview_plots(aggregated_complete, times_ms, event_types)
        
        # 4. STATISTICAL COMPARISON PLOTS
        self._create_statistical_comparison_plots(aggregated_complete, times_ms, event_types)
        
        print(f"     ✅ All comprehensive GFP plots saved in {self.plots_dir}")
    
    def _create_orig_vs_recon_plots(self, aggregated_complete, times_ms, event_types):
        """Create Original vs Reconstructed plots with statistical confidence bands."""
        
        print("     📈 Creating Original vs Reconstructed plots...")
        
        # Individual event plots
        for event_type in event_types:
            if (event_type in aggregated_complete['original'] and 
                event_type in aggregated_complete['reconstructed']):
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f'Global Field Power - {event_type}\nOriginal vs Reconstructed Data', fontsize=16)
                
                orig_data = aggregated_complete['original'][event_type]
                recon_data = aggregated_complete['reconstructed'][event_type]
                
                # Plot 1: Empirical statistics (mean ± std)
                axes[0].plot(times_ms, orig_data['mean'] * 1e6, 'blue', linewidth=3, label='Original')
                axes[0].fill_between(times_ms,
                                   (orig_data['mean'] - orig_data['std']) * 1e6,
                                   (orig_data['mean'] + orig_data['std']) * 1e6,
                                   alpha=0.3, color='blue', label='Original ±1 STD')
                
                axes[0].plot(times_ms, recon_data['mean'] * 1e6, 'red', linewidth=3, label='Reconstructed')
                axes[0].fill_between(times_ms,
                                   (recon_data['mean'] - recon_data['std']) * 1e6,
                                   (recon_data['mean'] + recon_data['std']) * 1e6,
                                   alpha=0.3, color='red', label='Reconstructed ±1 STD')
                
                axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                axes[0].set_xlabel('Time (ms)')
                axes[0].set_ylabel('Global Field Power (μV)')
                axes[0].set_title(f'Empirical Statistics (n={orig_data["n_subjects"]})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Chi-squared confidence intervals (if available)
                if (orig_data['chi2_ci_lower'] is not None and 
                    recon_data['chi2_ci_lower'] is not None):
                    
                    axes[1].plot(times_ms, orig_data['mean'] * 1e6, 'blue', linewidth=3, label='Original')
                    axes[1].fill_between(times_ms,
                                       orig_data['chi2_ci_lower'] * 1e6,
                                       orig_data['chi2_ci_upper'] * 1e6,
                                       alpha=0.3, color='blue', label='Original 95% Chi² CI')
                    
                    axes[1].plot(times_ms, recon_data['mean'] * 1e6, 'red', linewidth=3, label='Reconstructed')
                    axes[1].fill_between(times_ms,
                                       recon_data['chi2_ci_lower'] * 1e6,
                                       recon_data['chi2_ci_upper'] * 1e6,
                                       alpha=0.3, color='red', label='Reconstructed 95% Chi² CI')
                    
                    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1].set_xlabel('Time (ms)')
                    axes[1].set_ylabel('Global Field Power (μV)')
                    axes[1].set_title('Chi-Squared Confidence Intervals')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                else:
                    # No chi-squared CI available, show empirical SEM instead
                    axes[1].plot(times_ms, orig_data['mean'] * 1e6, 'blue', linewidth=3, label='Original')
                #    axes[1].fill_between(times_ms,
                #                       (orig_data['mean'] - orig_data['sem']) * 1e6,
                #                       (orig_data['mean'] + orig_data['sem']) * 1e6,
                #                       alpha=0.3, color='blue', label='Original ±1 SEM')
                    
                    axes[1].plot(times_ms, recon_data['mean'] * 1e6, 'red', linewidth=3, label='Reconstructed')
                 #   axes[1].fill_between(times_ms,
                 #                      (recon_data['mean'] - recon_data['sem']) * 1e6,
                #                       (recon_data['mean'] + recon_data['sem']) * 1e6,
                #                       alpha=0.3, color='red', label='Reconstructed ±1 SEM')
                    
                    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1].set_xlabel('Time (ms)')
                    axes[1].set_ylabel('Global Field Power (μV)')
                    axes[1].set_title('Standard Error of Mean')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(op.join(self.plots_dir, f'gfp_orig_vs_recon_{event_type}_statistics.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_difference_only_plots(self, aggregated_complete, times_ms, event_types):
        """Create difference-only plots."""
        
        print("     📈 Creating difference-only plots...")
        
        # Individual difference plots
        for event_type in event_types:
            if event_type in aggregated_complete['differences']:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                data = aggregated_complete['differences'][event_type]
                mean_diff = data['mean']
                std_diff = data['std']
             #   sem_diff = data['sem']
                n_subjects = data['n_subjects']
                
                # Plot mean difference with std band
                ax.plot(times_ms, mean_diff * 1e6, 'purple', linewidth=3, 
                       label=f'Mean Difference (n={n_subjects})')
                ax.fill_between(times_ms,
                               (mean_diff - std_diff) * 1e6,
                               (mean_diff + std_diff) * 1e6,
                               alpha=0.3, color='purple', label='±1 STD')
                
                # Also show SEM band (tighter, more conservative)
             #   ax.fill_between(times_ms,
             #              (mean_diff - std_diff) * 1e6,
             #                  (mean_diff + std_diff) * 1e6,
             #                  alpha=0.5, color='purple', label='±1 SEM')
                
                # Reference lines
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='No difference')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Stimulus onset')
                
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('GFP Difference (μV)\n(Original - Reconstructed)')
                ax.set_title(f'GFP Difference - {event_type}\n'
                           f'Mean ± STD/SEM across {n_subjects} subjects')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(op.join(self.plots_dir, f'gfp_difference_{event_type}_detailed.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_combined_overview_plots(self, aggregated_complete, times_ms, event_types):
        """Create combined overview plots showing all event types together."""
        
        print("     📈 Creating combined overview plots...")
        
        # 1. All event types - Original vs Reconstructed side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Global Field Power - All Event Types\nOriginal vs Reconstructed (Population Mean ± STD)', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # Original data
        for i, event_type in enumerate(event_types):
            if event_type in aggregated_complete['original']:
                data = aggregated_complete['original'][event_type]
                color = colors[i % len(colors)]
                ax1.plot(times_ms, data['mean'] * 1e6, color=color, linewidth=2, 
                        label=f'{event_type} (n={data["n_subjects"]})')
                ax1.fill_between(times_ms,
                               (data['mean'] - data['std']) * 1e6,
                               (data['mean'] + data['std']) * 1e6,
                               alpha=0.2, color=color)
        
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Global Field Power (μV)')
        ax1.set_title('Original Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reconstructed data
        for i, event_type in enumerate(event_types):
            if event_type in aggregated_complete['reconstructed']:
                data = aggregated_complete['reconstructed'][event_type]
                color = colors[i % len(colors)]
                ax2.plot(times_ms, data['mean'] * 1e6, color=color, linewidth=2, 
                        label=f'{event_type} (n={data["n_subjects"]})')
                ax2.fill_between(times_ms,
                               (data['mean'] - data['std']) * 1e6,
                               (data['mean'] + data['std']) * 1e6,
                               alpha=0.2, color=color)
        
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Global Field Power (μV)')
        ax2.set_title('Reconstructed Data')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'gfp_orig_vs_recon_all_events_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. All event types - Differences only
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('GFP Differences - All Event Types\n(Original - Reconstructed) Population Statistics', fontsize=16)
        
        for i, event_type in enumerate(event_types):
            if event_type in aggregated_complete['differences']:
                data = aggregated_complete['differences'][event_type]
                color = colors[i % len(colors)]
                
                ax.plot(times_ms, data['mean'] * 1e6, color=color, linewidth=2, 
                       label=f'{event_type} (n={data["n_subjects"]})')
                ax.fill_between(times_ms,
                               (data['mean'] - data['std']) * 1e6,
                               (data['mean'] + data['std']) * 1e6,
                               alpha=0.2, color=color)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('GFP Difference (μV)')
        ax.set_title('Population-Level GFP Differences\nPositive = Original > Reconstructed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'gfp_differences_all_events_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_comparison_plots(self, aggregated_complete, times_ms, event_types):
        """Create plots comparing empirical vs chi-squared statistics."""
        
        print("     📈 Creating statistical comparison plots...")
        
        # For each event type, compare empirical vs chi-squared approaches
        for event_type in event_types:
            if event_type in aggregated_complete['original']:
                orig_data = aggregated_complete['original'][event_type]
                
                # Only create if chi-squared CI is available
                if orig_data['chi2_ci_lower'] is not None:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                    fig.suptitle(f'Statistical Approach Comparison - {event_type}\n'
                               f'Empirical vs Chi-Squared Statistics', fontsize=16)
                    
                    # Plot 1: Empirical approach
                    ax1.plot(times_ms, orig_data['mean'] * 1e6, 'blue', linewidth=3, label='Mean GFP')
                    ax1.fill_between(times_ms,
                                   (orig_data['mean'] - orig_data['std']) * 1e6,
                                   (orig_data['mean'] + orig_data['std']) * 1e6,
                                   alpha=0.3, color='blue', label='±1 STD (empirical)')
                    
                    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    ax1.set_xlabel('Time (ms)')
                    ax1.set_ylabel('Global Field Power (μV)')
                    ax1.set_title('Empirical Statistics\n(Standard mean ± std)')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Chi-squared approach
                    ax2.plot(times_ms, orig_data['mean'] * 1e6, 'blue', linewidth=3, label='Mean GFP')
                    ax2.fill_between(times_ms,
                                   orig_data['chi2_ci_lower'] * 1e6,
                                   orig_data['chi2_ci_upper'] * 1e6,
                                   alpha=0.3, color='green', label='95% Chi² CI')
                    
                    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    ax2.set_xlabel('Time (ms)')
                    ax2.set_ylabel('Global Field Power (μV)')
                    ax2.set_title('Chi-Squared Statistics\n(Theoretical CI for std deviations)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(op.join(self.plots_dir, f'gfp_statistics_comparison_{event_type}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"     ✅ Created statistical comparison for {event_type}")
    
    def _load_subject_epochs(self, data_dir, subject_id):
        """Load EEG epochs data from .fif files for a single subject."""
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
        
        # Load original epochs
        epochs_orig = None
        if orig_files:
            epochs_file = orig_files[0]
            try:
                epochs_orig = mne.read_epochs(epochs_file, preload=True, verbose=False)
            except Exception as e:
                print(f"     ⚠️  Error loading original epochs for {subject_id}: {e}")
        
        # Load reconstructed epochs
        epochs_recon = None
        if recon_files:
            epochs_file = recon_files[0]
            try:
                epochs_recon = mne.read_epochs(epochs_file, preload=True, verbose=False)
            except Exception as e:
                print(f"     ⚠️  Error loading reconstructed epochs for {subject_id}: {e}")
        
        return epochs_orig, epochs_recon
    
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
            fig.suptitle(f'Global Field Power - {event_type} - All Subjects', fontsize=14)
            
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
                fig.suptitle(f'Global Field Power - All Event Types - All Subjects', fontsize=16)
                
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
                fig.suptitle(f'Global Field Power - All Event Types - All Subjects', fontsize=16)
                
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
            fig.suptitle(f'Global Field Power - All Event Types - All Subjects', fontsize=16)
            
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
        
        # Also create time series plot
        self._create_time_series_plot_all_events(event_types, epochs_orig, epochs_recon)
    
    def _create_time_series_plot_all_events(self, event_types, epochs_orig, epochs_recon=None):
        """Create time-series plot showing all events."""
        print("     📈 Creating time-series plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle(f'Global Field Power Time Series - All Subjects', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, event_type in enumerate(event_types):
            if event_type not in epochs_orig.event_id:
                continue
            
            color = colors[i % len(colors)]
            
            if HAS_NICE_EXT:
                plot_gfp(epochs_orig, conditions=[event_type], colors=[color],
                        labels=[event_type], ax=ax, fig_kwargs={}, sns_kwargs={})
            else:
                plot_gfp_local(epochs_orig, conditions=[event_type], colors=[color],
                              labels=[event_type], ax=ax, fig_kwargs={})
        
        ax.set_title('All Event Types - Original Data')
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'global_field_power_time_series.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_local_effect_plot(self, epochs_orig, epochs_recon=None):
        """Create plot showing Local Standard vs Local Deviant effect."""
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
                fig.suptitle(f'Local Effect Analysis - All Subjects', fontsize=16)
                
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
                fig.suptitle(f'Local Effect Analysis - All Subjects', fontsize=16)
                
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
            fig.suptitle(f'Local Effect Analysis - All Subjects', fontsize=16)
            
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
        """Create plot showing Global Standard vs Global Deviant effect."""
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
                fig.suptitle(f'Global Effect Analysis - All Subjects', fontsize=16)
                
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
                fig.suptitle(f'Global Effect Analysis - All Subjects', fontsize=16)
                
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
            fig.suptitle(f'Global Effect Analysis - All Subjects', fontsize=16)
            
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
        fig.suptitle(f'Absolute Difference: Original vs Reconstructed - All Subjects', fontsize=16)
        
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
    
    def _load_cached_gfp_metrics(self):
        """
        Load cached GFP metrics from individual subject analyses.
        
        Returns
        -------
        dict
            Dictionary with subject_id -> gfp_metrics, or None if no cache found
        """
        print("     🔍 Searching for cached GFP metrics...")
        
        cached_data = {}
        found_count = 0
        
        for subject_id, subject_info in self.subjects_data.items():
            # Reconstruct path to individual analysis results
            if '_ses-' in subject_id:
                subj_part, sess_part = subject_id.split('_ses-')
                if not subj_part.startswith('sub-'):
                    subj_part = f'sub-{subj_part}'
                session_name = f'ses-{sess_part}'
            else:
                subj_part = f'sub-{subject_id}' if not subject_id.startswith('sub-') else subject_id
                session_name = None
            
            # Build path to cached GFP metrics
            if session_name:
                metrics_file = op.join(self.results_dir, subj_part, session_name, 
                                     'individual_analysis', 'global_field_power', 
                                     'metrics', 'gfp_metrics.json')
            else:
                metrics_file = op.join(self.results_dir, subj_part, 
                                     'individual_analysis', 'global_field_power', 
                                     'metrics', 'gfp_metrics.json')
            
            # Try to load cached metrics
            if op.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    cached_data[subject_id] = metrics
                    found_count += 1
                    
                    if found_count <= 3:  # Show first few for debugging
                        print(f"     ✅ Loaded cache for {subject_id}: {len(metrics.get('event_types', {}))} events")
                        
                except Exception as e:
                    print(f"     ⚠️  Error loading cache for {subject_id}: {e}")
            else:
                if found_count <= 3:  # Only show first few missing files  
                    print(f"     ❌ No cache found for {subject_id}: {op.basename(metrics_file)}")
        
        if found_count > 3:
            print(f"     📊 Total: {found_count}/{len(self.subjects_data)} subjects have cached GFP metrics")
        
        # Return cached data if we have enough subjects
        if found_count >= len(self.subjects_data) * 0.5:  # At least 50% cached
            return cached_data
        else:
            print(f"     ⚠️  Only {found_count}/{len(self.subjects_data)} subjects have cached data")
            print("     🔄 Falling back to .fif file computation...")
            return None
    
    def _analyze_gfp_from_cache(self, cached_gfp_data, event_types):
        """
        Perform GFP analysis using cached metrics (much faster!).
        
        This method loads pre-computed GFP data and creates the same analysis
        plots without needing to reload and recompute .fif files.
        """
        print("     🚀 Analyzing GFP from cached metrics...")
        print(f"     📊 Processing {len(cached_gfp_data)} subjects with cached data")
        
        # Convert cached data to the format expected by our plotting methods
        subject_gfp_complete = {
            'original': {},
            'reconstructed': {},
            'differences': {}
        }
        
        # Get times from first subject (should be consistent)
        first_subject_data = next(iter(cached_gfp_data.values()))
        times = np.array(first_subject_data['times'])
        
        # Process each subject's cached data
        for subject_id, metrics in cached_gfp_data.items():
            subject_gfp_complete['original'][subject_id] = {}
            subject_gfp_complete['reconstructed'][subject_id] = {}
            subject_gfp_complete['differences'][subject_id] = {}
            
            for event_type in event_types:
                if event_type in metrics['event_types']:
                    event_data = metrics['event_types'][event_type]
                    
                    # Original GFP
                    if 'original' in event_data:
                        orig_gfp = np.array(event_data['original']['gfp_mean'])
                        subject_gfp_complete['original'][subject_id][event_type] = orig_gfp
                    
                    # Reconstructed GFP
                    if 'reconstructed' in event_data:
                        recon_gfp = np.array(event_data['reconstructed']['gfp_mean'])
                        subject_gfp_complete['reconstructed'][subject_id][event_type] = recon_gfp
                    
                    # Difference (if both exist)
                    if 'difference' in event_data:
                        diff_gfp = np.array(event_data['difference']['gfp_diff'])
                        subject_gfp_complete['differences'][subject_id][event_type] = diff_gfp
                    elif ('original' in event_data and 'reconstructed' in event_data):
                        # Compute difference from original and reconstructed
                        orig_gfp = np.array(event_data['original']['gfp_mean'])
                        recon_gfp = np.array(event_data['reconstructed']['gfp_mean'])
                        diff_gfp = orig_gfp - recon_gfp
                        subject_gfp_complete['differences'][subject_id][event_type] = diff_gfp
        
        # Get counts for reporting
        orig_count = len(subject_gfp_complete['original'])
        recon_count = len(subject_gfp_complete['reconstructed']) 
        diff_count = len(subject_gfp_complete['differences'])
        
        print(f"     📈 Extracted GFP data: {orig_count} orig, {recon_count} recon, {diff_count} diff")
        
        # Aggregate all data (original, reconstructed, and differences)
        print("     📊 Aggregating cached GFP data...")
        aggregated_complete = self._aggregate_complete_gfp_data(subject_gfp_complete, event_types, times)
        
        # Create comprehensive plots
        print("     📈 Creating Global Field Power plots from cached data...")
        self._create_comprehensive_gfp_plots(aggregated_complete, times, event_types)
        
        print("     ✅ Cached GFP analysis completed successfully!")


class GlobalAnalyzer:
    """Global analysis across multiple subjects."""
    
    def __init__(self, results_dir, output_dir, patient_labels_file=None, target_state=None, fif_data_dir=None, skip_gfp=False):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        self.target_state = target_state
        self.fif_data_dir = fif_data_dir  # Directory containing raw .fif files
        self.skip_gfp = skip_gfp  # Option to skip Global Field Power analysis
        self.mapper = MarkerNameMapper()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.plots_dir = op.join(output_dir, 'plots')
        self.data_dir = op.join(output_dir, 'data')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data containers
        self.subjects_data = {}
        self.global_scalar_data = {}
        self.global_topo_data = {}
        self.inconsistent_subjects = []  # Track subjects with inconsistent data shapes
        
        # Load patient labels if provided
        self.patient_labels = {}
        self.patient_labels_original = {}  # Store original diagnosis before grouping
        self.available_states = set()
        if self.patient_labels_file:
            self._load_patient_labels()
    
    def _load_patient_labels(self):
        """Load patient labels from CSV file with diagnosis grouping."""
        try:
            print(f"📋 Loading patient labels from: {self.patient_labels_file}")
            df = pd.read_csv(self.patient_labels_file)
            
            # Create a mapping from subject_session to state
            for _, row in df.iterrows():
                subject = row['subject']
                session = f"ses-{row['session']:02d}"  # Convert 1 -> ses-01, 2 -> ses-02
                state = row['state']
                
                # Skip subjects with n/a state
                if pd.isna(state) or state == 'n/a':
                    continue
                
                # Create key compatible with our subject_session format
                subject_session_key = f"{subject}_{session}"
                
                # Store original diagnosis before grouping
                self.patient_labels_original[subject_session_key] = state
                
                # Group diagnoses as requested:
                # - Merge MCS+ and MCS- into MCS
                # - Merge UWS and VS into VS/UWS (they are the same condition)
                if state in ['MCS+', 'MCS-']:
                    state = 'MCS'
                elif state == 'VS':
                    state = 'UWS'  # VS and UWS are the same, use UWS as standard
                
                self.patient_labels[subject_session_key] = state
                self.available_states.add(state)
            
            print(f"   ✓ Loaded labels for {len(self.patient_labels)} subject/sessions")
            print(f"   ✓ Available states (after grouping): {sorted(self.available_states)}")
            print(f"   ℹ️  Diagnosis grouping applied: MCS+/MCS- → MCS, VS → UWS")
            
            if self.target_state:
                filtered_count = sum(1 for state in self.patient_labels.values() if state == self.target_state)
                print(f"   🎯 Target state '{self.target_state}': {filtered_count} subject/sessions")
                
        except Exception as e:
            print(f"   ⚠️  Error loading patient labels: {e}")
            print("   ⚠️  Continuing without state filtering...")
    
    def _should_include_subject(self, subject_session_id):
        """Check if a subject/session should be included based on target state."""
        if not self.patient_labels_file or not self.target_state:
            return True  # Include all if no filtering
        
        # Check if this subject/session has the target state
        return self.patient_labels.get(subject_session_id) == self.target_state
    
    def collect_subject_data(self):
        """Collect data from all subjects with session support."""
        print("🔍 Collecting data from all subjects and sessions...")
        print(f"   Scanning directory: {self.results_dir}")
        
        subject_dirs = glob.glob(op.join(self.results_dir, 'sub-*'))
        print(f"   Found {len(subject_dirs)} potential subject directories")
        
        subjects_included = []
        subjects_skipped = []
        
        for subject_dir in subject_dirs:
            subject_id = op.basename(subject_dir).replace('sub-', '')
            print(f"\n   📂 Processing {subject_id}...")
            
            # Look for session directories within each subject
            session_dirs = glob.glob(op.join(subject_dir, 'ses-*'))
            
            if len(session_dirs) == 0:
                # Fallback: check if it's the old structure (no sessions)
                # Updated path: individual_analysis instead of compare_markers
                summary_file = op.join(subject_dir, 'individual_analysis', 'analysis_summary.json')
                features_dir = op.join(subject_dir, 'features_variable')
                
                if op.exists(summary_file) and op.exists(features_dir):
                    print(f"     📊 Found old structure for {subject_id}")
                    session_id = "legacy"
                    self._process_subject_session(subject_dir, subject_id, session_id, 
                                                  subjects_included, subjects_skipped)
                else:
                    print(f"     ❌ Skipping {subject_id}: no sessions found and no legacy structure")
                    subjects_skipped.append((f"{subject_id}/no-sessions", "no sessions or legacy structure"))
                continue
            
            print(f"     Found {len(session_dirs)} sessions for {subject_id}")
            
            for session_dir in session_dirs:
                session_id = op.basename(session_dir)
                print(f"     📂 Processing {subject_id}/{session_id}...")
                
                self._process_subject_session(session_dir, subject_id, session_id,
                                              subjects_included, subjects_skipped)
        
        print("\n📊 DATA COLLECTION SUMMARY:")
        print(f"   ✅ Successfully loaded: {len(subjects_included)} subject/sessions")
        print(f"      Subject/sessions: {subjects_included}")
        print(f"   ❌ Skipped: {len(subjects_skipped)} subject/sessions")
        for subj_id, reason in subjects_skipped:
            print(f"      {subj_id}: {reason}")
        
        if len(subjects_included) == 0:
            print("⚠️  WARNING: No subjects could be loaded for this analysis.")
            print("    This could be due to:")
            print("    1. No subjects matching the target state filter")
            print("    2. Missing data files in the directory structure") 
            print("    3. Incorrect data directory path")
            return []  # Return empty list instead of raising error
        
        # Validate consistency across subjects
        if len(subjects_included) > 1:
            first_subj = subjects_included[0]
            ref_scalar_shape = self.subjects_data[first_subj]['scalars_original'].shape
            ref_topo_shape = self.subjects_data[first_subj]['topos_original'].shape
            
            print("\n🔍 CROSS-SUBJECT CONSISTENCY CHECK:")
            print(f"   Reference (from {first_subj}): scalars={ref_scalar_shape}, topos={ref_topo_shape}")
            
            self.inconsistent_subjects = []
            for subj_id in subjects_included[1:]:
                subj_scalar_shape = self.subjects_data[subj_id]['scalars_original'].shape
                subj_topo_shape = self.subjects_data[subj_id]['topos_original'].shape
                
                if subj_scalar_shape != ref_scalar_shape or subj_topo_shape != ref_topo_shape:
                    self.inconsistent_subjects.append(subj_id)
                    print(f"   ❌ {subj_id}: scalars={subj_scalar_shape}, topos={subj_topo_shape}")
                else:
                    print(f"   ✅ {subj_id}: consistent")
            
            if len(self.inconsistent_subjects) > 0:
                print(f"\n⚠️  Warning: {len(self.inconsistent_subjects)} subjects have inconsistent data shapes!")
                print(f"   Inconsistent subjects: {self.inconsistent_subjects}")
                print("   These subjects will be EXCLUDED from topographic plots to prevent errors.")
                print("   They will still be included in scalar analysis.")
        
        print("\n✅ Data collection completed successfully!")
        print(f"   Ready for global analysis with {len(subjects_included)} subject/sessions")
        
        return subjects_included
    
    def _process_subject_session(self, session_dir, subject_id, session_id, 
                                 subjects_included, subjects_skipped):
        """Process a single subject/session combination."""
        
        # Check if analysis summary exists
        # Updated path: individual_analysis instead of compare_markers
        summary_file = op.join(session_dir, 'individual_analysis', 'analysis_summary.json')
        features_dir = op.join(session_dir, 'features_variable')
        
        # Create unique identifier for subject/session
        subject_session_id = f"{subject_id}_{session_id}"
        
        # Check state filtering first
        if not self._should_include_subject(subject_session_id):
            expected_state = self.patient_labels.get(subject_session_id, 'unknown')
            print(f"     ⏭️  Skipping {subject_id}/{session_id}: state '{expected_state}' != target '{self.target_state}'")
            subjects_skipped.append((subject_session_id, f"state mismatch: {expected_state} != {self.target_state}"))
            return
        
        if not op.exists(summary_file):
            print(f"     ❌ Skipping {subject_id}/{session_id}: no analysis summary found")
            subjects_skipped.append((subject_session_id, "no analysis summary"))
            return
            
        if not op.exists(features_dir):
            print(f"     ❌ Skipping {subject_id}/{session_id}: no features directory found")
            subjects_skipped.append((subject_session_id, "no features directory"))
            return
        
        try:
            # Load subject analysis summary
            print("     📄 Loading analysis summary...")
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Load feature data
            print("     📊 Loading feature data...")
            feature_files = ['scalars_original.npy', 'scalars_reconstructed.npy',
                           'topos_original.npy', 'topos_reconstructed.npy']
            
            for fname in feature_files:
                fpath = op.join(features_dir, fname)
                if not op.exists(fpath):
                    raise FileNotFoundError(f"Missing {fname}")
                print(f"        ✓ Found {fname}")
            
            scalars_orig = np.load(op.join(features_dir, 'scalars_original.npy'))
            scalars_recon = np.load(op.join(features_dir, 'scalars_reconstructed.npy'))
            topos_orig = np.load(op.join(features_dir, 'topos_original.npy'))
            topos_recon = np.load(op.join(features_dir, 'topos_reconstructed.npy'))
            
            # Validate data shapes
            print("     🔍 Validating data shapes...")
            print(f"        Scalars: {scalars_orig.shape} vs {scalars_recon.shape}")
            print(f"        Topos: {topos_orig.shape} vs {topos_recon.shape}")
            
            if scalars_orig.shape != scalars_recon.shape:
                raise ValueError(f"Scalar shape mismatch: {scalars_orig.shape} vs {scalars_recon.shape}")
            if topos_orig.shape != topos_recon.shape:
                raise ValueError(f"Topo shape mismatch: {topos_orig.shape} vs {topos_recon.shape}")
            if len(scalars_orig) != topos_orig.shape[0]:
                raise ValueError(f"Marker count mismatch: scalars {len(scalars_orig)} vs topos {topos_orig.shape[0]}")
            
            # Extract key metrics from summary
            scalar_corr = summary['scalar_metrics']['overall']['correlation']
            topo_corr = summary['topographic_metrics']['overall']['correlation']
            print(f"        📈 Metrics: scalar_corr={scalar_corr:.4f}, topo_corr={topo_corr:.4f}")
            
            # Store data with session-aware key
            subject_state = self.patient_labels.get(subject_session_id, 'unknown')
            self.subjects_data[subject_session_id] = {
                'summary': summary,
                'scalars_original': scalars_orig,
                'scalars_reconstructed': scalars_recon,
                'topos_original': topos_orig,
                'topos_reconstructed': topos_recon,
                'subject_id': subject_id,
                'session_id': session_id,
                'state': subject_state
            }
            
            subjects_included.append(subject_session_id)
            print(f"     ✅ Successfully loaded {subject_id}/{session_id}")
            
        except Exception as e:
            print(f"     ❌ Error loading {subject_id}/{session_id}: {e}")
            subjects_skipped.append((subject_session_id, str(e)))
            return
    
    def prepare_global_data(self):
        """Prepare global data structures."""
        print("Preparing global data structures...")
        
        subjects = list(self.subjects_data.keys())
        
        # Scalar data preparation
        self.global_scalar_data = {
            'subjects': subjects,
            'correlations': [],
            'mses': [],
            'maes': [],
            'cosine_similarities': [],
            'marker_data': {}  # marker_name -> {subjects: [], orig_vals: [], recon_vals: [], diffs: [], norm_sq_errors: [], norm_abs_errors: []}
        }
        
        # Filter out inconsistent subjects for topographic analysis
        # Scalar analysis can include all subjects since each marker is independent
        # But topo analysis needs consistent shapes for array operations
        topo_subjects = [s for s in subjects if s not in self.inconsistent_subjects]
        
        if len(self.inconsistent_subjects) > 0:
            print(f"  📊 Scalar analysis: {len(subjects)} subjects (all included)")
            print(f"  🗺️  Topo analysis: {len(topo_subjects)} subjects ({len(self.inconsistent_subjects)} excluded due to shape mismatch)")
        
        # Topographic data preparation - ONLY use consistent subjects
        self.global_topo_data = {
            'subjects': topo_subjects,  # Use filtered list
            'correlations': [],
            'mses': [],
            'maes': [],
            'nmses': [],
            'nrmses': [],
            'cosine_similarities': [],
            'topos_orig_all': [],
            'topos_recon_all': [],
            'n_markers': 0,
            'n_channels': 0
        }
        
        # Collect scalar data (all subjects)
        for subject_id in subjects:
            data = self.subjects_data[subject_id]
            scalar_metrics = data['summary']['scalar_metrics']['overall']
            
            self.global_scalar_data['correlations'].append(scalar_metrics['correlation'])
            self.global_scalar_data['mses'].append(scalar_metrics['mse'])
            self.global_scalar_data['maes'].append(scalar_metrics['mae'])
            self.global_scalar_data['cosine_similarities'].append(scalar_metrics['cosine_similarity'])
            
            # Per-marker data
            per_marker = data['summary']['scalar_metrics']['per_marker']
            for marker_name, marker_metrics in per_marker.items():
                if marker_name not in self.global_scalar_data['marker_data']:
                    self.global_scalar_data['marker_data'][marker_name] = {
                        'subjects': [], 'orig_vals': [], 'recon_vals': [], 
                        'diffs': [], 'norm_sq_errors': [], 'norm_abs_errors': []
                    }
                
                self.global_scalar_data['marker_data'][marker_name]['subjects'].append(subject_id)
                self.global_scalar_data['marker_data'][marker_name]['orig_vals'].append(marker_metrics['original_value'])
                self.global_scalar_data['marker_data'][marker_name]['recon_vals'].append(marker_metrics['reconstructed_value'])
                self.global_scalar_data['marker_data'][marker_name]['diffs'].append(marker_metrics['absolute_difference'])
                self.global_scalar_data['marker_data'][marker_name]['norm_sq_errors'].append(marker_metrics['norm_sq_error'])
                self.global_scalar_data['marker_data'][marker_name]['norm_abs_errors'].append(marker_metrics['norm_abs_error'])
        
        # Collect topographic data - ONLY from consistent subjects
        for subject_id in topo_subjects:
            data = self.subjects_data[subject_id]
            topo_metrics = data['summary']['topographic_metrics']['overall']
            
            self.global_topo_data['correlations'].append(topo_metrics['correlation'])
            self.global_topo_data['mses'].append(topo_metrics['mse'])
            self.global_topo_data['maes'].append(topo_metrics['mae'])
            self.global_topo_data['nmses'].append(topo_metrics['nmse'])
            self.global_topo_data['nrmses'].append(topo_metrics['nrmse'])
            self.global_topo_data['cosine_similarities'].append(topo_metrics['cosine_similarity'])
        
        # Collect time series error data (MSE/MAE per trial and sensor)
        self.global_timeseries_error_data = {
            'subjects': subjects,
            'mses': [],
            'maes': [],
            'mse_stds': [],
            'mae_stds': []
        }
        
        for subject_id in subjects:
            data = self.subjects_data[subject_id]
            # Check if time series error metrics exist
            if 'timeseries_error_metrics' in data['summary'] and data['summary']['timeseries_error_metrics'] is not None:
                ts_metrics = data['summary']['timeseries_error_metrics']['overall']
                self.global_timeseries_error_data['mses'].append(ts_metrics['mse'])
                self.global_timeseries_error_data['maes'].append(ts_metrics['mae'])
                self.global_timeseries_error_data['mse_stds'].append(ts_metrics.get('mse_std', 0))
                self.global_timeseries_error_data['mae_stds'].append(ts_metrics.get('mae_std', 0))
            else:
                print(f"     ⚠️  No time series error metrics for {subject_id}")
                self.global_timeseries_error_data['mses'].append(np.nan)
                self.global_timeseries_error_data['maes'].append(np.nan)
                self.global_timeseries_error_data['mse_stds'].append(np.nan)
                self.global_timeseries_error_data['mae_stds'].append(np.nan)
        
        # Collect topographic arrays - ONLY from topo_subjects (filtered to exclude inconsistent shapes)
        for subject_id in topo_subjects:
            data = self.subjects_data[subject_id]
            
            # Reconstruct topographic arrays from per-marker data
            # IMPORTANT: Use MarkerNameMapper order, NOT sorted order to maintain consistency
            per_marker_topo = data['summary']['topographic_metrics']['per_marker']
            topo_orig_list = []
            topo_recon_list = []
            
            # Use the same order as MarkerNameMapper to ensure consistency
            for i in range(len(per_marker_topo)):
                expected_marker_name = self.mapper.get_topo_name(i)
                if expected_marker_name in per_marker_topo:
                    topo_orig_list.append(per_marker_topo[expected_marker_name]['topo_original'])
                    topo_recon_list.append(per_marker_topo[expected_marker_name]['topo_reconstructed'])
                else:
                    # Fallback: find by index if name doesn't match
                    print(f"     ⚠️  Warning: Expected marker '{expected_marker_name}' at index {i} not found")
                    # Try to find the marker by index in the sorted keys
                    sorted_keys = sorted(per_marker_topo.keys())
                    if i < len(sorted_keys):
                        actual_marker_name = sorted_keys[i]
                        print(f"     🔄 Using '{actual_marker_name}' instead")
                        topo_orig_list.append(per_marker_topo[actual_marker_name]['topo_original'])
                        topo_recon_list.append(per_marker_topo[actual_marker_name]['topo_reconstructed'])
                    else:
                        raise ValueError(f"Cannot find marker for index {i}")
            
            # Verify we got all markers
            if len(topo_orig_list) != len(per_marker_topo):
                print(f"     ⚠️  Warning: Expected {len(per_marker_topo)} markers, got {len(topo_orig_list)}")
                print(f"     Expected markers: {[self.mapper.get_topo_name(i) for i in range(len(per_marker_topo))]}")
                print(f"     Available markers: {sorted(per_marker_topo.keys())}")
            
            self.global_topo_data['topos_orig_all'].append(np.array(topo_orig_list))
            self.global_topo_data['topos_recon_all'].append(np.array(topo_recon_list))
        
        # Set dimensions from first subject
        if topo_subjects:
            first_topo = self.global_topo_data['topos_orig_all'][0]
            self.global_topo_data['n_markers'] = first_topo.shape[0]
            self.global_topo_data['n_channels'] = first_topo.shape[1]
        
        print(f"  Prepared data for {len(subjects)} subjects")
        print(f"  Scalar markers: {len(self.global_scalar_data['marker_data'])}")
        if topo_subjects:
            print(f"  Topo dimensions: {self.global_topo_data['n_markers']} markers × {self.global_topo_data['n_channels']} channels")
            print(f"  Topo subjects: {len(topo_subjects)}")
        else:
            print(f"  ⚠️  No subjects available for topographic analysis (all have inconsistent shapes)")
    
    def create_scalar_global_plots(self):
        """Create global scalar analysis plots."""
        print("Creating global scalar plots...")
        
        subjects = self.global_scalar_data['subjects']
        
        # 1. Global correlation and MSE per subject
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        state_suffix = f" - State: {self.target_state}" if self.target_state else ""
        fig.suptitle(f'Global Scalar Analysis Across Subjects{state_suffix}', fontsize=16)
        
        # Correlation plot
        correlations = self.global_scalar_data['correlations']
        x_pos = np.arange(len(subjects))
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations, ddof=1) if len(correlations) > 1 else 0
        
        axes[0].plot(x_pos, correlations, 'o-', linewidth=2, markersize=8)
        axes[0].axhline(y=mean_corr, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_corr:.4f}')
        axes[0].axhline(y=mean_corr + std_corr, color='gray', linestyle=':', alpha=0.5, label='±1 STD')
        axes[0].axhline(y=mean_corr - std_corr, color='gray', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Subject ID')
        axes[0].set_ylabel('Correlation')
        axes[0].set_title(f'Global Correlation (Mean: {mean_corr:.4f} ± {std_corr:.4f})')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(subjects, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MSE plot
        mses = self.global_scalar_data['mses']
        mean_mse = np.mean(mses)
        std_mse = np.std(mses, ddof=1) if len(mses) > 1 else 0
        
        axes[1].plot(x_pos, mses, 'o-', linewidth=2, markersize=8, color='red')
        axes[1].axhline(y=mean_mse, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_mse:.4f}')
        axes[1].axhline(y=mean_mse + std_mse, color='gray', linestyle=':', alpha=0.5, label='±1 STD')
        axes[1].axhline(y=mean_mse - std_mse, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Subject ID')
        axes[1].set_ylabel('MSE')
        axes[1].set_title(f'Global MSE (Mean: {mean_mse:.4f} ± {std_mse:.4f})')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(subjects, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'scalar_global_correlation_mse.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmaps: subjects × markers - modified version
        marker_names = list(self.global_scalar_data['marker_data'].keys())
        n_markers = len(marker_names)
        n_subjects = len(subjects)
        
        print(f"     🔥 Creating scalar heatmaps ({n_subjects} subjects × {n_markers} markers)")
        
        # Prepare heatmap data
        absolute_diff_matrix = np.zeros((n_subjects, n_markers))
        normalized_diff_matrix = np.zeros((n_subjects, n_markers))
        
        for i, subject_id in enumerate(subjects):
            for j, marker_name in enumerate(marker_names):
                if subject_id in self.global_scalar_data['marker_data'][marker_name]['subjects']:
                    idx = self.global_scalar_data['marker_data'][marker_name]['subjects'].index(subject_id)
                    orig_val = self.global_scalar_data['marker_data'][marker_name]['orig_vals'][idx]
                    recon_val = self.global_scalar_data['marker_data'][marker_name]['recon_vals'][idx]
                    
                    # Absolute difference between original and reconstructed values
                    absolute_diff_matrix[i, j] = abs(orig_val - recon_val)
                    
                    # Normalized difference: |orig - recon| / mean(|orig|, |recon|)
                    mean_val = (abs(orig_val) + abs(recon_val)) / 2
                    normalized_diff_matrix[i, j] = abs(orig_val - recon_val) / (mean_val + 1e-8)
        
        # Create heatmaps
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('Scalar Features: Subject × Marker Analysis', fontsize=16)
        
        # Absolute difference heatmap
        im1 = axes[0].imshow(absolute_diff_matrix, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title('Absolute Difference |Original - Reconstructed|')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Subjects')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[0].set_yticks(range(n_subjects))
        axes[0].set_yticklabels(subjects)
        plt.colorbar(im1, ax=axes[0])
        
        # Normalized difference heatmap
        im2 = axes[1].imshow(normalized_diff_matrix, aspect='auto', cmap='Reds', vmin=0)
        axes[1].set_title('Normalized Difference |Original - Reconstructed| / Mean(|Original|, |Reconstructed|)')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Subjects')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[1].set_yticks(range(n_subjects))
        axes[1].set_yticklabels(subjects)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'scalar_heatmaps_subjects_markers.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # 3. Mean norm_sq_error and correlation per marker
        # Improved mean statistics per marker calculations
        mean_norm_sq_errors = []
        std_norm_sq_errors = []
        mean_correlations = []
        std_correlations = []
        mean_normalized_diffs = []
        std_normalized_diffs = []
        
        for marker_name in marker_names:
            # Get data for this marker across all subjects
            orig_vals = np.array(self.global_scalar_data['marker_data'][marker_name]['orig_vals'])
            recon_vals = np.array(self.global_scalar_data['marker_data'][marker_name]['recon_vals'])
            n_subjects = len(orig_vals)
            
            # 1. Norm Sq Error (using existing calculation)
            marker_norm_sq_errors = self.global_scalar_data['marker_data'][marker_name]['norm_sq_errors']
            mean_norm_sq_errors.append(np.mean(marker_norm_sq_errors))
            # Standard error of the mean: std / sqrt(N)
            std_norm_sq_errors.append(np.std(marker_norm_sq_errors, ddof=1) / np.sqrt(n_subjects) if n_subjects > 1 else 0)
            
            # 2. Correlation calculation (proper Pearson correlation)
            if np.std(orig_vals) > 1e-8 and np.std(recon_vals) > 1e-8:
                corr_coeff = np.corrcoef(orig_vals, recon_vals)[0, 1]
                # For multiple subjects, we have one correlation per marker
                mean_correlations.append(corr_coeff)
                std_correlations.append(0)  # Single correlation value per marker
            else:
                mean_correlations.append(0)
                std_correlations.append(0)
            
            # 3. Normalized difference: |orig - recon| / mean(|orig|, |recon|)
            normalized_diffs = []
            for orig, recon in zip(orig_vals, recon_vals):
                mean_abs_val = (abs(orig) + abs(recon)) / 2
                if mean_abs_val > 1e-8:
                    normalized_diffs.append(abs(orig - recon) / mean_abs_val)
                else:
                    normalized_diffs.append(0)
            
            mean_normalized_diffs.append(np.mean(normalized_diffs))
            # Standard error of the mean: std / sqrt(N)
            std_normalized_diffs.append(np.std(normalized_diffs, ddof=1) / np.sqrt(n_subjects) if n_subjects > 1 else 0)
        
        # Create figure with single subplot - Mean Normalized Difference only
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        fig.suptitle('Mean Normalized Difference per Marker Across Subjects', fontsize=16)
        
        x_pos = np.arange(len(marker_names))
        
        # Mean normalized difference plot
        ax.plot(x_pos, mean_normalized_diffs, 'o-', linewidth=2, markersize=6, color='blue')
        ax.fill_between(x_pos, 
                        np.array(mean_normalized_diffs) - np.array(std_normalized_diffs), 
                        np.array(mean_normalized_diffs) + np.array(std_normalized_diffs), 
                        alpha=0.3, color='blue')
        ax.set_xlabel('Markers')
        ax.set_ylabel('Mean Normalized Difference')
        ax.set_title('Mean Normalized Difference |Orig-Recon|/Mean(|Orig|,|Recon|) per Marker (±SEM)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(marker_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'scalar_mean_stats_per_marker.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        scalar_summary_data = pd.DataFrame({
            'Subject': subjects,
            'Correlation': self.global_scalar_data['correlations'],
            'MSE': self.global_scalar_data['mses'],
            'MAE': self.global_scalar_data['maes'],
            'Cosine_Similarity': self.global_scalar_data['cosine_similarities']
        })
        scalar_summary_data.to_csv(op.join(self.data_dir, 'scalar_global_summary.csv'), index=False)
        
        # Save per-marker statistics
        marker_stats_data = pd.DataFrame({
            'Marker': marker_names,
            'Mean_Norm_Sq_Error': mean_norm_sq_errors,
            'SEM_Norm_Sq_Error': std_norm_sq_errors,
            'Pearson_Correlation': mean_correlations,
            'Mean_Normalized_Difference': mean_normalized_diffs,
            'SEM_Normalized_Difference': std_normalized_diffs
        })
        marker_stats_data.to_csv(op.join(self.data_dir, 'scalar_marker_statistics.csv'), index=False)
    
    def create_topographic_global_plots(self):
        """Create global topographic analysis plots."""
        print("Creating global topographic plots...")
        
        subjects = self.global_topo_data['subjects']
        n_subjects = len(subjects)
        
        # Check if we have any subjects for topo analysis
        if n_subjects == 0:
            print("  ⚠️  No subjects available for topographic plots (all have inconsistent shapes)")
            print("  Skipping topographic plots...")
            return
        
        n_markers = self.global_topo_data['n_markers']
        n_channels = self.global_topo_data['n_channels']
        
        # Convert to numpy arrays - should work now since we filtered out inconsistent subjects
        try:
            topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])  # (n_subjects, n_markers, n_channels)
            topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
        except ValueError as e:
            print(f"  ❌ Error converting topo data to arrays: {e}")
            print(f"  This should not happen after filtering. Debugging info:")
            print(f"     Subjects: {subjects}")
            print(f"     Shapes: {[np.array(t).shape for t in self.global_topo_data['topos_orig_all']]}")
            return
        
        # 1. Global correlation and MSE per subject (same as scalars but for topo)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        state_suffix = f" - State: {self.target_state}" if self.target_state else ""
        fig.suptitle(f'Global Topographic Analysis Across Subjects{state_suffix}', fontsize=16)
        
        correlations = self.global_topo_data['correlations']
        nmses = self.global_topo_data['nmses']
        x_pos = np.arange(len(subjects))
        
        # Correlation plot
        axes[0].plot(x_pos, correlations, 'o-', linewidth=2, markersize=8)
        axes[0].fill_between(x_pos, 
                           np.array(correlations) - np.std(correlations), 
                           np.array(correlations) + np.std(correlations), 
                           alpha=0.3)
        axes[0].set_xlabel('Subject ID')
        axes[0].set_ylabel('Correlation')
        axes[0].set_title(f'Global Topo Correlation (Mean: {np.mean(correlations):.4f} ± {np.std(correlations):.4f})')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(subjects, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # NMSE plot
        axes[1].plot(x_pos, nmses, 'o-', linewidth=2, markersize=8, color='red')
        axes[1].fill_between(x_pos, 
                           np.array(nmses) - np.std(nmses), 
                           np.array(nmses) + np.std(nmses), 
                           alpha=0.3, color='red')
        axes[1].set_xlabel('Subject ID')
        axes[1].set_ylabel('NMSE')
        axes[1].set_title(f'Global Topo NMSE (Mean: {np.mean(nmses):.4f} ± {np.std(nmses):.4f})')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(subjects, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_global_correlation_nmse.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmaps: subjects × sensors (NMSE and correlation across markers)
        print(f"     🔥 Creating topographic subject × sensor heatmaps ({n_subjects} subjects × {n_channels} sensors)")
        
        sensor_nmse_matrix = np.zeros((n_subjects, n_channels))
        sensor_corr_matrix = np.zeros((n_subjects, n_channels))
        
        for i in range(n_subjects):
            for j in range(n_channels):
                # NMSE across markers for this subject and sensor
                mse_vals = (topos_orig_all[i, :, j] - topos_recon_all[i, :, j]) ** 2
                var_orig = np.var(topos_orig_all[i, :, j], ddof=1) if len(topos_orig_all[i, :, j]) > 1 else np.var(topos_orig_all[i, :, j])
                sensor_nmse_matrix[i, j] = np.mean(mse_vals) / (var_orig + 1e-8)
                # Correlation across markers for this subject and sensor
                if np.std(topos_orig_all[i, :, j]) > 1e-8 and np.std(topos_recon_all[i, :, j]) > 1e-8:
                    sensor_corr_matrix[i, j] = np.corrcoef(topos_orig_all[i, :, j], topos_recon_all[i, :, j])[0, 1]
                else:
                    sensor_corr_matrix[i, j] = 0
        
        # Create regular heatmaps
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle('Topographic: Subject × Sensor Analysis (Original)', fontsize=16)
        
        # NMSE heatmap - check for negative values and use symmetric range
        nmse_min, nmse_max = np.min(sensor_nmse_matrix), np.max(sensor_nmse_matrix)
        if nmse_min < 0:
            nmse_vmax = max(abs(nmse_min), abs(nmse_max))
            im1 = axes[0].imshow(sensor_nmse_matrix, aspect='auto', cmap='RdBu_r', 
                                vmin=-nmse_vmax, vmax=nmse_vmax)
        else:
            im1 = axes[0].imshow(sensor_nmse_matrix, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title('NMSE per Subject and Sensor (across markers)')
        axes[0].set_xlabel('Sensors')
        axes[0].set_ylabel('Subjects')
        axes[0].set_yticks(range(n_subjects))
        axes[0].set_yticklabels(subjects)
        plt.colorbar(im1, ax=axes[0])
        
        # Correlation heatmap - symmetric around 0
        corr_vmax = max(abs(np.nanmin(sensor_corr_matrix)), abs(np.nanmax(sensor_corr_matrix)))
        im2 = axes[1].imshow(sensor_corr_matrix, aspect='auto', cmap='RdBu_r', 
                            vmin=-corr_vmax, vmax=corr_vmax)
        axes[1].set_title('Correlation per Subject and Sensor (across markers)')
        axes[1].set_xlabel('Sensors')
        axes[1].set_ylabel('Subjects')
        axes[1].set_yticks(range(n_subjects))
        axes[1].set_yticklabels(subjects)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_sensors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create outlier-filtered versions
        print(f"     🧹 Creating outlier-filtered subject × sensor heatmaps...")
        
        # Filter outliers
        sensor_nmse_filtered, sensor_nmse_outlier_info = OutlierDetector.filter_outliers_2d(
            sensor_nmse_matrix, method='iqr', factor=1.5)
        sensor_corr_filtered, sensor_corr_outlier_info = OutlierDetector.filter_outliers_2d(
            sensor_corr_matrix, method='iqr', factor=1.5)
            
        print(f"        📊 Sensor NMSE outliers: {sensor_nmse_outlier_info['n_outliers']} ({sensor_nmse_outlier_info['outlier_percent']:.1f}%)")
        print(f"        📊 Sensor correlation outliers: {sensor_corr_outlier_info['n_outliers']} ({sensor_corr_outlier_info['outlier_percent']:.1f}%)")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(32, 10))
        fig.suptitle('Topographic: Subject × Sensor Analysis (Original vs Outlier-Filtered)', fontsize=16)
        
        # NMSE comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[0], sensor_nmse_matrix, sensor_nmse_filtered, sensor_nmse_outlier_info,
            'NMSE per Subject and Sensor (across markers)', 'Sensors', 'Subjects',
            yticklabels=subjects)
        
        # Correlation comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[1], sensor_corr_matrix, sensor_corr_filtered, sensor_corr_outlier_info,
            'Correlation per Subject and Sensor (across markers)', 'Sensors', 'Subjects',
            yticklabels=subjects)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_sensors_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save filtered versions separately
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle('Topographic: Subject × Sensor Analysis (Outlier-Filtered)', fontsize=16)
        
        # Filtered NMSE heatmap
        sensor_nmse_filt_min, sensor_nmse_filt_max = np.nanmin(sensor_nmse_filtered), np.nanmax(sensor_nmse_filtered)
        if sensor_nmse_filt_min < 0:
            sensor_nmse_filt_vmax = max(abs(sensor_nmse_filt_min), abs(sensor_nmse_filt_max))
            im1 = axes[0].imshow(sensor_nmse_filtered, aspect='auto', cmap='RdBu_r', 
                                vmin=-sensor_nmse_filt_vmax, vmax=sensor_nmse_filt_vmax)
        else:
            im1 = axes[0].imshow(sensor_nmse_filtered, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title(f'NMSE per Subject and Sensor (Filtered: {sensor_nmse_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[0].set_xlabel('Sensors')
        axes[0].set_ylabel('Subjects')
        axes[0].set_yticks(range(n_subjects))
        axes[0].set_yticklabels(subjects)
        plt.colorbar(im1, ax=axes[0])
        
        # Filtered correlation heatmap
        sensor_corr_filt_vmax = max(abs(np.nanmin(sensor_corr_filtered)), abs(np.nanmax(sensor_corr_filtered)))
        im2 = axes[1].imshow(sensor_corr_filtered, aspect='auto', cmap='RdBu_r', 
                            vmin=-sensor_corr_filt_vmax, vmax=sensor_corr_filt_vmax)
        axes[1].set_title(f'Correlation per Subject and Sensor (Filtered: {sensor_corr_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[1].set_xlabel('Sensors')
        axes[1].set_ylabel('Subjects')
        axes[1].set_yticks(range(n_subjects))
        axes[1].set_yticklabels(subjects)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_sensors_filtered.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmaps: subjects × markers (NMSE and correlation across sensors)
        
        marker_nmse_matrix = np.zeros((n_subjects, n_markers))
        marker_corr_matrix = np.zeros((n_subjects, n_markers))
        
        for i in range(n_subjects):
            for j in range(n_markers):
                # NMSE across sensors for this subject and marker
                mse_vals = (topos_orig_all[i, j, :] - topos_recon_all[i, j, :]) ** 2
                var_orig = np.var(topos_orig_all[i, j, :], ddof=1) if len(topos_orig_all[i, j, :]) > 1 else np.var(topos_orig_all[i, j, :])
                marker_nmse_matrix[i, j] = np.mean(mse_vals) / (var_orig + 1e-8)
                # Correlation across sensors for this subject and marker
                if np.std(topos_orig_all[i, j, :]) > 1e-8 and np.std(topos_recon_all[i, j, :]) > 1e-8:
                    marker_corr_matrix[i, j] = np.corrcoef(topos_orig_all[i, j, :], topos_recon_all[i, j, :])[0, 1]
                else:
                    marker_corr_matrix[i, j] = 0
        
        marker_names = [self.mapper.get_topo_name(i) for i in range(n_markers)]
        
        # Create regular heatmaps
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle('Topographic: Subject × Marker Analysis (Original)', fontsize=16)
        
        # NMSE heatmap - check for negative values and use symmetric range
        nmse_min, nmse_max = np.min(marker_nmse_matrix), np.max(marker_nmse_matrix)
        if nmse_min < 0:
            nmse_vmax = max(abs(nmse_min), abs(nmse_max))
            im1 = axes[0].imshow(marker_nmse_matrix, aspect='auto', cmap='RdBu_r', 
                                vmin=-nmse_vmax, vmax=nmse_vmax)
        else:
            im1 = axes[0].imshow(marker_nmse_matrix, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title('NMSE per Subject and Marker (across sensors)')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Subjects')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[0].set_yticks(range(n_subjects))
        axes[0].set_yticklabels(subjects)
        plt.colorbar(im1, ax=axes[0])
        
        # Correlation heatmap - symmetric around 0
        corr_vmax = max(abs(np.nanmin(marker_corr_matrix)), abs(np.nanmax(marker_corr_matrix)))
        im2 = axes[1].imshow(marker_corr_matrix, aspect='auto', cmap='RdBu_r', 
                            vmin=-corr_vmax, vmax=corr_vmax)
        axes[1].set_title('Correlation per Subject and Marker (across sensors)')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Subjects')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[1].set_yticks(range(n_subjects))
        axes[1].set_yticklabels(subjects)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_markers.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create outlier-filtered versions
        print(f"     🧹 Creating outlier-filtered subject × marker heatmaps...")
        
        # Filter outliers
        marker_nmse_filtered, marker_nmse_outlier_info = OutlierDetector.filter_outliers_2d(
            marker_nmse_matrix, method='iqr', factor=1.5)
        marker_corr_filtered, marker_corr_outlier_info = OutlierDetector.filter_outliers_2d(
            marker_corr_matrix, method='iqr', factor=1.5)
            
        print(f"        📊 Marker NMSE outliers: {marker_nmse_outlier_info['n_outliers']} ({marker_nmse_outlier_info['outlier_percent']:.1f}%)")
        print(f"        📊 Marker correlation outliers: {marker_corr_outlier_info['n_outliers']} ({marker_corr_outlier_info['outlier_percent']:.1f}%)")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(32, 10))
        fig.suptitle('Topographic: Subject × Marker Analysis (Original vs Outlier-Filtered)', fontsize=16)
        
        # NMSE comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[0], marker_nmse_matrix, marker_nmse_filtered, marker_nmse_outlier_info,
            'NMSE per Subject and Marker (across sensors)', 'Markers', 'Subjects',
            xticklabels=marker_names, yticklabels=subjects)
        
        # Correlation comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[1], marker_corr_matrix, marker_corr_filtered, marker_corr_outlier_info,
            'Correlation per Subject and Marker (across sensors)', 'Markers', 'Subjects',
            xticklabels=marker_names, yticklabels=subjects)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_markers_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save filtered versions separately
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle('Topographic: Subject × Marker Analysis (Outlier-Filtered)', fontsize=16)
        
        # Filtered NMSE heatmap
        marker_nmse_filt_min, marker_nmse_filt_max = np.nanmin(marker_nmse_filtered), np.nanmax(marker_nmse_filtered)
        if marker_nmse_filt_min < 0:
            marker_nmse_filt_vmax = max(abs(marker_nmse_filt_min), abs(marker_nmse_filt_max))
            im1 = axes[0].imshow(marker_nmse_filtered, aspect='auto', cmap='RdBu_r', 
                                vmin=-marker_nmse_filt_vmax, vmax=marker_nmse_filt_vmax)
        else:
            im1 = axes[0].imshow(marker_nmse_filtered, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title(f'NMSE per Subject and Marker (Filtered: {marker_nmse_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Subjects')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[0].set_yticks(range(n_subjects))
        axes[0].set_yticklabels(subjects)
        plt.colorbar(im1, ax=axes[0])
        
        # Filtered correlation heatmap
        marker_corr_filt_vmax = max(abs(np.nanmin(marker_corr_filtered)), abs(np.nanmax(marker_corr_filtered)))
        im2 = axes[1].imshow(marker_corr_filtered, aspect='auto', cmap='RdBu_r', 
                            vmin=-marker_corr_filt_vmax, vmax=marker_corr_filt_vmax)
        axes[1].set_title(f'Correlation per Subject and Marker (Filtered: {marker_corr_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Subjects')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[1].set_yticks(range(n_subjects))
        axes[1].set_yticklabels(subjects)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_subjects_markers_filtered.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmaps: sensors × markers (mean NMSE and correlation across subjects)
        print(f"     🔥 Creating topographic sensor × marker heatmaps ({n_channels} sensors × {n_markers} markers)")
        
        sensor_marker_nmse = np.zeros((n_channels, n_markers))
        sensor_marker_corr = np.zeros((n_channels, n_markers))
        
        for i in range(n_channels):
            for j in range(n_markers):
                # Mean NMSE across subjects for this sensor and marker
                orig_vals = [topos_orig_all[s, j, i] for s in range(n_subjects)]
                recon_vals = [topos_recon_all[s, j, i] for s in range(n_subjects)]
                mses = [(o - r) ** 2 for o, r in zip(orig_vals, recon_vals)]
                var_orig = np.var(orig_vals, ddof=1) if len(orig_vals) > 1 else np.var(orig_vals)
                sensor_marker_nmse[i, j] = np.mean(mses) / (var_orig + 1e-8)
                
                # Mean correlation across subjects for this sensor and marker
                if np.std(orig_vals) > 1e-8 and np.std(recon_vals) > 1e-8:
                    sensor_marker_corr[i, j] = np.corrcoef(orig_vals, recon_vals)[0, 1]
                else:
                    sensor_marker_corr[i, j] = 0
        
        # Create regular heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Topographic: Sensor × Marker Analysis (Original)', fontsize=16)
        
        # NMSE heatmap - check for negative values and use symmetric range
        nmse_min, nmse_max = np.min(sensor_marker_nmse), np.max(sensor_marker_nmse)
        if nmse_min < 0:
            nmse_vmax = max(abs(nmse_min), abs(nmse_max))
            im1 = axes[0].imshow(sensor_marker_nmse, aspect='auto', cmap='RdBu_r', 
                                vmin=-nmse_vmax, vmax=nmse_vmax)
        else:
            im1 = axes[0].imshow(sensor_marker_nmse, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title('Mean NMSE per Sensor and Marker')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Sensors')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        plt.colorbar(im1, ax=axes[0])
        
        # Correlation heatmap - symmetric around 0
        corr_vmax = max(abs(np.nanmin(sensor_marker_corr)), abs(np.nanmax(sensor_marker_corr)))
        im2 = axes[1].imshow(sensor_marker_corr, aspect='auto', cmap='RdBu_r', 
                            vmin=-corr_vmax, vmax=corr_vmax)
        axes[1].set_title('Mean Correlation per Sensor and Marker')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Sensors')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_sensors_markers_mean.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create outlier-filtered versions
        print(f"     🧹 Creating outlier-filtered sensor × marker heatmaps...")
        
        # Filter outliers
        sensor_marker_nmse_filtered, sensor_marker_nmse_outlier_info = OutlierDetector.filter_outliers_2d(
            sensor_marker_nmse, method='iqr', factor=1.5)
        sensor_marker_corr_filtered, sensor_marker_corr_outlier_info = OutlierDetector.filter_outliers_2d(
            sensor_marker_corr, method='iqr', factor=1.5)
            
        print(f"        📊 Sensor-marker NMSE outliers: {sensor_marker_nmse_outlier_info['n_outliers']} ({sensor_marker_nmse_outlier_info['outlier_percent']:.1f}%)")
        print(f"        📊 Sensor-marker correlation outliers: {sensor_marker_corr_outlier_info['n_outliers']} ({sensor_marker_corr_outlier_info['outlier_percent']:.1f}%)")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(32, 16))
        fig.suptitle('Topographic: Sensor × Marker Analysis (Original vs Outlier-Filtered)', fontsize=16)
        
        # NMSE comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[0], sensor_marker_nmse, sensor_marker_nmse_filtered, sensor_marker_nmse_outlier_info,
            'Mean NMSE per Sensor and Marker', 'Markers', 'Sensors',
            xticklabels=marker_names)
        
        # Correlation comparison
        OutlierDetector.create_comparison_heatmaps(
            fig, axes[1], sensor_marker_corr, sensor_marker_corr_filtered, sensor_marker_corr_outlier_info,
            'Mean Correlation per Sensor and Marker', 'Markers', 'Sensors',
            xticklabels=marker_names)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_sensors_markers_mean_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save filtered versions separately
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Topographic: Sensor × Marker Analysis (Outlier-Filtered)', fontsize=16)
        
        # Filtered NMSE heatmap
        sensor_marker_nmse_filt_min, sensor_marker_nmse_filt_max = np.nanmin(sensor_marker_nmse_filtered), np.nanmax(sensor_marker_nmse_filtered)
        if sensor_marker_nmse_filt_min < 0:
            sensor_marker_nmse_filt_vmax = max(abs(sensor_marker_nmse_filt_min), abs(sensor_marker_nmse_filt_max))
            im1 = axes[0].imshow(sensor_marker_nmse_filtered, aspect='auto', cmap='RdBu_r', 
                                vmin=-sensor_marker_nmse_filt_vmax, vmax=sensor_marker_nmse_filt_vmax)
        else:
            im1 = axes[0].imshow(sensor_marker_nmse_filtered, aspect='auto', cmap='Reds', vmin=0)
        axes[0].set_title(f'Mean NMSE per Sensor and Marker (Filtered: {sensor_marker_nmse_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Sensors')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        plt.colorbar(im1, ax=axes[0])
        
        # Filtered correlation heatmap
        sensor_marker_corr_filt_vmax = max(abs(np.nanmin(sensor_marker_corr_filtered)), abs(np.nanmax(sensor_marker_corr_filtered)))
        im2 = axes[1].imshow(sensor_marker_corr_filtered, aspect='auto', cmap='RdBu_r', 
                            vmin=-sensor_marker_corr_filt_vmax, vmax=sensor_marker_corr_filt_vmax)
        axes[1].set_title(f'Mean Correlation per Sensor and Marker (Filtered: {sensor_marker_corr_outlier_info["outlier_percent"]:.1f}% outliers removed)')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Sensors')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'topo_heatmaps_sensors_markers_mean_filtered.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save topographic data to CSV
        topo_summary_data = pd.DataFrame({
            'Subject': subjects,
            'Correlation': self.global_topo_data['correlations'],
            'MSE': self.global_topo_data['mses'],
            'MAE': self.global_topo_data['maes'],
            'NMSE': self.global_topo_data['nmses'],
            'NRMSE': self.global_topo_data['nrmses'],
            'Cosine_Similarity': self.global_topo_data['cosine_similarities']
        })
        topo_summary_data.to_csv(op.join(self.data_dir, 'topo_global_summary.csv'), index=False)
    
    def create_timeseries_error_plots(self):
        """Create global time series error (MSE/MAE) plots."""
        print("Creating global time series error plots...")
        
        subjects = self.global_timeseries_error_data['subjects']
        mses = np.array(self.global_timeseries_error_data['mses'])
        maes = np.array(self.global_timeseries_error_data['maes'])
        
        # Remove NaN values for statistics
        valid_mask = ~(np.isnan(mses) | np.isnan(maes))
        if not np.any(valid_mask):
            print("  ⚠️  No valid time series error data found. Skipping plots.")
            return
        
        valid_subjects = [s for s, v in zip(subjects, valid_mask) if v]
        valid_mses = mses[valid_mask]
        valid_maes = maes[valid_mask]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        state_suffix = f" - State: {self.target_state}" if self.target_state else ""
        fig.suptitle(f'Time Series Error Analysis Across Subjects{state_suffix}', fontsize=16, fontweight='bold')
        
        x_pos = np.arange(len(valid_subjects))
        
        # MSE plot
        mean_mse = np.mean(valid_mses)
        std_mse = np.std(valid_mses, ddof=1) if len(valid_mses) > 1 else 0
        
        axes[0].bar(x_pos, valid_mses, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axhline(y=mean_mse, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_mse:.6f}', alpha=0.7)
        axes[0].axhline(y=mean_mse + std_mse, color='gray', linestyle=':', alpha=0.5, label='±1 STD')
        axes[0].axhline(y=mean_mse - std_mse, color='gray', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Subject ID', fontsize=12)
        axes[0].set_ylabel('MSE', fontsize=12)
        axes[0].set_title(f'MSE per Subject\n(Mean: {mean_mse:.6f} ± {std_mse:.6f})', fontsize=14)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(valid_subjects, rotation=45, ha='right')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # MAE plot
        mean_mae = np.mean(valid_maes)
        std_mae = np.std(valid_maes, ddof=1) if len(valid_maes) > 1 else 0
        
        axes[1].bar(x_pos, valid_maes, color='coral', alpha=0.7, edgecolor='black')
        axes[1].axhline(y=mean_mae, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_mae:.6f}', alpha=0.7)
        axes[1].axhline(y=mean_mae + std_mae, color='gray', linestyle=':', alpha=0.5, label='±1 STD')
        axes[1].axhline(y=mean_mae - std_mae, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Subject ID', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title(f'MAE per Subject\n(Mean: {mean_mae:.6f} ± {std_mae:.6f})', fontsize=14)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(valid_subjects, rotation=45, ha='right')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'timeseries_error_global_mse_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved time series error plots")
        
        # Save summary data to CSV
        ts_error_summary_data = pd.DataFrame({
            'Subject': valid_subjects,
            'MSE': valid_mses,
            'MAE': valid_maes
        })
        ts_error_summary_data.to_csv(op.join(self.data_dir, 'timeseries_error_global_summary.csv'), index=False)
        
        # Save overall statistics to JSON
        ts_error_stats = {
            'overall_statistics': {
                'mse_mean': float(mean_mse),
                'mse_std': float(std_mse),
                'mse_min': float(np.min(valid_mses)),
                'mse_max': float(np.max(valid_mses)),
                'mae_mean': float(mean_mae),
                'mae_std': float(std_mae),
                'mae_min': float(np.min(valid_maes)),
                'mae_max': float(np.max(valid_maes)),
                'n_subjects': len(valid_subjects)
            },
            'per_subject': {
                subj: {
                    'mse': float(mse_val),
                    'mae': float(mae_val)
                }
                for subj, mse_val, mae_val in zip(valid_subjects, valid_mses, valid_maes)
            }
        }
        
        with open(op.join(self.data_dir, 'timeseries_error_statistics.json'), 'w') as f:
            json.dump(ts_error_stats, f, indent=2)
        
        print(f"  💾 Saved time series error statistics to JSON")
        print(f"     📊 Overall MSE: {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"     📊 Overall MAE: {mean_mae:.6f} ± {std_mae:.6f}")
    
    def create_mne_topomap_plots(self):
        """Create MNE topographic plots for original, reconstructed, and NMSE data."""
       # if not HAS_MNE:
       #     print("  ⚠️  Skipping MNE topomap plots - MNE-Python not available")
       #     return
            
        print("Creating MNE topographic plots...")
        
        subjects = self.global_topo_data['subjects']
        n_subjects = len(subjects)
        
        # Check if we have any subjects for topo analysis
        if n_subjects == 0:
            print("  ⚠️  No subjects available for MNE topomap plots (all have inconsistent shapes)")
            print("  Skipping MNE topomap plots...")
            return
        
        n_markers = self.global_topo_data['n_markers']
        n_channels = self.global_topo_data['n_channels']
        
        print(f"  📊 Data summary: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        
        # Check for shape consistency across subjects
        shapes_orig = [np.array(topo).shape for topo in self.global_topo_data['topos_orig_all']]
        shapes_recon = [np.array(topo).shape for topo in self.global_topo_data['topos_recon_all']]
        
        print(f"  🔍 Checking data shapes across subjects...")
        unique_shapes = set(shapes_orig)
        if len(unique_shapes) > 1:
            print(f"  ⚠️  WARNING: Found {len(unique_shapes)} different shapes across subjects:")
            for shape in unique_shapes:
                count = shapes_orig.count(shape)
                print(f"     - Shape {shape}: {count} subjects")
        
        # Convert to numpy arrays - handle potentially inhomogeneous shapes
        try:
            topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])  # (n_subjects, n_markers, n_channels)
            topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
            print(f"  ✅ Successfully converted data to arrays: {topos_orig_all.shape}")
        except ValueError as e:
            # Handle inhomogeneous shapes (different markers per subject)
            print(f"  ⚠️  ERROR: Subjects have different numbers of markers or channels")
            print(f"  Cannot create topomap plots with inconsistent data shapes")
            print(f"  Error: {e}")
            print(f"  💡 TIP: This usually happens when subjects have different acquisition parameters")
            print(f"  Skipping MNE topomap plots...")
            return
        
        # Calculate mean across subjects for each marker
        topos_orig_mean = np.mean(topos_orig_all, axis=0)  # (n_markers, n_channels)
        topos_recon_mean = np.mean(topos_recon_all, axis=0)
        
        # Calculate NMSE per marker (mean across subjects)
        topos_nmse = np.zeros((n_markers, n_channels))
        for m in range(n_markers):
            for ch in range(n_channels):
                # Calculate NMSE across subjects for this marker and channel
                mses = [(topos_orig_all[s, m, ch] - topos_recon_all[s, m, ch]) ** 2 for s in range(n_subjects)]
                orig_vals = [topos_orig_all[s, m, ch] for s in range(n_subjects)]
                var_orig = np.var(orig_vals, ddof=1) if len(orig_vals) > 1 else np.var(orig_vals)
                topos_nmse[m, ch] = np.mean(mses) / (var_orig + 1e-8)
        
        # Set up montage with proper sphere and outlines using helper function
        print(f"  📡 Setting up EGI montage for {n_channels} channels")
        info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_orig_mean)
        
        marker_names = [self.mapper.get_topo_name(i) for i in range(n_markers)]
        
        # Define marker groups according to specifications
        psd_non_normalized = [
            'PowerSpectralDensity_delta', 'PowerSpectralDensity_theta', 'PowerSpectralDensity_alpha', 
            'PowerSpectralDensity_beta', 'PowerSpectralDensity_gamma'
        ]
        
        psd_normalized = [
            'PowerSpectralDensity_deltan', 'PowerSpectralDensity_thetan', 'PowerSpectralDensity_alphan',
            'PowerSpectralDensity_betan', 'PowerSpectralDensity_gamman'
        ]
        
        psd_summary = [
            'PowerSpectralDensity_summary_se', 'PowerSpectralDensitySummary_summary_msf', 
            'PowerSpectralDensitySummary_summary_sef90', 'PowerSpectralDensitySummary_summary_sef95'
        ]
        
        entropy_markers = [
            'PermutationEntropy_default', 'SymbolicMutualInformation_weighted', 'KolmogorovComplexity_default'
        ]
        
        time_locked_topography_markers = [
            'ContingentNegativeVariation_default', 'TimeLockedTopography_p1', 'TimeLockedTopography_p3a', 'TimeLockedTopography_p3b', 'TimeLockedContrast_LSGS-LDGD', 'TimeLockedContrast_LSGD-LDGS', 'TimeLockedContrast_LD-LS', 'TimeLockedContrast_GD-GS'
        ]

        time_locked_contrast_markers = [
            'TimeLockedContrast_LSGS-LDGD', 'TimeLockedContrast_LSGD-LDGS', 'TimeLockedContrast_LD-LS', 'TimeLockedContrast_GD-GS', 'TimeLockedContrast_mmn', 'TimeLockedContrast_p3a', 'TimeLockedContrast_p3b'
        ]
        
        # Create mapping from marker name to index
        marker_name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        # Define the 5 plots with their respective marker groups
        plot_groups = [
            ('PSD Non-Normalized', psd_non_normalized),
            ('PSD Normalized', psd_normalized),
            ('PSD Summary', psd_summary),
            ('Entropy Markers', entropy_markers),
            ('Time-Locked Contrast Markers', time_locked_contrast_markers),
            ('Time-Locked Topography Markers', time_locked_topography_markers)
        ]
        
        plot_count = 0
        
        for plot_num, (plot_title, marker_group) in enumerate(plot_groups):
            # Find indices for markers that exist in this group
            marker_indices = []
            existing_markers = []
            
            for marker_name in marker_group:
                if marker_name in marker_name_to_idx:
                    marker_indices.append(marker_name_to_idx[marker_name])
                    existing_markers.append(marker_name)
                else:
                    print(f"  ⚠️  Marker '{marker_name}' not found in data")
            
            if len(marker_indices) == 0:
                print(f"  ⚠️  No markers found for {plot_title}, skipping plot")
                continue
        
            n_rows = len(marker_indices)
            
            fig, axes = plt.subplots(n_rows, 4, figsize=(25, max(4, n_rows * 3)))  # 4 columns: orig, recon, difference, nmse
          #  fig.suptitle(f'MNE Topographic Maps - {plot_title} ({len(marker_indices)} markers)', fontsize=16)
            
            # Handle single row case
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Plot each marker in this group
            for row, marker_idx in enumerate(marker_indices):
                marker_name = existing_markers[row]
                
                # Original data
                orig_data = topos_orig_mean[marker_idx, :]
                # Reconstructed data  
                recon_data = topos_recon_mean[marker_idx, :]
                # Difference data (Original - Reconstructed)
                diff_data = orig_data - recon_data
                
                # Find common scale for original and reconstructed (and use same for difference in 4th column)
                data_min = min(np.min(orig_data), np.min(recon_data))
                data_max = max(np.max(orig_data), np.max(recon_data))
                
                # For 3rd column difference, use symmetric scale around 0
                diff_abs_max = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
                diff_symmetric_min, diff_symmetric_max = -diff_abs_max, diff_abs_max
                
                # Plot original
                im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                                 vlim=(data_min, data_max), cmap='viridis',
                                                 sphere=sphere, outlines=outlines,
                                                 extrapolate='local',
                                                 res=256, sensors=True, contours=6)
                
  # Set title for original plot
                if len(marker_name) > 15:
                    title_orig = f'{marker_name[:15]}\n{marker_name[15:]} Original'
                else:
                    title_orig = f'{marker_name}\nOriginal'
                axes[row, 0].set_title(title_orig)
                    
                    # Plot reconstructed
                im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                                 vlim=(data_min, data_max), cmap='viridis',
                                                 sphere=sphere, outlines=outlines,
                                                 extrapolate='local',
                                         res=256, sensors=True, contours=6)
                # Set title for reconstructed plot
                if len(marker_name) > 15:
                    title_recon = f'{marker_name[:15]}\n{marker_name[15:]} Reconstructed'
                else:
                    title_recon = f'{marker_name}\nReconstructed'
                axes[row, 1].set_title(title_recon)
                    
                    # Plot difference (Original - Reconstructed) with symmetric scale
                im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                                 vlim=(diff_symmetric_min, diff_symmetric_max),
                                                 show=False, cmap='viridis',
                                                 sphere=sphere, outlines=outlines,
                                                 extrapolate='local',
                                         res=256, sensors=True, contours=6)
                # Handle long marker names
                if len(marker_name) > 15:
                    axes[row, 2].set_title(f'{marker_name[:15]}\n{marker_name[15:]} Difference')
                else:
                    axes[row, 2].set_title(f'{marker_name}\nDifference')
                    
                    # Plot difference again but with same scale as original/reconstructed
                im4, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 3],
                                                 vlim=(data_min, data_max),
                                                 show=False, cmap='viridis',
                                                 sphere=sphere, outlines=outlines,
                                                 extrapolate='local',
                                         res=256, sensors=True, contours=6)
                if len(marker_name) > 15:
                    axes[row, 3].set_title(f'{marker_name[:15]}\n{marker_name[15:]} Difference')
                else:
                    axes[row, 3].set_title(f'{marker_name}\nDifference')
                    
                    # Add colorbars for each row
                plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)
                plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
                plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
                plt.colorbar(im4, ax=axes[row, 3], shrink=0.8)
            
            plt.tight_layout()
            plt.savefig(op.join(self.plots_dir, f'mne_topomaps_{plot_title.lower().replace(" ", "_").replace("-", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_count += 1
            print(f"    ✅ Created {plot_title} plot with {len(marker_indices)} markers")
            
        print(f"  ✅ Created {plot_count} MNE topomap plots covering {n_markers} markers total")
        
        # Add custom 3-column plot for specific biomarkers
        self._create_custom_biomarker_topomap(topos_orig_mean, topos_recon_mean, info, marker_names, sphere, outlines)
        self._create_custom_biomarker_topomap_new(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # Add custom biomarker plot with relative difference scaling
        self._create_custom_biomarker_topomap_wilcoxon(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # Add custom FULL biomarker plot with symmetric difference and no FDR
        self._create_custom_full_biomarker_topomap_wilcoxon(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # Add custom FULL biomarker plot with Spearman correlation test
        self._create_custom_full_biomarker_topomap_spearman(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # Add custom FULL biomarker plot with symmetric difference and FDR correction
        self._create_custom_full_biomarker_topomap_wilcoxon_corrected(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # Add diagnosis-specific biomarker plots (3 columns: Original, Reconstructed, Difference)
        self._create_diagnosis_filtered_biomarker_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                           diagnosis_group=['UWS', 'VS'], sphere=sphere, outlines=outlines)
        self._create_diagnosis_filtered_biomarker_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                           diagnosis_group=['MCS+', 'MCS-'], sphere=sphere, outlines=outlines)
        
        # Add Wilcoxon distribution plots
        self._plot_wilcoxon_distributions(marker_names, topos_orig_all, topos_recon_all)

        print(f"  ✅ Created custom biomarker topographic plots")



    def _create_custom_biomarker_topomap(self, topos_orig_mean, topos_recon_mean, info, marker_names, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for specific biomarkers.
        
        Layout: Original | Reconstructed | Difference (Relative Scale) | Difference (Original Scale)
        Rows: Alpha-norm, Beta-norm, MMN, PermEntropy, Kolmogorov, P3b, SMI, CNV
        """
        print("  🎯 Creating custom biomarker topographic plots...")
        
        # Define specific biomarkers to plot
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'), 
            ('TimeLockedContrast_mmn', 'MMN'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('TimeLockedTopography_p3b', 'P3b'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation'),
            ('ContingentNegativeVariation_default', 'CNV')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Remove the main title (as requested)
        # fig.suptitle() - NOT ADDED
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference\n(Original Scale)', 'Difference\n(Relative Scale)']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference (relative)
            diff_abs_max = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_relative_min, diff_relative_max = -diff_abs_max, diff_abs_max
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed  
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            # Column 3: Difference (Original Scale - same as orig/recon)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles

            # Column 4: Difference (Relative Scale - symmetric around 0)
            im4, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 3],
                                         vlim=(diff_relative_min, diff_relative_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 3].set_title('')  # Remove individual titles
            
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for each column
            plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)
            plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)  
            plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            plt.colorbar(im4, ax=axes[row, 3], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'custom_biomarkers_orig_recon_diff.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom biomarker plot saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")

    def _create_custom_biomarker_topomap_new(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for specific biomarkers.
        
        Layout: Original | Reconstructed | Difference (Original Scale) | Wilcoxon Test
        Rows: Alpha-norm, Beta-norm, MMN, PermEntropy, Kolmogorov, P3b, SMI, CNV
        """
        print("  🎯 Creating custom biomarker topographic plots...")
        
        # Define specific biomarkers to plot
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'), 
            ('TimeLockedContrast_mmn', 'MMN'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('TimeLockedTopography_p3b', 'P3b'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation'),
            ('ContingentNegativeVariation_default', 'CNV')
        ]
        
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Compute Wilcoxon tests for the selected biomarkers if data is available
        wilcoxon_p_values = {}
        wilcoxon_p_values_corrected = {}
        if topos_orig_all is not None and topos_recon_all is not None:
            print(f"  🔬 Computing Wilcoxon signed-rank tests for selected biomarkers...")
            from scipy.stats import wilcoxon as wilcoxon_test
            
            n_subjects = topos_orig_all.shape[0]
            n_channels = topos_orig_all.shape[2]
            
            for marker_idx in biomarker_indices:
                p_values = np.zeros(n_channels)
                
                # Test each channel independently
                for channel_idx in range(n_channels):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, channel_idx]
                    recon_channel = topos_recon_all[:, marker_idx, channel_idx]
                    
                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                    orig_valid = orig_channel[valid_mask]
                    recon_valid = recon_channel[valid_mask]
                    
                    if len(orig_valid) >= 5:  # Need minimum samples for reliable test
                        try:
                            _, p_value = wilcoxon_test(orig_valid, recon_valid, alternative='two-sided')
                            p_values[channel_idx] = p_value
                        except:
                            p_values[channel_idx] = np.nan
                    else:
                        p_values[channel_idx] = np.nan
                
                wilcoxon_p_values[marker_idx] = p_values
            
            print(f"  ✅ Wilcoxon tests computed for {len(wilcoxon_p_values)} biomarkers")
            
            '''
            # Apply FDR correction (Benjamini-Hochberg) to p-values for each biomarker
            print(f"  🔬 Applying FDR correction (Benjamini-Hochberg method)...")
            
            # Try to import FDR correction function
            try:
                # Try scipy.stats.false_discovery_control (scipy >= 1.10)
                from scipy.stats import false_discovery_control
                use_scipy_fdr = True
            except ImportError:
                # Fallback to statsmodels
                try:
                    from statsmodels.stats.multitest import multipletests
                    use_scipy_fdr = False
                    print("     ℹ️  Using statsmodels for FDR correction")
                except ImportError:
                    print("     ⚠️  Warning: Could not import FDR correction. Using uncorrected p-values.")
                    wilcoxon_p_values_corrected = wilcoxon_p_values.copy()
                    use_scipy_fdr = None
            
            if use_scipy_fdr is not None:
                for marker_idx in biomarker_indices:
                    p_values = wilcoxon_p_values[marker_idx]
                    
                    # Identify valid (non-NaN) p-values
                    valid_mask = np.isfinite(p_values)
                    valid_p_values = p_values[valid_mask]
                    
                    if len(valid_p_values) > 0:
                        # Apply FDR correction
                        if use_scipy_fdr:
                            # scipy.stats.false_discovery_control
                            corrected_valid = false_discovery_control(valid_p_values, method='bh')
                        else:
                            # statsmodels.stats.multitest.multipletests
                            reject, corrected_valid, _, _ = multipletests(valid_p_values, method='fdr_bh')
                        
                        # Create corrected p-values array with NaNs preserved
                        p_values_corrected = np.full_like(p_values, np.nan)
                        p_values_corrected[valid_mask] = corrected_valid
                        
                        wilcoxon_p_values_corrected[marker_idx] = p_values_corrected
                    else:
                        # All NaN, keep as is
                        wilcoxon_p_values_corrected[marker_idx] = p_values.copy()
                
                print(f"  ✅ FDR correction applied to {len(wilcoxon_p_values_corrected)} biomarkers")
            '''
        # Create figure: 4 columns x n_biomarkers rows
        n_cols = 4 if wilcoxon_p_values else 3
        fig, axes = plt.subplots(n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Remove the main title (as requested)
        # fig.suptitle() - NOT ADDED
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon'] if wilcoxon_p_values else ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=38)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference (relative)
            diff_abs_max = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_relative_min, diff_relative_max = -diff_abs_max, diff_abs_max
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed  
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Original Scale - same as orig/recon)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Column 4: Wilcoxon Test p-values (FDR-corrected if available)
            im4 = None
            if wilcoxon_p_values_corrected and marker_idx in wilcoxon_p_values_corrected:
                # Use FDR-corrected p-values
                p_values = wilcoxon_p_values_corrected[marker_idx]
                # Convert p-values to discrete categories
                p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
                
                # Create discrete values based on p-value thresholds:
                # p < 0.01 -> 0 (black)
                # p < 0.05 -> 1 (gray)
                # p >= 0.05 -> 2 (white)
                discrete_values = np.zeros_like(p_values_clean)
                discrete_values[p_values_clean < 0.01] = 0  # Black
                discrete_values[(p_values_clean >= 0.01) & (p_values_clean < 0.05)] = 1  # Gray
                discrete_values[p_values_clean >= 0.05] = 2  # White
                
                # Create custom discrete colormap
                from matplotlib.colors import ListedColormap
                colors_map = ['black', 'gray', 'white']
                discrete_cmap = ListedColormap(colors_map)
                
                im4, _ = mne.viz.plot_topomap(discrete_values, info, axes=axes[row, 3],
                                             vlim=(0, 2),  # 0=black, 1=gray, 2=white
                                             show=False, cmap=discrete_cmap,
                                             sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
                axes[row, 3].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for each column
         #   plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)
         #   plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)  
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=18)
            
            # No colorbar for Wilcoxon column (discrete values with legend instead)
        
        plt.tight_layout()
        
        # Add legend for the Wilcoxon column
        if wilcoxon_p_values_corrected:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='gray', edgecolor='black', label='0.01 ≤ p < 0.05'),
                Patch(facecolor='white', edgecolor='black', label='p ≥ 0.05')
            ]
            # Add legend to the right of the plot
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=25, 
                      title='Wilcoxon p-value', title_fontsize=25)
        
        plt.savefig(op.join(self.plots_dir, 'custom_biomarkers_orig_recon_diff_new.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom biomarker plot saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")
    
    def _create_custom_biomarker_topomap_wilcoxon(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for specific biomarkers with relative difference scaling.
        
        Layout: Original | Reconstructed (with cbar) | Difference (Relative Scale with cbar) | Wilcoxon Test
        Rows: Alpha-norm, Beta-norm, MMN, PermEntropy, Kolmogorov, P3b, SMI, CNV
        """
        print("  🎯 Creating custom biomarker topographic plots with relative difference...")
        
        # Define specific biomarkers to plot
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'), 
            ('TimeLockedContrast_mmn', 'MMN'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('TimeLockedTopography_p3b', 'P3b'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation'),
            ('ContingentNegativeVariation_default', 'CNV')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Compute Wilcoxon tests for the selected biomarkers if data is available
        wilcoxon_p_values = {}
        wilcoxon_p_values_corrected = {}
        if topos_orig_all is not None and topos_recon_all is not None:
            print(f"  🔬 Computing Wilcoxon signed-rank tests for selected biomarkers...")
            from scipy.stats import wilcoxon as wilcoxon_test
            
            n_subjects = topos_orig_all.shape[0]
            n_channels = topos_orig_all.shape[2]
            
            for marker_idx in biomarker_indices:
                p_values = np.zeros(n_channels)
                
                # Test each channel independently
                for channel_idx in range(n_channels):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, channel_idx]
                    recon_channel = topos_recon_all[:, marker_idx, channel_idx]
                    
                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                    orig_valid = orig_channel[valid_mask]
                    recon_valid = recon_channel[valid_mask]
                    
                    if len(orig_valid) >= 5:  # Need minimum samples for reliable test
                        try:
                            _, p_value = wilcoxon_test(orig_valid, recon_valid, alternative='two-sided')
                            p_values[channel_idx] = p_value
                        except:
                            p_values[channel_idx] = np.nan
                    else:
                        p_values[channel_idx] = np.nan
                
                wilcoxon_p_values[marker_idx] = p_values
            
            print(f"  ✅ Wilcoxon tests computed for {len(wilcoxon_p_values)} biomarkers")
            
            # Apply FDR correction (Benjamini-Hochberg) to p-values for each biomarker
            print(f"  🔬 Applying FDR correction (Benjamini-Hochberg method)...")
            
            # Try to import FDR correction function
            try:
                # Try scipy.stats.false_discovery_control (scipy >= 1.10)
                from scipy.stats import false_discovery_control
                use_scipy_fdr = True
            except ImportError:
                # Fallback to statsmodels
                try:
                    from statsmodels.stats.multitest import multipletests
                    use_scipy_fdr = False
                    print("     ℹ️  Using statsmodels for FDR correction")
                except ImportError:
                    print("     ⚠️  Warning: Could not import FDR correction. Using uncorrected p-values.")
                    wilcoxon_p_values_corrected = wilcoxon_p_values.copy()
                    use_scipy_fdr = None
            
            if use_scipy_fdr is not None:
                for marker_idx in biomarker_indices:
                    p_values = wilcoxon_p_values[marker_idx]
                    
                    # Identify valid (non-NaN) p-values
                    valid_mask = np.isfinite(p_values)
                    valid_p_values = p_values[valid_mask]
                    
                    if len(valid_p_values) > 0:
                        # Apply FDR correction
                        if use_scipy_fdr:
                            # scipy.stats.false_discovery_control
                            corrected_valid = false_discovery_control(valid_p_values, method='bh')
                        else:
                            # statsmodels.stats.multitest.multipletests
                            reject, corrected_valid, _, _ = multipletests(valid_p_values, method='fdr_bh')
                        
                        # Create corrected p-values array with NaNs preserved
                        p_values_corrected = np.full_like(p_values, np.nan)
                        p_values_corrected[valid_mask] = corrected_valid
                        
                        wilcoxon_p_values_corrected[marker_idx] = p_values_corrected
                    else:
                        # All NaN, keep as is
                        wilcoxon_p_values_corrected[marker_idx] = p_values.copy()
                
                print(f"  ✅ FDR correction applied to {len(wilcoxon_p_values_corrected)} biomarkers")
        
        # Create figure: 4 columns x n_biomarkers rows
        n_cols = 4 if wilcoxon_p_values else 3
        fig, axes = plt.subplots(n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon'] if wilcoxon_p_values else ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=32)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Relative scale for difference: use actual min/max of difference
            diff_min = np.min(diff_data)
            diff_max = np.max(diff_data)
            
            # Column 1: Original (no colorbar)
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed (WITH colorbar)
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Relative Scale with viridis, WITH colorbar)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_min, diff_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Column 4: Wilcoxon Test p-values (FDR-corrected if available)
            im4 = None
            if wilcoxon_p_values_corrected and marker_idx in wilcoxon_p_values_corrected:
                # Use FDR-corrected p-values
                p_values = wilcoxon_p_values_corrected[marker_idx]
                # Convert p-values to discrete categories
                p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
                
                # Create discrete values based on p-value thresholds:
                # p < 0.01 -> 0 (black)
                # p < 0.05 -> 1 (gray)
                # p >= 0.05 -> 2 (white)
                discrete_values = np.zeros_like(p_values_clean)
                discrete_values[p_values_clean < 0.01] = 0  # Black
                discrete_values[(p_values_clean >= 0.01) & (p_values_clean < 0.05)] = 1  # Gray
                discrete_values[p_values_clean >= 0.05] = 2  # White
                
                # Create custom discrete colormap
                from matplotlib.colors import ListedColormap
                colors_map = ['black', 'gray', 'white']
                discrete_cmap = ListedColormap(colors_map)
                
                im4, _ = mne.viz.plot_topomap(discrete_values, info, axes=axes[row, 3],
                                             vlim=(0, 2),  # 0=black, 1=gray, 2=white
                                             show=False, cmap=discrete_cmap,
                                             sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
                axes[row, 3].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
            cbar2.ax.tick_params(labelsize=18)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=18)
            
            # No colorbar for Original or Wilcoxon columns
        
        plt.tight_layout()
        
        # Add legend for the Wilcoxon column
        if wilcoxon_p_values_corrected:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='gray', edgecolor='black', label='0.01 ≤ p < 0.05'),
                Patch(facecolor='white', edgecolor='black', label='p ≥ 0.05')
            ]
            # Add legend to the right of the plot
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=20, 
                      title='Wilcoxon p-value\n(FDR corrected)', title_fontsize=22)
        
        plt.savefig(op.join(self.plots_dir, 'custom_biomarkers_orig_recon_wilcoxon.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom biomarker plot (wilcoxon version) saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")
    
    def _create_custom_full_biomarker_topomap_wilcoxon(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for extended biomarker set with symmetric difference scaling.
        
        Layout: Original | Reconstructed (with cbar) | Difference (Symmetric RdBu_r with cbar) | Wilcoxon Test (NO FDR)
        Rows: Alpha-norm, Beta-norm, Delta-norm, Gamma-norm, Theta-norm, PermEntropy, Kolmogorov, SpectralEntropy, SMI
        """
        print("  🎯 Creating custom FULL biomarker topographic plots...")
        
        # Define specific biomarkers to plot (9 markers)
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'),
            ('PowerSpectralDensity_deltan', 'Delta Normalized'),
            ('PowerSpectralDensity_gamman', 'Gamma Normalized'),
            ('PowerSpectralDensity_thetan', 'Theta Normalized'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('PowerSpectralDensitySummary_summary_se', 'Spectral\nEntropy'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Compute Wilcoxon tests for the selected biomarkers if data is available
        # NO FDR CORRECTION - use uncorrected p-values
        wilcoxon_p_values = {}
        if topos_orig_all is not None and topos_recon_all is not None:
            print(f"  🔬 Computing Wilcoxon signed-rank tests for selected biomarkers (NO FDR correction)...")
            from scipy.stats import wilcoxon as wilcoxon_test
            
            n_subjects = topos_orig_all.shape[0]
            n_channels = topos_orig_all.shape[2]
            
            for marker_idx in biomarker_indices:
                p_values = np.zeros(n_channels)
                
                # Test each channel independently
                for channel_idx in range(n_channels):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, channel_idx]
                    recon_channel = topos_recon_all[:, marker_idx, channel_idx]
                    
                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                    orig_valid = orig_channel[valid_mask]
                    recon_valid = recon_channel[valid_mask]
                    
                    if len(orig_valid) >= 5:  # Need minimum samples for reliable test
                        try:
                            _, p_value = wilcoxon_test(orig_valid, recon_valid, alternative='two-sided')
                            p_values[channel_idx] = p_value
                        except:
                            p_values[channel_idx] = np.nan
                    else:
                        p_values[channel_idx] = np.nan
                
                wilcoxon_p_values[marker_idx] = p_values
            
            print(f"  ✅ Wilcoxon tests computed for {len(wilcoxon_p_values)} biomarkers")
        
        # Create figure: 4 columns x n_biomarkers rows
        n_cols = 4 if wilcoxon_p_values else 3
        fig, axes = plt.subplots(n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon'] if wilcoxon_p_values else ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=40)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference: use largest absolute value
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original (no colorbar)
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed (WITH colorbar)
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Symmetric RdBu_r with white at 0, WITH colorbar)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Column 4: Wilcoxon Test p-values (NO FDR correction)
            im4 = None
            if wilcoxon_p_values and marker_idx in wilcoxon_p_values:
                p_values = wilcoxon_p_values[marker_idx]
                # Convert p-values to discrete categories
                p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
                
                # Create discrete values based on p-value thresholds:
                # p < 0.01 -> 0 (black)
                # p < 0.05 -> 1 (gray)
                # p >= 0.05 -> 2 (white)
                discrete_values = np.zeros_like(p_values_clean)
                discrete_values[p_values_clean < 0.01] = 0  # Black
                discrete_values[(p_values_clean >= 0.01) & (p_values_clean < 0.05)] = 1  # Gray
                discrete_values[p_values_clean >= 0.05] = 2  # White
                
                # Create custom discrete colormap
                from matplotlib.colors import ListedColormap
                colors_map = ['black', 'gray', 'white']
                discrete_cmap = ListedColormap(colors_map)
                
                im4, _ = mne.viz.plot_topomap(discrete_values, info, axes=axes[row, 3],
                                             vlim=(0, 2),  # 0=black, 1=gray, 2=white
                                             show=False, cmap=discrete_cmap,
                                             sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
                axes[row, 3].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=40,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
            cbar2.ax.tick_params(labelsize=24)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=24)
            
            # No colorbar for Original or Wilcoxon columns
        
        plt.tight_layout()
        
        # Add legend for the Wilcoxon column
        if wilcoxon_p_values:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='gray', edgecolor='black', label='0.01 ≤ p < 0.05'),
                Patch(facecolor='white', edgecolor='black', label='p ≥ 0.05')
            ]
            # Add legend to the right of the plot
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=26, 
                      title='Wilcoxon p-value', title_fontsize=28)
        
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_wilcoxon.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")
    
    def _create_custom_full_biomarker_topomap_spearman(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for extended biomarker set with symmetric difference scaling.
        
        Layout: Original | Reconstructed (with cbar) | Difference (Symmetric RdBu_r with cbar) | Spearman Test
        Rows: Alpha-norm, Beta-norm, Delta-norm, Gamma-norm, Theta-norm, PermEntropy, Kolmogorov, SpectralEntropy, SMI
        """
        print("  🎯 Creating custom FULL biomarker topographic plots (Spearman)...")
        
        # Define specific biomarkers to plot (9 markers)
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'),
            ('PowerSpectralDensity_deltan', 'Delta Normalized'),
            ('PowerSpectralDensity_gamman', 'Gamma Normalized'),
            ('PowerSpectralDensity_thetan', 'Theta Normalized'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('PowerSpectralDensitySummary_summary_se', 'Spectral\nEntropy'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Compute Spearman correlation tests for the selected biomarkers if data is available
        spearman_p_values = {}
        if topos_orig_all is not None and topos_recon_all is not None:
            print(f"  🔬 Computing Spearman correlation tests for selected biomarkers...")
            from scipy.stats import spearmanr
            
            n_subjects = topos_orig_all.shape[0]
            n_channels = topos_orig_all.shape[2]
            
            for marker_idx in biomarker_indices:
                p_values = np.zeros(n_channels)
                
                # Test each channel independently
                for channel_idx in range(n_channels):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, channel_idx]
                    recon_channel = topos_recon_all[:, marker_idx, channel_idx]
                    
                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                    orig_valid = orig_channel[valid_mask]
                    recon_valid = recon_channel[valid_mask]
                    
                    if len(orig_valid) >= 5:  # Need minimum samples for reliable test
                        try:
                            _, p_value = spearmanr(orig_valid, recon_valid)
                            p_values[channel_idx] = p_value
                        except:
                            p_values[channel_idx] = np.nan
                    else:
                        p_values[channel_idx] = np.nan
                
                spearman_p_values[marker_idx] = p_values
            
            print(f"  ✅ Spearman tests computed for {len(spearman_p_values)} biomarkers")
        
        # Create figure: 4 columns x n_biomarkers rows
        n_cols = 4 if spearman_p_values else 3
        fig, axes = plt.subplots(n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Spearman'] if spearman_p_values else ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=40)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference: use largest absolute value
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original (no colorbar)
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed (WITH colorbar)
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Symmetric RdBu_r with white at 0, WITH colorbar)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Column 4: Spearman Test p-values
            im4 = None
            if spearman_p_values and marker_idx in spearman_p_values:
                p_values = spearman_p_values[marker_idx]
                # Convert p-values to discrete categories
                p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
                
                # Create discrete values based on p-value thresholds:
                # p < 0.01 -> 0 (black)
                # p < 0.05 -> 1 (gray)
                # p >= 0.05 -> 2 (white)
                discrete_values = np.zeros_like(p_values_clean)
                discrete_values[p_values_clean < 0.01] = 0  # Black
                discrete_values[(p_values_clean >= 0.01) & (p_values_clean < 0.05)] = 1  # Gray
                discrete_values[p_values_clean >= 0.05] = 2  # White
                
                # Create custom discrete colormap
                from matplotlib.colors import ListedColormap
                colors_map = ['black', 'gray', 'white']
                discrete_cmap = ListedColormap(colors_map)
                
                im4, _ = mne.viz.plot_topomap(discrete_values, info, axes=axes[row, 3],
                                             vlim=(0, 2),  # 0=black, 1=gray, 2=white
                                             show=False, cmap=discrete_cmap,
                                             sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
                axes[row, 3].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=40,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
            cbar2.ax.tick_params(labelsize=24)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=24)
            
            # No colorbar for Original or Spearman columns
        
        plt.tight_layout()
        
        # Add legend for the Spearman column
        if spearman_p_values:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='gray', edgecolor='black', label='0.01 ≤ p < 0.05'),
                Patch(facecolor='white', edgecolor='black', label='p ≥ 0.05')
            ]
            # Add legend to the right of the plot
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=26, 
                      title='Spearman p-value', title_fontsize=28)
        
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_spearman.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot (Spearman) saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")
    
    def _create_custom_full_biomarker_topomap_wilcoxon_corrected(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """
        Create custom 4-column topographic plot for extended biomarker set with symmetric difference scaling and FDR correction.
        
        Layout: Original | Reconstructed (with cbar) | Difference (Symmetric RdBu_r with cbar) | Wilcoxon Test (WITH FDR)
        Rows: Alpha-norm, Beta-norm, Delta-norm, Gamma-norm, Theta-norm, PermEntropy, Kolmogorov, SpectralEntropy, SMI
        """
        print("  🎯 Creating custom FULL biomarker topographic plots (FDR corrected)...")
        
        # Define specific biomarkers to plot (9 markers)
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'),
            ('PowerSpectralDensity_deltan', 'Delta Normalized'),
            ('PowerSpectralDensity_gamman', 'Gamma Normalized'),
            ('PowerSpectralDensity_thetan', 'Theta Normalized'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('PowerSpectralDensitySummary_summary_se', 'Spectral\nEntropy'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        available_biomarkers = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                available_biomarkers.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Compute Wilcoxon tests for the selected biomarkers if data is available
        wilcoxon_p_values = {}
        wilcoxon_p_values_corrected = {}
        if topos_orig_all is not None and topos_recon_all is not None:
            print(f"  🔬 Computing Wilcoxon signed-rank tests for selected biomarkers...")
            from scipy.stats import wilcoxon as wilcoxon_test
            
            n_subjects = topos_orig_all.shape[0]
            n_channels = topos_orig_all.shape[2]
            
            for marker_idx in biomarker_indices:
                p_values = np.zeros(n_channels)
                
                # Test each channel independently
                for channel_idx in range(n_channels):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, channel_idx]
                    recon_channel = topos_recon_all[:, marker_idx, channel_idx]
                    
                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                    orig_valid = orig_channel[valid_mask]
                    recon_valid = recon_channel[valid_mask]
                    
                    if len(orig_valid) >= 5:  # Need minimum samples for reliable test
                        try:
                            _, p_value = wilcoxon_test(orig_valid, recon_valid, alternative='two-sided')
                            p_values[channel_idx] = p_value
                        except:
                            p_values[channel_idx] = np.nan
                    else:
                        p_values[channel_idx] = np.nan
                
                wilcoxon_p_values[marker_idx] = p_values
            
            print(f"  ✅ Wilcoxon tests computed for {len(wilcoxon_p_values)} biomarkers")
            
            # Apply FDR correction (Benjamini-Hochberg) to p-values for each biomarker
            print(f"  🔬 Applying FDR correction (Benjamini-Hochberg method)...")
            
            # Try to import FDR correction function
            try:
                # Try scipy.stats.false_discovery_control (scipy >= 1.10)
                from scipy.stats import false_discovery_control
                use_scipy_fdr = True
            except ImportError:
                # Fallback to statsmodels
                try:
                    from statsmodels.stats.multitest import multipletests
                    use_scipy_fdr = False
                    print("     ℹ️  Using statsmodels for FDR correction")
                except ImportError:
                    print("     ⚠️  Warning: Could not import FDR correction. Using uncorrected p-values.")
                    wilcoxon_p_values_corrected = wilcoxon_p_values.copy()
                    use_scipy_fdr = None
            
            if use_scipy_fdr is not None:
                for marker_idx in biomarker_indices:
                    p_values = wilcoxon_p_values[marker_idx]
                    
                    # Identify valid (non-NaN) p-values
                    valid_mask = np.isfinite(p_values)
                    valid_p_values = p_values[valid_mask]
                    
                    if len(valid_p_values) > 0:
                        # Apply FDR correction
                        if use_scipy_fdr:
                            # scipy.stats.false_discovery_control
                            corrected_valid = false_discovery_control(valid_p_values, method='bh')
                        else:
                            # statsmodels.stats.multitest.multipletests
                            reject, corrected_valid, _, _ = multipletests(valid_p_values, method='fdr_bh')
                        
                        # Create corrected p-values array with NaNs preserved
                        p_values_corrected = np.full_like(p_values, np.nan)
                        p_values_corrected[valid_mask] = corrected_valid
                        
                        wilcoxon_p_values_corrected[marker_idx] = p_values_corrected
                    else:
                        # All NaN, keep as is
                        wilcoxon_p_values_corrected[marker_idx] = p_values.copy()
                
                print(f"  ✅ FDR correction applied to {len(wilcoxon_p_values_corrected)} biomarkers")
        
        # Create figure: 4 columns x n_biomarkers rows
        n_cols = 4 if wilcoxon_p_values else 3
        fig, axes = plt.subplots(n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon'] if wilcoxon_p_values else ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=32)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference: use largest absolute value
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original (no colorbar)
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed (WITH colorbar)
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Symmetric RdBu_r with white at 0, WITH colorbar)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Column 4: Wilcoxon Test p-values (FDR-corrected)
            im4 = None
            if wilcoxon_p_values_corrected and marker_idx in wilcoxon_p_values_corrected:
                # Use FDR-corrected p-values
                p_values = wilcoxon_p_values_corrected[marker_idx]
                # Convert p-values to discrete categories
                p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
                
                # Create discrete values based on p-value thresholds:
                # p < 0.01 -> 0 (black)
                # p < 0.05 -> 1 (gray)
                # p >= 0.05 -> 2 (white)
                discrete_values = np.zeros_like(p_values_clean)
                discrete_values[p_values_clean < 0.01] = 0  # Black
                discrete_values[(p_values_clean >= 0.01) & (p_values_clean < 0.05)] = 1  # Gray
                discrete_values[p_values_clean >= 0.05] = 2  # White
                
                # Create custom discrete colormap
                from matplotlib.colors import ListedColormap
                colors_map = ['black', 'gray', 'white']
                discrete_cmap = ListedColormap(colors_map)
                
                im4, _ = mne.viz.plot_topomap(discrete_values, info, axes=axes[row, 3],
                                             vlim=(0, 2),  # 0=black, 1=gray, 2=white
                                             show=False, cmap=discrete_cmap,
                                             sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
                axes[row, 3].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side only
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)  # Horizontal text
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
            cbar2.ax.tick_params(labelsize=18)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=18)
            
            # No colorbar for Original or Wilcoxon columns
        
        plt.tight_layout()
        
        # Add legend for the Wilcoxon column
        if wilcoxon_p_values_corrected:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='gray', edgecolor='black', label='0.01 ≤ p < 0.05'),
                Patch(facecolor='white', edgecolor='black', label='p ≥ 0.05')
            ]
            # Add legend to the right of the plot
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=20, 
                      title='Wilcoxon p-value\n(FDR corrected)', title_fontsize=22)
        
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_wilcoxon_corrected.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot (FDR corrected) saved with {n_biomarkers} markers")
        print(f"    📍 Biomarkers included: {biomarker_labels}")
    
    def _create_diagnosis_filtered_biomarker_topomap(self, topos_orig_all, topos_recon_all, info, marker_names, 
                                                       diagnosis_group=None, sphere=None, outlines='head'):
        """
        Create 3-column topographic plot (Original, Reconstructed, Difference) filtered by diagnosis group.
        
        Similar to custom_biomarkers_orig_recon_wilcoxon_corrected but without the Wilcoxon test column,
        and filtered to include only subjects with specific diagnoses.
        
        Parameters
        ----------
        topos_orig_all : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data for all subjects
        topos_recon_all : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data for all subjects
        info : mne.Info
            MNE info object with montage
        marker_names : list
            List of marker names
        diagnosis_group : list
            List of diagnoses to include (e.g., ['UWS', 'VS'] or ['MCS+', 'MCS-'])
        sphere : tuple or str
            Sphere definition for plotting
        outlines : dict or str
            Outlines definition for plotting
        """
        if diagnosis_group is None or not self.patient_labels_original:
            print("  ⚠️  No diagnosis filtering available, skipping diagnosis-filtered plots")
            return
        
        group_name = '_'.join(diagnosis_group).replace('+', 'plus').replace('-', 'minus')
        print(f"  🎯 Creating diagnosis-filtered biomarker plots for {diagnosis_group}...")
        
        # Filter subjects by diagnosis
        subject_ids = self.global_topo_data['subjects']
        filtered_indices = []
        filtered_subject_ids = []
        
        for idx, subject_id in enumerate(subject_ids):
            original_diagnosis = self.patient_labels_original.get(subject_id)
            if original_diagnosis in diagnosis_group:
                filtered_indices.append(idx)
                filtered_subject_ids.append(subject_id)
        
        if len(filtered_indices) == 0:
            print(f"     ⚠️  No subjects found with diagnoses {diagnosis_group}")
            return
        
        print(f"     ✓ Found {len(filtered_indices)} subjects with diagnoses {diagnosis_group}")
        print(f"     ✓ Subjects: {filtered_subject_ids}")
        
        # Filter the topographic data
        topos_orig_filtered = topos_orig_all[filtered_indices]
        topos_recon_filtered = topos_recon_all[filtered_indices]
        
        # Compute mean across filtered subjects
        topos_orig_mean = np.mean(topos_orig_filtered, axis=0)  # (n_markers, n_channels)
        topos_recon_mean = np.mean(topos_recon_filtered, axis=0)
        
        # Define the 9 biomarkers to plot (same as wilcoxon_corrected)
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'),
            ('PowerSpectralDensity_deltan', 'Delta Normalized'),
            ('PowerSpectralDensity_gamman', 'Gamma Normalized'),
            ('PowerSpectralDensity_thetan', 'Theta Normalized'),
            ('PermutationEntropy_default', 'Permutation\nEntropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov\nComplexity'),
            ('PowerSpectralDensitySummary_summary_se', 'Spectral\nEntropy'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual\nInformation')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                print(f"       ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"       ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print(f"     ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"     📊 Creating plot for {n_biomarkers} biomarkers, {len(filtered_indices)} subjects")
        
        # Create figure: 3 columns x n_biomarkers rows (wider figure for larger topoplots)
        fig, axes = plt.subplots(n_biomarkers, 3, figsize=(20, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=32)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            # Data for this marker
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            # Use same scale for original and reconstructed
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference: use largest absolute value
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original (no colorbar)
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')  # Remove individual titles
            
            # Column 2: Reconstructed (WITH colorbar)
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')  # Remove individual titles
            
            # Column 3: Difference (Symmetric RdBu_r, WITH colorbar)
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')  # Remove individual titles
            
            # Add row label (biomarker name) on the left side
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3 (smaller shrink to take less space)
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.7, aspect=20)
            cbar2.ax.tick_params(labelsize=16)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.7, aspect=20)
            cbar3.ax.tick_params(labelsize=16)
        
        # Use more generous spacing to prevent compression
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot with diagnosis group in filename
        filename = f'custom_biomarkers_orig_recon_diff_{group_name}.png'
        plt.savefig(op.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Diagnosis-filtered plot saved: {filename}")
        print(f"        Diagnoses: {diagnosis_group}")
        print(f"        N subjects: {len(filtered_indices)}")
        print(f"        Biomarkers: {biomarker_labels}")
    
    def _plot_wilcoxon_distributions(self, marker_names, topos_orig_all=None, topos_recon_all=None):
        """
        Plot distributions of differences passed to Wilcoxon test.
        
        Creates 8 unique images (one for each biomarker) with 4 subplots each.
        Each subplot shows 64 curves (electrodes), with:
        - X-axis: subject id
        - Y-axis: difference (recon - orig) for that subject and electrode
        
        Parameters
        ----------
        marker_names : list
            List of marker names
        topos_orig_all : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon_all : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        """
        print("  📊 Creating Wilcoxon distribution plots...")
        
        if topos_orig_all is None or topos_recon_all is None:
            print("     ⚠️  No topographic data available. Skipping Wilcoxon distribution plots.")
            return
        
        # Define the 8 biomarkers to plot (same as custom_biomarkers_orig_recon_diff_new.png)
        biomarker_specs = [
            ('PowerSpectralDensity_alphan', 'Alpha Normalized'),
            ('PowerSpectralDensity_betan', 'Beta Normalized'), 
            ('TimeLockedContrast_mmn', 'MMN'),
            ('PermutationEntropy_default', 'Permutation Entropy'),
            ('KolmogorovComplexity_default', 'Kolmogorov Complexity'),
            ('TimeLockedTopography_p3b', 'P3b'),
            ('SymbolicMutualInformation_weighted', 'Symbolic Mutual Information'),
            ('ContingentNegativeVariation_default', 'CNV')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        biomarker_names_full = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
                biomarker_names_full.append(marker_name)
                print(f"    ✅ Found {display_name}: marker index {idx}")
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_subjects = topos_orig_all.shape[0]
        n_channels = topos_orig_all.shape[2]
        subject_ids = np.arange(n_subjects)
        
        # Assume 256 channels (EGI256), split into 4 groups of 64
        channels_per_subplot = 64
        n_subplots = 4
        
        if n_channels != 256:
            print(f"     ⚠️  Expected 256 channels, got {n_channels}. Adjusting layout...")
            channels_per_subplot = n_channels // n_subplots
        
        # Create one figure per biomarker
        for marker_idx, display_name, marker_name_full in zip(biomarker_indices, biomarker_labels, biomarker_names_full):
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            axes = axes.flatten()
            
            fig.suptitle(f'Wilcoxon Test Distributions: {display_name}', fontsize=20, y=0.995)
            
            # For each subplot, plot 64 electrodes
            for subplot_idx in range(n_subplots):
                ax = axes[subplot_idx]
                
                # Determine channel range for this subplot
                start_ch = subplot_idx * channels_per_subplot
                end_ch = min(start_ch + channels_per_subplot, n_channels)
                
                # Plot each electrode as a separate curve
                for ch_idx in range(start_ch, end_ch):
                    # Get data for this marker and channel across all subjects
                    orig_channel = topos_orig_all[:, marker_idx, ch_idx]
                    recon_channel = topos_recon_all[:, marker_idx, ch_idx]
                    
                    # Compute difference: recon - orig (NOT orig - recon)
                    diff_channel = recon_channel - orig_channel
                    
                    # Plot the difference for each subject
                    ax.plot(subject_ids, diff_channel, alpha=0.3, linewidth=0.8)
                
                # Formatting
                ax.set_xlabel('Subject ID', fontsize=12)
                ax.set_ylabel('Difference (Recon - Orig)', fontsize=12)
                ax.set_title(f'Electrodes {start_ch}-{end_ch-1}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero difference')
                
                # Only add legend to first subplot
                if subplot_idx == 0:
                    ax.legend(loc='best', fontsize=10)
            
            plt.tight_layout()
            
            # Save figure
            filename = f'wilcoxon_distributions_{marker_name_full.replace("/", "_")}.png'
            save_path = op.join(self.plots_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    ✅ Saved: {filename}")
        
        print(f"  ✅ Created {len(biomarker_indices)} Wilcoxon distribution plots")
    
    def global_statistical_tests(self):
        """Perform global statistical tests."""
        print("Performing global statistical tests...")
        
        results = {}
        subjects = self.global_scalar_data['subjects']
        n_subjects = len(subjects)
        
        # ========== SCALAR TESTS ==========
        scalar_mses = self.global_scalar_data['mses']
        
        # Overall scalar MSE test (testing if MSE significantly different from 0)
        scalar_mse_t, scalar_mse_p = stats.ttest_1samp(scalar_mses, 0)
        
        results['scalar_tests'] = {
            'overall_mse_test': {
                'null_hypothesis': 'mean_mse = 0',
                'statistic': float(scalar_mse_t),
                'p_value': float(scalar_mse_p),
                'mean_mse': float(np.mean(scalar_mses)),
                'std_mse': float(np.std(scalar_mses))
            }
        }
        
        # Per-marker norm_sq_error tests for scalars
        marker_names = list(self.global_scalar_data['marker_data'].keys())
        per_marker_tests = {}
        
        for marker_name in marker_names:
            marker_norm_sq_errors = self.global_scalar_data['marker_data'][marker_name]['norm_sq_errors']
            if len(marker_norm_sq_errors) >= 2:  # Need at least 2 subjects for t-test
                t_stat, p_val = stats.ttest_1samp(marker_norm_sq_errors, 0)
                per_marker_tests[marker_name] = {
                    'norm_sq_error_test_statistic': float(t_stat),
                    'norm_sq_error_test_pvalue': float(p_val),
                    'mean_norm_sq_error': float(np.mean(marker_norm_sq_errors)),
                    'std_norm_sq_error': float(np.std(marker_norm_sq_errors)),
                    'n_subjects': len(marker_norm_sq_errors)
                }
        
        results['scalar_tests']['per_marker_tests'] = per_marker_tests
        
        # Per-subject norm_sq_error tests for scalars (across markers)
        per_subject_tests = {}
        for i, subject_id in enumerate(subjects):
            subject_norm_sq_errors = []
            for marker_name in marker_names:
                if subject_id in self.global_scalar_data['marker_data'][marker_name]['subjects']:
                    idx = self.global_scalar_data['marker_data'][marker_name]['subjects'].index(subject_id)
                    subject_norm_sq_errors.append(self.global_scalar_data['marker_data'][marker_name]['norm_sq_errors'][idx])
            
            if len(subject_norm_sq_errors) >= 2:  # Need at least 2 markers for t-test
                t_stat, p_val = stats.ttest_1samp(subject_norm_sq_errors, 0)
                per_subject_tests[subject_id] = {
                    'norm_sq_error_test_statistic': float(t_stat),
                    'norm_sq_error_test_pvalue': float(p_val),
                    'mean_norm_sq_error': float(np.mean(subject_norm_sq_errors)),
                    'std_norm_sq_error': float(np.std(subject_norm_sq_errors)),
                    'n_markers': len(subject_norm_sq_errors)
                }
        
        results['scalar_tests']['per_subject_tests'] = per_subject_tests
        
        # ========== TOPOGRAPHIC TESTS ==========
        topo_nmses = self.global_topo_data['nmses']
        
        # Overall topographic NMSE test
        topo_nmse_t, topo_nmse_p = stats.ttest_1samp(topo_nmses, 0)
        
        results['topo_tests'] = {
            'overall_nmse_test': {
                'null_hypothesis': 'mean_nmse = 0',
                'statistic': float(topo_nmse_t),
                'p_value': float(topo_nmse_p),
                'mean_nmse': float(np.mean(topo_nmses)),
                'std_nmse': float(np.std(topo_nmses))
            }
        }
        
        # Per-sensor and per-marker tests for topography
        if len(self.global_topo_data['topos_orig_all']) > 0:
            topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])
            topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
            n_markers = self.global_topo_data['n_markers']
            n_channels = self.global_topo_data['n_channels']
            
            # Per-sensor tests (MSE across markers, testing if mean MSE = 0)
            per_sensor_tests = {}
            for ch in range(n_channels):
                sensor_mses = []
                for s in range(n_subjects):
                    for m in range(n_markers):
                        mse = (topos_orig_all[s, m, ch] - topos_recon_all[s, m, ch]) ** 2
                        sensor_mses.append(mse)
                
                if len(sensor_mses) >= 2:
                    t_stat, p_val = stats.ttest_1samp(sensor_mses, 0)
                    per_sensor_tests[f'sensor_{ch}'] = {
                        'mse_test_statistic': float(t_stat),
                        'mse_test_pvalue': float(p_val),
                        'mean_mse': float(np.mean(sensor_mses)),
                        'std_mse': float(np.std(sensor_mses))
                    }
            
            results['topo_tests']['per_sensor_tests'] = per_sensor_tests
            
            # Per-marker tests (MSE across sensors, testing if mean MSE = 0)
            per_topo_marker_tests = {}
            topo_marker_names = [self.mapper.get_topo_name(i) for i in range(n_markers)]
            
            for m, marker_name in enumerate(topo_marker_names):
                marker_mses = []
                for s in range(n_subjects):
                    for ch in range(n_channels):
                        mse = (topos_orig_all[s, m, ch] - topos_recon_all[s, m, ch]) ** 2
                        marker_mses.append(mse)
                
                if len(marker_mses) >= 2:
                    t_stat, p_val = stats.ttest_1samp(marker_mses, 0)
                    per_topo_marker_tests[marker_name] = {
                        'mse_test_statistic': float(t_stat),
                        'mse_test_pvalue': float(p_val),
                        'mean_mse': float(np.mean(marker_mses)),
                        'std_mse': float(np.std(marker_mses))
                    }
            
            results['topo_tests']['per_marker_tests'] = per_topo_marker_tests
        
        # Save results
        with open(op.join(self.data_dir, 'global_statistical_tests.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-marker and per-sensor results as CSV for easy analysis
        if 'per_marker_tests' in results['scalar_tests']:
            scalar_marker_df = pd.DataFrame(results['scalar_tests']['per_marker_tests']).T
            scalar_marker_df.to_csv(op.join(self.data_dir, 'scalar_per_marker_tests.csv'))
        
        if 'per_subject_tests' in results['scalar_tests']:
            scalar_subject_df = pd.DataFrame(results['scalar_tests']['per_subject_tests']).T
            scalar_subject_df.to_csv(op.join(self.data_dir, 'scalar_per_subject_tests.csv'))
        
        if 'per_sensor_tests' in results['topo_tests']:
            topo_sensor_df = pd.DataFrame(results['topo_tests']['per_sensor_tests']).T
            topo_sensor_df.to_csv(op.join(self.data_dir, 'topo_per_sensor_tests.csv'))
        
        if 'per_marker_tests' in results['topo_tests']:
            topo_marker_df = pd.DataFrame(results['topo_tests']['per_marker_tests']).T
            topo_marker_df.to_csv(op.join(self.data_dir, 'topo_per_marker_tests.csv'))
        
        return results
    
    def create_statistical_tests_plots(self):
        """Create plots for statistical tests p-values."""
        print("Creating statistical tests plots...")
        
        subjects = self.global_scalar_data['subjects']
        n_subjects = len(subjects)
        
        # 1. SUBJECT-WISE TESTS (for each subject across all markers)
        self._create_subject_wise_statistical_tests_plot(subjects, n_subjects)
        
        # 2. MARKER-WISE TESTS (for each marker across all subjects)
        self._create_marker_wise_statistical_tests_plot(subjects, n_subjects)
    
    def _create_subject_wise_statistical_tests_plot(self, subjects, n_subjects):
        """Create statistical tests plot for each subject across markers."""
        print("  Creating subject-wise statistical tests plot...")
        
        # Collect p-values from individual subject analyses
        scalar_pvalues = {'paired_ttest': [], 'wilcoxon': [], 'one_sample_ttest': []}
        topo_pvalues = {'paired_ttest': [], 'wilcoxon': [], 'one_sample_ttest': []}
        
        for subject_id in subjects:
            data = self.subjects_data[subject_id]
            
            # Extract scalar statistical test results
            scalar_tests = data['summary']['scalar_statistical_tests']
            scalar_pvalues['paired_ttest'].append(scalar_tests['paired_ttest']['p_value'])
            scalar_pvalues['wilcoxon'].append(scalar_tests['wilcoxon']['p_value'])
            scalar_pvalues['one_sample_ttest'].append(scalar_tests['one_sample_ttest_on_differences']['p_value'])
            
            # Extract topographic statistical test results
            topo_tests = data['summary']['topographic_statistical_tests']
            topo_pvalues['paired_ttest'].append(topo_tests['paired_ttest']['p_value'])
            topo_pvalues['wilcoxon'].append(topo_tests['wilcoxon']['p_value'])
            topo_pvalues['one_sample_ttest'].append(topo_tests['one_sample_ttest_on_differences']['p_value'])
        
        # Create matrices
        test_names = ['Paired t-test', 'Wilcoxon', 'One-sample t-test']
        scalar_pvalue_matrix = np.array([
            scalar_pvalues['paired_ttest'],
            scalar_pvalues['wilcoxon'], 
            scalar_pvalues['one_sample_ttest']
        ])
        
        topo_pvalue_matrix = np.array([
            topo_pvalues['paired_ttest'],
            topo_pvalues['wilcoxon'],
            topo_pvalues['one_sample_ttest']
        ])
        
        # Find max p-value for colorbar scale
        max_pval_scalar = np.ceil(np.max(scalar_pvalue_matrix) * 10) / 10
        max_pval_topo = np.ceil(np.max(topo_pvalue_matrix) * 10) / 10
        max_pval_overall = max(max_pval_scalar, max_pval_topo)
        
        # Create figure - increased height
        fig, axes = plt.subplots(2, 1, figsize=(max(12, n_subjects * 0.8), 12))
        fig.suptitle('Statistical Tests P-values: Subject-wise Analysis\n(Tests performed for each subject across all markers)', fontsize=16)
        
        # Scalar tests heatmap
        im1 = axes[0].imshow(scalar_pvalue_matrix, aspect='auto', cmap='Reds', 
                            vmin=0, vmax=max_pval_overall)
        axes[0].set_title('Scalar Statistical Tests P-values')
        axes[0].set_xlabel('Subjects')
        axes[0].set_ylabel('Test Type')
        axes[0].set_xticks(range(n_subjects))
        axes[0].set_xticklabels(subjects, rotation=45)
        axes[0].set_yticks(range(len(test_names)))
        axes[0].set_yticklabels(test_names)
        axes[0].grid(False)
        
        # Add p-value text annotations
        for i in range(len(test_names)):
            for j in range(n_subjects):
                text = axes[0].text(j, i, f'{scalar_pvalue_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=axes[0])
        
        # Topographic tests heatmap
        im2 = axes[1].imshow(topo_pvalue_matrix, aspect='auto', cmap='Reds',
                            vmin=0, vmax=max_pval_overall)
        axes[1].set_title('Topographic Statistical Tests P-values')
        axes[1].set_xlabel('Subjects')
        axes[1].set_ylabel('Test Type')
        axes[1].set_xticks(range(n_subjects))
        axes[1].set_xticklabels(subjects, rotation=45)
        axes[1].set_yticks(range(len(test_names)))
        axes[1].set_yticklabels(test_names)
        axes[1].grid(False)
        
        # Add p-value text annotations
        for i in range(len(test_names)):
            for j in range(n_subjects):
                text = axes[1].text(j, i, f'{topo_pvalue_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'statistical_test_pvalues_subjects.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created subject-wise statistical tests plot")
        print(f"        Max scalar p-value: {max_pval_scalar:.3f}")
        print(f"        Max topo p-value: {max_pval_topo:.3f}")
    
    def _create_marker_wise_statistical_tests_plot(self, subjects, n_subjects):
        """Create statistical tests plot for each marker across subjects."""
        print("  Creating marker-wise statistical tests plot...")
        
        # Get marker names
        marker_names = list(self.global_scalar_data['marker_data'].keys())
        n_markers = len(marker_names)
        
        # Initialize containers for marker-wise tests
        scalar_marker_pvalues = {'paired_ttest': [], 'wilcoxon': [], 'one_sample_ttest': []}
        topo_marker_pvalues = {'paired_ttest': [], 'wilcoxon': [], 'one_sample_ttest': []}
        
        # For each marker, perform statistical tests across subjects
        for marker_name in marker_names:
            # Get original and reconstructed values for this marker across all subjects
            orig_vals = self.global_scalar_data['marker_data'][marker_name]['orig_vals']
            recon_vals = self.global_scalar_data['marker_data'][marker_name]['recon_vals']
            
            # Perform statistical tests for scalar data
            if len(orig_vals) >= 2:  # Need at least 2 subjects
                from scipy.stats import ttest_rel, wilcoxon, ttest_1samp
                
                # Paired t-test (original vs reconstructed for this marker)
                try:
                    _, p_paired = ttest_rel(orig_vals, recon_vals)
                    scalar_marker_pvalues['paired_ttest'].append(p_paired)
                except:
                    scalar_marker_pvalues['paired_ttest'].append(1.0)
                
                # Wilcoxon signed-rank test
                try:
                    _, p_wilcoxon = wilcoxon(orig_vals, recon_vals)
                    scalar_marker_pvalues['wilcoxon'].append(p_wilcoxon)
                except:
                    scalar_marker_pvalues['wilcoxon'].append(1.0)
                
                # One-sample t-test on differences
                try:
                    differences = np.array(orig_vals) - np.array(recon_vals)
                    _, p_one_sample = ttest_1samp(differences, 0)
                    scalar_marker_pvalues['one_sample_ttest'].append(p_one_sample)
                except:
                    scalar_marker_pvalues['one_sample_ttest'].append(1.0)
            else:
                # Not enough subjects for statistical tests
                scalar_marker_pvalues['paired_ttest'].append(1.0)
                scalar_marker_pvalues['wilcoxon'].append(1.0)
                scalar_marker_pvalues['one_sample_ttest'].append(1.0)
        
        # For topographic data, we need to extract per-marker data across subjects
        if len(self.global_topo_data['topos_orig_all']) > 0:
            topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])
            topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
            
            for marker_idx in range(min(n_markers, topos_orig_all.shape[1])):
                # Get data for this marker across all subjects and channels
                marker_orig_data = topos_orig_all[:, marker_idx, :].flatten()  # Flatten across subjects and channels
                marker_recon_data = topos_recon_all[:, marker_idx, :].flatten()
                
                # Perform statistical tests for topographic data
                if len(marker_orig_data) >= 2:
                    try:
                        _, p_paired = ttest_rel(marker_orig_data, marker_recon_data)
                        topo_marker_pvalues['paired_ttest'].append(p_paired)
                    except:
                        topo_marker_pvalues['paired_ttest'].append(1.0)
                    
                    try:
                        _, p_wilcoxon = wilcoxon(marker_orig_data, marker_recon_data)
                        topo_marker_pvalues['wilcoxon'].append(p_wilcoxon)
                    except:
                        topo_marker_pvalues['wilcoxon'].append(1.0)
                    
                    try:
                        differences = marker_orig_data - marker_recon_data
                        _, p_one_sample = ttest_1samp(differences, 0)
                        topo_marker_pvalues['one_sample_ttest'].append(p_one_sample)
                    except:
                        topo_marker_pvalues['one_sample_ttest'].append(1.0)
                else:
                    topo_marker_pvalues['paired_ttest'].append(1.0)
                    topo_marker_pvalues['wilcoxon'].append(1.0)
                    topo_marker_pvalues['one_sample_ttest'].append(1.0)
        else:
            # No topographic data available
            for _ in range(n_markers):
                topo_marker_pvalues['paired_ttest'].append(1.0)
                topo_marker_pvalues['wilcoxon'].append(1.0)
                topo_marker_pvalues['one_sample_ttest'].append(1.0)
        
        # Create matrices
        test_names = ['Paired t-test', 'Wilcoxon', 'One-sample t-test']
        scalar_marker_pvalue_matrix = np.array([
            scalar_marker_pvalues['paired_ttest'],
            scalar_marker_pvalues['wilcoxon'], 
            scalar_marker_pvalues['one_sample_ttest']
        ])
        
        topo_marker_pvalue_matrix = np.array([
            topo_marker_pvalues['paired_ttest'][:n_markers],
            topo_marker_pvalues['wilcoxon'][:n_markers],
            topo_marker_pvalues['one_sample_ttest'][:n_markers]
        ])
        
        # Find max p-value for colorbar scale
        max_pval_scalar_marker = np.ceil(np.max(scalar_marker_pvalue_matrix) * 10) / 10
        max_pval_topo_marker = np.ceil(np.max(topo_marker_pvalue_matrix) * 10) / 10
        max_pval_marker_overall = max(max_pval_scalar_marker, max_pval_topo_marker)
        
        # Create figure - increased height
        fig, axes = plt.subplots(2, 1, figsize=(max(15, n_markers * 0.6), 14))
        fig.suptitle('Statistical Tests P-values: Marker-wise Analysis\n(Tests performed for each marker across all subjects)', fontsize=16)
        
        # Scalar tests heatmap
        im1 = axes[0].imshow(scalar_marker_pvalue_matrix, aspect='auto', cmap='Reds', 
                            vmin=0, vmax=max_pval_marker_overall)
        axes[0].set_title('Scalar Statistical Tests P-values')
        axes[0].set_xlabel('Markers')
        axes[0].set_ylabel('Test Type')
        axes[0].set_xticks(range(n_markers))
        axes[0].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[0].set_yticks(range(len(test_names)))
        axes[0].set_yticklabels(test_names)
        axes[0].grid(False)
        
        # Add p-value text annotations
        for i in range(len(test_names)):
            for j in range(n_markers):
                # Adjust font size based on number of markers
                fontsize = max(6, min(9, 120 // n_markers))
                axes[0].text(j, i, f'{scalar_marker_pvalue_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=fontsize)
        
        plt.colorbar(im1, ax=axes[0])
        
        # Topographic tests heatmap
        im2 = axes[1].imshow(topo_marker_pvalue_matrix, aspect='auto', cmap='Reds',
                            vmin=0, vmax=max_pval_marker_overall)
        axes[1].set_title('Topographic Statistical Tests P-values')
        axes[1].set_xlabel('Markers')
        axes[1].set_ylabel('Test Type')
        axes[1].set_xticks(range(n_markers))
        axes[1].set_xticklabels(marker_names, rotation=45, ha='right')
        axes[1].set_yticks(range(len(test_names)))
        axes[1].set_yticklabels(test_names)
        axes[1].grid(False)
        
        # Add p-value text annotations
        for i in range(len(test_names)):
            for j in range(n_markers):
                # Adjust font size based on number of markers
                fontsize = max(6, min(9, 120 // n_markers))
                axes[1].text(j, i, f'{topo_marker_pvalue_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=fontsize)
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(op.join(self.plots_dir, 'statistical_test_pvalues_markers.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Created marker-wise statistical tests plot")
        print(f"        Max scalar p-value: {max_pval_scalar_marker:.3f}")
        print(f"        Max topo p-value: {max_pval_topo_marker:.3f}")
        print(f"        Number of markers tested: {n_markers}")
    
    def run_statistical_analysis(self):
        """
        Run statistical analysis using StatisticalAnalyzer.
        
        Creates statistics folder with results and plots subfolders.
        """
        print("🔍 Checking statistical analysis availability...")
        print(f"   HAS_STATISTICAL_ANALYSIS = {HAS_STATISTICAL_ANALYSIS}")
        print(f"   StatisticalAnalyzer = {StatisticalAnalyzer}")
        
        if not HAS_STATISTICAL_ANALYSIS:
            print("⚠️  StatisticalAnalyzer not available - skipping statistical analysis")
            print("   This means the Wilcoxon test and SSIM analysis will not run.")
            return None
        
        # Create statistics folder structure
        statistics_dir = op.join(self.output_dir, 'statistics')
        print(f"📁 Creating statistics directory: {statistics_dir}")
        os.makedirs(statistics_dir, exist_ok=True)
        
        try:
            # Initialize statistical analyzer with the statistics directory
            patient_labels_file = self.patient_labels_file
            print(f"📄 Using patient labels: {patient_labels_file}")
            
            # Create StatisticalAnalyzer
            print("🏗️  Initializing StatisticalAnalyzer...")
            stat_analyzer = StatisticalAnalyzer(
                results_dir=self.results_dir,
                output_dir=statistics_dir,
                patient_labels_file=patient_labels_file,
                data_dir=self.fif_data_dir
            )
            
            # Run the statistical analysis
            print("🔬 Running comprehensive statistical analysis...")
            print("   This includes:")
            print("   - Wilcoxon signed-rank tests for each marker")
            print("   - Structural Similarity Index (SSIM) computation")
            print("   - Analysis by patient groups (MCS, UWS, etc.)")
            
            stats_results = stat_analyzer.run_analysis(analyze_by_groups=True)
            
            if stats_results:
                print(f"✅ Statistical analysis completed successfully!")
                print(f"📊 Statistics plots saved in: {op.join(statistics_dir, 'plots')}")
                print(f"📈 Statistics results saved in: {op.join(statistics_dir, 'results')}")
                print(f"🔍 Found {len(stats_results)} analysis groups")
                
                # Show what was analyzed
                for group_name in stats_results.keys():
                    print(f"   ✓ Group '{group_name}' analyzed")
                    
            else:
                print("⚠️  Statistical analysis returned no results")
            
            return stats_results
            
        except Exception as e:
            print(f"❌ Statistical analysis failed: {e}")
            import traceback
            print("📋 Full error traceback:")
            traceback.print_exc()
            return None
    
    def run_analysis(self):
        """Run complete global analysis."""
        print("=" * 60)
        print("GLOBAL ANALYSIS ACROSS SUBJECTS")
        print("=" * 60)
        
        # Collect data
        subjects = self.collect_subject_data()
        if subjects is None or len(subjects) < 1:
            print("⚠️  WARNING: No subjects found for analysis")
            return {
                'subjects': [],
                'scalar_data': {},
                'topo_data': {},
                'statistical_tests': {},
                'target_state': self.target_state,
                'available_states': sorted(self.available_states) if self.available_states else None
            }
        elif len(subjects) < 2:
            print(f"⚠️  WARNING: Only {len(subjects)} subject found. Some analyses may be limited.")
            # Continue with single subject analysis
        
        # Prepare data structures
        self.prepare_global_data()
        
        # Create plots
      #  self.create_scalar_global_plots()
        # self.create_topographic_global_plots()  # Commented out for now
        
        # Create time series error plots
      #  self.create_timeseries_error_plots()
        
        # Create MNE topographic plots
        self.create_mne_topomap_plots()
        
        # Create Global Field Power plots (optional)
        if not self.skip_gfp:
            print("\n--- Global Field Power Analysis ---")
            gfp_analyzer = GlobalFieldPowerGlobal(self.output_dir, self.subjects_data, self.results_dir, self.fif_data_dir)
            gfp_analyzer.analyze_all_subjects()
        else:
            print("\n--- Global Field Power Analysis (SKIPPED) ---")
            print("🚫 Global Field Power analysis skipped as requested")
        
        # Statistical Analysis - NEW integrated approach
        print("\n--- Statistical Analysis ---")
        print("🔍 Debug: About to call run_statistical_analysis()")
        print(f"   HAS_STATISTICAL_ANALYSIS = {HAS_STATISTICAL_ANALYSIS}")
        print(f"   StatisticalAnalyzer = {StatisticalAnalyzer}")
        stats_results = self.run_statistical_analysis()
        print(f"   Returned stats_results = {type(stats_results)} with keys: {list(stats_results.keys()) if stats_results else None}")
        
        # Summary
    
        print("\n" + "=" * 60)
        print("GLOBAL ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Subjects analyzed: {len(subjects)}")
        print(f"Scalar correlation: {np.mean(self.global_scalar_data['correlations']):.4f} ± {np.std(self.global_scalar_data['correlations']):.4f}")
        print(f"Scalar MSE: {np.mean(self.global_scalar_data['mses']):.4f} ± {np.std(self.global_scalar_data['mses']):.4f}")
        print(f"Scalar MAE: {np.mean(self.global_scalar_data['maes']):.4f} ± {np.std(self.global_scalar_data['maes']):.4f}")
        print(f"Topo correlation: {np.mean(self.global_topo_data['correlations']):.4f} ± {np.std(self.global_topo_data['correlations']):.4f}")
        print(f"Topo MSE: {np.mean(self.global_topo_data['mses']):.4f} ± {np.std(self.global_topo_data['mses']):.4f}")
        print(f"Topo MAE: {np.mean(self.global_topo_data['maes']):.4f} ± {np.std(self.global_topo_data['maes']):.4f}")
        print(f"Topo NMSE: {np.mean(self.global_topo_data['nmses']):.4f} ± {np.std(self.global_topo_data['nmses']):.4f}")
        print(f"Topo NRMSE: {np.mean(self.global_topo_data['nrmses']):.4f} ± {np.std(self.global_topo_data['nrmses']):.4f}")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return {
            'subjects': subjects,
            'scalar_data': self.global_scalar_data,
            'topo_data': self.global_topo_data,
            'statistical_tests': stats_results,
            'target_state': self.target_state,
            'available_states': sorted(self.available_states) if self.available_states else None
        }
    

def run_state_based_analysis(results_dir, base_output_dir, patient_labels_file, fif_data_dir=None, skip_gfp=False):
    """Run analysis for all states and the complete dataset."""
    print("=" * 80)
    print("RUNNING STATE-BASED GLOBAL ANALYSIS")
    print("=" * 80)
    
    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    # First, load patient labels to get available states with proper grouping
    try:
        df = pd.read_csv(patient_labels_file)
        available_states = set()
        for _, row in df.iterrows():
            state = row['state']
            if pd.notna(state) and state != 'n/a':
                # Apply same grouping logic as in GlobalAnalyzer._load_patient_labels
                # Group diagnoses as requested:
                # - Merge MCS+ and MCS- into MCS
                # - Merge UWS and VS into VS/UWS (they are the same condition)
                if state in ['MCS+', 'MCS-']:
                    state = 'MCS'
                elif state == 'VS':
                    state = 'UWS'  # VS and UWS are the same, use UWS as standard
                
                available_states.add(state)
        
        available_states = sorted(available_states)
        print(f"📋 Found states (after grouping): {available_states}")
        print(f"ℹ️  Applied diagnosis grouping: MCS+/MCS- → MCS, VS → UWS")
    except Exception as e:
        print(f"❌ Error loading patient labels: {e}")
        return
    
    # Results container
    all_results = {}
    
    # 1. Run analysis for ALL subjects (no filtering)
    print(f"\n{'='*60}")
    print("ANALYZING ALL SUBJECTS")
    print(f"{'='*60}")
    
    all_output_dir = op.join(base_output_dir, 'all_subs')
    analyzer_all = GlobalAnalyzer(results_dir, all_output_dir, patient_labels_file, target_state=None, fif_data_dir=fif_data_dir)
    results_all = analyzer_all.run_analysis()
    all_results['all_subs'] = results_all
    
    # 2. Run analysis for each state
    for state in available_states:
        print(f"\n{'='*60}")
        print(f"ANALYZING STATE: {state}")
        print(f"{'='*60}")
        
        state_output_dir = op.join(base_output_dir, state)
        analyzer_state = GlobalAnalyzer(results_dir, state_output_dir, patient_labels_file, target_state=state, fif_data_dir=fif_data_dir, skip_gfp=skip_gfp)
        results_state = analyzer_state.run_analysis()
        
        if results_state and len(results_state['subjects']) > 0:
            all_results[state] = results_state
            print(f"✅ Completed analysis for state '{state}' with {len(results_state['subjects'])} subjects")
        else:
            print(f"⚠️  No subjects found for state '{state}' or analysis failed")
    
    # 3. Create summary across all states
    print(f"\n{'='*60}")
    print("CREATING CROSS-STATE SUMMARY")
    print(f"{'='*60}")
    
    summary_data = []
    for state_name, results in all_results.items():
        if results and len(results['subjects']) > 0:
            scalar_data = results['scalar_data']
            topo_data = results['topo_data']
            
            summary_data.append({
                'state': state_name,
                'n_subjects': len(results['subjects']),
                'scalar_correlation_mean': np.mean(scalar_data['correlations']),
                'scalar_correlation_std': np.std(scalar_data['correlations']),
                'scalar_mse_mean': np.mean(scalar_data['mses']),
                'scalar_mse_std': np.std(scalar_data['mses']),
                'topo_correlation_mean': np.mean(topo_data['correlations']),
                'topo_correlation_std': np.std(topo_data['correlations']),
                'topo_nmse_mean': np.mean(topo_data['nmses']),
                'topo_nmse_std': np.std(topo_data['nmses'])
            })
    
    # Save cross-state summary
    summary_df = pd.DataFrame(summary_data)
    summary_file = op.join(base_output_dir, f'cross_state_summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print("\n📊 CROSS-STATE SUMMARY:")
    print(summary_df.to_string(index=False))
    print(f"\n💾 Cross-state summary saved to: {summary_file}")
    
    print(f"\n{'='*80}")
    print("STATE-BASED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Results saved in: {base_output_dir}")
    print(f"📊 Summary across {len(all_results)} state groups")
    print(f"⏰ Timestamp: {timestamp}")
    
    return all_results, summary_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Global analysis across subjects')
    parser.add_argument('--results-dir', default='/Users/trinidad.borrell/Documents/Work/PhD/Proyects/nice/benchmark/results',
                       help='Results directory containing subject folders')
    parser.add_argument('--output-dir', help='Output directory for global analysis')
    parser.add_argument('--patient-labels', 
                       default='/Users/trinidad.borrell/Documents/Work/PhD/Proyects/nice/benchmark/py/metadata/patient_labels_with_controls.csv',
                       help='CSV file with patient labels and states')
    parser.add_argument('--data-dir', help='Data directory containing raw .fif files (for Global Field Power analysis)')
    parser.add_argument('--skip-gfp', action='store_true',
                       help='Skip Global Field Power analysis for faster execution')
    parser.add_argument('--single-state', help='Run analysis for a single state only')
    parser.add_argument('--no-state-analysis', action='store_true', 
                       help='Run traditional analysis without state filtering')
    
    args = parser.parse_args()
    
    # Default patient labels file
    patient_labels_file = args.patient_labels if op.exists(args.patient_labels) else None
    
    if args.no_state_analysis or not patient_labels_file:
        # Traditional analysis (backward compatibility)
        print("🔄 Running traditional global analysis (no state filtering)")
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = op.join(args.results_dir, 'global', f'global_results_{timestamp}')
        
        print(f"Starting global analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved with timestamp: {timestamp}")
        
        analyzer = GlobalAnalyzer(args.results_dir, output_dir, fif_data_dir=args.data_dir, skip_gfp=args.skip_gfp)
        results = analyzer.run_analysis()
        
        if results:
            print("\n✓ Global analysis complete!")
            print(f"📊 Plots saved in: {op.join(output_dir, 'plots')}")
            print(f"📈 Data saved in: {op.join(output_dir, 'data')}")
    
    elif args.single_state:
        # Single state analysis
        print(f"🎯 Running analysis for single state: {args.single_state}")
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = op.join(args.results_dir, 'global', args.single_state, f'global_results_{timestamp}')
        
        analyzer = GlobalAnalyzer(args.results_dir, output_dir, patient_labels_file, args.single_state, fif_data_dir=args.data_dir, skip_gfp=args.skip_gfp)
        results = analyzer.run_analysis()
        
        if results:
            print("\n✓ Single state analysis complete!")
            print(f"📊 Plots saved in: {op.join(output_dir, 'plots')}")
            print(f"📈 Data saved in: {op.join(output_dir, 'data')}")
    
    else:
        # State-based analysis (new default)
        print("🧩 Running state-based global analysis")
        
        if args.output_dir:
            base_output_dir = args.output_dir
        else:
            base_output_dir = op.join(args.results_dir, 'global')
        
        run_state_based_analysis(args.results_dir, base_output_dir, patient_labels_file, fif_data_dir=args.data_dir, skip_gfp=args.skip_gfp)


if __name__ == '__main__':
    main()