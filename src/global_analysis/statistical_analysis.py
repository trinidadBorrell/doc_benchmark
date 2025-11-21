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
from scipy.stats import ttest_rel, wilcoxon, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

# Try to import skimage for SSIM computation
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. SSIM computation will use custom implementation.")

# Try to import MNE for topographic plotting
try:
    import mne
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    HAS_MNE = True
    mne.set_log_level('WARNING')
except ImportError:
    HAS_MNE = False
    Path = None
    PathPatch = None
    print("Warning: MNE-Python not available. Topographic plots will be skipped.")

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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
    _egi256_outlines = {
        'ear1': np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
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
        print('  Defined EGI-256 sphere')
    else:
        sphere = None
    
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


class StatisticalAnalyzer:
    """Performs statistical analysis on EEG markers across subjects."""
    
    def __init__(self, results_dir, output_dir, patient_labels_file, data_dir=None):
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
        data_dir : str, optional
            Path to original data directory (for additional analysis)
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        self.data_dir = data_dir
        
        # Create output directories
        self.plots_dir = op.join(output_dir, 'plots')
        self.results_data_dir = op.join(output_dir, 'results')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_data_dir, exist_ok=True)
        
        # Load patient labels
        self.patient_labels = self._load_patient_labels()
        
        # Define full marker names (27 markers total, BEFORE filtering)
        all_marker_names = [
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
        
        # Define shortened marker names for plotting (BEFORE filtering)
        all_marker_short_names = [
            'PSD δ',
            'PSD δn',
            'PSD θ',
            'PSD θn',
            'PSD α',
            'PSD αn',
            'PSD β',
            'PSD βn',
            'PSD γ',
            'PSD γn',
            'PSD SE',
            'PSD MSF',
            'PSD SEF90',
            'PSD SEF95',
            'P Entropy',
            'SMI',
            'K Complexity',
            'CNV',
            'TLT P1',
            'TLT P3a',
            'TLT P3b',
            'TLC LSGS-LDGD',
            'TLC LSGD-LDGS',
            'TLC LD-LS',
            'TLC MMN',
            'TLC P3a',
            'TLC GD-GS',
            'TLC P3b'
        ]
        
        # Indices of non-normalized PSDs to exclude (delta, theta, alpha, beta, gamma)
        self.excluded_marker_indices = [0, 2, 4, 6, 8]
        
        # Create filtered marker lists (exclude non-normalized PSDs)
        self.marker_names = [name for idx, name in enumerate(all_marker_names) 
                            if idx not in self.excluded_marker_indices]
        self.marker_short_names = [name for idx, name in enumerate(all_marker_short_names) 
                                  if idx not in self.excluded_marker_indices]
        
        print(f"\n📊 Excluding non-normalized PSDs from analysis:")
        for idx in self.excluded_marker_indices:
            print(f"  - Excluded: {all_marker_names[idx]} ({all_marker_short_names[idx]})")
        print(f"  Analyzing {len(self.marker_names)} markers (down from {len(all_marker_names)})\n")
        
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
        skipped_subjects = []
        expected_shape = None
        
        # Find all subject directories
        subject_dirs = sorted(glob.glob(op.join(self.results_dir, 'sub-*')))
        
        # First pass: determine the most common shape
        all_shapes = []
        for subject_dir in subject_dirs:
            session_dirs = sorted(glob.glob(op.join(subject_dir, 'ses-*')))
            for session_dir in session_dirs:
                features_dir = op.join(session_dir, 'features_variable')
                topo_orig_file = op.join(features_dir, 'topos_original.npy')
                if op.exists(topo_orig_file):
                    shape = np.load(topo_orig_file).shape
                    all_shapes.append(shape)
        
        if all_shapes:
            # Find most common shape
            from collections import Counter
            shape_counts = Counter(all_shapes)
            expected_shape = shape_counts.most_common(1)[0][0]
            print(f"  Expected shape: {expected_shape} (found in {shape_counts[expected_shape]} samples)")
        
        # Second pass: load data with shape validation
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
                    
                    # Skip if shape doesn't match expected shape
                    if expected_shape is not None and topos_orig.shape != expected_shape:
                        print(f"  ⚠️  Skipped {subject_id}: shape {topos_orig.shape} != expected {expected_shape}")
                        skipped_subjects.append((subject_id, topos_orig.shape))
                        continue
                    
                    topos_orig_list.append(topos_orig)
                    topos_recon_list.append(topos_recon)
                    subject_ids.append(subject_id)
                    
                    print(f"  ✓ Loaded {subject_id}: {topos_orig.shape}")
        
        if skipped_subjects:
            print(f"\n⚠️  Skipped {len(skipped_subjects)} samples due to shape mismatch:")
            for subj_id, shape in skipped_subjects:
                print(f"    - {subj_id}: {shape}")
        
        print(f"\n✅ Loaded data for {len(subject_ids)} subjects")
        
        topos_orig_all = np.array(topos_orig_list)
        topos_recon_all = np.array(topos_recon_list)
        
        # Filter out non-normalized PSD markers
        print(f"\n🔧 Filtering out non-normalized PSDs...")
        print(f"  Original shape: {topos_orig_all.shape}")
        
        # Keep only markers NOT in excluded list
        all_marker_indices = np.arange(topos_orig_all.shape[1])
        keep_indices = [idx for idx in all_marker_indices if idx not in self.excluded_marker_indices]
        
        topos_orig_filtered = topos_orig_all[:, keep_indices, :]
        topos_recon_filtered = topos_recon_all[:, keep_indices, :]
        
        print(f"  Filtered shape: {topos_orig_filtered.shape}")
        print(f"  Removed {len(self.excluded_marker_indices)} non-normalized PSD markers\n")
        
        return {
            'original': topos_orig_filtered,
            'reconstructed': topos_recon_filtered,
            'subject_ids': subject_ids
        }
    
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
    
    def perform_wilcoxon_tests(self, topos_orig, topos_recon, subject_ids=None, group_name="all_subs"):
        """
        Perform channel-wise Wilcoxon signed-rank test for each marker.
        
        For each marker, performs Wilcoxon test at each channel independently,
        comparing original vs reconstructed across subjects.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        subject_ids : list, optional
            List of subject IDs
        group_name : str
            Name of the group being analyzed
            
        Returns
        -------
        dict
            Wilcoxon test results with p-values per channel per marker
        """
        print(f"\n🔬 Performing channel-wise Wilcoxon signed-rank tests for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        print(f"  Data shape: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        print(f"  Total tests: {n_markers * n_channels} ({n_markers} markers × {n_channels} channels)")
        
        results = {
            'group_name': group_name,
            'n_subjects': int(n_subjects),
            'n_markers': int(n_markers),
            'n_channels': int(n_channels),
            'markers': []
        }
        
        # For each marker, perform Wilcoxon test at each channel
        for marker_idx in range(n_markers):
            print(f"  Testing marker {marker_idx + 1}/{n_markers} ({self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f'M{marker_idx}'})...", end='\r')
            
            # Initialize arrays for this marker
            p_values = np.zeros(n_channels)
            statistics = np.zeros(n_channels)
            n_valid_per_channel = np.zeros(n_channels, dtype=int)
            
            # Test each channel independently
            for channel_idx in range(n_channels):
                # Get data for this marker and channel across all subjects
                orig_channel = topos_orig[:, marker_idx, channel_idx]  # (n_subjects,)
                recon_channel = topos_recon[:, marker_idx, channel_idx]  # (n_subjects,)
                
                # Remove NaN/Inf values
                valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                orig_valid = orig_channel[valid_mask]
                recon_valid = recon_channel[valid_mask]
                
                n_valid_per_channel[channel_idx] = len(orig_valid)
                
                if len(orig_valid) < 5:  # Need minimum samples for reliable test
                    p_values[channel_idx] = np.nan
                    statistics[channel_idx] = np.nan
                else:
                    try:
                        # Perform Wilcoxon signed-rank test
                        statistic, p_value = wilcoxon(orig_valid, recon_valid, alternative='two-sided')
                        p_values[channel_idx] = p_value
                        statistics[channel_idx] = statistic
                    except Exception as e:
                        # Handle cases where test fails (e.g., all zeros)
                        p_values[channel_idx] = np.nan
                        statistics[channel_idx] = np.nan
            
            # Compute summary statistics for this marker
            valid_p_values = p_values[np.isfinite(p_values)]
            
            marker_results = {
                'marker_idx': int(marker_idx),
                'marker_name': self.marker_names[marker_idx] if marker_idx < len(self.marker_names) else f"M{marker_idx}",
                'marker_short_name': self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}",
                'p_values': p_values.tolist(),  # p-value per channel
                'statistics': statistics.tolist(),  # Wilcoxon statistic per channel
                'n_valid_per_channel': n_valid_per_channel.tolist(),
                'n_valid_channels': int(np.sum(np.isfinite(p_values))),
                'mean_p_value': float(np.nanmean(p_values)),
                'median_p_value': float(np.nanmedian(p_values)),
                'min_p_value': float(np.nanmin(p_values)) if len(valid_p_values) > 0 else np.nan,
                'max_p_value': float(np.nanmax(p_values)) if len(valid_p_values) > 0 else np.nan,
                'n_channels_significant_05': int(np.sum(p_values < 0.05)),
                'n_channels_significant_01': int(np.sum(p_values < 0.01)),
                'n_channels_significant_001': int(np.sum(p_values < 0.001)),
                'proportion_channels_significant_05': float(np.sum(p_values < 0.05) / np.sum(np.isfinite(p_values))) if np.sum(np.isfinite(p_values)) > 0 else 0.0
            }
            
            results['markers'].append(marker_results)
        
        print(f"\n✅ Wilcoxon tests completed for {n_markers} markers × {n_channels} channels")
        
        # Overall summary statistics
        all_mean_p_values = [m['mean_p_value'] for m in results['markers'] if np.isfinite(m['mean_p_value'])]
        total_sig_05 = sum(m['n_channels_significant_05'] for m in results['markers'])
        total_sig_01 = sum(m['n_channels_significant_01'] for m in results['markers'])
        total_sig_001 = sum(m['n_channels_significant_001'] for m in results['markers'])
        
        results['summary'] = {
            'mean_p_value_across_all': float(np.mean(all_mean_p_values)) if all_mean_p_values else np.nan,
            'median_p_value_across_all': float(np.median(all_mean_p_values)) if all_mean_p_values else np.nan,
            'total_channels_significant_05': int(total_sig_05),
            'total_channels_significant_01': int(total_sig_01),
            'total_channels_significant_001': int(total_sig_001),
            'total_tests': int(n_markers * n_channels),
            'proportion_significant_05': float(total_sig_05 / (n_markers * n_channels))
        }
        
        print(f"  Summary: {total_sig_05}/{n_markers * n_channels} channel tests significant at p<0.05 " +
              f"({results['summary']['proportion_significant_05']:.1%})")
        
        return results
    
    def perform_ttest_tests(self, topos_orig, topos_recon, subject_ids=None, group_name="all_subs"):
        """
        Perform channel-wise paired t-test for each marker.
        
        For each marker, performs paired t-test at each channel independently,
        comparing original vs reconstructed across subjects.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        subject_ids : list, optional
            List of subject IDs
        group_name : str
            Name of the group being analyzed
            
        Returns
        -------
        dict
            Paired t-test results with p-values per channel per marker
        """
        print(f"\n🔬 Performing channel-wise paired t-tests for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        print(f"  Data shape: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        print(f"  Total tests: {n_markers * n_channels} ({n_markers} markers × {n_channels} channels)")
        
        results = {
            'group_name': group_name,
            'n_subjects': int(n_subjects),
            'n_markers': int(n_markers),
            'n_channels': int(n_channels),
            'markers': []
        }
        
        # For each marker, perform paired t-test at each channel
        for marker_idx in range(n_markers):
            print(f"  Testing marker {marker_idx + 1}/{n_markers} ({self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f'M{marker_idx}'})...", end='\r')
            
            # Initialize arrays for this marker
            p_values = np.zeros(n_channels)
            t_values = np.zeros(n_channels)
            n_valid_per_channel = np.zeros(n_channels, dtype=int)
            
            # Test each channel independently
            for channel_idx in range(n_channels):
                # Get data for this marker and channel across all subjects
                orig_channel = topos_orig[:, marker_idx, channel_idx]  # (n_subjects,)
                recon_channel = topos_recon[:, marker_idx, channel_idx]  # (n_subjects,)
                
                # Remove NaN/Inf values
                valid_mask = np.isfinite(orig_channel) & np.isfinite(recon_channel)
                orig_valid = orig_channel[valid_mask]
                recon_valid = recon_channel[valid_mask]
                
                n_valid_per_channel[channel_idx] = len(orig_valid)
                
                if len(orig_valid) < 5:  # Need minimum samples for reliable test
                    p_values[channel_idx] = np.nan
                    t_values[channel_idx] = np.nan
                else:
                    try:
                        # Perform paired t-test
                        t_stat, p_value = ttest_rel(orig_valid, recon_valid)
                        p_values[channel_idx] = p_value
                        t_values[channel_idx] = t_stat
                    except Exception as e:
                        # Handle cases where test fails (e.g., all zeros)
                        p_values[channel_idx] = np.nan
                        t_values[channel_idx] = np.nan
            
            # Compute summary statistics for this marker
            valid_p_values = p_values[np.isfinite(p_values)]
            
            marker_results = {
                'marker_idx': int(marker_idx),
                'marker_name': self.marker_names[marker_idx] if marker_idx < len(self.marker_names) else f"M{marker_idx}",
                'marker_short_name': self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}",
                'p_values': p_values.tolist(),  # p-value per channel
                't_values': t_values.tolist(),  # t-statistic per channel
                'n_valid_per_channel': n_valid_per_channel.tolist(),
                'n_valid_channels': int(np.sum(np.isfinite(p_values))),
                'mean_p_value': float(np.nanmean(p_values)),
                'median_p_value': float(np.nanmedian(p_values)),
                'min_p_value': float(np.nanmin(p_values)) if len(valid_p_values) > 0 else np.nan,
                'max_p_value': float(np.nanmax(p_values)) if len(valid_p_values) > 0 else np.nan,
                'n_channels_significant_05': int(np.sum(p_values < 0.05)),
                'n_channels_significant_01': int(np.sum(p_values < 0.01)),
                'n_channels_significant_001': int(np.sum(p_values < 0.001)),
                'proportion_channels_significant_05': float(np.sum(p_values < 0.05) / np.sum(np.isfinite(p_values))) if np.sum(np.isfinite(p_values)) > 0 else 0.0
            }
            
            results['markers'].append(marker_results)
        
        print(f"\n✅ Paired t-tests completed for {n_markers} markers × {n_channels} channels")
        
        # Overall summary statistics
        all_mean_p_values = [m['mean_p_value'] for m in results['markers'] if np.isfinite(m['mean_p_value'])]
        total_sig_05 = sum(m['n_channels_significant_05'] for m in results['markers'])
        total_sig_01 = sum(m['n_channels_significant_01'] for m in results['markers'])
        total_sig_001 = sum(m['n_channels_significant_001'] for m in results['markers'])
        
        results['summary'] = {
            'mean_p_value_across_all': float(np.mean(all_mean_p_values)) if all_mean_p_values else np.nan,
            'median_p_value_across_all': float(np.median(all_mean_p_values)) if all_mean_p_values else np.nan,
            'total_channels_significant_05': int(total_sig_05),
            'total_channels_significant_01': int(total_sig_01),
            'total_channels_significant_001': int(total_sig_001),
            'total_tests': int(n_markers * n_channels),
            'proportion_significant_05': float(total_sig_05 / (n_markers * n_channels))
        }
        
        print(f"  Summary: {total_sig_05}/{n_markers * n_channels} channel tests significant at p<0.05 " +
              f"({results['summary']['proportion_significant_05']:.1%})")
        
        return results
    
    def compute_structural_similarity(self, topos_orig, topos_recon, subject_ids=None, group_name="all_subs"):
        """
{{ ... }}
        Compute Structural Similarity Index (SSIM) between original and reconstructed topoplots.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        subject_ids : list, optional
            List of subject IDs
        group_name : str
            Name of the group being analyzed
            
        Returns
        -------
        dict
            SSIM results for each subject and marker
        """
        print(f"\n📊 Computing Structural Similarity Index (SSIM) for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        print(f"  Data shape: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        
        results = {
            'group_name': group_name,
            'n_subjects': int(n_subjects),
            'n_markers': int(n_markers), 
            'n_channels': int(n_channels),
            'subjects': [],
            'marker_means': [],
            'overall_stats': {}
        }
        
        # Store SSIM values for each subject and marker
        all_ssim_values = np.zeros((n_subjects, n_markers))
        
        for subj_idx in range(n_subjects):
            subject_id = subject_ids[subj_idx] if subject_ids else f"sub_{subj_idx:03d}"
            print(f"  Processing subject {subj_idx + 1}/{n_subjects}: {subject_id}...", end='\r')
            
            subject_results = {
                'subject_id': subject_id,
                'subject_idx': int(subj_idx),
                'markers': []
            }
            
            for marker_idx in range(n_markers):
                orig_topo = topos_orig[subj_idx, marker_idx, :]  # (n_channels,)
                recon_topo = topos_recon[subj_idx, marker_idx, :]  # (n_channels,)
                
                # Remove NaN/inf values
                valid_mask = np.isfinite(orig_topo) & np.isfinite(recon_topo)
                if np.sum(valid_mask) < 2:
                    ssim_value = np.nan
                else:
                    orig_valid = orig_topo[valid_mask]
                    recon_valid = recon_topo[valid_mask]
                    
                    try:
                        if HAS_SKIMAGE:
                            # Use scikit-image SSIM (need to reshape to 2D)
                            # For EEG topos, we treat the channel dimension as a 1D signal
                            # We can reshape to a square-ish 2D array for SSIM
                            n_valid = len(orig_valid)
                            side_length = int(np.ceil(np.sqrt(n_valid)))
                            
                            # Pad to square
                            padded_orig = np.pad(orig_valid, (0, side_length**2 - n_valid), mode='constant', constant_values=0)
                            padded_recon = np.pad(recon_valid, (0, side_length**2 - n_valid), mode='constant', constant_values=0)
                            
                            # Reshape to 2D
                            orig_2d = padded_orig.reshape(side_length, side_length)
                            recon_2d = padded_recon.reshape(side_length, side_length)
                            
                            # Compute SSIM
                            ssim_value = ssim(orig_2d, recon_2d, data_range=max(np.max(orig_2d) - np.min(orig_2d),
                                                                                np.max(recon_2d) - np.min(recon_2d)))
                        else:
                            # Custom SSIM implementation for 1D signals
                            ssim_value = self._custom_ssim_1d(orig_valid, recon_valid)
                            
                    except Exception as e:
                        print(f"\n⚠️  Error computing SSIM for subject {subject_id}, marker {marker_idx}: {e}")
                        ssim_value = np.nan
                
                all_ssim_values[subj_idx, marker_idx] = ssim_value
                
                subject_results['markers'].append({
                    'marker_idx': int(marker_idx),
                    'ssim': float(ssim_value) if np.isfinite(ssim_value) else None,
                    'n_valid_channels': int(np.sum(valid_mask))
                })
            
            results['subjects'].append(subject_results)
        
        print(f"\n✅ SSIM computation completed for {n_subjects} subjects")
        
        # Compute marker-wise statistics
        for marker_idx in range(n_markers):
            marker_ssim_values = all_ssim_values[:, marker_idx]
            valid_values = marker_ssim_values[np.isfinite(marker_ssim_values)]
            
            if len(valid_values) > 0:
                marker_stats = {
                    'marker_idx': int(marker_idx),
                    'n_valid_subjects': int(len(valid_values)),
                    'mean_ssim': float(np.mean(valid_values)),
                    'median_ssim': float(np.median(valid_values)),
                    'std_ssim': float(np.std(valid_values)),
                    'min_ssim': float(np.min(valid_values)),
                    'max_ssim': float(np.max(valid_values)),
                    'q25_ssim': float(np.percentile(valid_values, 25)),
                    'q75_ssim': float(np.percentile(valid_values, 75))
                }
            else:
                marker_stats = {
                    'marker_idx': int(marker_idx),
                    'n_valid_subjects': 0,
                    'error': 'No valid SSIM values computed'
                }
            
            results['marker_means'].append(marker_stats)
        
        # Overall statistics
        valid_ssim_values = all_ssim_values[np.isfinite(all_ssim_values)]
        if len(valid_ssim_values) > 0:
            results['overall_stats'] = {
                'n_total_comparisons': int(n_subjects * n_markers),
                'n_valid_comparisons': int(len(valid_ssim_values)),
                'overall_mean_ssim': float(np.mean(valid_ssim_values)),
                'overall_median_ssim': float(np.median(valid_ssim_values)),
                'overall_std_ssim': float(np.std(valid_ssim_values)),
                'overall_min_ssim': float(np.min(valid_ssim_values)),
                'overall_max_ssim': float(np.max(valid_ssim_values))
            }
        
        print(f"  Overall mean SSIM: {results['overall_stats'].get('overall_mean_ssim', 'N/A'):.4f}")
        
        return results
    
    def _custom_ssim_1d(self, x, y):
        """
        Custom SSIM implementation for 1D signals (EEG topographies).
        
        Parameters
        ----------
        x, y : array-like
            Input signals
            
        Returns
        -------
        float
            SSIM value between -1 and 1
        """
        # Constants to avoid division by zero
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to numpy arrays
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Compute means
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        
        # Compute variances and covariance
        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        
        # Compute SSIM
        numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def plot_wilcoxon_results(self, wilcoxon_results, topos_orig, group_name="all_subs"):
        """
        Create summary visualization for Wilcoxon test results.
        
        Parameters
        ----------
        wilcoxon_results : dict
            Results from perform_wilcoxon_tests (channel-wise)
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data (needed for custom sphere calculation)
        group_name : str
            Name of the group for plot title
        """
        print(f"\n📊 Plotting Wilcoxon test summary for {group_name}...")
        
        valid_markers = wilcoxon_results['markers']
        if not valid_markers:
            print("  No valid results to plot")
            return
        
        # Extract data for plotting - use mean p-value across channels
        marker_indices = [m['marker_idx'] for m in valid_markers]
        mean_p_values = [m['mean_p_value'] for m in valid_markers]
        prop_sig = [m['proportion_channels_significant_05'] for m in valid_markers]
        
        # Create marker labels with short names
        marker_labels = []
        for idx in marker_indices:
            if idx < len(self.marker_short_names):
                marker_labels.append(f"{self.marker_short_names[idx]}")
            else:
                marker_labels.append(f"M{idx}")
        
        # Create dual-axis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Top panel: Mean p-value per marker
        colors = []
        for p in mean_p_values:
            if p < 0.001:
                colors.append('darkred')
            elif p < 0.01:
                colors.append('red')
            elif p < 0.05:
                colors.append('orange')
            else:
                colors.append('lightgray')
        
        bars1 = ax1.bar(marker_labels, mean_p_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add significance threshold lines
        ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
        ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='p = 0.01')
        ax1.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, label='p = 0.001')
        
        ax1.set_xlabel('Marker', fontsize=11, weight='bold')
        ax1.set_ylabel('Mean p-value across channels', fontsize=11, weight='bold')
        ax1.set_title(f'Channel-wise Wilcoxon Test: Mean p-value per Marker\n'
                     f'Group: {group_name.upper()} (n={wilcoxon_results["n_subjects"]} subjects)', 
                     fontsize=13, weight='bold', pad=15)
        ax1.set_yscale('log')
        ax1.set_ylim(0.0001, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom panel: Proportion of significant channels per marker
        colors_prop = ['darkred' if p > 0.5 else 'red' if p > 0.2 else 'orange' if p > 0.05 else 'lightgray' 
                       for p in prop_sig]
        bars2 = ax2.bar(marker_labels, prop_sig, color=colors_prop, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        ax2.set_xlabel('Marker', fontsize=11, weight='bold')
        ax2.set_ylabel('Proportion of channels with p<0.05', fontsize=11, weight='bold')
        ax2.set_title('Spatial Extent of Significant Differences', fontsize=12, weight='bold', pad=10)
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add summary text box
        if 'summary' in wilcoxon_results:
            summary = wilcoxon_results['summary']
            textstr = (f"Total tests: {summary['total_tests']}\n"
                      f"Significant (p<0.05): {summary['total_channels_significant_05']} "
                      f"({summary['proportion_significant_05']:.1%})")
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'wilcoxon_summary_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved Wilcoxon summary plot: {plot_path}")
        
        # Also create topographic p-value maps
        self.plot_wilcoxon_topomaps(wilcoxon_results, topos_orig, group_name)
    
    def plot_wilcoxon_topomaps(self, wilcoxon_results, topos_orig, group_name="all_subs"):
        """
        Create topographic maps showing p-values per channel for each marker.
        
        Parameters
        ----------
        wilcoxon_results : dict
            Results from perform_wilcoxon_tests (channel-wise)
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data (needed for custom sphere calculation)
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping topographic p-value plots")
            return
        
        print(f"\n🗺️  Creating topographic p-value maps for {group_name}...")
        
        n_markers = len(wilcoxon_results['markers'])
        n_channels = wilcoxon_results['n_channels']
        
        # Compute mean topography for sphere calculation
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Plot topomaps in grid (5 rows max, adaptive columns)
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx, marker_data in enumerate(wilcoxon_results['markers']):
            p_values = np.array(marker_data['p_values'])
            marker_name = marker_data.get('marker_short_name', f"M{marker_idx}")
            
            # Transform p-values to -log10(p) for better visualization
            # Replace NaN with 0
            p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
            neg_log_p = -np.log10(p_values_clean)
            
            # Cap at reasonable range
            neg_log_p = np.clip(neg_log_p, 0, 10)
            
            ax = axes[marker_idx]
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                neg_log_p, 
                info,
                axes=ax,
                show=False,
                cmap='RdYlBu_r',
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            n_sig = marker_data['n_channels_significant_05']
            prop_sig = marker_data['proportion_channels_significant_05']
            ax.set_title(f"{marker_name}\n{n_sig} ch sig ({prop_sig:.1%})", 
                        fontsize=9, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('-log₁₀(p-value)', fontsize=11, weight='bold')
        
        # Add reference lines for significance thresholds
        cbar.ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--', linewidth=2, alpha=0.7)
        cbar.ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', linewidth=2, alpha=0.7)
        cbar.ax.text(3.5, -np.log10(0.05), 'p=0.05', fontsize=8, color='orange', weight='bold')
        cbar.ax.text(3.5, -np.log10(0.01), 'p=0.01', fontsize=8, color='red', weight='bold')
        
        plt.suptitle(f'Topographic Distribution of p-values (Wilcoxon Test)\n'
                    f'Group: {group_name.upper()} | n={wilcoxon_results["n_subjects"]} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'wilcoxon_topomaps_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved Wilcoxon topographic maps: {plot_path}")
    
    def plot_ttest_results(self, ttest_results, topos_orig, group_name="all_subs"):
        """
        Plot paired t-test results (mean p-value per marker).
        
        Parameters
        ----------
        ttest_results : dict
            Results from perform_ttest_tests (channel-wise)
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data (needed for custom sphere calculation)
        group_name : str
            Name of the group for plot title
        """
        print(f"\n📊 Plotting paired t-test summary for {group_name}...")
        
        valid_markers = ttest_results['markers']
        if not valid_markers:
            print("  No valid results to plot")
            return
        
        # Extract data for plotting - use mean p-value across channels
        marker_indices = [m['marker_idx'] for m in valid_markers]
        mean_p_values = [m['mean_p_value'] for m in valid_markers]
        prop_sig = [m['proportion_channels_significant_05'] for m in valid_markers]
        
        # Create marker labels with short names
        marker_labels = []
        for idx in marker_indices:
            if idx < len(self.marker_short_names):
                marker_labels.append(f"{self.marker_short_names[idx]}")
            else:
                marker_labels.append(f"M{idx}")
        
        # Create dual-axis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Top panel: Mean p-value per marker
        colors = []
        for p in mean_p_values:
            if p < 0.001:
                colors.append('darkred')
            elif p < 0.01:
                colors.append('red')
            elif p < 0.05:
                colors.append('orange')
            else:
                colors.append('lightgray')
        
        bars1 = ax1.bar(marker_labels, mean_p_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add significance threshold lines
        ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
        ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='p = 0.01')
        ax1.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, label='p = 0.001')
        
        ax1.set_xlabel('Marker', fontsize=11, weight='bold')
        ax1.set_ylabel('Mean p-value across channels', fontsize=11, weight='bold')
        ax1.set_title(f'Channel-wise Paired T-Test: Mean p-value per Marker\n'
                     f'Group: {group_name.upper()} (n={ttest_results["n_subjects"]} subjects)', 
                     fontsize=13, weight='bold', pad=15)
        ax1.set_yscale('log')
        ax1.set_ylim(0.0001, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom panel: Proportion of significant channels per marker
        colors_prop = ['darkred' if p > 0.5 else 'red' if p > 0.2 else 'orange' if p > 0.05 else 'lightgray' 
                       for p in prop_sig]
        bars2 = ax2.bar(marker_labels, prop_sig, color=colors_prop, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        ax2.set_xlabel('Marker', fontsize=11, weight='bold')
        ax2.set_ylabel('Proportion of channels with p<0.05', fontsize=11, weight='bold')
        ax2.set_title('Spatial Extent of Significant Differences', fontsize=12, weight='bold', pad=10)
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add summary text box
        if 'summary' in ttest_results:
            summary = ttest_results['summary']
            textstr = (f"Total tests: {summary['total_tests']}\n"
                      f"Significant (p<0.05): {summary['total_channels_significant_05']} "
                      f"({summary['proportion_significant_05']:.1%})")
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'ttest_summary_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved t-test summary plot: {plot_path}")
        
        # Also create topographic p-value maps
        self.plot_ttest_topomaps(ttest_results, topos_orig, group_name)
    
    def plot_ttest_topomaps(self, ttest_results, topos_orig, group_name="all_subs"):
        """
        Create topographic maps showing p-values per channel for each marker (t-test).
        
        Parameters
        ----------
        ttest_results : dict
            Results from perform_ttest_tests (channel-wise)
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data (needed for custom sphere calculation)
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping topographic p-value plots")
            return
        
        print(f"\n🗺️  Creating topographic p-value maps (t-test) for {group_name}...")
        
        n_markers = len(ttest_results['markers'])
        n_channels = ttest_results['n_channels']
        
        # Compute mean topography for sphere calculation
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Plot topomaps in grid (5 rows max, adaptive columns)
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx, marker_data in enumerate(ttest_results['markers']):
            p_values = np.array(marker_data['p_values'])
            marker_name = marker_data.get('marker_short_name', f"M{marker_idx}")
            
            # Transform p-values to -log10(p) for better visualization
            # Replace NaN with 0
            p_values_clean = np.where(np.isfinite(p_values), p_values, 1.0)
            neg_log_p = -np.log10(p_values_clean)
            
            # Cap at reasonable range
            neg_log_p = np.clip(neg_log_p, 0, 10)
            
            ax = axes[marker_idx]
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                neg_log_p, 
                info,
                axes=ax,
                show=False,
                cmap='RdYlBu_r',
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            n_sig = marker_data['n_channels_significant_05']
            prop_sig = marker_data['proportion_channels_significant_05']
            ax.set_title(f"{marker_name}\n{n_sig} ch sig ({prop_sig:.1%})", 
                        fontsize=9, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('-log₁₀(p-value)', fontsize=11, weight='bold')
        
        # Add reference lines for significance thresholds
        cbar.ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--', linewidth=2, alpha=0.7)
        cbar.ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', linewidth=2, alpha=0.7)
        cbar.ax.text(3.5, -np.log10(0.05), 'p=0.05', fontsize=8, color='orange', weight='bold')
        cbar.ax.text(3.5, -np.log10(0.01), 'p=0.01', fontsize=8, color='red', weight='bold')
        
        plt.suptitle(f'Topographic Distribution of p-values (Paired T-Test)\n'
                    f'Group: {group_name.upper()} | n={ttest_results["n_subjects"]} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'ttest_topomaps_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved t-test topographic maps: {plot_path}")
    
    def plot_std_topomaps(self, topos_orig, topos_recon, group_name="all_subs"):
        """
        Create topographic maps showing standard deviation of differences across subjects.
        
        Shows the inter-subject variability in reconstruction error for each marker.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping std topographic plots")
            return
        
        print(f"\n🗺️  Creating standard deviation topomaps for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        
        # Compute mean for setting up montage with proper sphere
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Compute differences and std across subjects
        differences = topos_orig - topos_recon  # (n_subjects, n_markers, n_channels)
        std_diff = np.std(differences, axis=0)  # (n_markers, n_channels)
        
        # Plot topomaps in grid (same structure as Wilcoxon/T-test topomaps)
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx in range(n_markers):
            marker_name = self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}"
            
            std_data = std_diff[marker_idx, :]
            ax = axes[marker_idx]
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                std_data,
                info,
                axes=ax,
                show=False,
                cmap='viridis',
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            mean_std = np.mean(std_data)
            max_std = np.max(std_data)
            ax.set_title(f"{marker_name}\nMean STD: {mean_std:.3f} | Max: {max_std:.3f}", 
                        fontsize=9, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Standard Deviation', fontsize=11, weight='bold')
        
        plt.suptitle(f'Standard Deviation of Differences (Original - Reconstructed)\n'
                    f'Group: {group_name.upper()} | n={n_subjects} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'std_topomaps_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved std topographic maps: {plot_path}")
    
    def plot_mean_diff_topomaps(self, topos_orig, topos_recon, group_name="all_subs"):
        """
        Create topographic maps showing mean difference (original - reconstructed) across subjects.
        
        Shows the average reconstruction error for each marker and channel.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping mean difference topographic plots")
            return
        
        print(f"\n🗺️  Creating mean difference topomaps for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        
        # Compute mean for setting up montage with proper sphere
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Compute mean difference across subjects
        differences = topos_orig - topos_recon  # (n_subjects, n_markers, n_channels)
        mean_diff = np.mean(differences, axis=0)  # (n_markers, n_channels)
        
        # Plot topomaps in grid (same structure as Wilcoxon/T-test/STD topomaps)
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx in range(n_markers):
            marker_name = self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}"
            
            mean_diff_data = mean_diff[marker_idx, :]
            ax = axes[marker_idx]
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                mean_diff_data,
                info,
                axes=ax,
                show=False,
                cmap='RdBu_r',  # Diverging colormap: red=positive diff, blue=negative diff
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            mean_val = np.mean(mean_diff_data)
            abs_max = np.max(np.abs(mean_diff_data))
            ax.set_title(f"{marker_name}\nMean: {mean_val:.3f} | |Max|: {abs_max:.3f}", 
                        fontsize=9, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Mean Difference\n(Original - Reconstructed)', fontsize=11, weight='bold')
        
        # Add reference line at 0
        cbar.ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
        
        plt.suptitle(f'Mean Difference Across Subjects (Original - Reconstructed)\n'
                    f'Group: {group_name.upper()} | n={n_subjects} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'mean_diff_topomaps_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved mean difference topographic maps: {plot_path}")
    
    def plot_std_topomaps_scaled(self, topos_orig, topos_recon, group_name="all_subs"):
        """
        Create topographic maps showing standard deviation of differences across subjects,
        with colorbar scale based on original/reconstructed marker ranges.
        
        This provides context for how large the STD is relative to the actual marker values.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping scaled std topographic plots")
            return
        
        print(f"\n🗺️  Creating scaled standard deviation topomaps for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        
        # Compute mean for setting up montage with proper sphere
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Compute differences and std across subjects
        differences = topos_orig - topos_recon  # (n_subjects, n_markers, n_channels)
        std_diff = np.std(differences, axis=0)  # (n_markers, n_channels)
        
        # Plot topomaps in grid
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx in range(n_markers):
            marker_name = self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}"
            
            std_data = std_diff[marker_idx, :]
            ax = axes[marker_idx]
            
            # Compute scale based on original and reconstructed marker values
            orig_marker_data = topos_orig[:, marker_idx, :]  # (n_subjects, n_channels)
            recon_marker_data = topos_recon[:, marker_idx, :]  # (n_subjects, n_channels)
            
            # Get min/max across both original and reconstructed
            vmin = min(np.nanmin(orig_marker_data), np.nanmin(recon_marker_data))
            vmax = max(np.nanmax(orig_marker_data), np.nanmax(recon_marker_data))
            
            # Plot topomap with scaled colorbar
            im, _ = mne.viz.plot_topomap(
                std_data,
                info,
                axes=ax,
                show=False,
                cmap='viridis',
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            mean_std = np.mean(std_data)
            max_std = np.max(std_data)
            ax.set_title(f"{marker_name}\nMean STD: {mean_std:.3f} | Max: {max_std:.3f}\nScale: [{vmin:.2f}, {vmax:.2f}]", 
                        fontsize=8, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Note: Each topomap has its own colorbar scale, so we don't add a shared colorbar
        
        plt.suptitle(f'Standard Deviation of Differences (Scaled to Original/Reconstructed Range)\n'
                    f'Group: {group_name.upper()} | n={n_subjects} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1.0, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'std_topomaps_scaled_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved scaled std topographic maps: {plot_path}")
    
    def plot_mean_diff_topomaps_scaled(self, topos_orig, topos_recon, group_name="all_subs"):
        """
        Create topographic maps showing mean difference (original - reconstructed) across subjects,
        with colorbar scale based on original/reconstructed marker ranges.
        
        This provides context for how large the mean difference is relative to the actual marker values.
        
        Parameters
        ----------
        topos_orig : array, shape (n_subjects, n_markers, n_channels)
            Original topographic data
        topos_recon : array, shape (n_subjects, n_markers, n_channels)
            Reconstructed topographic data
        group_name : str
            Name of the group for plot title
        """
        if not HAS_MNE:
            print("  ⚠️  MNE not available, skipping scaled mean difference topographic plots")
            return
        
        print(f"\n🗺️  Creating scaled mean difference topomaps for {group_name}...")
        
        n_subjects, n_markers, n_channels = topos_orig.shape
        
        # Compute mean for setting up montage with proper sphere
        topos_mean = np.mean(topos_orig, axis=0)  # (n_markers, n_channels)
        
        # Set up montage with proper sphere and outlines
        try:
            info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_mean)
        except Exception as e:
            print(f"  ⚠️  Could not set up montage: {e}")
            return
        
        # Compute mean difference across subjects
        differences = topos_orig - topos_recon  # (n_subjects, n_markers, n_channels)
        mean_diff = np.mean(differences, axis=0)  # (n_markers, n_channels)
        
        # Plot topomaps in grid
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
        if n_markers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for marker_idx in range(n_markers):
            marker_name = self.marker_short_names[marker_idx] if marker_idx < len(self.marker_short_names) else f"M{marker_idx}"
            
            mean_diff_data = mean_diff[marker_idx, :]
            ax = axes[marker_idx]
            
            # Compute scale based on original and reconstructed marker values
            orig_marker_data = topos_orig[:, marker_idx, :]  # (n_subjects, n_channels)
            recon_marker_data = topos_recon[:, marker_idx, :]  # (n_subjects, n_channels)
            
            # Get min/max across both original and reconstructed
            vmin = min(np.nanmin(orig_marker_data), np.nanmin(recon_marker_data))
            vmax = max(np.nanmax(orig_marker_data), np.nanmax(recon_marker_data))
            
            # Plot topomap with scaled colorbar
            im, _ = mne.viz.plot_topomap(
                mean_diff_data,
                info,
                axes=ax,
                show=False,
                cmap='RdBu_r',  # Diverging colormap
                sphere=sphere,
                outlines=outlines,
                contours=4,
                size=2
            )
            
            # Add title with marker name and stats
            mean_val = np.mean(mean_diff_data)
            abs_max = np.max(np.abs(mean_diff_data))
            ax.set_title(f"{marker_name}\nMean: {mean_val:.3f} | |Max|: {abs_max:.3f}\nScale: [{vmin:.2f}, {vmax:.2f}]", 
                        fontsize=8, weight='bold')
        
        # Remove extra axes
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')
        
        # Note: Each topomap has its own colorbar scale, so we don't add a shared colorbar
        
        plt.suptitle(f'Mean Difference Across Subjects (Scaled to Original/Reconstructed Range)\n'
                    f'Group: {group_name.upper()} | n={n_subjects} subjects',
                    fontsize=14, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1.0, 0.96])
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'mean_diff_topomaps_scaled_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved scaled mean difference topographic maps: {plot_path}")
    
    def plot_ssim_results(self, ssim_results, group_name="all_subs"):
        """
        Plot SSIM results.
        
        Parameters
        ----------
        ssim_results : dict
            Results from compute_structural_similarity
        group_name : str
            Name of the group for plot title
        """
        print(f"\n📊 Plotting SSIM results for {group_name}...")
        
        valid_markers = [m for m in ssim_results['marker_means'] if 'error' not in m]
        if not valid_markers:
            print("  No valid results to plot")
            return
        
        # Extract data for plotting
        marker_indices = [m['marker_idx'] for m in valid_markers]
        mean_ssim = [m['mean_ssim'] for m in valid_markers]
        std_ssim = [m['std_ssim'] for m in valid_markers]
        
        # Create marker labels with short names
        marker_labels = []
        for idx in marker_indices:
            if idx < len(self.marker_short_names):
                marker_labels.append(f"{self.marker_short_names[idx]}")
            else:
                marker_labels.append(f"M{idx}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar plot with error bars
        bars = ax.bar(marker_labels, mean_ssim, yerr=std_ssim, 
                     alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5,
                     capsize=5)
        
        # Add value labels on bars
        for bar, ssim_val in zip(bars, mean_ssim):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{ssim_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel('Marker', fontsize=12, weight='bold')
        ax.set_ylabel('Structural Similarity Index (SSIM)', fontsize=12, weight='bold')
        ax.set_title(f'Structural Similarity: Original vs Reconstructed Topographic Markers\n'
                    f'Group: {group_name.upper()} (n={ssim_results["n_subjects"]})', 
                    fontsize=14, weight='bold', pad=20)
        
        # Set reasonable y-axis limits (SSIM ranges from -1 to 1, but usually positive)
        ax.set_ylim(0, 1.1)
        
        # Rotate x-axis labels if too many markers
        if len(marker_labels) > 10:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add reference lines
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect similarity')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good similarity')
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Moderate similarity')
        ax.legend(loc='upper right')
        
        # Add summary text box
        if 'overall_stats' in ssim_results and ssim_results['overall_stats']:
            overall = ssim_results['overall_stats']
            textstr = (f"Overall Statistics:\n"
                      f"Mean SSIM: {overall.get('overall_mean_ssim', 'N/A'):.3f}\n"
                      f"Median SSIM: {overall.get('overall_median_ssim', 'N/A'):.3f}\n"
                      f"Std SSIM: {overall.get('overall_std_ssim', 'N/A'):.3f}")
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = op.join(self.plots_dir, f'ssim_results_{group_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved SSIM plot: {plot_path}")
    
    def plot_ssmi_results(self, ssmi_results):
        """Plot SSMI results."""
        print("\n📊 Plotting SSMI results...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot SSMI per marker
        marker_indices = np.arange(len(ssmi_results['ssmi_per_marker']))
        
        # Create marker labels with short names
        marker_labels = []
        for idx in marker_indices:
            if idx < len(self.marker_short_names):
                marker_labels.append(self.marker_short_names[idx])
            else:
                marker_labels.append(f"M{idx}")
        
        ax1.bar(marker_labels, ssmi_results['ssmi_per_marker'], alpha=0.7, color='coral')
        ax1.axhline(y=ssmi_results['mean_ssmi_per_marker'], color='red', linestyle='--', 
                   label=f'Mean: {ssmi_results["mean_ssmi_per_marker"]:.6f}')
        ax1.set_xlabel('Marker', fontsize=11, weight='bold')
        ax1.set_ylabel('SSMI', fontsize=11, weight='bold')
        ax1.set_title('SSMI per Marker', fontsize=12, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
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
    
    def analyze_group(self, topos_orig, topos_recon, subject_ids, group_name="all_subs"):
        """
        Perform complete statistical analysis for a specific group.
        
        Parameters
        ----------
        topos_orig : array
            Original topographic data
        topos_recon : array
            Reconstructed topographic data
        subject_ids : list
            Subject IDs
        group_name : str
            Name of the group
            
        Returns
        -------
        dict
            Complete analysis results for the group
        """
        print(f"\n🔬 Analyzing group: {group_name}")
        
        # Run all statistical tests
        wilcoxon_results = self.perform_wilcoxon_tests(topos_orig, topos_recon, subject_ids, group_name)
        ttest_results = self.perform_ttest_tests(topos_orig, topos_recon, subject_ids, group_name)
        ssim_results = self.compute_structural_similarity(topos_orig, topos_recon, subject_ids, group_name)
        ssmi_results = self.compute_ssmi(topos_orig, topos_recon)
        
        # Create plots
        self.plot_wilcoxon_results(wilcoxon_results, topos_orig, group_name)
        self.plot_ttest_results(ttest_results, topos_orig, group_name)
        self.plot_ssim_results(ssim_results, group_name)
        self.plot_ssmi_results(ssmi_results)
        self.plot_std_topomaps(topos_orig, topos_recon, group_name)
        self.plot_mean_diff_topomaps(topos_orig, topos_recon, group_name)
        
        # Create scaled plots (with original/reconstructed marker ranges)
        self.plot_std_topomaps_scaled(topos_orig, topos_recon, group_name)
        self.plot_mean_diff_topomaps_scaled(topos_orig, topos_recon, group_name)
        
        # Prepare results
        group_results = {
            'group_name': group_name,
            'analysis_date': datetime.now().isoformat(),
            'n_subjects': len(subject_ids),
            'subject_ids': subject_ids,
            'wilcoxon_test': wilcoxon_results,
            'ttest': ttest_results,
            'ssim_analysis': ssim_results,
            'ssmi': ssmi_results
        }
        
        # Save results as JSON
        json_path = op.join(self.results_data_dir, f'statistical_results_{group_name}.json')
        with open(json_path, 'w') as f:
            json.dump(group_results, f, indent=2)
        
        print(f"✅ Saved {group_name} results to: {json_path}")
        
        return group_results
    
    def run_analysis(self, analyze_by_groups=True):
        """Run complete statistical analysis."""
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Load data
        data = self.load_topographic_data()
        topos_orig = data['original']
        topos_recon = data['reconstructed']
        subject_ids = data['subject_ids']
        
        all_results = {}
        
        # Analyze all subjects together
        all_results['all_subs'] = self.analyze_group(topos_orig, topos_recon, subject_ids, "all_subs")
        
        # Analyze by groups if requested and patient labels are available
        if analyze_by_groups and hasattr(self, 'patient_labels') and self.patient_labels is not None:
            print("\n🔍 Analyzing by patient groups...")
            
            # Create group mappings
            patient_df = self.patient_labels.copy()
            
            # Map subject IDs to groups
            subject_groups = {}
            for subj_id in subject_ids:
                # Look for this subject in patient labels
                # Try different possible ID formats
                possible_ids = [subj_id, f"sub-{subj_id}", subj_id.replace('sub-', '')]
                
                group_found = False
                for possible_id in possible_ids:
                    if possible_id in patient_df['subject'].values:
                        row = patient_df[patient_df['subject'] == possible_id].iloc[0]
                        # Use 'state' column as group (UWS, MCS-, MCS+, EMCS, etc.)
                        group = row.get('state', row.get('diagnostic_crs_final', 'Unknown'))
                        subject_groups[subj_id] = group
                        group_found = True
                        break
                
                if not group_found:
                    print(f"⚠️  Subject {subj_id} not found in patient labels")
                    subject_groups[subj_id] = 'Unknown'
            
            # Group subjects by their labels
            groups = {}
            for subj_id, group in subject_groups.items():
                if group not in groups:
                    groups[group] = []
                groups[group].append(subj_id)
            
            print(f"Found groups: {list(groups.keys())}")
            
            # Analyze each group with sufficient subjects
            for group_name, group_subjects in groups.items():
                if len(group_subjects) >= 3:  # Minimum subjects for meaningful analysis
                    print(f"\n📊 Analyzing group '{group_name}' with {len(group_subjects)} subjects")
                    
                    # Get indices of subjects in this group
                    group_indices = [i for i, subj_id in enumerate(subject_ids) if subj_id in group_subjects]
                    
                    if group_indices:
                        group_topos_orig = topos_orig[group_indices]
                        group_topos_recon = topos_recon[group_indices]
                        group_subj_ids = [subject_ids[i] for i in group_indices]
                        
                        all_results[group_name] = self.analyze_group(
                            group_topos_orig, group_topos_recon, group_subj_ids, group_name
                        )
                else:
                    print(f"⏭️  Skipping group '{group_name}' - insufficient subjects ({len(group_subjects)} < 3)")
        
        # Save combined results
        combined_json_path = op.join(self.results_data_dir, 'all_statistical_results.json')
        with open(combined_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Saved combined results to: {combined_json_path}")
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*60)
        
        return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Statistical analysis across multiple subjects')
    parser.add_argument('--results-dir', required=True, help='Path to results directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for statistical results')
    parser.add_argument('--patient-labels', required=True, help='Path to patient labels CSV file')
    parser.add_argument('--data-dir', help='Path to original data directory (optional)')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = StatisticalAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        patient_labels_file=args.patient_labels,
        data_dir=args.data_dir
    )
    
    analyzer.run_analysis()


if __name__ == '__main__':
    main()

