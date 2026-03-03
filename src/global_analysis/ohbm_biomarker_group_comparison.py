"""Biomarker Group Comparison - 9x4 Grid Topoplot Analysis

Creates a comprehensive topographic comparison plot showing Spearman correlations
between original and reconstructed EEG biomarkers across 4 subject groups:
- MCS (MCS+ and MCS-)
- UWS (VS and UWS)  
- Control Local Global
- Control Resting State

Rows: 9 biomarkers (Alpha, Beta, Delta, Gamma, Theta normalized power,
       Permutation Entropy pe_theta, Spectral Entropy, Kolmogorov Complexity,
       Symbolic Mutual Information wsmi_theta)
Columns: 4 subject groups with FDR-corrected Spearman test results

Author: Based on global_topoplots_minimal.py structure
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
from pathlib import Path
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Import sklearn for mutual information computation
try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import normalized_mutual_info_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Mutual information computation will be skipped.")

# Import statsmodels for FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. FDR correction will be skipped.")

# Import MNE for topographic plotting
try:
    import mne
    HAS_MNE = True
    mne.set_log_level('WARNING')
except ImportError as e:
    HAS_MNE = False
    print(f"Warning: MNE-Python not available. Topographic plots will be skipped. Error: {e}")

# Set plotting style
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
        "axes.titlesize": 22,
        "axes.labelsize": "large",
        "ytick.labelsize": 22,
        "xtick.labelsize": 22,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)


class MarkerNameMapper:
    """Maps marker indices to human-readable names."""
    
    def __init__(self):
        # Updated marker names matching new pipeline
        self.marker_names = [
            'delta_power_spectralpower',
            'delta_relative_spectralpower',
            'theta_power_spectralpower',
            'theta_relative_spectralpower',
            'alpha_power_spectralpower',
            'alpha_relative_spectralpower',
            'beta_power_spectralpower',
            'beta_relative_spectralpower',
            'gamma_power_spectralpower',
            'gamma_relative_spectralpower',
            'spectral_entropy_spectralpower',
            'msf_psdsummary',
            'sef90_psdsummary',
            'sef95_psdsummary',
            'pe_theta_permutationentropy',
            'kolmogorov_complexity_kolmogorovcomplexity',
            'cnv_detailed_cnvslope',
            'p1_topography_timelockedtopo',
            'p3a_topography_timelockedtopo',
            'p3b_topography_timelockedtopo',
            'timelockedcontrast_lsgs_ldgd_timelockedcontrast',
            'timelockedcontrast_lsgd_ldgs_timelockedcontrast',
            'timelockedcontrast_ld_ls_timelockedcontrast',
            'timelockedcontrast_mmn_timelockedcontrast',
            'timelockedcontrast_p3a_timelockedcontrast',
            'timelockedcontrast_gd_gs_timelockedcontrast',
            'Timelockedcontrast_p3b_timelockedcontrast',
            'wsmi_theta_symbolicmutualinformation'
        ]
        
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


def _prepare_egi256_sphere_and_outlines(evoked):
    """Prepare sphere and outlines for EGI-256 topographic plotting."""
    
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
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    path = Path(vertices=vertices, codes=codes)
    
    def patch():
        return PathPatch(path, alpha=0.1)
    
    outlines['patch'] = patch
    
    return this_sphere, outlines


def _setup_montage_and_sphere(n_channels, topos_mean=None):
    """Set up MNE montage, info object, sphere, and outlines for topographic plotting."""
    
    if n_channels == 256:
        print('  Setting up EGI-256 montage with custom sphere and outlines')
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage, on_missing='warn')
        
        if topos_mean is not None:
            evoked = mne.EvokedArray(topos_mean.T, info, tmin=0)
            sphere, outlines = _prepare_egi256_sphere_and_outlines(evoked)
        else:
            sphere = 'auto'
            outlines = 'head'
            
    elif n_channels == 128:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage)
        sphere = 'auto'
        outlines = 'head'
        
    elif n_channels == 64:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-64_1.0')
        info = mne.create_info(montage.ch_names, 250, ch_types='eeg')
        info.set_montage(montage)
        sphere = 'auto'
        outlines = 'head'
        
    else:
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


def load_patient_labels(patient_labels_file):
    """Load patient labels from CSV file."""
    try:
        print(f"📋 Loading patient labels from: {patient_labels_file}")
        df = pd.read_csv(patient_labels_file)
        
        patient_labels = {}
        for _, row in df.iterrows():
            subject = row['subject']
            session = f"ses-{row['session']:02d}"
            state = row['diagnostic_crs_final']  # Use diagnostic_crs_final instead of state
            
            if pd.isna(state) or state == 'n/a':
                continue
            
            subject_session_key = f"{subject}_{session}"
            patient_labels[subject_session_key] = state
        
        print(f"   ✓ Loaded labels for {len(patient_labels)} subject/sessions")
        return patient_labels
        
    except Exception as e:
        print(f"   ⚠️  Error loading patient labels: {e}")
        return {}


def load_group_topo_data(data_dir, group_name, patient_labels=None, diagnostic_filter=None, allow_heterogeneous=False):
    """Load topographic data from a specific directory for a subject group."""
    print(f"🔍 Loading {group_name} data from: {data_dir}")
    
    if not op.exists(data_dir):
        print(f"   ❌ Directory not found: {data_dir}")
        return None, None, None
    
    # Find all subjects with orig/recon directories
    subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('sub-')]
    print(f"📁 Found {len(subject_dirs)} subject directories")
    
    subjects_data = []
    subject_ids = []
    
    for subject_dir in sorted(subject_dirs):
        subject_path = op.join(data_dir, subject_dir)
        if not op.isdir(subject_path):
            continue
            
        subject_id = subject_dir.replace('sub-', '')
        
        # Find all sessions for this subject
        try:
            session_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
        except PermissionError:
            print(f"    ⚠️  Permission denied accessing {subject_path}")
            continue
            
        for session_dir in sorted(session_dirs):
            session_path = op.join(subject_path, session_dir)
            subject_session_id = f"{subject_id}_{session_dir}"
            
            # Apply diagnostic filtering for DoC patients
            if patient_labels and diagnostic_filter:
                diagnosis = patient_labels.get(subject_session_id)
                if diagnosis not in diagnostic_filter:
                    continue
            
            # Load topographic data
            try:
                # Check for orig and recon directories
                orig_dir = op.join(session_path, "orig")
                recon_dir = op.join(session_path, "recon")
                
                topos_orig_file = op.join(orig_dir, f"topos_sub-{subject_id}_{session_dir}.npz")
                topos_recon_file = op.join(recon_dir, f"topos_sub-{subject_id}_{session_dir}.npz")
                
                # Check files exist
                if not op.exists(topos_orig_file) or not op.exists(topos_recon_file):
                    continue
                
                # Load and extract topographic data
                def extract_topo_markers(npz_data):
                    marker_names = sorted(npz_data.files)
                    if not marker_names:
                        raise ValueError("No markers found in .npz file")
                    topo_data = np.array([npz_data[name] for name in marker_names])
                    return topo_data, marker_names
                
                topos_orig, marker_names = extract_topo_markers(np.load(topos_orig_file))
                topos_recon, _ = extract_topo_markers(np.load(topos_recon_file))
                
                # Validate data shape
                if topos_orig.ndim != 2 or topos_recon.ndim != 2:
                    continue
                
                if topos_orig.shape != topos_recon.shape:
                    continue
                
                subjects_data.append({
                    'topos_original': topos_orig,
                    'topos_reconstructed': topos_recon,
                    'marker_names': marker_names,
                    'subject_id': subject_id,
                    'session_id': session_dir,
                })
                subject_ids.append(subject_session_id)
                
            except Exception as e:
                print(f"     ❌ Error loading {subject_id}/{session_dir}: {e}")
                continue
    
    if not subjects_data:
        print(f"   ❌ No valid data found for {group_name}")
        return None, None, None
    
    print(f"   ✓ Loaded {len(subjects_data)} subjects for {group_name}")
    
    if allow_heterogeneous:
        # For Control RS: keep all subjects with different shapes, return list of dicts
        print(f"   ✓ Heterogeneous mode: keeping all {len(subjects_data)} subjects with varying shapes")
        return subjects_data, subject_ids, None
    else:
        # For other groups: validate shape consistency across subjects
        print(f"   🔍 Validating shape consistency across subjects...")
        expected_shape = None
        valid_subjects_data = []
        
        for i, subject_data in enumerate(subjects_data):
            current_shape = subject_data['topos_original'].shape
            
            if expected_shape is None:
                expected_shape = current_shape
                print(f"      Expected shape set to: {expected_shape}")
            elif current_shape != expected_shape:
                print(f"      ⚠️  Skipping subject {subject_data['subject_id']}: shape mismatch {current_shape} vs expected {expected_shape}")
                continue
            
            valid_subjects_data.append(subject_data)
        
        if len(valid_subjects_data) == 0:
            print(f"   ❌ No subjects with consistent data shapes found for {group_name}")
            return None, None, None
        
        print(f"   ✓ {len(valid_subjects_data)} subjects with consistent shapes retained")
        
        # Convert to numpy arrays
        topos_orig_all = np.array([s['topos_original'] for s in valid_subjects_data])
        topos_recon_all = np.array([s['topos_reconstructed'] for s in valid_subjects_data])
        marker_names = valid_subjects_data[0]['marker_names']
        
        return topos_orig_all, topos_recon_all, marker_names


def get_markers_for_group(group_data):
    """Get all available markers for a group, handling both homogeneous and heterogeneous data."""
    if 'subjects_list' in group_data:  # heterogeneous (Control RS)
        return set().union(*[s['marker_names'] for s in group_data['subjects_list']])
    else:  # homogeneous (other groups)
        return set(group_data['marker_names'])


def get_marker_data_for_group(group_data, marker_name):
    """Get stacked data for a specific marker, handling both homogeneous and heterogeneous groups."""
    if 'subjects_list' in group_data:  # heterogeneous (Control RS)
        # Filter subjects that have this marker
        subjects_with_marker = [s for s in group_data['subjects_list'] if marker_name in s['marker_names']]
        
        if not subjects_with_marker:
            return None, None, 0
        
        # Get marker index (should be same for all subjects with this marker)
        marker_idx = subjects_with_marker[0]['marker_names'].index(marker_name)
        
        # Stack data for subjects with this marker (create 3D arrays)
        topos_orig_stacked = np.array([s['topos_original'][marker_idx] for s in subjects_with_marker])[:, np.newaxis, :]
        topos_recon_stacked = np.array([s['topos_reconstructed'][marker_idx] for s in subjects_with_marker])[:, np.newaxis, :]
        
        return topos_orig_stacked, topos_recon_stacked, len(subjects_with_marker)
        
    else:  # homogeneous (other groups)
        if marker_name not in group_data['marker_names']:
            return None, None, 0
            
        marker_idx = group_data['marker_names'].index(marker_name)
        topos_orig_stacked = group_data['topos_orig'][:, marker_idx:marker_idx+1, :]
        topos_recon_stacked = group_data['topos_recon'][:, marker_idx:marker_idx+1, :]
        
        return topos_orig_stacked, topos_recon_stacked, group_data['n_subjects']


def perform_spearman_fdr_test(topos_orig_all, topos_recon_all, marker_idx, marker_name, group_name):
    """Perform electrode-wise Spearman correlation test with FDR correction."""
    
    n_channels = topos_orig_all.shape[2]
    p_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            orig_ch = topos_orig_all[:, marker_idx, ch]  # shape: (n_subjects,)
            recon_ch = topos_recon_all[:, marker_idx, ch]
            corr, p = stats.spearmanr(orig_ch, recon_ch)
            p_values[ch] = p
        except:
            p_values[ch] = 1.0  # Conservative: assume no significance
    
    # Apply FDR correction across electrodes
    if HAS_STATSMODELS:
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        significant_count = np.sum(p_values_corrected < 0.05)
        print(f"  FDR: {marker_name} {group_name}: {significant_count}/{len(p_values_corrected)} significant channels")
    else:
        p_values_corrected = p_values
        print("  ⚠️  statsmodels not available, using uncorrected p-values")
    
    return p_values_corrected


def perform_wilcoxon_fdr_test(topos_orig_all, topos_recon_all, marker_idx, marker_name, group_name):
    """Perform electrode-wise Wilcoxon signed-rank test with FDR correction."""
    
    n_channels = topos_orig_all.shape[2]
    p_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            orig_ch = topos_orig_all[:, marker_idx, ch]  # shape: (n_subjects,)
            recon_ch = topos_recon_all[:, marker_idx, ch]
            stat, p = stats.wilcoxon(orig_ch, recon_ch)
            p_values[ch] = p
        except:
            p_values[ch] = 1.0  # Conservative: assume no significance
    
    # Apply FDR correction across electrodes
    if HAS_STATSMODELS:
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        significant_count = np.sum(p_values_corrected < 0.05)
        print(f"  FDR: {marker_name} {group_name}: {significant_count}/{len(p_values_corrected)} significant channels")
    else:
        p_values_corrected = p_values
        print("  ⚠️  statsmodels not available, using uncorrected p-values")
    
    return p_values_corrected


def compute_spearman_correlations(topos_orig_all, topos_recon_all, marker_idx, marker_name, group_name):
    """Compute electrode-wise Spearman correlation coefficients between original and reconstructed data."""
    
    n_channels = topos_orig_all.shape[2]
    correlation_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            orig_ch = topos_orig_all[:, marker_idx, ch]  # shape: (n_subjects,)
            recon_ch = topos_recon_all[:, marker_idx, ch]
            corr, p = stats.spearmanr(orig_ch, recon_ch)
            correlation_values[ch] = corr if not np.isnan(corr) else 0.0
        except:
            correlation_values[ch] = 0.0  # Conservative: assume no correlation
    
    mean_corr = np.mean(np.abs(correlation_values))
    max_corr = np.max(np.abs(correlation_values))
    print(f"  Corr: {marker_name} {group_name}: mean |ρ| = {mean_corr:.3f}, max |ρ| = {max_corr:.3f}")
    
    return correlation_values


def compute_spearman_correlations_with_pvalues(topos_orig_all, topos_recon_all, marker_idx, marker_name, group_name):
    """Compute electrode-wise Spearman correlation coefficients and p-values between original and reconstructed data."""
    
    n_channels = topos_orig_all.shape[2]
    correlation_values = np.zeros(n_channels)
    p_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            orig_ch = topos_orig_all[:, marker_idx, ch]  # shape: (n_subjects,)
            recon_ch = topos_recon_all[:, marker_idx, ch]
            corr, p = stats.spearmanr(orig_ch, recon_ch)
            correlation_values[ch] = corr if not np.isnan(corr) else 0.0
            p_values[ch] = p if not np.isnan(p) else 1.0
        except:
            correlation_values[ch] = 0.0  # Conservative: assume no correlation
            p_values[ch] = 1.0  # Conservative: assume not significant
    
    # Apply FDR correction
    if HAS_STATSMODELS:
        try:
            _, p_values_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        except:
            p_values_fdr = p_values
    else:
        p_values_fdr = p_values
    
    mean_corr = np.mean(np.abs(correlation_values))
    max_corr = np.max(np.abs(correlation_values))
    n_significant = np.sum(p_values_fdr < 0.05)
    print(f"  Corr: {marker_name} {group_name}: mean |ρ| = {mean_corr:.3f}, max |ρ| = {max_corr:.3f}, sig electrodes = {n_significant}/{n_channels}")
    
    return correlation_values, p_values_fdr


def perform_mutual_information(topos_orig_all, topos_recon_all, marker_idx, marker_name, group_name):
    """Perform electrode-wise normalized mutual information computation between original and reconstructed data."""
    
    if not HAS_SKLEARN:
        print("  ⚠️  sklearn not available, returning zeros")
        return np.zeros(topos_orig_all.shape[2])
    
    n_channels = topos_orig_all.shape[2]
    n_subjects = topos_orig_all.shape[0]
    
    # Check if we have enough subjects for meaningful MI computation
    if n_subjects < 5:
        print(f"  ⚠️  {marker_name} {group_name}: insufficient subjects ({n_subjects}), returning zeros")
        return np.zeros(n_channels)
    
    mi_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            orig_ch = topos_orig_all[:, marker_idx, ch]  # shape: (n_subjects,)
            recon_ch = topos_recon_all[:, marker_idx, ch]
            
            # Check for sufficient variance
            if np.var(orig_ch) < 1e-10 or np.var(recon_ch) < 1e-10:
                mi_values[ch] = 0.0
                continue
            
            # Use mutual_info_regression for continuous variables
            # Reshape orig_ch as feature matrix, recon_ch as target
            X = orig_ch.reshape(-1, 1)
            y = recon_ch
            
            # Compute mutual information
            mi = mutual_info_regression(X, y, random_state=42)[0]
            
            # Normalize MI to [0, 1] range using proper entropy estimation
            # Use histogram with proper probability (not density)
            def estimate_entropy(data, n_bins=None):
                if n_bins is None:
                    n_bins = min(5, len(data)//2)
                if n_bins < 2:
                    return 0.0
                hist, _ = np.histogram(data, bins=n_bins, density=False)
                hist = hist.astype(float)
                hist = hist / hist.sum()  # Convert to probabilities
                hist = hist[hist > 0]  # Remove zero probabilities
                if len(hist) == 0:
                    return 0.0
                return -np.sum(hist * np.log(hist + 1e-10))
            
            h_orig = estimate_entropy(orig_ch)
            h_recon = estimate_entropy(recon_ch)
            
            # Use symmetric normalization: 2*MI/(H(X)+H(Y))
            if (h_orig + h_recon) > 1e-10:
                mi_normalized = 2 * mi / (h_orig + h_recon)
                mi_values[ch] = min(mi_normalized, 1.0)  # Cap at 1.0 for numerical stability
            else:
                mi_values[ch] = 0.0
                
        except Exception as e:
            mi_values[ch] = 0.0  # Conservative: assume no mutual information
    
    mean_mi = np.mean(mi_values)
    max_mi = np.max(mi_values)
    print(f"  MI: {marker_name} {group_name}: mean MI = {mean_mi:.3f}, max MI = {max_mi:.3f}")
    
    return mi_values


def create_biomarker_comparison_grid(output_dir):
    """Create 9x4 grid topographic comparison plot with Spearman tests."""
    
    print("=" * 60)
    print("Creating Biomarker Group Comparison Grid")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
                
                # Print shape distribution
                shapes = [s['topos_original'].shape[0] for s in subjects_list]
                shape_counts = {}
                for shape in shapes:
                    shape_counts[shape] = shape_counts.get(shape, 0) + 1
                print(f"      Shape distribution: {shape_counts}")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    marker_names_ref = first_group['marker_names']
    
    # Set up montage and sphere
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure: 9 rows × 4 columns
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(20, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=30, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Perform Spearman test and create p-value map
                p_values_corrected = perform_spearman_fdr_test(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(256)  # 256 channels
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im, _ = mne.viz.plot_topomap(p_map, info, axes=ax,
                                            vlim=(0, 2), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for p-values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, ticks=[0, 1, 2])
                    cbar.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                    cbar.ax.tick_params(labelsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_spearman_values_extended_with_fdr(output_dir):
    """Create extended spearman correlation grid with FDR-corrected p-values for MCS, UWS, Control groups only."""
    
    print("=" * 60)
    print("Creating Extended Biomarker Group Comparison Grid (Spearman Values with FDR)")
    print("=" * 60)
    
    # Define data directories (exclude Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None
    }
    
        
    # Define extended biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information'),
        ('cnv_detailed_cnvslope', 'CNV_slope'),
        ('p1_topography_timelockedtopo', 'P1_tlt'),
        ('p3a_topography_timelockedtopo', 'P3a_tlt'),
        ('p3b_topography_timelockedtopo', 'P3b_tlt'),
        ('timelockedcontrast_lsgs_ldgd_timelockedcontrast', 'LSGS_LDGS_tlc'),
        ('timelockedcontrast_lsgd_ldgs_timelockedcontrast', 'LSGD_LDGS_tlc'),
        ('timelockedcontrast_ld_ls_timelockedcontrast', 'LD_LS_tlc'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN_tlc'),
        ('timelockedcontrast_p3a_timelockedcontrast', 'P3a_tlc'),
        ('timelockedcontrast_gd_gs_timelockedcontrast', 'GD_GS_tlc'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b_tlc')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=False
        )
        
        # Other groups: return numpy arrays
        topos_orig, topos_recon, marker_names = result
        
        if topos_orig is not None:
            groups_data[group_name] = {
                'topos_orig': topos_orig,
                'topos_recon': topos_recon,
                'marker_names': marker_names,
                'n_subjects': topos_orig.shape[0],
                'is_homogeneous': True
            }
            print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
        else:
            print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating extended Spearman values grid plot with FDR correction ({len(available_biomarkers)} biomarkers × {len(groups_data)} groups)")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(12, max(16, n_biomarkers * 1.5)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Perform Spearman test and create p-value map (same as existing FDR function)
                p_values_corrected = perform_spearman_fdr_test(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create discrete p-value map for visualization (same as existing FDR function)
                p_map = np.zeros(256)  # 256 channels
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors (same as existing FDR function)
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im, _ = mne.viz.plot_topomap(p_map, info, axes=ax,
                                            vlim=(0, 2), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for p-values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, ticks=[0, 1, 2])
                    cbar.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                    cbar.ax.tick_params(labelsize=20)
                    
                    # Add significance legend
                   # n_significant = np.sum(p_values_corrected < 0.05)
                   # ax.text(1.1, 0.5, f'p<0.05 FDR\n{n_significant} sig', 
                   #        transform=ax.transAxes, fontsize=12, 
                   #        verticalalignment='center', color='black')
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_spearman_values_extended_with_fdr.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Extended biomarker group comparison grid (Spearman values with FDR) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_spearman_values_extended(output_dir):
    """Create extended spearman correlation grid with additional topographic markers for MCS, UWS, Control groups only."""
    
    print("=" * 60)
    print("Creating Extended Biomarker Group Comparison Grid (Spearman Values)")
    print("=" * 60)
    
    # Define data directories (exclude Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None
    }
    
        
    # Define extended biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information'),
        ('cnv_detailed_cnvslope', 'CNV_slope'),
        ('p1_topography_timelockedtopo', 'P1_tlt'),
        ('p3a_topography_timelockedtopo', 'P3a_tlt'),
        ('p3b_topography_timelockedtopo', 'P3b_tlt'),
        ('timelockedcontrast_lsgs_ldgd_timelockedcontrast', 'LSGS_LDGS_tlc'),
        ('timelockedcontrast_lsgd_ldgs_timelockedcontrast', 'LSGD_LDGS_tlc'),
        ('timelockedcontrast_ld_ls_timelockedcontrast', 'LD_LS_tlc'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN_tlc'),
        ('timelockedcontrast_p3a_timelockedcontrast', 'P3a_tlc'),
        ('timelockedcontrast_gd_gs_timelockedcontrast', 'GD_GS_tlc'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b_tlc')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=False
        )
        
        # Other groups: return numpy arrays
        topos_orig, topos_recon, marker_names = result
        
        if topos_orig is not None:
            groups_data[group_name] = {
                'topos_orig': topos_orig,
                'topos_recon': topos_recon,
                'marker_names': marker_names,
                'n_subjects': topos_orig.shape[0],
                'is_homogeneous': True
            }
            print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
        else:
            print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating extended Spearman values grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(12, max(16, n_biomarkers * 1.5)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute Spearman correlation values
                correlation_values = compute_spearman_correlations(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create correlation map for visualization (continuous values -1 to 1)
                corr_map = correlation_values
                
                # Use RdBu_r colormap for correlation values (-1 to 1)
                cmap = 'RdBu_r'
                
                # Plot correlation map
                im, _ = mne.viz.plot_topomap(corr_map, info, axes=ax,
                                            vlim=(-1, 1), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for correlation values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=20)
                    cbar.set_label('Spearman ρ', fontsize=15)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_spearman_values_extended.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Extended biomarker group comparison grid (Spearman values) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_spearman_values(output_dir):
    """Create 9x4 grid topographic comparison plot showing Spearman correlation values."""
    
    print("=" * 60)
    print("Creating Biomarker Group Comparison Grid (Spearman Values)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating Spearman values grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    if 'is_homogeneous' in first_group:
        topos_orig_ref = first_group['topos_orig']
        topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    else:
        # For heterogeneous data, use first subject
        topos_orig_ref_mean = first_group['subjects_list'][0]['topos_original']
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref_mean.shape[1], topos_orig_ref_mean)
    
    # Create figure: 9 rows × 4 columns
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(20, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=30, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute Spearman correlation values
                correlation_values = compute_spearman_correlations(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create correlation map for visualization (continuous values -1 to 1)
                corr_map = correlation_values
                
                # Use RdBu_r colormap for correlation values (-1 to 1)
                cmap = 'RdBu_r'
                
                # Plot correlation map
                im, _ = mne.viz.plot_topomap(corr_map, info, axes=ax,
                                            vlim=(-1, 1), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for correlation values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=20)
                    cbar.set_label('Spearman ρ', fontsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_spearman_values.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid (Spearman values) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_wilcoxon(output_dir):
    """Create 9x4 grid topographic comparison plot with Wilcoxon tests."""
    
    print("=" * 60)
    print("Creating Biomarker Group Comparison Grid (Wilcoxon)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating Wilcoxon grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    if 'is_homogeneous' in first_group:
        topos_orig_ref = first_group['topos_orig']
        topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    else:
        # For heterogeneous data, use first subject
        topos_orig_ref_mean = first_group['subjects_list'][0]['topos_original']
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref_mean.shape[1], topos_orig_ref_mean)
    
    # Create figure: 9 rows × 4 columns
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(20, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=25)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Perform Wilcoxon test and create p-value map
                p_values_corrected = perform_wilcoxon_fdr_test(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(256)  # 256 channels
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im, _ = mne.viz.plot_topomap(p_map, info, axes=ax,
                                            vlim=(0, 2), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for p-values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, ticks=[0, 1, 2])
                    cbar.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                    cbar.ax.tick_params(labelsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_wilcoxon.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid (Wilcoxon) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_mi(output_dir):
    """Create 9x4 grid topographic comparison plot with Mutual Information."""
    
    print("=" * 60)
    print("Creating Biomarker Group Comparison Grid (Mutual Information)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating MI grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    if 'is_homogeneous' in first_group:
        topos_orig_ref = first_group['topos_orig']
        topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    else:
        # For heterogeneous data, use first subject
        topos_orig_ref_mean = first_group['subjects_list'][0]['topos_original']
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref_mean.shape[1], topos_orig_ref_mean)
    
    # Create figure: 9 rows × 4 columns
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(20, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=25)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Perform Mutual Information computation
                mi_values = perform_mutual_information(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create MI map for visualization (continuous values 0-1)
                mi_map = mi_values
                
                # Use viridis colormap for continuous MI values
                cmap = 'viridis'
                
                # Plot MI map
                im, _ = mne.viz.plot_topomap(mi_map, info, axes=ax,
                                            vlim=(0, 1), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for MI values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=20)
                    cbar.set_label('Normalized MI', fontsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_mi.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid (Mutual Information) saved to: {output_path}")
    return output_path


def main():
    """Main function to create Spearman, Wilcoxon, Mutual Information grid, Mutual Information boxplot, Spearman correlation boxplot, and Spearman correlation boxplot with p < 0.05."""
    
    # Configuration
    output_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/OHBM"
    
    print("=" * 80)
    print("Creating Biomarker Group Comparison Plots")
    print("=" * 80)
    
    # Create the Spearman comparison plot
    print("\n1. Creating Spearman correlation test plot...")
    spearman_plot_path = create_biomarker_comparison_grid(output_dir)
    
    # Create the Wilcoxon comparison plot
    print("\n2. Creating Wilcoxon signed-rank test plot...")
    wilcoxon_plot_path = create_biomarker_comparison_grid_wilcoxon(output_dir)
    
    # Create the Mutual Information comparison plot
    print("\n3. Creating Mutual Information grid plot...")
    mi_plot_path = create_biomarker_comparison_grid_mi(output_dir)
    
    # Create the Mutual Information boxplot
    print("\n4. Creating Mutual Information boxplot (subject level)...")
    mi_boxplot_path = create_mutual_information_boxplot(output_dir)
    
    # Create the Spearman correlation boxplot
    print("\n5. Creating Spearman correlation boxplot (subject level)...")
    spearman_boxplot_path = create_spearman_correlation_boxplot(output_dir)
    
    # Create the Spearman correlation boxplot with p < 0.05
    print("\n6. Creating Spearman correlation boxplot with p < 0.05 (subject level)...")
    spearman_p05_boxplot_path = create_spearman_correlation_boxplot_p05(output_dir)
    
    # Create the Spearman correlation values plot
    print("\n7. Creating Spearman correlation values plot...")
    spearman_values_plot_path = create_biomarker_comparison_grid_spearman_values(output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("Biomarker group comparison completed successfully!")
    print("=" * 80)
    
    if spearman_plot_path:
        print(f"✅ Spearman plot saved: {spearman_plot_path}")
    else:
        print("❌ Failed to create Spearman plot")
    
    if wilcoxon_plot_path:
        print(f"✅ Wilcoxon plot saved: {wilcoxon_plot_path}")
    else:
        print("❌ Failed to create Wilcoxon plot")
    
    if mi_plot_path:
        print(f"✅ Mutual Information plot saved: {mi_plot_path}")
    else:
        print("❌ Failed to create Mutual Information plot")
    
    if mi_boxplot_path:
        print(f"✅ Mutual Information boxplot saved: {mi_boxplot_path}")
    else:
        print("❌ Failed to create Mutual Information boxplot")
    
    if spearman_boxplot_path:
        print(f"✅ Spearman correlation boxplot saved: {spearman_boxplot_path}")
    else:
        print("❌ Failed to create Spearman correlation boxplot")
    
    if spearman_p05_boxplot_path:
        print(f"✅ Spearman correlation boxplot with p < 0.05 saved: {spearman_p05_boxplot_path}")
    else:
        print("❌ Failed to create Spearman correlation boxplot with p < 0.05")
    
    if spearman_values_plot_path:
        print(f"✅ Spearman correlation values plot saved: {spearman_values_plot_path}")
    else:
        print("❌ Failed to create Spearman correlation values plot")
    
    print("=" * 80)


def compute_subject_level_mi(topo_orig, topo_recon, marker_name, subject_id):
    """Compute normalized mutual information between original and reconstructed topoplot for a single subject."""
    
    if not HAS_SKLEARN:
        return 0.0
    
    try:
        # topo_orig and topo_recon are 1D arrays of channel values for this marker
        # Check for sufficient variance
        if np.var(topo_orig) < 1e-10 or np.var(topo_recon) < 1e-10:
            return 0.0
        
        # Use mutual_info_regression for continuous variables
        # Reshape orig as feature matrix, recon as target
        X = topo_orig.reshape(-1, 1)
        y = topo_recon
        
        # Compute mutual information
        mi = mutual_info_regression(X, y, random_state=42)[0]
        
        # Normalize MI to [0, 1] range using proper entropy estimation
        # Use histogram with proper probability (not density)
        def estimate_entropy(data, n_bins=None):
            if n_bins is None:
                n_bins = min(10, len(data)//2)
            if n_bins < 2:
                return 0.0
            hist, _ = np.histogram(data, bins=n_bins, density=False)
            hist = hist.astype(float)
            hist = hist / hist.sum()  # Convert to probabilities
            hist = hist[hist > 0]  # Remove zero probabilities
            if len(hist) == 0:
                return 0.0
            return -np.sum(hist * np.log(hist + 1e-10))
        
        h_orig = estimate_entropy(topo_orig)
        h_recon = estimate_entropy(topo_recon)
        
        # Use symmetric normalization: 2*MI/(H(X)+H(Y))
        if (h_orig + h_recon) > 1e-10:
            mi_normalized = 2 * mi / (h_orig + h_recon)
            mi_normalized = min(mi_normalized, 1.0)  # Cap at 1.0 for numerical stability
            return mi_normalized
        else:
            return 0.0
            
    except Exception as e:
        print(f"    Error computing MI for {subject_id} {marker_name}: {e}")
        return 0.0


def create_mutual_information_boxplot(output_dir):
    """Create boxplot showing subject-level mutual information across groups and markers."""
    
    print("=" * 60)
    print("Creating Mutual Information Boxplot (Subject Level)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot (same as MI grid)
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Computing subject-level MI for {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Compute subject-level MI for all marker × group combinations
    mi_data = {}  # mi_data[marker_name][group_name] = [mi_values_per_subject]
    
    for marker_name, display_name in available_biomarkers:
        print(f"\n🔍 Computing MI for {display_name}...")
        mi_data[marker_name] = {}
        
        for group_name, group_data in groups_data.items():
            print(f"  Processing {group_name}...")
            mi_values = []
            
            if 'subjects_list' in group_data:  # heterogeneous (Control RS)
                for subject_data in group_data['subjects_list']:
                    if marker_name in subject_data['marker_names']:
                        marker_idx = subject_data['marker_names'].index(marker_name)
                        topo_orig = subject_data['topos_original'][marker_idx]
                        topo_recon = subject_data['topos_reconstructed'][marker_idx]
                        
                        mi_val = compute_subject_level_mi(
                            topo_orig, topo_recon, marker_name, subject_data['subject_id']
                        )
                        mi_values.append(mi_val)
                        
            else:  # homogeneous (other groups)
                if marker_name in group_data['marker_names']:
                    marker_idx = group_data['marker_names'].index(marker_name)
                    
                    for subj_idx in range(group_data['n_subjects']):
                        topo_orig = group_data['topos_orig'][subj_idx, marker_idx, :]
                        topo_recon = group_data['topos_recon'][subj_idx, marker_idx, :]
                        subject_id = f"subj_{subj_idx}"
                        
                        mi_val = compute_subject_level_mi(
                            topo_orig, topo_recon, marker_name, subject_id
                        )
                        mi_values.append(mi_val)
            
            mi_data[marker_name][group_name] = mi_values
            print(f"    {group_name}: {len(mi_values)} subjects, mean MI = {np.mean(mi_values):.3f}" if mi_values else f"    {group_name}: 0 subjects")
    
    # Create boxplot figure
    n_biomarkers = len(available_biomarkers)
    fig, axes = plt.subplots(n_biomarkers, 1, figsize=(22, max(16, n_biomarkers * 2)), sharex=True)
    
    if n_biomarkers == 1:
        axes = [axes]  # Make it iterable
    
    group_names = list(groups_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Standard matplotlib colors
    
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        ax = axes[row]
        
        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        
        for group_name in group_names:
            mi_values = mi_data[marker_name][group_name]
            if mi_values:  # Only add if data exists
                boxplot_data.append(mi_values)
                boxplot_labels.append(group_name)
        
        if boxplot_data:
            # Create boxplot
            bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(boxplot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add grid and formatting
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('MI', fontsize=20)
            ax.set_ylim([-0.3, 1.15])  # Consistent y-axis for all rows
            
            # Add row label on the left
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=25, rotation=0)
            
            # Add individual subject points over the boxplot
            for i, (group_name, values) in enumerate([(lbl, data) for lbl, data in zip(boxplot_labels, boxplot_data)]):
                # Add jitter to x positions to avoid overlap
                x_positions = np.random.normal(i+1, 0.05, size=len(values))
                ax.scatter(x_positions, values, alpha=0.6, s=20, color=colors[i % len(colors)], zorder=3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_ylabel('Normalized MI', fontsize=15)
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=25, rotation=90)
    
    # Set x-axis label for bottom plot
   # axes[-1].set_xlabel('Subject Group', fontsize=18)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'mutual_information_plt.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Mutual Information boxplot saved to: {output_path}")
    return output_path


def compute_subject_level_spearman(topo_orig, topo_recon, marker_name, subject_id):
    """Compute Spearman correlation between original and reconstructed topoplot for a single subject."""
    
    try:
        # topo_orig and topo_recon are 1D arrays of channel values for this marker
        # Check for sufficient variance
        if np.var(topo_orig) < 1e-10 or np.var(topo_recon) < 1e-10:
            return 0.0
        
        # Compute Spearman correlation
        corr, p = stats.spearmanr(topo_orig, topo_recon)
        
        # Return correlation coefficient (ignore p-value for this visualization)
        if np.isnan(corr):
            return 0.0
        return corr
            
    except Exception as e:
        print(f"    Error computing Spearman for {subject_id} {marker_name}: {e}")
        return 0.0


def create_spearman_correlation_boxplot(output_dir):
    """Create boxplot showing subject-level Spearman correlation across groups and markers."""
    
    print("=" * 60)
    print("Creating Spearman Correlation Boxplot (Subject Level)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot (same as MI grid)
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Computing subject-level Spearman correlation for {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Compute subject-level Spearman correlation for all marker × group combinations
    spearman_data = {}  # spearman_data[marker_name][group_name] = [spearman_values_per_subject]
    
    for marker_name, display_name in available_biomarkers:
        print(f"\n🔍 Computing Spearman correlation for {display_name}...")
        spearman_data[marker_name] = {}
        
        for group_name, group_data in groups_data.items():
            print(f"  Processing {group_name}...")
            spearman_values = []
            
            if 'subjects_list' in group_data:  # heterogeneous (Control RS)
                for subject_data in group_data['subjects_list']:
                    if marker_name in subject_data['marker_names']:
                        marker_idx = subject_data['marker_names'].index(marker_name)
                        topo_orig = subject_data['topos_original'][marker_idx]
                        topo_recon = subject_data['topos_reconstructed'][marker_idx]
                        
                        spearman_val = compute_subject_level_spearman(
                            topo_orig, topo_recon, marker_name, subject_data['subject_id']
                        )
                        spearman_values.append(spearman_val)
                        
            else:  # homogeneous (other groups)
                if marker_name in group_data['marker_names']:
                    marker_idx = group_data['marker_names'].index(marker_name)
                    
                    for subj_idx in range(group_data['n_subjects']):
                        topo_orig = group_data['topos_orig'][subj_idx, marker_idx, :]
                        topo_recon = group_data['topos_recon'][subj_idx, marker_idx, :]
                        subject_id = f"subj_{subj_idx}"
                        
                        spearman_val = compute_subject_level_spearman(
                            topo_orig, topo_recon, marker_name, subject_id
                        )
                        spearman_values.append(spearman_val)
            
            spearman_data[marker_name][group_name] = spearman_values
            print(f"    {group_name}: {len(spearman_values)} subjects, mean Spearman = {np.mean(spearman_values):.3f}" if spearman_values else f"    {group_name}: 0 subjects")
    
    # Create boxplot figure
    n_biomarkers = len(available_biomarkers)
    figsize=(22, max(18, n_biomarkers * 2))
    fig, axes = plt.subplots(n_biomarkers, 1, figsize=figsize, sharex=True)
    
    if n_biomarkers == 1:
        axes = [axes]  # Make it iterable
    
    group_names = list(groups_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Standard matplotlib colors
    
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        ax = axes[row]
        
        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        
        for group_name in group_names:
            spearman_values = spearman_data[marker_name][group_name]
            if spearman_values:  # Only add if data exists
                boxplot_data.append(spearman_values)
                boxplot_labels.append(group_name)
        
        if boxplot_data:
            # Create boxplot
            bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(boxplot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add grid and formatting
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Spearman ρ', fontsize=20)
            ax.set_ylim([-1.05, 1.05])  # Consistent y-axis for all rows (correlation range [-1,1] with padding)
            
            # Add row label on the left
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=30, rotation=0)
            
            # Add individual subject points over the boxplot
            for i, (group_name, values) in enumerate([(lbl, data) for lbl, data in zip(boxplot_labels, boxplot_data)]):
                # Add jitter to x positions to avoid overlap
                x_positions = np.random.normal(i+1, 0.05, size=len(values))
                ax.scatter(x_positions, values, alpha=0.6, s=20, color=colors[i % len(colors)], zorder=3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_ylabel('Spearman ρ', fontsize=10)
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=18, rotation=90)
    
    # Set x-axis label for bottom plot
   # axes[-1].set_xlabel('Subject Group', fontsize=18)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'spearman_correlation_plt.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Spearman correlation boxplot saved to: {output_path}")
    return output_path


def create_spearman_correlation_boxplot_p05(output_dir):
    """Create boxplot showing subject-level Spearman correlation across groups and markers with p < 0.05 significance (no correction)."""
    
    print("=" * 60)
    print("Creating Spearman Correlation Boxplot with p < 0.05 (Subject Level)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define biomarkers to plot (same as MI grid)
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Computing subject-level Spearman correlation for {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Compute subject-level Spearman correlation for all marker × group combinations
    spearman_data = {}  # spearman_data[marker_name][group_name] = [spearman_values_per_subject]
    
    for marker_name, display_name in available_biomarkers:
        print(f"\n🔍 Computing Spearman correlation for {display_name}...")
        spearman_data[marker_name] = {}
        
        for group_name, group_data in groups_data.items():
            print(f"  Processing {group_name}...")
            spearman_values = []
            
            if 'subjects_list' in group_data:  # heterogeneous (Control RS)
                for subject_data in group_data['subjects_list']:
                    if marker_name in subject_data['marker_names']:
                        marker_idx = subject_data['marker_names'].index(marker_name)
                        topo_orig = subject_data['topos_original'][marker_idx]
                        topo_recon = subject_data['topos_reconstructed'][marker_idx]
                        
                        spearman_val = compute_subject_level_spearman(
                            topo_orig, topo_recon, marker_name, subject_data['subject_id']
                        )
                        spearman_values.append(spearman_val)
                        
            else:  # homogeneous (other groups)
                if marker_name in group_data['marker_names']:
                    marker_idx = group_data['marker_names'].index(marker_name)
                    
                    for subj_idx in range(group_data['n_subjects']):
                        topo_orig = group_data['topos_orig'][subj_idx, marker_idx, :]
                        topo_recon = group_data['topos_recon'][subj_idx, marker_idx, :]
                        subject_id = f"subj_{subj_idx}"
                        
                        spearman_val = compute_subject_level_spearman(
                            topo_orig, topo_recon, marker_name, subject_id
                        )
                        spearman_values.append(spearman_val)
            
            spearman_data[marker_name][group_name] = spearman_values
            print(f"    {group_name}: {len(spearman_values)} subjects, mean Spearman = {np.mean(spearman_values):.3f}" if spearman_values else f"    {group_name}: 0 subjects")
    
    # Perform statistical tests with p < 0.05 threshold
    print(f"\n🧪 Performing statistical tests with p < 0.05 threshold...")
    group_names = list(groups_data.keys())
    n_groups = len(group_names)
    n_biomarkers = len(available_biomarkers)
    
    # Collect all p-values for significance testing
    all_p_values = []
    all_comparisons = []  # List of (marker_name, group1, group2) tuples
    
    for marker_name, display_name in available_biomarkers:
        print(f"\n  Testing {display_name}...")
        
        # Perform all pairwise comparisons
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                group1 = group_names[i]
                group2 = group_names[j]
                
                values1 = spearman_data[marker_name][group1]
                values2 = spearman_data[marker_name][group2]
                
                if len(values1) > 0 and len(values2) > 0:
                    # Check for sufficient variance before performing test
                    var1 = np.var(values1)
                    var2 = np.var(values2)
                    
                    if var1 < 1e-10 or var2 < 1e-10:
                        print(f"    {group1} vs {group2}: skipped (zero variance)")
                        continue
                    
                    # Perform Mann-Whitney U test
                    stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    
                    all_p_values.append(p_value)
                    all_comparisons.append((marker_name, group1, group2))
                    print(f"    {group1} vs {group2}: p = {p_value:.6f}")
    
    # Apply simple p < 0.05 threshold (no correction)
    if len(all_p_values) > 0:
        print(f"\n   Simple p < 0.05 threshold applied to {len(all_p_values)} comparisons")
        n_significant = 0
        
        # Store significant comparisons
        significant_comparisons = {}  # significant_comparisons[marker_name] = [(group1, group2, p_value), ...]
        
        for i, (marker_name, group1, group2) in enumerate(all_comparisons):
            p_value = all_p_values[i]
            if p_value < 0.05:
                if marker_name not in significant_comparisons:
                    significant_comparisons[marker_name] = []
                significant_comparisons[marker_name].append((group1, group2, p_value))
                n_significant += 1
                print(f"    {marker_name}: {group1} vs {group2}: p = {p_value:.6f} *")
        
        print(f"   Significant with p < 0.05: {n_significant}/{len(all_p_values)} comparisons")
    else:
        print("   No valid comparisons")
        significant_comparisons = {}
    
    # Create boxplot figure with significance markers
    fig, axes = plt.subplots(n_biomarkers, 1, figsize=(18, max(18, n_biomarkers * 2.5)), sharex=True)
    
    if n_biomarkers == 1:
        axes = [axes]  # Make it iterable
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Standard matplotlib colors
    
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        ax = axes[row]
        
        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        
        for group_name in group_names:
            spearman_values = spearman_data[marker_name][group_name]
            if spearman_values:  # Only add if data exists
                boxplot_data.append(spearman_values)
                boxplot_labels.append(group_name)
        
        if boxplot_data:
            # Create boxplot
            bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(boxplot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add grid and formatting
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Spearman ρ', fontsize=18)
            ax.set_ylim([-1.05, 1.05])  # Consistent y-axis for all rows (correlation range [-1,1] with padding)
            
            # Add row label on the left
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=18, rotation=0)
            
            # Add individual subject points over the boxplot
            for i, (group_name, values) in enumerate([(lbl, data) for lbl, data in zip(boxplot_labels, boxplot_data)]):
                # Add jitter to x positions to avoid overlap
                x_positions = np.random.normal(i+1, 0.05, size=len(values))
                ax.scatter(x_positions, values, alpha=0.6, s=20, color=colors[i % len(colors)], zorder=3)
            
            # Add significance markers
            if marker_name in significant_comparisons:
                for idx, (group1, group2, p_value) in enumerate(significant_comparisons[marker_name]):
                    idx1 = boxplot_labels.index(group1)
                    idx2 = boxplot_labels.index(group2)
                    
                    # Get max values for positioning
                    max_val1 = max(boxplot_data[idx1])
                    max_val2 = max(boxplot_data[idx2])
                    y_pos = max(max_val1, max_val2) + 0.05  # Position above boxes
                    
                    # Ensure asterisk stays within y-axis limits (1.05)
                    if y_pos > 0.98:  # Leave room for asterisk and line
                        y_pos = 0.98
                    
                    # Adjust for multiple significant comparisons to avoid overlap
                    y_offset = idx * 0.03  # Stack multiple comparisons vertically
                    y_line = y_pos + y_offset
                    
                    # Ensure still within bounds
                    if y_line > 0.98:
                        y_line = 0.98 - (idx * 0.01)  # Adjust downward if needed
                    
                    # Draw connecting line and asterisk
                    x1, x2 = idx1 + 1, idx2 + 1  # Boxplot positions are 1-indexed
                    
                    # Draw horizontal line
                    ax.plot([x1, x2], [y_line, y_line], 'k-', linewidth=1)
                    
                    # Draw vertical lines down to boxes
                    ax.plot([x1, x1], [y_line - 0.02, y_line], 'k-', linewidth=1)
                    ax.plot([x2, x2], [y_line - 0.02, y_line], 'k-', linewidth=1)
                    
                    # Add asterisk
                    ax.text((x1 + x2) / 2, y_line + 0.02, '*', ha='center', va='bottom', 
                           fontsize=20, fontweight='bold')
                    
                    # Add p-value text (optional, can be commented out for cleaner look)
                    # ax.text((x1 + x2) / 2, y_line + 0.05, f'p={p_value:.3f}', 
                    #        ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_ylabel('Spearman ρ', fontsize=10)
            ax.text(-0.15, 0.5, display_name, transform=ax.transAxes,
                   ha='right', va='center', fontsize=18, rotation=90)
    
    # Add significance explanation at the bottom
    fig.text(0.5, 0.02, '* Significant with p < 0.05 (no multiple comparison correction)', 
            ha='center', va='bottom', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for significance explanation
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'spearman_correlation_p05_plt.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Spearman correlation boxplot with p < 0.05 saved to: {output_path}")
    return output_path


def create_biomarker_difference_grid(output_dir):
    """Create grid topographic plot showing original vs reconstructed differences with row-wise shared color bars."""
    
    print("=" * 60)
    print("Creating Biomarker Difference Grid (Δ)")
    print("=" * 60)
    
    # Define data directories (exclude Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None
    }
    
    # Define biomarkers to plot (same as original spearman plot)
    biomarker_specs = [
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('beta_relative_spectralpower', 'Beta Normalized'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('gamma_relative_spectralpower', 'Gamma Normalized'),
        ('theta_relative_spectralpower', 'Theta Normalized'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity'),
        ('wsmi_theta_symbolicmutualinformation', 'Symbolic Mutual Information')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=False
        )
        
        # Other groups: return numpy arrays
        topos_orig, topos_recon, marker_names = result
        
        if topos_orig is not None:
            groups_data[group_name] = {
                'topos_orig': topos_orig,
                'topos_recon': topos_recon,
                'marker_names': marker_names,
                'n_subjects': topos_orig.shape[0],
                'is_homogeneous': True
            }
            print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
        else:
            print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating difference grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(15, max(16, n_biomarkers * 1.5)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Process each row to compute differences and determine row-wise color limits
    row_diff_maps = []
    row_vlims = []
    
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        row_diffs = []
        
        for col, group_name in enumerate(group_names):
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute mean difference across subjects for this marker and group
                # orig_data shape: (n_subjects, 1, n_channels)
                # recon_data shape: (n_subjects, 1, n_channels)
                mean_orig = np.mean(marker_orig_data[:, 0, :], axis=0)  # shape: (n_channels,)
                mean_recon = np.mean(marker_recon_data[:, 0, :], axis=0)  # shape: (n_channels,)
                diff_map = mean_orig - mean_recon  # shape: (n_channels,)
                row_diffs.append(diff_map)
            else:
                # Marker not available - use zeros
                row_diffs.append(np.zeros(256))
        
        row_diff_maps.append(row_diffs)
        
        # Compute row-wise color limits (max absolute value across all groups in this row)
        if row_diffs:
            all_diffs = np.concatenate(row_diffs)
            vmax = np.max(np.abs(all_diffs))
            row_vlims.append((-vmax, vmax))
        else:
            row_vlims.append((-1, 1))
    
    # Plot each biomarker × group combination with row-wise shared color limits
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Use pre-computed difference map
                diff_map = row_diff_maps[row][col]
                
                # Use RdBu_r colormap (same as spearman) with row-wise limits
                cmap = 'RdBu_r'
                vlim = row_vlims[row]
                
                # Plot difference map
                im, _ = mne.viz.plot_topomap(diff_map, info, axes=ax,
                                            vlim=vlim, cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for difference values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=15)
                    cbar.set_label('Δ', fontsize=20, rotation=0)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_differences.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid (differences) saved to: {output_path}")
    return output_path


def create_biomarker_difference_grid_selected(output_dir):
    """Create grid topographic plot showing original vs reconstructed differences for selected biomarkers."""
    
    print("=" * 60)
    print("Creating Selected Biomarker Difference Grid (Δ)")
    print("=" * 60)
    
    # Define data directories (exclude Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None
    }
    
    # Define selected biomarkers to plot with Greek symbols
    biomarker_specs = [
        ('cnv_detailed_cnvslope', 'CNV'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=False
        )
        
        # Other groups: return numpy arrays
        topos_orig, topos_recon, marker_names = result
        
        if topos_orig is not None:
            groups_data[group_name] = {
                'topos_orig': topos_orig,
                'topos_recon': topos_recon,
                'marker_names': marker_names,
                'n_subjects': topos_orig.shape[0],
                'is_homogeneous': True
            }
            print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
        else:
            print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating selected difference grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(14, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Process each row to compute differences and determine row-wise color limits
    row_diff_maps = []
    row_vlims = []
    
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        row_diffs = []
        
        for col, group_name in enumerate(group_names):
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute mean difference across subjects for this marker and group
                # orig_data shape: (n_subjects, 1, n_channels)
                # recon_data shape: (n_subjects, 1, n_channels)
                mean_orig = np.mean(marker_orig_data[:, 0, :], axis=0)  # shape: (n_channels,)
                mean_recon = np.mean(marker_recon_data[:, 0, :], axis=0)  # shape: (n_channels,)
                diff_map = mean_orig - mean_recon  # shape: (n_channels,)
                row_diffs.append(diff_map)
            else:
                # Marker not available - use zeros
                row_diffs.append(np.zeros(256))
        
        row_diff_maps.append(row_diffs)
        
        # Compute row-wise color limits (max absolute value across all groups in this row)
        if row_diffs:
            all_diffs = np.concatenate(row_diffs)
            vmax = np.max(np.abs(all_diffs))
            row_vlims.append((-vmax, vmax))
        else:
            row_vlims.append((-1, 1))
    
    # Plot each biomarker × group combination with row-wise shared color limits
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Use pre-computed difference map
                diff_map = row_diff_maps[row][col]
                
                # Use RdBu_r colormap (same as spearman) with row-wise limits
                cmap = 'RdBu_r'
                vlim = row_vlims[row]
                
                # Plot difference map
                im, _ = mne.viz.plot_topomap(diff_map, info, axes=ax,
                                            vlim=vlim, cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for difference values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=20)
                    cbar.set_label('Δ', fontsize=20, rotation=0, labelpad=15)  # Increase labelpad value
                    offset_text = cbar.ax.yaxis.get_offset_text()
                    offset_text.set_fontsize(12)
                    offset_text.set_horizontalalignment('right')  # Align to the right

            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_differences_selected.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Selected biomarker group comparison grid (differences) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_selected_fdr(output_dir):
    """Create grid topographic comparison plot with FDR-corrected Spearman tests for selected biomarkers."""
    
    print("=" * 60)
    print("Creating Selected Biomarker Group Comparison Grid (FDR)")
    print("=" * 60)
    
    # Define data directories
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control LG': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control LG': None,
        'Control RS': None
    }
    
    # Define selected biomarkers to plot with Greek symbols
    biomarker_specs = [
        ('cnv_detailed_cnvslope', 'CNV'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Use heterogeneous loading for Control RS
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list)
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating selected FDR grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    if 'is_homogeneous' in first_group:
        topos_orig_ref = first_group['topos_orig']
        topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    else:
        # For heterogeneous data, use first subject
        topos_orig_ref_mean = first_group['subjects_list'][0]['topos_original']
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref_mean.shape[1], topos_orig_ref_mean)
    
    # Create figure: 8 rows × 4 columns
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(20, max(16, n_biomarkers * 2)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=30, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Perform Spearman test and create p-value map
                p_values_corrected = perform_spearman_fdr_test(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(256)  # 256 channels
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im, _ = mne.viz.plot_topomap(p_map, info, axes=ax,
                                            vlim=(0, 2), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for p-values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, ticks=[0, 1, 2])
                    cbar.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                    cbar.ax.tick_params(labelsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_selected_fdr.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Selected biomarker group comparison grid (FDR) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_spearman_values_selected(output_dir):
    """Create spearman correlation grid with selected biomarkers for MCS, UWS, Control groups only."""
    
    print("=" * 60)
    print("Creating Selected Biomarker Spearman Values Grid")
    print("=" * 60)
    
    # Define data directories (exclude Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None
    }
    
    # Define selected biomarkers to plot
    biomarker_specs = [
        ('cnv_detailed_cnvslope', 'CNV'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=False
        )
        
        # Other groups: return numpy arrays
        topos_orig, topos_recon, marker_names = result
        
        if topos_orig is not None:
            groups_data[group_name] = {
                'topos_orig': topos_orig,
                'topos_recon': topos_recon,
                'marker_names': marker_names,
                'n_subjects': topos_orig.shape[0],
                'is_homogeneous': True
            }
            print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
        else:
            print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in each group individually
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating selected spearman values grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    topos_orig_ref = first_group['topos_orig']
    topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(14, max(16, n_biomarkers * 1.5)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=30, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute Spearman correlation values
                correlation_values = compute_spearman_correlations(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create correlation map for visualization (continuous values -1 to 1)
                corr_map = correlation_values
                
                # Use RdBu_r colormap for correlation values (-1 to 1)
                cmap = 'RdBu_r'
                
                # Plot correlation map
                im, _ = mne.viz.plot_topomap(corr_map, info, axes=ax,
                                            vlim=(-1, 1), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for correlation values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=15)
                    cbar.set_label('Spearman ρ', fontsize=20)
            else:  # Marker not available in this group
                # Show "N/A" text
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       fontsize=20, ha='center', va='center', color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_spearman_values_selected.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Selected biomarker group comparison grid (Spearman values) saved to: {output_path}")
    return output_path


def create_biomarker_comparison_grid_spearman_values_selected_with_rs(output_dir):
    """Create spearman correlation grid with selected biomarkers for MCS, UWS, Control, Control RS groups."""
    
    print("=" * 60)
    print("Creating Selected Biomarker Spearman Values Grid (with Control RS)")
    print("=" * 60)
    
    # Define data directories (include Control RS)
    data_dirs = {
        'MCS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'UWS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_DoC',
        'Control': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data_control_lg',
        'Control RS': '/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data'
    }
    
    # Load patient labels for diagnostic filtering
    patient_labels_file = "/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv"
    patient_labels = load_patient_labels(patient_labels_file)
    
    # Define diagnostic filters
    diagnostic_filters = {
        'MCS': ['MCS+', 'MCS-'],
        'UWS': ['VS', 'UWS'],
        'Control': None,
        'Control RS': None
    }
    
    # Define selected biomarkers to plot
    biomarker_specs = [
        ('cnv_detailed_cnvslope', 'CNV'),
        ('timelockedcontrast_p3b_timelockedcontrast', 'P3b'),
        ('timelockedcontrast_mmn_timelockedcontrast', 'MMN'),
        ('delta_relative_spectralpower', 'Delta Normalized'),
        ('alpha_relative_spectralpower', 'Alpha Normalized'),
        ('spectral_entropy_spectralpower', 'Spectral Entropy'),
        ('pe_theta_permutationentropy', 'Permutation Entropy'),
        ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov Complexity')
    ]
    
    # Load data for all groups
    groups_data = {}
    for group_name, data_dir in data_dirs.items():
        diagnostic_filter = diagnostic_filters[group_name]
        
        # Control RS needs heterogeneous mode
        allow_heterogeneous = (group_name == 'Control RS')
        
        result = load_group_topo_data(
            data_dir, group_name, patient_labels, diagnostic_filter, allow_heterogeneous=allow_heterogeneous
        )
        
        if allow_heterogeneous:
            # Control RS: return list of dicts
            subjects_list, subject_ids, _ = result
            if subjects_list is not None:
                groups_data[group_name] = {
                    'subjects_list': subjects_list,  # List of dicts with varying shapes
                    'subject_ids': subject_ids,
                    'n_subjects_total': len(subjects_list),
                    'is_homogeneous': False
                }
                print(f"   ✓ {group_name}: {len(subjects_list)} subjects loaded (heterogeneous)")
                
                # Print shape distribution
                shapes = [s['topos_original'].shape[0] for s in subjects_list]
                shape_counts = {}
                for shape in shapes:
                    shape_counts[shape] = shape_counts.get(shape, 0) + 1
                print(f"      Shape distribution: {shape_counts}")
            else:
                print(f"   ❌ {group_name}: No data loaded")
        else:
            # Other groups: return numpy arrays
            topos_orig, topos_recon, marker_names = result
            
            if topos_orig is not None:
                groups_data[group_name] = {
                    'topos_orig': topos_orig,
                    'topos_recon': topos_recon,
                    'marker_names': marker_names,
                    'n_subjects': topos_orig.shape[0],
                    'is_homogeneous': True
                }
                print(f"   ✓ {group_name}: {topos_orig.shape[0]} subjects loaded")
            else:
                print(f"   ❌ {group_name}: No data loaded")
    
    if not groups_data:
        print("❌ No data loaded for any group. Exiting.")
        return
    
    # Find markers available in each group (allowing missing markers per group)
    group_markers = {}
    for group_name, group_data in groups_data.items():
        group_markers[group_name] = get_markers_for_group(group_data)
    
    # Filter biomarkers to those available in at least one group
    available_biomarkers = []
    for marker_name, display_name in biomarker_specs:
        # Check if marker is available in at least one group
        available_in_any_group = any(marker_name in markers for markers in group_markers.values())
        if available_in_any_group:
            available_biomarkers.append((marker_name, display_name))
            print(f"   ✓ {display_name}: available in {[g for g, markers in group_markers.items() if marker_name in markers]}")
        else:
            print(f"   ❌ {display_name}: not available in any group, skipping")
    
    if not available_biomarkers:
        print("❌ No biomarkers available in all groups. Exiting.")
        return
    
    print(f"📊 Creating selected spearman values grid plot with {len(available_biomarkers)} biomarkers × {len(groups_data)} groups")
    
    # Get reference data for montage setup
    first_group = list(groups_data.values())[0]
    if first_group['is_homogeneous']:
        topos_orig_ref = first_group['topos_orig']
        topos_orig_ref_mean = np.mean(first_group['topos_orig'], axis=0)
    else:
        # For heterogeneous data, use the first subject
        topos_orig_ref = first_group['topos_orig'][0]
        topos_orig_ref_mean = topos_orig_ref
    
    # Set up montage and sphere
    info, sphere, outlines = _setup_montage_and_sphere(topos_orig_ref.shape[2], topos_orig_ref_mean)
    
    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_groups = len(groups_data)
    fig, axes = plt.subplots(n_biomarkers, n_groups, figsize=(16, max(16, n_biomarkers * 1.5)))
    
    # Handle single row/column cases
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Add column titles at the top
    group_names = list(groups_data.keys())
    for col, title in enumerate(group_names):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=30)
    
    # Plot each biomarker × group combination
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        # Add row label
        axes[row, 0].text(-0.3, 0.5, display_name, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=25, rotation=0)
        
        for col, group_name in enumerate(group_names):
            ax = axes[row, col]
            group_data = groups_data[group_name]
            
            # Get marker data (handle missing markers and heterogeneous groups)
            marker_orig_data, marker_recon_data, n_subjects_for_marker = get_marker_data_for_group(group_data, marker_name)
            
            if marker_orig_data is not None:  # Marker exists in this group
                # Compute Spearman correlation values
                correlation_values = compute_spearman_correlations(
                    marker_orig_data, marker_recon_data, 0, marker_name, group_name
                )
                
                # Create correlation map for visualization (continuous values -1 to 1)
                corr_map = correlation_values
                
                # Use RdBu_r colormap for correlation values (-1 to 1)
                cmap = 'RdBu_r'
                
                # Plot correlation map
                im, _ = mne.viz.plot_topomap(corr_map, info, axes=ax,
                                            vlim=(-1, 1), cmap=cmap,
                                            show=False, sphere=sphere, outlines=outlines,
                                            extrapolate='local', res=256, sensors=True, contours=6)
                ax.set_title('')
                
                # Add colorbar for correlation values (only in rightmost column)
                if col == n_groups - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=15)
                    cbar.set_label('Spearman ρ', fontsize=15)
            else:  # Marker not available in this group
                # Show "X" text for missing markers
                ax.text(0.5, 0.5, 'X', transform=ax.transAxes,
                       fontsize=30, ha='center', va='center', color='red', weight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid_spearman_values_selected_with_rs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Selected biomarker group comparison grid (Spearman values with Control RS) saved to: {output_path}")
    return output_path


def main():
    """Main function to run all biomarker comparison analyses."""
    
    print("=" * 80)
    print("OHBM Biomarker Group Comparison Analysis")
    print("=" * 80)
    
    # Define output directory
    output_dir = '/data/project/eeg_foundation/src/doc_benchmark/results/figures/ohbm_biomarker_comparison'
    
    # Create all plots
    print("\n📊 Creating all biomarker comparison plots...")
    
    # Original plots
    create_biomarker_comparison_grid(output_dir)
    create_biomarker_comparison_grid_spearman_values(output_dir)
    create_biomarker_comparison_grid_wilcoxon(output_dir)
    create_biomarker_comparison_grid_mi(output_dir)
    
    # New extended plots
    create_biomarker_comparison_grid_spearman_values_extended(output_dir)
    create_biomarker_comparison_grid_spearman_values_extended_with_fdr(output_dir)
    create_biomarker_difference_grid(output_dir)
    
    # New selected biomarker plots
    create_biomarker_difference_grid_selected(output_dir)
    create_biomarker_comparison_grid_selected_fdr(output_dir)
    create_biomarker_comparison_grid_spearman_values_selected(output_dir)
    create_biomarker_comparison_grid_spearman_values_selected_with_rs(output_dir)
    
    print("\n✅ All biomarker comparison analyses completed!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
