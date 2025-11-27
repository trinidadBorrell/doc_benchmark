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
            
    
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.tight_layout(pad=1.5)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, 'biomarker_group_comparison_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biomarker group comparison grid saved to: {output_path}")
    return output_path


def main():
    """Main function to create biomarker group comparison plot."""
    
    # Configuration
    output_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/OHBM"
    
    # Create the comparison plot
    plot_path = create_biomarker_comparison_grid(output_dir)
    
    if plot_path:
        print("\n" + "=" * 60)
        print("Biomarker group comparison completed successfully!")
        print(f"Plot saved: {plot_path}")
        print("=" * 60)
    else:
        print("\nFailed to create biomarker group comparison plot.")


if __name__ == "__main__":
    main()
