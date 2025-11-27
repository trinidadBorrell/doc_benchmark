"""Global topographic analysis - Minimal version for 10 specific plots.

Creates only these plots:
1. custom_biomarkers_orig_recon_diff_MCSplus_MCSminus.png
2. custom_biomarkers_orig_recon_diff_UWS_VS.png  
3. custom_biomarkers_orig_recon_diff.png
4. custom_full_biomarkers_orig_recon_spearman.png
5. custom_full_biomarkers_orig_recon_wilcoxon_corrected.png
6. custom_full_biomarkers_orig_recon_wilcoxon.png
7. custom_full_biomarkers_orig_recon_wilcoxon_MCSplus_MCSminus.png
8. custom_full_biomarkers_orig_recon_wilcoxon_UWS_VS.png
9. custom_full_biomarkers_orig_recon_spearman_MCSplus_MCSminus.png
10. custom_full_biomarkers_orig_recon_spearman_UWS_VS.png

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
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
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
            'Timelockedcontrast_p3b_timelockedcontrast'
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


class GlobalTopoAnalyzer:
    """Global topographic analysis - Minimal version for 10 specific plots."""
    
    def __init__(self, results_dir, output_dir, patient_labels_file=None):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        self.mapper = MarkerNameMapper()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.plots_dir = op.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Data containers
        self.subjects_data = {}
        self.global_topo_data = {}
        self.patient_labels_original = {}
        
        # Load patient labels if provided
        if self.patient_labels_file:
            self._load_patient_labels()
    
    def _load_patient_labels(self):
        """Load patient labels from CSV file."""
        try:
            print(f"📋 Loading patient labels from: {self.patient_labels_file}")
            df = pd.read_csv(self.patient_labels_file)
            
            for _, row in df.iterrows():
                subject = row['subject']
                session = f"ses-{row['session']:02d}"
                state = row['state']
                
                if pd.isna(state) or state == 'n/a':
                    continue
                
                subject_session_key = f"{subject}_{session}"
                self.patient_labels_original[subject_session_key] = state
            
            print(f"   ✓ Loaded labels for {len(self.patient_labels_original)} subject/sessions")
            
        except Exception as e:
            print(f"   ⚠️  Error loading patient labels: {e}")
    
    def collect_subject_data(self):
        """Collect data from computed_data directory with topos_*.npz files."""
        print("🔍 Scanning for topos_*.npz files in computed_data...")
        
        # Scan computed_data directory for topos files
        computed_data_dir = op.join(self.results_dir, "computed_data")
        if not op.exists(computed_data_dir):
            raise ValueError(f"Computed data directory not found: {computed_data_dir}")
        
        # Find all subjects with orig/recon directories
        subject_dirs = [d for d in os.listdir(computed_data_dir) if d.startswith('sub-')]
        print(f"📁 Found {len(subject_dirs)} subject directories")
        
        for subject_dir in sorted(subject_dirs):
            subject_path = op.join(computed_data_dir, subject_dir)
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
                print(f"  📊 Processing {subject_session_id}")
                self._process_subject_session(subject_id, session_dir, subject_session_id, session_path)
        
        # Return list of collected subjects
        return list(self.subjects_data.keys())
    
    def _process_subject_session(self, subject_id, session_id, subject_session_id, session_path):
        """Process a single subject/session from .npz files."""
        try:
            # Check for orig and recon directories
            orig_dir = op.join(session_path, "orig")
            recon_dir = op.join(session_path, "recon")
            
            topos_orig_file = op.join(orig_dir, f"topos_sub-{subject_id}_{session_id}.npz")
            topos_recon_file = op.join(recon_dir, f"topos_sub-{subject_id}_{session_id}.npz")
            
            # Check files exist
            if not op.exists(topos_orig_file):
                print(f"     ⏭️  Skipping {subject_session_id}: missing orig topos file")
                return
            if not op.exists(topos_recon_file):
                print(f"     ⏭️  Skipping {subject_session_id}: missing recon topos file")
                return
            
            # Load and extract topographic data
            def extract_topo_markers(npz_data):
                marker_names = sorted(npz_data.files)
                if not marker_names:
                    raise ValueError("No markers found in .npz file")
                topo_data = np.array([npz_data[name] for name in marker_names])
                return topo_data, marker_names
            
            topos_orig, marker_names = extract_topo_markers(np.load(topos_orig_file))
            topos_recon, _ = extract_topo_markers(np.load(topos_recon_file))
            
            # Validate data shape (should be n_markers x n_channels)
            if topos_orig.ndim != 2 or topos_recon.ndim != 2:
                raise ValueError(f"Expected 2D topographic data, got orig: {topos_orig.shape}, recon: {topos_recon.shape}")
            
            if topos_orig.shape != topos_recon.shape:
                raise ValueError(f"Shape mismatch between orig and recon: orig {topos_orig.shape} vs recon {topos_recon.shape}")
            
            # Store data
            self.subjects_data[subject_session_id] = {
                'topos_original': topos_orig,
                'topos_reconstructed': topos_recon,
                'marker_names': marker_names,
                'subject_id': subject_id,
                'session_id': session_id,
                'n_markers': topos_orig.shape[0],
                'n_channels': topos_orig.shape[1]
            }
            
            print(f"     ✓ Loaded {subject_session_id}: {topos_orig.shape[0]} markers × {topos_orig.shape[1]} channels")
            
        except Exception as e:
            print(f"     ❌ Error loading {subject_id}/{session_id}: {e}")
    
    def prepare_global_data(self):
        """Prepare global data structures from .npz files."""
        print("Preparing global data structures...")
        
        subjects = list(self.subjects_data.keys())
        
        if not subjects:
            print("  ⚠️  No subjects loaded!")
            return
        
        self.global_topo_data = {
            'subjects': [],
            'topos_orig_all': [],
            'topos_recon_all': [],
            'n_markers': 0,
            'n_channels': 0
        }
        
        # Collect topographic arrays from all subjects with shape validation
        expected_shape = None
        valid_subjects = []
        
        for subject_id in subjects:
            data = self.subjects_data[subject_id]
            
            # Validate shape consistency across subjects
            current_shape = data['topos_original'].shape
            if expected_shape is None:
                expected_shape = current_shape
                print(f"  Expected shape set to: {expected_shape}")
            elif current_shape != expected_shape:
                print(f"  ⚠️  Skipping {subject_id}: shape mismatch {current_shape} vs expected {expected_shape}")
                continue
            
            # Add to valid subjects list
            valid_subjects.append(subject_id)
            
            # Directly use the loaded topographic data
            self.global_topo_data['topos_orig_all'].append(data['topos_original'])
            self.global_topo_data['topos_recon_all'].append(data['topos_reconstructed'])
        
        # Check if we have any valid subjects after shape validation
        if len(self.global_topo_data['topos_orig_all']) == 0:
            print("  ❌ No subjects with consistent topographic data found!")
            return
        
        # Update subjects list to only include valid ones
        self.global_topo_data['subjects'] = valid_subjects
        
        # Convert to numpy arrays (subjects × markers × channels)
        self.global_topo_data['topos_orig_all'] = np.array(self.global_topo_data['topos_orig_all'])
        self.global_topo_data['topos_recon_all'] = np.array(self.global_topo_data['topos_recon_all'])
        
        # Set dimensions from first subject
        self.global_topo_data['n_markers'] = self.global_topo_data['topos_orig_all'].shape[1]
        self.global_topo_data['n_channels'] = self.global_topo_data['topos_orig_all'].shape[2]
        
        # Compute mean across subjects (markers × channels)
        self.global_topo_data['topos_orig_mean'] = np.mean(self.global_topo_data['topos_orig_all'], axis=0)
        self.global_topo_data['topos_recon_mean'] = np.mean(self.global_topo_data['topos_recon_all'], axis=0)
        
        # Get marker names from first valid subject
        first_valid_subject = valid_subjects[0]
        first_subject_data = self.subjects_data[first_valid_subject]
        self.global_topo_data['marker_names'] = first_subject_data['marker_names']
        
        print(f"  Prepared data for {len(valid_subjects)} subjects")
        print(f"  Topo dimensions: {self.global_topo_data['n_markers']} markers × {self.global_topo_data['n_channels']} channels")
        print(f"  Data shape: {self.global_topo_data['topos_orig_all'].shape} (subjects × markers × channels)")
    
    def _create_custom_biomarker_topomap(self, topos_orig_mean, topos_recon_mean, info, marker_names, sphere=None, outlines='head'):
        """Create custom 3-column topographic plot for specific biomarkers."""
        
        # Define the 8 biomarkers to plot (WITHOUT SymbolicMutualInformation)
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
            else:
                print(f"    ⚠️  {display_name} ({marker_name}) not found in data")
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        print(f"  📊 Creating plot for {n_biomarkers} biomarkers")
        
        # Create figure: 3 columns x n_biomarkers rows
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
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.7, aspect=20)
            cbar2.ax.tick_params(labelsize=16)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.7, aspect=20)
            cbar3.ax.tick_params(labelsize=16)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot
        plt.savefig(op.join(self.plots_dir, 'custom_biomarkers_orig_recon_diff.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom biomarker plot saved with {n_biomarkers} markers")
    
    def _create_diagnosis_filtered_biomarker_topomap(self, topos_orig_all, topos_recon_all, info, marker_names, 
                                                      diagnosis_group=None, sphere=None, outlines='head'):
        """Create 3-column topographic plot filtered by diagnosis group."""
        
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
        
        # Filter the topographic data
        topos_orig_filtered = topos_orig_all[filtered_indices]
        topos_recon_filtered = topos_recon_all[filtered_indices]
        
        # Compute mean across filtered subjects
        topos_orig_mean = np.mean(topos_orig_filtered, axis=0)
        topos_recon_mean = np.mean(topos_recon_filtered, axis=0)
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print(f"     ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Create figure: 3 columns x n_biomarkers rows
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
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=32,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.7, aspect=20)
            cbar2.ax.tick_params(labelsize=16)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.7, aspect=20)
            cbar3.ax.tick_params(labelsize=16)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot with diagnosis group in filename
        filename = f'custom_biomarkers_orig_recon_diff_{group_name}.png'
        plt.savefig(op.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Diagnosis-filtered plot saved: {filename}")
    
    def _create_custom_full_biomarker_topomap_wilcoxon(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """Create custom 4-column topographic plot with Wilcoxon test."""
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Collect all p-values for FDR correction
        p_values = []
        
        if topos_orig_all is not None and topos_recon_all is not None:
            for marker_idx in biomarker_indices:
                orig_vals = topos_orig_all[:, marker_idx, :].flatten()
                recon_vals = topos_recon_all[:, marker_idx, :].flatten()
                
                try:
                    _, p_val = stats.wilcoxon(orig_vals, recon_vals)
                    p_values.append(p_val)
                except:
                    p_values.append(1.0)
            
            # Apply FDR correction (Benjamini-Hochberg)
            if HAS_STATSMODELS:
                _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            else:
                print("  ⚠️  statsmodels not available, skipping FDR correction")
                p_values_corrected = p_values
        else:
            p_values_corrected = [1.0] * len(biomarker_indices)
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(25, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon Test (FDR)']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label, p_val_corr) in enumerate(zip(biomarker_indices, biomarker_labels, p_values_corrected)):
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Column 4: Electrode-wise Wilcoxon test
            if topos_orig_all is not None and topos_recon_all is not None:
                # Perform electrode-wise Wilcoxon signed-rank test
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
                else:
                    p_values_corrected = p_values
                    print("  ⚠️  statsmodels not available, using uncorrected p-values")
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(n_channels)
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im4, _ = mne.viz.plot_topomap(p_map, info, axes=axes[row, 3],
                                             vlim=(0, 2), cmap=cmap,
                                             show=False, sphere=sphere, outlines=outlines,
                                             extrapolate='local', res=256, sensors=True)
                axes[row, 3].set_title('')
                
                # Add colorbar for p-values
                cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.6, aspect=20, ticks=[0, 1, 2])
                cbar4.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                cbar4.ax.tick_params(labelsize=12)
            else:
                axes[row, 3].text(0.5, 0.5, 'Wilcoxon\nTest\nNo Data', 
                                  transform=axes[row, 3].transAxes,
                                  fontsize=16, ha='center', va='center')
                axes[row, 3].set_xlim(0, 1)
                axes[row, 3].set_ylim(0, 1)
                axes[row, 3].axis('off')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.6, aspect=20)
            cbar2.ax.tick_params(labelsize=14)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.6, aspect=20)
            cbar3.ax.tick_params(labelsize=14)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_wilcoxon.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot saved with {n_biomarkers} markers")
    
    def _create_custom_full_biomarker_topomap_spearman(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """Create custom 4-column topographic plot with Spearman test."""
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(25, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Spearman Test']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Column 4: Electrode-wise Spearman test
            if topos_orig_all is not None and topos_recon_all is not None:
                # Perform electrode-wise Spearman correlation test
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
                else:
                    p_values_corrected = p_values
                    print("  ⚠️  statsmodels not available, using uncorrected p-values")
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(n_channels)
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im4, _ = mne.viz.plot_topomap(p_map, info, axes=axes[row, 3],
                                             vlim=(0, 2), cmap=cmap,
                                             show=False, sphere=sphere, outlines=outlines,
                                             extrapolate='local', res=256, sensors=True)
                axes[row, 3].set_title('')
                
                # Add colorbar for p-values
                cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.6, aspect=20, ticks=[0, 1, 2])
                cbar4.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                cbar4.ax.tick_params(labelsize=12)
            else:
                axes[row, 3].text(0.5, 0.5, 'Spearman\nTest\nNo Data', 
                                  transform=axes[row, 3].transAxes,
                                  fontsize=16, ha='center', va='center')
                axes[row, 3].set_xlim(0, 1)
                axes[row, 3].set_ylim(0, 1)
                axes[row, 3].axis('off')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.6, aspect=20)
            cbar2.ax.tick_params(labelsize=14)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.6, aspect=20)
            cbar3.ax.tick_params(labelsize=14)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_spearman.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot (Spearman) saved with {n_biomarkers} markers")
    
    def _create_custom_full_biomarker_topomap_wilcoxon_corrected(self, topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all=None, topos_recon_all=None, sphere=None, outlines='head'):
        """Create custom 4-column topographic plot with FDR-corrected Wilcoxon test."""
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print("  ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Collect all p-values for FDR correction
        p_values = []
        
        if topos_orig_all is not None and topos_recon_all is not None:
            for marker_idx in biomarker_indices:
                orig_vals = topos_orig_all[:, marker_idx, :].flatten()
                recon_vals = topos_recon_all[:, marker_idx, :].flatten()
                
                try:
                    _, p_val = stats.wilcoxon(orig_vals, recon_vals)
                    p_values.append(p_val)
                except:
                    p_values.append(1.0)
            
            # Apply FDR correction (Benjamini-Hochberg)
            if HAS_STATSMODELS:
                _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            else:
                print("  ⚠️  statsmodels not available, skipping FDR correction")
                p_values_corrected = p_values
        else:
            p_values_corrected = [1.0] * len(biomarker_indices)
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(25, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon Test (FDR)']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label, p_val_corr) in enumerate(zip(biomarker_indices, biomarker_labels, p_values_corrected)):
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Column 4: Electrode-wise FDR-corrected Wilcoxon test
            if topos_orig_all is not None and topos_recon_all is not None:
                # Perform electrode-wise Wilcoxon signed-rank test
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
                else:
                    p_values_corrected = p_values
                    print("  ⚠️  statsmodels not available, using uncorrected p-values")
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(n_channels)
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im4, _ = mne.viz.plot_topomap(p_map, info, axes=axes[row, 3],
                                             vlim=(0, 2), cmap=cmap,
                                             show=False, sphere=sphere, outlines=outlines,
                                             extrapolate='local', res=256, sensors=True)
                axes[row, 3].set_title('')
                
                # Add colorbar for p-values
                cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.6, aspect=20, ticks=[0, 1, 2])
                cbar4.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                cbar4.ax.tick_params(labelsize=12)
            else:
                axes[row, 3].text(0.5, 0.5, 'Wilcoxon\nTest\nNo Data', 
                                  transform=axes[row, 3].transAxes,
                                  fontsize=16, ha='center', va='center')
                axes[row, 3].set_xlim(0, 1)
                axes[row, 3].set_ylim(0, 1)
                axes[row, 3].axis('off')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.6, aspect=20)
            cbar2.ax.tick_params(labelsize=14)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.6, aspect=20)
            cbar3.ax.tick_params(labelsize=14)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot
        plt.savefig(op.join(self.plots_dir, 'custom_full_biomarkers_orig_recon_wilcoxon_corrected.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Custom FULL biomarker plot (FDR corrected) saved with {n_biomarkers} markers")
    
    def _create_diagnosis_specific_wilcoxon_topomap(self, topos_orig_all, topos_recon_all, info, marker_names, 
                                                    diagnosis_group=None, sphere=None, outlines='head'):
        """Create diagnosis-specific 4-column topographic plot with Wilcoxon test."""
        
        if diagnosis_group is None or not self.patient_labels_original:
            print("  ⚠️  No diagnosis filtering available, skipping diagnosis-specific wilcoxon plots")
            return
        
        group_name = '_'.join(diagnosis_group).replace('+', 'plus').replace('-', 'minus')
        print(f"  🎯 Creating diagnosis-specific wilcoxon plots for {diagnosis_group}...")
        
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
        
        if len(filtered_indices) < 6:
            print(f"     ⚠️  Warning: Only {len(filtered_indices)} subjects found with diagnoses {diagnosis_group}. Wilcoxon test requires N≥6 for reliable results.")
        
        print(f"     ✓ Found {len(filtered_indices)} subjects with diagnoses {diagnosis_group}")
        
        # Filter the topographic data
        topos_orig_filtered = topos_orig_all[filtered_indices]
        topos_recon_filtered = topos_recon_all[filtered_indices]
        
        # Compute mean across filtered subjects
        topos_orig_mean = np.mean(topos_orig_filtered, axis=0)
        topos_recon_mean = np.mean(topos_recon_filtered, axis=0)
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print(f"     ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Collect all p-values for FDR correction
        p_values = []
        
        for marker_idx in biomarker_indices:
            orig_vals = topos_orig_filtered[:, marker_idx, :].flatten()
            recon_vals = topos_recon_filtered[:, marker_idx, :].flatten()
            
            try:
                _, p_val = stats.wilcoxon(orig_vals, recon_vals)
                p_values.append(p_val)
            except:
                p_values.append(1.0)
        
        # Apply FDR correction (Benjamini-Hochberg)
        if HAS_STATSMODELS:
            _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        else:
            print("     ⚠️  statsmodels not available, skipping FDR correction")
            p_values_corrected = p_values
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(25, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Wilcoxon Test (FDR)']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label, p_val_corr) in enumerate(zip(biomarker_indices, biomarker_labels, p_values_corrected)):
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Column 4: Electrode-wise Wilcoxon test
            if topos_orig_filtered is not None and topos_recon_filtered is not None:
                # Perform electrode-wise Wilcoxon signed-rank test
                n_channels = topos_orig_filtered.shape[2]
                p_values = np.zeros(n_channels)
                
                for ch in range(n_channels):
                    try:
                        orig_ch = topos_orig_filtered[:, marker_idx, ch]  # shape: (n_subjects,)
                        recon_ch = topos_recon_filtered[:, marker_idx, ch]
                        stat, p = stats.wilcoxon(orig_ch, recon_ch)
                        p_values[ch] = p
                    except:
                        p_values[ch] = 1.0  # Conservative: assume no significance
                
                # Apply FDR correction across electrodes
                if HAS_STATSMODELS:
                    _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
                else:
                    p_values_corrected = p_values
                    print("  ⚠️  statsmodels not available, using uncorrected p-values")
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(n_channels)
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im4, _ = mne.viz.plot_topomap(p_map, info, axes=axes[row, 3],
                                             vlim=(0, 2), cmap=cmap,
                                             show=False, sphere=sphere, outlines=outlines,
                                             extrapolate='local', res=256, sensors=True)
                axes[row, 3].set_title('')
                
                # Add colorbar for p-values
                cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.6, aspect=20, ticks=[0, 1, 2])
                cbar4.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                cbar4.ax.tick_params(labelsize=12)
            else:
                axes[row, 3].text(0.5, 0.5, 'Wilcoxon\nTest\nNo Data', 
                                  transform=axes[row, 3].transAxes,
                                  fontsize=16, ha='center', va='center')
                axes[row, 3].set_xlim(0, 1)
                axes[row, 3].set_ylim(0, 1)
                axes[row, 3].axis('off')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.6, aspect=20)
            cbar2.ax.tick_params(labelsize=14)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.6, aspect=20)
            cbar3.ax.tick_params(labelsize=14)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot with diagnosis group in filename
        filename = f'custom_full_biomarkers_orig_recon_wilcoxon_{group_name}.png'
        plt.savefig(op.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Diagnosis-specific wilcoxon plot saved: {filename}")
    
    def _create_diagnosis_specific_spearman_topomap(self, topos_orig_all, topos_recon_all, info, marker_names, 
                                                    diagnosis_group=None, sphere=None, outlines='head'):
        """Create diagnosis-specific 4-column topographic plot with Spearman test."""
        
        if diagnosis_group is None or not self.patient_labels_original:
            print("  ⚠️  No diagnosis filtering available, skipping diagnosis-specific spearman plots")
            return
        
        group_name = '_'.join(diagnosis_group).replace('+', 'plus').replace('-', 'minus')
        print(f"  🎯 Creating diagnosis-specific spearman plots for {diagnosis_group}...")
        
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
        
        if len(filtered_indices) < 6:
            print(f"     ⚠️  Warning: Only {len(filtered_indices)} subjects found with diagnoses {diagnosis_group}. Spearman test requires N≥6 for reliable results.")
        
        print(f"     ✓ Found {len(filtered_indices)} subjects with diagnoses {diagnosis_group}")
        
        # Filter the topographic data
        topos_orig_filtered = topos_orig_all[filtered_indices]
        topos_recon_filtered = topos_recon_all[filtered_indices]
        
        # Compute mean across filtered subjects
        topos_orig_mean = np.mean(topos_orig_filtered, axis=0)
        topos_recon_mean = np.mean(topos_recon_filtered, axis=0)
        
        # Define the 8 biomarkers to plot
        biomarker_specs = [
            ('alpha_relative_spectralpower', 'Alpha Normalized'),
            ('beta_relative_spectralpower', 'Beta Normalized'),
            ('delta_relative_spectralpower', 'Delta Normalized'),
            ('gamma_relative_spectralpower', 'Gamma Normalized'),
            ('theta_relative_spectralpower', 'Theta Normalized'),
            ('pe_theta_permutationentropy', 'Permutation\nEntropy'),
            ('kolmogorov_complexity_kolmogorovcomplexity', 'Kolmogorov\nComplexity'),
            ('spectral_entropy_spectralpower', 'Spectral\nEntropy')
        ]
        
        # Find indices for these markers
        biomarker_indices = []
        biomarker_labels = []
        
        for marker_name, display_name in biomarker_specs:
            if marker_name in marker_names:
                idx = marker_names.index(marker_name)
                biomarker_indices.append(idx)
                biomarker_labels.append(display_name)
        
        if not biomarker_indices:
            print(f"     ❌ No requested biomarkers found in data")
            return
        
        n_biomarkers = len(biomarker_indices)
        
        # Create figure: 4 columns x n_biomarkers rows
        fig, axes = plt.subplots(n_biomarkers, 4, figsize=(25, max(12, n_biomarkers * 2.5)))
        
        # Handle single row case
        if n_biomarkers == 1:
            axes = axes.reshape(1, -1)
        
        # Add column titles at the top
        column_titles = ['Original', 'Reconstructed', 'Difference', 'Spearman Test']
        for col, title in enumerate(column_titles):
            axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                             ha='center', va='bottom', fontsize=25)
        
        # Plot each biomarker
        for row, (marker_idx, label) in enumerate(zip(biomarker_indices, biomarker_labels)):
            orig_data = topos_orig_mean[marker_idx, :]
            recon_data = topos_recon_mean[marker_idx, :]
            diff_data = orig_data - recon_data
            
            # Find common scale for original and reconstructed
            orig_min, orig_max = np.min(orig_data), np.max(orig_data)
            recon_min, recon_max = np.min(recon_data), np.max(recon_data)
            
            common_min = min(orig_min, recon_min)
            common_max = max(orig_max, recon_max)
            
            # Symmetric scale for difference
            diff_max_abs = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
            diff_vmin, diff_vmax = -diff_max_abs, diff_max_abs
            
            # Column 1: Original
            im1, _ = mne.viz.plot_topomap(orig_data, info, axes=axes[row, 0],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 0].set_title('')
            
            # Column 2: Reconstructed
            im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                         vlim=(common_min, common_max),
                                         show=False, cmap='viridis',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 1].set_title('')
            
            # Column 3: Difference
            im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                         vlim=(diff_vmin, diff_vmax),
                                         show=False, cmap='RdBu_r',
                                         sphere=sphere, outlines=outlines,
                                         extrapolate='local',
                                         res=256, sensors=True, contours=6)
            axes[row, 2].set_title('')
            
            # Column 4: Electrode-wise Spearman test
            if topos_orig_filtered is not None and topos_recon_filtered is not None:
                # Perform electrode-wise Spearman correlation test
                n_channels = topos_orig_filtered.shape[2]
                p_values = np.zeros(n_channels)
                
                for ch in range(n_channels):
                    try:
                        orig_ch = topos_orig_filtered[:, marker_idx, ch]  # shape: (n_subjects,)
                        recon_ch = topos_recon_filtered[:, marker_idx, ch]
                        corr, p = stats.spearmanr(orig_ch, recon_ch)
                        p_values[ch] = p
                    except:
                        p_values[ch] = 1.0  # Conservative: assume no significance
                
                # Apply FDR correction across electrodes
                if HAS_STATSMODELS:
                    _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
                else:
                    p_values_corrected = p_values
                    print("  ⚠️  statsmodels not available, using uncorrected p-values")
                
                # Create discrete p-value map for visualization
                p_map = np.zeros(n_channels)
                p_map[p_values_corrected < 0.01] = 0      # black: p < 0.01
                p_map[(p_values_corrected >= 0.01) & (p_values_corrected < 0.05)] = 1  # gray: 0.01 ≤ p < 0.05
                p_map[p_values_corrected >= 0.05] = 2    # white: p ≥ 0.05
                
                # Create custom colormap for discrete colors
                cmap = ListedColormap(['black', 'gray', 'white'])
                
                # Plot p-value map
                im4, _ = mne.viz.plot_topomap(p_map, info, axes=axes[row, 3],
                                             vlim=(0, 2), cmap=cmap,
                                             show=False, sphere=sphere, outlines=outlines,
                                             extrapolate='local', res=256, sensors=True)
                axes[row, 3].set_title('')
                
                # Add colorbar for p-values
                cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.6, aspect=20, ticks=[0, 1, 2])
                cbar4.ax.set_yticklabels(['p<0.01', '0.01≤p<0.05', 'p≥0.05'])
                cbar4.ax.tick_params(labelsize=12)
            else:
                axes[row, 3].text(0.5, 0.5, 'Spearman\nTest\nNo Data', 
                                  transform=axes[row, 3].transAxes,
                                  fontsize=16, ha='center', va='center')
                axes[row, 3].set_xlim(0, 1)
                axes[row, 3].set_ylim(0, 1)
                axes[row, 3].axis('off')
            
            # Add row label
            axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                             ha='right', va='center', fontsize=25,
                             rotation=0)
            
            # Add colorbars for columns 2 and 3
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.6, aspect=20)
            cbar2.ax.tick_params(labelsize=14)
            
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.6, aspect=20)
            cbar3.ax.tick_params(labelsize=14)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout(pad=1.5)
        
        # Save plot with diagnosis group in filename
        filename = f'custom_full_biomarkers_orig_recon_spearman_{group_name}.png'
        plt.savefig(op.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Diagnosis-specific spearman plot saved: {filename}")
    
    def create_mne_topomap_plots(self):
        """Create the 10 required MNE topographic plots."""
        if not HAS_MNE:
            print("  ⚠️  Skipping MNE topomap plots - MNE-Python not available")
            return
            
        print("Creating MNE topographic plots...")
        
        subjects = self.global_topo_data['subjects']
        n_subjects = len(subjects)
        
        if n_subjects == 0:
            print("  ⚠️  No subjects available for MNE topomap plots")
            return
        
        n_markers = self.global_topo_data['n_markers']
        n_channels = self.global_topo_data['n_channels']
        
        print(f"  📊 Data summary: {n_subjects} subjects, {n_markers} markers, {n_channels} channels")
        
        # Convert to numpy arrays
        topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])
        topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
        
        # Calculate mean across subjects for each marker
        topos_orig_mean = np.mean(topos_orig_all, axis=0)
        topos_recon_mean = np.mean(topos_recon_all, axis=0)
        
        # Set up montage with proper sphere and outlines
        print(f"  📡 Setting up EGI montage for {n_channels} channels")
        info, sphere, outlines = _setup_montage_and_sphere(n_channels, topos_orig_mean)
        
        marker_names = [self.mapper.get_topo_name(i) for i in range(n_markers)]
        
        # Create the 6 required plots:
        
        # 1. All subjects plot
        print(f"  🎯 Creating plot for all subjects...")
        self._create_custom_biomarker_topomap(topos_orig_mean, topos_recon_mean, info, marker_names, sphere, outlines)
        
        # 2. Diagnosis-specific plots
        print(f"  🎯 Creating diagnosis-filtered plots...")
        self._create_diagnosis_filtered_biomarker_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                           diagnosis_group=['UWS', 'VS'], sphere=sphere, outlines=outlines)
        self._create_diagnosis_filtered_biomarker_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                           diagnosis_group=['MCS+', 'MCS-'], sphere=sphere, outlines=outlines)
        
        # 3. Statistical test plots
        print(f"  🎯 Creating statistical test plots...")
        self._create_custom_full_biomarker_topomap_wilcoxon(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        self._create_custom_full_biomarker_topomap_spearman(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        self._create_custom_full_biomarker_topomap_wilcoxon_corrected(topos_orig_mean, topos_recon_mean, info, marker_names, topos_orig_all, topos_recon_all, sphere, outlines)
        
        # 4. Diagnosis-specific statistical test plots
        print(f"  🎯 Creating diagnosis-specific statistical test plots...")
        self._create_diagnosis_specific_wilcoxon_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                         diagnosis_group=['MCS+', 'MCS-'], sphere=sphere, outlines=outlines)
        self._create_diagnosis_specific_wilcoxon_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                         diagnosis_group=['UWS', 'VS'], sphere=sphere, outlines=outlines)
        self._create_diagnosis_specific_spearman_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                        diagnosis_group=['MCS+', 'MCS-'], sphere=sphere, outlines=outlines)
        self._create_diagnosis_specific_spearman_topomap(topos_orig_all, topos_recon_all, info, marker_names, 
                                                        diagnosis_group=['UWS', 'VS'], sphere=sphere, outlines=outlines)
        
        print(f"  ✅ Created 10 custom biomarker topographic plots")
    
    def run_analysis(self):
        """Run the minimal topographic analysis for the 10 required plots."""
        print("=" * 60)
        print("GLOBAL TOPOGRAPHIC ANALYSIS - MINIMAL VERSION")
        print("=" * 60)
        
        # Collect data
        subjects = self.collect_subject_data()
        if len(subjects) == 0:
            print("⚠️  No subjects found for analysis")
            return
        
        # Prepare data structures
        self.prepare_global_data()
        
        # Create the 10 required plots
        print("\n--- Creating Required Plots ---")
        self.create_mne_topomap_plots()
        
        print("=" * 60)
        print("TOPOGRAPHIC ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Subjects analyzed: {len(subjects)}")
        print(f"Plots saved to: {self.plots_dir}")
        print(f"Created 10 plots:")
        print(f"  1. custom_biomarkers_orig_recon_diff.png")
        print(f"  2. custom_biomarkers_orig_recon_diff_MCSplus_MCSminus.png")
        print(f"  3. custom_biomarkers_orig_recon_diff_UWS_VS.png")
        print(f"  4. custom_full_biomarkers_orig_recon_wilcoxon.png")
        print(f"  5. custom_full_biomarkers_orig_recon_spearman.png")
        print(f"  6. custom_full_biomarkers_orig_recon_wilcoxon_corrected.png")
        print(f"  7. custom_full_biomarkers_orig_recon_wilcoxon_MCSplus_MCSminus.png")
        print(f"  8. custom_full_biomarkers_orig_recon_wilcoxon_UWS_VS.png")
        print(f"  9. custom_full_biomarkers_orig_recon_spearman_MCSplus_MCSminus.png")
        print(f"  10. custom_full_biomarkers_orig_recon_spearman_UWS_VS.png")
        print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Global topographic analysis - Minimal version for 10 plots')
    parser.add_argument('--results-dir', 
                       default='/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS',
                       help='Results directory containing subject folders')
    parser.add_argument('--output-dir', 
                       default='/data/project/eeg_foundation/src/doc_benchmark/results/new_results/GLOBAL/control_rs',
                       help='Output directory for topographic analysis')
    parser.add_argument('--patient-labels', 
                       default='/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv',
                       help='CSV file with patient labels and states')
    
    args = parser.parse_args()
    
    # Check if patient labels file exists
    patient_labels_file = args.patient_labels if op.exists(args.patient_labels) else None
    
    print(f"Starting minimal global topographic analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyzer = GlobalTopoAnalyzer(args.results_dir, args.output_dir, patient_labels_file)
    analyzer.run_analysis()
    
    print("\n✓ Minimal global topographic analysis complete!")


if __name__ == '__main__':
    main()
