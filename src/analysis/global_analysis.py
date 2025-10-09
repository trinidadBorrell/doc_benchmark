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

# Try to import MNE for topographic plotting
try:
    import mne
    HAS_MNE = True
    mne.set_log_level('WARNING')
except ImportError:
    HAS_MNE = False
    print("Warning: MNE-Python not available. Topographic plots will be skipped.")

from scipy.stats import zscore

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


class GlobalAnalyzer:
    """Global analysis across multiple subjects."""
    
    def __init__(self, results_dir, output_dir, patient_labels_file=None, target_state=None):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        self.target_state = target_state
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
        
        # Load patient labels if provided
        self.patient_labels = {}
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
                
                # Group diagnoses as requested:
                # - Merge MCS+ and MCS- into MCS
                # - Merge UWS and VS into VS/UWS (they are the same condition)
                if state in ['MCS+', 'MCS-']:
                    state = 'MCS'
                elif state == 'VS':
                    state = 'UWS'  # VS and UWS are the same, use UWS as standard
                
                # Create key compatible with our subject_session format
                subject_session_key = f"{subject}_{session}"
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
            
            inconsistent_subjects = []
            for subj_id in subjects_included[1:]:
                subj_scalar_shape = self.subjects_data[subj_id]['scalars_original'].shape
                subj_topo_shape = self.subjects_data[subj_id]['topos_original'].shape
                
                if subj_scalar_shape != ref_scalar_shape or subj_topo_shape != ref_topo_shape:
                    inconsistent_subjects.append(subj_id)
                    print(f"   ❌ {subj_id}: scalars={subj_scalar_shape}, topos={subj_topo_shape}")
                else:
                    print(f"   ✅ {subj_id}: consistent")
            
            if len(inconsistent_subjects) > 0:
                print(f"\n⚠️  Warning: {len(inconsistent_subjects)} subjects have inconsistent data shapes!")
                print("   This may cause issues in global analysis. Consider reprocessing these subjects.")
        
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
        
        # Topographic data preparation
        self.global_topo_data = {
            'subjects': subjects,
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
        
        # Collect scalar data
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
        
        # Collect topographic data
        for subject_id in subjects:
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
        if subjects:
            first_topo = self.global_topo_data['topos_orig_all'][0]
            self.global_topo_data['n_markers'] = first_topo.shape[0]
            self.global_topo_data['n_channels'] = first_topo.shape[1]
        
        print(f"  Prepared data for {len(subjects)} subjects")
        print(f"  Scalar markers: {len(self.global_scalar_data['marker_data'])}")
        print(f"  Topo dimensions: {self.global_topo_data['n_markers']} markers × {self.global_topo_data['n_channels']} channels")
    
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
        n_markers = self.global_topo_data['n_markers']
        n_channels = self.global_topo_data['n_channels']
        
        # Convert to numpy arrays
        topos_orig_all = np.array(self.global_topo_data['topos_orig_all'])  # (n_subjects, n_markers, n_channels)
        topos_recon_all = np.array(self.global_topo_data['topos_recon_all'])
        
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
        if not HAS_MNE:
            print("  ⚠️  Skipping MNE topomap plots - MNE-Python not available")
            return
            
        print("Creating MNE topographic plots...")
        
        subjects = self.global_topo_data['subjects']
        n_subjects = len(subjects)
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
        
        # Create montage based on the number of channels
        try:
            print(f"  !!!!!!!!Creating montage for {n_channels} channels")
            if n_channels == 64:
                # Use biosemi64 montage which is specifically designed for 64 electrodes
                montage = mne.channels.make_standard_montage('biosemi64')
                # Create info with the exact channel names from the montage
                info = mne.create_info(montage.ch_names, 100, 'eeg')
                info.set_montage(montage)
                print(f"  ✅ Created biosemi64 montage for {n_channels} channels")
                
            elif n_channels == 32:
                montage = mne.channels.make_standard_montage('biosemi32')
                info = mne.create_info(montage.ch_names, 100, 'eeg')
                info.set_montage(montage)
                print(f"  ✅ Created biosemi32 montage for {n_channels} channels")
                
            elif n_channels == 128:
                montage = mne.channels.make_standard_montage('biosemi128')
                info = mne.create_info(montage.ch_names, 100, 'eeg')
                info.set_montage(montage)
                print(f"  ✅ Created biosemi128 montage for {n_channels} channels")
                
            elif n_channels == 256:
                # Try EGI_256 first (most appropriate for your data), then fallback to biosemi256
                try:
                    print(f"  Creating GSN-HydroCel-256 montage for {n_channels} channels")
                    montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
                    info = mne.create_info(ch_names =montage.ch_names, sfreq =250, ch_types ='eeg')
                    info.set_montage(montage)
                except:
                    montage = mne.channels.make_standard_montage('biosemi256')
                    info = mne.create_info(montage.ch_names, 250, 'eeg')
                    info.set_montage(montage)
                    print(f"  ✅ Created biosemi256 montage for {n_channels} channels")
                
            elif n_channels <= 256:
                # For other channel counts, try to use standard_1020 which has good coverage
                montage = mne.channels.make_standard_montage('standard_1020')
                
                # Check if we have enough channels in the montage
                if len(montage.ch_names) >= n_channels:
                    # Take only the first n_channels
                    available_channels = montage.ch_names[:n_channels]
                    montage.ch_names = available_channels
                    
                    # Filter the dig points to match the selected channels
                    # Keep fiducials (first 3 points) and selected EEG channels
                    fiducials = montage.dig[:3]  # nasion, lpa, rpa
                    eeg_dig = montage.dig[3:3+n_channels]  # EEG channels
                    montage.dig = fiducials + eeg_dig
                    
                    info = mne.create_info(available_channels, 100, 'eeg')
                    info.set_montage(montage)
                    print(f"  ✅ Created standard_1020 montage for {n_channels} channels")
                else:
                    # Not enough channels in montage, fall back to spherical layout
                    print(f"  ⚠️  standard_1020 montage has only {len(montage.ch_names)} channels, need {n_channels}")
                    raise ValueError("Not enough channels in standard montage")
                
            else:
                raise ValueError(f"Unsupported channel count: {n_channels}")
            
        except Exception as e:
            print(f"  ⚠️  Could not create standard montage for {n_channels} channels: {e}")
            print("  Creating spherical layout montage...")
            
            # Create channel names matching typical EEG conventions
            ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
            info = mne.create_info(ch_names, 100, 'eeg')
            
            # Create a spherical layout for the electrodes
            # Use MNE's built-in sphere layout which ensures electrodes stay within head boundary
            from mne.channels.layout import _auto_topomap_coords
            pos = _auto_topomap_coords(info, picks=None, sphere=None, ignore_overlap=True)
            
            # Create montage with proper electrode positions
            montage_dict = dict(zip(ch_names, pos))
            montage = mne.channels.make_dig_montage(montage_dict, coord_frame='head')
            info.set_montage(montage)
            print(f"   Created spherical layout montage for {n_channels} channels")
        
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
                                                 vlim=(data_min, data_max), 
                                                 show=False, cmap='viridis')
                    
                # Set title for original plot
                if len(marker_name) > 15:
                    title_orig = f'{marker_name[:15]}\n{marker_name[15:]} Original'
                else:
                    title_orig = f'{marker_name}\nOriginal'
                axes[row, 0].set_title(title_orig)
                    
                    # Plot reconstructed
                im2, _ = mne.viz.plot_topomap(recon_data, info, axes=axes[row, 1],
                                                 vlim=(data_min, data_max),
                                                 show=False, cmap='viridis')
                # Set title for reconstructed plot
                if len(marker_name) > 15:
                    title_recon = f'{marker_name[:15]}\n{marker_name[15:]} Reconstructed'
                else:
                    title_recon = f'{marker_name}\nReconstructed'
                axes[row, 1].set_title(title_recon)
                    
                    # Plot difference (Original - Reconstructed) with symmetric scale
                im3, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 2],
                                                 vlim=(diff_symmetric_min, diff_symmetric_max),
                                                 show=False, cmap='viridis')
                # Handle long marker names
                if len(marker_name) > 15:
                    axes[row, 2].set_title(f'{marker_name[:15]}\n{marker_name[15:]} Difference')
                else:
                    axes[row, 2].set_title(f'{marker_name}\nDifference')
                    
                    # Plot difference again but with same scale as original/reconstructed
                im4, _ = mne.viz.plot_topomap(diff_data, info, axes=axes[row, 3],
                                                 vlim=(data_min, data_max),
                                                 show=False, cmap='viridis')
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
        self.create_scalar_global_plots()
        # self.create_topographic_global_plots()  # Commented out for now
        
        # Create time series error plots
        self.create_timeseries_error_plots()
        
        # Create MNE topographic plots
        self.create_mne_topomap_plots()
        
        # Create statistical tests plots - DISABLED (moved to statistical_analysis.py)
        # self.create_statistical_tests_plots()
        
        # Statistical tests - DISABLED (moved to statistical_analysis.py)
        # stats_results = self.global_statistical_tests()
        stats_results = None  # Statistical tests moved to statistical_analysis.py
        
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


def run_state_based_analysis(results_dir, base_output_dir, patient_labels_file):
    """Run analysis for all states and the complete dataset."""
    print("=" * 80)
    print("RUNNING STATE-BASED GLOBAL ANALYSIS")
    print("=" * 80)
    
    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    # First, load patient labels to get available states
    try:
        df = pd.read_csv(patient_labels_file)
        available_states = set()
        for _, row in df.iterrows():
            state = row['state']
            if pd.notna(state) and state != 'n/a':
                available_states.add(state)
        available_states = sorted(available_states)
        print(f"📋 Found states: {available_states}")
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
    analyzer_all = GlobalAnalyzer(results_dir, all_output_dir, patient_labels_file, target_state=None)
    results_all = analyzer_all.run_analysis()
    all_results['all_subs'] = results_all
    
    # 2. Run analysis for each state
    for state in available_states:
        print(f"\n{'='*60}")
        print(f"ANALYZING STATE: {state}")
        print(f"{'='*60}")
        
        state_output_dir = op.join(base_output_dir, state)
        analyzer_state = GlobalAnalyzer(results_dir, state_output_dir, patient_labels_file, target_state=state)
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
        
        analyzer = GlobalAnalyzer(args.results_dir, output_dir)
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
        
        analyzer = GlobalAnalyzer(args.results_dir, output_dir, patient_labels_file, args.single_state)
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
        
        run_state_based_analysis(args.results_dir, base_output_dir, patient_labels_file)


if __name__ == '__main__':
    main()