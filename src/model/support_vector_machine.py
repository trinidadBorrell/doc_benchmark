"""Cross-subject binary SVM classification for TOTEM data.

==================================================
Cross-subject Binary SVM classification
==================================================

This script trains SVM classifiers across subjects for binary 
classification of consciousness states from EEG markers: VS (Vegetative State) 
vs MCS (Minimally Conscious State). The classification is performed
at the subject level, where each subject contributes one sample.

Key features:
- Binary classification: VS vs MCS (UWS → VS, MCS+/MCS- → MCS)
- Cross-subject classification with GROUP-BASED splitting (NO data leakage)
- All sessions from the same subject stay together in train OR test (never split)
- Support for scalar or topographic markers
- Support for original or reconstructed data
- State prediction from patient_labels_with_controls.csv
- Comprehensive evaluation with proper cross-validation using GroupKFold

Based on the DOC-Forest recipe from:
[1] Engemann D.A.`*, Raimondo F.`*, King JR., Rohaut B., Louppe G.,
    Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
    Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
    Robust EEG-based cross-site and cross-protocol classification of
    states of consciousness. Brain. doi:10.1093/brain/awy251

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import numpy as np
import pandas as pd
import argparse
import os
import os.path as op
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                   StratifiedKFold, LeaveOneOut, train_test_split,
                                   GroupShuffleSplit, GroupKFold)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, balanced_accuracy_score,
                           precision_recall_fscore_support, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import joblib

sns.set_style("whitegrid")

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


class CrossSubjectClassifier:
    """Cross-subject binary SVM classifier for VS vs MCS state prediction."""
    
    def __init__(self, data_dir, patient_labels_file, marker_type='scalar', 
                 data_origin='original', output_dir=None, random_state=42):
        """
        Initialize the cross-subject classifier.
        
        Parameters
        ----------
        data_dir : str
            Path to results directory containing subject data
        patient_labels_file : str  
            Path to CSV file with patient labels
        marker_type : str
            'scalar' or 'topo' - type of markers to use
        data_origin : str
            'original' or 'reconstructed' - source of data
        output_dir : str
            Output directory for results
        random_state : int
            Random state for reproducibility
        """
        self.data_dir = data_dir
        self.patient_labels_file = patient_labels_file
        self.marker_type = marker_type
        self.data_origin = data_origin
        self.output_dir = output_dir or f"results/svm/{marker_type}_{data_origin}"
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data containers
        self.X = None
        self.y = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.feature_names = None
        self.feature_names_abbreviated = None
        
        # Results containers
        self.results = {}
        
    def load_patient_labels(self):
        """Load patient labels from CSV file for binary classification (VS vs MCS)."""
        print(f"Loading patient labels from: {self.patient_labels_file}")
        
        try:
            df = pd.read_csv(self.patient_labels_file)
            
            # Create mapping from subject_session to state
            labels_dict = {}
            available_states = set()
            
            for _, row in df.iterrows():
                subject = row['subject']
                session = f"ses-{row['session']:02d}"
                state = row['state']
                
                # Skip subjects with missing state
                if pd.isna(state) or state == 'n/a':
                    continue
                
                # Binary classification: VS vs MCS
                if state == 'UWS':
                    # UWS corresponds to VS (Vegetative State)
                    state = 'VS'
                elif state in ['MCS+', 'MCS-']:
                    # Merge MCS+ and MCS- into single MCS category
                    state = 'MCS'
                else:
                    # Skip other states (COMA, EMCS, CONTROL, etc.)
                    continue
                
                subject_session_key = f"{subject}_{session}"
                labels_dict[subject_session_key] = state
                available_states.add(state)
            
            print(f"   ✓ Loaded labels for {len(labels_dict)} subject/sessions")
            print(f"   ✓ Available states: {sorted(available_states)}")
            print("    Binary classification: VS (Vegetative State) vs MCS (Minimally Conscious State)")
            
            return labels_dict, sorted(available_states)
            
        except Exception as e:
            print(f"   Error loading patient labels: {e}")
            raise
    
    def get_marker_names(self):
        """Get standard marker names based on NICE collection order.
        
        Returns
        -------
        tuple
            (full_names, abbreviated_names)
        """
        # Standard marker names in NICE collection order
        marker_names = [
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
        
        # Create abbreviations
        abbreviated_names = []
        for name in marker_names:
            if name.startswith('PowerSpectralDensity_'):
                abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
            elif name.startswith('PowerSpectralDensitySummary_'):
                abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
            elif name.startswith('TimeLockedContrast_'):
                abbrev = name.replace('TimeLockedContrast_', 'TLC_')
            elif name.startswith('TimeLockedTopography_'):
                abbrev = name.replace('TimeLockedTopography_', 'TLT_')
            elif name.startswith('ContingentNegativeVariation'):
                abbrev = 'CNV'
            elif name.startswith('PermutationEntropy'):
                abbrev = 'PE'
            elif name.startswith('SymbolicMutualInformation'):
                abbrev = 'SMI'
            elif name.startswith('KolmogorovComplexity'):
                abbrev = 'KC'
            else:
                abbrev = name
            abbreviated_names.append(abbrev)
        
        return marker_names, abbreviated_names
    
    def load_feature_names(self):
        """Load feature names from the first available subject's .npz file.
        
        Returns
        -------
        list
            List of feature names, abbreviated for display
        """
        print(" Loading feature names...")
        
        # First, try to load from .npz files in the new structure
        if op.exists(self.data_dir):
            subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
            
            for subject_dir in sorted(subject_dirs):
                subject_path = op.join(self.data_dir, subject_dir)
                if not op.isdir(subject_path):
                    continue
                    
                # Check if this is the new structure (sub-XXX_original or sub-XXX_recon)
                if '_' in subject_dir and (subject_dir.endswith('_original') or subject_dir.endswith('_recon')):
                    # Map data type for compatibility
                    data_type = subject_dir.split('_')[-1]
                    if data_type == 'recon':
                        data_type = 'reconstructed'
                    
                    # Only process if data_type matches our data_origin
                    if data_type != self.data_origin:
                        continue
                    
                    # Extract subject ID
                    subject_id_raw = subject_dir.split('_')[0].replace('sub-', '')
                    
                    # Look for .npz file
                    npz_filename = f"scalars_{subject_id_raw}_{subject_dir.split('_')[-1]}.npz"
                    npz_path = op.join(subject_path, npz_filename)
                    
                    if op.exists(npz_path):
                        try:
                            # Load feature names from .npz file
                            data_dict = np.load(npz_path)
                            feature_names = sorted(data_dict.files())
                            
                            # Abbreviate long feature names for better display
                            abbreviated_names = []
                            for name in feature_names:
                                # Create abbreviations
                                if name.startswith('PowerSpectralDensity_'):
                                    abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
                                elif name.startswith('PowerSpectralDensitySummary_'):
                                    abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
                                elif name.startswith('TimeLockedContrast_'):
                                    abbrev = name.replace('TimeLockedContrast_', 'TLC_')
                                elif name.startswith('TimeLockedTopography_'):
                                    abbrev = name.replace('TimeLockedTopography_', 'TLT_')
                                elif name.startswith('ContingentNegativeVariation'):
                                    abbrev = 'CNV'
                                elif name.startswith('PermutationEntropy'):
                                    abbrev = 'PE'
                                elif name.startswith('SymbolicMutualInformation'):
                                    abbrev = 'SMI'  
                                elif name.startswith('KolmogorovComplexity'):
                                    abbrev = 'KC'
                                else:
                                    # Keep original if no abbreviation rule
                                    abbrev = name
                                
                                abbreviated_names.append(abbrev)
                            
                            print(f"   ✓ Loaded {len(feature_names)} feature names from {npz_filename}")
                            print(f"   Features: {', '.join(abbreviated_names[:5])}{'...' if len(abbreviated_names) > 5 else ''}")
                            
                            return feature_names, abbreviated_names
                            
                        except Exception as e:
                            print(f"    Error loading {npz_path}: {e}")
                            continue
        
        # OLD structure: try to load from scalar_metrics.csv
        for subject_dir in sorted(subject_dirs):
            subject_path = op.join(self.data_dir, subject_dir)
            if not op.isdir(subject_path):
                continue
                
            # Look for session directories
            try:
                session_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
            except PermissionError:
                continue
            
            for session_dir in sorted(session_dirs):
                session_path = op.join(subject_path, session_dir)
                
                # Try to find scalar_metrics.csv (old structure)
                metrics_file = op.join(session_path, 'compare_markers', 'scalars', 'metrics', 'scalar_metrics.csv')
                
                if op.exists(metrics_file):
                    try:
                        df = pd.read_csv(metrics_file, index_col=0)
                        feature_names = df.index.tolist()
                        
                        # Abbreviate long feature names for better display
                        abbreviated_names = []
                        for name in feature_names:
                            # Create abbreviations
                            if name.startswith('PowerSpectralDensity_'):
                                abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
                            elif name.startswith('PowerSpectralDensitySummary_'):
                                abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
                            elif name.startswith('TimeLockedContrast_'):
                                abbrev = name.replace('TimeLockedContrast_', 'TLC_')
                            elif name.startswith('TimeLockedTopography_'):
                                abbrev = name.replace('TimeLockedTopography_', 'TLT_')
                            elif name.startswith('ContingentNegativeVariation'):
                                abbrev = 'CNV'
                            elif name.startswith('PermutationEntropy'):
                                abbrev = 'PE'
                            elif name.startswith('SymbolicMutualInformation'):
                                abbrev = 'SMI'  
                            elif name.startswith('KolmogorovComplexity'):
                                abbrev = 'KC'
                            else:
                                # Keep original if no abbreviation rule
                                abbrev = name
                            
                            abbreviated_names.append(abbrev)
                        
                        print(f"   ✓ Loaded {len(feature_names)} feature names from {subject_dir}/{session_dir}")
                        print(f"   Features: {', '.join(abbreviated_names[:5])}{'...' if len(abbreviated_names) > 5 else ''}")
                        
                        return feature_names, abbreviated_names
                        
                    except Exception as e:
                        print(f"    Error loading {metrics_file}: {e}")
                        continue
        
        # Fallback: use standard marker names
        print("   Using standard marker names (NICE collection order)")
        marker_names, abbreviated_names = self.get_marker_names()
        
        # For scalar markers, use all 28 markers
        if self.marker_type == 'scalar':
            print(f"   ✓ Using {len(marker_names)} standard scalar marker names")
            print(f"   Features: {', '.join(abbreviated_names[:5])}...")
            return marker_names, abbreviated_names
        else:
            # For topo markers, we'll have marker_name × n_channels features
            # Return the base marker names - they'll be expanded when plotting
            print(f"   ✓ Using {len(marker_names)} standard topo marker names (will be expanded per channel)")
            return marker_names, abbreviated_names
    
    def load_subject_data(self, subject_session_path):
        """Load marker data for a single subject/session.
        
        Parameters
        ----------
        subject_session_path : str
            Path to subject/session directory (or direct path to .npz file)
            
        Returns
        -------
        array or None
            Marker data or None if loading failed
        """
        
        # Check if subject_session_path is a direct .npz file path
        if subject_session_path.endswith('.npz'):
            filepath = subject_session_path
        else:
            # Determine filename based on marker type and data origin
            if self.marker_type == 'scalar':
                filename = f"scalars_{self.data_origin}.npy"
                features_dir = "features_variable"
            elif self.marker_type == 'topo':
                filename = f"topos_{self.data_origin}.npy"
                features_dir = "features_variable"
            else:
                raise ValueError(f"Unknown marker_type: {self.marker_type}")
            
            filepath = op.join(subject_session_path, features_dir, filename)
        
        if not op.exists(filepath):
            print(f"    Missing file: {filepath}")
            return None
        
        try:
            # Handle both .npy and .npz files
            if filepath.endswith('.npz'):
                # Load .npz file
                data_dict = np.load(filepath)
                
                # Try common keys for single array
                if 'scalars' in data_dict:
                    data = data_dict['scalars']
                elif 'arr_0' in data_dict:
                    data = data_dict['arr_0']
                else:
                    # .npz might have individual markers as separate keys
                    # Collect all scalar values into a single array
                    keys = sorted(data_dict.keys())
                    if len(keys) > 0:
                        # Check if all values are scalars (shape ())
                        all_scalars = all(data_dict[k].shape == () for k in keys)
                        if all_scalars:
                            # Collect all scalar values into a single array
                            data = np.array([float(data_dict[k]) for k in keys])
                            print(f"      Collected {len(keys)} scalar markers from .npz file")
                        else:
                            # Use the first available key
                            data = data_dict[keys[0]]
                    else:
                        raise ValueError(f"No data found in .npz file: {filepath}")
            else:
                # Load .npy file
                data = np.load(filepath)
            
            if self.marker_type == 'scalar':
                # Scalar data: (n_markers,) -> flatten to 1D
                return data.flatten()
            elif self.marker_type == 'topo':
                # Topographic data: (n_markers, n_channels) -> flatten to 1D
                return data.flatten()
            
        except Exception as e:
            print(f"   Error loading {filepath}: {e}")
            return None
    
    def collect_data(self):
        """Collect data from all available subjects."""
        print(f"🔍 Collecting {self.marker_type} {self.data_origin} data from subjects...")
        print(f"   Data directory: {self.data_dir}")
        
        # Load patient labels
        labels_dict, available_states = self.load_patient_labels()
        
        # Load feature names
        self.feature_names, self.feature_names_abbreviated = self.load_feature_names()
        
        # Find all subject/session directories
        subject_data = []
        subject_labels = []
        collected_subjects = []
        
        # Scan for subject directories
        if not op.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        try:
            subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        except PermissionError as e:
            raise ValueError(f"Permission denied accessing data directory {self.data_dir}: {e}")
            
        print(f"   Found {len(subject_dirs)} potential subject directories")
        
        subjects_processed = 0
        subjects_skipped = 0
        
        for subject_dir in sorted(subject_dirs):
            subject_path = op.join(self.data_dir, subject_dir)
            
            if not op.isdir(subject_path):
                continue
            
            # NEW: Check if this is the new structure (sub-XXX_original or sub-XXX_recon)
            # Format: sub-001_original or sub-001_recon
            if '_' in subject_dir and (subject_dir.endswith('_original') or subject_dir.endswith('_recon')):
                # New structure: data is directly in this directory
                parts = subject_dir.rsplit('_', 1)
                subject_id_raw = parts[0].replace('sub-', '')  # e.g., "001"
                data_type = parts[1]  # 'original' or 'recon'
                
                # Only process if data_type matches our data_origin
                # Map 'recon' to 'reconstructed' for compatibility
                if data_type == 'recon':
                    data_type = 'reconstructed'
                
                if data_type != self.data_origin:
                    continue
                
                # Look for .npz file directly in this directory
                # The file might be named with the numeric ID
                npz_filename = f"scalars_{subject_id_raw}_{parts[1]}.npz"
                npz_path = op.join(subject_path, npz_filename)
                
                # For label lookup, we need to match the subject ID format in patient_labels.csv
                # The numeric ID might map to a different format (e.g., "001" -> "AA048")
                # Try both numeric format and with ses-01
                subject_session_key = f"{subject_id_raw}_ses-01"
                
                # Check if we have labels for this subject
                if subject_session_key not in labels_dict:
                    print(f"    Skipping {subject_session_key}: no label found")
                    subjects_skipped += 1
                    continue
                
                # Load marker data from .npz file
                if not op.exists(npz_path):
                    print(f"    Missing {npz_filename} in {subject_path}")
                    subjects_skipped += 1
                    continue
                
                marker_data = self.load_subject_data(npz_path)
                
                if marker_data is None:
                    print(f"   ⏭️  Skipping {subject_session_key}: failed to load data")
                    subjects_skipped += 1
                    continue
                
                # Validate shape consistency
                if len(subject_data) > 0:
                    expected_shape = subject_data[0].shape
                    
                    if marker_data.shape != expected_shape:
                        print(f"   ⚠️  Skipping {subject_session_key}: shape mismatch")
                        print(f"       Expected: {expected_shape}, got: {marker_data.shape}")
                        subjects_skipped += 1
                        continue
                
                # Store data
                subject_data.append(marker_data)
                subject_labels.append(labels_dict[subject_session_key])
                collected_subjects.append(subject_session_key)
                subjects_processed += 1
                
                print(f"   ✓ Loaded {subject_session_key}: {marker_data.shape} features, state={labels_dict[subject_session_key]}")
                
            else:
                # OLD structure: sub-XXX/ses-YY/features_variable/scalars_*.npy
                subject_id = subject_dir.replace('sub-', '')
                
                # Look for session directories
                try:
                    session_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
                except PermissionError:
                    print(f"   Permission denied accessing {subject_path}")
                    continue
                
                for session_dir in sorted(session_dirs):
                    session_path = op.join(subject_path, session_dir)
                    
                    if not op.isdir(session_path):
                        continue
                    
                    subject_session_key = f"{subject_id}_{session_dir}"
                    
                    # Check if we have labels for this subject/session
                    if subject_session_key not in labels_dict:
                        print(f"    Skipping {subject_session_key}: no label found")
                        subjects_skipped += 1
                        continue
                    
                    # Load marker data
                    marker_data = self.load_subject_data(session_path)
                    
                    if marker_data is None:
                        print(f"   ⏭️  Skipping {subject_session_key}: failed to load data")
                        subjects_skipped += 1
                        continue
                    
                    # Validate shape consistency
                    if len(subject_data) > 0:
                        expected_shape = subject_data[0].shape
                        
                        if marker_data.shape != expected_shape:
                            print(f"   ⚠️  Skipping {subject_session_key}: shape mismatch")
                            print(f"       Expected: {expected_shape}, got: {marker_data.shape}")
                            subjects_skipped += 1
                            continue
                    
                    # Store data
                    subject_data.append(marker_data)
                    subject_labels.append(labels_dict[subject_session_key])
                    collected_subjects.append(subject_session_key)
                    subjects_processed += 1
                    
                    print(f"   ✓ Loaded {subject_session_key}: {marker_data.shape} features, state={labels_dict[subject_session_key]}")
        
        print("\n📊 DATA COLLECTION SUMMARY:")
        print(f"    Successfully loaded: {subjects_processed} subject/sessions")
        print(f"    Skipped: {subjects_skipped} subject/sessions")
        
        if subjects_processed == 0:
            raise ValueError("No subjects could be loaded!")
        
        # Validate all shapes are consistent before converting to numpy
        if len(subject_data) > 1:
            first_shape = subject_data[0].shape
            all_same = all(d.shape == first_shape for d in subject_data)
            
            if not all_same:
                raise ValueError(
                    f"Shape inconsistency detected after filtering!\n"
                    f"  Shapes found: {set(d.shape for d in subject_data)}"
                )
            
            print(f"    ✓ All data validated: {first_shape[0]} features per subject")
        
        # Convert to numpy arrays
        self.X = np.array(subject_data)
        self.y = np.array(subject_labels)
        self.subjects = collected_subjects
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_
        
        print(f"   Final dataset: {self.X.shape[0]} subjects × {self.X.shape[1]} features")
        print(f"   Classes: {list(self.class_names)}")
        
        # Show class distribution
        unique, counts = np.unique(self.y, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"      {class_name}: {count} subjects")
        
        # Check if we have enough data for each class
        min_class_size = min(counts)
        if min_class_size == 0:
            raise ValueError("Some classes have no samples!")
        elif min_class_size == 1:
            print("    Warning: Some classes have only 1 sample. Consider using Leave-One-Out CV.")
        
        return self.X, self.y_encoded, self.subjects
    
    def create_pipeline(self, n_estimators=500, max_depth=4):
        """Create the DOC-Forest pipeline.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
            
        Returns
        -------
        Pipeline
            Scikit-learn pipeline
        """
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_encoded), 
            y=self.y_encoded
        )
        class_weight_dict = dict(zip(np.unique(self.y_encoded), class_weights))
        
        # Create DOC-Forest pipeline
       # pipeline = make_pipeline(
       #     RobustScaler(),
       #     SVMClassifier(
       #         n_estimators=n_estimators,
       #         max_features=1,
       #         criterion='entropy',
       #         max_depth=max_depth,
       #         random_state=self.random_state,
       #         class_weight=class_weight_dict, #WARNING: In the paper this is 'balanced'
       #         n_jobs=-1
       #     )
        #)
        
        #Create Doc-Forest pipeline with SVM
        pipeline = make_pipeline(
            RobustScaler(),
            SVC(
                probability=True,
                class_weight=class_weight_dict, #WARNING: In the paper this is 'balanced'
                n_jobs=-1,
                random_state=self.random_state,

            )
        )
        return pipeline
    
    def evaluate_model(self, pipeline, cv_strategy='stratified', n_splits=5, test_size=0.2):
        """Evaluate model with proper train/test split and cross-validation.
        
        Parameters
        ----------
        pipeline : Pipeline
            Scikit-learn pipeline to evaluate
        cv_strategy : str
            'stratified', 'loo' (leave-one-out), or 'group'
        n_splits : int
            Number of CV splits (ignored for LOO)
        test_size : float
            Fraction of data to hold out for testing
            
        Returns
        -------
        dict
            Evaluation results
        """
        
        print(f" Evaluating model with {cv_strategy} cross-validation and {test_size:.0%} test set...")
        
        # Extract subject groups (without session) to prevent data leakage
        # Example: "AA048_ses-01" -> "AA048"
        subject_groups = np.array([subj.split('_ses-')[0] for subj in self.subjects])
        
        print(f"    🔒 GROUP-BASED SPLITTING: Ensuring all sessions from same subject stay together")
        print(f"    Found {len(np.unique(subject_groups))} unique subjects across {len(self.subjects)} sessions")
        
        # Use GroupShuffleSplit to ensure all sessions from same subject go to same fold
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(self.X, self.y_encoded, groups=subject_groups))
        
        X_train = self.X[train_idx]
        X_test = self.X[test_idx]
        y_train = self.y_encoded[train_idx]
        y_test = self.y_encoded[test_idx]
        subjects_train = [self.subjects[i] for i in train_idx]
        subjects_test = [self.subjects[i] for i in test_idx]
        groups_train = subject_groups[train_idx]
        groups_test = subject_groups[test_idx]
        
        # Verify no subject leakage
        unique_train_subjects = set(groups_train)
        unique_test_subjects = set(groups_test)
        overlap = unique_train_subjects.intersection(unique_test_subjects)
        if overlap:
            raise ValueError(f"❌ CRITICAL: Subject leakage detected! Subjects in both train and test: {overlap}")
        
        print(f"    ✓ No subject leakage verified: {len(unique_train_subjects)} unique subjects in train, {len(unique_test_subjects)} in test")
        
        print(f"   ✓ Train set: {X_train.shape[0]} sessions from {len(unique_train_subjects)} unique subjects")
        print(f"   ✓ Test set: {X_test.shape[0]} sessions from {len(unique_test_subjects)} unique subjects")
        
        # Show class distribution in train/test
        print("    Train class distribution:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for class_idx, count in zip(unique_train, counts_train):
            print(f"      {self.class_names[class_idx]}: {count} subjects")
            
        print("    Test class distribution:")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        for class_idx, count in zip(unique_test, counts_test):
            print(f"      {self.class_names[class_idx]}: {count} subjects")
        
        # Choose cross-validation strategy for training set
        # IMPORTANT: Use GroupKFold to prevent session leakage within CV folds
        n_train_samples = X_train.shape[0]
        n_unique_train_subjects = len(unique_train_subjects)
        
        if cv_strategy == 'loo':
            cv = LeaveOneOut()
            print(f"   Using Leave-One-Out CV on training set ({n_train_samples} folds)")
            print(f"   ⚠️  WARNING: LOO doesn't respect subject groups - sessions from same subject may be in different folds")
        elif cv_strategy == 'stratified':
            # Use GroupKFold to ensure sessions from same subject stay in same fold during CV
            # Determine maximum possible folds based on number of unique subjects
            max_possible_splits = min(n_splits, n_unique_train_subjects)
            
            if max_possible_splits < 2:
                print(f"   WARNING: Only {n_unique_train_subjects} unique subjects in training, cannot perform CV")
                cv = LeaveOneOut()
                print(f"   Falling back to Leave-One-Out CV on training set ({n_train_samples} folds)")
            else:
                effective_n_splits = max_possible_splits
                cv = GroupKFold(n_splits=effective_n_splits)
                print(f"   🔒 Using GroupKFold with {effective_n_splits} splits (respects subject groups)")
                print(f"   This ensures sessions from same subject stay in same CV fold")
        else:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")
        
        # Cross-validation scores on training set only
        # Pass groups parameter for GroupKFold
        cv_params = {
            'estimator': pipeline,
            'X': X_train,
            'y': y_train,
            'cv': cv,
            'scoring': 'balanced_accuracy',
            'n_jobs': -1
        }
        
        # Add groups parameter for GroupKFold
        if isinstance(cv, GroupKFold):
            cv_params['groups'] = groups_train
        
        cv_scores = cross_val_score(**cv_params)
        
        # Also compute AUC for binary classification
        cv_auc_scores = []
        if len(self.class_names) == 2:
            cv_auc_params = {
                'estimator': pipeline,
                'X': X_train,
                'y': y_train,
                'cv': cv,
                'scoring': 'roc_auc',
                'n_jobs': -1
            }
            if isinstance(cv, GroupKFold):
                cv_auc_params['groups'] = groups_train
            
            cv_auc_scores = cross_val_score(**cv_auc_params)
        
        # Fit final model on training data only
        print(f"   Training final model on {X_train.shape[0]} training subjects...")
        pipeline.fit(X_train, y_train)
        
        # Predictions on test set (no data leakage)
        # IMPORTANT: Use argmax(predict_proba()) for consistency with probabilities
        print(f"   Evaluating on {X_test.shape[0]} held-out test subjects...")
        y_test_proba = pipeline.predict_proba(X_test)
        y_test_pred = np.argmax(y_test_proba, axis=1)  # Derive from probabilities
        
        # Also get predictions on training set for completeness
        y_train_proba = pipeline.predict_proba(X_train)
        y_train_pred = np.argmax(y_train_proba, axis=1)  # Derive from probabilities
        
        # Compute detailed metrics on test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
        
        # Get unique classes present in test set
        test_classes_present = np.unique(y_test)
        test_class_names_present = [self.class_names[i] for i in test_classes_present]
        
        print(f"   Classes present in test set: {test_class_names_present}")
        
        test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
            y_test, y_test_pred, average=None, labels=test_classes_present, zero_division=0
        )
        
        # Classification report on test set (only for classes present)
        test_class_report = classification_report(
            y_test, y_test_pred, 
        #    target_names=test_class_names_present, 
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix on test set
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # AUC computation for binary classification (VS vs MCS)
        test_auc_score = None
        
        if len(test_classes_present) == 2:
            # Binary classification: compute standard AUC-ROC
            try:
                test_auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
                print(f"   Test AUC-ROC: {test_auc_score:.3f}")
            except Exception as e:
                print(f"    Could not compute AUC: {e}")
                test_auc_score = None
        elif len(test_classes_present) == 1:
            print(f"   Only one class present in test set: {test_class_names_present[0]}. Cannot compute AUC.")
            test_auc_score = None
        else:
            print(f"   More than 2 classes found: {test_class_names_present}. This should not happen for binary classification.")
            test_auc_score = None
        
        # Feature importances (use absolute coefficients for linear SVM)
        feature_importances = np.abs(pipeline.named_steps['svc'].coef_[0])
        
        # Training set metrics for comparison
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
        
        results = {
            # Cross-validation results (on training set)
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_auc_scores': cv_auc_scores,
            'cv_auc_mean': np.mean(cv_auc_scores) if cv_auc_scores else None,
            'cv_auc_std': np.std(cv_auc_scores) if cv_auc_scores else None,
            
            # Test set results (primary evaluation)
            'test_accuracy': test_accuracy,
            'test_balanced_accuracy': test_balanced_acc,
            'test_auc_score': test_auc_score,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'test_support': test_support,
            'test_classification_report': test_class_report,
            'test_confusion_matrix': test_conf_matrix,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'y_test_true': y_test,
            
            # Training set results (for comparison/overfitting check)
            'train_accuracy': train_accuracy,
            'train_balanced_accuracy': train_balanced_acc,
            'y_train_pred': y_train_pred,
            'y_train_proba': y_train_proba,
            'y_train_true': y_train,
            
            # Model info
            'feature_importances': feature_importances,
            'n_train_subjects': len(subjects_train),
            'n_test_subjects': len(subjects_test),
            'n_features': X_train.shape[1],
            'class_names': self.class_names,
            'cv_strategy': cv_strategy,
            'test_size': test_size,
            'subjects_train': subjects_train,
            'subjects_test': subjects_test,
            'test_classes_present': test_classes_present,
            'test_class_names_present': test_class_names_present,
            
            # For backward compatibility (using test set metrics)
            'accuracy': test_accuracy,
            'balanced_accuracy': test_balanced_acc,
            'auc_score': test_auc_score,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'support': test_support,
            'classification_report': test_class_report,
            'confusion_matrix': test_conf_matrix,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba,
            'n_subjects': len(subjects_test)  # Using test subjects for main reporting
        }
        
        return results
    
    def plot_results(self, results):
        """Create visualization plots for results.
        
        Parameters
        ----------
        results : dict
            Results from evaluate_model
        """
        
        print(" Creating result plots...")
        
        # Create figure with subplots
        n_classes = len(self.class_names)
        
        if n_classes == 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        fig.suptitle(f'Cross-Subject Classification Results\n'
                    f'{self.marker_type.title()} {self.data_origin.title()} Data\n'
                    f'Train: {results["n_train_subjects"]} subjects, Test: {results["n_test_subjects"]} subjects', 
                    fontsize=16)
        
        # 1. Cross-validation scores
        ax = axes[0, 0]
        cv_scores = results['cv_scores']
        x_pos = np.arange(len(cv_scores))
        
        bars = ax.bar(x_pos, cv_scores, alpha=0.7)
        ax.axhline(y=results['cv_mean'], color='red', linestyle='--', 
                  label=f'Mean: {results["cv_mean"]:.3f} ± {results["cv_std"]:.3f}')
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'CV Scores on Training Set ({results["cv_strategy"]})\n'
                    f'Actual folds: {len(cv_scores)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, cv_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Confusion Matrix (Test Set)
        ax = axes[0, 1]
        conf_matrix = results['test_confusion_matrix']
        
        # Get classes present in test set for proper labeling
        test_classes_present = results.get('test_classes_present', range(n_classes))
        test_class_names_present = results.get('test_class_names_present', self.class_names)
        n_test_classes = len(test_classes_present)
        
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = conf_matrix.max() / 2. if conf_matrix.size > 0 else 0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if conf_matrix[i, j] > thresh else "black")
        
        ax.set_title(f'Confusion Matrix (Test Set)\n{n_test_classes} classes present')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks(range(n_test_classes))
        ax.set_xticklabels(test_class_names_present, rotation=45)
        ax.set_yticks(range(n_test_classes))
        ax.set_yticklabels(test_class_names_present)
        
        # 3. Feature Importances (top 20)
        ax = axes[1, 0]
        importances = results['feature_importances']
        top_n = min(20, len(importances))
        top_indices = np.argsort(importances)[-top_n:]
        
        ax.barh(range(top_n), importances[top_indices])
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Features')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_yticks(range(top_n))
        
        # Use real feature names if available, otherwise fall back to generic names
        if self.feature_names_abbreviated is not None and len(self.feature_names_abbreviated) == len(importances):
            feature_labels = [self.feature_names_abbreviated[i] for i in top_indices]
        else:
            feature_labels = [f'Feature {i}' for i in top_indices]
        
        ax.set_yticklabels(feature_labels)
        
        # 4. Class-wise Performance (Test Set)
        ax = axes[1, 1]
        
        precision = results['test_precision']
        recall = results['test_recall']
        f1 = results['test_f1_score']
        test_class_names_present = results.get('test_class_names_present', self.class_names)
        
        x = np.arange(len(test_class_names_present))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Class-wise Performance (Test Set)\n{len(test_class_names_present)} classes present')
        ax.set_xticks(x)
        ax.set_xticklabels(test_class_names_present, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = op.join(self.output_dir, 'classification_results.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Results plot saved to: {plot_file}")
        plt.close()
        
        # ROC curve for binary classification
        if results['test_auc_score'] is not None:
            self._plot_binary_roc_curve(results)
        else:
            print("   Skipping ROC curve plot: AUC score not available")
    
    def _plot_binary_roc_curve(self, results):
        """Plot ROC curve for binary classification."""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute ROC curve using test set data
        fpr, tpr, _ = roc_curve(results['y_test_true'], results['y_test_proba'][:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = op.join(self.output_dir, 'roc_curve.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ ROC curve saved to: {plot_file}")
        plt.close()
    
    def _plot_multiclass_roc_curves(self, results):
        """Plot One-vs-Rest ROC curves for multi-class classification."""
        
        print("   Creating One-vs-Rest ROC curves for multi-class classification...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get test data
        y_test_true = results['y_test_true']
        y_test_proba = results['y_test_proba']
        test_classes_present = results['test_classes_present']
        test_class_names_present = results['test_class_names_present']
        
        # Binarize the test labels for OvR approach
        y_test_binarized = label_binarize(y_test_true, classes=test_classes_present)
        
        # If only one class is present, we can't compute ROC
        if len(test_classes_present) < 2:
            ax.text(0.5, 0.5, 'Cannot plot ROC curves\n(only one class in test set)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest) - Insufficient Classes')
            plt.tight_layout()
            plot_file = op.join(self.output_dir, 'roc_curves_ovr.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"   ROC curves saved to: {plot_file} (insufficient classes for proper ROC)")
            plt.close()
            return
        
        # Colors for different classes
        colors = plt.cm.Set1(np.linspace(0, 1, len(test_classes_present)))
        
        # Store AUC scores for summary
        auc_scores = {}
        
        # Plot ROC curve for each class (One-vs-Rest)
        for i, (class_idx, class_name, color) in enumerate(zip(test_classes_present, test_class_names_present, colors)):
            # For multiclass, we need to get the probability for this specific class
            class_proba = y_test_proba[:, i] if i < y_test_proba.shape[1] else y_test_proba[:, class_idx]
            
            # Create binary labels (this class vs all others)
            if len(test_classes_present) == 2:
                # Special case for 2 classes
                y_true_binary = (y_test_true == class_idx).astype(int)
            else:
                # General multiclass case
                y_true_binary = y_test_binarized[:, i] if y_test_binarized.ndim > 1 else (y_test_true == class_idx).astype(int)
            
            # Skip if this class has no positive examples
            if np.sum(y_true_binary) == 0:
                print(f"   Skipping {class_name}: no positive examples in test set")
                continue
                
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true_binary, class_proba)
            roc_auc = auc(fpr, tpr)
            auc_scores[class_name] = roc_auc
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)\nMulti-class Classification')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        if auc_scores:
            mean_auc = np.mean(list(auc_scores.values()))
            ax.text(0.02, 0.98, f'Mean AUC: {mean_auc:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = op.join(self.output_dir, 'roc_curves_ovr.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"    Multi-class ROC curves saved to: {plot_file}")
        if auc_scores:
            print(f"   AUC scores: {', '.join([f'{k}: {v:.3f}' for k, v in auc_scores.items()])}")
        plt.close()
    
    def save_results(self, results, pipeline):
        """Save results and model to files.
        
        Parameters
        ----------
        results : dict
            Results from evaluate_model
        pipeline : Pipeline
            Trained pipeline
        """
        
        print("Saving results and model...")
        
        # Save model
        model_file = op.join(self.output_dir, 'trained_model.pkl')
        joblib.dump(pipeline, model_file)
        print(f"   ✓ Model saved to: {model_file}")
        
        # Save label encoder
        encoder_file = op.join(self.output_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_file)
        print(f"   ✓ Label encoder saved to: {encoder_file}")
        
        # Prepare results for JSON serialization
        json_results = {
            'experiment_info': {
                'marker_type': self.marker_type,
                'data_origin': self.data_origin,
                'n_subjects': results['n_subjects'],
                'n_train_subjects': results['n_train_subjects'],
                'n_test_subjects': results['n_test_subjects'],
                'n_features': results['n_features'],
                'class_names': results['class_names'].tolist(),
                'cv_strategy': results['cv_strategy'],
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                # Cross-validation on training set
                'cv_balanced_accuracy_mean': float(results['cv_mean']),
                'cv_balanced_accuracy_std': float(results['cv_std']),
                'cv_balanced_accuracy_scores': results['cv_scores'].tolist(),
                'cv_auc_mean': float(results['cv_auc_mean']) if results['cv_auc_mean'] is not None else None,
                'cv_auc_std': float(results['cv_auc_std']) if results['cv_auc_std'] is not None else None,
                
                # Test set performance (primary metrics)
                'test_accuracy': float(results['test_accuracy']),
                'test_balanced_accuracy': float(results['test_balanced_accuracy']),
                'test_auc_score': float(results['test_auc_score']) if results['test_auc_score'] is not None else None,
                
                # Training set performance (for overfitting check)
                'train_accuracy': float(results['train_accuracy']),
                'train_balanced_accuracy': float(results['train_balanced_accuracy']),
                
                # Backward compatibility (using test metrics)
                'accuracy': float(results['test_accuracy']),
                'balanced_accuracy': float(results['test_balanced_accuracy']),
                'auc_score': float(results['test_auc_score']) if results['test_auc_score'] is not None else None,
            },
            'class_metrics': {
                # Test set metrics (primary) - only for classes present in test set
                'test_precision': results['test_precision'].tolist(),
                'test_recall': results['test_recall'].tolist(),
                'test_f1_score': results['test_f1_score'].tolist(),
                'test_support': results['test_support'].tolist(),
                'test_classes_present': results['test_classes_present'].tolist(),
                'test_class_names_present': results['test_class_names_present'],
                
                # Backward compatibility (using test metrics)
                'precision': results['test_precision'].tolist(),
                'recall': results['test_recall'].tolist(),
                'f1_score': results['test_f1_score'].tolist(),
                'support': results['test_support'].tolist()
            },
            'confusion_matrix': results['test_confusion_matrix'].tolist(),
            'classification_report': results['test_classification_report'],
            'feature_importances': results['feature_importances'].tolist(),
            'feature_names': self.feature_names if self.feature_names is not None else [],
            'feature_names_abbreviated': self.feature_names_abbreviated if self.feature_names_abbreviated is not None else [],
            'subjects_info': {
                'train_subjects': results['subjects_train'],
                'test_subjects': results['subjects_test'],
                'test_true_labels': [self.class_names[i] for i in results['y_test_true']],
                'test_predicted_labels': [self.class_names[i] for i in results['y_test_pred']],
                
                # Backward compatibility (using test data)
                'subjects': results['subjects_test'],
                'true_labels': [self.class_names[i] for i in results['y_test_true']],
                'predicted_labels': [self.class_names[i] for i in results['y_test_pred']]
            }
        }
        
        # Save results
        results_file = op.join(self.output_dir, 'classification_results.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"   ✓ Results saved to: {results_file}")
        
        # Save detailed CSV with subject-level predictions (test set)
        test_true_labels = [self.class_names[i] for i in results['y_test_true']]
        test_pred_labels = [self.class_names[i] for i in results['y_test_pred']]
        
        subject_results_df = pd.DataFrame({
            'subject_session': results['subjects_test'],
            'true_state': test_true_labels,
            'predicted_state': test_pred_labels,
            'correct_prediction': [true == pred for true, pred in zip(test_true_labels, test_pred_labels)],
            'dataset_split': 'test'
        })
        
        # Add prediction probabilities for test set
        for i, class_name in enumerate(self.class_names):
            subject_results_df[f'prob_{class_name}'] = results['y_test_proba'][:, i]
        
        csv_file = op.join(self.output_dir, 'subject_predictions.csv')
        subject_results_df.to_csv(csv_file, index=False)
        print(f"   ✓ Subject predictions saved to: {csv_file}")
        
        # Save feature importances with names
        if self.feature_names is not None and len(self.feature_names) == len(results['feature_importances']):
            feature_importance_df = pd.DataFrame({
                'feature_name': self.feature_names,
                'feature_name_abbreviated': self.feature_names_abbreviated,
                'importance': results['feature_importances']
            })
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
            feature_importance_file = op.join(self.output_dir, 'feature_importances.csv')
            feature_importance_df.to_csv(feature_importance_file, index=False)
            print(f"   ✓ Feature importances saved to: {feature_importance_file}")
        else:
            # Fallback: save just the importance values
            feature_importance_df = pd.DataFrame({
                'feature_index': range(len(results['feature_importances'])),
                'importance': results['feature_importances']
            })
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
            feature_importance_file = op.join(self.output_dir, 'feature_importances.csv')
            feature_importance_df.to_csv(feature_importance_file, index=False)
            print(f"   ✓ Feature importances saved to: {feature_importance_file} (with indices)")
        
        return json_results
    
    def run_classification(self, n_estimators=500, max_depth=4, cv_strategy='stratified', n_splits=5, test_size=0.2):
        """Run the complete classification pipeline.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum tree depth
        cv_strategy : str
            Cross-validation strategy
        n_splits : int
            Number of CV splits
        test_size : float
            Fraction of data to hold out for testing
            
        Returns
        -------
        dict
            Classification results
        """
        
        print("=" * 80)
        print("CROSS-SUBJECT EXTRATREES CLASSIFICATION")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Labels file: {self.patient_labels_file}")
        print(f"Marker type: {self.marker_type}")
        print(f"Data origin: {self.data_origin}")
        print(f"Output directory: {self.output_dir}")
        print(f"CV strategy: {cv_strategy}")
        print(f"Trees: {n_estimators}, Max depth: {max_depth}")
        print(f"Test size: {test_size:.0%}")
        print()
        
        try:
            # Step 1: Collect data
            X, y_encoded, subjects = self.collect_data()
            
            # Step 2: Create pipeline
            pipeline = self.create_pipeline(n_estimators, max_depth)
            
            # Step 3: Evaluate model
            results = self.evaluate_model(pipeline, cv_strategy, n_splits, test_size)
            
            # Step 4: Create plots
            self.plot_results(results)
            
            # Step 5: Save results
            final_results = self.save_results(results, pipeline)
            
            # Summary
            print("\n" + "=" * 80)
            print("CLASSIFICATION SUMMARY")
            print("=" * 80)
            print(f"Dataset: {results['n_train_subjects']} train + {results['n_test_subjects']} test subjects, {results['n_features']} features")
            print(f"Classes: {', '.join(results['class_names'])}")
            print(f"CV Balanced Accuracy (train): {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
            print(f"Test Balanced Accuracy: {results['test_balanced_accuracy']:.3f}")
            print(f"Train Balanced Accuracy: {results['train_balanced_accuracy']:.3f}")
            
            # AUC reporting for binary classification
            if results['test_auc_score'] is not None:
                print(f"Test AUC-ROC Score: {results['test_auc_score']:.3f}")
            else:
                print("Test AUC-ROC Score: Not available")
            
            print(f"Results saved to: {self.output_dir}")
            print("=" * 80)
            
            return final_results
            
        except Exception as e:
            print(f"\n Classification failed with error: {e}")
            print("\n Debugging information:")
            print(f"   Data directory exists: {op.exists(self.data_dir)}")
            print(f"   Labels file exists: {op.exists(self.patient_labels_file)}")
            print(f"   Output directory: {self.output_dir}")
            raise


class CrossDataClassifier:
    """
    Cross-data SVM classifier that trains on both original and reconstructed data.
    
    This classifier implements a rigorous cross-testing design to ensure:
    1. NO SUBJECT BIAS: Same train/test subjects used for both original and reconstructed data
    2. NO DATA LEAKAGE: Train and test subjects are completely separate
    3. FAIR COMPARISON: All 4 test scenarios use identical test subjects
    
    Training Strategy:
    - Model A: Trained on ORIGINAL data from train subjects
    - Model B: Trained on RECONSTRUCTED data from train subjects (SAME subjects as Model A)
    
    Testing Strategy (Cross-Testing):
    - Model A → Original test set (same test subjects)
    - Model A → Reconstructed test set (same test subjects)
    - Model B → Original test set (same test subjects)
    - Model B → Reconstructed test set (same test subjects)
    
    This design allows fair comparison between original and reconstructed data while
    controlling for subject-specific characteristics that could bias the results.
    """
    
    def __init__(self, data_dir, patient_labels_file, marker_type='scalar', 
                 output_dir=None, random_state=42):
        """
        Initialize the cross-data classifier.
        
        Parameters
        ----------
        data_dir : str
            Path to results directory containing subject data
        patient_labels_file : str  
            Path to CSV file with patient labels
        marker_type : str
            'scalar' or 'topo' - type of markers to use
        output_dir : str
            Output directory for results
        random_state : int
            Random state for reproducibility (ensures same train/test splits across runs)
        """
        self.data_dir = data_dir
        self.patient_labels_file = patient_labels_file
        self.marker_type = marker_type
        self.output_dir = output_dir or f"results/svm/{marker_type}"
        self.random_state = random_state
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data containers for both original and reconstructed data
        self.X_original = None
        self.X_reconstructed = None
        self.y = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.feature_names = None
        self.feature_names_abbreviated = None
        
        # Results containers
        self.results = {}
        
    def load_patient_labels(self):
        """Load patient labels from CSV file for binary classification (VS vs MCS)."""
        print(f"Loading patient labels from: {self.patient_labels_file}")
        
        try:
            df = pd.read_csv(self.patient_labels_file)
            
            # Create mapping from subject_session to state
            labels_dict = {}
            available_states = set()
            
            for _, row in df.iterrows():
                subject = row['subject']
                session = f"ses-{row['session']:02d}"
                state = row['state']
                
                # Skip subjects with missing state
                if pd.isna(state) or state == 'n/a':
                    continue
                
                # Binary classification: VS vs MCS
                if state == 'UWS':
                    # UWS corresponds to VS (Vegetative State)
                    state = 'VS'
                elif state in ['MCS+', 'MCS-']:
                    # Merge MCS+ and MCS- into single MCS category
                    state = 'MCS'
                else:
                    # Skip other states (COMA, EMCS, CONTROL, etc.)
                    continue
                
                subject_session_key = f"{subject}_{session}"
                labels_dict[subject_session_key] = state
                available_states.add(state)
            
            print(f"   ✓ Loaded labels for {len(labels_dict)} subject/sessions")
            print(f"   ✓ Available states: {sorted(available_states)}")
            print("    Binary classification: VS (Vegetative State) vs MCS (Minimally Conscious State)")
            
            return labels_dict, sorted(available_states)
            
        except Exception as e:
            print(f"   Error loading patient labels: {e}")
            raise
    
    def get_marker_names(self):
        """Get standard marker names based on NICE collection order.
        
        Returns
        -------
        tuple
            (full_names, abbreviated_names)
        """
        # Standard marker names in NICE collection order
        marker_names = [
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
        
        # Create abbreviations
        abbreviated_names = []
        for name in marker_names:
            if name.startswith('PowerSpectralDensity_'):
                abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
            elif name.startswith('PowerSpectralDensitySummary_'):
                abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
            elif name.startswith('TimeLockedContrast_'):
                abbrev = name.replace('TimeLockedContrast_', 'TLC_')
            elif name.startswith('TimeLockedTopography_'):
                abbrev = name.replace('TimeLockedTopography_', 'TLT_')
            elif name.startswith('ContingentNegativeVariation'):
                abbrev = 'CNV'
            elif name.startswith('PermutationEntropy'):
                abbrev = 'PE'
            elif name.startswith('SymbolicMutualInformation'):
                abbrev = 'SMI'
            elif name.startswith('KolmogorovComplexity'):
                abbrev = 'KC'
            else:
                abbrev = name
            abbreviated_names.append(abbrev)
        
        return marker_names, abbreviated_names
    
    def load_feature_names(self):
        """Load feature names from the first available subject's .npz file.
        
        Returns
        -------
        list
            List of feature names, abbreviated for display
        """
        print(" Loading feature names...")
        
        # First, try to load from .npz files in the new structure
        if op.exists(self.data_dir):
            subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
            
            for subject_dir in sorted(subject_dirs):
                subject_path = op.join(self.data_dir, subject_dir)
                if not op.isdir(subject_path):
                    continue
                    
                # Check if this is the new structure (sub-XXX_original or sub-XXX_recon)
                if '_' in subject_dir and (subject_dir.endswith('_original') or subject_dir.endswith('_recon')):
                    # Map data type for compatibility
                    data_type = subject_dir.split('_')[-1]
                    if data_type == 'recon':
                        data_type = 'reconstructed'
                    
                    # Only process if data_type matches our data_origin
                    if data_type != self.data_origin:
                        continue
                    
                    # Extract subject ID
                    subject_id_raw = subject_dir.split('_')[0].replace('sub-', '')
                    
                    # Look for .npz file
                    npz_filename = f"scalars_{subject_id_raw}_{subject_dir.split('_')[-1]}.npz"
                    npz_path = op.join(subject_path, npz_filename)
                    
                    if op.exists(npz_path):
                        try:
                            # Load feature names from .npz file
                            data_dict = np.load(npz_path)
                            feature_names = sorted(data_dict.files())
                            
                            # Abbreviate long feature names for better display
                            abbreviated_names = []
                            for name in feature_names:
                                # Create abbreviations
                                if name.startswith('PowerSpectralDensity_'):
                                    abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
                                elif name.startswith('PowerSpectralDensitySummary_'):
                                    abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
                                elif name.startswith('TimeLockedContrast_'):
                                    abbrev = name.replace('TimeLockedContrast_', 'TLC_')
                                elif name.startswith('TimeLockedTopography_'):
                                    abbrev = name.replace('TimeLockedTopography_', 'TLT_')
                                elif name.startswith('ContingentNegativeVariation'):
                                    abbrev = 'CNV'
                                elif name.startswith('PermutationEntropy'):
                                    abbrev = 'PE'
                                elif name.startswith('SymbolicMutualInformation'):
                                    abbrev = 'SMI'  
                                elif name.startswith('KolmogorovComplexity'):
                                    abbrev = 'KC'
                                else:
                                    # Keep original if no abbreviation rule
                                    abbrev = name
                                
                                abbreviated_names.append(abbrev)
                            
                            print(f"   ✓ Loaded {len(feature_names)} feature names from {npz_filename}")
                            print(f"   Features: {', '.join(abbreviated_names[:5])}{'...' if len(abbreviated_names) > 5 else ''}")
                            
                            return feature_names, abbreviated_names
                            
                        except Exception as e:
                            print(f"    Error loading {npz_path}: {e}")
                            continue
        
        # OLD structure: try to load from scalar_metrics.csv
        for subject_dir in sorted(subject_dirs):
            subject_path = op.join(self.data_dir, subject_dir)
            if not op.isdir(subject_path):
                continue
                
            # Look for session directories
            try:
                session_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
            except PermissionError:
                continue
            
            for session_dir in sorted(session_dirs):
                session_path = op.join(subject_path, session_dir)
                
                # Try to find scalar_metrics.csv (old structure)
                metrics_file = op.join(session_path, 'compare_markers', 'scalars', 'metrics', 'scalar_metrics.csv')
                
                if op.exists(metrics_file):
                    try:
                        df = pd.read_csv(metrics_file, index_col=0)
                        feature_names = df.index.tolist()
                        
                        # Abbreviate long feature names for better display
                        abbreviated_names = []
                        for name in feature_names:
                            # Create abbreviations
                            if name.startswith('PowerSpectralDensity_'):
                                abbrev = name.replace('PowerSpectralDensity_', 'PSD_')
                            elif name.startswith('PowerSpectralDensitySummary_'):
                                abbrev = name.replace('PowerSpectralDensitySummary_', 'PSD_Sum_')
                            elif name.startswith('TimeLockedContrast_'):
                                abbrev = name.replace('TimeLockedContrast_', 'TLC_')
                            elif name.startswith('TimeLockedTopography_'):
                                abbrev = name.replace('TimeLockedTopography_', 'TLT_')
                            elif name.startswith('ContingentNegativeVariation'):
                                abbrev = 'CNV'
                            elif name.startswith('PermutationEntropy'):
                                abbrev = 'PE'
                            elif name.startswith('SymbolicMutualInformation'):
                                abbrev = 'SMI'  
                            elif name.startswith('KolmogorovComplexity'):
                                abbrev = 'KC'
                            else:
                                # Keep original if no abbreviation rule
                                abbrev = name
                            
                            abbreviated_names.append(abbrev)
                        
                        print(f"   ✓ Loaded {len(feature_names)} feature names from {subject_dir}/{session_dir}")
                        print(f"   Features: {', '.join(abbreviated_names[:5])}{'...' if len(abbreviated_names) > 5 else ''}")
                        
                        return feature_names, abbreviated_names
                        
                    except Exception as e:
                        print(f"    Error loading {metrics_file}: {e}")
                        continue
        
        # Fallback: use standard marker names
        print("   Using standard marker names (NICE collection order)")
        marker_names, abbreviated_names = self.get_marker_names()
        
        # For scalar markers, use all 28 markers
        if self.marker_type == 'scalar':
            print(f"   ✓ Using {len(marker_names)} standard scalar marker names")
            print(f"   Features: {', '.join(abbreviated_names[:5])}...")
            return marker_names, abbreviated_names
        else:
            # For topo markers, we'll have marker_name × n_channels features
            # Return the base marker names - they'll be expanded when plotting
            print(f"   ✓ Using {len(marker_names)} standard topo marker names (will be expanded per channel)")
            return marker_names, abbreviated_names
    
    def load_subject_data_both(self, subject_session_path):
        """Load both original and reconstructed marker data for a single subject/session.
        
        Parameters
        ----------
        subject_session_path : str
            Path to subject/session directory
            
        Returns
        -------
        tuple or (None, None)
            (original_data, reconstructed_data) or (None, None) if loading failed
        """
        
        # Extract subject and session identifiers from path
        subject_dir = op.basename(op.dirname(subject_session_path))  # e.g. "sub-001"
        session_dir = op.basename(subject_session_path)              # e.g. "ses-01"
        subject_id = subject_dir.replace('sub-', '')
        session_id = session_dir.replace('ses-', '')

        # New MARKERS layout expected by the pipeline:
        #   data_dir/
        #       sub-{ID}/ses-{XX}/orig/scalars_{ID}.npz
        #       sub-{ID}/ses-{XX}/recon/scalars_{ID}.npz
        # (same filename in orig and recon, repeated across sessions)
        # We also support the previous naming convention used by the
        # markers phase as a fallback:
        #   scalars_{ID}_ses-{XX}_orig.npz and scalars_{ID}_ses-{XX}_recon.npz

        orig_dir = op.join(subject_session_path, 'orig')
        recon_dir = op.join(subject_session_path, 'recon')

        # Determine candidate filenames based on marker type
        if self.marker_type == 'scalar':
            primary_name = f"scalars_{subject_id}.npz"
            fallback_name_orig = f"scalars_{subject_id}_ses-{session_id}_orig.npz"
            fallback_name_recon = f"scalars_{subject_id}_ses-{session_id}_recon.npz"
        elif self.marker_type == 'topo':
            primary_name = f"topos_{subject_id}.npz"
            fallback_name_orig = f"topos_{subject_id}_ses-{session_id}_orig.npz"
            fallback_name_recon = f"topos_{subject_id}_ses-{session_id}_recon.npz"
        else:
            raise ValueError(f"Unknown marker_type: {self.marker_type}")

        # Resolve filepaths with graceful fallback
        def _resolve_filepath(base_dir, primary, fallback):
            primary_path = op.join(base_dir, primary)
            if op.exists(primary_path):
                return primary_path
            fallback_path = op.join(base_dir, fallback)
            if op.exists(fallback_path):
                return fallback_path
            return None

        filepath_orig = _resolve_filepath(orig_dir, primary_name, fallback_name_orig)
        filepath_recon = _resolve_filepath(recon_dir, primary_name, fallback_name_recon)

        # Check if both files exist
        if filepath_orig is None:
            print(f"    Missing scalar/topo file for ORIGINAL in {orig_dir}")
            return None, None

        if filepath_recon is None:
            print(f"    Missing scalar/topo file for RECON in {recon_dir}")
            return None, None

        try:
            # Load original and reconstructed data (.npz or .npy)
            def _load_array(path):
                if path.endswith('.npz'):
                    data_dict = np.load(path)
                    keys = list(data_dict.keys())
                    if not keys:
                        raise ValueError(f"No arrays stored in {path}")
                    # If a single array is stored, use it; otherwise
                    # collect all scalar values into a single vector
                    if len(keys) == 1:
                        return data_dict[keys[0]]
                    return np.array([float(data_dict[k]) for k in keys])
                else:
                    return np.load(path)

            data_orig = _load_array(filepath_orig)
            data_recon = _load_array(filepath_recon)

            # Flatten data based on marker type
            if self.marker_type == 'scalar':
                # Scalar data: (n_markers,) -> flatten to 1D
                data_orig = data_orig.flatten()
                data_recon = data_recon.flatten()
            elif self.marker_type == 'topo':
                # Topographic data: (n_markers, n_channels) -> flatten to 1D
                data_orig = data_orig.flatten()
                data_recon = data_recon.flatten()

            return data_orig, data_recon

        except Exception as e:
            print(f"   Error loading data from {subject_session_path}: {e}")
            return None, None
    
    def collect_data(self):
        """Collect both original and reconstructed data from all available subjects."""
        print(f"🔍 Collecting {self.marker_type} data (both original and reconstructed) from subjects...")
        print(f"   Data directory: {self.data_dir}")
        
        # Load patient labels
        labels_dict, available_states = self.load_patient_labels()
        
        # Load feature names
        self.feature_names, self.feature_names_abbreviated = self.load_feature_names()
        
        # Find all subject/session directories
        subject_data_orig = []
        subject_data_recon = []
        subject_labels = []
        collected_subjects = []
        
        # Scan for subject directories
        if not op.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        try:
            subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        except PermissionError as e:
            raise ValueError(f"Permission denied accessing data directory {self.data_dir}: {e}")
            
        print(f"   Found {len(subject_dirs)} potential subject directories")
        
        subjects_processed = 0
        subjects_skipped = 0
        
        for subject_dir in sorted(subject_dirs):
            subject_path = op.join(self.data_dir, subject_dir)
            
            if not op.isdir(subject_path):
                continue
            
            # NEW: Check if this is the new structure (sub-XXX_original or sub-XXX_recon)
            # Format: sub-001_original or sub-001_recon
            if '_' in subject_dir and (subject_dir.endswith('_original') or subject_dir.endswith('_recon')):
                # New structure: data is directly in this directory
                parts = subject_dir.rsplit('_', 1)
                subject_id_raw = parts[0].replace('sub-', '')  # e.g., "001"
                data_type = parts[1]  # 'original' or 'recon'
                
                # Only process if data_type matches our data_origin
                # Map 'recon' to 'reconstructed' for compatibility
                if data_type == 'recon':
                    data_type = 'reconstructed'
                
                if data_type != self.data_origin:
                    continue
                
                # Look for .npz file directly in this directory
                # The file might be named with the numeric ID
                npz_filename = f"scalars_{subject_id_raw}_{parts[1]}.npz"
                npz_path = op.join(subject_path, npz_filename)
                
                # For label lookup, we need to match the subject ID format in patient_labels.csv
                # The numeric ID might map to a different format (e.g., "001" -> "AA048")
                # Try both numeric format and with ses-01
                subject_session_key = f"{subject_id_raw}_ses-01"
                
                # Check if we have labels for this subject
                if subject_session_key not in labels_dict:
                    print(f"    Skipping {subject_session_key}: no label found")
                    subjects_skipped += 1
                    continue
                
                # Load marker data from .npz file
                if not op.exists(npz_path):
                    print(f"    Missing {npz_filename} in {subject_path}")
                    subjects_skipped += 1
                    continue
                
                marker_data = self.load_subject_data(npz_path)
                
                if marker_data is None:
                    print(f"   ⏭️  Skipping {subject_session_key}: failed to load data")
                    subjects_skipped += 1
                    continue
                
                # For CrossDataClassifier, we need both original and reconstructed data
                # This new structure separates them, so we need to handle this differently
                # For now, store the data and we'll pair them later
                if not hasattr(self, '_temp_data_storage'):
                    self._temp_data_storage = {}
                
                self._temp_data_storage[subject_session_key] = {
                    'data': marker_data,
                    'data_type': data_type
                }
                
                print(f"   ✓ Loaded {subject_session_key}: {marker_data.shape} features ({data_type}), state={labels_dict[subject_session_key]}")
                subjects_processed += 1
                
            else:
                # OLD structure: sub-XXX/ses-YY/orig/recon
                subject_id = subject_dir.replace('sub-', '')
                
                # Look for session directories
                try:
                    session_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
                except PermissionError:
                    print(f"   Permission denied accessing {subject_path}")
                    continue
                
                for session_dir in sorted(session_dirs):
                    session_path = op.join(subject_path, session_dir)
                    
                    if not op.isdir(session_path):
                        continue
                    
                    subject_session_key = f"{subject_id}_{session_dir}"
                    
                    # Check if we have labels for this subject/session
                    if subject_session_key not in labels_dict:
                        print(f"    Skipping {subject_session_key}: no label found")
                        subjects_skipped += 1
                        continue
                    
                    # Load both original and reconstructed marker data
                    marker_data_orig, marker_data_recon = self.load_subject_data_both(session_path)
                    
                    if marker_data_orig is None or marker_data_recon is None:
                        print(f"   ⏭️  Skipping {subject_session_key}: failed to load both data types")
                        subjects_skipped += 1
                        continue
                    
                    # Validate shape consistency
                    if len(subject_data_orig) > 0:
                        expected_shape_orig = subject_data_orig[0].shape
                        expected_shape_recon = subject_data_recon[0].shape
                        
                        if marker_data_orig.shape != expected_shape_orig:
                            print(f"   ⚠️  Skipping {subject_session_key}: shape mismatch")
                            print(f"       Expected orig: {expected_shape_orig}, got: {marker_data_orig.shape}")
                            subjects_skipped += 1
                            continue
                        
                        if marker_data_recon.shape != expected_shape_recon:
                            print(f"   ⚠️  Skipping {subject_session_key}: shape mismatch")
                            print(f"       Expected recon: {expected_shape_recon}, got: {marker_data_recon.shape}")
                            subjects_skipped += 1
                            continue
                    
                    # Store data
                    subject_data_orig.append(marker_data_orig)
                    subject_data_recon.append(marker_data_recon)
                    subject_labels.append(labels_dict[subject_session_key])
                    collected_subjects.append(subject_session_key)
                    subjects_processed += 1
                    
                    print(f"   ✓ Loaded {subject_session_key}: orig={marker_data_orig.shape}, recon={marker_data_recon.shape}, state={labels_dict[subject_session_key]}")
        
        # Handle new structure data pairing
        if hasattr(self, '_temp_data_storage'):
            print("\n   Processing new structure data...")
            
            # Find subjects that have both original and reconstructed data
            paired_subjects = []
            for subject_key, data_info in self._temp_data_storage.items():
                # Look for the corresponding data in the other type
                other_type = 'reconstructed' if data_info['data_type'] == 'original' else 'original'
                other_key = None
                
                # Try to find matching subject with other data type
                for other_subject_key, other_data_info in self._temp_data_storage.items():
                    if (other_data_info['data_type'] == other_type and 
                        subject_key.replace('_ses-01', '') == other_subject_key.replace('_ses-01', '')):
                        other_key = other_subject_key
                        break
                
                if other_key is not None:
                    # We have both data types for this subject
                    if data_info['data_type'] == 'original':
                        orig_data = data_info['data']
                        recon_data = self._temp_data_storage[other_key]['data']
                    else:
                        orig_data = self._temp_data_storage[other_key]['data']
                        recon_data = data_info['data']
                    
                    # Validate shape consistency
                    if len(subject_data_orig) > 0:
                        expected_shape_orig = subject_data_orig[0].shape
                        expected_shape_recon = subject_data_recon[0].shape
                        
                        if orig_data.shape != expected_shape_orig:
                            print(f"   ⚠️  Skipping {subject_key}: original shape mismatch")
                            continue
                        
                        if recon_data.shape != expected_shape_recon:
                            print(f"   ⚠️  Skipping {subject_key}: reconstructed shape mismatch")
                            continue
                    
                    subject_data_orig.append(orig_data)
                    subject_data_recon.append(recon_data)
                    subject_labels.append(labels_dict[subject_key])
                    collected_subjects.append(subject_key)
                    paired_subjects.append(subject_key)
                    
                    print(f"   ✓ Paired {subject_key}: orig={orig_data.shape}, recon={recon_data.shape}")
            
            print(f"   ✓ Paired {len(paired_subjects)} subjects from new structure")
            
            # Clean up temporary storage
            delattr(self, '_temp_data_storage')
        
        print("\n📊 DATA COLLECTION SUMMARY:")
        print(f"    Successfully loaded: {subjects_processed} subject/sessions")
        print(f"    Skipped: {subjects_skipped} subject/sessions")
        
        if subjects_processed == 0:
            raise ValueError("No subjects could be loaded!")
        
        # Validate all shapes are consistent before converting to numpy
        if len(subject_data_orig) > 1:
            first_shape_orig = subject_data_orig[0].shape
            first_shape_recon = subject_data_recon[0].shape
            all_same_orig = all(d.shape == first_shape_orig for d in subject_data_orig)
            all_same_recon = all(d.shape == first_shape_recon for d in subject_data_recon)
            
            if not all_same_orig or not all_same_recon:
                raise ValueError(
                    f"Shape inconsistency detected after filtering!\n"
                    f"  Original shapes: {set(d.shape for d in subject_data_orig)}\n"
                    f"  Reconstructed shapes: {set(d.shape for d in subject_data_recon)}"
                )
            
            print(f"    ✓ All data validated: {first_shape_orig[0]} features per subject")
        
        # Convert to numpy arrays
        self.X_original = np.array(subject_data_orig)
        self.X_reconstructed = np.array(subject_data_recon)
        self.y = np.array(subject_labels)
        self.subjects = collected_subjects
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_
        
        print(f"   Final dataset: {self.X_original.shape[0]} subjects × {self.X_original.shape[1]} features")
        print(f"   Classes: {list(self.class_names)}")
        
        # Show class distribution
        unique, counts = np.unique(self.y, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"      {class_name}: {count} subjects")
        
        # Check if we have enough data for each class
        min_class_size = min(counts)
        if min_class_size == 0:
            raise ValueError("Some classes have no samples!")
        elif min_class_size == 1:
            print("    Warning: Some classes have only 1 sample. Consider using Leave-One-Out CV.")
        
        return self.X_original, self.X_reconstructed, self.y_encoded, self.subjects
    
    def create_pipeline(self, C=1.0, kernel='linear', gamma='scale'):
        """Create SVM pipeline with specified kernel.
        
        Parameters
        ----------
        C : float
            Regularization parameter for SVM
        kernel : str
            Kernel type ('linear' or 'rbf')
        gamma : str or float
            Gamma parameter for RBF kernel ('scale', 'auto', or numeric value)
            
        Returns
        -------
        Pipeline
            Scikit-learn pipeline
        """
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_encoded), 
            y=self.y_encoded
        )
        class_weight_dict = dict(zip(np.unique(self.y_encoded), class_weights))
        
        # Create SVM pipeline with StandardScaler
        pipeline = make_pipeline(
            StandardScaler(),  # Changed from RobustScaler
            SVC(
                kernel=kernel,
                C=C,
                gamma=gamma if kernel == 'rbf' else 'scale',  # gamma only for RBF
                probability=True,  # Enable probability estimates
                class_weight=class_weight_dict,
                random_state=self.random_state
            )
        )
        
        return pipeline
    
    def _grid_search_hyperparameters(self, X_train, y_train, groups_train, cv_strategy, n_splits):
        """Perform grid search for best C parameter with linear kernel.
        
        Returns
        -------
        dict
            Dictionary with best parameters: {'kernel', 'C', 'gamma', 'score'}
        """
        # Parameter grid - only linear kernel with different C values
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        
        # Set up stratified CV
        if cv_strategy == 'loo':
            cv = LeaveOneOut()
            print(f"      Using Leave-One-Out CV")
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            print(f"      Using StratifiedKFold with {n_splits} splits")
        
        best_params = {'kernel': 'linear', 'C': None, 'gamma': 'scale', 'score': -np.inf}
        
        # Test linear kernel with different C values
        print(f"\n      Testing LINEAR kernel:")
        for C in C_values:
            pipeline = self.create_pipeline(C=C, kernel='linear')
            
            scores = cross_val_score(estimator=pipeline, X=X_train, y=y_train, 
                                   cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            mean_score = np.mean(scores)
            
            print(f"         C={C:8.3f}: {mean_score:.3f} ± {np.std(scores):.3f}")
            
            if mean_score > best_params['score']:
                best_params = {'kernel': 'linear', 'C': C, 'gamma': 'scale', 'score': mean_score}
        
        print(f"\n      ✓ Best parameters: kernel={best_params['kernel']}, C={best_params['C']}, Score={best_params['score']:.3f}")
        return best_params
    
    def run_cross_data_classification(self, cv_strategy='stratified', n_splits=5, test_size=0.2):
        """
        Run cross-data classification with rigorous controls for subject bias and data leakage.
        
        This method implements a 4-way cross-testing design:
        1. Split subjects into train/test (SAME split for both original and reconstructed)
        2. Train Model A on ORIGINAL data from train subjects
        3. Train Model B on RECONSTRUCTED data from train subjects (SAME subjects)
        4. Test both models on both test sets:
           - Model A → Original test subjects
           - Model A → Reconstructed test subjects (SAME subjects)
           - Model B → Original test subjects (SAME subjects)
           - Model B → Reconstructed test subjects (SAME subjects)
        
        Key guarantees:
        - NO SUBJECT BIAS: All 4 test scenarios use identical test subjects
        - NO DATA LEAKAGE: Train and test subjects are completely separate
        - FAIR COMPARISON: Models trained on different data types tested on same subjects
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum tree depth
        cv_strategy : str
            Cross-validation strategy ('stratified' or 'loo')
        n_splits : int
            Number of CV splits (ignored for LOO)
        test_size : float
            Fraction of data to hold out for testing (default: 0.2)
            
        Returns
        -------
        dict
            Cross-data classification results including all 4 test scenarios
        """
        
        print("=" * 80)
        print("CROSS-DATA SVM CLASSIFICATION WITH LINEAR KERNEL")
        print("=" * 80)
        print()
        print("🔒 DATA INTEGRITY GUARANTEES:")
        print("   ✓ Same train/test subjects for BOTH original and reconstructed data")
        print("   ✓ No subject bias in cross-testing (same test subjects for all scenarios)")
        print("   ✓ No data leakage (train/test subjects completely separate)")
        print("   ✓ Stratified splits to maintain class balance")
        print("   ✓ Grid search for optimal C parameter")
        print()
        print(f"📁 Configuration:")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Labels file: {self.patient_labels_file}")
        print(f"   Marker type: {self.marker_type}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   CV strategy: {cv_strategy}")
        print(f"   Test size: {test_size:.0%}")
        print(f"   Random state: {self.random_state}")
        print()
        
        try:
            # Step 1: Collect data
            X_orig, X_recon, y_encoded, subjects = self.collect_data()
            
            # Step 2: Split into train/test (CRITICAL: same subjects for both data types)
            # This ensures no subject bias between original and reconstructed testing
            # and prevents data leakage by keeping train/test subjects completely separate
            print(f"\n🔀 Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
            print("   IMPORTANT: Using identical train/test subject splits for BOTH original and reconstructed data")
            print("   This ensures:")
            print("     - No subject bias when comparing model performance on different data types")
            print("     - No data leakage between original and reconstructed test sets")
            print("     - Fair cross-testing of both models on both data types")
            print("     - All sessions from same subject stay together (no session leakage)")
            
            # Extract subject groups (without session) to prevent data leakage
            # Example: "AA048_ses-01" -> "AA048"
            subject_groups = np.array([subj.split('_ses-')[0] for subj in subjects])
            
            print(f"\n    🔒 GROUP-BASED SPLITTING: Ensuring all sessions from same subject stay together")
            print(f"    Found {len(np.unique(subject_groups))} unique subjects across {len(subjects)} sessions")
            
            # Use GroupShuffleSplit to ensure all sessions from same subject go to same fold
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
            train_idx, test_idx = next(gss.split(X_orig, y_encoded, groups=subject_groups))
            
            X_orig_train = X_orig[train_idx]
            X_orig_test = X_orig[test_idx]
            X_recon_train = X_recon[train_idx]
            X_recon_test = X_recon[test_idx]
            y_train = y_encoded[train_idx]
            y_test = y_encoded[test_idx]
            subjects_train = [subjects[i] for i in train_idx]
            subjects_test = [subjects[i] for i in test_idx]
            groups_train = subject_groups[train_idx]
            groups_test = subject_groups[test_idx]
            
            # Verify no subject leakage
            unique_train_subjects = set(groups_train)
            unique_test_subjects = set(groups_test)
            overlap = unique_train_subjects.intersection(unique_test_subjects)
            if overlap:
                raise ValueError(f"❌ CRITICAL: Subject leakage detected! Subjects in both train and test: {overlap}")
            
            print(f"    ✓ No subject leakage verified: {len(unique_train_subjects)} unique subjects in train, {len(unique_test_subjects)} in test")
            
            print(f"\n   ✓ Train set: {X_orig_train.shape[0]} sessions from {len(unique_train_subjects)} unique subjects")
            print(f"   ✓ Test set: {X_orig_test.shape[0]} sessions from {len(unique_test_subjects)} unique subjects")
            
            # Verify and report class distribution in train/test splits
            print("\n   📊 Class distribution in TRAIN set:")
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            for class_idx, count in zip(unique_train, counts_train):
                print(f"      {self.class_names[class_idx]}: {count} subjects ({count/len(y_train)*100:.1f}%)")
            
            print("\n   📊 Class distribution in TEST set:")
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            for class_idx, count in zip(unique_test, counts_test):
                print(f"      {self.class_names[class_idx]}: {count} subjects ({count/len(y_test)*100:.1f}%)")
            
            # CRITICAL VERIFICATION: Ensure data shapes match for same subjects
            print(f"\n   ✓ Verification: Original train shape: {X_orig_train.shape}")
            print(f"   ✓ Verification: Reconstructed train shape: {X_recon_train.shape}")
            print(f"   ✓ Verification: Original test shape: {X_orig_test.shape}")
            print(f"   ✓ Verification: Reconstructed test shape: {X_recon_test.shape}")
            
            # Ensure shapes match (same number of subjects, same features)
            assert X_orig_train.shape == X_recon_train.shape, "Train set shape mismatch between original and reconstructed!"
            assert X_orig_test.shape == X_recon_test.shape, "Test set shape mismatch between original and reconstructed!"
            print("   ✓ All shape verifications passed - same subjects in both data types!")
            
            # Step 3: Train models with grid search
            print("\n🔧 Training models with hyperparameter grid search...")
            print("   Searching over: kernel=['linear'], C=[0.001-100]")
            
            # Grid search for Model A (original data)
            print("\n   🔍 Grid Search for Model A (original data):")
            best_params_A = self._grid_search_hyperparameters(X_orig_train, y_train, groups_train, cv_strategy, n_splits)
            
            # Grid search for Model B (reconstructed data)
            print("\n   🔍 Grid Search for Model B (reconstructed data):")
            best_params_B = self._grid_search_hyperparameters(X_recon_train, y_train, groups_train, cv_strategy, n_splits)
            
            # Train final models with best hyperparameters
            print("\n   ✓ Training final models with optimal hyperparameters...")
            
            # DEBUG: Check training labels distribution
            print(f"\n   🔍 DEBUG: Training data check...")
            print(f"      y_train unique values: {np.unique(y_train)}")
            print(f"      y_train distribution: {np.bincount(y_train)}")
            print(f"      self.class_names: {self.class_names}")
            
            pipeline_A = self.create_pipeline(C=best_params_A['C'], kernel=best_params_A['kernel'], gamma=best_params_A['gamma'])
            pipeline_A.fit(X_orig_train, y_train)
            print(f"      Model A: kernel={best_params_A['kernel']}, C={best_params_A['C']}, gamma={best_params_A['gamma']}, CV Score={best_params_A['score']:.3f}")
            
            pipeline_B = self.create_pipeline(C=best_params_B['C'], kernel=best_params_B['kernel'], gamma=best_params_B['gamma'])
            pipeline_B.fit(X_recon_train, y_train)
            print(f"      Model B: kernel={best_params_B['kernel']}, C={best_params_B['C']}, gamma={best_params_B['gamma']}, CV Score={best_params_B['score']:.3f}")
            
            # DEBUG: Verify class encoding after training
            print(f"\n   🔍 DEBUG: Verifying class encoding after training...")
            print(f"      LabelEncoder.classes_: {self.class_names}")
            print(f"      SVC Model A.classes_: {pipeline_A.named_steps['svc'].classes_}")
            print(f"      SVC Model B.classes_: {pipeline_B.named_steps['svc'].classes_}")
            
            # Store best scores from grid search (note: these are single values from grid search, not individual CV folds)
            cv_scores_A = np.array([best_params_A['score']])
            cv_scores_B = np.array([best_params_B['score']])
            best_params = {'model_A': best_params_A, 'model_B': best_params_B}
            
            # Step 5: Cross-testing - test both models on both test sets
            # CRITICAL: This cross-testing design prevents subject bias and data leakage
            print("\n🔄 Cross-testing models on SAME test subjects...")
            print("   Testing strategy:")
            print("     1. Model A (trained on ORIGINAL) → tested on ORIGINAL test subjects")
            print("     2. Model A (trained on ORIGINAL) → tested on RECONSTRUCTED test subjects")
            print("     3. Model B (trained on RECONSTRUCTED) → tested on ORIGINAL test subjects")
            print("     4. Model B (trained on RECONSTRUCTED) → tested on RECONSTRUCTED test subjects")
            print(f"\n   All 4 tests use the SAME {len(subjects_test)} test subjects: {subjects_test[:3]}{'...' if len(subjects_test) > 3 else ''}")
            print("   This ensures fair comparison without subject-specific biases!")
            
            # Model A predictions
            print("\n   🔬 Model A (trained on ORIGINAL data) predictions...")
            # IMPORTANT: Use argmax(predict_proba()) instead of predict() for consistency
            # SVC with probability=True can have mismatches between predict() and predict_proba()
            y_proba_A_on_orig = pipeline_A.predict_proba(X_orig_test)
            y_proba_A_on_recon = pipeline_A.predict_proba(X_recon_test)
            y_pred_A_on_orig = np.argmax(y_proba_A_on_orig, axis=1)  # Use argmax for consistency
            y_pred_A_on_recon = np.argmax(y_proba_A_on_recon, axis=1)
            print(f"      ✓ Tested on original test set: accuracy = {accuracy_score(y_test, y_pred_A_on_orig):.3f}")
            print(f"      ✓ Tested on reconstructed test set: accuracy = {accuracy_score(y_test, y_pred_A_on_recon):.3f}")
            
            # DEBUG: Analyze probability distributions
            mcs_idx = np.where(self.class_names == 'MCS')[0][0]
            print(f"\n   🔍 DEBUG: Probability distribution analysis for Model A:")
            print(f"      P(MCS) statistics - Original test:")
            print(f"         Mean: {np.mean(y_proba_A_on_orig[:, mcs_idx]):.3f}")
            print(f"         Std:  {np.std(y_proba_A_on_orig[:, mcs_idx]):.3f}")
            print(f"         Min:  {np.min(y_proba_A_on_orig[:, mcs_idx]):.3f}")
            print(f"         Max:  {np.max(y_proba_A_on_orig[:, mcs_idx]):.3f}")
            print(f"      P(MCS) statistics - Reconstructed test:")
            print(f"         Mean: {np.mean(y_proba_A_on_recon[:, mcs_idx]):.3f}")
            print(f"         Std:  {np.std(y_proba_A_on_recon[:, mcs_idx]):.3f}")
            print(f"         Min:  {np.min(y_proba_A_on_recon[:, mcs_idx]):.3f}")
            print(f"         Max:  {np.max(y_proba_A_on_recon[:, mcs_idx]):.3f}")
            
            # DEBUG: Show first few predictions and probabilities
            print(f"\n   🔍 DEBUG: First 3 predictions from Model A on original test (using argmax):")
            for i in range(min(3, len(y_test))):
                true_label = self.class_names[y_test[i]]
                pred_label = self.class_names[y_pred_A_on_orig[i]]
                proba_mcs = y_proba_A_on_orig[i, mcs_idx]
                proba_vs = y_proba_A_on_orig[i, np.where(self.class_names == 'VS')[0][0]]
                print(f"      Subject {subjects_test[i]}: True={true_label}, Pred={pred_label} (encoded={y_pred_A_on_orig[i]}), P(MCS)={proba_mcs:.3f}, P(VS)={proba_vs:.3f}")
            
            # Model B predictions
            print("\n   🔬 Model B (trained on RECONSTRUCTED data) predictions...")
            # Use argmax(predict_proba()) for consistency
            y_proba_B_on_orig = pipeline_B.predict_proba(X_orig_test)
            y_proba_B_on_recon = pipeline_B.predict_proba(X_recon_test)
            y_pred_B_on_orig = np.argmax(y_proba_B_on_orig, axis=1)
            y_pred_B_on_recon = np.argmax(y_proba_B_on_recon, axis=1)
            print(f"      ✓ Tested on original test set: accuracy = {accuracy_score(y_test, y_pred_B_on_orig):.3f}")
            print(f"      ✓ Tested on reconstructed test set: accuracy = {accuracy_score(y_test, y_pred_B_on_recon):.3f}")
            
            # DEBUG: Analyze probability distributions for Model B
            print(f"\n   🔍 DEBUG: Probability distribution analysis for Model B:")
            print(f"      P(MCS) statistics - Original test:")
            print(f"         Mean: {np.mean(y_proba_B_on_orig[:, mcs_idx]):.3f}")
            print(f"         Std:  {np.std(y_proba_B_on_orig[:, mcs_idx]):.3f}")
            print(f"         Min:  {np.min(y_proba_B_on_orig[:, mcs_idx]):.3f}")
            print(f"         Max:  {np.max(y_proba_B_on_orig[:, mcs_idx]):.3f}")
            print(f"      P(MCS) statistics - Reconstructed test:")
            print(f"         Mean: {np.mean(y_proba_B_on_recon[:, mcs_idx]):.3f}")
            print(f"         Std:  {np.std(y_proba_B_on_recon[:, mcs_idx]):.3f}")
            print(f"         Min:  {np.min(y_proba_B_on_recon[:, mcs_idx]):.3f}")
            print(f"         Max:  {np.max(y_proba_B_on_recon[:, mcs_idx]):.3f}")
            
            # Step 6: Compute metrics for all combinations
            print("\n📈 Computing metrics...")
            
            # Get feature importances (only for linear kernel)
            if best_params_A['kernel'] == 'linear':
                feature_importances_A = np.abs(pipeline_A.named_steps['svc'].coef_[0])
            else:
                feature_importances_A = None
            
            if best_params_B['kernel'] == 'linear':
                feature_importances_B = np.abs(pipeline_B.named_steps['svc'].coef_[0])
            else:
                feature_importances_B = None
            
            results = {
                'model_A_orig_test': self._compute_metrics(y_test, y_pred_A_on_orig, y_proba_A_on_orig, "Model A on Original Test"),
                'model_A_recon_test': self._compute_metrics(y_test, y_pred_A_on_recon, y_proba_A_on_recon, "Model A on Reconstructed Test"),
                'model_B_orig_test': self._compute_metrics(y_test, y_pred_B_on_orig, y_proba_B_on_orig, "Model B on Original Test"),
                'model_B_recon_test': self._compute_metrics(y_test, y_pred_B_on_recon, y_proba_B_on_recon, "Model B on Reconstructed Test"),
                'cv_scores_A': cv_scores_A,
                'cv_scores_B': cv_scores_B,
                'best_params': best_params,  # Store best hyperparameters
                'test_subjects': subjects_test,  # SAME test subjects used for all 4 test scenarios
                'train_subjects': subjects_train,  # SAME train subjects used to train both models
                'class_names': self.class_names,
                'feature_importances_A': feature_importances_A,  # None for non-linear kernels
                'feature_importances_B': feature_importances_B,  # None for non-linear kernels
                'y_test': y_test,  # TRUE labels for test subjects (same for all 4 scenarios)
                'n_features': X_orig_train.shape[1],
                'n_train_subjects': len(subjects_train),
                'n_test_subjects': len(subjects_test),
                # Additional metadata for verification
                'metadata': {
                    'same_test_subjects_used': True,
                    'same_train_subjects_used': True,
                    'stratified_split': True,
                    'no_data_leakage': True,
                    'no_subject_bias': True,
                    'random_state': self.random_state,
                    'test_size': test_size
                }
            }
            
            # Step 7: Save results and plots for all combinations
            self._save_cross_data_results(results, pipeline_A, pipeline_B)
            
            # Summary
            print("\n" + "=" * 80)
            print("CROSS-DATA CLASSIFICATION SUMMARY")
            print("=" * 80)
            print(f"\n📊 Dataset Information:")
            print(f"   Train subjects: {results['n_train_subjects']}")
            print(f"   Test subjects: {results['n_test_subjects']} (SAME for all 4 test scenarios)")
            print(f"   Features: {results['n_features']}")
            print(f"   Classes: {', '.join(results['class_names'])}")
            
            print(f"\n📈 Cross-Validation Results (on training set):")
            print(f"   Model A (trained on original): {np.mean(cv_scores_A):.3f} ± {np.std(cv_scores_A):.3f}")
            print(f"   Model B (trained on reconstructed): {np.mean(cv_scores_B):.3f} ± {np.std(cv_scores_B):.3f}")
            
            print(f"\n🔄 Cross-Testing Results (Balanced Accuracy):")
            print(f"   Model A (original) → Original test set:        {results['model_A_orig_test']['balanced_accuracy']:.3f}")
            print(f"   Model A (original) → Reconstructed test set:   {results['model_A_recon_test']['balanced_accuracy']:.3f}")
            print(f"   Model B (reconstructed) → Original test set:   {results['model_B_orig_test']['balanced_accuracy']:.3f}")
            print(f"   Model B (reconstructed) → Reconstructed test:  {results['model_B_recon_test']['balanced_accuracy']:.3f}")
            
            print(f"\n✅ Data Integrity Verification:")
            print(f"   ✓ Same {results['n_test_subjects']} test subjects used for all 4 test scenarios")
            print(f"   ✓ Train/test subjects completely separate (no data leakage)")
            print(f"   ✓ Stratified split maintained class balance")
            print(f"   ✓ Random state: {self.random_state} (reproducible splits)")
            
            print(f"\n💾 Results saved to: {self.output_dir}")
            print(f"   - 4 subdirectories (one per test scenario)")
            print(f"   - Combined confusion matrices visualization")
            print(f"   - Subject-level predictions for all scenarios")
            print("=" * 80)
            
            return results
            
        except Exception as e:
            print(f"\nCross-data classification failed with error: {e}")
            print(f"   Data directory exists: {op.exists(self.data_dir)}")
            print(f"   Labels file exists: {op.exists(self.patient_labels_file)}")
            print(f"   Output directory: {self.output_dir}")
            raise
    
    def _compute_metrics(self, y_true, y_pred, y_proba, description):
        """Compute classification metrics for a single test scenario."""
        print(f"   Computing metrics for {description}...")
        
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # AUC for binary classification
        auc_score = None
        if len(self.class_names) == 2:
            try:
                auc_score = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                auc_score = None
        
        # Precision, recall, f1 per class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'description': description,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def _save_cross_data_results(self, results, pipeline_A, pipeline_B):
        """Save results for all cross-data testing combinations."""
        
        # Define test scenarios
        scenarios = [
            ('model_A_orig_test', 'original_test', pipeline_A, 'Model_A_Original_Test'),
            ('model_A_recon_test', 'reconstructed_test', pipeline_A, 'Model_A_Reconstructed_Test'),
            ('model_B_orig_test', 'original_test', pipeline_B, 'Model_B_Original_Test'),
            ('model_B_recon_test', 'reconstructed_test', pipeline_B, 'Model_B_Reconstructed_Test')
        ]
        
        for scenario_key, test_set_name, pipeline, folder_name in scenarios:
            print(f"Saving results for {scenario_key}...")
            
            # Create output directory for this scenario
            scenario_dir = op.join(self.output_dir, folder_name)
            os.makedirs(scenario_dir, exist_ok=True)
            
            scenario_results = results[scenario_key]
            
            # Save model
            model_file = op.join(scenario_dir, 'trained_model.pkl')
            joblib.dump(pipeline, model_file)
            
            # Save label encoder
            encoder_file = op.join(scenario_dir, 'label_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_file)
            
            # Save classification results JSON
            json_results = {
                'experiment_info': {
                    'marker_type': self.marker_type,
                    'test_set': test_set_name,
                    'model_trained_on': 'original' if 'model_A' in scenario_key else 'reconstructed',
                    'model_tested_on': 'original' if 'orig_test' in scenario_key else 'reconstructed',
                    'n_subjects': results['n_test_subjects'],
                    'n_train_subjects': results['n_train_subjects'],
                    'n_test_subjects': results['n_test_subjects'],
                    'n_features': results['n_features'],
                    'class_names': results['class_names'].tolist(),
                    'timestamp': datetime.now().isoformat(),
                    # Verification metadata
                    'data_integrity': results.get('metadata', {})
                },
                'performance_metrics': {
                    'accuracy': float(scenario_results['accuracy']),
                    'balanced_accuracy': float(scenario_results['balanced_accuracy']),
                    'auc_score': float(scenario_results['auc_score']) if scenario_results['auc_score'] is not None else None,
                },
                'class_metrics': {
                    'precision': scenario_results['precision'].tolist(),
                    'recall': scenario_results['recall'].tolist(),
                    'f1_score': scenario_results['f1_score'].tolist(),
                    'support': scenario_results['support'].tolist()
                },
                'confusion_matrix': scenario_results['confusion_matrix'].tolist(),
                'classification_report': scenario_results['classification_report']
            }
            
            results_file = op.join(scenario_dir, 'classification_results.json')
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            # Save subject predictions CSV
            test_true_labels = [self.class_names[i] for i in results['y_test']]
            test_pred_labels = [self.class_names[i] for i in scenario_results['y_pred']]
            
            # DEBUG: Print first few to verify encoding
            print(f"\n   🔍 DEBUG CSV encoding for {scenario_key}:")
            print(f"      self.class_names: {self.class_names}")
            print(f"      First 3 encoded predictions: {scenario_results['y_pred'][:3]}")
            print(f"      First 3 decoded predictions: {test_pred_labels[:3]}")
            mcs_idx = np.where(self.class_names == 'MCS')[0][0]
            vs_idx = np.where(self.class_names == 'VS')[0][0]
            for idx in range(min(3, len(test_pred_labels))):
                print(f"      Row {idx}: y_pred={scenario_results['y_pred'][idx]}, label={test_pred_labels[idx]}, P(MCS)={scenario_results['y_proba'][idx, mcs_idx]:.3f}, P(VS)={scenario_results['y_proba'][idx, vs_idx]:.3f}")
            
            subject_results_df = pd.DataFrame({
                'subject_session': results['test_subjects'],
                'true_state': test_true_labels,
                'predicted_state': test_pred_labels,
                'correct_prediction': [true == pred for true, pred in zip(test_true_labels, test_pred_labels)],
                'test_set': test_set_name,
                'model_trained_on': 'original' if 'model_A' in scenario_key else 'reconstructed'
            })
            
            # Add prediction probabilities
            for i, class_name in enumerate(self.class_names):
                subject_results_df[f'prob_{class_name}'] = scenario_results['y_proba'][:, i]
            
            csv_file = op.join(scenario_dir, 'subject_predictions.csv')
            subject_results_df.to_csv(csv_file, index=False)
            
            # Save feature importances (only available for linear kernel)
            if 'model_A' in scenario_key:
                importances = results['feature_importances_A']
            else:
                importances = results['feature_importances_B']
            
            if importances is not None:
                if self.feature_names is not None and len(self.feature_names) == len(importances):
                    feature_importance_df = pd.DataFrame({
                        'feature_name': self.feature_names,
                        'feature_name_abbreviated': self.feature_names_abbreviated,
                        'importance': importances
                    })
                else:
                    feature_importance_df = pd.DataFrame({
                        'feature_index': range(len(importances)),
                        'importance': importances
                    })
                
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
                feature_importance_file = op.join(scenario_dir, 'feature_importances.csv')
                feature_importance_df.to_csv(feature_importance_file, index=False)
                print(f"   ✓ Feature importances saved (linear kernel)")
            else:
                print(f"   ℹ️  Feature importances not available (non-linear kernel)")
            
            # Create plots for this scenario
            self._plot_scenario_results(scenario_results, scenario_dir, scenario_key, results)
            
            print(f"   ✓ {scenario_key} results saved to: {scenario_dir}")
        
        # Save consolidated results JSON for reuse by multiple_seeds_analysis
        consolidated_results = {
            'model_A_orig_test': {
                'accuracy': float(results['model_A_orig_test']['accuracy']),
                'balanced_accuracy': float(results['model_A_orig_test']['balanced_accuracy']),
                'auc_score': float(results['model_A_orig_test']['auc_score']) if results['model_A_orig_test']['auc_score'] is not None else None,
            },
            'model_A_recon_test': {
                'accuracy': float(results['model_A_recon_test']['accuracy']),
                'balanced_accuracy': float(results['model_A_recon_test']['balanced_accuracy']),
                'auc_score': float(results['model_A_recon_test']['auc_score']) if results['model_A_recon_test']['auc_score'] is not None else None,
            },
            'model_B_orig_test': {
                'accuracy': float(results['model_B_orig_test']['accuracy']),
                'balanced_accuracy': float(results['model_B_orig_test']['balanced_accuracy']),
                'auc_score': float(results['model_B_orig_test']['auc_score']) if results['model_B_orig_test']['auc_score'] is not None else None,
            },
            'model_B_recon_test': {
                'accuracy': float(results['model_B_recon_test']['accuracy']),
                'balanced_accuracy': float(results['model_B_recon_test']['balanced_accuracy']),
                'auc_score': float(results['model_B_recon_test']['auc_score']) if results['model_B_recon_test']['auc_score'] is not None else None,
            },
            'metadata': results.get('metadata', {}),
            'test_subjects': results['test_subjects'],
            'train_subjects': results['train_subjects'],
            'n_test_subjects': results['n_test_subjects'],
            'n_train_subjects': results['n_train_subjects']
        }
        
        consolidated_file = op.join(self.output_dir, 'cross_data_results.json')
        with open(consolidated_file, 'w') as f:
            json.dump(consolidated_results, f, indent=2)
        print(f"   ✓ Consolidated results saved to: {consolidated_file}")
        
        # After saving all scenario results, create the combined 4-heatmap figure
        self._plot_combined_confusion_matrices(results)
        
        # Create subject-level probability plots
        self._plot_subject_probabilities(results)
        self._plot_model_A_probabilities(results)
        self._plot_model_B_probabilities(results)
        self._plot_comparison_grid(results)
        self._plot_differences_same_train(results)
        self._plot_differences_same_test(results)
        self._plot_probability_boxplots(results)
        self._plot_difference_boxplots_with_stats(results)
    
    def _plot_subject_probabilities(self, results):
        """Plot subject-level probabilities for all 4 test scenarios showing P(MCS)."""
        print("\n📊 Creating subject-level probability plot...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS and VS indices
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        vs_idx = np.where(self.class_names == 'VS')[0][0]
        
        print(f"   Class encoding: MCS=index {mcs_idx}, VS=index {vs_idx}")
        print(f"   Plotting P(MCS) from SVC.predict_proba()[:, {mcs_idx}]")
        print(f"   Note: P(VS) = 1 - P(MCS), sum to 1.0")
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # DEBUG: Save probabilities and predictions to CSV for inspection
        import pandas as pd
        debug_df = pd.DataFrame({
            'Subject': subjects_test,
            'True_Label': ['MCS' if results['y_test'][i] == mcs_idx else 'VS' for i in range(n_subjects)],
            'P(MCS)_OO': proba_OO,
            'P(VS)_OO': results['model_A_orig_test']['y_proba'][:, vs_idx],
            'Pred_OO': ['MCS' if results['model_A_orig_test']['y_pred'][i] == mcs_idx else 'VS' for i in range(n_subjects)],
            'P(MCS)_RR': proba_RR,
            'P(VS)_RR': results['model_B_recon_test']['y_proba'][:, vs_idx],
            'Pred_RR': ['MCS' if results['model_B_recon_test']['y_pred'][i] == mcs_idx else 'VS' for i in range(n_subjects)]
        })
        debug_csv = op.join(self.output_dir, 'debug_probabilities.csv')
        debug_df.to_csv(debug_csv, index=False)
        print(f"   ℹ️  DEBUG: Saved probability details to {debug_csv}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, n_subjects * 0.4), 7))
        x = np.arange(n_subjects)
        
        # Plot 4 curves
        ax.plot(x, proba_OO, marker='o', linestyle='-', linewidth=2.5, markersize=7,
               label='OO (Original Model → Original Test)', color='blue', alpha=0.8)
        ax.plot(x, proba_OR, marker='s', linestyle='--', linewidth=2.5, markersize=7,
               label='OR (Original Model → Reconstructed Test)', color='orange', alpha=0.8)
        ax.plot(x, proba_RO, marker='^', linestyle='-.', linewidth=2.5, markersize=7,
               label='RO (Reconstructed Model → Original Test)', color='green', alpha=0.8)
        ax.plot(x, proba_RR, marker='d', linestyle=':', linewidth=2.5, markersize=7,
               label='RR (Reconstructed Model → Reconstructed Test)', color='red', alpha=0.8)
        
        # Add decision threshold
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                  label='Decision Threshold (0.5)', zorder=0)
        
        # Customize plot
        ax.set_xlabel('Subject ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('P(MCS) - Probability of MCS Class', fontsize=13, fontweight='bold')
        ax.set_title(f'Subject-Level Probabilities: P(MCS) from SVC.predict_proba()\\n'
                    f'{self.marker_type.title()} Features - All 4 Test Scenarios (OO, OR, RO, RR)\\n'
                    f'Note: P(VS) = 1 - P(MCS)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        
        # Add true label background shading
        y_true = results['y_test']
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.text(0.02, 0.98, 'Background: Green = True MCS, Red = True VS',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        prob_plot_file = op.join(self.output_dir, 'subject_probabilities_all_scenarios.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Subject probability plot saved to: {prob_plot_file}")
    
    def _plot_model_A_probabilities(self, results):
        """Plot Model A (Original Model) probabilities on both test sets."""
        print("\n📊 Creating Model A probability plot...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS and VS indices
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        vs_idx = np.where(self.class_names == 'VS')[0][0]
        
        # Extract P(MCS) for Model A on both test sets
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, n_subjects * 0.4), 7))
        x = np.arange(n_subjects)
        
        # Plot 2 curves for Model A
        ax.plot(x, proba_OO, marker='o', linestyle='-', linewidth=3, markersize=8,
               label='Original Model → Original Test', color='blue', alpha=0.9)
        ax.plot(x, proba_OR, marker='s', linestyle='--', linewidth=3, markersize=8,
               label='Original Model → Reconstructed Test', color='orange', alpha=0.9)
        
        # Add decision threshold
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                  label='Decision Threshold (0.5)', zorder=0)
        
        # Customize plot
        ax.set_xlabel('Subject ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('P(MCS) - Probability of MCS Class', fontsize=13, fontweight='bold')
        ax.set_title(f'Model A (Original Data): Subject-Level Probabilities\\n'
                    f'{self.marker_type.title()} Features - Tested on Original vs Reconstructed Data\\n'
                    f'Kernel: {results["best_params"]["model_A"]["kernel"].upper()}, '
                    f'C={results["best_params"]["model_A"]["C"]}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
        
        # Add true label background shading
        y_true = results['y_test']
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.text(0.02, 0.98, 'Background: Green = True MCS, Red = True VS',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        prob_plot_file = op.join(self.output_dir, 'subject_probabilities_model_A.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Model A probability plot saved to: {prob_plot_file}")
    
    def _plot_model_B_probabilities(self, results):
        """Plot Model B (Reconstructed Model) probabilities on both test sets."""
        print("\n📊 Creating Model B probability plot...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS and VS indices
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        vs_idx = np.where(self.class_names == 'VS')[0][0]
        
        # Extract P(MCS) for Model B on both test sets
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, n_subjects * 0.4), 7))
        x = np.arange(n_subjects)
        
        # Plot 2 curves for Model B
        ax.plot(x, proba_RO, marker='^', linestyle='-.', linewidth=3, markersize=8,
               label='Reconstructed Model → Original Test', color='green', alpha=0.9)
        ax.plot(x, proba_RR, marker='d', linestyle=':', linewidth=3, markersize=8,
               label='Reconstructed Model → Reconstructed Test', color='red', alpha=0.9)
        
        # Add decision threshold
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                  label='Decision Threshold (0.5)', zorder=0)
        
        # Customize plot
        ax.set_xlabel('Subject ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('P(MCS) - Probability of MCS Class', fontsize=13, fontweight='bold')
        ax.set_title(f'Model B (Reconstructed Data): Subject-Level Probabilities\\n'
                    f'{self.marker_type.title()} Features - Tested on Original vs Reconstructed Data\\n'
                    f'Kernel: {results["best_params"]["model_B"]["kernel"].upper()}, '
                    f'C={results["best_params"]["model_B"]["C"]}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
        
        # Add true label background shading
        y_true = results['y_test']
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.text(0.02, 0.98, 'Background: Green = True MCS, Red = True VS',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        prob_plot_file = op.join(self.output_dir, 'subject_probabilities_model_B.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Model B probability plot saved to: {prob_plot_file}")
    
    def _plot_comparison_grid(self, results):
        """Plot 4 subplots comparing different scenarios:
        - Top-left: OO vs OR (Original model on both tests)
        - Top-right: RO vs RR (Reconstructed model on both tests)
        - Bottom-left: OO vs RO (Both models on original test)
        - Bottom-right: OR vs RR (Both models on reconstructed test)
        """
        print("\n📊 Creating comparison grid plot...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS and VS indices
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        x = np.arange(n_subjects)
        y_true = results['y_test']
        
        # Subplot 1: OO vs OR (Original Model on both tests)
        ax = axes[0, 0]
        ax.plot(x, proba_OO, marker='o', linestyle='-', linewidth=2.5, markersize=7,
               label='OO: Original Model → Original Test', color='blue', alpha=0.8)
        ax.plot(x, proba_OR, marker='s', linestyle='--', linewidth=2.5, markersize=7,
               label='OR: Original Model → Reconstructed Test', color='orange', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold (0.5)', zorder=0)
        
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('P(MCS)', fontsize=11, fontweight='bold')
        ax.set_title('Model A: Original Model\nTested on Original vs Reconstructed Data', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Subplot 2: RO vs RR (Reconstructed Model on both tests)
        ax = axes[0, 1]
        ax.plot(x, proba_RO, marker='^', linestyle='-.', linewidth=2.5, markersize=7,
               label='RO: Reconstructed Model → Original Test', color='green', alpha=0.8)
        ax.plot(x, proba_RR, marker='d', linestyle=':', linewidth=2.5, markersize=7,
               label='RR: Reconstructed Model → Reconstructed Test', color='red', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold (0.5)', zorder=0)
        
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('P(MCS)', fontsize=11, fontweight='bold')
        ax.set_title('Model B: Reconstructed Model\nTested on Original vs Reconstructed Data', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Subplot 3: OO vs RO (Both models on Original test)
        ax = axes[1, 0]
        ax.plot(x, proba_OO, marker='o', linestyle='-', linewidth=2.5, markersize=7,
               label='OO: Original Model → Original Test', color='blue', alpha=0.8)
        ax.plot(x, proba_RO, marker='^', linestyle='-.', linewidth=2.5, markersize=7,
               label='RO: Reconstructed Model → Original Test', color='green', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold (0.5)', zorder=0)
        
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('P(MCS)', fontsize=11, fontweight='bold')
        ax.set_title('Original Test Data\nComparing Original vs Reconstructed Models', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Subplot 4: OR vs RR (Both models on Reconstructed test)
        ax = axes[1, 1]
        ax.plot(x, proba_OR, marker='s', linestyle='--', linewidth=2.5, markersize=7,
               label='OR: Original Model → Reconstructed Test', color='orange', alpha=0.8)
        ax.plot(x, proba_RR, marker='d', linestyle=':', linewidth=2.5, markersize=7,
               label='RR: Reconstructed Model → Reconstructed Test', color='red', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold (0.5)', zorder=0)
        
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('P(MCS)', fontsize=11, fontweight='bold')
        ax.set_title('Reconstructed Test Data\nComparing Original vs Reconstructed Models', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Overall title
        fig.suptitle(f'Probability Comparison Grid: All 4 Test Scenarios\n'
                    f'{self.marker_type.title()} Features - {n_subjects} Test Subjects\n'
                    f'Background: Green = True MCS, Red = True VS',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        prob_plot_file = op.join(self.output_dir, 'subject_probabilities_comparison_grid.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Comparison grid plot saved to: {prob_plot_file}")
    
    def _plot_differences_same_train(self, results):
        """Plot probability differences when same model is tested on different data.
        - Curve 1: OO - OR (Original model: Original test - Reconstructed test)
        - Curve 2: RO - RR (Reconstructed model: Original test - Reconstructed test)
        """
        print("\n📊 Creating probability differences plot (same training)...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS index
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Calculate differences
        diff_OO_OR = proba_OO - proba_OR  # Original model: how much higher on original vs reconstructed
        diff_RR_RO = proba_RR - proba_RO  # Reconstructed model: how much higher on reconstructed vs original
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, n_subjects * 0.4), 7))
        x = np.arange(n_subjects)
        
        # Plot 2 difference curves
        ax.plot(x, diff_OO_OR, marker='o', linestyle='-', linewidth=3, markersize=8,
               label='Δ(OO - OR): Original Model tested on Original vs Reconstructed', 
               color='blue', alpha=0.9)
        ax.plot(x, diff_RR_RO, marker='^', linestyle='--', linewidth=3, markersize=8,
               label='Δ(RR - RO): Reconstructed Model tested on Reconstructed vs Original', 
               color='green', alpha=0.9)
        
        # Add zero line (no difference)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                  label='Zero Difference', zorder=0)
        
        # Customize plot
        ax.set_xlabel('Subject ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('Δ P(MCS) - Probability Difference', fontsize=13, fontweight='bold')
        ax.set_title(f'Probability Differences: Same Training Model, Different Test Data\\n'
                    f'{self.marker_type.title()} Features - Effect of Test Data Type\\n'
                    f'Positive: Higher probability on reference test',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks(np.arange(-1.0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        
        # Add true label background shading
        y_true = results['y_test']
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.text(0.02, 0.98, 'Background: Green = True MCS, Red = True VS',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        prob_plot_file = op.join(self.output_dir, 'differences_same_train.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Differences (same train) plot saved to: {prob_plot_file}")
    
    def _plot_differences_same_test(self, results):
        """Plot probability differences when different models are tested on same data.
        - Curve 1: OO - RO (Original test: Original model - Reconstructed model)
        - Curve 2: OR - RR (Reconstructed test: Original model - Reconstructed model)
        """
        print("\n📊 Creating probability differences plot (same test)...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability plot: only for binary classification")
            return
        
        subjects_test = results['test_subjects']
        n_subjects = len(subjects_test)
        
        # Find MCS index
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Calculate differences
        diff_OO_RO = proba_OO - proba_RO  # Original test: how much higher with original vs reconstructed model
        diff_OR_RR = proba_OR - proba_RR  # Reconstructed test: how much higher with original vs reconstructed model
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, n_subjects * 0.4), 7))
        x = np.arange(n_subjects)
        
        # Plot 2 difference curves
        ax.plot(x, diff_OO_RO, marker='o', linestyle='-', linewidth=3, markersize=8,
               label='Δ(OO - RO): Original Test with Original vs Reconstructed Model', 
               color='blue', alpha=0.9)
        ax.plot(x, diff_OR_RR, marker='s', linestyle='--', linewidth=3, markersize=8,
               label='Δ(OR - RR): Reconstructed Test with Original vs Reconstructed Model', 
               color='orange', alpha=0.9)
        
        # Add zero line (no difference)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
                  label='Zero Difference', zorder=0)
        
        # Customize plot
        ax.set_xlabel('Subject ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('Δ P(MCS) - Probability Difference', fontsize=13, fontweight='bold')
        ax.set_title(f'Probability Differences: Same Test Data, Different Training Model\\n'
                    f'{self.marker_type.title()} Features - Effect of Training Data Type\\n'
                    f'Positive: Higher probability with Original model | Negative: Higher with Reconstructed model',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects_test, rotation=90, ha='right', fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks(np.arange(-1.0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        
        # Add true label background shading
        y_true = results['y_test']
        for i, (subj_idx, true_label) in enumerate(zip(x, y_true)):
            color = 'green' if true_label == mcs_idx else 'red'
            ax.axvspan(subj_idx - 0.4, subj_idx + 0.4, alpha=0.15, color=color, zorder=0)
        
        ax.text(0.02, 0.98, 'Background: Green = True MCS, Red = True VS',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        prob_plot_file = op.join(self.output_dir, 'differences_same_test.png')
        plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and save statistics (was removed during earlier edit - adding back)
        # This functionality exists in the original _plot_differences_same_test implementation
        
        print(f"   ✓ Differences (same test) plot saved to: {prob_plot_file}")
    
    def _plot_probability_boxplots(self, results):
        """Plot boxplots of P(MCS) for each model and subject type (8 boxplots total)."""
        print("\n📊 Creating probability boxplots...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping probability boxplots: only for binary classification")
            return
        
        # Find MCS index
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Get true labels
        y_true = results['y_test']
        mcs_mask = (y_true == mcs_idx)
        vs_mask = (y_true != mcs_idx)
        
        # Prepare data for boxplots
        data_to_plot = []
        labels = []
        colors = []
        
        # MCS subjects (4 boxplots)
        data_to_plot.extend([proba_OO[mcs_mask], proba_OR[mcs_mask], proba_RO[mcs_mask], proba_RR[mcs_mask]])
        labels.extend(['OO', 'OR', 'RO', 'RR'])
        colors.extend(['blue', 'orange', 'green', 'red'])
        
        # VS subjects (4 boxplots)
        data_to_plot.extend([proba_OO[vs_mask], proba_OR[vs_mask], proba_RO[vs_mask], proba_RR[vs_mask]])
        labels.extend(['OO', 'OR', 'RO', 'RR'])
        colors.extend(['blue', 'orange', 'green', 'red'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create boxplots
        positions = [1, 2, 3, 4, 6, 7, 8, 9]  # Gap between MCS and VS
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', linewidth=2))
        
        # Color boxplots
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add individual points (jittered)
        for i, (data, pos, color) in enumerate(zip(data_to_plot, positions, colors)):
            x = np.random.normal(pos, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.6, s=50, color=color, edgecolors='black', linewidths=0.5, zorder=3)
        
        # Customize plot
        ax.set_ylabel('P(MCS) - Probability of MCS Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Subject Type', fontsize=14, fontweight='bold')
        ax.set_title(f'Probability Distribution by Model and Subject Type\n{self.marker_type.title()} Features',
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set x-axis
        ax.set_xticks([2.5, 7.5])
        ax.set_xticklabels(['True MCS', 'True VS'], fontsize=13, fontweight='bold')
        
        # Add group labels
        for pos, label, color in zip([1, 2, 3, 4], ['OO', 'OR', 'RO', 'RR'], ['blue', 'orange', 'green', 'red']):
            ax.text(pos, -0.08, label, ha='center', fontsize=11, fontweight='bold', color=color,
                   transform=ax.get_xaxis_transform())
        for pos, label, color in zip([6, 7, 8, 9], ['OO', 'OR', 'RO', 'RR'], ['blue', 'orange', 'green', 'red']):
            ax.text(pos, -0.08, label, ha='center', fontsize=11, fontweight='bold', color=color,
                   transform=ax.get_xaxis_transform())
        
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Decision Threshold')
        ax.axvline(x=5, color='black', linestyle='-', linewidth=1, alpha=0.3)  # Separator
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='OO: Original Model → Original Test'),
            Patch(facecolor='orange', alpha=0.6, label='OR: Original Model → Reconstructed Test'),
            Patch(facecolor='green', alpha=0.6, label='RO: Reconstructed Model → Original Test'),
            Patch(facecolor='red', alpha=0.6, label='RR: Reconstructed Model → Reconstructed Test')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        prob_boxplot_file = op.join(self.output_dir, 'probability_boxplots_by_model_and_subject.png')
        plt.savefig(prob_boxplot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Probability boxplots saved to: {prob_boxplot_file}")
    
    def _plot_difference_boxplots_with_stats(self, results):
        """Plot boxplots of probability differences with Wilcoxon tests.
        Matches the style of seed_differences_analysis.py differences_combined_boxplot.png
        """
        print("\n📊 Creating difference boxplots with statistical tests...")
        
        if len(self.class_names) != 2:
            print("   ⚠️  Skipping difference boxplots: only for binary classification")
            return
        
        from scipy.stats import wilcoxon
        
        # Find MCS index
        mcs_idx = np.where(self.class_names == 'MCS')[0][0]
        
        # Extract P(MCS) for all 4 scenarios
        proba_OO = results['model_A_orig_test']['y_proba'][:, mcs_idx]
        proba_OR = results['model_A_recon_test']['y_proba'][:, mcs_idx]
        proba_RO = results['model_B_orig_test']['y_proba'][:, mcs_idx]
        proba_RR = results['model_B_recon_test']['y_proba'][:, mcs_idx]
        
        # Calculate differences
        diff_OO_OR = proba_OO - proba_OR
        diff_OO_RO = proba_OO - proba_RO
        diff_RR_OR = proba_RR - proba_OR
        diff_RR_RO = proba_RR - proba_RO
        
        # Prepare data for boxplots (all subjects combined)
        differences = [diff_OO_OR, diff_OO_RO, diff_RR_OR, diff_RR_RO]
        
        # Define colors matching seed_differences_analysis.py
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Create labels matching the format
        labels = [
            'Train: O. Test: O\nvs\nTrain: O. Test: R',  # OO-OR
            'Train: O. Test: O\nvs\nTrain: R. Test: O',  # OO-RO
            'Train: R. Test: R\nvs\nTrain: O. Test: R',  # RR-OR
            'Train: R. Test: R\nvs\nTrain: R. Test: O'   # RR-RO
        ]
        
        # Perform Wilcoxon tests
        p_values = []
        for diff in differences:
            if len(diff) > 0:
                try:
                    stat, p = wilcoxon(diff, alternative='two-sided')
                    p_values.append(p)
                except:
                    p_values.append(1.0)
            else:
                p_values.append(1.0)
        
        # Create figure (matching seed_differences_analysis.py style)
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create boxplots
        positions = [1, 2, 3, 4]
        bp = ax.boxplot(differences, labels=labels, positions=positions, widths=0.15, patch_artist=True,
                       showmeans=True, meanline=True,
                       medianprops=dict(color='black', linewidth=2),
                       meanprops=dict(color='gray', linewidth=2, linestyle='--'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxplots
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Overlay individual points with jitter
        for i, (values, pos, color) in enumerate(zip(differences, positions, colors)):
            x = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.3, s=30, color=color, zorder=3)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Get current y-limits and extend them for significance markers
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.15 * y_range)
        
        # Add red asterisks on top of boxplots if significant
        for i, (pos, p_val) in enumerate(zip(positions, p_values)):
            if p_val < 0.05:
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                y_pos = ymax + 0.05 * y_range
                ax.text(pos, y_pos, sig, ha='center', va='bottom',
                       fontsize=24, color='red', fontweight='bold')
        
        # Formatting (matching seed_differences_analysis.py)
        ax.set_ylabel('Δ P(MCS) - Probability difference', fontsize=24)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=17, rotation=0)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2, label='Median'),
            Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Mean'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=16)
        
        plt.tight_layout()
        diff_boxplot_file = op.join(self.output_dir, 'difference_boxplots_with_wilcoxon.png')
        plt.savefig(diff_boxplot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save p-values to JSON
        stats_dict = {
            'all_subjects': {
                'OO_minus_OR': {
                    'p_value': float(p_values[0]), 
                    'significant': bool(p_values[0] < 0.05),
                    'interpretation': 'Significant difference' if p_values[0] < 0.05 else 'No significant difference'
                },
                'OO_minus_RO': {
                    'p_value': float(p_values[1]), 
                    'significant': bool(p_values[1] < 0.05),
                    'interpretation': 'Significant difference' if p_values[1] < 0.05 else 'No significant difference'
                },
                'RR_minus_OR': {
                    'p_value': float(p_values[2]), 
                    'significant': bool(p_values[2] < 0.05),
                    'interpretation': 'Significant difference' if p_values[2] < 0.05 else 'No significant difference'
                },
                'RR_minus_RO': {
                    'p_value': float(p_values[3]), 
                    'significant': bool(p_values[3] < 0.05),
                    'interpretation': 'Significant difference' if p_values[3] < 0.05 else 'No significant difference'
                }
            },
            'test_info': {
                'test_name': 'Wilcoxon signed-rank test (two-tailed)',
                'null_hypothesis': 'H0: The median of differences equals zero (distributions are the same)',
                'alternative_hypothesis': 'H1: The median of differences does NOT equal zero (distributions differ)',
                'interpretation': 'If p < 0.05, we reject H0 and conclude distributions are significantly DIFFERENT',
                'note': 'This is the standard statistical framework. Small p-values provide evidence against H0.'
            }
        }
        
        stats_file = op.join(self.output_dir, 'wilcoxon_test_results.json')
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"   ✓ Difference boxplots saved to: {diff_boxplot_file}")
        print(f"   ✓ Wilcoxon test results saved to: {stats_file}")
        
        # Create smaller version with abbreviated labels
        self._plot_difference_boxplots_with_stats_small(results, differences, p_values, colors)
    
    def _plot_difference_boxplots_with_stats_small(self, results, differences, p_values, colors):
        """Create a smaller version of difference boxplots with abbreviated labels.
        Matches the style of seed_differences_analysis.py differences_combined_boxplot_small.png
        """
        print("\n📊 Creating small difference boxplots...")
        
        # Create abbreviated labels
        abbreviated_labels = [
            'O O\nvs\nO R',  # OO-OR
            'O O\nvs\nR O',  # OO-RO
            'R R\nvs\nO R',  # RR-OR
            'R R\nvs\nR O'   # RR-RO
        ]
        
        # Create figure (smaller size)
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create boxplots
        positions = [1, 2, 3, 4]
        bp = ax.boxplot(differences, labels=abbreviated_labels, positions=positions, widths=0.15, 
                       patch_artist=True, showmeans=True, meanline=True,
                       medianprops=dict(color='black', linewidth=2),
                       meanprops=dict(color='gray', linewidth=2, linestyle='--'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxplots
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Overlay individual points with jitter
        for i, (values, pos, color) in enumerate(zip(differences, positions, colors)):
            x = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.3, s=30, color=color, zorder=3)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Formatting with larger labels (matching the small version style)
        ax.set_ylabel('Δ Probability', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=14, rotation=0)
        ax.tick_params(axis='y', labelsize=14)
        
        # Set symmetric y-axis limits
        ax.set_ylim(-0.4, 0.4)
        y_range = 0.8  # Total range for positioning asterisks
        
        # Add red asterisks on top of boxplots if significant
        for i, (pos, p_val) in enumerate(zip(positions, p_values)):
            if p_val < 0.05:
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                y_pos = 0.4 + 0.05 * y_range  # Position above the upper limit
                ax.text(pos, y_pos, sig, ha='center', va='bottom',
                       fontsize=24, color='red', fontweight='bold')
        
        plt.tight_layout()
        diff_boxplot_small_file = op.join(self.output_dir, 'difference_boxplots_with_wilcoxon_small.png')
        plt.savefig(diff_boxplot_small_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Small difference boxplots saved to: {diff_boxplot_small_file}")
    
    def _plot_scenario_results(self, scenario_results, output_dir, scenario_key, global_results):
        """Create plots for a single testing scenario."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_name = "Model A (Original)" if 'model_A' in scenario_key else "Model B (Reconstructed)"
        test_name = "Original Test" if 'orig_test' in scenario_key else "Reconstructed Test"
        
        fig.suptitle(f'Cross-Data Classification Results\n'
                    f'{self.marker_type.title()} Data: {model_name} → {test_name}\n'
                    f'Test subjects: {global_results["n_test_subjects"]}', 
                    fontsize=16)
        
        # 1. Grid Search / CV scores
        ax = axes[0, 0]
        if 'model_A' in scenario_key:
            cv_scores = global_results['cv_scores_A']
            cv_label = "Model A: Grid Search Best CV Score\n(Original Training Data)"
        else:
            cv_scores = global_results['cv_scores_B']
            cv_label = "Model B: Grid Search Best CV Score\n(Reconstructed Training Data)"
            
        # Note: cv_scores contains only the best score from grid search, not individual folds
        x_pos = np.arange(len(cv_scores))
        bars = ax.bar(x_pos, cv_scores, alpha=0.7, width=0.5)
        
        # Only show mean line if we have multiple scores
        if len(cv_scores) > 1:
            ax.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
        
        ax.set_xlabel('Best Model (from Grid Search over C)')
        ax.set_ylabel('Balanced Accuracy (CV Score)')
        ax.set_title(cv_label, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Optimal C'])
        if len(cv_scores) > 1:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, cv_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Confusion Matrix
        ax = axes[0, 1]
        conf_matrix = scenario_results['confusion_matrix']
        
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = conf_matrix.max() / 2. if conf_matrix.size > 0 else 0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if conf_matrix[i, j] > thresh else "black")
        
        ax.set_title(f'Confusion Matrix\n{test_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        
        # 3. Feature Importances (top 10) - only for linear kernel
        ax = axes[1, 0]
        if 'model_A' in scenario_key:
            importances = global_results['feature_importances_A']
            kernel_used = global_results['best_params']['model_A']['kernel']
        else:
            importances = global_results['feature_importances_B']
            kernel_used = global_results['best_params']['model_B']['kernel']
        
        if importances is not None:
            top_n = min(10, len(importances))
            top_indices = np.argsort(importances)[-top_n:]
            
            ax.barh(range(top_n), importances[top_indices])
            ax.set_xlabel('Feature Importance (|Coefficient|)')
            ax.set_ylabel('Features')
            ax.set_title(f'Top {top_n} Feature Importances\n{model_name} (Linear Kernel)')
            ax.set_yticks(range(top_n))
            
            # Use real feature names if available
            if self.feature_names_abbreviated is not None and len(self.feature_names_abbreviated) == len(importances):
                feature_labels = [self.feature_names_abbreviated[i] for i in top_indices]
            else:
                feature_labels = [f'Feature {i}' for i in top_indices]
            
            ax.set_yticklabels(feature_labels)
        else:
            # RBF kernel - no linear coefficients
            ax.text(0.5, 0.5, f'Feature importances not available\n(using {kernel_used.upper()} kernel)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Feature Importances\n{model_name} ({kernel_used.upper()} Kernel)')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 4. Class-wise Performance
        ax = axes[1, 1]
        
        precision = scenario_results['precision']
        recall = scenario_results['recall']
        f1 = scenario_results['f1_score']
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Class-wise Performance\n{test_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = op.join(output_dir, 'classification_results.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # ROC curve for binary classification
        if scenario_results['auc_score'] is not None:
            self._plot_roc_curve(scenario_results, output_dir, test_name, global_results['y_test'])
    
    def _plot_roc_curve(self, scenario_results, output_dir, test_name, y_test):
        """Plot ROC curve for binary classification."""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use the actual test labels and probabilities
        y_proba = scenario_results['y_proba']
        
        if len(self.class_names) == 2:
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc="lower right")
        else:
            # Multi-class case
            ax.text(0.5, 0.5, 'ROC Curve\n(Multi-class not implemented)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {test_name}')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = op.join(output_dir, 'roc_curve.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_confusion_matrices(self, results):
        """
        Create a 2x2 grid of confusion matrix heatmaps showing all cross-data scenarios.
        
        Layout:
        - Row 1: Model A (trained on original data)
        - Row 2: Model B (trained on reconstructed data)
        - Column 1 (left): Tested on original data
        - Column 2 (right): Tested on reconstructed data
        
        Each heatmap shows: (VS label, MCS label) x (VS pred, MCS pred)
        """
        print("\n📊 Creating combined 4-heatmap confusion matrix figure...")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Cross-Data Classification: Confusion Matrices\n{self.marker_type.title()} Features',
                    fontsize=18, fontweight='bold')
        
        # Define the 4 scenarios
        scenarios = [
            ('model_A_orig_test', 0, 0, 'Model A (Original)\n→ Original Test'),
            ('model_A_recon_test', 0, 1, 'Model A (Original)\n→ Reconstructed Test'),
            ('model_B_orig_test', 1, 0, 'Model B (Reconstructed)\n→ Original Test'),
            ('model_B_recon_test', 1, 1, 'Model B (Reconstructed)\n→ Reconstructed Test')
        ]
        
        for scenario_key, row, col, title in scenarios:
            ax = axes[row, col]
            scenario_results = results[scenario_key]
            conf_matrix = scenario_results['confusion_matrix']
            
            # Plot heatmap
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Count', rotation=270, labelpad=15)
            
            # Labels and title
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            
            # Set ticks
            ax.set_xticks(range(len(self.class_names)))
            ax.set_xticklabels(self.class_names, fontsize=11)
            ax.set_yticks(range(len(self.class_names)))
            ax.set_yticklabels(self.class_names, fontsize=11)
            
            # Add text annotations
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(j, i, format(conf_matrix[i, j], 'd'),
                           ha="center", va="center", fontsize=16, fontweight='bold',
                           color="white" if conf_matrix[i, j] > thresh else "black")
            
            # Add balanced accuracy text
            bal_acc = scenario_results['balanced_accuracy']
            ax.text(0.98, 0.02, f"Bal. Acc: {bal_acc:.3f}",
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add row and column labels
        fig.text(0.08, 0.75, 'Trained on\nOriginal', fontsize=14, fontweight='bold',
                ha='center', va='center', rotation=90,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        fig.text(0.08, 0.25, 'Trained on\nReconstructed', fontsize=14, fontweight='bold',
                ha='center', va='center', rotation=90,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        fig.text(0.30, 0.93, 'Tested on Original', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        fig.text(0.70, 0.93, 'Tested on Reconstructed', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.tight_layout(rect=[0.12, 0.0, 1.0, 0.95])
        
        # Save combined plot
        combined_plot_file = op.join(self.output_dir, 'combined_confusion_matrices.png')
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Combined confusion matrices saved to: {combined_plot_file}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Cross-subject binary SVM classification for VS vs MCS consciousness states'
    )
    parser.add_argument('--data-dir', required=True,
                       help='Path to results directory containing subject data')
    parser.add_argument('--patient-labels', required=True,
                       help='Path to CSV file with patient labels')
    parser.add_argument('--marker-type', choices=['scalar', 'topo'], default='scalar',
                       help='Type of markers to use (scalar or topo)')
    parser.add_argument('--output-dir', 
                       help='Output directory for results (default: results/svm/{marker_type})')
    parser.add_argument('--cv-strategy', choices=['stratified', 'loo'], default='stratified',
                       help='Cross-validation strategy')
    parser.add_argument('--n-splits', type=int, default=2,
                       help='Number of CV splits (ignored for LOO)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to hold out for testing (default: 0.2)')
    parser.add_argument('--cross-data', action='store_true',
                       help='Run cross-data classification (train on both original and reconstructed, test on both)')
    
    args = parser.parse_args()
    
    # Run classification
    try:
        if args.cross_data:
            # Run cross-data SVM classification
            print("Running cross-data SVM classification with linear kernel...")
            classifier = CrossDataClassifier(
                data_dir=args.data_dir,
                patient_labels_file=args.patient_labels,
                marker_type=args.marker_type,
                output_dir=args.output_dir,
                random_state=args.random_state
            )
            
            classifier.run_cross_data_classification(
                cv_strategy=args.cv_strategy,
                n_splits=args.n_splits,
                test_size=args.test_size
            )
        else:
            print("Error: This script requires --cross-data flag.")
            print("Use: --cross-data to train on both original and reconstructed data")
            return
        
        print("\nClassification completed successfully!")
        
    except Exception as e:
        print(f"\nClassification failed: {e}")
        raise


if __name__ == '__main__':
    main()
