"""Cross-subject binary ExtraTrees classification for TOTEM data.

==================================================
Cross-subject Binary ExtraTrees classification
==================================================

This script trains ExtraTrees classifiers across subjects for binary 
classification of consciousness states from EEG markers: VS (Vegetative State) 
vs MCS (Minimally Conscious State). The classification is performed
at the subject level, where each subject contributes one sample.

Key features:
- Binary classification: VS vs MCS (UWS → VS, MCS+/MCS- → MCS)
- Cross-subject classification (no data leakage)
- Support for scalar or topographic markers
- Support for original or reconstructed data
- State prediction from patient_labels_with_controls.csv
- Comprehensive evaluation with proper cross-validation

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

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import (cross_val_score, 
                                   StratifiedKFold, LeaveOneOut, train_test_split)
from sklearn.ensemble import ExtraTreesClassifier
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
    """Cross-subject binary ExtraTrees classifier for VS vs MCS state prediction."""
    
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
        self.output_dir = output_dir or f"results/extratrees/{marker_type}_{data_origin}"
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
    
    def load_feature_names(self):
        """Load feature names from the first available subject's scalar_metrics.csv file.
        
        Returns
        -------
        list
            List of feature names, abbreviated for display
        """
        print(" Loading feature names from scalar_metrics.csv...")
        
        # Find first available subject directory
        if not op.exists(self.data_dir):
            print(f"   Data directory not found: {self.data_dir}")
            return None
        
        subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        
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
                
                # Try to find scalar_metrics.csv
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
        
        print("    No scalar_metrics.csv files found in data directory")
        # Return generic feature names if no metrics file found
        if self.marker_type == 'scalar':
            print("   Generating generic scalar feature names...")
            generic_names = [f"marker_{i:02d}" for i in range(1, 29)]  # Assume 28 markers
            return generic_names, generic_names
        return None, None
    
    def load_subject_data(self, subject_session_path):
        """Load marker data for a single subject/session.
        
        Parameters
        ----------
        subject_session_path : str
            Path to subject/session directory
            
        Returns
        -------
        array or None
            Marker data or None if loading failed
        """
        
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
            print(f"    Missing {filename} in {subject_session_path}")
            return None
        
        try:
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
            subject_id = subject_dir.replace('sub-', '')
            subject_path = op.join(self.data_dir, subject_dir)
            
            if not op.isdir(subject_path):
                continue
            
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
                    print(f"   Skipping {subject_session_key}: failed to load data")
                    subjects_skipped += 1
                    continue
                
                # Store data
                subject_data.append(marker_data)
                subject_labels.append(labels_dict[subject_session_key])
                collected_subjects.append(subject_session_key)
                subjects_processed += 1
                
                print(f"   ✓ Loaded {subject_session_key}: {marker_data.shape} features, state={labels_dict[subject_session_key]}")
        
        print("\n DATA COLLECTION SUMMARY:")
        print(f"    Successfully loaded: {subjects_processed} subject/sessions")
        print(f"    Skipped: {subjects_skipped} subject/sessions")
        
        if subjects_processed == 0:
            raise ValueError("No subjects could be loaded!")
        
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
        pipeline = make_pipeline(
            RobustScaler(),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_features=1,
                criterion='entropy',
                max_depth=max_depth,
                random_state=self.random_state,
                class_weight=class_weight_dict, #WARNING: In the paper this is 'balanced'
                n_jobs=-1
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
        
        # First, split into train and test sets to avoid data leakage
        print(f"    Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        X_train, X_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
            self.X, self.y_encoded, self.subjects, 
            test_size=test_size, 
            random_state=self.random_state, 
        #    stratify=self.y_encoded --> not stratified with all dataset (data leakage), it is done just in the cv splits
        )
        
        print(f"   ✓ Train set: {X_train.shape[0]} subjects")
        print(f"   ✓ Test set: {X_test.shape[0]} subjects")
        
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
        n_train_samples = X_train.shape[0]
        
        if cv_strategy == 'loo':
            cv = LeaveOneOut()
            print(f"   Using Leave-One-Out CV on training set ({n_train_samples} folds)")
        elif cv_strategy == 'stratified':
            # Check minimum class size for stratified CV on training set
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            min_class_size = np.min(class_counts)
            
            # For stratified CV, we need at least 2 samples per class for n_splits=2
            max_possible_splits = min_class_size
            effective_n_splits = min(n_splits, n_train_samples, max_possible_splits)
            
            if max_possible_splits < 1:
                print("   WARNING: Some classes have only 1 sample in training, using Leave-One-Out instead")
                cv = LeaveOneOut()
                print(f"   Using Leave-One-Out CV on training set ({n_train_samples} folds)")
            elif effective_n_splits < n_splits:
                print(f"   WARNING:  Reducing CV splits from {n_splits} to {effective_n_splits} (limited by min training class size: {min_class_size})")
                cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=self.random_state)
                print(f"   Using Stratified {effective_n_splits}-fold CV on training set")
            else:
                cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=self.random_state)
                print(f"   Using Stratified {effective_n_splits}-fold CV on training set")
        else:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")
        
        # Cross-validation scores on training set only
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=cv, scoring='balanced_accuracy', n_jobs=-1
        )
        
        # Also compute AUC for binary classification
        cv_auc_scores = []
        if len(self.class_names) == 2:
            cv_auc_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
        
        # Fit final model on training data only
        print(f"   Training final model on {X_train.shape[0]} training subjects...")
        pipeline.fit(X_train, y_train)
        
        # Predictions on test set (no data leakage)
        print(f"   Evaluating on {X_test.shape[0]} held-out test subjects...")
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)
        
        # Also get predictions on training set for completeness
        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)
        
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
        
        # Feature importances
        feature_importances = pipeline.named_steps['extratreesclassifier'].feature_importances_
        
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
    """Cross-data ExtraTrees classifier that trains on both original and reconstructed data."""
    
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
            Random state for reproducibility
        """
        self.data_dir = data_dir
        self.patient_labels_file = patient_labels_file
        self.marker_type = marker_type
        self.output_dir = output_dir or f"results/extratrees/{marker_type}"
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
    
    def load_feature_names(self):
        """Load feature names from the first available subject's scalar_metrics.csv file."""
        print(" Loading feature names from scalar_metrics.csv...")
        
        # Find first available subject directory
        if not op.exists(self.data_dir):
            print(f"   Data directory not found: {self.data_dir}")
            return None
        
        subject_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        
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
                
                # Try to find scalar_metrics.csv
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
        
        print("    No scalar_metrics.csv files found in data directory")
        # Return generic feature names if no metrics file found
        if self.marker_type == 'scalar':
            print("   Generating generic scalar feature names...")
            generic_names = [f"marker_{i:02d}" for i in range(1, 29)]  # Assume 28 markers
            return generic_names, generic_names
        return None, None
    
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
        
        # Determine filename based on marker type
        if self.marker_type == 'scalar':
            filename_orig = "scalars_original.npy"
            filename_recon = "scalars_reconstructed.npy"
            features_dir = "features_variable"
        elif self.marker_type == 'topo':
            filename_orig = "topos_original.npy"
            filename_recon = "topos_reconstructed.npy"
            features_dir = "features_variable"
        else:
            raise ValueError(f"Unknown marker_type: {self.marker_type}")
        
        filepath_orig = op.join(subject_session_path, features_dir, filename_orig)
        filepath_recon = op.join(subject_session_path, features_dir, filename_recon)
        
        # Check if both files exist
        if not op.exists(filepath_orig):
            print(f"    Missing {filename_orig} in {subject_session_path}")
            return None, None
        
        if not op.exists(filepath_recon):
            print(f"    Missing {filename_recon} in {subject_session_path}")
            return None, None
        
        try:
            # Load original data
            data_orig = np.load(filepath_orig)
            data_recon = np.load(filepath_recon)
            
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
            subject_id = subject_dir.replace('sub-', '')
            subject_path = op.join(self.data_dir, subject_dir)
            
            if not op.isdir(subject_path):
                continue
            
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
                    print(f"   Skipping {subject_session_key}: failed to load both data types")
                    subjects_skipped += 1
                    continue
                
                # Store data
                subject_data_orig.append(marker_data_orig)
                subject_data_recon.append(marker_data_recon)
                subject_labels.append(labels_dict[subject_session_key])
                collected_subjects.append(subject_session_key)
                subjects_processed += 1
                
                print(f"   ✓ Loaded {subject_session_key}: orig={marker_data_orig.shape}, recon={marker_data_recon.shape}, state={labels_dict[subject_session_key]}")
        
        print("\n DATA COLLECTION SUMMARY:")
        print(f"    Successfully loaded: {subjects_processed} subject/sessions")
        print(f"    Skipped: {subjects_skipped} subject/sessions")
        
        if subjects_processed == 0:
            raise ValueError("No subjects could be loaded!")
        
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
        pipeline = make_pipeline(
            RobustScaler(),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_features=1,
                criterion='entropy',
                max_depth=max_depth,
                random_state=self.random_state,
                class_weight=class_weight_dict,
                n_jobs=-1
            )
        )
        
        return pipeline
    
    def run_cross_data_classification(self, n_estimators=500, max_depth=4, cv_strategy='stratified', n_splits=5, test_size=0.2):
        """
        Run cross-data classification:
        - Train Model A on original data
        - Train Model B on reconstructed data  
        - Test both models on both test sets (original and reconstructed, same subjects)
        
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
            Cross-data classification results
        """
        
        print("=" * 80)
        print("CROSS-DATA EXTRATREES CLASSIFICATION")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Labels file: {self.patient_labels_file}")
        print(f"Marker type: {self.marker_type}")
        print(f"Output directory: {self.output_dir}")
        print(f"CV strategy: {cv_strategy}")
        print(f"Trees: {n_estimators}, Max depth: {max_depth}")
        print(f"Test size: {test_size:.0%}")
        print()
        
        try:
            # Step 1: Collect data
            X_orig, X_recon, y_encoded, subjects = self.collect_data()
            
            # Step 2: Split into train/test (same subjects for both data types)
            print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
            
            (X_orig_train, X_orig_test, X_recon_train, X_recon_test, 
             y_train, y_test, subjects_train, subjects_test) = train_test_split(
                X_orig, X_recon, y_encoded, subjects, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=y_encoded
            )
            
            print(f"   ✓ Train set: {X_orig_train.shape[0]} subjects")
            print(f"   ✓ Test set: {X_orig_test.shape[0]} subjects")
            
            # Step 3: Train models
            print("\n🔧 Training models...")
            
            # Model A: trained on original data
            print("   Training Model A (original data)...")
            pipeline_A = self.create_pipeline(n_estimators, max_depth)
            pipeline_A.fit(X_orig_train, y_train)
            
            # Model B: trained on reconstructed data
            print("   Training Model B (reconstructed data)...")
            pipeline_B = self.create_pipeline(n_estimators, max_depth)
            pipeline_B.fit(X_recon_train, y_train)
            
            # Step 4: Cross-validation on training sets
            print("\n📊 Running cross-validation...")
            
            # CV for Model A on original training data
            if cv_strategy == 'loo':
                cv = LeaveOneOut()
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            
            cv_scores_A = cross_val_score(pipeline_A, X_orig_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            cv_scores_B = cross_val_score(pipeline_B, X_recon_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            
            print(f"   Model A CV: {np.mean(cv_scores_A):.3f} ± {np.std(cv_scores_A):.3f}")
            print(f"   Model B CV: {np.mean(cv_scores_B):.3f} ± {np.std(cv_scores_B):.3f}")
            
            # Step 5: Cross-testing - test both models on both test sets
            print("\n🔄 Cross-testing models...")
            
            # Model A predictions
            print("   Model A predictions...")
            y_pred_A_on_orig = pipeline_A.predict(X_orig_test)  # Model A on original test
            y_pred_A_on_recon = pipeline_A.predict(X_recon_test)  # Model A on reconstructed test
            y_proba_A_on_orig = pipeline_A.predict_proba(X_orig_test)
            y_proba_A_on_recon = pipeline_A.predict_proba(X_recon_test)
            
            # Model B predictions
            print("   Model B predictions...")
            y_pred_B_on_orig = pipeline_B.predict(X_orig_test)  # Model B on original test
            y_pred_B_on_recon = pipeline_B.predict(X_recon_test)  # Model B on reconstructed test
            y_proba_B_on_orig = pipeline_B.predict_proba(X_orig_test)
            y_proba_B_on_recon = pipeline_B.predict_proba(X_recon_test)
            
            # Step 6: Compute metrics for all combinations
            print("\n📈 Computing metrics...")
            
            results = {
                'model_A_orig_test': self._compute_metrics(y_test, y_pred_A_on_orig, y_proba_A_on_orig, "Model A on Original Test"),
                'model_A_recon_test': self._compute_metrics(y_test, y_pred_A_on_recon, y_proba_A_on_recon, "Model A on Reconstructed Test"),
                'model_B_orig_test': self._compute_metrics(y_test, y_pred_B_on_orig, y_proba_B_on_orig, "Model B on Original Test"),
                'model_B_recon_test': self._compute_metrics(y_test, y_pred_B_on_recon, y_proba_B_on_recon, "Model B on Reconstructed Test"),
                'cv_scores_A': cv_scores_A,
                'cv_scores_B': cv_scores_B,
                'test_subjects': subjects_test,
                'train_subjects': subjects_train,
                'class_names': self.class_names,
                'feature_importances_A': pipeline_A.named_steps['extratreesclassifier'].feature_importances_,
                'feature_importances_B': pipeline_B.named_steps['extratreesclassifier'].feature_importances_,
                'y_test': y_test,
                'n_features': X_orig_train.shape[1],
                'n_train_subjects': len(subjects_train),
                'n_test_subjects': len(subjects_test)
            }
            
            # Step 7: Save results and plots for all combinations
            self._save_cross_data_results(results, pipeline_A, pipeline_B)
            
            # Summary
            print("\n" + "=" * 80)
            print("CROSS-DATA CLASSIFICATION SUMMARY")
            print("=" * 80)
            print(f"Dataset: {results['n_train_subjects']} train + {results['n_test_subjects']} test subjects, {results['n_features']} features")
            print(f"Classes: {', '.join(results['class_names'])}")
            print(f"Model A (original) CV: {np.mean(cv_scores_A):.3f} ± {np.std(cv_scores_A):.3f}")
            print(f"Model B (reconstructed) CV: {np.mean(cv_scores_B):.3f} ± {np.std(cv_scores_B):.3f}")
            print("\nCross-testing results:")
            print(f"  Model A → Original test: {results['model_A_orig_test']['balanced_accuracy']:.3f}")
            print(f"  Model A → Reconstructed test: {results['model_A_recon_test']['balanced_accuracy']:.3f}")
            print(f"  Model B → Original test: {results['model_B_orig_test']['balanced_accuracy']:.3f}")
            print(f"  Model B → Reconstructed test: {results['model_B_recon_test']['balanced_accuracy']:.3f}")
            print(f"Results saved to: {self.output_dir}")
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
                    'n_features': results['n_features'],
                    'class_names': results['class_names'].tolist(),
                    'timestamp': datetime.now().isoformat()
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
            
            # Save feature importances
            if 'model_A' in scenario_key:
                importances = results['feature_importances_A']
            else:
                importances = results['feature_importances_B']
                
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
            
            # Create plots for this scenario
            self._plot_scenario_results(scenario_results, scenario_dir, scenario_key, results)
            
            print(f"   ✓ {scenario_key} results saved to: {scenario_dir}")
        
        # After saving all scenario results, create the combined 4-heatmap figure
        self._plot_combined_confusion_matrices(results)
    
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
        
        # 1. CV scores comparison (if available)
        ax = axes[0, 0]
        if 'model_A' in scenario_key:
            cv_scores = global_results['cv_scores_A']
            cv_label = "Model A CV (Original Training)"
        else:
            cv_scores = global_results['cv_scores_B']
            cv_label = "Model B CV (Reconstructed Training)"
            
        x_pos = np.arange(len(cv_scores))
        bars = ax.bar(x_pos, cv_scores, alpha=0.7)
        ax.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(cv_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, cv_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
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
        
        # 3. Feature Importances (top 10)
        ax = axes[1, 0]
        if 'model_A' in scenario_key:
            importances = global_results['feature_importances_A']
        else:
            importances = global_results['feature_importances_B']
            
        top_n = min(10, len(importances))
        top_indices = np.argsort(importances)[-top_n:]
        
        ax.barh(range(top_n), importances[top_indices])
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Features')
        ax.set_title(f'Top {top_n} Feature Importances\n{model_name}')
        ax.set_yticks(range(top_n))
        
        # Use real feature names if available
        if self.feature_names_abbreviated is not None and len(self.feature_names_abbreviated) == len(importances):
            feature_labels = [self.feature_names_abbreviated[i] for i in top_indices]
        else:
            feature_labels = [f'Feature {i}' for i in top_indices]
        
        ax.set_yticklabels(feature_labels)
        
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
        description='Cross-subject binary ExtraTrees classification for VS vs MCS consciousness states'
    )
    parser.add_argument('--data-dir', required=True,
                       help='Path to results directory containing subject data')
    parser.add_argument('--patient-labels', required=True,
                       help='Path to CSV file with patient labels')
    parser.add_argument('--marker-type', choices=['scalar', 'topo'], default='scalar',
                       help='Type of markers to use (scalar or topo)')
    parser.add_argument('--data-origin', choices=['original', 'reconstructed'], default='original',
                       help='Source of data (original or reconstructed)')
    parser.add_argument('--output-dir', 
                       help='Output directory for results (default: results/extratrees/{marker_type}_{data_origin})')
    parser.add_argument('--n-estimators', type=int, default=500,
                       help='Number of trees in the forest')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum depth of trees')
    parser.add_argument('--cv-strategy', choices=['stratified', 'loo'], default='stratified',
                       help='Cross-validation strategy')
    parser.add_argument('--n-splits', type=int, default=4,
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
            # Use new CrossDataClassifier for cross-data classification
            print("Running cross-data classification...")
            classifier = CrossDataClassifier(
                data_dir=args.data_dir,
                patient_labels_file=args.patient_labels,
                marker_type=args.marker_type,
                output_dir=args.output_dir,
                random_state=args.random_state
            )
            
            classifier.run_cross_data_classification(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                cv_strategy=args.cv_strategy,
                n_splits=args.n_splits,
                test_size=args.test_size
            )
        else:
            # Use original CrossSubjectClassifier for single-data classification
            print("Running single-data classification...")
            classifier = CrossSubjectClassifier(
                data_dir=args.data_dir,
                patient_labels_file=args.patient_labels,
                marker_type=args.marker_type,
                data_origin=args.data_origin,
                output_dir=args.output_dir,
                random_state=args.random_state
            )
            
            classifier.run_classification(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                cv_strategy=args.cv_strategy,
                n_splits=args.n_splits,
                test_size=args.test_size
            )
        
        print("\nClassification completed successfully!")
        
    except Exception as e:
        print(f"\nClassification failed: {e}")
        raise


if __name__ == '__main__':
    main()
