#!/usr/bin/env python3
"""
ICM Predictions Computation Module

This module handles ONLY the computation/prediction logic.
NO visualization code should be in this file.

Computes predictions and saves results to pkl files for visualization.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
from .reductions import (
    reduce_to_scalar,
    get_lg_reduction_config,
)

logger = logging.getLogger(__name__)

# Pretrained models directory (project root, not in src/)
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "trained_models"

class ICMPredictor:
    """
    ICM Prediction class that replicates NICE framework functionality.
    
    This class handles both univariate and multivariate predictions for consciousness
    classification (VS/UWS vs MCS) using the same markers and methods as NICE.
    """
    
    def __init__(self, config_params=None, models_dir=None):
        """
        Initialize the ICM predictor.
        
        Parameters
        ----------
        config_params : dict, optional
            Configuration parameters for prediction. If None, uses default NICE parameters.
        models_dir : str or Path, optional
            Directory containing pretrained models. If None, uses default location.
        """
        if config_params is None:
            config_params = {}
            
        # Default configuration matching NICE framework
        self.ml_classes = config_params.get('ml_classes', ['VS/UWS', 'MCS'])
        self.target = config_params.get('target', 'Label')
        self.reductions = config_params.get('reductions', [
            'icm/lg/egi256/trim_mean80', 'icm/lg/egi256/std',
            'icm/lg/egi256gfp/trim_mean80', 'icm/lg/egi256gfp/std'
        ])
        self.clf_types = config_params.get('clf_type', ['gssvm', 'et-reduced'])
        
        # Set models directory
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        
        # Load pretrained models (will raise error if not found)
        self.classifiers, self.feature_cols, self.training_metadata = self._load_pretrained_models()
        
        # Marker groups for univariate analysis (matching NICE structure)
        self.marker_groups = {
            'Information Theory': [
                'PermutationEntropy/default',
                'KolmogorovComplexity/default'
            ],
            'Spectral': [
                'PowerSpectralDensity/alpha',
                'PowerSpectralDensity/alphan',
                'PowerSpectralDensity/beta',
                'PowerSpectralDensity/betan',
                'PowerSpectralDensity/delta',
                'PowerSpectralDensity/deltan',
                'PowerSpectralDensity/theta',
                'PowerSpectralDensity/thetan',
                'PowerSpectralDensity/gamma',
                'PowerSpectralDensity/gamman',
                'PowerSpectralDensity/summary_se',
                'PowerSpectralDensitySummary/summary_msf',
                'PowerSpectralDensitySummary/summary_sef90',
                'PowerSpectralDensitySummary/summary_sef95',
            ],
            'Connectivity': [
                'SymbolicMutualInformation/weighted'
            ],
            'ERPs': [
                'ContingentNegativeVariation/default',
                'WindowDecoding/local',
                'WindowDecoding/global',
                'TimeLockedContrast/mmn',
                'TimeLockedContrast/p3a',
                'TimeLockedContrast/p3b',
                'TimeLockedTopography/p1',
                'TimeLockedTopography/p3a',
                'TimeLockedTopography/p3b',
                'TimeLockedContrast/LD-LS',
                'TimeLockedContrast/GD-GS',
                'TimeLockedContrast/LSGD-LDGS',
                'TimeLockedContrast/LSGS-LDGD',
            ]
        }
        
    def _load_pretrained_models(self):
        """
        Load pretrained models and metadata from disk.
        
        Returns
        -------
        classifiers : dict
            Dictionary of {clf_name: trained_model}
        feature_cols : list
            List of feature column names
        training_metadata : dict
            Training metadata
            
        Raises
        ------
        FileNotFoundError
            If models directory or required files are not found
        """
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}\n"
                f"Please ensure trained models are available."
            )
        
        logger.info(f"Loading pretrained models from {self.models_dir}")
        
        # Load feature columns
        features_path = self.models_dir / "feature_columns.joblib"
        if not features_path.exists():
            raise FileNotFoundError(f"Feature columns not found: {features_path}")
        feature_cols = joblib.load(features_path)
        logger.info(f"Loaded {len(feature_cols)} feature columns")
        
        # Load training metadata
        metadata_path = self.models_dir / "training_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
        training_metadata = joblib.load(metadata_path)
        logger.info(f"Loaded training metadata: {training_metadata['n_samples']} samples, {training_metadata['n_features']} features")
        
        # Load classifiers (fail if any requested classifier is missing)
        classifiers = {}
        for clf_name in self.clf_types:
            model_path = self.models_dir / f"{clf_name}_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Required classifier model not found: {model_path}\n"
                    f"Expected classifiers: {self.clf_types}"
                )
            
            clf = joblib.load(model_path)
            classifiers[clf_name] = clf
            logger.info(f"Loaded {clf_name} model")
            
            # Load and log metadata
            clf_metadata_path = self.models_dir / f"{clf_name}_metadata.joblib"
            if clf_metadata_path.exists():
                clf_metadata = joblib.load(clf_metadata_path)
                logger.info(f"  CV AUC: {clf_metadata['cv_score_mean']:.3f} ± {clf_metadata['cv_score_std']:.3f}")
                logger.info(f"  In-sample accuracy: {clf_metadata['in_sample_score']:.3f}")
        
        if not classifiers:
            raise ValueError(f"No classifiers loaded from {self.models_dir}")
        
        # Load univariate models (optional - for better univariate predictions)
        univariate_path = self.models_dir / "univariate_models.joblib"
        if univariate_path.exists():
            self.univariate_models = joblib.load(univariate_path)
            logger.info(f"Loaded {len(self.univariate_models)} univariate models")
        else:
            self.univariate_models = None
            logger.warning("Univariate models not found - will use simplified approach")
        
        return classifiers, feature_cols, training_metadata
    
    def predict(self, markers_data, summary=None):
        """
        Main prediction function using pretrained models.
        
        Parameters
        ----------
        markers_data : dict
            Dictionary containing marker data in NICE format
        summary : dict, optional
            Not used (kept for API compatibility)
            
        Returns
        -------
        dict
            Dictionary containing prediction results with 'multivariate' and 'univariate' keys
        """
        logger.info("Starting ICM predictions with pretrained models...")
        
        # Extract features from subject's data using NICE reductions
        subject_features_df, subject_feature_names = self._extract_features_with_reductions(markers_data)
        
        if subject_features_df.empty:
            logger.warning("No features extracted from subject data")
            return self._create_empty_prediction_result()
        
        logger.info(f"Extracted {len(subject_feature_names)} features from subject data")
        
        # Prepare subject data (align features with pretrained model's expected features)
        X_subject = self._align_subject_features(subject_features_df, self.feature_cols, subject_feature_names)
        
        if X_subject is None:
            logger.error("Failed to align subject features with model features")
            return self._create_empty_prediction_result()
        
        logger.info(f"Aligned subject features: {X_subject.shape}")
        
        # Perform multivariate predictions with pretrained models
        multivariate_results = self._multivariate_prediction_pretrained(X_subject)
        
        # Perform univariate predictions
        if self.univariate_models is not None:
            univariate_results = self._univariate_prediction_pretrained(X_subject, self.feature_cols)
        else:
            univariate_results = self._univariate_prediction_simple(X_subject, self.feature_cols)
        
        # Combine results
        prediction_results = {
            'multivariate': multivariate_results,
            'univariate': univariate_results,
            'summary': {
                'n_features': len(self.feature_cols),
                'n_samples_train': self.training_metadata['n_samples'],
                'n_samples_subject': X_subject.shape[0],
                'classes': self.ml_classes,
                'target': self.target
            }
        }
        
        logger.info("ICM predictions completed successfully")
        return prediction_results
    
    def _extract_features_with_reductions(self, markers_data):
        """
        Extract features from HDF5 data using NICE reduction pipeline.
        
        Applies all 4 reductions to each marker to match training data format.
        
        Parameters
        ----------
        markers_data : dict
            Dictionary containing HDF5 marker data
            
        Returns
        -------
        pd.DataFrame
            Feature matrix (1 row x N features)
        list
            Feature column names matching training CSV format
        """
        features = {}  # Dict to store computed features
        
        # Get epochs info if available (needed for reshaping)
        n_epochs = None
        n_channels = None
        
        if 'epoch_info' in markers_data:
            epochs = markers_data['epoch_info']
            n_epochs = len(epochs)
            n_channels = len(epochs.ch_names)
            logger.info(f"Found epochs info: {n_epochs} epochs × {n_channels} channels")
        else:
            # Try to infer from data dimensions
            logger.warning("No epoch_info found, will try to infer dimensions from data")
            n_epochs = 1193  # Default for current dataset
            n_channels = 256
        
        # Define all 4 reduction types
        reduction_types = [
            'icm/lg/egi256/trim_mean80',
            'icm/lg/egi256/std',
            'icm/lg/egi256gfp/trim_mean80',
            'icm/lg/egi256gfp/std',
        ]
        
        # PowerSpectralDensity - Absolute power
        # CRITICAL: Training data has PSD in dB scale (10*log10), so we must convert
        if 'per_channel_spectral' in markers_data:
            spectral_data = markers_data['per_channel_spectral']
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                if band in spectral_data:
                    raw_data = np.array(spectral_data[band])
                    
                    # Check if data is in linear scale (all positive, typical range >0.001)
                    # If so, convert to dB scale to match training data
                    is_linear_scale = np.all(raw_data > 0) and np.mean(raw_data) > 0.001
                    if is_linear_scale:
                        logger.info(f"PSD/{band}: Converting from linear to dB scale (10*log10)")
                        band_data_db = 10 * np.log10(raw_data)
                        logger.info(f"  Before: min={np.min(raw_data):.6f}, max={np.max(raw_data):.6f}")
                        logger.info(f"  After:  min={np.min(band_data_db):.3f}, max={np.max(band_data_db):.3f}")
                    else:
                        logger.info(f"PSD/{band}: Already in dB scale (has negative values or very small values)")
                        band_data_db = raw_data
                    
                    for reduction_type in reduction_types:
                        try:
                            scalar_value = reduce_to_scalar(
                                band_data_db, n_epochs, n_channels,
                                marker_type='PowerSpectralDensity',
                                reduction_type=reduction_type,
                                is_connectivity=False
                            )
                            key = f"nice/marker/PowerSpectralDensity/{band}_{reduction_type}"
                            features[key] = scalar_value
                            logger.info(f"  {key} = {scalar_value:.6f}")
                        except Exception as e:
                            logger.debug(f"Failed PSD/{band}/{reduction_type}: {e}")
        
        # PowerSpectralDensity - Normalized power (deltan, thetan, etc.)
        if 'normalized_spectral' in markers_data:
            normalized_data = markers_data['normalized_spectral']
            for band_norm in ['deltan', 'thetan', 'alphan', 'betan', 'gamman']:
                if band_norm in normalized_data:
                    for reduction_type in reduction_types:
                        try:
                            scalar_value = reduce_to_scalar(
                                normalized_data[band_norm], n_epochs, n_channels,
                                marker_type='PowerSpectralDensity',
                                reduction_type=reduction_type,
                                is_connectivity=False
                            )
                            key = f"nice/marker/PowerSpectralDensity/{band_norm}_{reduction_type}"
                            features[key] = scalar_value
                        except Exception as e:
                            logger.debug(f"Failed PSD/{band_norm}/{reduction_type}: {e}")
        
        # PowerSpectralDensity/summary_se - Spectral Entropy
        if 'spectral_entropy' in markers_data:
            for reduction_type in reduction_types:
                try:
                    scalar_value = reduce_to_scalar(
                        markers_data['spectral_entropy'], n_epochs, n_channels,
                        marker_type='PowerSpectralDensity',
                        reduction_type=reduction_type,
                        is_connectivity=False
                    )
                    key = f"nice/marker/PowerSpectralDensity/summary_se_{reduction_type}"
                    features[key] = scalar_value
                except Exception as e:
                    logger.debug(f"Failed spectral entropy/{reduction_type}: {e}")
        
        # PowerSpectralDensitySummary (MSF, SEF90, SEF95)
        for summary_type in ['msf', 'sef90', 'sef95']:
            key = f"{summary_type}_per_channel"
            if key in markers_data:
                for reduction_type in reduction_types:
                    try:
                        scalar_value = reduce_to_scalar(
                            markers_data[key], n_epochs, n_channels,
                            marker_type='PowerSpectralDensitySummary',
                            reduction_type=reduction_type,
                            is_connectivity=False
                        )
                        fkey = f"nice/marker/PowerSpectralDensitySummary/summary_{summary_type}_{reduction_type}"
                        features[fkey] = scalar_value
                    except Exception as e:
                        logger.debug(f"Failed {summary_type}/{reduction_type}: {e}")
        
        # PermutationEntropy - Use ONLY theta band (tau=8)
        if 'pe_theta' in markers_data:
            for reduction_type in reduction_types:
                try:
                    scalar_value = reduce_to_scalar(
                        markers_data['pe_theta'], n_epochs, n_channels,
                        marker_type='PermutationEntropy',
                        reduction_type=reduction_type,
                        is_connectivity=False
                    )
                    fkey = f"nice/marker/PermutationEntropy/default_{reduction_type}"
                    features[fkey] = scalar_value
                except Exception as e:
                    logger.debug(f"Failed PE/theta/{reduction_type}: {e}")
        
        # SymbolicMutualInformation - Use ONLY theta band
        if 'wsmi_theta' in markers_data:
            for reduction_type in reduction_types:
                try:
                    scalar_value = reduce_to_scalar(
                        markers_data['wsmi_theta'], n_epochs, n_channels,
                        marker_type='SymbolicMutualInformation',
                        reduction_type=reduction_type,
                        is_connectivity=True
                    )
                    fkey = f"nice/marker/SymbolicMutualInformation/weighted_{reduction_type}"
                    features[fkey] = scalar_value
                except Exception as e:
                    logger.debug(f"Failed WSMI/theta/{reduction_type}: {e}")
        
        # KolmogorovComplexity
        if 'kolmogorov_complexity' in markers_data:
            for reduction_type in reduction_types:
                try:
                    scalar_value = reduce_to_scalar(
                        markers_data['kolmogorov_complexity'], n_epochs, n_channels,
                        marker_type='KolmogorovComplexity',
                        reduction_type=reduction_type,
                        is_connectivity=False
                    )
                    fkey = f"nice/marker/KolmogorovComplexity/default_{reduction_type}"
                    features[fkey] = scalar_value
                except Exception as e:
                    logger.debug(f"Failed Kolmogorov/{reduction_type}: {e}")
        
        # ContingentNegativeVariation
        if 'cnv_data' in markers_data and 'cnv_slopes_trials' in markers_data['cnv_data']:
            cnv_slopes = markers_data['cnv_data']['cnv_slopes_trials']
            for reduction_type in reduction_types:
                try:
                    config = get_lg_reduction_config(reduction_type, 'ContingentNegativeVariation')
                    channels_fun = config['channels_fun']
                    epochs_fun = config['epochs_fun']
                    cnv_roi = config['picks'].get('channels', None)
                    
                    data_filtered = cnv_slopes[:, cnv_roi] if cnv_roi is not None else cnv_slopes
                    data_per_epoch = channels_fun(data_filtered, axis=1)
                    scalar_value = float(epochs_fun(data_per_epoch, axis=0))
                    
                    fkey = f"nice/marker/ContingentNegativeVariation/default_{reduction_type}"
                    features[fkey] = scalar_value
                except Exception as e:
                    logger.debug(f"Failed CNV/{reduction_type}: {e}")
        
        # TimeLockedTopography (p1, p3a, p3b)
        # Junifer stores as: (1, epochs, channels, times) or (epochs, channels, times)
        # No ROI specified in config → uses all 256 channels
        # Average across time → apply reduce_to_scalar
        for topo_type in ['p1', 'p3a', 'p3b']:
            topo_key = f"{topo_type}_topography_timelockedtopo"
            if topo_key in markers_data:
                # Convert to numpy array (junifer data comes in correct shape)
                topo_data = np.array(markers_data[topo_key])
                
                # Squeeze leading dimension if present: (1, epochs, channels, times) -> (epochs, channels, times)
                if topo_data.ndim == 4 and topo_data.shape[0] == 1:
                    topo_data = topo_data.squeeze(0)
                
                # TimeLockedTopography uses no ROI → 256 channels
                n_channels_topo = 256
                
                if topo_data.ndim == 3 and topo_data.shape[1] == n_channels_topo:
                    # Data is (epochs, channels, times) - average across time
                    topo_time_avg = np.mean(topo_data, axis=2)  # -> (epochs, channels)
                    
                    # Apply standard reduction pipeline for each reduction type
                    for reduction_type in reduction_types:
                        try:
                            scalar_value = reduce_to_scalar(
                                topo_time_avg, n_epochs, n_channels_topo,
                                marker_type='TimeLockedTopography',
                                reduction_type=reduction_type,
                                is_connectivity=False
                            )
                            
                            fkey = f"nice/marker/TimeLockedTopography/{topo_type}_{reduction_type}"
                            features[fkey] = scalar_value
                        except Exception as e:
                            logger.warning(f"Failed TimeLockedTopography/{topo_type}/{reduction_type}: {e}")
                else:
                    logger.warning(f"Unexpected shape for {topo_key}: {topo_data.shape}, expected (epochs, {n_channels_topo}, times)")
        
        # TimeLockedContrast (LSGS-LDGD, LSGD-LDGS, LD-LS, mmn, p3a, GD-GS, p3b)
        # Junifer stores as: (1, combined_epochs, channels, times) or (combined_epochs, channels, times)
        # Note: Junifer concatenates epochs from BOTH conditions (A and B) and stores ALL 256 channels
        # ROI filtering (scalp/mmn/p3a/p3b) is applied during reduce_to_scalar, not at storage
        # Average across time → apply reduce_to_scalar with ROI filtering
        
        # Get epoch counts per condition for contrasts
        epochs_obj = markers_data['epoch_info']
        epoch_counts = {}
        for event_id, event_code in epochs_obj.event_id.items():
            epoch_counts[event_id] = sum(epochs_obj.events[:, 2] == event_code)
        
        contrast_mapping = {
            'timelockedcontrast_lsgs_ldgd_timelockedcontrast': ('LSGS-LDGD', epoch_counts.get('LSGS', 0) + epoch_counts.get('LDGD', 0)),
            'timelockedcontrast_lsgd_ldgs_timelockedcontrast': ('LSGD-LDGS', epoch_counts.get('LSGD', 0) + epoch_counts.get('LDGS', 0)),
            'timelockedcontrast_ld_ls_timelockedcontrast': ('LD-LS', epoch_counts.get('LDGS', 0) + epoch_counts.get('LDGD', 0) + epoch_counts.get('LSGS', 0) + epoch_counts.get('LSGD', 0)),
            'timelockedcontrast_mmn_timelockedcontrast': ('mmn', epoch_counts.get('LDGS', 0) + epoch_counts.get('LDGD', 0) + epoch_counts.get('LSGS', 0) + epoch_counts.get('LSGD', 0)),
            'timelockedcontrast_p3a_timelockedcontrast': ('p3a', epoch_counts.get('LDGS', 0) + epoch_counts.get('LDGD', 0) + epoch_counts.get('LSGS', 0) + epoch_counts.get('LSGD', 0)),
            'timelockedcontrast_gd_gs_timelockedcontrast': ('GD-GS', epoch_counts.get('LSGD', 0) + epoch_counts.get('LDGD', 0) + epoch_counts.get('LSGS', 0) + epoch_counts.get('LDGS', 0)),
            'timelockedcontrast_p3b_timelockedcontrast': ('p3b', epoch_counts.get('LSGD', 0) + epoch_counts.get('LDGD', 0) + epoch_counts.get('LSGS', 0) + epoch_counts.get('LDGS', 0)),
        }
        
        for contrast_key, (contrast_name, n_epochs_contrast) in contrast_mapping.items():
            if contrast_key in markers_data:
                # Convert to numpy array (junifer data comes in correct shape)
                contrast_data = np.array(markers_data[contrast_key])
                
                # Squeeze leading dimension if present: (1, epochs, channels, times) -> (epochs, channels, times)
                if contrast_data.ndim == 4 and contrast_data.shape[0] == 1:
                    contrast_data = contrast_data.squeeze(0)
                
                if contrast_data.ndim == 3 and contrast_data.shape[1] == n_channels:
                    # Data is (combined_epochs, channels, times) - average across time
                    contrast_time_avg = np.mean(contrast_data, axis=2)  # -> (combined_epochs, channels)
                    
                    # Apply standard reduction pipeline for each reduction type
                    # reduce_to_scalar will apply ROI filtering based on marker_type
                    for reduction_type in reduction_types:
                        try:
                            marker_type = f'TimeLockedContrast/{contrast_name}' if contrast_name in ['mmn', 'p3a', 'p3b'] else 'TimeLockedContrast'
                            
                            scalar_value = reduce_to_scalar(
                                contrast_time_avg, n_epochs_contrast, n_channels,
                                marker_type=marker_type,
                                reduction_type=reduction_type,
                                is_connectivity=False
                            )
                            
                            fkey = f"nice/marker/TimeLockedContrast/{contrast_name}_{reduction_type}"
                            features[fkey] = scalar_value
                        except Exception as e:
                            logger.warning(f"Failed TimeLockedContrast/{contrast_name}/{reduction_type}: {e}")
                else:
                    logger.warning(f"Unexpected shape for {contrast_key}: {contrast_data.shape}, expected (epochs, {n_channels}, times)")
        
        # WindowDecoding (local, global)
        # Already a scalar (cross-validated accuracy) - use same value for all 4 reductions
        for wd_type in ['local', 'global']:
            wd_key = f"window_decoding_{wd_type}"
            if wd_key in markers_data:
                wd_data = markers_data[wd_key]
                try:
                    if isinstance(wd_data, (list, np.ndarray)):
                        scalar_value = float(np.array(wd_data).flatten()[0])
                    else:
                        scalar_value = float(wd_data)
                    
                    # WindowDecoding is already a single accuracy value - use for all 4 reductions
                    for reduction_type in reduction_types:
                        fkey = f"nice/marker/WindowDecoding/{wd_type}_{reduction_type}"
                        features[fkey] = scalar_value
                except Exception as e:
                    logger.warning(f"Failed to extract WindowDecoding/{wd_type}: {e}")
        
        # Convert to DataFrame
        if features:
            feature_df = pd.DataFrame([features])
            feature_names = list(features.keys())
            logger.info(f"Extracted {len(feature_names)} features using NICE reductions")
            return feature_df, feature_names
        else:
            logger.warning("No features extracted with NICE reductions")
            return pd.DataFrame(), []
    
    def _align_subject_features(self, subject_features_df, training_feature_cols, subject_feature_names):
        """
        Align subject features with training feature columns.
        
        Parameters
        ----------
        subject_features_df : pd.DataFrame
            Subject feature DataFrame (1 row)
        training_feature_cols : list
            List of training feature column names
        subject_feature_names : list
            List of subject feature names
            
        Returns
        -------
        X_subject : np.ndarray
            Aligned feature vector (1 x n_features)
        """
        # For single subject, we have feature values but need to match training format
        # Training format: marker_name_reduction (e.g., "nice/marker/PermutationEntropy/default_icm/lg/egi256/trim_mean80")
        # Subject format: same as training (marker_name_reduction)
        
        # Create mapping from subject features to indices for fast lookup
        subject_feature_map = {name: idx for idx, name in enumerate(subject_feature_names)}
        
        # Create aligned feature vector
        X_subject = np.zeros((1, len(training_feature_cols)))
        
        matched_features = 0
        unmatched_features = []
        
        for train_idx, train_col in enumerate(training_feature_cols):
            # Try exact match first
            if train_col in subject_feature_map:
                subj_idx = subject_feature_map[train_col]
                X_subject[0, train_idx] = subject_features_df.iloc[0, subj_idx]
                matched_features += 1
            else:
                unmatched_features.append(train_col)
        
        if matched_features < len(training_feature_cols) * 0.5:
            logger.warning(f"Only matched {matched_features}/{len(training_feature_cols)} features between subject and training data")
            if len(unmatched_features) <= 10:
                logger.warning(f"Unmatched training features: {unmatched_features}")
            else:
                logger.warning(f"Unmatched training features (first 10): {unmatched_features[:10]}")
            # Log first few subject features for debugging
            logger.warning(f"Subject feature examples (first 5): {subject_feature_names[:5]}")
        else:
            logger.info(f"Matched {matched_features}/{len(training_feature_cols)} features")
        
        return X_subject
    
    def _multivariate_prediction_pretrained(self, X_subject):
        """
        Perform multivariate prediction using pretrained models.
        
        Parameters
        ----------
        X_subject : np.ndarray
            Subject feature vector (1 × n_features)
            
        Returns
        -------
        dict
            Multivariate prediction results with feature importance
        """
        multivariate_results = {}
        
        logger.info("Predicting with pretrained models...")
        
        for clf_name, clf in self.classifiers.items():
            logger.info(f"  Using {clf_name}...")
            
            try:
                # Predict on subject using pretrained model
                probas = clf.predict_proba(X_subject)
                
                # Load saved metadata
                clf_metadata_path = self.models_dir / f"{clf_name}_metadata.joblib"
                clf_metadata = joblib.load(clf_metadata_path) if clf_metadata_path.exists() else {}
                
                # Extract feature importance
                feature_importance = self._extract_feature_importance(clf, clf_name)
                
                multivariate_results[clf_name] = {
                    self.ml_classes[0]: float(probas[0, 0]),
                    self.ml_classes[1]: float(probas[0, 1]),
                    'cv_score_mean': clf_metadata.get('cv_score_mean', None),
                    'in_sample_score': clf_metadata.get('in_sample_score', None),
                    'feature_importance': feature_importance
                }
                
                logger.info(f"    Predicted: {self.ml_classes[0]}={probas[0,0]:.3f}, {self.ml_classes[1]}={probas[0,1]:.3f}")
                if feature_importance is not None:
                    logger.info(f"    Extracted feature importance for {len(feature_importance)} features")
                
            except Exception as e:
                logger.error(f"Failed to predict with {clf_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return multivariate_results
    
    def _extract_feature_importance(self, clf, clf_name):
        """
        Extract feature importance from trained classifier.
        
        Parameters
        ----------
        clf : sklearn Pipeline
            Trained classifier pipeline
        clf_name : str
            Classifier name
            
        Returns
        -------
        dict or None
            Dictionary mapping feature indices to importance scores
        """
        try:
            # Get the actual classifier from the pipeline
            if hasattr(clf, 'named_steps'):
                classifier = clf.named_steps.get('classifier', None)
                if classifier is None:
                    return None
            else:
                classifier = clf
            
            # For SVC with GridSearchCV, get the best estimator
            if hasattr(classifier, 'best_estimator_'):
                classifier = classifier.best_estimator_
            
            # Extract importance based on classifier type
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models (ExtraTreesClassifier)
                # Get feature selector if present
                if hasattr(clf, 'named_steps') and 'selector' in clf.named_steps:
                    selector = clf.named_steps['selector']
                    selected_features = selector.get_support(indices=True)
                    importance = np.zeros(len(self.feature_cols))
                    importance[selected_features] = classifier.feature_importances_
                else:
                    importance = classifier.feature_importances_
                
                # Create dictionary with feature names
                feature_importance = {
                    self.feature_cols[i]: float(importance[i])
                    for i in range(len(importance))
                }
                return feature_importance
                
            elif hasattr(classifier, 'coef_'):
                # Linear models (SVC with linear kernel)
                # Get feature selector if present
                if hasattr(clf, 'named_steps') and 'selector' in clf.named_steps:
                    selector = clf.named_steps['selector']
                    selected_features = selector.get_support(indices=True)
                    importance = np.zeros(len(self.feature_cols))
                    importance[selected_features] = np.abs(classifier.coef_[0])
                else:
                    importance = np.abs(classifier.coef_[0])
                
                # Create dictionary with feature names
                feature_importance = {
                    self.feature_cols[i]: float(importance[i])
                    for i in range(len(importance))
                }
                return feature_importance
            
            else:
                logger.debug(f"No feature importance available for {clf_name}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract feature importance for {clf_name}: {e}")
            return None
    
    def _univariate_prediction_pretrained(self, X_subject, feature_cols):
        """
        Univariate prediction using pretrained logistic regression models.
        
        This matches the NICE framework approach - each feature gets its own trained model.
        
        Parameters
        ----------
        X_subject : np.ndarray
            Subject feature vector (1 × n_features)
        feature_cols : list
            List of feature column names
            
        Returns
        -------
        pd.DataFrame
            Univariate prediction results per marker
        """
        logger.info("Performing univariate analysis with pretrained models...")
        
        univariate_results = []
        
        for idx, feature_col in enumerate(feature_cols):
            try:
                # Get feature value
                feature_value = X_subject[0, idx]
                
                # Skip if feature is missing
                if np.isnan(feature_value):
                    logger.debug(f"Skipping {feature_col}: NaN value")
                    continue
                
                # Get pretrained univariate model for this feature
                if feature_col not in self.univariate_models:
                    logger.debug(f"No univariate model for {feature_col}")
                    continue
                
                clf = self.univariate_models[feature_col]
                
                # Predict probabilities using trained model
                X_feature = np.array([[feature_value]])
                probas = clf.predict_proba(X_feature)
                score = clf.score(X_feature, [1])  # Dummy score, just for structure
                
                # Parse marker and reduction from column name
                # Format: nice/marker/MarkerName/subtype_icm/lg/egi256/reduction
                # We want to extract: Marker=nice/marker/MarkerName/subtype, Reduction=icm/lg/egi256/reduction
                parts = feature_col.split('_')
                if len(parts) >= 2:
                    # Find the last underscore that separates marker from reduction
                    # The reduction part starts with "icm/"
                    for i in range(len(parts) - 1, -1, -1):
                        if parts[i].startswith('icm/') or (i > 0 and parts[i-1].endswith('/lg') or parts[i-1].endswith('/rs')):
                            marker = '_'.join(parts[:i])
                            reduction = '_'.join(parts[i:])
                            break
                    else:
                        marker = '_'.join(parts[:-1])
                        reduction = parts[-1]
                else:
                    marker = feature_col
                    reduction = 'unknown'
                
                univariate_results.append({
                    'Marker': marker,
                    'Reduction': reduction,
                    'Score': float(score),
                    self.ml_classes[0]: float(probas[0, 0]),
                    self.ml_classes[1]: float(probas[0, 1]),
                    'Feature_Value': float(feature_value)
                })
                
            except Exception as e:
                logger.debug(f"Skipped univariate analysis for {feature_col}: {e}")
                continue
        
        if univariate_results:
            univariate_df = pd.DataFrame(univariate_results)
            logger.info(f"Generated univariate predictions for {len(univariate_df)} markers")
        else:
            univariate_df = pd.DataFrame()
            logger.warning("No univariate predictions generated")
        
        return univariate_df
    
    def _univariate_prediction_simple(self, X_subject, feature_cols):
        """
        Simplified univariate prediction (fallback if no pretrained models).
        
        Parameters
        ----------
        X_subject : np.ndarray
            Subject feature vector (1 × n_features)
        feature_cols : list
            List of feature column names
            
        Returns
        -------
        pd.DataFrame
            Univariate prediction results per marker
        """
        logger.warning("Using simplified univariate analysis (no pretrained models)")
        logger.warning("Results may not be accurate - retrain models with univariate support")
        
        univariate_results = []
        
        for idx, feature_col in enumerate(feature_cols):
            try:
                # Extract single feature value
                feature_value = X_subject[0, idx]
                
                # Skip if feature is missing
                if np.isnan(feature_value):
                    continue
                
                # Parse marker and reduction
                parts = feature_col.split('_')
                if len(parts) >= 2:
                    for i in range(len(parts) - 1, -1, -1):
                        if parts[i].startswith('icm/') or (i > 0 and parts[i-1].endswith('/lg') or parts[i-1].endswith('/rs')):
                            marker = '_'.join(parts[:i])
                            reduction = '_'.join(parts[i:])
                            break
                    else:
                        marker = '_'.join(parts[:-1])
                        reduction = parts[-1]
                else:
                    marker = feature_col
                    reduction = 'unknown'
                
                # Dummy prediction - not meaningful without trained models
                univariate_results.append({
                    'Marker': marker,
                    'Reduction': reduction,
                    'Score': 0.5,
                    self.ml_classes[0]: 0.5,
                    self.ml_classes[1]: 0.5,
                    'Feature_Value': float(feature_value)
                })
                
            except Exception as e:
                logger.debug(f"Skipped univariate analysis for {feature_col}: {e}")
                continue
        
        if univariate_results:
            univariate_df = pd.DataFrame(univariate_results)
        else:
            univariate_df = pd.DataFrame()
        
        return univariate_df
    
    def _create_empty_prediction_result(self):
        """Create empty prediction result structure."""
        return {
            'multivariate': {},
            'univariate': pd.DataFrame(),
            'summary': {
                'n_features': 0,
                'n_samples': 0,
                'classes': self.ml_classes,
                'target': self.target
            }
        }


def predict_icm_lg(markers_data, summary=None, config_params=None):
    """
    Main prediction function for ICM Local-Global paradigm.
    
    This function replicates the NICE _predict_lg functionality.
    
    Parameters
    ----------
    markers_data : dict
        Dictionary containing marker data from HDF5/pickle files
    summary : dict, optional
        Summary data containing labels and metadata
    config_params : dict, optional
        Configuration parameters for prediction
        
    Returns
    -------
    dict
        Dictionary containing prediction results
    """
    if config_params is None:
        config_params = {}
        
    # Set LG-specific defaults
    if 'ml_classes' not in config_params:
        config_params['ml_classes'] = ['VS/UWS', 'MCS']
        
    if 'target' not in config_params:
        config_params['target'] = 'Label'
        
    if 'reductions' not in config_params:
        config_params['reductions'] = [
            'icm/lg/egi256/trim_mean80', 'icm/lg/egi256/std',
            'icm/lg/egi256gfp/trim_mean80', 'icm/lg/egi256gfp/std'
        ]
        
    if 'clf_type' not in config_params:
        config_params['clf_type'] = ['gssvm', 'et-reduced']
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}
    elif 'clf_select' not in config_params:
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}
    
    # Handle MCS+/MCS- mapping (from NICE framework)
    if summary is not None and isinstance(summary, dict):
        if config_params['target'] in summary:
            target_data = summary[config_params['target']]
            if hasattr(target_data, 'replace'):
                summary[config_params['target']] = target_data.replace(
                    {'MCS+': 'MCS', 'MCS-': 'MCS'}
                )
    
    # Create predictor and run predictions
    predictor = ICMPredictor(config_params)
    results = predictor.predict(markers_data, summary)
    
    logger.info("ICM LG predictions completed")
    return results


def predict_icm_rs(markers_data, summary=None, config_params=None):
    """
    Main prediction function for ICM Resting State paradigm.
    
    This function replicates the NICE _predict_rs functionality.
    
    Parameters
    ----------
    markers_data : dict
        Dictionary containing marker data from HDF5/pickle files
    summary : dict, optional
        Summary data containing labels and metadata
    config_params : dict, optional
        Configuration parameters for prediction
        
    Returns
    -------
    dict
        Dictionary containing prediction results
    """
    if config_params is None:
        config_params = {}
        
    # Set RS-specific defaults
    if 'ml_classes' not in config_params:
        config_params['ml_classes'] = ['VS/UWS', 'MCS']
        
    if 'target' not in config_params:
        config_params['target'] = 'Diagnosis'  # Different from LG
        
    if 'reductions' not in config_params:
        config_params['reductions'] = [
            'icm/rs/egi256/trim_mean80', 'icm/rs/egi256/std',
            'icm/rs/egi256gfp/trim_mean80', 'icm/rs/egi256gfp/std'
        ]
        
    if 'clf_type' not in config_params:
        config_params['clf_type'] = ['gssvm', 'et-reduced']
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}
    elif 'clf_select' not in config_params:
        config_params['clf_select'] = {'gssvm': 20., 'et-reduced': None}
    
    # Handle MCS+/MCS- mapping (from NICE framework)
    if summary is not None and isinstance(summary, dict):
        if config_params['target'] in summary:
            target_data = summary[config_params['target']]
            if hasattr(target_data, 'replace'):
                summary[config_params['target']] = target_data.replace(
                    {'MCS+': 'MCS', 'MCS-': 'MCS'}
                )
    
    # Create predictor and run predictions
    predictor = ICMPredictor(config_params)
    results = predictor.predict(markers_data, summary)
    
    logger.info("ICM RS predictions completed")
    return results


# Convenience function for backward compatibility
def predict(markers_data, summary=None, config_params=None, paradigm='lg'):
    """
    General prediction function that can handle both LG and RS paradigms.
    
    Parameters
    ----------
    markers_data : dict
        Dictionary containing marker data
    summary : dict, optional
        Summary data containing labels and metadata
    config_params : dict, optional
        Configuration parameters for prediction
    paradigm : str, default='lg'
        Paradigm type ('lg' for Local-Global, 'rs' for Resting State)
        
    Returns
    -------
    dict
        Dictionary containing prediction results
    """
    if paradigm.lower() == 'lg':
        return predict_icm_lg(markers_data, summary, config_params)
    elif paradigm.lower() == 'rs':
        return predict_icm_rs(markers_data, summary, config_params)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}. Use 'lg' or 'rs'.")


# load_training_summary removed - no longer needed with pretrained models


def compute_prediction_data(report_data, task='lg', output_dir="./tmp_computed_data"):
    """
    Compute predictions and save results to pkl file (NO PLOTTING).
    
    Parameters
    ----------
    report_data : dict
        Report data dictionary containing marker data
    output_dir : str or Path
        Directory to save computed prediction results
    """
    logger.info(f"Computing prediction data for {task.upper()} paradigm...")
    
    # Run predictions using the ICM predictor with pretrained models
    # Errors will propagate and break the pipeline if models/data are missing
    if task == 'lg':
        prediction_results = predict_icm_lg(
            markers_data=report_data,
            summary=None,  # Not needed with pretrained models
            config_params={
                'ml_classes': ['VS/UWS', 'MCS'],
                'target': 'Label',  # Column name in training CSV
                'clf_type': ['gssvm', 'et-reduced']
            }
        )
    else:  # rs
        prediction_results = predict_icm_rs(
            markers_data=report_data,
            summary=None,  # Not needed with pretrained models
            config_params={
                'ml_classes': ['VS/UWS', 'MCS'],
                'target': 'Label',  # Column name in training CSV
                'clf_type': ['gssvm', 'et-reduced']
            }
        )
    
    # Add task information to results
    if prediction_results:
        prediction_results['summary']['task'] = task.upper()
    
    if not prediction_results:
        raise ValueError("No prediction results generated - prediction failed")
    
    # Save prediction results to pkl
    output_path = Path(output_dir) / "prediction_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(prediction_results, f)
    logger.info(f"✅ Prediction results saved to {output_path}")
