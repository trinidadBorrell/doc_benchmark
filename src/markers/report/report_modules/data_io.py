"""
Consolidated Data I/O for Next-ICM Reports

All data loading, reading, and adaptation in one place:
- HDF5 feature reading (from Junifer)
- MNE preprocessing data reading
- Data transformation and alignment
- Channel alignment utilities

This consolidates what was previously split across 4 files:
- junifer_hdf5_reader_final.py
- dumped_data_reader.py
- data_loaders.py
- data_adapters.py
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import mne
import numpy as np

try:
    from junifer.storage import HDF5FeatureStorage
except ImportError:
    print("Warning: junifer not found. HDF5 reading may not work.")

logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: HDF5 Feature Reading (from Junifer)
# =============================================================================

class HDF5Reader:
    """Read Junifer HDF5 feature storage files.
    
    Uses junifer's built-in HDF5FeatureStorage to read features.
    """

    def __init__(self, hdf5_path: Union[str, Path]):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        self.storage = HDF5FeatureStorage(str(self.hdf5_path))
        self._features_cache = None

    def _get_features(self) -> Dict[str, Dict[str, Any]]:
        """Get cached feature list."""
        if self._features_cache is None:
            self._features_cache = self.storage.list_features()
        return self._features_cache

    def list_features(self) -> List[str]:
        """List all available feature names."""
        features = self._get_features()
        return [meta["name"] for meta in features.values()]

    def read_feature(self, feature_name: str) -> Dict[str, Any]:
        """Read a feature from the HDF5 file."""
        try:
            return self.storage.read(feature_name=feature_name)
        except Exception as e:
            available = self.list_features()
            raise ValueError(
                f"Failed to read feature '{feature_name}': {e}. "
                f"Available features: {available}"
            )


# =============================================================================
# Part 2: MNE Preprocessing Data Reading
# =============================================================================

class PreprocessingReader:
    """Read MNE preprocessing data from dumped files."""

    def __init__(self, dump_base_path: str = "./input"):
        self.dump_path = Path(dump_base_path)

    def load_preprocessing_data(self) -> Dict[str, Any]:
        """Load preprocessing data from dumped files.
        
        Loads cleaned epochs and preprocessing metadata.
        Original epochs are not needed since all info is in the metadata.
        """
        cleaned_epochs_file = self.dump_path / "03_artifact_rejected_eeg.fif"
        bad_channels_metadata_file = self.dump_path / "03_bad_channels_metadata.pkl"

        if not cleaned_epochs_file.exists():
            raise FileNotFoundError(f"Cleaned epochs file not found: {cleaned_epochs_file}")
        if not bad_channels_metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {bad_channels_metadata_file}")

        # Load cleaned epochs (for all analysis)
        cleaned_epochs = mne.read_epochs(str(cleaned_epochs_file), preload=True)
        
        # Load metadata containing preprocessing info
        with open(bad_channels_metadata_file, "rb") as f:
            metadata = pickle.load(f)

        # Extract preprocessing_info from pkl
        preprocessing_info = metadata.get("preprocessing_info", {})
        bad_channels = preprocessing_info.get("bad_channels_detected", [])
        bad_epochs_indices = preprocessing_info.get("bad_epochs_detected", [])
        
        # Calculate epoch counts
        n_epochs_after = len(cleaned_epochs)
        n_bad_epochs = len(bad_epochs_indices) if bad_epochs_indices is not None and len(bad_epochs_indices) > 0 else 0
        n_epochs_before = n_epochs_after + n_bad_epochs
        
        logger.info(f"Loaded cleaned epochs: {n_epochs_after}")
        logger.info(f"Bad epochs from metadata: {n_bad_epochs}")
        logger.info(f"Total epochs before rejection: {n_epochs_before}")
        logger.info(f"Bad channels detected: {len(bad_channels)} - {bad_channels}")

        return {
            "epochs": cleaned_epochs,           # For all analysis (main data)
            "bad_channels": bad_channels,
            "bad_epochs": bad_epochs_indices,
            "n_epochs_before_rejection": n_epochs_before,
            "n_epochs_after_rejection": n_epochs_after,
            "preprocessing_info": preprocessing_info,  # Full preprocessing metadata from pkl
        }


# =============================================================================
# Part 3: Data Adapter (for nice_ext compatibility)
# =============================================================================

class MarkerDataAdapter:
    """Adapter to make data compatible with nice_ext plotting functions.
    
    This wraps data to provide the interface that nice_ext expects.
    """

    def __init__(self, data, name, ch_info=None, intercepts=None):
        self.data_ = np.asarray(data, dtype=np.float64)
        self.name_ = name
        self.ch_info_ = ch_info
        # For CNV markers that have both slopes and intercepts
        self.intercepts_ = np.asarray(intercepts, dtype=np.float64) if intercepts is not None else None

    def _get_title(self):
        """Get marker title for nice_ext plotting."""
        return self.name_

    def reduce_to_topo(self, reduction_func=None, picks=None):
        """Reduce data to topographical representation."""
        if reduction_func is None:
            reduced_data = self.data_.copy()
        else:
            # If data is 2D (trials x channels), reduce across trials (axis 0)
            if self.data_.ndim == 2:
                reduced_data = reduction_func(self.data_, axis=0)
            else:
                reduced_data = reduction_func(self.data_)

        if reduced_data is None:
            reduced_data = self.data_.copy()

        # Ensure at least 1D array (not scalar)
        reduced_data = np.atleast_1d(reduced_data)
        
        # Flatten if more than 1D
        if reduced_data.ndim > 1:
            reduced_data = reduced_data.flatten()

        # Apply picks
        if picks is not None:
            if isinstance(picks, slice):
                reduced_data = reduced_data[picks]
            elif hasattr(picks, "__iter__"):
                reduced_data = reduced_data[picks]

        # Handle both full montage and good-channels-only data
        if hasattr(self.ch_info_, "ch_names") and hasattr(self.ch_info_, "get"):
            all_channels = self.ch_info_.ch_names
            bad_channels = self.ch_info_.get("bads", [])

            # If data already aligned to full montage with NaNs, return as-is
            if len(reduced_data) == len(all_channels) and np.any(np.isnan(reduced_data)):
                return reduced_data

            # Otherwise, filter to good channels only
            good_indices = [
                i for i, ch in enumerate(all_channels) if ch not in bad_channels
            ]
            if len(reduced_data) == len(good_indices):
                # Data is good-channels-only, expand to full montage
                full_data = np.full(len(all_channels), np.nan)
                full_data[good_indices] = reduced_data
                return full_data

        return reduced_data


def align_data_to_eeg_montage(
    data, mne_info, fill_value=np.nan, bad_channels=None
):
    """Align data to EEG montage, handling channel mismatches and bad channels.
    
    Parameters
    ----------
    data : array-like
        Data to align (can be dict or array)
    mne_info : mne.Info
        MNE Info object with channel information
    fill_value : float
        Value to use for missing/bad channels
    bad_channels : list, optional
        List of bad channel names
    
    Returns
    -------
    aligned_data : np.ndarray
        Data aligned to EEG montage
    eeg_info : mne.Info
        Info object for EEG channels only
    """
    if bad_channels is None:
        bad_channels = []

    # Extract EEG channels from mne_info
    ch_names = mne_info["ch_names"]
    eeg_channel_indices = [
        i for i, ch in enumerate(ch_names)
        if ch.startswith("E") and ch[1:].isdigit()
    ]
    eeg_info = mne.pick_info(mne_info, eeg_channel_indices, copy=True)
    n_eeg_channels = len(eeg_info["ch_names"])
    
    # Set bad channels in info
    bad_eeg_channels = [ch for ch in bad_channels if ch in eeg_info["ch_names"]]
    eeg_info["bads"] = bad_eeg_channels
    good_eeg_channels = [ch for ch in eeg_info["ch_names"] if ch not in bad_eeg_channels]

    # Handle dict input
    if isinstance(data, dict):
        aligned_data = np.full(n_eeg_channels, fill_value)
        for i, ch_name in enumerate(eeg_info["ch_names"]):
            if ch_name in data and ch_name not in bad_eeg_channels:
                aligned_data[i] = data[ch_name]
        return aligned_data, eeg_info

    # Handle array input
    data_flat = np.asarray(data).flatten()

    # Case 1: Data matches good channels
    if len(data_flat) == len(good_eeg_channels):
        aligned_data = np.full(n_eeg_channels, fill_value)
        good_indices = [
            i for i, ch in enumerate(eeg_info["ch_names"])
            if ch not in bad_eeg_channels
        ]
        aligned_data[good_indices] = data_flat
        return aligned_data, eeg_info

    # Case 2: Data matches full EEG montage
    if len(data_flat) == n_eeg_channels:
        return data_flat.copy(), eeg_info

    # Case 3: Data is larger (includes non-EEG channels)
    if len(eeg_channel_indices) <= len(data_flat):
        eeg_data = data_flat[eeg_channel_indices]
        
        if len(eeg_data) == len(good_eeg_channels):
            aligned_data = np.full(n_eeg_channels, fill_value)
            good_indices = [
                i for i, ch in enumerate(eeg_info["ch_names"])
                if ch not in bad_eeg_channels
            ]
            aligned_data[good_indices] = eeg_data
        elif len(eeg_data) == n_eeg_channels:
            aligned_data = eeg_data.copy()
        else:
            raise ValueError(
                f"Cannot align extracted EEG data: {len(eeg_data)} channels doesn't match "
                f"good ({len(good_eeg_channels)}) or total ({n_eeg_channels}) EEG channels"
            )
        return aligned_data, eeg_info

    raise ValueError(
        f"Cannot align data: {len(data_flat)} values don't match "
        f"EEG channels (good: {len(good_eeg_channels)}, total: {n_eeg_channels})"
    )


# =============================================================================
# Part 4: Report Data Loader (Main Orchestrator)
# =============================================================================

class ReportDataLoader:
    """Load and transform all data for report generation.
    
    This combines HDF5 reading, preprocessing data, and transformations.
    """

    def __init__(self, hdf5_path: Union[str, Path], fif_path: Union[str, Path] = None, skip_preprocessing: bool = False):
        self.hdf5_reader = HDF5Reader(hdf5_path)
        self.fif_path = Path(fif_path) if fif_path else None
        self.skip_preprocessing = skip_preprocessing
        if not skip_preprocessing:
            self.preprocessing_reader = PreprocessingReader()
        else:
            self.preprocessing_reader = None

    def load_all_data(self) -> Dict[str, Any]:
        """Load all data from HDF5 and preprocessing files."""
        logger.info("Loading HDF5 features...")
        report_data = self._load_hdf5_features()
        
        if self.skip_preprocessing:
            # Load epochs directly from FIF file without preprocessing metadata
            logger.info(f"Loading epochs from FIF file: {self.fif_path}")
            epochs = mne.read_epochs(str(self.fif_path), preload=True)
            report_data["epoch_info"] = epochs
            report_data["metadata"] = {
                "preprocessing_info": {},
                "bad_channels": [],
                "bad_epochs": [],
                "n_epochs_before_rejection": len(epochs),
                "n_epochs_after_rejection": len(epochs)
            }
            logger.info(f"Loaded {len(epochs)} epochs from FIF file")
        else:
            # Load with preprocessing metadata
            logger.info("Loading preprocessing data...")
            preprocessing_data = self.preprocessing_reader.load_preprocessing_data()
            
            # Add preprocessing data to report_data
            report_data["epoch_info"] = preprocessing_data["epochs"]  # Cleaned epochs for analysis
            report_data["metadata"] = {
                "preprocessing_info": preprocessing_data["preprocessing_info"],  # Full preprocessing JSON from pkl
                "bad_channels": preprocessing_data["bad_channels"],
                "bad_epochs": preprocessing_data["bad_epochs"],
                "n_epochs_before_rejection": preprocessing_data["n_epochs_before_rejection"],
                "n_epochs_after_rejection": preprocessing_data["n_epochs_after_rejection"]
            }
        
        logger.info(f"Loaded {len(report_data)} data sections")
        return report_data

    def _load_hdf5_features(self) -> Dict[str, Any]:
        """Load all features from HDF5 file."""
        features = self.hdf5_reader.list_features()
        report_data = {}

        for feature_name in features:
            try:
                feature_data = self.hdf5_reader.read_feature(feature_name)
                self._process_feature(feature_name, feature_data, report_data)
            except Exception as e:
                logger.warning(f"Could not process feature '{feature_name}': {e}")

        # Create CNV data structure
        self._create_cnv_data(report_data)
        
        # Organize connectivity data
        self._organize_connectivity_data(report_data)
        
        # Organize spectral data
        self._organize_spectral_data(report_data)

        return report_data

    def _process_feature(self, feature_name: str, feature_data: Dict, report_data: Dict):
        """Process a single feature and add to report_data.
        
        Maps HDF5 feature names to internal keys expected by computation modules.
        All features follow pattern: EEG_<name>_<markertype>
        """
        data = feature_data["data"]
        
        # SpectralPower - Absolute power (delta_power, theta_power, etc.)
        if "_power_spectralpower" in feature_name:
            band = feature_name.replace("EEG_", "").replace("_power_spectralpower", "")
            key = f"per_channel_{band}"
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # SpectralPower - Normalized/Relative power (delta_relative, theta_relative, etc.)
        elif "_relative_spectralpower" in feature_name:
            band = feature_name.replace("EEG_", "").replace("_relative_spectralpower", "")
            key = f"{band}n_spectralpower"  # deltan, thetan, etc.
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # SpectralPower - Spectral entropy
        elif "spectral_entropy_spectralpower" in feature_name:
            report_data["spectral_entropy"] = data
            logger.info(f"Loaded {feature_name} -> spectral_entropy")
        
        # PowerSpectralDensitySummary (msf, sef90, sef95)
        elif "psdsummary" in feature_name:
            summary_type = feature_name.replace("EEG_", "").replace("_psdsummary", "")
            key = f"{summary_type}_per_channel"
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # PermutationEntropy
        elif "permutationentropy" in feature_name:
            # Extract band from name like EEG_pe_theta_permutationentropy
            parts = feature_name.replace("EEG_", "").split("_")
            band = parts[1]  # theta, alpha, beta, gamma
            key = f"pe_{band}"
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # SymbolicMutualInformation (WSMI)
        elif "symbolicmutualinformation" in feature_name:
            # Extract band from name like EEG_wsmi_theta_symbolicmutualinformation
            parts = feature_name.replace("EEG_", "").split("_")
            band = parts[1]  # theta, alpha, beta, gamma
            key = f"wsmi_{band}"
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # KolmogorovComplexity
        elif "kolmogorovcomplexity" in feature_name:
            report_data["kolmogorov_complexity"] = data
            logger.info(f"Loaded {feature_name} -> kolmogorov_complexity")
        
        # ContingentNegativeVariation
        elif "cnv" in feature_name.lower():
            key = feature_name.replace("EEG_", "")
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # TimeLockedTopography
        elif "timelockedtopo" in feature_name:
            key = feature_name.replace("EEG_", "")
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # TimeLockedContrast
        elif "timelockedcontrast" in feature_name:
            key = feature_name.replace("EEG_", "")
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")
        
        # WindowDecoding
        elif "windowdecoding" in feature_name:
            # EEG_window_decoding_local_windowdecoding -> window_decoding_local
            key = feature_name.replace("EEG_", "").replace("_windowdecoding", "")
            report_data[key] = data
            logger.info(f"Loaded {feature_name} -> {key}")

    def _extract_band_from_name(self, feature_name: str) -> str:
        """Extract frequency band from feature name."""
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            if band in feature_name:
                return band
        return None

    def _create_cnv_data(self, report_data: Dict):
        """Create CNV data structure from slopes and intercepts."""
        # Look for CNV data with either naming convention
        slope_key = None
        intercept_key = None
        
        if "cnv_slopes_trials" in report_data:
            slope_key = "cnv_slopes_trials"
            intercept_key = "cnv_intercepts_trials"
        elif "cnv_detailed_cnvslope" in report_data:
            slope_key = "cnv_detailed_cnvslope"
            intercept_key = "cnv_detailed_cnvintercept"
        
        if slope_key and intercept_key:
            slopes = report_data[slope_key]
            intercepts = report_data[intercept_key]
            
            # Reshape CNV data if needed
            if slopes.ndim == 2 and slopes.shape[1] == 1:
                # Data is flattened as (total_elements, 1) - try common EGI montages
                total_elements = slopes.shape[0]
                
                # Try common EGI channel counts (prioritize 256 - full EGI montage)
                for n_ch in [256, 265, 257, 234, 129, 128]:
                    if total_elements % n_ch == 0:
                        n_trials = total_elements // n_ch
                        slopes = slopes.reshape(n_trials, n_ch)
                        intercepts = intercepts.reshape(n_trials, n_ch)
                        logger.info(f"Reshaped CNV data from ({total_elements}, 1) to ({n_trials}, {n_ch})")
                        break
                else:
                    logger.warning(f"Could not determine proper CNV reshape for {slopes.shape}")
            
            report_data["cnv_data"] = {
                "cnv_slopes_trials": slopes,
                "cnv_intercepts_trials": intercepts,
            }
            logger.info(f"Created cnv_data structure: {slopes.shape[0]} trials, {slopes.shape[1]} channels")
        else:
            logger.warning("No CNV slope/intercept data found in report_data")

    def _organize_connectivity_data(self, report_data: Dict):
        """Organize connectivity features."""
        connectivity_keys = [k for k in report_data.keys() if k.startswith("smi_") or k.startswith("wsmi_")]
        if connectivity_keys:
            report_data["connectivity_data"] = {k: report_data[k] for k in connectivity_keys}
            report_data["per_band_connectivity"] = {
                k: report_data[k] for k in connectivity_keys if k.startswith("wsmi_")
            }
            logger.info(f"Loaded connectivity data: {list(report_data['connectivity_data'].keys())}")

    def _organize_spectral_data(self, report_data: Dict):
        """Organize spectral features by band."""
        # Absolute power per channel
        per_channel_spectral = {}
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            key = f"per_channel_{band}"
            if key in report_data:
                per_channel_spectral[band] = report_data[key]
        
        if per_channel_spectral:
            report_data["per_channel_spectral"] = per_channel_spectral
            logger.info(f"Organized per-channel spectral data: {list(per_channel_spectral.keys())}")
        
        # Normalized power (aggregated)
        normalized_spectral = {}
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            key = f"{band}n_spectralpower"  # deltan, thetan, etc.
            if key in report_data:
                normalized_spectral[f"{band}n"] = report_data[key]
        
        if normalized_spectral:
            report_data["normalized_spectral"] = normalized_spectral
            logger.info(f"Organized normalized spectral data: {list(normalized_spectral.keys())}")
