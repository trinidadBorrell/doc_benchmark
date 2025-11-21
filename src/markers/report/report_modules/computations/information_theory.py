"""
Information theory analysis computation functions.

This module contains all computation logic for information theory analysis.
"""

import logging
import numpy as np
from pathlib import Path
from collections import OrderedDict

from ..data_io import MarkerDataAdapter, align_data_to_eeg_montage

logger = logging.getLogger(__name__)


def compute_information_theory_data(epochs, report_data, output_dir="./tmp_computed_data"):
    """
    Compute all information theory data and save to pkl (NO PLOTTING).
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object
    report_data : dict
        Report data dictionary
    output_dir : str or Path
        Directory to save computed data
        
    Returns
    -------
    dict
        Dictionary with paths to saved data files
    """
    from .topos import compute_marker_topo
    
    logger.info("Computing information theory analysis data...")
    
    output_paths = {}
    
    try:
        # 1. Compute per-band Permutation Entropy
        pe_path = _compute_per_band_permutation_entropy(epochs, report_data, output_dir)
        if pe_path:
            output_paths['permutation_entropy'] = pe_path

        # 2. Compute Kolmogorov complexity
        kc_path = _compute_kolmogorov_complexity(epochs, report_data, output_dir)
        if kc_path:
            output_paths['kolmogorov_complexity'] = kc_path
        
        # 3. Compute generic per-channel information theory measures
        per_channel_info_theory = {}
        if "kolmogorov_per_channel" in report_data:
            per_channel_info_theory["kolmogorov_complexity"] = (
                report_data["kolmogorov_per_channel"]
            )
        if "permutation_entropy_per_channel" in report_data:
            per_channel_info_theory["permutation_entropy"] = (
                report_data["permutation_entropy_per_channel"]
            )

        if per_channel_info_theory and epochs is not None:
            for measure_name, measure_data in per_channel_info_theory.items():
                try:
                    data_array = np.asarray(measure_data, dtype=np.float64)
                    if data_array.ndim == 1 and len(data_array) == 64:
                        adapter = MarkerDataAdapter(
                            data=data_array,
                            name=measure_name,
                            ch_info=epochs.info,
                        )
                        measure_path = Path(output_dir) / f"info_theory_{measure_name}.pkl"
                        compute_marker_topo(
                            adapter,
                            reduction=lambda x: x,
                            outlines="egi",
                            output_path=measure_path,
                        )
                        logger.info(f"Computed {measure_name} information theory data")
                        output_paths[f'info_theory_{measure_name}'] = measure_path
                except Exception as e:
                    logger.warning(f"Could not compute {measure_name} data: {e}")
        
        logger.info("✅ Information theory computations complete")

    except Exception as e:
        logger.error(f"Failed to compute information theory data: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")

    return output_paths


def _compute_per_band_permutation_entropy(epochs, report_data, output_dir):
    """Compute per-band Permutation Entropy topoplots data."""
    from .topos import compute_markers_topos
    
    try:
        logger.info("Computing per-band Permutation Entropy...")

        pe_markers = OrderedDict()
        band_configs = [
            ("delta", "pe_delta"),
            ("theta", "pe_theta"),
            ("alpha", "pe_alpha"),
            ("beta", "pe_beta"),
            ("gamma", "pe_gamma"),
        ]

        # Get bad channels from preprocessing info
        preprocessing_bad_channels = []
        if "metadata" in report_data:
            preprocessing_info = report_data["metadata"].get("preprocessing_info", {})
            preprocessing_bad_channels = preprocessing_info.get("bad_channels_detected", [])
            logger.info(f"Using preprocessing bad channels for PE: {preprocessing_bad_channels}")

        for band_name, marker_name in band_configs:
            pe_key = f"pe_{band_name}"
            if pe_key not in report_data:
                logger.warning(f"PE data key '{pe_key}' not found in report data")
                continue
            
            # Convert to numpy array (junifer data comes in correct shape)
            pe_data = np.array(report_data[pe_key])
            logger.info(f"Found PE {band_name} band data in key '{pe_key}' with shape: {pe_data.shape}")
            
            # Aggregate across epochs if data is (epochs, channels)
            if pe_data.ndim == 3 and pe_data.shape[0] == 1:
                # Shape is (1, epochs, channels) - squeeze first dimension
                pe_data = pe_data.squeeze(0)
            
            if pe_data.ndim == 2:
                # Data is (epochs, channels) - aggregate across epochs
                logger.info(f"Aggregating PE {band_name} across {pe_data.shape[0]} epochs")
                pe_data = np.mean(pe_data, axis=0)
                logger.info(f"Aggregated PE {band_name}: shape {pe_data.shape}")
            elif pe_data.ndim == 1:
                # Data is already aggregated (channels,)
                logger.info(f"PE {band_name} already aggregated: shape {pe_data.shape}")
            else:
                logger.warning(f"Unexpected PE {band_name} shape: {pe_data.shape}")

            try:
                pe_data_aligned, eeg_info = align_data_to_eeg_montage(
                    pe_data,
                    epochs.info,
                    fill_value=np.nan,
                    bad_channels=preprocessing_bad_channels,
                )
                logger.info(
                    f"Aligned PE {band_name} band: {len(pe_data)} -> {len(pe_data_aligned)} channels"
                )
            except ValueError as e:
                logger.warning(f"Cannot align PE {band_name} band data: {e}")
                continue

            pe_marker = MarkerDataAdapter(
                data=pe_data_aligned,
                name=f"pe_{band_name}",
                ch_info=eeg_info,
            )
            pe_markers[f"nice/marker/PermutationEntropy/{band_name}"] = pe_marker
            logger.info(
                f"Loaded PE {band_name} band: mean={np.mean(pe_data_aligned):.4f}, "
                f"range=[{np.min(pe_data_aligned):.4f}, {np.max(pe_data_aligned):.4f}]"
            )

        if not pe_markers:
            logger.warning("No PE band data available for topoplots")
            logger.info(f"Available data keys: {list(report_data.keys())}")
            return None

        s_reductions = {}
        s_picks = {}

        def identity_reduction(data):
            return data

        for pe_key in pe_markers.keys():
            s_reductions[pe_key] = identity_reduction
            s_picks[pe_key] = None

        # Compute topographies
        pe_path = Path(output_dir) / "permutation_entropy_bands.pkl"
        compute_markers_topos(
            pe_markers, s_reductions, s_picks, outlines="egi/256",
            output_path=pe_path,
        )
        logger.info("✅ Permutation Entropy topographies computed")
        return pe_path

    except Exception as e:
        logger.error(f"Failed to create per-band PE topoplots: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None


def _compute_kolmogorov_complexity(epochs, report_data, output_dir):
    """Compute Kolmogorov complexity topoplot data"""
    from .topos import compute_markers_topos
    
    try:
        logger.info("Computing Kolmogorov complexity...")

        kc_data = None
        if "kolmogorov_per_channel" in report_data:
            kc_data = report_data["kolmogorov_per_channel"]
        elif "kolmogorov_complexity" in report_data:
            kc_data = report_data["kolmogorov_complexity"]

        if kc_data is None:
            logger.warning("No Kolmogorov complexity data found")
            return None

        # Convert to numpy array (junifer data comes in correct shape)
        kc_data = np.array(kc_data)
        logger.info(f"Kolmogorov complexity data shape: {kc_data.shape}")
        
        # Aggregate across epochs if data is (epochs, channels)
        if kc_data.ndim == 3 and kc_data.shape[0] == 1:
            # Shape is (1, epochs, channels) - squeeze first dimension
            kc_data = kc_data.squeeze(0)
        
        if kc_data.ndim == 2:
            # Data is (epochs, channels) - aggregate across epochs
            logger.info(f"Aggregating Kolmogorov across {kc_data.shape[0]} epochs")
            kc_data = np.mean(kc_data, axis=0)
            logger.info(f"Aggregated Kolmogorov: shape {kc_data.shape}")
        elif kc_data.ndim == 1:
            # Data is already aggregated (channels,)
            logger.info(f"Kolmogorov already aggregated: shape {kc_data.shape}")
        else:
            logger.warning(f"Unexpected Kolmogorov shape: {kc_data.shape}")
            return None

        try:
            kc_data_aligned, eeg_info = align_data_to_eeg_montage(
                kc_data, epochs.info, fill_value=np.nan
            )
            logger.info(
                f"Aligned Kolmogorov complexity: {len(kc_data)} -> {len(kc_data_aligned)} channels"
            )
        except ValueError as e:
            logger.warning(f"Cannot align Kolmogorov complexity data: {e}")
            return None

        kc_marker = MarkerDataAdapter(
            data=kc_data_aligned,
            name="kolmogorov_complexity",
            ch_info=eeg_info,
        )

        kc_markers = OrderedDict()
        kc_markers["nice/marker/KolmogorovComplexity/complexity"] = kc_marker

        logger.info(
            f"Loaded Kolmogorov complexity: mean={np.mean(kc_data_aligned):.4f}, "
            f"range=[{np.min(kc_data_aligned):.4f}, {np.max(kc_data_aligned):.4f}]"
        )

        s_reductions = {}
        s_picks = {}

        def identity_reduction(data):
            return data

        for kc_key in kc_markers.keys():
            s_reductions[kc_key] = identity_reduction
            s_picks[kc_key] = None

        # Compute topography
        kc_path = Path(output_dir) / "kolmogorov_complexity.pkl"
        compute_markers_topos(
            kc_markers, s_reductions, s_picks, outlines="egi/256",
            output_path=kc_path,
        )
        logger.info("✅ Kolmogorov complexity computed")
        return kc_path

    except Exception as e:
        logger.error(f"Failed to create Kolmogorov complexity topoplot: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None
