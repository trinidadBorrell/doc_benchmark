"""
Spectral analysis computation functions.

This module contains all computation logic for spectral analysis.
"""

import logging
import numpy as np
from pathlib import Path
from collections import OrderedDict

from ..data_io import MarkerDataAdapter, align_data_to_eeg_montage

logger = logging.getLogger(__name__)


def compute_spectral_analysis_data(epochs, report_data, output_dir="./tmp_computed_data"):
    """
    Compute all spectral analysis data and save to pkl (NO PLOTTING).
    
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
    logger.info("Computing spectral analysis data...")
    
    output_paths = {}
    
    try:
        per_channel_spectral = report_data.get("per_channel_spectral", {})
        normalized_spectral = report_data.get("normalized_spectral", {})
        
        if not per_channel_spectral and not normalized_spectral:
            logger.warning("No spectral data found")
            return output_paths

        preprocessing_bad_channels = _get_preprocessing_bad_channels(report_data)

        # 1. Compute normalized spectral bands
        bands_path = _compute_normalized_spectral_bands(
            epochs, normalized_spectral, preprocessing_bad_channels, output_dir
        )
        if bands_path:
            output_paths['spectral_bands_normalized'] = bands_path

        # 2. Compute absolute power (log scale)
        absolute_path = _compute_spectral_power_log(
            epochs, per_channel_spectral, preprocessing_bad_channels, output_dir
        )
        if absolute_path:
            output_paths['spectral_absolute_power'] = absolute_path

        # 3. Compute spectral summaries
        summaries_path = _compute_spectral_summaries(epochs, report_data, output_dir)
        if summaries_path:
            output_paths['spectral_summaries'] = summaries_path

        logger.info("✅ Spectral computations complete")

    except Exception as e:
        logger.error(f"Failed to compute spectral data: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")

    return output_paths


def _get_preprocessing_bad_channels(report_data):
    """Extract bad channels from preprocessing info"""
    preprocessing_bad_channels = []
    if "metadata" in report_data:
        preprocessing_info = report_data["metadata"].get("preprocessing_info", {})
        preprocessing_bad_channels = preprocessing_info.get("bad_channels_detected", [])
        logger.info(f"Using preprocessing bad channels: {preprocessing_bad_channels}")
    return preprocessing_bad_channels


def _compute_normalized_spectral_bands(epochs, normalized_spectral, preprocessing_bad_channels, output_dir):
    """Compute normalized spectral bands data.
    
    Note: Data from HDF5 is already in dB scale (10*log10 of relative power),
    so no additional transformation is needed.
    """
    from .topos import compute_markers_topos
    
    try:
        spectral_bands = OrderedDict()
        # Normalized bands have 'n' suffix: deltan, thetan, etc.
        band_names = ["delta_normalized", "theta_normalized", "alpha_normalized", "beta_normalized", "gamma_normalized"]

        for band in band_names:
            if band in normalized_spectral:
                # Convert to numpy array (junifer data comes in correct shape)
                band_data = np.array(normalized_spectral[band])
                logger.info(f"Found {band} band data with shape: {band_data.shape}")
                
                # Aggregate across epochs if data is (epochs, channels)
                if band_data.ndim == 3 and band_data.shape[0] == 1:
                    # Shape is (1, epochs, channels) - squeeze first dimension
                    band_data = band_data.squeeze(0)
                
                if band_data.ndim == 2:
                    # Data is (epochs, channels) - aggregate across epochs
                    logger.info(f"Aggregating {band} across {band_data.shape[0]} epochs")
                    band_data = np.mean(band_data, axis=0)
                    logger.info(f"Aggregated {band}: shape {band_data.shape}")
                elif band_data.ndim == 1:
                    # Data is already aggregated (channels,)
                    logger.info(f"{band} already aggregated: shape {band_data.shape}")
                else:
                    logger.warning(f"Unexpected {band} shape: {band_data.shape}")
                    continue

                try:
                    band_data_aligned, eeg_info = align_data_to_eeg_montage(
                        band_data,
                        epochs.info,
                        fill_value=0.0,
                        bad_channels=preprocessing_bad_channels,
                    )
                    logger.info(
                        f"Aligned {band} band: {len(band_data)} -> {len(band_data_aligned)} channels"
                    )
                except ValueError as e:
                    logger.warning(f"Cannot align {band} band data: {e}")
                    continue

                band_marker = MarkerDataAdapter(
                    data=band_data_aligned,
                    name=f"spectral_{band}",
                    ch_info=eeg_info,
                )
                spectral_bands[f"nice/marker/PowerSpectralDensity/{band}"] = band_marker
                logger.info(
                    f"Loaded {band} band (dB): mean={np.mean(band_data_aligned):.2f}, "
                    f"range=[{np.min(band_data_aligned):.2f}, {np.max(band_data_aligned):.2f}]"
                )

        if not spectral_bands:
            logger.warning("No spectral bands available")
            return None

        s_reductions = {}
        s_picks = {}
        
        def identity_reduction(data):
            return data

        for band_key in spectral_bands.keys():
            s_reductions[band_key] = identity_reduction
            s_picks[band_key] = None

        # Compute topographies
        bands_path = Path(output_dir) / "spectral_bands_normalized.pkl"
        compute_markers_topos(
            spectral_bands, s_reductions, s_picks, outlines="egi/256",
            output_path=bands_path
        )
        logger.info("✅ Normalized spectral bands computed")
        return bands_path

    except Exception as e:
        logger.error(f"Failed to compute normalized spectral bands: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None


def _compute_spectral_power_log(epochs, per_channel_spectral, preprocessing_bad_channels, output_dir):
    """Compute absolute spectral power (non-normalized, log scale).
    
    Note: Data from HDF5 is already in dB scale (10*log10 of absolute power),
    so no additional transformation is needed.
    """
    from .topos import compute_markers_topos
    
    try:
        logger.info("Computing absolute spectral power...")
        spectral_absolute_bands = OrderedDict()
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]

        for band in band_names:
            if band in per_channel_spectral:
                # Convert to numpy array (junifer data comes in correct shape)
                band_data = np.array(per_channel_spectral[band])
                logger.info(f"Found {band} band for absolute power with shape: {band_data.shape}")
                
                # Aggregate across epochs if data is (epochs, channels)
                if band_data.ndim == 3 and band_data.shape[0] == 1:
                    # Shape is (1, epochs, channels) - squeeze first dimension
                    band_data = band_data.squeeze(0)
                
                if band_data.ndim == 2:
                    # Data is (epochs, channels) - aggregate across epochs
                    logger.info(f"Aggregating {band} across {band_data.shape[0]} epochs")
                    band_data = np.mean(band_data, axis=0)
                    logger.info(f"Aggregated {band}: shape {band_data.shape}")
                elif band_data.ndim == 1:
                    # Data is already aggregated (channels,)
                    logger.info(f"{band} already aggregated: shape {band_data.shape}")
                else:
                    logger.warning(f"Unexpected {band} shape: {band_data.shape}")
                    continue

                try:
                    band_data_aligned, eeg_info = align_data_to_eeg_montage(
                        band_data,
                        epochs.info,
                        fill_value=0.0,
                        bad_channels=preprocessing_bad_channels,
                    )
                except ValueError as e:
                    logger.warning(f"Cannot align {band} band data for log power: {e}")
                    continue

                band_marker = MarkerDataAdapter(
                    data=band_data_aligned,
                    name=f"absolute_power_{band}",
                    ch_info=eeg_info,
                )
                spectral_absolute_bands[
                    f"nice/marker/PowerSpectralDensityAbsolute/{band}"
                ] = band_marker

        if not spectral_absolute_bands:
            logger.warning("No absolute spectral power data available")
            return None

        s_reductions = {}
        s_picks = {}
        
        def identity_reduction(data):
            return data

        for band_key in spectral_absolute_bands.keys():
            s_reductions[band_key] = identity_reduction
            s_picks[band_key] = None

        absolute_path = Path(output_dir) / "spectral_absolute_power.pkl"
        compute_markers_topos(
            spectral_absolute_bands, s_reductions, s_picks, outlines="egi/256",
            output_path=absolute_path
        )
        logger.info("✅ Absolute spectral power computed")
        return absolute_path

    except Exception as e:
        logger.error(f"Failed to compute spectral power (log): {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None


def _compute_spectral_summaries(epochs, report_data, output_dir):
    """Compute spectral summaries (SE, MSF, SEF90, SEF95)"""
    from .topos import compute_markers_topos
    
    try:
        logger.info("Computing spectral summaries...")

        per_channel_spectral_summaries = {}
        if "msf_per_channel" in report_data:
            per_channel_spectral_summaries["MSF"] = report_data["msf_per_channel"]
        if "sef90_per_channel" in report_data:
            per_channel_spectral_summaries["SEF90"] = report_data["sef90_per_channel"]
        if "sef95_per_channel" in report_data:
            per_channel_spectral_summaries["SEF95"] = report_data["sef95_per_channel"]

        if not per_channel_spectral_summaries:
            logger.warning("No spectral summary data available")
            return None

        spectral_summaries = OrderedDict()
        summary_names = ["MSF", "SEF90", "SEF95"]
        nice_names = {"MSF": "msf", "SEF90": "sef90", "SEF95": "sef95"}

        for summary in summary_names:
            if summary in per_channel_spectral_summaries:
                summary_data = per_channel_spectral_summaries[summary]
                
                if hasattr(summary_data, "shape") and len(summary_data.shape) > 1:
                    summary_data = summary_data.flatten()

                try:
                    summary_data_aligned, eeg_info = align_data_to_eeg_montage(
                        summary_data,
                        epochs.info,
                        fill_value=np.nan,
                    )
                except ValueError as e:
                    logger.warning(f"Cannot align {summary} summary data: {e}")
                    continue

                summary_marker = MarkerDataAdapter(
                    data=summary_data_aligned,
                    name=f"spectral_summary_{summary}",
                    ch_info=eeg_info,
                )
                spectral_summaries[nice_names[summary]] = summary_marker

        if not spectral_summaries:
            logger.warning("No spectral summary data available")
            return None

        s_reductions = {}
        s_picks = {}
        
        def identity_reduction(data):
            return data

        for summary_key in spectral_summaries.keys():
            s_reductions[summary_key] = identity_reduction
            s_picks[summary_key] = None

        summaries_path = Path(output_dir) / "spectral_summaries.pkl"
        compute_markers_topos(
            spectral_summaries, s_reductions, s_picks, outlines="egi/256",
            output_path=summaries_path
        )
        logger.info("✅ Spectral summaries computed")
        return summaries_path

    except Exception as e:
        logger.error(f"Failed to compute spectral summaries: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None
