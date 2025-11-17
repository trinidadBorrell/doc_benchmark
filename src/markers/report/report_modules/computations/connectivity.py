"""
Connectivity analysis computation functions.

This module contains all computation logic for connectivity analysis.
"""

import logging
import numpy as np
from pathlib import Path
from collections import OrderedDict

from ..data_io import MarkerDataAdapter, align_data_to_eeg_montage

logger = logging.getLogger(__name__)


def compute_connectivity_analysis_data(epochs, report_data, output_dir="./tmp_computed_data"):
    """
    Compute connectivity analysis data and save to pkl (NO PLOTTING).
    
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

    logger.info("Computing connectivity analysis data...")
    
    output_paths = {}
    
    # 1. Compute per-band WSMI topoplots
    wsmi_path = _compute_per_band_connectivity(epochs, report_data, output_dir)
    if wsmi_path:
        output_paths['wsmi_bands'] = wsmi_path

    # 2. Compute mutual information topography if available
    try:
        connectivity_data = report_data.get("connectivity_data", {})

        if connectivity_data:
            if (
                "mutual_information" in report_data
                and "per_channel" in report_data["mutual_information"]
            ):
                connectivity_per_ch = report_data["mutual_information"]["per_channel"]

                eeg_channels = [
                    ch
                    for ch in epochs.ch_names
                    if ch.startswith("E") and ch[1:].isdigit()
                ]
                eeg_indices = [
                    epochs.ch_names.index(ch)
                    for ch in eeg_channels
                    if ch in epochs.ch_names
                ]

                if len(eeg_indices) == 256:
                    connectivity_values = np.zeros(256, dtype=np.float64)

                    for i, ch_name in enumerate(eeg_channels):
                        if ch_name in connectivity_per_ch:
                            connectivity_values[i] = connectivity_per_ch[ch_name]

                    import mne
                    eeg_info = mne.pick_info(epochs.info, eeg_indices, copy=True)
                    eeg_info["description"] = "egi/256"

                    logger.info("Using real connectivity data: 256 EEG channels")
                    
                adapter = MarkerDataAdapter(
                    data=connectivity_values,
                    name="mutual_information",
                    ch_info=eeg_info,
                )

                # Compute marker topo data
                logger.info("Computing mutual information topography...")
                mi_topo_path = Path(output_dir) / "mutual_info_topo.pkl"
                compute_marker_topo(
                    adapter,
                    reduction=lambda x: x,
                    outlines="egi/256",
                    output_path=mi_topo_path,
                )
                logger.info("✅ Mutual information topography computed")
                output_paths['mutual_info'] = mi_topo_path

    except Exception as e:
        logger.warning(f"Could not compute connectivity topography: {e}")
    
    logger.info("✅ Connectivity computations complete")
    return output_paths


def _compute_per_band_connectivity(epochs, report_data, output_dir="./tmp_computed_data"):
    """Compute per-band WSMI connectivity topoplots data."""
    from .topos import compute_markers_topos
    
    try:
        logger.info("Computing per-band WSMI connectivity topoplots")

        band_configs = [
            ("delta", "wsmi_delta"),
            ("theta", "wsmi_theta"),
            ("alpha", "wsmi_alpha"),
            ("beta", "wsmi_beta"),
            ("gamma", "wsmi_gamma"),
        ]

        wsmi_markers = OrderedDict()
        bands_found = 0

        for band_name, marker_pattern in band_configs:
            wsmi_data = None

            for key in report_data.keys():
                if marker_pattern in key.lower():
                    wsmi_data = report_data[key]
                    logger.info(
                        f"Found WSMI {band_name} band data in key '{key}' with shape: {np.array(wsmi_data).shape}"
                    )
                    break

            if wsmi_data is not None:
                wsmi_data = np.array(wsmi_data)
                if wsmi_data.ndim == 2 and wsmi_data.shape[1] == 1:
                    wsmi_data = wsmi_data.flatten()
                    logger.info(f"Flattened WSMI {band_name} data to shape: {wsmi_data.shape}")
                
                # Reshape connectivity data from upper triangle to full matrix
                # WSMI is stored as upper triangle (no diagonal): epochs × (ch×(ch-1)/2)
                n_epochs = len(epochs)
                n_channels = len(epochs.ch_names)
                n_pairs = n_channels * (n_channels - 1) // 2
                
                if len(wsmi_data) == n_epochs * n_pairs:
                    logger.info(f"Reshaping WSMI from ({len(wsmi_data)},) to ({n_epochs}, {n_pairs}) then aggregating")
                    
                    # Reshape to (epochs, n_pairs)
                    wsmi_reshaped = wsmi_data.reshape(n_epochs, n_pairs)
                    
                    # Reconstruct symmetric matrices for each epoch
                    wsmi_matrices = np.zeros((n_epochs, n_channels, n_channels))
                    for epoch_idx in range(n_epochs):
                        triu_indices = np.triu_indices(n_channels, k=1)
                        wsmi_matrices[epoch_idx][triu_indices] = wsmi_reshaped[epoch_idx]
                        wsmi_matrices[epoch_idx] = wsmi_matrices[epoch_idx] + wsmi_matrices[epoch_idx].T
                    
                    # Aggregate: median across channels_y, then mean across epochs
                    wsmi_median_per_ch = np.median(wsmi_matrices, axis=2)  # (epochs, channels)
                    wsmi_aggregated = np.mean(wsmi_median_per_ch, axis=0)   # (channels,)
                    
                    logger.info(f"Aggregated WSMI across {n_epochs} epochs: final shape {wsmi_aggregated.shape}")
                    wsmi_data = wsmi_aggregated
                else:
                    logger.warning(
                        f"WSMI data size {len(wsmi_data)} doesn't match expected "
                        f"epochs×pairs ({n_epochs}×{n_pairs}={n_epochs*n_pairs}). "
                        f"Using first {n_channels} values only."
                    )
                    wsmi_data = wsmi_data[:n_channels]

                try:
                    wsmi_data_aligned, eeg_info = align_data_to_eeg_montage(
                        wsmi_data, epochs.info, fill_value=np.nan
                    )
                    logger.info(
                        f"Aligned WSMI {band_name} band: {len(wsmi_data)} -> {len(wsmi_data_aligned)} channels"
                    )

                    wsmi_marker = MarkerDataAdapter(
                        data=wsmi_data_aligned,
                        name=f"wsmi_{band_name}",
                        ch_info=eeg_info,
                    )
                    wsmi_markers[f"nice/marker/WSMI/{band_name}"] = wsmi_marker
                    bands_found += 1
                    logger.info(
                        f"Loaded WSMI {band_name} band: mean={np.mean(wsmi_data_aligned):.4f}, "
                        f"range=[{np.min(wsmi_data_aligned):.4f}, {np.max(wsmi_data_aligned):.4f}]"
                    )

                except ValueError as e:
                    logger.warning(f"Cannot align WSMI {band_name} band data: {e}")
                    continue
            else:
                logger.warning(
                    f"No data found for WSMI {band_name} band marker '{marker_pattern}'"
                )

        if bands_found == 0:
            logger.warning("No WSMI band data available for topoplots")
            logger.info(f"Available data keys: {list(report_data.keys())}")
            return None

        wsmi_reductions = {}
        wsmi_picks = {}
        for band_key in wsmi_markers.keys():
            wsmi_reductions[band_key] = lambda x: x
            wsmi_picks[band_key] = None

        # Compute WSMI topographies
        logger.info("Computing WSMI topographies...")
        wsmi_topo_path = Path(output_dir) / "wsmi_bands_topo.pkl"
        compute_markers_topos(
            wsmi_markers,
            wsmi_reductions,
            picks=wsmi_picks,
            outlines="egi/256",
            output_path=wsmi_topo_path,
        )
        logger.info("✅ WSMI topographies computed")
        return wsmi_topo_path

    except Exception as e:
        logger.error(f"Failed to create WSMI connectivity topoplots: {e}")
        import traceback
        traceback.print_exc()
        return None
