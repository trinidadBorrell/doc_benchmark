"""Shared utility functions for marker computation (scalars and topographies)."""

import json
import numpy as np
from pathlib import Path
from scipy import stats


def trimmean80(data, axis=None):
    """Compute 80% trimmed mean (trim 10% from each tail)"""
    return stats.trim_mean(data, proportiontocut=0.1, axis=axis)


def create_256_to_64_roi_mapping():
    """Create mapping from 256-channel ROIs to 64-channel ROIs using the JSON mapping.

    Returns
    -------
    function
        Mapping function for ROI conversion
    """
    # Load the mapping file
    mapping_file = (
        Path(__file__).parent / ".." / ".." / "data" / "egi256_biosemi64.json"
    )

    with open(mapping_file, "r") as f:
        mapping_data = json.load(f)

    # Get the recombination groups (biosemi64 -> 256 electrodes)
    recombination_groups = mapping_data["recombination_groups"]

    # Create reverse mapping: 256 electrode number -> 64 channel index
    electrode_256_to_ch_64 = {}

    # Define the actual channel order from the fif file
    ch_64_names_ordered = [
        "Fp1",
        "AF7",
        "AF3",
        "F1",
        "F3",
        "F5",
        "F7",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "C1",
        "C3",
        "C5",
        "T7",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "P1",
        "P3",
        "P5",
        "P7",
        "P9",
        "PO7",
        "PO3",
        "O1",
        "Iz",
        "Oz",
        "POz",
        "Pz",
        "CPz",
        "Fpz",
        "Fp2",
        "AF8",
        "AF4",
        "AFz",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT8",
        "FC6",
        "FC4",
        "FC2",
        "FCz",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP8",
        "CP6",
        "CP4",
        "CP2",
        "P2",
        "P4",
        "P6",
        "P8",
        "P10",
        "PO8",
        "PO4",
        "O2",
    ]

    # Convert electrode names to indices and create mapping
    for ch_64_name, electrode_256_list in recombination_groups.items():
        if ch_64_name in ch_64_names_ordered:
            ch_64_idx = ch_64_names_ordered.index(ch_64_name)

            # Convert electrode names like "E33" to electrode numbers
            for electrode_name in electrode_256_list:
                electrode_num = int(electrode_name[1:])
                electrode_256_to_ch_64[electrode_num] = ch_64_idx

    def map_256_roi_to_64(roi_256):
        """Map a 256-channel ROI to corresponding 64-channel indices"""
        mapped_channels = set()
        for electrode_num in roi_256:
            if electrode_num in electrode_256_to_ch_64:
                mapped_channels.add(electrode_256_to_ch_64[electrode_num])
        return np.array(sorted(mapped_channels))

    return map_256_roi_to_64


def get_electrode_mapping(n_channels):
    """Get electrode mappings based on the number of channels.

    Parameters
    ----------
    n_channels : int
        Number of available channels

    Returns
    -------
    dict
        Dictionary with electrode mappings for different ROIs
    """
    if n_channels == 256:  # EGI 256-channel system
        scalp_roi = np.arange(224)
        cnv_roi = np.array([5, 6, 13, 14, 15, 21, 22]) - 1
        mmn_roi = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185]) - 1
        p3b_roi = np.array([8, 44, 80, 99, 100, 109, 118, 127, 128, 131, 185]) - 1
        p3a_roi = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185]) - 1

    elif n_channels == 64:  # Standard 64-channel system (biosemi64)
        scalp_roi = np.arange(64)

        map_roi = create_256_to_64_roi_mapping()

        # Original 256-channel ROIs (1-based)
        cnv_roi_256 = np.array([5, 6, 13, 14, 15, 21, 22])
        mmn_roi_256 = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])
        p3b_roi_256 = np.array([8, 44, 80, 99, 100, 109, 118, 127, 128, 131, 185])
        p3a_roi_256 = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])

        cnv_roi = map_roi(cnv_roi_256)
        mmn_roi = map_roi(mmn_roi_256)
        p3b_roi = map_roi(p3b_roi_256)
        p3a_roi = map_roi(p3a_roi_256)
    else:
        raise ValueError(f"Unsupported number of channels: {n_channels}")

    # Filter out channels that don't exist
    cnv_roi = cnv_roi[cnv_roi < n_channels]
    mmn_roi = mmn_roi[mmn_roi < n_channels]
    p3b_roi = p3b_roi[p3b_roi < n_channels]
    p3a_roi = p3a_roi[p3a_roi < n_channels]

    return {
        "scalp_roi": scalp_roi,
        "cnv_roi": cnv_roi,
        "mmn_roi": mmn_roi,
        "p3b_roi": p3b_roi,
        "p3a_roi": p3a_roi,
    }
