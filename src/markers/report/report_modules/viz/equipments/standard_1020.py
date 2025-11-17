# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import mne
import numpy as np
from mne.utils import logger

from .montages import define_equipment
from .rois import define_rois

# Standard 10-20 montage channel names (64 channels)
_standard_1020_64_ch_names = [
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

# Channel names configuration for standard_1020 equipment
_standard_1020_ch_names = {"64": _standard_1020_64_ch_names}

# Standard 10-20 montage configurations for standard_1020
_standard_1020_montages = {"64": "standard_1020"}

# ROI definitions for standard 10-20 montage (64 channels)
# Based on standard electrode positions and typical ERP component locations
_standard_1020_rois = {
    # Frontal midline (Fz region) - typically used for MMN, CNV
    "Fz": np.array([37]),  # Fz electrode index
    "cnv": np.array([2, 3, 36, 37, 38, 39]),  # AF3, F1, AFz, Fz, F2, AF4
    "mmn": np.array(
        [2, 3, 36, 37, 38, 39, 10, 46]
    ),  # Frontocentral: AF3, F1, AFz, Fz, F2, AF4, FC1, FCz
    # Central midline (Cz region) - typically used for P3a
    "Cz": np.array([47]),  # Cz electrode index
    "p3a": np.array(
        [10, 11, 45, 46, 47, 48, 49]
    ),  # Frontocentral: FC1, C1, FC2, FCz, Cz, C2, C4
    # Parietal midline (Pz region) - typically used for P3b
    "Pz": np.array([30]),  # Pz electrode index
    "p3b": np.array(
        [18, 19, 29, 30, 31, 54, 55, 56]
    ),  # Centroparietal: CP1, P1, POz, Pz, CPz, CP2, P2, CP4
    # All scalp electrodes (all 64 channels are EEG)
    "scalp": np.arange(64),
    # No non-scalp channels in standard 10-20 EEG
    "nonscalp": None,
}


def _prepare_standard_1020_layout(ch_config):
    """Prepare layout for standard 10-20 montage using MNE's built-in functionality"""
    from matplotlib.patches import Circle
    from mne.viz.topomap import _prepare_topomap_plot

    # Create info object for the specified channel configuration
    ch_names = _standard_1020_ch_names[ch_config]
    info = mne.create_info(ch_names, 1, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")

    # Use MNE's standard topomap preparation for standard_1020
    # Unpack all 7 values like the EGI layout function does
    _, pos, _, _, _, this_sphere, clip_origin = _prepare_topomap_plot(
        info, "eeg", sphere=None
    )

    # Create outlines dictionary following EGI pattern
    # For standard_1020, we'll create a simple circular outline using all electrode positions
    n_electrodes = len(pos)

    # Create a circular outline using the outermost electrodes
    # Find electrodes on the perimeter (approximate circle)
    angles = np.arctan2(pos[:, 1], pos[:, 0])
    radii = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)

    # Sort by angle to create a circular outline
    sorted_indices = np.argsort(angles)
    perimeter_indices = sorted_indices[
        radii[sorted_indices] > np.percentile(radii, 75)
    ]

    # Create outline coordinates
    outline_pos = pos[perimeter_indices, :]
    outlines = {}
    outlines["head"] = (outline_pos[:, 0], outline_pos[:, 1])
    outlines["outer"] = outlines["head"]  # Same as head for standard_1020

    # Add patch function for compatibility
    def patch():
        return Circle(
            (0, 0), radius=np.max(radii), color="white", alpha=0.1, fill=False
        )

    outlines["mask_pos"] = outlines["outer"]
    outlines["patch"] = patch
    outlines["clip_radius"] = clip_origin

    return this_sphere, outlines


def register():
    """Register standard_1020 montage with nice_ext equipment system"""
    logger.info("Defining standard_1020")
    define_equipment(
        "standard_1020",
        _standard_1020_montages,
        _standard_1020_ch_names,
        _prepare_standard_1020_layout,
    )
    define_rois("standard_1020/64", _standard_1020_rois)
