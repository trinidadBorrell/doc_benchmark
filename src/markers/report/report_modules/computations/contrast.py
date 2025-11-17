# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Computation functions for contrast analysis.
Separates heavy statistical computations from data extraction.
"""

import pickle
import numpy as np
import scipy.stats
from pathlib import Path
from mne.utils import logger


def compute_contrast(
    epochs,
    conditions,
    method=np.mean,
    roi_name=None,
    roi_channels=None,
    paired=False,
    output_path=None,
):
    """
    Compute contrast analysis with statistical tests between two conditions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    conditions : list
        List of two conditions to compare
    method : callable
        Reduction method (e.g., np.mean, trim_mean80)
    roi_name : str or None
        ROI name for channel selection
    roi_channels : list or None
        Specific ROI channels
    paired : bool
        Whether to use paired t-test
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed contrast data
    """
    from ..viz.equipments import get_roi_ch_names
    
    logger.info("Starting contrast analysis computations...")
    
    if len(conditions) != 2:
        raise ValueError('Need 2 conditions to make a contrast.')
    
    # Get all conditions
    all_conditions = (
        sum(conditions, []) if isinstance(conditions[0], list)
        else conditions)
    epochs_filtered = epochs[all_conditions].copy()
    
    # Apply ROI filtering if specified
    if roi_name is not None:
        if roi_channels is None:
            roi_channels = get_roi_ch_names(
                config=epochs_filtered.info['description'], roi_name=roi_name)
        epochs_filtered.pick_channels(roi_channels)
        this_data = epochs_filtered.get_data().mean(1, keepdims=True)
        to_drop = epochs_filtered.info['ch_names'][1:]
        to_rename = epochs_filtered.info['ch_names'][0]
        epochs_filtered.drop_channels(to_drop)
        epochs_filtered._data = this_data
        epochs_filtered.rename_channels({to_rename: 'ROI-MEAN'})
    
    # Compute evoked responses for each condition
    logger.info(f"Computing evoked responses for conditions: {conditions}")
    evokeds_data = []
    evokeds_stderr_data = []
    
    for c in conditions:
        evoked_data = method(epochs_filtered[c].get_data(), axis=0)
        stderr_data = scipy.stats.sem(epochs_filtered[c].get_data(), axis=0)
        evokeds_data.append(evoked_data)
        evokeds_stderr_data.append(stderr_data)
    
    # Compute contrast (difference)
    contrast_data = evokeds_data[0] - evokeds_data[1]
    
    # Compute statistical tests
    logger.info("Computing statistical tests...")
    epochs_a = epochs_filtered[conditions[0]]
    epochs_b = epochs_filtered[conditions[1]]
    
    if paired is False:
        t_val, p_val = scipy.stats.ttest_ind(
            epochs_a.get_data(), epochs_b.get_data(), axis=0,
            equal_var=False)
    else:
        t_val, p_val = scipy.stats.ttest_rel(
            epochs_a.get_data(), epochs_b.get_data(), axis=0)
    
    # Compute -log10(p) for visualization
    mlog10_p_val = -np.log10(p_val)
    
    # Compile all computed data
    computed_data = {
        "contrast_data": contrast_data,
        "evokeds_data": evokeds_data,
        "evokeds_stderr_data": evokeds_stderr_data,
        "p_val": p_val,
        "mlog10_p_val": mlog10_p_val,
        "t_val": t_val,
        "info": epochs_filtered.info,
        "tmin": epochs_filtered.tmin,
        "conditions": conditions,
        "method": method.__name__ if hasattr(method, '__name__') else str(method),
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving computed data to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(computed_data, f)
        logger.info("Computed data saved successfully")
    
    logger.info("Contrast analysis computations completed")
    return computed_data


def compute_contrast_1samp(
    epochs,
    conditions,
    method=np.mean,
    roi_name=None,
    output_path=None,
):
    """
    Compute one-sample contrast analysis (evoked responses without comparison).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    conditions : list or str
        List of conditions or 'all' for all conditions
    method : callable
        Reduction method (e.g., np.mean, trim_mean80)
    roi_name : str or None
        ROI name for channel selection
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed evoked data
    """
    from ..viz.equipments import get_roi_ch_names
    
    logger.info("Starting one-sample contrast computations...")
    
    if conditions == 'all':
        conditions = [list(epochs.event_id.keys())]
    
    all_conditions = (
        sum(conditions, []) if isinstance(conditions[0], list)
        else conditions)
    epochs_filtered = epochs[all_conditions].copy()
    
    # Apply ROI filtering if specified
    if roi_name is not None:
        roi_channels = get_roi_ch_names(
            config=epochs_filtered.info['description'], roi_name=roi_name)
        epochs_filtered.pick_channels(roi_channels)
        this_data = epochs_filtered.get_data().mean(1, keepdims=True)
        to_drop = epochs_filtered.info['ch_names'][1:]
        to_rename = epochs_filtered.info['ch_names'][0]
        epochs_filtered.drop_channels(to_drop)
        epochs_filtered._data = this_data
        epochs_filtered.rename_channels({to_rename: 'ROI-MEAN'})
    
    # Compute evoked responses
    logger.info(f"Computing evoked responses for conditions: {conditions}")
    evokeds_data = []
    evokeds_stderr_data = []
    
    for c in conditions:
        evoked_data = method(epochs_filtered[c].get_data(), axis=0)
        stderr_data = scipy.stats.sem(epochs_filtered[c].get_data(), axis=0)
        evokeds_data.append(evoked_data)
        evokeds_stderr_data.append(stderr_data)
    
    # Compile all computed data
    computed_data = {
        "evokeds_data": evokeds_data,
        "evokeds_stderr_data": evokeds_stderr_data,
        "info": epochs_filtered.info,
        "tmin": epochs_filtered.tmin,
        "conditions": conditions,
        "method": method.__name__ if hasattr(method, '__name__') else str(method),
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving computed data to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(computed_data, f)
        logger.info("Computed data saved successfully")
    
    logger.info("One-sample contrast computations completed")
    return computed_data
