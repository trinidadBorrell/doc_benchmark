# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import scipy

import re
from collections import namedtuple

import mne

from .equipments import get_roi_ch_names


def _evoked_apply_method(epochs, method):
    data = method(epochs, axis=0)
    out = mne.evoked.EvokedArray(data, epochs.info, epochs.tmin)
    return out


def get_evoked(epochs, condition, method=np.mean, roi_name=None):
    epochs = epochs[condition]

    if roi_name is not None:
        roi_channels = get_roi_ch_names(
            config=epochs.info['description'], roi_name=roi_name)
        epochs.pick_channels(roi_channels)
        this_data = epochs.get_data().mean(1, keepdims=True)
        to_drop = epochs.info['ch_names'][1:]
        to_rename = epochs.info['ch_names'][0]
        epochs.drop_channels(to_drop)
        epochs._data = this_data
        epochs.rename_channels({to_rename: 'ROI-MEAN'})

    evoked = _evoked_apply_method(epochs[condition], method)
    evoked_stderr = _evoked_apply_method(epochs[condition], scipy.stats.sem)

    return evoked, evoked_stderr


def get_contrast(computed_data_path=None, computed_data=None):
    """
    Get contrast results from pre-computed data.
    Returns Evoked objects for compatibility with existing code.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
        
    Returns
    -------
    evoked_out : mne.Evoked
        Contrast evoked (difference between conditions)
    evokeds : list of mne.Evoked
        Evoked responses for each condition
    evokeds_stderr : list of mne.Evoked
        Standard errors for each condition
    stats : namedtuple
        Statistical results with p_val and mlog10_p_val as Evoked objects
    """
    import pickle
    
    # Load computed data from pickle if path provided
    if computed_data is None:
        if computed_data_path is None:
            raise ValueError("Either computed_data_path or computed_data must be provided")
        with open(computed_data_path, "rb") as f:
            computed_data = pickle.load(f)
    
    # Extract computed data
    contrast_data = computed_data["contrast_data"]
    evokeds_data = computed_data["evokeds_data"]
    evokeds_stderr_data = computed_data["evokeds_stderr_data"]
    p_val = computed_data["p_val"]
    mlog10_p_val = computed_data["mlog10_p_val"]
    info = computed_data["info"]
    tmin = computed_data["tmin"]
    
    # Create Evoked objects for compatibility
    evoked_out = mne.evoked.EvokedArray(contrast_data, info, tmin)
    evokeds = [mne.evoked.EvokedArray(data, info, tmin) for data in evokeds_data]
    evokeds_stderr = [mne.evoked.EvokedArray(data, info, tmin) for data in evokeds_stderr_data]
    
    # Create stats namedtuple
    stats = namedtuple('stats', 'p_val mlog10_p_val')
    stats.p_val = mne.evoked.EvokedArray(p_val, info, tmin)
    stats.mlog10_p_val = mne.evoked.EvokedArray(mlog10_p_val, info, tmin)
    
    return evoked_out, evokeds, evokeds_stderr, stats


def get_contrast_1samp(computed_data_path=None, computed_data=None):
    """
    Get one-sample contrast results from pre-computed data.
    Returns Evoked objects for compatibility with existing code.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
        
    Returns
    -------
    evokeds : list of mne.Evoked
        Evoked responses for each condition
    evokeds_stderr : list of mne.Evoked
        Standard errors for each condition
    """
    import pickle
    
    # Load computed data from pickle if path provided
    if computed_data is None:
        if computed_data_path is None:
            raise ValueError("Either computed_data_path or computed_data must be provided")
        with open(computed_data_path, "rb") as f:
            computed_data = pickle.load(f)
    
    # Extract computed data
    evokeds_data = computed_data["evokeds_data"]
    evokeds_stderr_data = computed_data["evokeds_stderr_data"]
    info = computed_data["info"]
    tmin = computed_data["tmin"]
    
    # Create Evoked objects for compatibility
    evokeds = [mne.evoked.EvokedArray(data, info, tmin) for data in evokeds_data]
    evokeds_stderr = [mne.evoked.EvokedArray(data, info, tmin) for data in evokeds_stderr_data]
    
    return evokeds, evokeds_stderr


def fname_regexp_event(fname, regex_map, event_id):
    found = 0
    match = 0
    for reg, evt in regex_map.items():
        if re.match(reg, fname) is not None:
            found = found + 1
            match = evt
    if found == 0:
        raise ValueError('No regexp match for {}'.format(fname))
    elif found > 1:
        raise ValueError('More than one match for {}'.format(fname))

    return event_id[match]
