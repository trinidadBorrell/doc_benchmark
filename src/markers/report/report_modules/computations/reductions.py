"""
NICE-style reduction functions for ICM prediction pipeline.

These reductions convert multi-dimensional marker data (epochs × channels × ...)
into scalar features suitable for machine learning.

Based on next_icm/nice_ext/api/reductions.py and next_icm/lg/reductions.py
"""

import numpy as np
from scipy.stats import trim_mean
from scipy.stats import entropy as scipy_entropy
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Core Statistical Functions (from NICE framework)
# =============================================================================

def trim_mean80(a, axis=0):
    """Trimmed mean keeping 80% of center data (removes 10% from each tail)"""
    return trim_mean(a, proportiontocut=0.1, axis=axis)


def trim_mean90(a, axis=0):
    """Trimmed mean keeping 90% of center data (removes 5% from each tail)"""
    return trim_mean(a, proportiontocut=0.05, axis=axis)


def entropy(a, axis=0):
    """Shannon entropy along specified axis"""
    if axis is None:
        return scipy_entropy(a.flatten())
    # Apply entropy along axis
    return np.apply_along_axis(scipy_entropy, axis, a)


# =============================================================================
# Data Reshaping Utilities
# =============================================================================

def reshape_flattened_data(data, n_epochs, n_channels):
    """
    Reshape flattened Junifer data back to (epochs × channels).
    
    Parameters
    ----------
    data : np.ndarray
        Flattened data of shape (n_epochs*n_channels, 1) or (n_epochs*n_channels,)
    n_epochs : int
        Number of epochs
    n_channels : int
        Number of channels
        
    Returns
    -------
    np.ndarray
        Reshaped data of shape (n_epochs, n_channels)
    """
    data_flat = np.asarray(data).flatten()
    expected_size = n_epochs * n_channels
    
    if len(data_flat) != expected_size:
        raise ValueError(
            f"Data size {len(data_flat)} doesn't match epochs×channels "
            f"({n_epochs}×{n_channels}={expected_size})"
        )
    
    return data_flat.reshape(n_epochs, n_channels)


def reconstruct_connectivity_matrix(upper_tri_data, n_epochs, n_channels):
    """
    Reconstruct full symmetric connectivity matrix from upper triangle (no diagonal).
    
    WSMI data is stored as upper triangle without diagonal: n_channels×(n_channels-1)/2
    per epoch.
    
    Parameters
    ----------
    upper_tri_data : np.ndarray
        Flattened upper triangle data
    n_epochs : int
        Number of epochs
    n_channels : int
        Number of channels
        
    Returns
    -------
    np.ndarray
        Full connectivity matrices of shape (n_epochs, n_channels, n_channels)
    """
    data_flat = np.asarray(upper_tri_data).flatten()
    n_pairs = n_channels * (n_channels - 1) // 2
    expected_size = n_epochs * n_pairs
    
    if len(data_flat) != expected_size:
        raise ValueError(
            f"Data size {len(data_flat)} doesn't match epochs×pairs "
            f"({n_epochs}×{n_pairs}={expected_size})"
        )
    
    # Reshape to (epochs, n_pairs)
    data_reshaped = data_flat.reshape(n_epochs, n_pairs)
    
    # Reconstruct full matrices
    matrices = np.zeros((n_epochs, n_channels, n_channels))
    
    for epoch_idx in range(n_epochs):
        # Get upper triangle indices (k=1 means exclude diagonal)
        triu_indices = np.triu_indices(n_channels, k=1)
        
        # Fill upper triangle
        matrices[epoch_idx][triu_indices] = data_reshaped[epoch_idx]
        
        # Make symmetric (copy upper to lower)
        matrices[epoch_idx] = matrices[epoch_idx] + matrices[epoch_idx].T
    
    return matrices


# =============================================================================
# ROI Filtering (from next_icm/nice_ext/equipments/rois.py)
# =============================================================================

def get_scalp_roi_indices(n_channels=256):
    """
    Get indices of scalp channels for EGI montage.
    
    For EGI/256, non-scalp channels are typically the face/neck electrodes.
    Based on NICE framework ROI definitions.
    
    Parameters
    ----------
    n_channels : int
        Total number of channels (256 or 128)
        
    Returns
    -------
    np.ndarray
        Indices of scalp channels
    """
    if n_channels == 256:
        # EGI/256: exclude face/neck channels (approximate - based on typical montages)
        # Channels 1-234 are typically scalp, 235-256 are face/neck
        # This is a simplification - ideally load from ROI config
        return np.arange(0, 234)  # First 234 channels
    elif n_channels == 128:
        # EGI/128: similar pattern
        return np.arange(0, 120)
    else:
        logger.warning(f"Unknown montage with {n_channels} channels, using all channels")
        return np.arange(n_channels)


def get_cnv_roi_indices(n_channels=256):
    """CNV ROI - frontocentral channels"""
    if n_channels == 256:
        # Approximate frontocentral region for CNV
        # Based on typical EGI/256 layout
        return np.array([5, 6, 7, 11, 12, 13, 106, 112, 129])  # Fz, FCz region
    else:
        # Fallback to central channels
        return np.arange(0, min(20, n_channels))


def get_mmn_roi_indices(n_channels=256):
    """MMN ROI - frontocentral channels"""
    # Similar to CNV for MMN
    return get_cnv_roi_indices(n_channels)


def get_p3a_roi_indices(n_channels=256):
    """P3a ROI - frontocentral channels"""
    if n_channels == 256:
        return np.array([5, 6, 7, 11, 12, 13, 62, 106, 112])  # Fz, Cz region
    else:
        return np.arange(0, min(20, n_channels))


def get_p3b_roi_indices(n_channels=256):
    """P3b ROI - parietal channels"""
    if n_channels == 256:
        return np.array([31, 37, 54, 55, 61, 62, 78, 79, 80])  # Pz region
    else:
        return np.arange(min(40, n_channels//2), min(60, n_channels))


# =============================================================================
# NICE Reduction Pipeline
# =============================================================================

def apply_reduction_pipeline(data, reduction_config, picks_config=None):
    """
    Apply NICE-style reduction pipeline to marker data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with labeled axes
    reduction_config : list of dict
        List of reduction steps, each with 'axis' and 'function' keys
    picks_config : dict, optional
        Channel/epoch picks per axis
        
    Returns
    -------
    scalar or np.ndarray
        Reduced data
    """
    result = data.copy()
    
    for step in reduction_config:
        axis_name = step['axis']
        func = step['function']
        
        # Apply picks if specified
        if picks_config and axis_name in picks_config:
            picks = picks_config[axis_name]
            if picks is not None:
                # Apply picks along the appropriate axis
                result = np.take(result, picks, axis=0)  # Simplified - adjust axis as needed
        
        # Apply reduction function
        result = func(result, axis=0)
    
    return result


# =============================================================================
# LG-Specific Reduction Configurations
# =============================================================================

def get_lg_reduction_config(reduction_type, marker_type=None):
    """
    Get reduction configuration for LG protocol.
    
    Maps reduction type strings to specific reduction functions and ROI picks.
    
    Parameters
    ----------
    reduction_type : str
        One of: 'icm/lg/egi256/trim_mean80', 'icm/lg/egi256/std',
                'icm/lg/egi256gfp/trim_mean80', 'icm/lg/egi256gfp/std'
    marker_type : str, optional
        Marker type for special handling (e.g., 'ContingentNegativeVariation')
        
    Returns
    -------
    dict
        Configuration with 'channels_fun', 'epochs_fun', 'picks', and optionally 'times_fun'
    """
    # Parse reduction type
    if 'gfp' in reduction_type:
        channels_fun = np.std  # GFP = std across channels
    else:
        channels_fun = np.mean  # Regular = mean across channels
    
    if 'trim_mean80' in reduction_type:
        epochs_fun = trim_mean80
    elif 'std' in reduction_type:
        epochs_fun = np.std
    else:
        epochs_fun = np.mean  # Default
    
    # Get appropriate ROI indices
    picks = {}
    times_fun = None
    
    if marker_type == 'ContingentNegativeVariation':
        picks['channels'] = get_cnv_roi_indices()
    elif marker_type in ['TimeLockedContrast/mmn', 'mmn']:
        picks['channels'] = get_mmn_roi_indices()
        times_fun = np.mean  # ERP markers need time averaging
    elif marker_type in ['TimeLockedContrast/p3a', 'p3a']:
        picks['channels'] = get_p3a_roi_indices()
        times_fun = np.mean
    elif marker_type in ['TimeLockedContrast/p3b', 'p3b']:
        picks['channels'] = get_p3b_roi_indices()
        times_fun = np.mean
    elif marker_type in ['TimeLockedTopography', 'TimeLockedContrast']:
        picks['channels'] = get_scalp_roi_indices()
        times_fun = np.mean  # ERP markers need time averaging
    else:
        # Default: scalp ROI for most markers
        picks['channels'] = get_scalp_roi_indices()
    
    config = {
        'channels_fun': channels_fun,
        'epochs_fun': epochs_fun,
        'picks': picks
    }
    
    if times_fun is not None:
        config['times_fun'] = times_fun
    
    return config


# =============================================================================
# High-Level Reduction Functions
# =============================================================================

def reduce_to_scalar(data, n_epochs, n_channels, marker_type, reduction_type, is_connectivity=False):
    """
    Reduce marker data to a single scalar value following NICE pipeline.
    
    Parameters
    ----------
    data : np.ndarray
        Flattened data from Junifer HDF5
    n_epochs : int
        Number of epochs
    n_channels : int
        Number of channels
    marker_type : str
        Type of marker
    reduction_type : str
        Reduction configuration string (e.g., 'icm/lg/egi256/trim_mean80')
    is_connectivity : bool
        Whether data is connectivity matrix (needs special handling)
        
    Returns
    -------
    float
        Scalar feature value
    """
    # Get reduction configuration
    config = get_lg_reduction_config(reduction_type, marker_type)
    channels_fun = config['channels_fun']
    epochs_fun = config['epochs_fun']
    scalp_roi = config['picks'].get('channels', None)
    
    # Reshape data
    if is_connectivity:
        # Reconstruct full connectivity matrices
        data_reshaped = reconstruct_connectivity_matrix(data, n_epochs, n_channels)
        # (epochs, channels, channels)
        
        # Apply median across channels_y dimension (axis=2)
        data_reshaped = np.median(data_reshaped, axis=2)  # → (epochs, channels)
    else:
        # Normal reshape to (epochs, channels)
        data_reshaped = reshape_flattened_data(data, n_epochs, n_channels)
    
    # Apply ROI filtering on channels
    if scalp_roi is not None:
        data_reshaped = data_reshaped[:, scalp_roi]  # (epochs, scalp_channels)
    
    # Aggregate across channels
    data_per_epoch = channels_fun(data_reshaped, axis=1)  # → (epochs,)
    
    # Aggregate across epochs
    scalar_value = epochs_fun(data_per_epoch, axis=0)  # → scalar
    
    return float(scalar_value)
