# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Computation functions for topography analysis.
Separates heavy computations from plotting for better modularity.
"""

import pickle
import numpy as np
import mne
from pathlib import Path
from mne.utils import logger


def process_topos_for_visualization(
    topos,
    reference_good_idx,
    non_scalp,
    vmin,
    same_scale=False,
    ignore_non_scalp=False,
):
    """
    Post-process topographies for visualization by handling non-scalp channels.
    
    This separates the raw computation from visualization-specific transformations.
    Non-scalp channels are set to vmin so they appear as the minimum value in plots.
    
    Parameters
    ----------
    topos : list of np.ndarray
        Raw topography arrays
    reference_good_idx : np.ndarray
        Indices of good channels
    non_scalp : np.ndarray or None
        Indices of non-scalp channels
    vmin : float
        Minimum value for scaling (used for same_scale=True)
    same_scale : bool
        Whether using same scale across all markers
    ignore_non_scalp : bool
        Whether to skip non-scalp processing
        
    Returns
    -------
    list of np.ndarray
        Processed topographies ready for visualization
    """
    processed_topos = []
    
    for topo in topos:
        topo = topo.copy()  # Don't modify original
        
        # Calculate vmin for this topo if not using same_scale
        if same_scale is False:
            topo_vmin = np.nanmin(topo)
        else:
            topo_vmin = vmin
        
        # Handle non-scalp channels
        if non_scalp is not None and not ignore_non_scalp:
            if len(topo) == len(reference_good_idx):
                # Map non_scalp indices to good channels positions
                non_scalp_good = []
                for idx in non_scalp:
                    if idx < len(reference_good_idx):
                        if idx in reference_good_idx:
                            good_pos = np.where(reference_good_idx == idx)[0]
                            if len(good_pos) > 0:
                                non_scalp_good.append(good_pos[0])
                
                if non_scalp_good:
                    topo[non_scalp_good] = topo_vmin
            else:
                # Fallback: apply directly if dimensions match
                valid_non_scalp = [idx for idx in non_scalp if idx < len(topo)]
                if valid_non_scalp:
                    topo[valid_non_scalp] = topo_vmin
        
        processed_topos.append(topo)
    
    return processed_topos


def compute_markers_topos(
    markers,
    reductions,
    picks=None,
    outlines="head",
    same_scale=False,
    ignore_non_scalp=False,
    output_path=None,
):
    """
    Compute topography data for multiple markers.
    
    Parameters
    ----------
    markers : dict
        Dictionary of marker objects
    reductions : dict
        Dictionary of reduction functions for each marker
    picks : dict or None
        Dictionary of channel picks for each marker
    outlines : str
        Equipment/montage specification
    same_scale : bool
        Whether to use same scale across all markers
    ignore_non_scalp : bool
        Whether to ignore non-scalp channels
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed data for plotting
    """
    from ..viz.equipments import get_roi, prepare_layout
    
    logger.info("Starting markers topography computations...")
    
    # Get ROIs
    scalp_roi = get_roi(config=outlines, roi_name="scalp")
    non_scalp = get_roi(config=outlines, roi_name="nonscalp")
    
    # CRITICAL: Use consistent channel info for all markers
    ch_info = markers[list(markers.keys())[0]].ch_info_
    
    # Create unified list of bad channels
    all_bad_channels = set()
    for marker in markers.values():
        all_bad_channels.update(marker.ch_info_.get("bads", []))
    
    # Create consistent ch_info with all bad channels
    consistent_ch_info = ch_info.copy()
    consistent_ch_info["bads"] = list(all_bad_channels)
    
    # Prepare layout
    logger.info("Preparing layout and positions...")
    sphere, outlines_data = prepare_layout(outlines, info=consistent_ch_info)
    _, pos, _, _, _, this_sphere, clip_origin = (
        mne.viz.topomap._prepare_topomap_plot(
            consistent_ch_info, "eeg", sphere=sphere
        )
    )
    
    # Calculate good indices based on consistent channel info
    reference_good_idx = mne.pick_channels(
        consistent_ch_info["ch_names"],
        include=[],
        exclude=consistent_ch_info["bads"],
    )
    reference_good_idx = np.array(reference_good_idx, dtype=int)
    
    # Create mask for good channels only
    if scalp_roi is not None and len(scalp_roi) > 0:
        scalp_good_channels = np.intersect1d(scalp_roi, reference_good_idx)
        mask = np.in1d(reference_good_idx, scalp_good_channels)
    else:
        mask = np.ones(len(reference_good_idx), dtype=bool)
    
    # Compute topographies for all markers
    logger.info(f"Computing topographies for {len(markers)} markers...")
    topos = []
    marker_names = []
    
    for name, marker in markers.items():
        logger.info(f"Computing topography for marker: {name}")
        
        # Apply reduction to marker
        topo = marker.reduce_to_topo(reductions[name], picks[name])
        
        # Align topo to reference dimensions
        if len(topo) != len(reference_good_idx):
            if len(topo) == len(marker.ch_info_["ch_names"]):
                topo = topo[reference_good_idx]
            elif len(topo) < len(reference_good_idx):
                aligned_topo = np.full(len(reference_good_idx), np.nan)
                aligned_topo[: len(topo)] = topo
                topo = aligned_topo
            else:
                topo = topo[: len(reference_good_idx)]
        
        topos.append(topo)
        marker_names.append(name)
    
    # Calculate scale if using same_scale
    vmin = -np.inf
    vmax = np.inf
    if same_scale is True:
        vmin = np.nanmin(topos)
        vmax = np.nanmax(topos)
    
    # Post-process topographies for visualization
    # This handles non-scalp channels by setting them to vmin
    processed_topos = process_topos_for_visualization(
        topos=topos,
        reference_good_idx=reference_good_idx,
        non_scalp=non_scalp,
        vmin=vmin,
        same_scale=same_scale,
        ignore_non_scalp=ignore_non_scalp,
    )
    
    # Compile all computed data
    computed_data = {
        "topos": processed_topos,
        "marker_names": marker_names,
        "pos": pos,
        "mask": mask,
        "vmin": vmin,
        "vmax": vmax,
        "same_scale": same_scale,
        "markers": markers,  # Keep reference for text mapping
        "ch_info": consistent_ch_info,  # Save for regenerating sphere/outlines during plotting
        # Note: sphere and outlines not saved (contain unpicklable functions)
        # They will be regenerated during plotting from ch_info
        "scalp_roi": scalp_roi,
        "non_scalp": non_scalp,
        "reference_good_idx": reference_good_idx,
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving computed data to {output_path}")
        
        # Create a copy for pickling without the full outlines (has unpicklable patch)
        pickle_data = computed_data.copy()
        pickle_data.pop('outlines_full', None)  # Remove unpicklable version
        
        with open(output_path, "wb") as f:
            pickle.dump(pickle_data, f)
        logger.info("Computed data saved successfully")
    
    logger.info("Markers topography computations completed")
    return computed_data


def compute_marker_topo(
    marker,
    reduction,
    picks=None,
    outlines="head",
    output_path=None,
):
    """
    Compute topography data for a single marker.
    
    Parameters
    ----------
    marker : object
        Marker object
    reduction : callable
        Reduction function
    picks : array or None
        Channel picks
    outlines : str
        Equipment/montage specification
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed data for plotting
    """
    logger.info("Computing single marker topography...")
    
    name = "single"
    markers = {name: marker}
    reductions = {name: reduction}
    picks_dict = {name: picks}
    
    return compute_markers_topos(
        markers,
        reductions,
        picks=picks_dict,
        outlines=outlines,
        output_path=output_path,
    )
