"""
Computation module for evoked data analysis.
Handles all heavy computations before visualization.
"""

import logging
import mne
import numpy as np
import pickle
import scipy.stats as scistats
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_cnv_analysis(
    cnv,
    reduction_func,
    outlines="head",
    stat_psig=0.05,
    stat_pvmin=1,
    stat_pvmax=1e-5,
    epochs=None,
    rois=None,
    n_permutations=1000,  # Reduced from 10000 to speed up computation
    output_path=None,
):
    """
    Compute all CNV analysis data including statistical tests and topographies.
    
    Parameters
    ----------
    cnv : CNVMarkerAdapter
        CNV marker object with slopes and intercepts
    reduction_func : callable
        Function to reduce data across trials
    outlines : str
        Montage configuration
    stat_psig : float
        P-value threshold for significance
    stat_pvmin : float
        Minimum p-value for colormap
    stat_pvmax : float
        Maximum p-value for colormap
    epochs : mne.Epochs
        Epochs object for additional computations
    rois : list
        List of ROI names to analyze
    n_permutations : int
        Number of permutations for cluster test
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed data for plotting
    """
    from ..viz.equipments import (
        get_roi,
        prepare_layout,
    )
    
    if rois is None:
        rois = ["Cz"]
    
    logger.info("Starting CNV analysis computations...")
    
    # Get montage and ROI information
    scalp_roi = get_roi(config=outlines, roi_name="scalp")
    non_scalp = get_roi(config=outlines, roi_name="nonscalp")
    rois_chs = [
        get_roi(config=outlines, roi_name=roi_name) for roi_name in rois
    ]
    
    # Adjacency not needed since we skip cluster testing for CNV
    # adjacency = get_ch_adjacency_montage(
    #     config=outlines, pick_names=montage_names
    # )
    sphere, outlines_data = prepare_layout(outlines, info=cnv.ch_info_)
    
    # Compute topography
    topo = cnv.reduce_to_topo(reduction_func)
    # Filter non_scalp to valid indices
    if non_scalp is not None and len(non_scalp) > 0:
        valid_non_scalp = [idx for idx in non_scalp if idx < len(topo)]
        if len(valid_non_scalp) > 0:
            topo[valid_non_scalp] = 0.0
    
    slopes = cnv.data_
    intercepts = cnv.intercepts_
    
    # Mass univariate one sample t-test
    _, p_topo = scistats.ttest_1samp(cnv.data_, popmean=0, axis=0)
    p_topo = -np.log10(p_topo)
    # Filter non_scalp to valid indices
    if non_scalp is not None and len(non_scalp) > 0:
        valid_non_scalp = [idx for idx in non_scalp if idx < len(p_topo)]
        if len(valid_non_scalp) > 0:
            p_topo[valid_non_scalp] = 0.0
    
    # Skip cluster permutation test (too slow for CNV data)
    cluster_mask = None
    n_clusters = 0
    obs = None
    clusters = []
    p_clusters = np.array([])
    
    # Get electrode positions for CNV plotting
    from mne.viz.topomap import _prepare_topomap_plot
    
    _, pos, _, _, _, _, _ = _prepare_topomap_plot(
        epochs.info, "eeg", sphere=sphere
    )
    
    # CNV now has full 256 channels - no expansion needed
    # Handle topo data filtering to good channels
    good_channels = mne.pick_channels(
        epochs.ch_names, include=[], exclude=epochs.info.get("bads", [])
    )
    
    if len(topo) == len(good_channels):
        topo_filtered = topo
    elif len(topo) == len(epochs.ch_names):
        topo_filtered = topo[good_channels]
    else:
        pos_len = len(pos)
        topo_filtered = (
            topo[:pos_len]
            if len(topo) >= pos_len
            else np.pad(topo, (0, pos_len - len(topo)), constant_values=np.nan)
        )
    
    # Create mask for scalp channels
    mask = np.in1d(np.arange(len(topo)), scalp_roi)
    if len(topo) == len(good_channels):
        mask_filtered = mask
    elif len(topo) == len(epochs.ch_names):
        mask_filtered = mask[good_channels]
    else:
        pos_len = len(pos)
        mask_filtered = (
            mask[:pos_len]
            if len(mask) >= pos_len
            else np.pad(mask, (0, pos_len - len(mask)), constant_values=False)
        )
    
    # Compute ROI-specific data
    roi_data = []
    for roi_name, roi in zip(rois, rois_chs):
        # Filter ROI channels to only valid indices
        n_channels = slopes.shape[1]
        valid_roi = [ch for ch in roi if ch < n_channels]
        
        if len(valid_roi) == 0:
            logger.warning(f"No valid channels for ROI {roi_name} (data has {n_channels} channels)")
            continue
        
        roi_cnv = slopes[:, valid_roi].mean(axis=1)
        roi_intercept = intercepts[:, valid_roi].mean(axis=1)
        _, p = scistats.ttest_1samp(roi_cnv, popmean=0)
        
        mean_slope = roi_cnv.mean(axis=0)
        mean_intercept = roi_intercept.mean(axis=0)
        cnv_line = [mean_intercept, mean_intercept + 0.6 * mean_slope]
        
        # Compute average evoked response for this ROI
        try:
            # Average epochs across all conditions for this ROI
            roi_epochs = epochs.copy().pick_channels(
                [epochs.ch_names[i] for i in valid_roi if i < len(epochs.ch_names)]
            )
            evoked = roi_epochs.average()
            # Calculate SEM across epochs
            data_sem = roi_epochs.get_data().std(axis=0) / np.sqrt(len(roi_epochs))
            evoked_stderr = evoked.copy()
            evoked_stderr.data = data_sem
        except Exception as e:
            logger.warning(f"Could not compute evoked for {roi_name}: {e}")
            evoked = None
            evoked_stderr = None
        
        roi_data.append({
            "roi_name": roi_name,
            "roi_cnv": roi_cnv,
            "roi_intercept": roi_intercept,
            "p_value": p,
            "mean_slope": mean_slope,
            "mean_intercept": mean_intercept,
            "cnv_line": cnv_line,
            "evoked": evoked,
            "evoked_stderr": evoked_stderr,
        })
    
    # Calculate stat parameters
    stat_logpsig = -np.log10(stat_psig)
    stat_vmin = np.log10(stat_pvmin)
    stat_vmax = -np.log10(stat_pvmax)
    
    # Calculate vminmax for colorbar (ignore NaN values)
    vminmax = max(np.abs(np.nanmin(topo_filtered)), np.abs(np.nanmax(topo_filtered)))
    
    # Prepare outlines for pickling (remove unpicklable 'patch' function)
    # Note: outlines_data is the dict, outlines is the string config
    picklable_outlines = {}
    if outlines_data is not None:
        picklable_outlines = {k: v for k, v in outlines_data.items() if k != 'patch'}
    
    # Compile all computed data
    computed_data = {
        "topo": topo,
        "topo_filtered": topo_filtered,
        "p_topo": p_topo,
        "obs": obs,
        "clusters": clusters,
        "p_clusters": p_clusters,
        "cluster_mask": cluster_mask,
        "n_clusters": n_clusters,
        "slopes": slopes,
        "intercepts": intercepts,
        "scalp_roi": scalp_roi,
        "non_scalp": non_scalp,
        "mask": mask,
        "mask_filtered": mask_filtered,
        "pos": pos,
        "sphere": sphere,
        "outlines": picklable_outlines,  # Use picklable version without 'patch'
        "rois": rois,
        "rois_chs": rois_chs,
        "roi_data": roi_data,
        "stat_psig": stat_psig,
        "stat_logpsig": stat_logpsig,
        "stat_vmin": stat_vmin,
        "stat_vmax": stat_vmax,
        "stat_pvmin": stat_pvmin,
        "stat_pvmax": stat_pvmax,
        "vminmax": vminmax,
        "ch_info": epochs.info,  # Full montage info to match 256-element data arrays
        # Note: sphere and outlines not saved (contain unpicklable functions)
        # They will be regenerated during plotting from ch_info
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(computed_data, f)
    return computed_data


def compute_cluster_test(
    epochs,
    conditions,
    labels,
    p_threshold=1e-2,
    f_threshold=10,
    n_permutations=1000,
    shift_time=0,
    event_times=None,
    output_path=None,
):
    """
    Compute cluster test analysis for comparing conditions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    conditions : list
        List of condition names to compare
    labels : list
        Labels for the conditions
    p_threshold : float
        P-value threshold for significance
    f_threshold : float or str
        F-threshold for cluster test, or 'auto'
    shift_time : float
        Time shift for plotting
    event_times : dict
        Event times for reference
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict or None
        Dictionary containing all computed data for plotting, or None if no clusters
    """
    from ..viz.equipments import get_roi_ch_names, prepare_layout
    from ..viz.equipments.montages import get_ch_adjacency
    from scipy import stats
    
    logger.info("Starting cluster test computations...")
    
    # Ensure description is set to egi/256 for consistent montage
    if epochs.info.get("description") is None:
        epochs.info["description"] = "egi/256"
    
    # Filter to scalp channels
    scalp_roi_names = get_roi_ch_names(epochs.info["description"], "scalp")
    epochs_filtered = epochs.copy().pick_channels(scalp_roi_names)
    adjacency = get_ch_adjacency(epochs_filtered)
    
    # Prepare data for cluster test
    X = [epochs_filtered[k].get_data() for k in conditions]
    X = [np.transpose(x, (0, 2, 1)) for x in X]
    
    # Calculate f_threshold if auto
    if f_threshold == "auto":
        f_threshold = stats.distributions.f.ppf(
            1.0 - 1e-5 / 2.0, X[0].shape[0] - 1, X[1].shape[0] - 1
        )
        logger.info(f"Using generated f_threshold {f_threshold} from p {1e-5}")
    
    # Run cluster test (expensive!)
    logger.info(f"Running spatio-temporal cluster test with {n_permutations} permutations...")
    cluster_stats = mne.stats.spatio_temporal_cluster_test(
        X,
        n_permutations=n_permutations,
        tail=0,
        n_jobs=-1,
        threshold=f_threshold,
        adjacency=adjacency,
    )
    
    T_obs, clusters, p_values, _ = cluster_stats
    sort_idx = np.argsort(p_values)
    p_values = p_values[sort_idx]
    clusters = [clusters[x] for x in sort_idx]
    
    if len(clusters) == 0:
        logger.info("No clusters found")
        return None
    
    # Determine number of significant clusters
    n_clusters = np.sum(p_values < p_threshold)
    if n_clusters == 0:
        logger.info(
            f"No significant cluster. Plotting lowest p-value ({p_values[0]})."
        )
        n_clusters = 1
    else:
        logger.info(f"Found {n_clusters} significant clusters.")
    
    # Get layout info - use egi/256 explicitly for consistent montage
    sphere, outlines = prepare_layout("egi/256", info=epochs_filtered.info)
    
    # CRITICAL: Extract 2D electrode positions for proper topomap plotting
    # This ensures the montage is correctly applied
    from mne.viz.topomap import _prepare_topomap_plot
    _, pos, _, _, _, _, _ = _prepare_topomap_plot(
        epochs_filtered.info, "eeg", sphere=sphere
    )
    logger.info(f"Extracted electrode positions: {pos.shape}")
    
    # Process each cluster
    logger.info(f"Processing {n_clusters} clusters...")
    cluster_data = []
    for i in range(n_clusters):
        time_inds, space_inds = clusters[i]
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        cluster_ch_names = [epochs_filtered.ch_names[i] for i in ch_inds]
        f_map = T_obs[time_inds, ...].mean(axis=0)
        
        # Compute evoked responses for cluster ROI channels
        # We need to average across cluster channels to match NICE's behavior
        from ..viz.contrast import get_evoked
        
        # Get evoked for each condition WITH cluster averaging
        # This matches NICE's get_contrast(roi_channels=cluster_ch_names) behavior
        evokeds = []
        evokeds_stderr = []
        for cond in conditions:
            # Pick cluster channels and average across them
            epochs_cluster = epochs_filtered.copy().pick_channels(cluster_ch_names)
            epochs_cond = epochs_cluster[cond]
            
            # Average across channels (same as NICE's roi_channels logic)
            this_data = epochs_cond.get_data().mean(1, keepdims=True)
            to_drop = epochs_cond.info['ch_names'][1:]
            to_rename = epochs_cond.info['ch_names'][0]
            epochs_cond.drop_channels(to_drop)
            epochs_cond._data = this_data
            epochs_cond.rename_channels({to_rename: 'ROI-MEAN'})
            
            # Now compute evoked and stderr on averaged data
            evoked, evoked_stderr = get_evoked(epochs_cond, cond, method=np.mean)
            evokeds.append(evoked)
            evokeds_stderr.append(evoked_stderr)
        
        # Create mask for this cluster
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        sig_times = (epochs_filtered.times[time_inds] + shift_time) * 1000
        
        # Create significance mask for evoked plot
        sig_mask = [x in time_inds for x in range(evokeds[0].times.shape[0])]
        
        cluster_data.append({
            "ch_inds": ch_inds,
            "time_inds": time_inds,
            "cluster_ch_names": cluster_ch_names,
            "f_map": f_map,
            "mask": mask,
            "sig_times": sig_times,
            "evokeds": evokeds,
            "evokeds_stderr": evokeds_stderr,
            "sig_mask": sig_mask,
            "p_value": p_values[i],
        })
    
    # Compile all computed data
    computed_data = {
        "T_obs": T_obs,
        "clusters": clusters,
        "p_values": p_values,
        "n_clusters": n_clusters,
        "cluster_data": cluster_data,
        "epochs_info": epochs_filtered.info,
        "pos": pos,  # Save 2D electrode positions for plotting
        "labels": labels,
        "shift_time": shift_time,
        "event_times": event_times,
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving computed data to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(computed_data, f)
        logger.info("Computed data saved successfully")
    
    logger.info("Cluster test computations completed")
    return computed_data


def compute_gfp(
    epochs,
    conditions=None,
    shift_time=0,
    roi_name=None,
    method=None,
    event_times=None,
    output_path=None,
):
    """
    Compute GFP (Global Field Power) data for plotting.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    conditions : list
        List of condition names
    shift_time : float
        Time shift for plotting
    roi_name : str or None
        ROI name to filter channels
    method : callable
        Reduction function to apply (e.g., trim_mean80)
    event_times : dict
        Event times for reference
    output_path : str or Path
        Path to save the computed results as pickle
        
    Returns
    -------
    dict
        Dictionary containing all computed data for plotting
    """
    from ..viz.equipments import get_roi
    from ..viz.stats import compute_gfp as compute_gfp_stats
    
    logger.info("Starting GFP computations...")
    
    if conditions is None:
        conditions = list(epochs.event_id.keys())
    
    if method is None:
        from ..viz.reductions import trim_mean80
        method = trim_mean80
    
    this_times = (epochs.times + shift_time) * 1e3
    
    # Compute GFP for each condition
    gfp_data = []
    for condition in conditions:
        logger.info(f"Computing GFP for condition: {condition}")
        data = epochs[condition].get_data()
        
        # Apply ROI filtering if specified
        if roi_name is not False:
            if roi_name is not None:
                roi = get_roi(
                    config=epochs.info["description"], roi_name=roi_name
                )
                data = data[:, roi, :]
            else:
                roi = get_roi(
                    config=epochs.info["description"], roi_name="scalp"
                )
                data = data[:, roi, :]
        
        # Apply reduction method
        data = method(data, axis=0)
        
        # Compute GFP and confidence intervals
        gfp, ci1, ci2 = compute_gfp_stats(data)
        
        gfp_data.append({
            "condition": condition,
            "gfp": gfp,
            "ci1": ci1,
            "ci2": ci2,
        })
    
    # Compile all computed data
    computed_data = {
        "gfp_data": gfp_data,
        "this_times": this_times,
        "event_times": event_times,
        "shift_time": shift_time,
    }
    
    # Save to pickle if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving computed data to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(computed_data, f)
        logger.info("Computed data saved successfully")
    
    logger.info("GFP computations completed")
    return computed_data
