# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import matplotlib.pyplot as plt
import mne
import numpy as np

from .equipments import get_roi, prepare_layout
from .utils import _map_marker_to_text


def plot_markers_topos(
    computed_data_path=None,
    computed_data=None,
    units=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    """
    Plot markers topographies from pre-computed data.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
    units : list
        Units for each marker
    fig_kwargs : dict
        Figure kwargs
    sns_kwargs : dict
        Seaborn style kwargs
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mne.utils import logger

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    # Load computed data from pickle if path provided
    if computed_data is None:
        if computed_data_path is None:
            raise ValueError("Either computed_data_path or computed_data must be provided")
        logger.info(f"Loading computed data from {computed_data_path}")
        with open(computed_data_path, "rb") as f:
            computed_data = pickle.load(f)
    
    # Extract pre-computed values
    topos = computed_data["topos"]
    marker_names = computed_data["marker_names"]
    pos = computed_data["pos"]
    mask = computed_data["mask"]
    vmin = computed_data["vmin"]
    vmax = computed_data["vmax"]
    same_scale = computed_data["same_scale"]
    ch_info = computed_data["ch_info"]
    
    # Regenerate sphere and outlines from ch_info (they can't be pickled)
    from .equipments import prepare_layout
    sphere, outlines_pickled = prepare_layout(ch_info["description"], info=ch_info)
    markers = computed_data["markers"]
    
    # CRITICAL: Regenerate full outlines with patch function for proper boundary clipping
    # The pickled outlines are missing the 'patch' function which is needed for clipping
    from .equipments import prepare_layout
    
    # Get ch_info from first marker to regenerate outlines properly
    first_marker = list(markers.values())[0] if markers else None
    equipment_config = 'egi/256'  # Default
    ch_info = None
    
    if first_marker and hasattr(first_marker, 'ch_info_'):
        ch_info = first_marker.ch_info_
        if hasattr(ch_info, 'get'):
            equipment_config = ch_info.get('description', 'egi/256')
        elif hasattr(ch_info, '__getitem__'):
            try:
                equipment_config = ch_info['description']
            except (KeyError, TypeError):
                pass
    
    # Regenerate full outlines with patch function for proper clipping
    try:
        sphere_regen, outlines_full = prepare_layout(equipment_config, info=ch_info)
        outlines = outlines_full  # This has the patch function
        # Update sphere if it was None
        if sphere is None:
            sphere = sphere_regen
        logger.info(f"Successfully regenerated outlines with patch for {equipment_config}")
    except Exception as e:
        logger.warning(f"Could not regenerate outlines with patch: {e}")
        # Fallback to pickled outlines (will have clipping issues)
        outlines = outlines_pickled
    
    # Setup plotting
    n_axes = len(topos)
    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(3 * n_axes if n_axes > 1 else 4, 4))
    if units is None:
        units = [r""] * n_axes
    
    fig, axes = plt.subplots(1, n_axes, **fig_kwargs)
    if n_axes == 1:
        axes = [axes]
    
    mask_params = dict(
        marker="+",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=0,
        markersize=1,
    )
    
    # Plot each marker
    for ax, topo, name, unit in zip(axes, topos, marker_names, units):
        # Calculate individual scale if not using same_scale
        if same_scale is False:
            topo_vmin = np.nanmin(topo)
            topo_vmax = np.nanmax(topo)
        else:
            topo_vmin = vmin
            topo_vmax = vmax
        
        # Handle NaN values
        nan_idx = np.isnan(topo)
        
        # Get marker for text mapping
        marker_obj = markers.get(name)
        title = _map_marker_to_text(marker_obj) if marker_obj else name
        
        plot_topomap_multi_cbar(
            topo[~nan_idx],
            pos[~nan_idx],
            ax,
            title,
            cmap="viridis",
            outlines=outlines,
            mask=mask,
            mask_params=mask_params,
            sensors=False,
            unit=unit,
            vmin=topo_vmin,
            vmax=topo_vmax,
            sphere=sphere,  # Add sphere for proper boundary clipping
        )
    
    plt.tight_layout()
    return fig


def plot_marker_topo(
    computed_data_path=None,
    computed_data=None,
    unit=None,
    fig_kwargs=None,
):
    """
    Plot single marker topography from pre-computed data.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
    unit : str
        Unit for the marker
    fig_kwargs : dict
        Figure kwargs
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    return plot_markers_topos(
        computed_data_path=computed_data_path,
        computed_data=computed_data,
        units=[unit] if unit else None,
        fig_kwargs=fig_kwargs,
    )


# Update MNE one with mask, mask_params and sensors parameters
def plot_topomap_multi_cbar(
    data,
    pos,
    ax,
    title=None,
    unit=None,
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
    colorbar=True,
    outlines="head",
    cbar_format="%0.3f",
    contours=0,
    extrapolate="local",
    mask=None,
    mask_params=None,
    sensors=True,
    sphere=None,
):
    """Low level plot topography for a single marker

    Parameters
    ----------
    data : numpy.ndarray of float, shape (n_sensors,)
        The data show.
    pos : numpy.ndarray of float, shape (n_sensors, 2)
        The positions of the sensors.
    ax : instance of Axis
        The axes to plot on.
    title : str | None
        The axes title to show. Defaults to None (no title will be shown).
    cbar_format : str
        The colorbar format. Defaults to '%0.3f'
    """
    mne.viz.topomap._hide_frame(ax)
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    cmap = mne.viz.utils._setup_cmap(cmap)
    if title is not None:
        ax.set_title(title, fontsize=10)
    im, _ = mne.viz.plot_topomap(
        data,
        pos,
        vlim=(vmin, vmax),
        axes=ax,
        cmap=cmap[0],
        image_interp="cubic",  # bilinear changed to cubic for newer MNE
        contours=contours,
        outlines=outlines,
        show=False,
        mask=mask,
        mask_params=mask_params,
        sensors=sensors,
        extrapolate=extrapolate,
        sphere=sphere,
    )

    if colorbar is True:
        cbar, cax = mne.viz.topomap._add_colorbar(ax, im, cmap)
        cbar.set_ticks((vmin, vmax))
        if unit is not None:
            cbar.ax.set_title(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)


def plot_topo_equipment(
    topo,
    equipment,
    ch_info=None,
    ax=None,
    symmetric_scale=False,
    unit="",
    label="",
    cmap="viridis",
):
    scalp_roi = get_roi(config=equipment, roi_name="scalp")
    non_scalp = get_roi(config=equipment, roi_name="nonscalp")

    sphere, outlines, info = prepare_layout(
        equipment, info=ch_info, return_info=True
    )
    _, pos, _, _, _, _, _ = mne.viz.topomap._prepare_topomap_plot(
        ch_info, "eeg", sphere=sphere
    )

    mask = np.in1d(np.arange(len(pos)), scalp_roi)
    mask_params = dict(
        marker="+",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=0,
        markersize=1,
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    vmin = np.min(topo[scalp_roi])
    vmax = np.max(topo[scalp_roi])
    neutral = vmin

    if symmetric_scale is True:
        absmax = max(abs(vmin), abs(vmax))
        vmin = -absmax
        vmax = absmax
        neutral = 0.0
        cmap = "RdBu_r"

    if non_scalp is not None:
        topo[non_scalp] = neutral

    plot_topomap_multi_cbar(
        topo,
        pos,
        ax,
        label,
        cmap=cmap,
        outlines=outlines,
        mask=mask,
        mask_params=mask_params,
        sensors=False,
        unit=unit,
        vmin=vmin,
        vmax=vmax,
    )

    return fig, ax


def plot_topos_equipments(
    topos,
    names,
    equipment,
    ch_info=None,
    same_scale=True,
    symmetric_scale=False,
    units=None,
    is_stat=False,
    ncols=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .utils import get_stat_colormap

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    if units is None:
        units = ""
    if not isinstance(units, list):
        units = [units] * len(names)

    scalp_roi = get_roi(config=equipment, roi_name="scalp")
    non_scalp = get_roi(config=equipment, roi_name="nonscalp")
    vmin, vmax = 0, 0

    sphere, outlines, ch_info = prepare_layout(
        equipment, info=ch_info, return_info=True
    )
    _, pos, _, _, _, _, _ = mne.viz.topomap._prepare_topomap_plot(
        ch_info, "eeg", sphere=sphere
    )

    mask = np.in1d(np.arange(len(pos)), scalp_roi)
    mask_params = dict(
        marker="+",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=0,
        markersize=1,
    )

    if same_scale is True:
        vmin = np.nanmin(np.c_[topos][:, scalp_roi])
        vmax = np.nanmax(np.c_[topos][:, scalp_roi])

    cmap = "viridis"
    if symmetric_scale is True:
        cmap = "RdBu_r"
        if same_scale is True:
            vabsmax = max(abs(vmin), abs(vmax))
            vmin = -vabsmax
            vmax = vabsmax

    n_axes = len(names)
    if ncols is None:
        ncols = n_axes
        nrows = 1
    else:
        nrows = int(np.ceil(n_axes / ncols))

    fig = None
    if fig_kwargs is None:
        fig_kwargs = dict(
            figsize=(
                3 * ncols if ncols > 1 else 4,
                3 * nrows if nrows > 1 else 4,
            )
        )
    fig, axes = plt.subplots(nrows, ncols, **fig_kwargs)
    if n_axes == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, name, unit, topo in zip(axes, names, units, topos):
        if same_scale is False:
            vmin = np.nanmin(topo[scalp_roi])
            vmax = np.nanmax(topo[scalp_roi])
            if symmetric_scale is True:
                vabsmax = max(abs(vmin), abs(vmax))
                vmin = -vabsmax
                vmax = vabsmax
        if non_scalp is not None:
            if is_stat is True or symmetric_scale is True:
                topo[non_scalp] = 0
            else:
                topo[non_scalp] = vmin

        nan_idx = np.isnan(topo)

        if is_stat is True:
            vmin = np.log10(1)
            vmax = -np.log10(1e-5)
            psig = -np.log10(0.05)
            cmap = get_stat_colormap(psig, vmin, vmax)

            unit = r"$-log_{10}(p)$"

        plot_topomap_multi_cbar(
            topo[~nan_idx],
            pos[~nan_idx],
            ax,
            name,
            cmap=cmap,
            outlines=outlines,
            mask=mask,
            mask_params=mask_params,
            sensors=False,
            unit=unit,
            vmin=vmin,
            vmax=vmax,
            extrapolate="local",
        )
    plt.tight_layout()
    return fig, axes
