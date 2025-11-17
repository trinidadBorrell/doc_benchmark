# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import mne
import numpy as np
from mne.utils import logger

from .equipments import (
    get_roi,
    prepare_layout,
)
from .topos import plot_topomap_multi_cbar
from .utils import get_stat_colormap


def plot_cluster_test(
    computed_data_path=None,
    computed_data=None,
    sns_kwargs=None,
):
    """
    Plot cluster test analysis from pre-computed data.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
    sns_kwargs : dict
        Seaborn style kwargs
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure, or None if no clusters
    """
    import pickle
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    
    # Handle case where no clusters were found
    if computed_data is None:
        logger.info("No clusters found in computed data")
        return None
    
    # Extract pre-computed values
    n_clusters = computed_data["n_clusters"]
    cluster_data = computed_data["cluster_data"]
    epochs_info = computed_data["epochs_info"]
    labels = computed_data["labels"]
    shift_time = computed_data["shift_time"]
    event_times = computed_data["event_times"]
    
    # Ensure description is set to egi/256 for consistent montage across all topoplots
    if epochs_info.get("description") is None:
        epochs_info["description"] = "egi/256"
    
    # Regenerate sphere and outlines from epochs_info (they can't be pickled)
    from .equipments import prepare_layout
    sphere, outlines = prepare_layout("egi/256", info=epochs_info)
    
    # Extract 2D electrode positions (or recompute if not in pickle for backward compatibility)
    if "pos" in computed_data:
        pos = computed_data["pos"]
        logger.info("Using pre-computed electrode positions from pickle")
    else:
        logger.warning("Positions not in pickle - recomputing from epochs_info")
        from mne.viz.topomap import _prepare_topomap_plot
        _, pos, _, _, _, _, _ = _prepare_topomap_plot(
            epochs_info, "eeg", sphere=sphere
        )
    
    # Create figure
    fig_cluster = plt.figure(figsize=(12, 3 * n_clusters))
    gs = gridspec.GridSpec(n_clusters, 2, width_ratios=[1, 3])
    
    # Plot each cluster
    for i in range(n_clusters):
        cluster = cluster_data[i]
        f_map = cluster["f_map"]
        mask = cluster["mask"]
        sig_times = cluster["sig_times"]
        evokeds = cluster["evokeds"]
        evokeds_stderr = cluster["evokeds_stderr"]
        sig_mask = cluster["sig_mask"]
        p_value = cluster["p_value"]
        
        # Plot Topo
        ax_topo = plt.subplot(gs[2 * i])
        plot_topomap_multi_cbar(
            f_map,
            pos=pos,  # Use extracted 2D positions, not epochs_info
            outlines=outlines,
            colorbar=False,
            mask=mask,
            ax=ax_topo,
            cmap="Reds",
            sphere=sphere,
        )
        image = ax_topo.images[0]
        divider = make_axes_locatable(ax_topo)
        ax_colorbar = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.1f} - {:0.1f} ms)".format(*sig_times[[0, -1]])
        )

        # Plot evokeds
        ax_evoked = plt.subplot(gs[2 * i + 1])
        plot_evoked(
            evokeds,
            std_errs=evokeds_stderr,
            colors=["r", "b"],
            labels=labels,
            ax=ax_evoked,
            shift_time=shift_time,
            sig_mask=sig_mask,
            event_times=event_times,
            sns_kwargs=sns_kwargs,
        )

        ax_evoked.axvline(0, color=".5", lw=0.5)
        handles, legend_labels = ax_evoked.get_legend_handles_labels()
        sig = mpl.patches.Patch(color="r", alpha=0.5, label=f"SEM {legend_labels[0]}")
        handles.append(sig)
        sig = mpl.patches.Patch(color="b", alpha=0.5, label=f"SEM {legend_labels[1]}")
        handles.append(sig)
        sig = mpl.patches.Patch(
            color="orange",
            label=f"Smallest found cluster p-value (p={p_value:.2})",
        )
        handles.append(sig)
        ax_evoked.legend(handles=handles, loc="upper left")

    plt.subplots_adjust(left=0.04, right=0.97)
    return fig_cluster


def plot_cnv(
    computed_data_path=None,
    computed_data=None,
    color="teal",
    event_times=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    """
    Plot CNV analysis from pre-computed data.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
    color : str
        Color for plotting
    event_times : dict
        Event times for plotting
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
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(14, 8))

    # Load computed data from pickle if path provided
    if computed_data is None:
        if computed_data_path is None:
            raise ValueError("Either computed_data_path or computed_data must be provided")
        logger.info(f"Loading computed data from {computed_data_path}")
        with open(computed_data_path, "rb") as f:
            computed_data = pickle.load(f)
    
    # Extract all pre-computed values
    topo_filtered = computed_data["topo_filtered"]
    p_topo = computed_data["p_topo"]
    obs = computed_data["obs"]
    p_clusters = computed_data["p_clusters"]
    cluster_mask = computed_data["cluster_mask"]
    n_clusters = computed_data["n_clusters"]
    mask_filtered = computed_data["mask_filtered"]
    mask = computed_data["mask"]
    pos = computed_data["pos"]
    roi_data = computed_data["roi_data"]
    ch_info = computed_data["ch_info"]
    
    # Regenerate sphere and outlines from ch_info (they can't be pickled)
    from .equipments import prepare_layout
    sphere, outlines_pickled = prepare_layout(ch_info["description"], info=ch_info)
    
    # CRITICAL: Regenerate outlines with patch function for proper boundary clipping
    from .equipments import prepare_layout
    equipment_config = ch_info.get('description', 'egi/256') if hasattr(ch_info, 'get') else 'egi/256'
    try:
        sphere_regen, outlines = prepare_layout(equipment_config, info=ch_info)
        if sphere is None:
            sphere = sphere_regen
    except Exception as e:
        logger.warning(f"Could not regenerate CNV outlines: {e}")
        outlines = outlines_pickled
    stat_psig = computed_data["stat_psig"]
    stat_logpsig = computed_data["stat_logpsig"]
    stat_vmin = computed_data["stat_vmin"]
    stat_vmax = computed_data["stat_vmax"]
    stat_pvmin = computed_data["stat_pvmin"]
    stat_pvmax = computed_data["stat_pvmax"]
    vminmax = computed_data["vminmax"]
    ch_info = computed_data["ch_info"]
    
    # Get stat colormap
    stat_cmap = get_stat_colormap(stat_logpsig, stat_vmin, stat_vmax)
    
    # Create figure and grid
    fig = plt.figure(figsize=fig_kwargs.get("figsize", (14, 8)))
    gs = gridspec.GridSpec(1 + len(roi_data), 6)

    # Setup axes based on cluster presence
    ax_topo_cluster = None
    cmap_cl = None
    if n_clusters == 0:
        logger.info("No cluster found")
        ax_topo = plt.subplot(gs[0, 1:3])
        ax_topo_stat = plt.subplot(gs[0, 3:5])
    else:
        if n_clusters == 1 and np.sum(p_clusters < stat_psig) == 0:
            cmap_cl = "Greys"
        else:
            cmap_cl = "Reds"
        ax_topo = plt.subplot(gs[0, 0:2])
        ax_topo_stat = plt.subplot(gs[0, 2:4])
        ax_topo_cluster = plt.subplot(gs[0, 4:])

    mask_params = dict(
        marker="+",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=0,
        markersize=1,
    )

    # Plot main CNV topomap
    im, _ = mne.viz.plot_topomap(
        topo_filtered,
        pos,
        vlim=(-vminmax, vminmax),
        axes=ax_topo,
        cmap="RdBu_r",
        image_interp="cubic",
        contours=0,
        extrapolate="local",
        mask=mask_filtered,
        mask_params=mask_params,
        sensors=False,
        outlines=outlines,
        sphere=sphere,
    )
    ax_topo.set_title("CNV", fontsize=10)

    # Plot statistical topomap
    plot_topomap_multi_cbar(
        p_topo,
        pos=ch_info,
        ax=ax_topo_stat,
        title="-log10(p)",
        cmap=stat_cmap,
        vmin=stat_vmin,
        vmax=stat_vmax,
        outlines=outlines,
        mask=mask,
        mask_params=mask_params,
        sensors=False,
        unit="-log10(p)",
        sphere=sphere,
    )

    # Configure colorbars
    try:
        if len(fig.axes) > 0 and len(fig.axes[0].images) > 0:
            cbar = fig.axes[0].images[0].colorbar
            if cbar is not None:
                cbar.set_ticks([-vminmax, 0, vminmax])
    except (IndexError, AttributeError):
        pass

    try:
        if len(fig.axes) > 1 and len(fig.axes[1].images) > 0:
            cbar = fig.axes[1].images[0].colorbar
            if cbar is not None:
                cbar.set_ticks([stat_vmin, stat_logpsig, stat_vmax])
                cbar.set_ticklabels(
                    [f"p={stat_pvmin}", f"p={stat_psig}", f"p={stat_pvmax}"]
                )
    except (IndexError, AttributeError):
        pass

    # Plot cluster if present
    if n_clusters != 0 and ax_topo_cluster is not None:
        plot_topomap_multi_cbar(
            abs(obs),
            pos=ch_info,
            outlines=outlines,
            colorbar=False,
            title="Cluster",
            mask=cluster_mask,
            ax=ax_topo_cluster,
            cmap=cmap_cl,
            sphere=sphere,
        )
        image = ax_topo_cluster.images[0]
        divider = make_axes_locatable(ax_topo_cluster)
        ax_colorbar = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(image, cax=ax_colorbar)
        ax_colorbar.set_title(r"$\|T\|$")
        ax_topo_cluster.set_xlabel(f"\nCluster\np-value={p_clusters[0]:0.4f}")

    # Plot ROI-specific data
    for i, roi_info in enumerate(roi_data):
        roi_name = roi_info["roi_name"]
        roi_cnv = roi_info["roi_cnv"]
        p = roi_info["p_value"]
        mean_slope = roi_info["mean_slope"]
        cnv_line = roi_info["cnv_line"]
        evoked = roi_info["evoked"]
        evoked_stderr = roi_info["evoked_stderr"]
        
        if p < 0.05:
            p_color = color
            if p < 1e-4:
                p_label = "p < 0.0001"
            else:
                p_label = f"p = {round(p, 4)}"
        else:
            p_color = "silver"
            p_label = f"p = {round(p, 4)}"

        # Plot CNV line (skip evoked plotting if not available)
        ax_roi = plt.subplot(gs[1 + i, 0:4])
        if evoked is not None:
            plot_evoked(
                [evoked],
                std_errs=[evoked_stderr] if evoked_stderr is not None else None,
                colors=[color],
                shift_time=0,
                ax=ax_roi,
                event_times=event_times,
                sns_kwargs=sns_kwargs,
            )
        ax_roi.set_title(f"Around {roi_name}", pad=20)
        ax_roi.plot([0, 600], cnv_line, color=p_color, ls="--", linewidth=2)
        ax_roi.axhline(0, color=".5", lw=0.5, ls="--")
        ax_roi.set_xlim([0, 600])
        ax_roi.set_xlabel('Time (ms)')
        ax_roi.set_ylabel('Amplitude (µV)')

        lab_erp = mpl.patches.Patch(
            color=color, alpha=0.2, label=r"$ERP\/(\mu \pm SEM)$"
        )
        lab_slope = mpl.lines.Line2D(
            [0], [0], color=p_color, ls="--", label="CNV slope"
        )
        ax_roi.legend(handles=[lab_erp, lab_slope], loc="upper left")

        # Plot histogram
        ax_hist = plt.subplot(gs[1 + i, 4:6])
        sns.distplot(roi_cnv, color=p_color, ax=ax_hist)
        ax_hist.axvline(mean_slope, color=p_color)
        ax_hist.axvline(0, color="black", lw=0.8, ls="--")
        ax_hist.set_title(f"CNV slope at {roi_name}", pad=20)
        ax_hist.set_xlabel(r"$CNV\ Slope\ (\mu{V})$")
        ax_hist.set_ylabel("Distribution")
        lab_slope = mpl.lines.Line2D(
            [0], [0], color=p_color, label=f"Slope = {mean_slope:.2f}"
        )
        lab_p = mpl.patches.Patch(color=p_color, label=p_label)
        ax_hist.legend(handles=[lab_slope, lab_p], loc="upper left")
    
    plt.tight_layout()
    return fig


def plot_gfp(
    computed_data_path=None,
    computed_data=None,
    colors=None,
    linestyles=None,
    labels=None,
    ax=None,
    sig_mask=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    """
    Plot GFP from pre-computed data.
    
    Parameters
    ----------
    computed_data_path : str or Path
        Path to pickle file containing computed data
    computed_data : dict
        Pre-computed data dictionary (if not loading from file)
    colors : list
        Colors for each condition
    linestyles : list
        Line styles for each condition
    labels : list
        Labels for each condition
    ax : matplotlib.axes.Axes
        Axes to plot on
    sig_mask : array
        Significance mask
    fig_kwargs : dict
        Figure kwargs
    sns_kwargs : dict
        Seaborn style kwargs
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure
    """
    import pickle
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    
    # Load computed data from pickle if path provided
    if computed_data is None:
        if computed_data_path is None:
            raise ValueError("Either computed_data_path or computed_data must be provided")
        logger.info(f"Loading computed data from {computed_data_path}")
        with open(computed_data_path, "rb") as f:
            computed_data = pickle.load(f)
    
    # Extract pre-computed values
    gfp_data = computed_data["gfp_data"]
    this_times = computed_data["this_times"]
    event_times = computed_data["event_times"]
    shift_time = computed_data["shift_time"]
    
    # Set up colors, linestyles, and labels
    if colors is None:
        colors = [None for _ in gfp_data]
    if linestyles is None:
        linestyles = ["-" for _ in gfp_data]
    if labels is None:
        labels = [None for _ in gfp_data]
    
    # Plot each condition
    for gfp_info, color, ls, label in zip(gfp_data, colors, linestyles, labels):
        if label is None:
            label = gfp_info["condition"]
        
        gfp = gfp_info["gfp"]
        ci1 = gfp_info["ci1"]
        ci2 = gfp_info["ci2"]
        
        lines = ax.plot(
            this_times, gfp * 1e6, color=color, linestyle=ls, label=label
        )

        ax.fill_between(
            this_times,
            y1=ci1 * 1e6,
            y2=ci2 * 1e6,
            color=lines[0].get_color(),
            alpha=0.5,
        )
    
    if sig_mask is not None:
        for i in np.where(sig_mask)[0]:
            ax.axvline(this_times[i], alpha=0.5, color="orange")
    
    handles, legend_labels = ax.get_legend_handles_labels()
    for color in np.unique(colors):
        if color is not None:
            sig = mpl.patches.Patch(color=color, alpha=0.5, label=r"$\chi^{2}$ CI")
            handles.append(sig)
    
    if event_times is not None:
        times = this_times * 1e3
        xticks = list(event_times.keys())
        xticks.insert(0, times[0])
        xticks.extend([times[-1]])
        ax.set_xticks(np.unique(xticks))
        for t, s in event_times.items():
            t += shift_time * 1e3
            ax.axvline(t, color="black", lw=0.5, ls="--")
            ax.text(x=t, y=ax.get_ylim()[1], s=s, horizontalalignment="center")
    
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r"Evoked Response ($\mu{V}$)")
    ax.set_xlabel("Time (ms)")
    ax.legend(handles=handles, loc="upper left")
    return fig


def plot_evoked(
    evokeds,
    std_errs=None,
    colors=None,
    linestyles=None,
    shift_time=0,
    labels=None,
    ax=None,
    event_times=None,
    sig_mask=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    
    # Ensure evokeds is a list
    if not isinstance(evokeds, (list, tuple)):
        evokeds = [evokeds]
    
    if std_errs is None:
        std_errs = [None for x in evokeds]
    if colors is None:
        colors = [None for x in evokeds]
    if linestyles is None:
        linestyles = ["-" for x in evokeds]
    if labels is None:
        labels = [f"{i}" for i in range(len(evokeds))]
    max_val = -np.inf
    min_val = np.inf
    this_times = None
    for evoked, color, ls, label, std_err in zip(
        evokeds, colors, linestyles, labels, std_errs
    ):
        if label is None:
            label = f"{evoked}"
        this_times = (evoked.times + shift_time) * 1e3
        data = evoked.data
        # Average across channels if multi-channel
        if data.ndim > 1:
            data = data.mean(axis=0)
        data = np.squeeze(data)
        lines = ax.plot(
            this_times, data * 1e6, color=color, linestyle=ls, label=label
        )
        this_max_val = np.max(data)
        this_min_val = np.min(data)
        if std_err is not None:
            stderr_data = std_err.data
            # Average across channels if multi-channel
            if stderr_data.ndim > 1:
                stderr_data = stderr_data.mean(axis=0)
            stderr_data = np.squeeze(stderr_data)
            ax.fill_between(
                this_times,
                y1=(data + stderr_data) * 1e6,
                y2=(data - stderr_data) * 1e6,
                color=lines[0].get_color(),
                alpha=0.2,
            )
            this_max_val += np.max(std_err.data)
            this_min_val -= np.max(std_err.data)
        max_val = max(this_max_val, max_val)
        min_val = min(this_min_val, min_val)

    if sig_mask is not None:
        for i in np.where(sig_mask)[0]:
            ax.axvline(this_times[i], alpha=0.3, color="orange")
    max_val = np.ceil(max_val * 1e6)
    min_val = np.floor(min_val * 1e6)
    step = 0.5
    if abs(max_val) + abs(min_val) >= 5:
        step = 1.0

    if abs(max_val) + abs(min_val) < step:
        step = (abs(max_val) + abs(min_val)) / 4

    # Ensure step is not too small to prevent infinite loops
    step = max(step, 0.1)

    if event_times is not None:
        # times = this_times
        # xticks = list(event_times.keys())
        # xticks.insert(0, times[0])
        # xticks.extend([times[-1]])
        # ax.set_xticks(np.unique(xticks))
        for t, s in event_times.items():
            t += shift_time * 1e3
            ax.axvline(t, color="black", lw=0.5, ls="--")
            ax.text(x=t, y=ax.get_ylim()[1], s=s, horizontalalignment="center")

    ax.set_yticks(np.arange(min_val, max_val + 0.1, step))
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r"Evoked Response ($\mu V$)")
    ax.set_xlabel("Time (ms)")
    if fig is not None:
        ax.legend(loc="upper left")
    return fig


def plot_butterfly(
    evoked,
    color=None,
    linestyle=None,
    shift_time=0,
    labels=None,
    ax=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    if linestyle is None:
        linestyle = "-"
    if color is None:
        color = "b"

    this_times = (evoked.times + shift_time) * 1e3
    data = np.squeeze(evoked.data).T
    ax.plot(this_times, data * 1e6, color=color, linestyle=linestyle, lw=0.1)
    max_val = np.max(data)
    min_val = np.min(data)

    max_val = np.ceil(max_val * 1e6)
    min_val = np.floor(min_val * 1e6)
    step = 2
    if abs(max_val) + abs(min_val) >= 10:
        step = 5

    # Ensure step is not too small to prevent infinite loops
    step = max(step, 0.1)

    ax.set_yticks(np.arange(min_val, max_val + 0.1, step))
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r"Evoked Response ($\mu V$)")
    ax.set_xlabel("Time (ms)")
    return fig


def plot_evoked_topomap(evoked, **kwargs):
    t_evoked = evoked.copy()
    
    # Ensure description is set to egi/256 for consistent montage
    if t_evoked.info.get("description") is None:
        t_evoked.info["description"] = "egi/256"
    
    scalp_roi = get_roi(config=t_evoked.info["description"], roi_name="scalp")
    non_scalp = get_roi(
        config=t_evoked.info["description"], roi_name="nonscalp"
    )
    if non_scalp is not None and len(non_scalp) > 0:
        t_evoked.data[non_scalp, :] = 0.0

    sphere, outlines = prepare_layout(
        t_evoked.info["description"], info=t_evoked.info
    )
    nchans, ntimes = t_evoked.data.shape
    mask = np.in1d(np.arange(nchans), scalp_roi)
    mask = np.tile(mask[:, None], (1, ntimes))
    mask_params = dict(
        marker="+",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=0,
        markersize=1,
    )

    kwargs["mask"] = mask
    kwargs["mask_params"] = mask_params
    kwargs["sensors"] = False
    kwargs["outlines"] = outlines
    kwargs["sphere"] = sphere
    sns_kwargs = kwargs.get("sns_kwargs", None)
    if sns_kwargs is not None:
        import seaborn as sns

        sns.set(**sns_kwargs)
    kwargs.pop("sns_kwargs", None)  # Remove if present, ignore if not

    return mne.viz.plot_evoked_topomap(t_evoked, **kwargs)


def plot_ttest(
    p_vals,
    labels,
    ticks,
    times,
    n_times_thresh,
    n_chans_thresh,
    colors=None,
    sns_kwargs=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    st = int(np.floor((n_times_thresh - 1) / 2))
    end = int(np.ceil((n_times_thresh - 1) / 2))
    this_times = times[st:-end]
    fig_p_vals, ax_pvals = plt.subplots(1, 1, figsize=(12, 8))
    if colors is None:
        colors = [None] * len(p_vals)
    for p_val, label, col in zip(p_vals, labels, colors):
        values = np.zeros(len(this_times))
        for tick in ticks:
            mask = p_val < tick
            time_mask = [
                np.convolve(
                    mask[i, :], np.ones((n_times_thresh,)), mode="valid"
                )
                == n_times_thresh
                for i in range(p_val.shape[0])
            ]
            this_count = np.sum(np.c_[time_mask], axis=0)
            values[this_count > n_chans_thresh] += 1
        ax_pvals.plot(this_times * 1e3, values, color=col)

    ax_pvals.set_xlabel("Time (ms)")
    ax_pvals.set_ylabel("p value")
    ax_pvals.set_xlim(this_times[[0, -1]] * 1e3)
    ax_pvals.set_ylim([0, len(ticks) + 1])
    ax_pvals.set_yticklabels([0] + ticks)
    ax_pvals.legend(ax_pvals.lines, labels, loc="upper left")

    return fig_p_vals
