#!/usr/bin/env python3
"""
Next-ICM Report - PHASE 2: Generate Plots

This script generates the HTML report from pre-computed data.
This runs instantly since all heavy computations were done in Phase 1.

Usage:
    python generate_plots.py [--skip-clustering] [--subject_id SUBJECT_ID] \
                             [--h5_file H5_FILE] [--fif_file FIF_FILE] \
                             [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
import mne

# Import data loader and viz modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from matplotlib import gridspec

from report_modules.data_io import ReportDataLoader
from report_modules.viz import (
    plot_gfp,
    plot_cnv,
    plot_evoked_topomap,
    plot_ttest,
    get_stat_colormap,
    get_contrast,
    plot_evoked,
    plot_cluster_test,
    plot_markers_topos,
    plot_marker_topo,
    render_prediction_summary,
)
from report_modules.viz.equipments import prepare_layout


def main():
    """Generate HTML report from pre-computed data"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PHASE 2: Generate Next-ICM report from pre-computed data"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering plots (must match what was used in compute phase)",
    )
    parser.add_argument(
        "--subject_id",
        default="AA048",
        help="Subject ID for report generation",
    )
    parser.add_argument(
        "--h5_file",
        default="./input/icm_complete_features.h5",
        help="Path to HDF5 file with features",
    )
    parser.add_argument(
        "--fif_file",
        default="./input/03_artifact_rejected_eeg.fif",
        help="Path to FIF file with epochs data",
    )
    parser.add_argument(
        "--data_dir",
        default="./tmp_computed_data",
        help="Directory containing pre-computed data (pickle files)",
    )
    parser.add_argument(
        "--output_dir",
        default="./reports",
        help="Output directory for HTML report",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if computed data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"❌ Computed data directory not found: {data_path}")
        logger.error("Please run compute_data.py first to generate the data.")
        sys.exit(1)

    # Generate plots from pre-computed data
    try:
        logger.info(f"Generating report for subject {args.subject_id}")
        logger.info(f"Loading data from: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Load data
        logger.info("="*60)
        logger.info("Loading data...")
        logger.info("="*60)
        
        hdf5_file = Path(args.h5_file)
        fif_file = Path(args.fif_file)
        
        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
        if not fif_file.exists():
            raise FileNotFoundError(f"FIF file not found: {fif_file}")
        
        logger.info(f"Loading HDF5 features from: {hdf5_file}")
        logger.info(f"Loading epochs from: {fif_file}")
        
        loader = ReportDataLoader(hdf5_file, fif_file, skip_preprocessing=True)
        report_data = loader.load_all_data()
        
        # Load epochs object
        epoch_info = report_data.get("epoch_info")
        if epoch_info is not None:
            epochs = epoch_info
            epochs.info["description"] = "egi/256"
            logger.info("Loaded epochs and set montage to 'egi/256'")
        else:
            logger.error("No epochs object found in data!")
            sys.exit(1)

        # ========== PHASE 2: GENERATE ALL PLOTS ==========
        logger.info("="*60)
        logger.info("PHASE 2: Generating all plots from computed data...")
        logger.info(f"Loading data from: {data_path}")
        logger.info("="*60)

        report = mne.Report(title=f"Task: Local-Global - subject: {args.subject_id}")

        # === PREPROCESSING PLOTS ===
        # Skip preprocessing plots if no preprocessing metadata available
        preprocessing_info = report_data.get("metadata", {}).get("preprocessing_info", {})
        if preprocessing_info:
            logger.info("Adding preprocessing plots...")
            _add_preprocessing_plots(report, epochs, report_data)
            gc.collect()
        else:
            logger.info("Skipping preprocessing plots (no preprocessing metadata available)")

        # === DIAGNOSTIC PLOTS ===
        logger.info("Adding diagnostic plots...")
        _add_diagnostic_plots(report, epochs, data_path)
        gc.collect()

        # === ERP PLOTS ===
        logger.info("Adding ERP plots...")
        _add_erp_plots(report, epochs, args.skip_clustering, data_path)
        gc.collect()

        # === CNV PLOTS ===
        logger.info("Adding CNV plots...")
        _add_cnv_plots(report, data_path)
        gc.collect()

        # === SPECTRAL PLOTS ===
        logger.info("Adding spectral plots...")
        _add_spectral_plots(report, data_path)
        gc.collect()

        # === CONNECTIVITY PLOTS ===
        logger.info("Adding connectivity plots...")
        _add_connectivity_plots(report, data_path)
        gc.collect()

        # === INFORMATION THEORY PLOTS ===
        logger.info("Adding information theory plots...")
        _add_information_theory_plots(report, data_path)
        gc.collect()

        # === PREDICTION ANALYSIS ===
        logger.info("Adding prediction analysis...")
        _add_prediction_plots(report, data_path)
        gc.collect()

        logger.info("="*60)
        logger.info("✅ PHASE 2 COMPLETE: All plots added to report")
        logger.info("="*60)

        # Save report
        output_path = Path(args.output_dir) / f"next_icm_report_{args.subject_id}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(str(output_path), overwrite=True)

        logger.info(f"✅ Next-ICM report generated: {output_path}")
        
        logger.info("✅ Report generated successfully!")
        print(f"\n{'='*80}")
        print(f"SUCCESS: Report generated at {output_path}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
        import traceback
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def _add_preprocessing_plots(report, epochs, report_data):
    """Add preprocessing plots"""
    import json
    
    # Get preprocessing info
    metadata = report_data.get("metadata", {})
    preprocessing_info = metadata.get("preprocessing_info", {})
    real_bad_channels = preprocessing_info.get("bad_channels_detected", [])
    bad_epochs = preprocessing_info.get("bad_epochs_detected", [])
    
    # Filter to only EEG channels
    eeg_bad_channels = [ch for ch in real_bad_channels if ch.startswith("E") and ch[1:].isdigit()]
    
    # 1. Bad Channels plot
    if epochs is not None:
        if epochs.info.get("description") is None:
            epochs.info["description"] = "egi/256"
        try:
            evoked = epochs.average()
            bad_idx = [i for i, ch in enumerate(evoked.ch_names) if ch in eeg_bad_channels]
            all_data = evoked.data * 1e6
            
            # Use automatic scale based on data percentiles for better contrast
            # Get good channels data only for more accurate range
            good_idx = [i for i, ch in enumerate(evoked.ch_names) if ch not in eeg_bad_channels]
            if len(good_idx) > 0:
                good_data = evoked.data[good_idx] * 1e6
                vmin_pct = np.percentile(good_data, 5)
                vmax_pct = np.percentile(good_data, 95)
            else:
                vmin_pct = np.percentile(all_data, 5)
                vmax_pct = np.percentile(all_data, 95)
            
            clim = {"eeg": [vmin_pct, vmax_pct]}
            fig, axes = plt.subplots(1, 1, figsize=(16, 8))
            plt.title("Bad Channels")
            fig_image = evoked.plot_image(picks="eeg", exclude=[], cmap="gray", clim=clim, axes=axes)
            times = evoked.times * 1e3
            
            for ax in fig_image.axes:
                if 1 <= len(ax.images) <= 2:
                    im = ax.images[0]
                    data = np.array(im.get_array())
                    if len(bad_idx) > 0:
                        mask = np.ones(data.shape, dtype=bool)
                        mask[bad_idx] = False
                        bad_data_masked = np.ma.masked_array(data, mask)
                        # Use automatic scale for bad channels based on their actual data range
                        bad_data_values = data[bad_idx]
                        vmin_bad = np.percentile(bad_data_values, 5)
                        vmax_bad = np.percentile(bad_data_values, 95)
                        im2 = ax.imshow(bad_data_masked, cmap="RdYlBu_r", interpolation="nearest",
                                      origin="lower", extent=[times[0], times[-1], 0, data.shape[0]],
                                      aspect="auto", vmin=vmin_bad, vmax=vmax_bad)
                    ax.set_xlabel("time (ms)")
            
            if len(fig_image.axes) > 1:
                fig_image.delaxes(fig_image.axes[1])
            
            divider = make_axes_locatable(fig_image.axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("regular", labelpad=-40, y=-0.05)
            cbar.ax.set_title(r"$\mu V$")
            
            if len(bad_idx) > 0:
                cax2 = divider.append_axes("right", size="5%", pad=0.5)
                cbar2 = plt.colorbar(im2, cax=cax2)
                cbar2.set_label("bads", labelpad=-43, y=-0.05)
                cbar2.ax.set_title(r"$\mu V$")
            
            new_axes = divider.append_axes("right", size="100%", pad=0.5)
            topo_mask = np.zeros(len(evoked.ch_names), dtype=bool)
            topo_mask[bad_idx] = True
            sphere, outlines = prepare_layout(epochs.info.get("description", "egi/256"), info=epochs.info)
            
            mne.viz.plot_topomap(np.ones(len(evoked.data)), pos=evoked.info, mask=topo_mask,
                               contours=0, cmap="gray", names=[" " * 10 + x for x in evoked.ch_names],
                               sensors=False, axes=new_axes, outlines=outlines,
                               mask_params={"markeredgecolor": "red", "marker": "x", "markersize": 7},
                               extrapolate="local", sphere=None)
            
            fig.subplots_adjust(wspace=0, right=1)
            fig.tight_layout()
            report.add_figure(fig, title="Bad Channels", section="Preprocessing")
            plt.close(fig)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not create bad channels plot: {e}")
    
    # 2. Bad Epochs
    good_epochs = len(epochs) if epochs is not None else 0
    total_epochs = good_epochs + len(bad_epochs)
    bad_percentage = (len(bad_epochs) / total_epochs * 100) if total_epochs > 0 else 0.0
    good_percentage = (good_epochs / total_epochs * 100) if total_epochs > 0 else 0.0
    
    html = f"""
    <div class="card">
        <div class="card-header"><h4>Epochs count</h4></div>
        <div class="card-body">
            <table class="table table-striped">
                <thead><tr><th>Reason</th><th># Epochs</th><th>Percentage</th></tr></thead>
                <tbody>
                    <tr><td>Artifacted</td><td>{len(bad_epochs)}</td><td>{bad_percentage:.2f} %</td></tr>
                    <tr><td>Good</td><td>{good_epochs}</td><td>{good_percentage:.2f} %</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    """
    report.add_html(html, title="Bad Epochs", section="Preprocessing")
    
    # 3. Preprocessing Summary
    preprocessing_json = json.dumps(preprocessing_info, indent=2, default=str)
    html = f"""
    <div style="margin-top: 30px;">
        <h4>Full Preprocessing Metadata</h4>
        <details>
            <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">Click to expand/collapse full preprocessing JSON</summary>
            <pre style="background-color: #f5f5f5; padding: 15px; border: 1px solid #ddd; border-radius: 5px; overflow-x: auto; max-height: 600px; font-size: 11px; font-family: monospace;">{preprocessing_json}</pre>
        </details>
    </div>
    """
    report.add_html(html, title="Preprocessing Summary", section="Preprocessing")


def _add_diagnostic_plots(report, epochs, data_path):
    """Add diagnostic plots"""
    # 1. Local Global Paradigm
    fig = mne.viz.plot_events(epochs.events, epochs.info["sfreq"], event_id=epochs.event_id)
    ax = fig.get_axes()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5))
    report.add_figure(fig, title="Local Global Paradigm", section="Diagnostic")
    plt.close(fig)
    
    # 2. GFP plots
    gfp_path = Path(data_path) / "diagnostic_gfp.pkl"
    if gfp_path.exists():
        fig = plot_gfp(computed_data_path=gfp_path, colors=["b", "b", "r", "r"],
                      linestyles=["-", "--", "-", "--"], fig_kwargs={"figsize": (12, 6)},
                      sns_kwargs={"style": "darkgrid"})
        report.add_figure(fig, title="All Blocks: Global Field Power", section="Diagnostic")
        plt.close(fig)


def _add_cnv_plots(report, data_path):
    """Add CNV plots"""
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}
    cnv_path = Path(data_path) / "cnv_computed_data.pkl"
    if cnv_path.exists():
        fig_cnv = plot_cnv(computed_data_path=cnv_path, event_times=event_times, sns_kwargs={"style": "white"})
        report.add_figure(fig_cnv, title="Contingent Negative Variation", section="CNV")
        plt.close(fig_cnv)


def _add_spectral_plots(report, data_path):
    """Add spectral plots"""
    # 1. Normalized spectral bands
    bands_path = Path(data_path) / "spectral_bands_normalized.pkl"
    if bands_path.exists():
        fig = plot_markers_topos(computed_data_path=bands_path)
        report.add_figure(fig, title="Spectral Power Bands Normalized (Delta, Theta, Alpha, Beta, Gamma)", section="Spectral")
        plt.close(fig)
    
    # 2. Absolute power
    absolute_path = Path(data_path) / "spectral_absolute_power.pkl"
    if absolute_path.exists():
        fig = plot_markers_topos(computed_data_path=absolute_path)
        report.add_figure(fig, title="Absolute Spectral Power (Log Scale) - Delta, Theta, Alpha, Beta, Gamma", section="Spectral")
        plt.close(fig)
    
    # 3. Spectral summaries
    summaries_path = Path(data_path) / "spectral_summaries.pkl"
    if summaries_path.exists():
        fig = plot_markers_topos(computed_data_path=summaries_path)
        report.add_figure(fig, title="Spectral Summaries (SE, MSF, SEF90, SEF95)", section="Spectral")
        plt.close(fig)


def _add_connectivity_plots(report, data_path):
    """Add connectivity plots"""
    # 1. WSMI topoplots
    wsmi_path = Path(data_path) / "wsmi_bands_topo.pkl"
    if wsmi_path.exists():
        fig = plot_markers_topos(computed_data_path=wsmi_path)
        report.add_figure(fig, title="WSMI Connectivity per Frequency Band", section="Connectivity")
        plt.close(fig)
    
    # 2. Mutual information
    mi_path = Path(data_path) / "mutual_info_topo.pkl"
    if mi_path.exists():
        fig = plot_marker_topo(computed_data_path=mi_path)
        report.add_figure(fig, title="Connectivity: Mutual Information Topography", section="Information Theory")
        plt.close(fig)


def _add_information_theory_plots(report, data_path):
    """Add information theory plots"""
    # 1. Permutation Entropy
    pe_path = Path(data_path) / "permutation_entropy_bands.pkl"
    if pe_path.exists():
        fig = plot_markers_topos(computed_data_path=pe_path)
        report.add_figure(fig, title="Permutation Entropy per Frequency Band", section="Information Theory")
        plt.close(fig)
    
    # 2. Kolmogorov complexity
    kc_path = Path(data_path) / "kolmogorov_complexity.pkl"
    if kc_path.exists():
        fig = plot_markers_topos(computed_data_path=kc_path)
        report.add_figure(fig, title="Kolmogorov Complexity", section="Information Theory")
        plt.close(fig)
    
    # 3. Generic per-channel measures
    for measure_name in ["kolmogorov_complexity", "permutation_entropy"]:
        measure_path = Path(data_path) / f"info_theory_{measure_name}.pkl"
        if measure_path.exists():
            fig = plot_marker_topo(computed_data_path=measure_path)
            report.add_figure(fig, title=f"Information Theory: {measure_name.upper()}", section="Information Theory")
            plt.close(fig)


def _add_erp_plots(report, epochs, skip_clustering, data_path):
    """Add ERP plots"""
    logger = logging.getLogger(__name__)
    
    local_labels = ["Local Deviant", "Local Standard"]
    global_labels = ["Global Deviant", "Global Standard"]
    time_shift = -0.6
    plot_times = np.arange(0.64, 1.336, 0.02) + time_shift
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}
    stat_psig = 0.05
    stat_logpsig = -np.log10(stat_psig)
    
    # LOCAL EFFECT
    local_gfp_path = Path(data_path) / "local_effect_gfp.pkl"
    fig_gfp = plot_gfp(computed_data_path=local_gfp_path, colors=["r", "b"], labels=local_labels,
                       fig_kwargs=dict(figsize=(12, 6)), sns_kwargs=dict(style="darkgrid"))
    
    local_contrast_path = Path(data_path) / "local_effect_contrast.pkl"
    evoked, _, _, local_contrast = get_contrast(computed_data_path=local_contrast_path)
    evoked.shift_time(time_shift)
    
    fig_topo = plot_evoked_topomap(evoked, times=plot_times, ch_type="eeg", contours=0, cmap="RdBu_r",
                                   cbar_fmt="%0.3f", average=0.04, units=r"$\mu{V}$", ncols=10, nrows="auto",
                                   extrapolate="local", sns_kwargs=dict(style="white"))
    
    # Get actual data range for proper statistical color scaling
    local_contrast.mlog10_p_val.shift_time(time_shift)
    stat_data = np.array(local_contrast.mlog10_p_val.data)
    stat_vmin_data = np.nanmin(stat_data)
    stat_vmax_data = np.nanmax(stat_data)
    
    # Use data range but ensure it includes the significance threshold properly
    stat_vmin_actual = max(0, stat_vmin_data)  # Don't go below 0 for -log10(p)
    stat_vmax_actual = max(stat_vmax_data, stat_logpsig + 2.0)  # Ensure significance is well-positioned
    
    # Generate colormap based on actual data range
    stat_cmap = get_stat_colormap(stat_logpsig, stat_vmin_actual, stat_vmax_actual)
    
    # The local_contrast.mlog10_p_val is already shifted, so don't shift it again
    # local_contrast.mlog10_p_val.shift_time(time_shift)  # REMOVED - double shift
    
    # Use all plot_times - they should work now
    fig_topo_stat = plot_evoked_topomap(local_contrast.mlog10_p_val, times=plot_times, ch_type="eeg",
                                       contours=0, cmap=stat_cmap, scalings=1, cbar_fmt="%0.3f", average=0.04,
                                       units="-log10(p)", ncols=10, nrows="auto", extrapolate="local",
                                       vlim=(stat_vmin_actual, stat_vmax_actual),  # Use vlim instead of vmin/vmax
                                       sns_kwargs=dict(style="white"))
    
    gc.collect()
    
    # Cluster test
    fig_cluster = None
    if not skip_clustering:
        local_cluster_path = Path(data_path) / "local_cluster_test.pkl"
        if local_cluster_path.exists():
            fig_cluster = plot_cluster_test(computed_data_path=local_cluster_path, sns_kwargs={"style": "darkgrid"})
    
    # Add Local Effect figures
    figs = [fig_gfp, fig_topo, fig_topo_stat]
    captions = ["Local Effect: Global Field Power", "Local Effect: Topographies", "Local Effect: Topographies (-log10(p))"]
    if fig_cluster is not None:
        figs.insert(0, fig_cluster)
        captions.insert(0, "Local Effect: Cluster Permutation Test")
    
    for fig, caption in zip(figs, captions):
        if fig is not None:
            report.add_figure(fig, title=caption, section="ERP")
            plt.close(fig)
    
    # GLOBAL EFFECT (similar structure)
    global_gfp_path = Path(data_path) / "global_effect_gfp.pkl"
    fig_gfp = plot_gfp(computed_data_path=global_gfp_path, colors=["r", "b"], labels=global_labels,
                       fig_kwargs=dict(figsize=(12, 6)), sns_kwargs=dict(style="darkgrid"))
    
    global_contrast_path = Path(data_path) / "global_effect_contrast.pkl"
    evoked, _, _, global_contrast = get_contrast(computed_data_path=global_contrast_path)
    evoked.shift_time(time_shift)
    
    fig_topo = plot_evoked_topomap(evoked, times=plot_times, ch_type="eeg", contours=0, cmap="RdBu_r",
                                   cbar_fmt="%0.3f", average=0.04, units=r"$\mu{V}$", ncols=10, nrows="auto",
                                   extrapolate="local", sns_kwargs=dict(style="white"))
    
    # Get actual data range for proper statistical color scaling (global effect)
    global_contrast.mlog10_p_val.shift_time(time_shift)
    stat_data_global = np.array(global_contrast.mlog10_p_val.data)
    stat_vmin_global = max(0, np.nanmin(stat_data_global))  # Don't go below 0 for -log10(p)
    stat_vmax_global = max(np.nanmax(stat_data_global), stat_logpsig + 2.0)  # Ensure significance is well-positioned
    
    # Generate colormap based on actual data range
    stat_cmap_global = get_stat_colormap(stat_logpsig, stat_vmin_global, stat_vmax_global)
    
    # The global_contrast.mlog10_p_val is also already shifted, don't shift it again
    # global_contrast.mlog10_p_val.shift_time(time_shift)  # REMOVED - double shift
    
    fig_topo_stat = plot_evoked_topomap(global_contrast.mlog10_p_val, times=plot_times, ch_type="eeg",
                                       contours=0, cmap=stat_cmap_global, scalings=1, cbar_fmt="%0.3f", average=0.04,
                                       units="-log10(p)", ncols=10, nrows="auto", extrapolate="local",
                                       vlim=(stat_vmin_global, stat_vmax_global),  # Use vlim instead of vmin/vmax
                                       sns_kwargs=dict(style="white"))
    
    gc.collect()
    
    fig_cluster = None
    if not skip_clustering:
        global_cluster_path = Path(data_path) / "global_cluster_test.pkl"
        if global_cluster_path.exists():
            fig_cluster = plot_cluster_test(computed_data_path=global_cluster_path, sns_kwargs={"style": "darkgrid"})
    
    # Add Global Effect figures
    figs = [fig_gfp, fig_topo, fig_topo_stat]
    captions = ["Global Effect: Global Field Power", "Global Effect: Topographies", "Global Effect: Topographies (-log10(p))"]
    if fig_cluster is not None:
        figs.append(fig_cluster)
        captions.append("Global Effect: Cluster Permutation Test")
    
    for fig, caption in zip(figs, captions):
        if fig is not None:
            report.add_figure(fig, title=caption, section="ERP")
            plt.close(fig)
    
    # ERP Statistical Analysis
    try:
        n_chans_thresh = 10
        n_times_thresh = 5
        p_vals_ticks = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        fig_stats = plot_ttest([local_contrast.p_val.data, global_contrast.p_val.data],
                              ["Local Effect", "Global Effect"], p_vals_ticks, (epochs.times - 0.6),
                              n_times_thresh=n_times_thresh, n_chans_thresh=n_chans_thresh, colors=["b", "r"])
        caption = f"ERP Statistical Analysis (> {n_chans_thresh} channels, > {n_times_thresh} samples)"
        report.add_figure(fig_stats, title=caption, section="ERP")
        plt.close(fig_stats)
    except Exception as e:
        logger.warning(f"ERP statistical analysis failed: {e}")
    
    # ROI Analysis
    rois = ["Fz", "Cz", "Pz"]
    fig_rois, axes = plt.subplots(len(rois), 2, figsize=(14, 2 * len(rois)))
    
    for i, roi_name in enumerate(rois):
        # Local Effect
        local_roi_path = Path(data_path) / f"local_effect_contrast_{roi_name}.pkl"
        this_evoked, evokeds, evokeds_stderr, this_contrast = get_contrast(computed_data_path=local_roi_path)
        sig_mask = np.squeeze(this_contrast.p_val.data < stat_psig)
        plot_evoked(evokeds, std_errs=evokeds_stderr, colors=["r", "b"], labels=["Deviant", "Standard"],
                   shift_time=time_shift, ax=axes[i, 0], sig_mask=sig_mask, event_times=event_times,
                   sns_kwargs=dict(style="darkgrid"))
        col_title = "" if i != 0 else "Local Effect \n\n"
        axes[i, 0].set_title(f"{col_title}Around {roi_name}")
        
        # Global Effect
        global_roi_path = Path(data_path) / f"global_effect_contrast_{roi_name}.pkl"
        this_evoked, evokeds, evokeds_stderr, this_contrast = get_contrast(computed_data_path=global_roi_path)
        sig_mask = np.squeeze(this_contrast.p_val.data < stat_psig)
        plot_evoked(evokeds, std_errs=evokeds_stderr, colors=["b", "r"], labels=None,
                   shift_time=time_shift, ax=axes[i, 1], sig_mask=sig_mask, event_times=event_times,
                   sns_kwargs=dict(style="darkgrid"))
        col_title = "" if i != 0 else "Global Effect \n\n"
        axes[i, 1].set_title(f"{col_title}Around {roi_name}")
    
    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(mpl.patches.Patch(color="r", alpha=0.5, label="Deviant SEM"))
    handles.append(mpl.patches.Patch(color="b", alpha=0.5, label="Standard SEM"))
    handles.append(mpl.patches.Patch(color="orange", label=f"Significative (p<{stat_psig})"))
    axes[0, 0].legend(handles=handles, bbox_to_anchor=(-0.1, 1.0))
    plt.subplots_adjust(hspace=0.6, left=0.18, right=0.98)
    
    report.add_figure(fig_rois, title="ERP ROI Analysis", section="ERP")
    plt.close(fig_rois)
    gc.collect()


def _add_prediction_plots(report, data_path):
    """Add prediction plots from precomputed results"""
    import pickle
    logger = logging.getLogger(__name__)
    
    # Load prediction results
    prediction_path = Path(data_path) / "prediction_results.pkl"
    if not prediction_path.exists():
        logger.warning("No prediction results found - skipping prediction section")
        html = """
        <div class="alert alert-info">
            <h4>Prediction Analysis</h4>
            <p>Prediction analysis requires trained classification models and patient data.</p>
            <p>This section will be populated when real prediction results are available.</p>
        </div>
        """
        report.add_html(html, title="Prediction Summary", section="Prediction")
        return
    
    try:
        with open(prediction_path, 'rb') as f:
            prediction_results = pickle.load(f)
        
        if not prediction_results or 'multivariate' not in prediction_results:
            logger.warning("Invalid prediction results")
            return
        
        multivariate = prediction_results.get('multivariate', {})
        univariate = prediction_results.get('univariate', None)
        summary_info = prediction_results.get('summary', {})
        
        # 1. Create prediction summary HTML
        prediction_summary = {
            "step": ["Classes:", "Target:", "Features:", "Samples:"],
            "params": [
                {"classes": str(summary_info.get('classes', ['VS/UWS', 'MCS']))},
                {"target": summary_info.get('target', 'Label')},
                {"n_features": str(summary_info.get('n_features', 0))},
                {"n_samples": str(summary_info.get('n_samples', 1))},
            ],
            "result": [[""], [""], [""], [""]],
        }
        
        for clf_name, clf_results in multivariate.items():
            if isinstance(clf_results, dict):
                vs_prob = clf_results.get('VS/UWS', 0.0)
                mcs_prob = clf_results.get('MCS', 0.0)
                prediction_summary["step"].append(f"{clf_name.upper()}:")
                prediction_summary["params"].append({
                    "classifier": clf_name,
                    "VS/UWS": f"{vs_prob:.3f}",
                    "MCS": f"{mcs_prob:.3f}"
                })
                prediction_summary["result"].append([
                    f"VS/UWS: {vs_prob:.1%}, MCS: {mcs_prob:.1%}"
                ])
        
        html = render_prediction_summary(prediction_summary)
        report.add_html(html, title="Prediction Summary", section="Prediction")
        
        # 2. Create multivariate visualizations
        sns.set()
        sns.set_color_codes()
        
        for clf_type, results in multivariate.items():
            if not isinstance(results, dict):
                continue
            
            fig_overall = plt.figure(figsize=(8, 2))
            vs_prob = results.get('VS/UWS', 0.0)
            mcs_prob = results.get('MCS', 0.0)
            
            if clf_type == 'et-reduced':
                # Binary decision visualization
                bins = [0, 1] if mcs_prob > 0.5 else [1, 0]
                bars = plt.barh([0, 0], bins, 1, [0, bins[0]],
                              color=['r', 'b'], alpha=0.7, edgecolor='0.2')
                sns.despine(left=True, bottom=True)
                ax = fig_overall.get_axes()[0]
                ax.set_xticklabels([])
                ax.set_yticks([0])
                ax.set_yticklabels(['Result:'])
                ax.set_xlim(0, 1)
                plt.subplots_adjust(bottom=0.5, top=0.6, left=0.45, right=0.55)
            else:
                # Probability bars
                bars = plt.barh([0, 0], [vs_prob, mcs_prob], 1, [0, vs_prob],
                              color=['r', 'b'], alpha=0.7, edgecolor='0.2')
                ax = fig_overall.get_axes()[0]
                [ax.axvline(x, color='0.2', lw=0.4) for x in np.arange(0, 1.01, 0.1)]
                plt.xticks(np.arange(0, 1.01, 0.1))
                sns.despine(left=True, bottom=False)
                ax.yaxis.set_visible(False)
                ax.set_xlim(0, 1)
                plt.subplots_adjust(bottom=0.5, top=0.6)
            
            plt.legend((bars[0], bars[1]), ['VS/UWS', 'MCS'], loc=9,
                     bbox_to_anchor=(0.5, -1.5), ncol=2)
            
            clf_display_name = 'Extra Trees' if clf_type == 'et-reduced' else 'Gaussian SVM'
            report.add_figure(fig_overall, title=clf_display_name, section='Multivariate')
            plt.close(fig_overall)
        
        logger.info("Added multivariate prediction visualizations")
        
        # 3. Create univariate visualizations if available
        if univariate is not None and not univariate.empty and len(univariate) > 0:
            import pandas as pd
            
            logger.info(f"Creating univariate visualizations for {len(univariate)} markers")
            
            # Violin plot
            stacked_df = univariate.set_index(['Marker', 'Reduction'])[['VS/UWS', 'MCS']].stack()
            stacked_df.index.names = ['Marker', 'Reduction', 'Label']
            stacked_df.name = 'P'
            stacked_df = stacked_df.reset_index()
            
            fig_violin = plt.figure(figsize=(8, 2))
            sns.violinplot(x='P', y='Label', data=stacked_df, orient='h',
                         inner='quartile', palette=['r', 'b'], cut=0)
            plt.axvline(0.5, color='k', linestyle='--')
            plt.xlabel('Probability')
            plt.ylabel('Probability Density')
            plt.title(f'Summary of univariate prediction ({len(univariate)} markers)')
            plt.subplots_adjust(bottom=0.25)
            
            report.add_figure(fig_violin, title='Univariate Summary I', section='Univariate Summaries')
            plt.close(fig_violin)
            
            # Grouped bar plot
            marker_groups = {
                'Information Theory': ['PermutationEntropy', 'KolmogorovComplexity'],
                'Spectral': ['PowerSpectralDensity', 'PowerSpectralDensitySummary'],
                'Connectivity': ['SymbolicMutualInformation'],
                'ERPs': ['ContingentNegativeVariation', 'WindowDecoding', 'TimeLockedContrast', 'TimeLockedTopography']
            }
            
            grouped_data = {}
            for group_name, marker_types in marker_groups.items():
                group_markers = []
                for _, row in univariate.iterrows():
                    marker_name = row['Marker']
                    if any(marker_type in marker_name for marker_type in marker_types):
                        group_markers.append(row)
                if group_markers:
                    grouped_data[group_name] = pd.DataFrame(group_markers)
            
            if grouped_data:
                n_groups = len(grouped_data)
                group_sizes = [len(df) for df in grouped_data.values()]
                
                fig_bars = plt.figure(figsize=(12, max(6, sum(group_sizes) * 0.5)))
                gs = gridspec.GridSpec(n_groups, 1, height_ratios=group_sizes)
                
                bar_height = 3
                for i, (group_name, group_df) in enumerate(grouped_data.items()):
                    ax = plt.subplot(gs[i])
                    
                    n_markers = len(group_df)
                    ys = np.arange(n_markers) * bar_height
                    bottoms = np.c_[ys, ys].ravel()
                    
                    vs_probs = group_df['VS/UWS'].values
                    mcs_probs = group_df['MCS'].values
                    
                    xs = np.c_[vs_probs, mcs_probs].ravel()
                    lefts = np.c_[np.zeros(n_markers), vs_probs].ravel()
                    
                    ax.barh(y=bottoms, width=xs, left=lefts, height=3,
                            edgecolor='0.2', color=['r', 'b'], alpha=0.7)
                    
                    ax.set_ylim([-bar_height/2, bar_height * n_markers - bar_height/2])
                    
                    labels = []
                    for _, row in group_df.iterrows():
                        marker_parts = row['Marker'].split('/')
                        marker_label = marker_parts[-1] if len(marker_parts) >= 2 else row['Marker']
                        reduction_label = row['Reduction']
                        labels.append(f'{marker_label} ({reduction_label})')
                    
                    ax.set_yticks(range(0, bar_height * n_markers, bar_height))
                    ax.set_yticklabels(labels)
                    ax.set_xlim(0, 1)
                    ax.set_xticks(np.arange(0, 1.01, 0.1))
                    ax.axvline(0.5, color='k', linestyle='--', linewidth=1)
                    ax.axvline(0.25, color='k', linestyle='--', linewidth=1)
                    ax.axvline(0.75, color='k', linestyle='--', linewidth=1)
                    ax.set_title(group_name)
                    
                    if i == 0:
                        ax.legend(['VS/UWS', 'MCS'], loc='upper right', bbox_to_anchor=(1.13, 1.1))
                
                plt.subplots_adjust(bottom=0.05, top=0.92)
                plt.suptitle('Probability of being VS/UWS vs MCS')
                
                report.add_figure(fig_bars, title='Univariate Summary II', section='Univariate Summaries')
                plt.close(fig_bars)
            
            logger.info("Added univariate prediction visualizations")
        
        logger.info("✅ All prediction plots added to report")
        
    except Exception as e:
        logger.error(f"Failed to create prediction plots: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
