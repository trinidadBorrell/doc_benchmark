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
        default="001",
        help="Subject ID for report generation (e.g., '001')",
    )
    parser.add_argument(
        "--session",
        default="01",
        help="Session number for report generation (e.g., '01')",
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
        "--data_dir_original",
        default="/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/sub-001_original",
        help="Directory containing pre-computed data for ORIGINAL (pickle files)",
    )
    parser.add_argument(
        "--data_dir_recon",
        default="/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/sub-001_recon",
        help="Directory containing pre-computed data for RECONSTRUCTED (pickle files)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for HTML report (default: auto-generated from subject/session)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if computed data directories exist
    data_path_original = Path(args.data_dir_original)
    data_path_recon = Path(args.data_dir_recon)
    if not data_path_original.exists():
        logger.error(f"❌ Original computed data directory not found: {data_path_original}")
        logger.error("Please run compute_data.py first to generate the data.")
        sys.exit(1)
    if not data_path_recon.exists():
        logger.error(f"❌ Reconstructed computed data directory not found: {data_path_recon}")
        logger.error("Please run compute_data.py first to generate the data.")
        sys.exit(1)

    # Generate plots from pre-computed data
    try:
        # Create output directory structure
        if args.output_dir is None:
            base_results_dir = Path("/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS")
            output_dir = base_results_dir / f"sub-{args.subject_id}" / f"ses-{args.session}" / "report"
        else:
            output_dir = Path(args.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating report for subject {args.subject_id}, session {args.session}")
        logger.info(f"Loading ORIGINAL data from: {args.data_dir_original}")
        logger.info(f"Loading RECONSTRUCTED data from: {args.data_dir_recon}")
        logger.info(f"Output directory: {output_dir}")
        
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

        # ========== PHASE 2: GENERATE ALL PLOTS (COMPARISON) ==========
        logger.info("="*60)
        logger.info("PHASE 2: Generating comparison plots from computed data...")
        logger.info(f"Original data: {data_path_original}")
        logger.info(f"Reconstructed data: {data_path_recon}")
        logger.info("="*60)

        report = mne.Report(title=f"Task: Local-Global - subject: {args.subject_id} - COMPARISON: Original vs Reconstructed")

        # === PREPROCESSING PLOTS ===
        # Skip preprocessing plots if no preprocessing metadata available
        preprocessing_info = report_data.get("metadata", {}).get("preprocessing_info", {})
        if preprocessing_info:
            logger.info("Adding preprocessing plots...")
            _add_preprocessing_plots(report, epochs, report_data, output_dir)
            gc.collect()
        else:
            logger.info("Skipping preprocessing plots (no preprocessing metadata available)")

        # === DIAGNOSTIC PLOTS ===
        logger.info("Adding diagnostic plots...")
        _add_diagnostic_plots(report, epochs, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === ERP PLOTS ===
        logger.info("Adding ERP plots...")
        _add_erp_plots(report, epochs, args.skip_clustering, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === CNV PLOTS ===
        logger.info("Adding CNV plots...")
        _add_cnv_plots(report, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === SPECTRAL PLOTS ===
        logger.info("Adding spectral plots...")
        _add_spectral_plots(report, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === CONNECTIVITY PLOTS ===
        logger.info("Adding connectivity plots...")
        _add_connectivity_plots(report, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === INFORMATION THEORY PLOTS ===
        logger.info("Adding information theory plots...")
        _add_information_theory_plots(report, data_path_original, data_path_recon, output_dir)
        gc.collect()

        # === PREDICTION ANALYSIS ===
        logger.info("Adding prediction analysis...")
        _add_prediction_plots(report, data_path_original, data_path_recon, output_dir)
        gc.collect()

        logger.info("="*60)
        logger.info("✅ PHASE 2 COMPLETE: All plots added to report")
        logger.info("="*60)

        # Save report
        output_path = output_dir / f"sub-{args.subject_id}_ses-{args.session}_report_comparison.html"
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


def _add_preprocessing_plots(report, epochs, report_data, save_dir=None):
    """Add preprocessing plots"""
    import json
    import pickle
    logger = logging.getLogger(__name__)
    
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
            
            # Save plot and data
            if save_dir:
                plot_name = "preprocessing_bad_channels"
                fig.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                # Save data
                np.savez(
                    Path(save_dir) / f"{plot_name}.npz",
                    evoked_data=evoked.data,
                    times=evoked.times,
                    ch_names=evoked.ch_names,
                    bad_channels=eeg_bad_channels,
                    bad_idx=bad_idx
                )
                logger.info(f"   💾 Saved: {plot_name}")
            
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


def _create_side_by_side_figure(fig_left, fig_right, title_left="ORIGINAL", title_right="RECONSTRUCTED"):
    """Create a single figure with two subfigures side-by-side"""
    import io
    from PIL import Image
    
    # Convert figures to images
    def fig_to_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        return img, buf
    
    img_left, buf_left = fig_to_img(fig_left) if fig_left else (None, None)
    img_right, buf_right = fig_to_img(fig_right) if fig_right else (None, None)
    
    # Create combined figure
    fig_combined = plt.figure(figsize=(24, 10))
    
    if img_left and img_right:
        # Both exist - show side by side
        ax1 = fig_combined.add_subplot(1, 2, 1)
        ax2 = fig_combined.add_subplot(1, 2, 2)
        
        ax1.imshow(img_left)
        ax1.axis('off')
        ax1.set_title(title_left, fontsize=14, fontweight='bold', pad=10)
        
        ax2.imshow(img_right)
        ax2.axis('off')
        ax2.set_title(title_right, fontsize=14, fontweight='bold', pad=10)
    elif img_left:
        ax = fig_combined.add_subplot(1, 1, 1)
        ax.imshow(img_left)
        ax.axis('off')
        ax.set_title(title_left, fontsize=14, fontweight='bold', pad=10)
    elif img_right:
        ax = fig_combined.add_subplot(1, 1, 1)
        ax.imshow(img_right)
        ax.axis('off')
        ax.set_title(title_right, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Clean up buffers
    if buf_left:
        buf_left.close()
    if buf_right:
        buf_right.close()
    
    return fig_combined


def _create_vertical_stacked_figure(fig_top, fig_bottom, title_top="ORIGINAL", title_bottom="RECONSTRUCTED"):
    """Create a single figure with two subfigures stacked vertically"""
    import io
    from PIL import Image
    
    # Convert figures to images
    def fig_to_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        return img, buf
    
    img_top, buf_top = fig_to_img(fig_top) if fig_top else (None, None)
    img_bottom, buf_bottom = fig_to_img(fig_bottom) if fig_bottom else (None, None)
    
    # Create combined figure with vertical stacking
    fig_combined = plt.figure(figsize=(16, 20))
    
    if img_top and img_bottom:
        # Both exist - stack vertically
        ax1 = fig_combined.add_subplot(2, 1, 1)
        ax2 = fig_combined.add_subplot(2, 1, 2)
        
        ax1.imshow(img_top)
        ax1.axis('off')
        ax1.set_title(title_top, fontsize=14, fontweight='bold', pad=10)
        
        ax2.imshow(img_bottom)
        ax2.axis('off')
        ax2.set_title(title_bottom, fontsize=14, fontweight='bold', pad=10)
    elif img_top:
        ax = fig_combined.add_subplot(1, 1, 1)
        ax.imshow(img_top)
        ax.axis('off')
        ax.set_title(title_top, fontsize=14, fontweight='bold', pad=10)
    elif img_bottom:
        ax = fig_combined.add_subplot(1, 1, 1)
        ax.imshow(img_bottom)
        ax.axis('off')
        ax.set_title(title_bottom, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Clean up buffers
    if buf_top:
        buf_top.close()
    if buf_bottom:
        buf_bottom.close()
    
    return fig_combined


def _add_diagnostic_plots(report, epochs, data_path_original, data_path_recon, save_dir=None):
    """Add diagnostic plots (side-by-side comparison)"""
    logger = logging.getLogger(__name__)
    # 1. Local Global Paradigm (same for both)
    fig = mne.viz.plot_events(epochs.events, epochs.info["sfreq"], event_id=epochs.event_id)
    ax = fig.get_axes()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.02, 1), 
             frameon=True, fontsize=9, title="Trial Types")
    plt.tight_layout()
    
    # Save plot and data
    if save_dir:
        plot_name = "diagnostic_paradigm"
        fig.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
        np.savez(
            Path(save_dir) / f"{plot_name}.npz",
            events=epochs.events,
            event_id=list(epochs.event_id.values()),
            event_names=list(epochs.event_id.keys()),
            sfreq=epochs.info["sfreq"]
        )
        logger.info(f"   💾 Saved: {plot_name}")
    
    report.add_figure(fig, title="Local Global Paradigm", section="Diagnostic")
    plt.close(fig)
    
    # 2. GFP plots (side-by-side)
    gfp_path_original = Path(data_path_original) / "diagnostic_gfp.pkl"
    gfp_path_recon = Path(data_path_recon) / "diagnostic_gfp.pkl"
    
    if gfp_path_original.exists() or gfp_path_recon.exists():
        import pickle
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        
        # Load data to get shared y-axis limits
        y_values = []
        gfp_data_orig, gfp_data_recon = None, None
        
        if gfp_path_original.exists():
            with open(gfp_path_original, 'rb') as f:
                gfp_data_orig = pickle.load(f)
                for evoked in gfp_data_orig.get('evokeds', []):
                    gfp = np.sqrt((evoked.data ** 2).mean(axis=0)) * 1e6
                    # Only collect valid values
                    valid_gfp = gfp[np.isfinite(gfp)]
                    if len(valid_gfp) > 0:
                        y_values.extend([valid_gfp.min(), valid_gfp.max()])
        
        if gfp_path_recon.exists():
            with open(gfp_path_recon, 'rb') as f:
                gfp_data_recon = pickle.load(f)
                for evoked in gfp_data_recon.get('evokeds', []):
                    gfp = np.sqrt((evoked.data ** 2).mean(axis=0)) * 1e6
                    # Only collect valid values
                    valid_gfp = gfp[np.isfinite(gfp)]
                    if len(valid_gfp) > 0:
                        y_values.extend([valid_gfp.min(), valid_gfp.max()])
        
        # Compute shared limits only if we have valid values
        use_shared_ylim = len(y_values) > 0
        if use_shared_ylim:
            y_min = min(y_values)
            y_max = max(y_values)
            # Add 5% margin
            y_margin = (y_max - y_min) * 0.05
            y_lim = [y_min - y_margin, y_max + y_margin]
        
        # Original (left)
        if gfp_path_original.exists():
            plot_gfp(computed_data_path=gfp_path_original, colors=["b", "b", "r", "r"],
                    linestyles=["-", "--", "-", "--"], ax=axes[0],
                    sns_kwargs={"style": "darkgrid"})
            if use_shared_ylim:
                axes[0].set_ylim(y_lim)
            axes[0].set_title("ORIGINAL: Global Field Power", fontweight='bold')
        
        # Reconstructed (right)
        if gfp_path_recon.exists():
            plot_gfp(computed_data_path=gfp_path_recon, colors=["b", "b", "r", "r"],
                    linestyles=["-", "--", "-", "--"], ax=axes[1],
                    sns_kwargs={"style": "darkgrid"})
            if use_shared_ylim:
                axes[1].set_ylim(y_lim)
            axes[1].set_title("RECONSTRUCTED: Global Field Power", fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot and data
        if save_dir:
            plot_name = "diagnostic_gfp_comparison"
            fig.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
            # Save data from both datasets
            save_data = {}
            if gfp_data_orig:
                save_data['original_evokeds'] = [ev.data for ev in gfp_data_orig.get('evokeds', [])]
                save_data['original_times'] = gfp_data_orig.get('evokeds', [{}])[0].times if gfp_data_orig.get('evokeds') else []
            if gfp_data_recon:
                save_data['reconstructed_evokeds'] = [ev.data for ev in gfp_data_recon.get('evokeds', [])]
                save_data['reconstructed_times'] = gfp_data_recon.get('evokeds', [{}])[0].times if gfp_data_recon.get('evokeds') else []
            np.savez(Path(save_dir) / f"{plot_name}.npz", **save_data)
            logger.info(f"   💾 Saved: {plot_name}")
        
        report.add_figure(fig, title="All Blocks: Global Field Power Comparison", section="Diagnostic")
        plt.close(fig)


def _add_cnv_plots(report, data_path_original, data_path_recon, save_dir=None):
    """Add CNV plots (vertically stacked with matched scales)"""
    import pickle
    import logging
    logger = logging.getLogger(__name__)
    
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}
    cnv_path_original = Path(data_path_original) / "cnv_computed_data.pkl"
    cnv_path_recon = Path(data_path_recon) / "cnv_computed_data.pkl"
    
    fig_orig, fig_recon = None, None
    
    # Load both datasets to compute shared scales
    if cnv_path_original.exists() and cnv_path_recon.exists():
        with open(cnv_path_original, 'rb') as f:
            data_orig = pickle.load(f)
        with open(cnv_path_recon, 'rb') as f:
            data_recon = pickle.load(f)
        
        # Compute combined vminmax
        vminmax_orig = data_orig.get('vminmax', None)
        vminmax_recon = data_recon.get('vminmax', None)
        
        logger.info(f"📊 CNV SCALE MATCHING:")
        logger.info(f"   Original vminmax: {vminmax_orig}")
        logger.info(f"   Reconstructed vminmax: {vminmax_recon}")
        
        if vminmax_orig is not None and vminmax_recon is not None:
            vminmax_combined = max(abs(float(vminmax_orig)), abs(float(vminmax_recon)))
            logger.info(f"   ✓ Combined vminmax (symmetric): ±{vminmax_combined}")
            data_orig['vminmax'] = vminmax_combined
            data_recon['vminmax'] = vminmax_combined
        else:
            logger.warning(f"   ⚠️  Could not match scales - missing vminmax values")
        
        # Generate both plots
        fig_orig = plot_cnv(computed_data=data_orig, event_times=event_times, 
                           sns_kwargs={"style": "white"})
        fig_recon = plot_cnv(computed_data=data_recon, event_times=event_times,
                            sns_kwargs={"style": "white"})
        
        # Save individual plots and data if save_dir provided
        if save_dir and fig_orig:
            save_path = Path(save_dir) / "cnv_original.png"
            fig_orig.savefig(save_path, dpi=150, bbox_inches='tight')
            # Save data
            np.savez(
                Path(save_dir) / "cnv_original.npz",
                evokeds=[ev.data for ev in data_orig.get('evokeds', [])],
                times=data_orig.get('evokeds', [{}])[0].times if data_orig.get('evokeds') else [],
                vminmax=data_orig.get('vminmax')
            )
            logger.info(f"   💾 Saved: cnv_original")
        if save_dir and fig_recon:
            save_path = Path(save_dir) / "cnv_reconstructed.png"
            fig_recon.savefig(save_path, dpi=150, bbox_inches='tight')
            # Save data
            np.savez(
                Path(save_dir) / "cnv_reconstructed.npz",
                evokeds=[ev.data for ev in data_recon.get('evokeds', [])],
                times=data_recon.get('evokeds', [{}])[0].times if data_recon.get('evokeds') else [],
                vminmax=data_recon.get('vminmax')
            )
            logger.info(f"   💾 Saved: cnv_reconstructed")
    else:
        # Load individual plots if only one exists
        if cnv_path_original.exists():
            fig_orig = plot_cnv(computed_data_path=cnv_path_original, event_times=event_times, 
                               sns_kwargs={"style": "white"})
            if save_dir and fig_orig:
                save_path = Path(save_dir) / "cnv_original.png"
                fig_orig.savefig(save_path, dpi=150, bbox_inches='tight')
        if cnv_path_recon.exists():
            fig_recon = plot_cnv(computed_data_path=cnv_path_recon, event_times=event_times,
                                sns_kwargs={"style": "white"})
            if save_dir and fig_recon:
                save_path = Path(save_dir) / "cnv_reconstructed.png"
                fig_recon.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Create vertically stacked figure
    if fig_orig or fig_recon:
        fig_combined = _create_vertical_stacked_figure(fig_orig, fig_recon, 
                                                       "ORIGINAL: Contingent Negative Variation",
                                                       "RECONSTRUCTED: Contingent Negative Variation")
        report.add_figure(fig_combined, title="CNV - Comparison", section="CNV")
        
        # Save combined figure and data
        if save_dir:
            save_path = Path(save_dir) / "cnv_comparison.png"
            fig_combined.savefig(save_path, dpi=150, bbox_inches='tight')
            # Save combined data
            save_data = {}
            if data_orig:
                save_data['original_evokeds'] = [ev.data for ev in data_orig.get('evokeds', [])]
                save_data['original_times'] = data_orig.get('evokeds', [{}])[0].times if data_orig.get('evokeds') else []
                save_data['original_vminmax'] = data_orig.get('vminmax')
            if data_recon:
                save_data['reconstructed_evokeds'] = [ev.data for ev in data_recon.get('evokeds', [])]
                save_data['reconstructed_times'] = data_recon.get('evokeds', [{}])[0].times if data_recon.get('evokeds') else []
                save_data['reconstructed_vminmax'] = data_recon.get('vminmax')
            np.savez(Path(save_dir) / "cnv_comparison.npz", **save_data)
            logger.info(f"   💾 Saved: cnv_comparison")
        
        plt.close(fig_combined)
        if fig_orig:
            plt.close(fig_orig)
        if fig_recon:
            plt.close(fig_recon)


def _add_spectral_plots(report, data_path_original, data_path_recon, save_dir=None):
    """Add spectral plots (side-by-side comparison with matched scales)"""
    import pickle
    logger = logging.getLogger(__name__)
    
    # Helper to match scales and plot side-by-side
    def plot_with_matched_scales(path_orig, path_recon, title_base, section, plot_prefix):
        if not path_orig.exists() and not path_recon.exists():
            return
        
        # Load data
        data_orig, data_recon = None, None
        if path_orig.exists():
            with open(path_orig, 'rb') as f:
                data_orig = pickle.load(f)
        if path_recon.exists():
            with open(path_recon, 'rb') as f:
                data_recon = pickle.load(f)
        
        # Match scales if both exist
        if data_orig and data_recon:
            vmin_o, vmax_o = data_orig.get('vmin'), data_orig.get('vmax')
            vmin_r, vmax_r = data_recon.get('vmin'), data_recon.get('vmax')
            
            if vmin_o is not None and vmin_r is not None:
                if isinstance(vmin_o, (list, np.ndarray)):
                    vmin_combined = [min(vo, vr) for vo, vr in zip(vmin_o, vmin_r)]
                    vmax_combined = [max(vo, vr) for vo, vr in zip(vmax_o, vmax_r)]
                else:
                    vmin_combined = min(vmin_o, vmin_r)
                    vmax_combined = max(vmax_o, vmax_r)
                
                data_orig['vmin'] = vmin_combined
                data_orig['vmax'] = vmax_combined
                data_recon['vmin'] = vmin_combined
                data_recon['vmax'] = vmax_combined
        
        # Generate figures
        fig_orig = plot_markers_topos(computed_data=data_orig) if data_orig else None
        fig_recon = plot_markers_topos(computed_data=data_recon) if data_recon else None
        
        # Create side-by-side figure
        if fig_orig or fig_recon:
            fig_combined = _create_side_by_side_figure(fig_orig, fig_recon,
                                                       f"ORIGINAL: {title_base}",
                                                       f"RECONSTRUCTED: {title_base}")
            
            # Save plot and data
            if save_dir:
                plot_name = f"{section.lower()}_{plot_prefix}"
                fig_combined.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                # Save data
                save_data = {}
                if data_orig:
                    save_data['original_evokeds'] = data_orig.get('evokeds', [])
                    save_data['original_vmin'] = data_orig.get('vmin')
                    save_data['original_vmax'] = data_orig.get('vmax')
                if data_recon:
                    save_data['reconstructed_evokeds'] = data_recon.get('evokeds', [])
                    save_data['reconstructed_vmin'] = data_recon.get('vmin')
                    save_data['reconstructed_vmax'] = data_recon.get('vmax')
                np.savez(Path(save_dir) / f"{plot_name}.npz", **save_data)
                logger.info(f"   💾 Saved: {plot_name}")
            
            report.add_figure(fig_combined, title=f"{title_base} - Comparison", section=section)
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)
    
    # 1. Normalized spectral bands
    plot_with_matched_scales(
        Path(data_path_original) / "spectral_bands_normalized.pkl",
        Path(data_path_recon) / "spectral_bands_normalized.pkl",
        "Spectral Power Bands Normalized",
        "Spectral",
        "bands_normalized"
    )
    
    # 2. Absolute power
    plot_with_matched_scales(
        Path(data_path_original) / "spectral_absolute_power.pkl",
        Path(data_path_recon) / "spectral_absolute_power.pkl",
        "Absolute Spectral Power",
        "Spectral",
        "absolute_power"
    )
    
    # 3. Spectral summaries
    plot_with_matched_scales(
        Path(data_path_original) / "spectral_summaries.pkl",
        Path(data_path_recon) / "spectral_summaries.pkl",
        "Spectral Summaries",
        "Spectral",
        "summaries"
    )


def _add_connectivity_plots(report, data_path_original, data_path_recon, save_dir=None):
    """Add connectivity plots (vertically stacked with matched scales)"""
    import pickle
    logger = logging.getLogger(__name__)
    
    # Helper for markers topos
    def plot_markers_with_matched_scales(path_orig, path_recon, title_base, section):
        if not path_orig.exists() and not path_recon.exists():
            return
        
        data_orig, data_recon = None, None
        if path_orig.exists():
            with open(path_orig, 'rb') as f:
                data_orig = pickle.load(f)
        if path_recon.exists():
            with open(path_recon, 'rb') as f:
                data_recon = pickle.load(f)
        
        if data_orig and data_recon:
            vmin_o, vmax_o = data_orig.get('vmin'), data_orig.get('vmax')
            vmin_r, vmax_r = data_recon.get('vmin'), data_recon.get('vmax')
            
            if vmin_o is not None and vmin_r is not None:
                if isinstance(vmin_o, (list, np.ndarray)):
                    vmin_combined = [min(vo, vr) for vo, vr in zip(vmin_o, vmin_r)]
                    vmax_combined = [max(vo, vr) for vo, vr in zip(vmax_o, vmax_r)]
                else:
                    vmin_combined = min(vmin_o, vmin_r)
                    vmax_combined = max(vmax_o, vmax_r)
                
                data_orig['vmin'] = vmin_combined
                data_orig['vmax'] = vmax_combined
                data_recon['vmin'] = vmin_combined
                data_recon['vmax'] = vmax_combined
        
        # Generate figures
        fig_orig = plot_markers_topos(computed_data=data_orig) if data_orig else None
        fig_recon = plot_markers_topos(computed_data=data_recon) if data_recon else None
        
        # Create vertically stacked figure
        if fig_orig or fig_recon:
            fig_combined = _create_vertical_stacked_figure(fig_orig, fig_recon,
                                                           f"ORIGINAL: {title_base}",
                                                           f"RECONSTRUCTED: {title_base}")
            report.add_figure(fig_combined, title=f"{title_base} - Comparison", section=section)
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)
    
    # Helper for single marker topo
    def plot_marker_with_matched_scales(path_orig, path_recon, title_base, section):
        if not path_orig.exists() and not path_recon.exists():
            return
        
        data_orig, data_recon = None, None
        if path_orig.exists():
            with open(path_orig, 'rb') as f:
                data_orig = pickle.load(f)
        if path_recon.exists():
            with open(path_recon, 'rb') as f:
                data_recon = pickle.load(f)
        
        if data_orig and data_recon:
            vmin_o, vmax_o = data_orig.get('vmin'), data_orig.get('vmax')
            vmin_r, vmax_r = data_recon.get('vmin'), data_recon.get('vmax')
            
            if vmin_o is not None and vmin_r is not None:
                vmin_combined = min(vmin_o, vmin_r)
                vmax_combined = max(vmax_o, vmax_r)
                data_orig['vmin'] = vmin_combined
                data_orig['vmax'] = vmax_combined
                data_recon['vmin'] = vmin_combined
                data_recon['vmax'] = vmax_combined
        
        # Generate figures
        fig_orig = plot_marker_topo(computed_data=data_orig) if data_orig else None
        fig_recon = plot_marker_topo(computed_data=data_recon) if data_recon else None
        
        # Create vertically stacked figure
        if fig_orig or fig_recon:
            fig_combined = _create_vertical_stacked_figure(fig_orig, fig_recon,
                                                           f"ORIGINAL: {title_base}",
                                                           f"RECONSTRUCTED: {title_base}")
            report.add_figure(fig_combined, title=f"{title_base} - Comparison", section=section)
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)
    
    # 1. WSMI topoplots
    plot_markers_with_matched_scales(
        Path(data_path_original) / "wsmi_bands_topo.pkl",
        Path(data_path_recon) / "wsmi_bands_topo.pkl",
        "WSMI Connectivity per Frequency Band",
        "Connectivity"
    )
    
    # 2. Mutual information
    plot_marker_with_matched_scales(
        Path(data_path_original) / "mutual_info_topo.pkl",
        Path(data_path_recon) / "mutual_info_topo.pkl",
        "Mutual Information Topography",
        "Information Theory"
    )


def _add_information_theory_plots(report, data_path_original, data_path_recon, save_dir=None):
    """Add information theory plots (side-by-side comparison with matched scales)"""
    import pickle
    logger = logging.getLogger(__name__)
    
    # Helper for markers topos
    def plot_markers_with_matched_scales(path_orig, path_recon, title_base, section):
        if not path_orig.exists() and not path_recon.exists():
            return
        
        data_orig, data_recon = None, None
        if path_orig.exists():
            with open(path_orig, 'rb') as f:
                data_orig = pickle.load(f)
        if path_recon.exists():
            with open(path_recon, 'rb') as f:
                data_recon = pickle.load(f)
        
        if data_orig and data_recon:
            vmin_o, vmax_o = data_orig.get('vmin'), data_orig.get('vmax')
            vmin_r, vmax_r = data_recon.get('vmin'), data_recon.get('vmax')
            
            if vmin_o is not None and vmin_r is not None:
                if isinstance(vmin_o, (list, np.ndarray)):
                    vmin_combined = [min(vo, vr) for vo, vr in zip(vmin_o, vmin_r)]
                    vmax_combined = [max(vo, vr) for vo, vr in zip(vmax_o, vmax_r)]
                else:
                    vmin_combined = min(vmin_o, vmin_r)
                    vmax_combined = max(vmax_o, vmax_r)
                
                data_orig['vmin'] = vmin_combined
                data_orig['vmax'] = vmax_combined
                data_recon['vmin'] = vmin_combined
                data_recon['vmax'] = vmax_combined
        
        # Generate figures
        fig_orig = plot_markers_topos(computed_data=data_orig) if data_orig else None
        fig_recon = plot_markers_topos(computed_data=data_recon) if data_recon else None
        
        # Create vertically stacked figure
        if fig_orig or fig_recon:
            fig_combined = _create_vertical_stacked_figure(fig_orig, fig_recon,
                                                           f"ORIGINAL: {title_base}",
                                                           f"RECONSTRUCTED: {title_base}")
            report.add_figure(fig_combined, title=f"{title_base} - Comparison", section=section)
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)
    
    # Helper for single marker topo
    def plot_marker_with_matched_scales(path_orig, path_recon, title_base, section):
        if not path_orig.exists() and not path_recon.exists():
            return
        
        data_orig, data_recon = None, None
        if path_orig.exists():
            with open(path_orig, 'rb') as f:
                data_orig = pickle.load(f)
        if path_recon.exists():
            with open(path_recon, 'rb') as f:
                data_recon = pickle.load(f)
        
        if data_orig and data_recon:
            vmin_o, vmax_o = data_orig.get('vmin'), data_orig.get('vmax')
            vmin_r, vmax_r = data_recon.get('vmin'), data_recon.get('vmax')
            
            if vmin_o is not None and vmin_r is not None:
                vmin_combined = min(vmin_o, vmin_r)
                vmax_combined = max(vmax_o, vmax_r)
                data_orig['vmin'] = vmin_combined
                data_orig['vmax'] = vmax_combined
                data_recon['vmin'] = vmin_combined
                data_recon['vmax'] = vmax_combined
        
        # Generate figures
        fig_orig = plot_marker_topo(computed_data=data_orig) if data_orig else None
        fig_recon = plot_marker_topo(computed_data=data_recon) if data_recon else None
        
        # Create vertically stacked figure
        if fig_orig or fig_recon:
            fig_combined = _create_vertical_stacked_figure(fig_orig, fig_recon,
                                                           f"ORIGINAL: {title_base}",
                                                           f"RECONSTRUCTED: {title_base}")
            report.add_figure(fig_combined, title=f"{title_base} - Comparison", section=section)
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)
    
    # 1. Permutation Entropy
    plot_markers_with_matched_scales(
        Path(data_path_original) / "permutation_entropy_bands.pkl",
        Path(data_path_recon) / "permutation_entropy_bands.pkl",
        "Permutation Entropy per Frequency Band",
        "Information Theory"
    )
    
    # 2. Kolmogorov complexity
    plot_markers_with_matched_scales(
        Path(data_path_original) / "kolmogorov_complexity.pkl",
        Path(data_path_recon) / "kolmogorov_complexity.pkl",
        "Kolmogorov Complexity",
        "Information Theory"
    )
    
    # 3. Generic per-channel measures
    for measure_name in ["kolmogorov_complexity", "permutation_entropy"]:
        plot_marker_with_matched_scales(
            Path(data_path_original) / f"info_theory_{measure_name}.pkl",
            Path(data_path_recon) / f"info_theory_{measure_name}.pkl",
            measure_name.upper(),
            "Information Theory"
        )


def _add_erp_plots(report, epochs, skip_clustering, data_path_original, data_path_recon, save_dir=None):
    """Add ERP plots (side-by-side comparison)"""
    logger = logging.getLogger(__name__)
    
    logger.info("Generating side-by-side ERP plots...")
    
    # Generate figures for both datasets
    figs_orig = _generate_erp_figures(epochs, skip_clustering, data_path_original, "ORIGINAL", save_dir)
    figs_recon = _generate_erp_figures(epochs, skip_clustering, data_path_recon, "RECONSTRUCTED", save_dir)
    
    # Combine and add to report
    for key in figs_orig.keys():
        fig_orig = figs_orig.get(key)
        fig_recon = figs_recon.get(key)
        
        if fig_orig or fig_recon:
            fig_combined = _create_side_by_side_figure(fig_orig, fig_recon,
                                                       f"ORIGINAL: {key}",
                                                       f"RECONSTRUCTED: {key}")
            
            # Save plot
            if save_dir:
                plot_name = f"erp_{key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
                fig_combined.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                logger.info(f"   💾 Saved: {plot_name}")
            
            report.add_figure(fig_combined, title=f"{key} - Comparison", section="ERP")
            plt.close(fig_combined)
            if fig_orig:
                plt.close(fig_orig)
            if fig_recon:
                plt.close(fig_recon)


def _generate_erp_figures(epochs, skip_clustering, data_path, label_prefix, save_dir=None):
    """Helper function to generate ERP figures for a single dataset - returns dict of figures"""
    logger = logging.getLogger(__name__)
    figures = {}
    
    local_labels = ["Local Deviant", "Local Standard"]
    global_labels = ["Global Deviant", "Global Standard"]
    time_shift = -0.6
    plot_times = np.arange(0.64, 1.336, 0.02) + time_shift
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}
    stat_psig = 0.05
    stat_logpsig = -np.log10(stat_psig)
    
    # LOCAL EFFECT
    local_gfp_path = Path(data_path) / "local_effect_gfp.pkl"
    if local_gfp_path.exists():
        fig_gfp = plot_gfp(computed_data_path=local_gfp_path, colors=["r", "b"], labels=local_labels,
                           fig_kwargs=dict(figsize=(12, 6)), sns_kwargs=dict(style="darkgrid"))
        figures["Local Effect - GFP"] = fig_gfp
    else:
        figures["Local Effect - GFP"] = None
    
    local_contrast_path = Path(data_path) / "local_effect_contrast.pkl"
    if local_contrast_path.exists():
        evoked, _, _, local_contrast = get_contrast(computed_data_path=local_contrast_path)
        evoked.shift_time(time_shift)
        
        fig_topo = plot_evoked_topomap(evoked, times=plot_times, ch_type="eeg", contours=0, cmap="RdBu_r",
                                       cbar_fmt="%0.3f", average=0.04, units=r"$\mu{V}$", ncols=10, nrows="auto",
                                       extrapolate="local", sns_kwargs=dict(style="white"))
        figures["Local Effect - Topographies"] = fig_topo
        
        # Statistical color scaling
        local_contrast.mlog10_p_val.shift_time(time_shift)
        stat_data = np.array(local_contrast.mlog10_p_val.data)
        stat_vmin_actual = max(0, np.nanmin(stat_data))
        stat_vmax_actual = max(np.nanmax(stat_data), stat_logpsig + 2.0)
        stat_cmap = get_stat_colormap(stat_logpsig, stat_vmin_actual, stat_vmax_actual)
        
        fig_topo_stat = plot_evoked_topomap(local_contrast.mlog10_p_val, times=plot_times, ch_type="eeg",
                                           contours=0, cmap=stat_cmap, scalings=1, cbar_fmt="%0.3f", average=0.04,
                                           units="-log10(p)", ncols=10, nrows="auto", extrapolate="local",
                                           vlim=(stat_vmin_actual, stat_vmax_actual),
                                           sns_kwargs=dict(style="white"))
        figures["Local Effect - Topographies (-log10(p))"] = fig_topo_stat
        gc.collect()
    else:
        figures["Local Effect - Topographies"] = None
        figures["Local Effect - Topographies (-log10(p))"] = None
    
    # Cluster test
    if not skip_clustering:
        local_cluster_path = Path(data_path) / "local_cluster_test.pkl"
        if local_cluster_path.exists():
            fig_cluster = plot_cluster_test(computed_data_path=local_cluster_path, sns_kwargs={"style": "darkgrid"})
            figures["Local Effect - Cluster"] = fig_cluster
        else:
            figures["Local Effect - Cluster"] = None
    else:
        figures["Local Effect - Cluster"] = None
    
    # GLOBAL EFFECT
    global_gfp_path = Path(data_path) / "global_effect_gfp.pkl"
    if global_gfp_path.exists():
        fig_gfp = plot_gfp(computed_data_path=global_gfp_path, colors=["r", "b"], labels=global_labels,
                           fig_kwargs=dict(figsize=(12, 6)), sns_kwargs=dict(style="darkgrid"))
        figures["Global Effect - GFP"] = fig_gfp
    else:
        figures["Global Effect - GFP"] = None
    
    global_contrast_path = Path(data_path) / "global_effect_contrast.pkl"
    if global_contrast_path.exists():
        evoked, _, _, global_contrast = get_contrast(computed_data_path=global_contrast_path)
        evoked.shift_time(time_shift)
        
        fig_topo = plot_evoked_topomap(evoked, times=plot_times, ch_type="eeg", contours=0, cmap="RdBu_r",
                                       cbar_fmt="%0.3f", average=0.04, units=r"$\mu{V}$", ncols=10, nrows="auto",
                                       extrapolate="local", sns_kwargs=dict(style="white"))
        figures["Global Effect - Topographies"] = fig_topo
        
        # Statistical color scaling
        global_contrast.mlog10_p_val.shift_time(time_shift)
        stat_data_global = np.array(global_contrast.mlog10_p_val.data)
        stat_vmin_global = max(0, np.nanmin(stat_data_global))
        stat_vmax_global = max(np.nanmax(stat_data_global), stat_logpsig + 2.0)
        stat_cmap_global = get_stat_colormap(stat_logpsig, stat_vmin_global, stat_vmax_global)
        
        fig_topo_stat = plot_evoked_topomap(global_contrast.mlog10_p_val, times=plot_times, ch_type="eeg",
                                           contours=0, cmap=stat_cmap_global, scalings=1, cbar_fmt="%0.3f", average=0.04,
                                           units="-log10(p)", ncols=10, nrows="auto", extrapolate="local",
                                           vlim=(stat_vmin_global, stat_vmax_global),
                                           sns_kwargs=dict(style="white"))
        figures["Global Effect - Topographies (-log10(p))"] = fig_topo_stat
        gc.collect()
    else:
        figures["Global Effect - Topographies"] = None
        figures["Global Effect - Topographies (-log10(p))"] = None
    
    # Cluster test
    if not skip_clustering:
        global_cluster_path = Path(data_path) / "global_cluster_test.pkl"
        if global_cluster_path.exists():
            fig_cluster = plot_cluster_test(computed_data_path=global_cluster_path, sns_kwargs={"style": "darkgrid"})
            figures["Global Effect - Cluster"] = fig_cluster
        else:
            figures["Global Effect - Cluster"] = None
    else:
        figures["Global Effect - Cluster"] = None
    
    # Return the dictionary of figures
    return figures


def _add_prediction_plots(report, data_path_original, data_path_recon, save_dir=None):
    """Add prediction plots from precomputed results (side-by-side comparison)"""
    import pickle
    logger = logging.getLogger(__name__)
    
    # Process original predictions
    prediction_path_original = Path(data_path_original) / "prediction_results.pkl"
    if prediction_path_original.exists():
        logger.info("Processing ORIGINAL prediction data...")
        _process_prediction_data(report, prediction_path_original, "ORIGINAL", save_dir)
    
    # Process reconstructed predictions
    prediction_path_recon = Path(data_path_recon) / "prediction_results.pkl"
    if prediction_path_recon.exists():
        logger.info("Processing RECONSTRUCTED prediction data...")
        _process_prediction_data(report, prediction_path_recon, "RECONSTRUCTED", save_dir)
    
    # If neither exists, show info message
    if not prediction_path_original.exists() and not prediction_path_recon.exists():
        logger.warning("No prediction results found - skipping prediction section")
        html = """
        <div class="alert alert-info">
            <h4>Prediction Analysis</h4>
            <p>Prediction analysis requires trained classification models and patient data.</p>
            <p>This section will be populated when real prediction results are available.</p>
        </div>
        """
        report.add_html(html, title="Prediction Summary", section="Prediction")


def _process_prediction_data(report, prediction_path, label_prefix, save_dir=None):
    """Helper function to process prediction data for a single dataset"""
    import pickle
    logger = logging.getLogger(__name__)
    
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
        report.add_html(html, title=f"{label_prefix}: Prediction Summary", section="Prediction")
        
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
            
            # Save plot and data
            if save_dir:
                plot_name = f"prediction_{label_prefix.lower()}_{clf_type}"
                fig_overall.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                np.savez(
                    Path(save_dir) / f"{plot_name}.npz",
                    vs_prob=vs_prob,
                    mcs_prob=mcs_prob,
                    classifier=clf_type
                )
                logger.info(f"   💾 Saved: {plot_name}")
            
            report.add_figure(fig_overall, title=f"{label_prefix}: {clf_display_name}", section='Multivariate')
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
            
            # Save plot and data
            if save_dir:
                plot_name = f"prediction_{label_prefix.lower()}_univariate_violin"
                fig_violin.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                np.savez(
                    Path(save_dir) / f"{plot_name}.npz",
                    univariate_data=univariate.to_dict(),
                    n_markers=len(univariate)
                )
                logger.info(f"   💾 Saved: {plot_name}")
            
            report.add_figure(fig_violin, title=f'{label_prefix}: Univariate Summary I', section='Univariate Summaries')
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
                
                # Save plot and data
                if save_dir:
                    plot_name = f"prediction_{label_prefix.lower()}_univariate_bars"
                    fig_bars.savefig(Path(save_dir) / f"{plot_name}.png", dpi=150, bbox_inches='tight')
                    np.savez(
                        Path(save_dir) / f"{plot_name}.npz",
                        grouped_data={k: v.to_dict() for k, v in grouped_data.items()}
                    )
                    logger.info(f"   💾 Saved: {plot_name}")
                
                report.add_figure(fig_bars, title=f'{label_prefix}: Univariate Summary II', section='Univariate Summaries')
                plt.close(fig_bars)
            
            logger.info("Added univariate prediction visualizations")
        
        logger.info("✅ All prediction plots added to report")
        
    except Exception as e:
        logger.error(f"Failed to create prediction plots: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
