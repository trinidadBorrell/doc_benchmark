"""Control RS Plots for CBraMod - Original vs Reconstructed Topographic Comparison

Creates a grid topographic comparison plot showing:
- Column 1: Original topos (mean across subjects)
- Column 2: Reconstructed topos (mean across subjects)
- Column 3: Relative difference ((recon - orig) / orig)

Rows: Biomarkers available in the data

Data paths:
- Original: /data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data/sub-{ID}/ses-01/orig/topos_sub-{ID}_ses-01.npz
- Reconstructed: /data/project/eeg_foundation/src/doc_benchmark/results/CBraMod/MARKERS/computed_data/sub-{ID}/ses-01/recon/topos_sub-{ID}_ses-01.npz

Author: Generated script
"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import MNE for topographic plotting
try:
    import mne

    HAS_MNE = True
    mne.set_log_level("WARNING")
except ImportError as e:
    HAS_MNE = False
    print(
        f"Warning: MNE-Python not available. Topographic plots will be skipped. Error: {e}"
    )

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 22,
        "axes.labelsize": "large",
        "ytick.labelsize": 22,
        "xtick.labelsize": 22,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)


# Marker display names mapping
MARKER_DISPLAY_NAMES = {
    "alpha_relative_spectralpower": "Alpha Normalized",
    "beta_relative_spectralpower": "Beta Normalized",
    "delta_relative_spectralpower": "Delta Normalized",
    "gamma_relative_spectralpower": "Gamma Normalized",
    "theta_relative_spectralpower": "Theta Normalized",
    "pe_theta_permutationentropy": "Permutation Entropy",
    "spectral_entropy_spectralpower": "Spectral Entropy",
    "kolmogorov_complexity_kolmogorovcomplexity": "Kolmogorov Complexity",
    "wsmi_theta_symbolicmutualinformation": "Symbolic Mutual Information",
    "alpha_power_spectralpower": "Alpha Power",
    "beta_power_spectralpower": "Beta Power",
    "delta_power_spectralpower": "Delta Power",
    "gamma_power_spectralpower": "Gamma Power",
    "theta_power_spectralpower": "Theta Power",
    "msf_psdsummary": "Mean Spectral Frequency",
    "sef90_psdsummary": "SEF90",
    "sef95_psdsummary": "SEF95",
}


def _prepare_egi256_sphere_and_outlines(evoked):
    """Prepare sphere and outlines for EGI-256 topographic plotting."""

    _egi256_outlines = {
        "ear1": np.array([190, 191, 201, 209, 218, 217, 216, 208, 200, 190]),
        "ear2": np.array([81, 72, 66, 67, 68, 73, 82, 92, 91, 81]),
        "outer": np.array(
            [
                9,
                17,
                24,
                30,
                31,
                36,
                45,
                243,
                240,
                241,
                242,
                246,
                250,
                255,
                90,
                101,
                110,
                119,
                132,
                144,
                164,
                173,
                186,
                198,
                207,
                215,
                228,
                232,
                236,
                239,
                238,
                237,
                233,
                9,
            ]
        ),
    }

    sphere_ch_names = ["E137", "E26", "E69", "E202"]
    ch_names = evoked.ch_names
    ch_idx = [ch_names.index(ch) for ch in sphere_ch_names if ch in ch_names]

    if len(ch_idx) == 4:
        pos_3d = np.stack([evoked.info["chs"][idx]["loc"][:3] for idx in ch_idx])
        radius = np.abs(pos_3d[[2, 3], 0]).mean()
        x = pos_3d[0, 0]
        y = pos_3d[-1, 1]
        z = pos_3d[:, -1].mean()
        sphere = (x, y, z, radius)

    # Get 2D positions for topomap
    _, pos, _, _, _, this_sphere, clip_origin = mne.viz.topomap._prepare_topomap_plot(
        evoked.info, "eeg", sphere=sphere
    )

    # Build the outlines dictionary properly
    outlines = {}
    codes = []
    vertices = []
    for k, v in _egi256_outlines.items():
        t_verts = pos[v, :]
        outlines[k] = (t_verts[:, 0], t_verts[:, 1])
        t_codes = 2 * np.ones(v.shape[0])
        t_codes[0] = 1
        codes.append(t_codes)
        vertices.append(t_verts)

    vertices = np.concatenate(vertices, axis=0)
    codes = np.concatenate(codes, axis=0)

    # Add all required keys for MNE
    outlines["mask_pos"] = outlines["outer"]
    outlines["clip_radius"] = clip_origin

    # Create path patch
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    path = Path(vertices=vertices, codes=codes)

    def patch():
        return PathPatch(path, alpha=0.1)

    outlines["patch"] = patch

    return this_sphere, outlines


def _setup_montage_and_sphere(n_channels, topos_mean=None):
    """Set up MNE montage, info object, sphere, and outlines for topographic plotting."""

    if n_channels == 256:
        print("  Setting up EGI-256 montage with custom sphere and outlines")
        montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
        info = mne.create_info(montage.ch_names, 250, ch_types="eeg")
        info.set_montage(montage, on_missing="warn")

        if topos_mean is not None:
            evoked = mne.EvokedArray(topos_mean.T, info, tmin=0)
            sphere, outlines = _prepare_egi256_sphere_and_outlines(evoked)
        else:
            sphere = "auto"
            outlines = "head"

    elif n_channels == 128:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        info = mne.create_info(montage.ch_names, 250, ch_types="eeg")
        info.set_montage(montage)
        sphere = "auto"
        outlines = "head"

    elif n_channels == 64:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-64_1.0")
        info = mne.create_info(montage.ch_names, 250, ch_types="eeg")
        info.set_montage(montage)
        sphere = "auto"
        outlines = "head"

    else:
        print(
            f"  Warning: No standard montage for {n_channels} channels, creating generic layout"
        )
        ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names, 100, "eeg")
        from mne.channels.layout import _auto_topomap_coords

        pos = _auto_topomap_coords(info, picks=None, sphere=None, ignore_overlap=True)
        montage_dict = dict(zip(ch_names, pos))
        montage = mne.channels.make_dig_montage(montage_dict, coord_frame="head")
        info.set_montage(montage)
        sphere = "auto"
        outlines = "head"

    return info, sphere, outlines


def load_control_rs_data(orig_base_dir, recon_base_dir):
    """Load topographic data for Control RS subjects from original and CBraMod reconstructed paths."""

    print("=" * 60)
    print("Loading Control RS Data for CBraMod Comparison")
    print("=" * 60)
    print(f"Original data: {orig_base_dir}")
    print(f"Reconstructed data: {recon_base_dir}")

    if not op.exists(orig_base_dir):
        print(f"   Error: Original directory not found: {orig_base_dir}")
        return None

    if not op.exists(recon_base_dir):
        print(f"   Error: Reconstructed directory not found: {recon_base_dir}")
        return None

    # Find all subjects in original directory
    subject_dirs = [d for d in os.listdir(orig_base_dir) if d.startswith("sub-")]
    print(f"Found {len(subject_dirs)} subject directories in original data")

    subjects_data = []

    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.replace("sub-", "")

        # Build file paths
        orig_file = op.join(
            orig_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "orig",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        recon_file = op.join(
            recon_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )

        # Check if both files exist
        if not op.exists(orig_file):
            print(f"   Skipping {subject_id}: original file not found")
            continue
        if not op.exists(recon_file):
            print(f"   Skipping {subject_id}: reconstructed file not found")
            continue

        try:
            # Load data
            orig_data = np.load(orig_file)
            recon_data = np.load(recon_file)

            # Get marker names
            orig_markers = sorted(orig_data.files)
            recon_markers = sorted(recon_data.files)

            # Find common markers
            common_markers = sorted(set(orig_markers) & set(recon_markers))

            if not common_markers:
                print(f"   Skipping {subject_id}: no common markers")
                continue

            # Extract data for common markers
            topos_orig = np.array([orig_data[m] for m in common_markers])
            topos_recon = np.array([recon_data[m] for m in common_markers])

            # Validate shapes
            if topos_orig.shape != topos_recon.shape:
                print(
                    f"   Skipping {subject_id}: shape mismatch orig={topos_orig.shape} vs recon={topos_recon.shape}"
                )
                continue

            subjects_data.append(
                {
                    "subject_id": subject_id,
                    "topos_original": topos_orig,
                    "topos_reconstructed": topos_recon,
                    "marker_names": common_markers,
                    "n_channels": topos_orig.shape[1],
                }
            )

        except Exception as e:
            print(f"   Error loading {subject_id}: {e}")
            continue

    if not subjects_data:
        print("No valid subjects found!")
        return None

    print(f"Successfully loaded {len(subjects_data)} subjects")

    # Find common markers across all subjects
    all_marker_sets = [set(s["marker_names"]) for s in subjects_data]
    common_markers_all = sorted(set.intersection(*all_marker_sets))

    print(f"Common markers across all subjects: {len(common_markers_all)}")
    for m in common_markers_all:
        print(f"   - {m}")

    # Filter subjects to those with consistent channel count
    channel_counts = [s["n_channels"] for s in subjects_data]
    most_common_channels = max(set(channel_counts), key=channel_counts.count)

    filtered_subjects = [
        s for s in subjects_data if s["n_channels"] == most_common_channels
    ]
    print(f"Subjects with {most_common_channels} channels: {len(filtered_subjects)}")

    if not filtered_subjects:
        print("No subjects with consistent channel count!")
        return None

    # Stack data for common markers
    n_subjects = len(filtered_subjects)
    n_markers = len(common_markers_all)
    n_channels = most_common_channels

    topos_orig_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_recon_all = np.zeros((n_subjects, n_markers, n_channels))

    for i, subj in enumerate(filtered_subjects):
        for j, marker in enumerate(common_markers_all):
            marker_idx = subj["marker_names"].index(marker)
            topos_orig_all[i, j, :] = subj["topos_original"][marker_idx]
            topos_recon_all[i, j, :] = subj["topos_reconstructed"][marker_idx]

    return {
        "topos_orig": topos_orig_all,
        "topos_recon": topos_recon_all,
        "marker_names": common_markers_all,
        "n_subjects": n_subjects,
        "n_channels": n_channels,
        "subject_ids": [s["subject_id"] for s in filtered_subjects],
    }


def create_control_rs_comparison_grid(output_dir, biomarker_filter=None):
    """Create grid plot: Original | Reconstructed | Relative Difference.

    Args:
        output_dir: Directory to save the output plot
        biomarker_filter: Optional list of biomarker names to include. If None, uses all available.
    """

    print("=" * 60)
    print("Creating Control RS CBraMod Comparison Grid")
    print("=" * 60)

    # Define data directories
    orig_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data"
    recon_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/CBraMod/MARKERS/computed_data"

    # Load data
    data = load_control_rs_data(orig_base_dir, recon_base_dir)

    if data is None:
        print("Failed to load data. Exiting.")
        return None

    # Get marker names and filter if specified
    marker_names = data["marker_names"]

    # Exclude absolute power markers (keep only normalized/relative ones)
    excluded_markers = [
        "alpha_power_spectralpower",
        "beta_power_spectralpower",
        "delta_power_spectralpower",
        "gamma_power_spectralpower",
        "theta_power_spectralpower",
    ]

    if biomarker_filter:
        # Filter to specified biomarkers that exist in data
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in biomarker_filter
            if m in marker_names and m not in excluded_markers
        ]
    else:
        # Use all available markers except excluded ones
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in marker_names
            if m not in excluded_markers
        ]

    if not available_biomarkers:
        print("No biomarkers available after filtering. Exiting.")
        return None

    print(f"Plotting {len(available_biomarkers)} biomarkers")

    # Compute mean across subjects
    topos_orig_mean = np.mean(data["topos_orig"], axis=0)  # (n_markers, n_channels)
    topos_recon_mean = np.mean(data["topos_recon"], axis=0)  # (n_markers, n_channels)

    # Compute relative difference: (recon - orig) / |orig|
    # Use small epsilon to avoid division by zero
    eps = 1e-10
    relative_diff = (topos_recon_mean - topos_orig_mean) / (
        np.abs(topos_orig_mean) + eps
    )

    # Set up montage
    info, sphere, outlines = _setup_montage_and_sphere(
        data["n_channels"], topos_orig_mean
    )

    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_cols = 3  # Original, Reconstructed, Relative Difference

    fig, axes = plt.subplots(
        n_biomarkers, n_cols, figsize=(15, max(12, n_biomarkers * 2))
    )

    # Handle single row case
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)

    # Add column titles
    col_titles = ["Original", "Reconstructed", "Relative Difference"]
    for col, title in enumerate(col_titles):
        axes[0, col].text(
            0.5,
            1.15,
            title,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=24,
        )

    # Store images for colorbars
    orig_recon_images = []
    diff_images = []

    # Plot each biomarker
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        marker_idx = marker_names.index(marker_name)

        # Add row label
        axes[row, 0].text(
            -0.25,
            0.5,
            display_name,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=18,
            rotation=0,
        )

        # Get data for this marker
        orig_data = topos_orig_mean[marker_idx]
        recon_data = topos_recon_mean[marker_idx]
        diff_data = relative_diff[marker_idx]

        # Compute shared vmin/vmax for original and reconstructed
        vmin_shared = min(np.nanmin(orig_data), np.nanmin(recon_data))
        vmax_shared = max(np.nanmax(orig_data), np.nanmax(recon_data))

        # Compute vmin/vmax for relative difference (symmetric around 0)
        diff_abs_max = np.nanmax(np.abs(diff_data))
        vmin_diff = -diff_abs_max
        vmax_diff = diff_abs_max

        # Plot Original (column 0)
        im_orig, _ = mne.viz.plot_topomap(
            orig_data,
            info,
            axes=axes[row, 0],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 0].set_title("")

        # Plot Reconstructed (column 1)
        im_recon, _ = mne.viz.plot_topomap(
            recon_data,
            info,
            axes=axes[row, 1],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 1].set_title("")

        # Plot Relative Difference (column 2)
        im_diff, _ = mne.viz.plot_topomap(
            diff_data,
            info,
            axes=axes[row, 2],
            vlim=(vmin_diff, vmax_diff),
            cmap="RdBu_r",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 2].set_title("")

        # Store for colorbars (use last row's images)
        if row == n_biomarkers - 1:
            orig_recon_images.append((im_orig, vmin_shared, vmax_shared))
            diff_images.append((im_diff, vmin_diff, vmax_diff))

        # Add individual colorbars for each row
        # Colorbar for Original/Reconstructed (shared scale)
        cbar_orig = plt.colorbar(
            im_orig, ax=axes[row, 1], shrink=0.7, aspect=15, pad=0.02
        )
        cbar_orig.ax.tick_params(labelsize=10)

        # Colorbar for Relative Difference
        cbar_diff = plt.colorbar(
            im_diff, ax=axes[row, 2], shrink=0.7, aspect=15, pad=0.02
        )
        cbar_diff.ax.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.15)

    # Add subtitle with subject count
    # fig.suptitle(f'Control RS: Original vs CBraMod Reconstructed (N={data["n_subjects"]} subjects)',
    #              fontsize=20, y=1.02)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, "control_rs_CBraMod_comparison_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
    return output_path


def create_control_rs_comparison_grid_selected(output_dir):
    """Create grid plot with selected biomarkers commonly used in analysis."""

    # Define selected biomarkers (same as in ohbm_biomarker_group_comparison.py)
    selected_biomarkers = [
        "alpha_relative_spectralpower",
        "beta_relative_spectralpower",
        "delta_relative_spectralpower",
        "gamma_relative_spectralpower",
        "theta_relative_spectralpower",
        "pe_theta_permutationentropy",
        "spectral_entropy_spectralpower",
        "kolmogorov_complexity_kolmogorovcomplexity",
        "wsmi_theta_symbolicmutualinformation",
    ]

    return create_control_rs_comparison_grid(
        output_dir, biomarker_filter=selected_biomarkers
    )


def load_control_rs_data_three_sources(orig_base_dir, cbramod_base_dir, totem_base_dir):
    """Load topographic data for Control RS subjects from original, CBraMod, and TOTEM paths."""

    print("=" * 60)
    print("Loading Control RS Data for Three-Way Comparison")
    print("=" * 60)
    print(f"Original data: {orig_base_dir}")
    print(f"CBraMod reconstructed: {cbramod_base_dir}")
    print(f"TOTEM reconstructed: {totem_base_dir}")

    for base_dir, name in [
        (orig_base_dir, "Original"),
        (cbramod_base_dir, "CBraMod"),
        (totem_base_dir, "TOTEM"),
    ]:
        if not op.exists(base_dir):
            print(f"   Error: {name} directory not found: {base_dir}")
            return None

    # Find all subjects in original directory
    subject_dirs = [d for d in os.listdir(orig_base_dir) if d.startswith("sub-")]
    print(f"Found {len(subject_dirs)} subject directories in original data")

    subjects_data = []

    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.replace("sub-", "")

        # Build file paths
        orig_file = op.join(
            orig_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "orig",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        cbramod_file = op.join(
            cbramod_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        totem_file = op.join(
            totem_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )

        # Check if all files exist
        if not op.exists(orig_file):
            print(f"   Skipping {subject_id}: original file not found")
            continue
        if not op.exists(cbramod_file):
            print(f"   Skipping {subject_id}: CBraMod file not found")
            continue
        if not op.exists(totem_file):
            print(f"   Skipping {subject_id}: TOTEM file not found")
            continue

        try:
            # Load data
            orig_data = np.load(orig_file)
            cbramod_data = np.load(cbramod_file)
            totem_data = np.load(totem_file)

            # Get marker names
            orig_markers = sorted(orig_data.files)
            cbramod_markers = sorted(cbramod_data.files)
            totem_markers = sorted(totem_data.files)

            # Find common markers across all three
            common_markers = sorted(
                set(orig_markers) & set(cbramod_markers) & set(totem_markers)
            )

            if not common_markers:
                print(f"   Skipping {subject_id}: no common markers")
                continue

            # Extract data for common markers
            topos_orig = np.array([orig_data[m] for m in common_markers])
            topos_cbramod = np.array([cbramod_data[m] for m in common_markers])
            topos_totem = np.array([totem_data[m] for m in common_markers])

            # Validate shapes
            if not (topos_orig.shape == topos_cbramod.shape == topos_totem.shape):
                print(f"   Skipping {subject_id}: shape mismatch")
                continue

            subjects_data.append(
                {
                    "subject_id": subject_id,
                    "topos_original": topos_orig,
                    "topos_cbramod": topos_cbramod,
                    "topos_totem": topos_totem,
                    "marker_names": common_markers,
                    "n_channels": topos_orig.shape[1],
                }
            )

        except Exception as e:
            print(f"   Error loading {subject_id}: {e}")
            continue

    if not subjects_data:
        print("No valid subjects found!")
        return None

    print(f"Successfully loaded {len(subjects_data)} subjects")

    # Find common markers across all subjects
    all_marker_sets = [set(s["marker_names"]) for s in subjects_data]
    common_markers_all = sorted(set.intersection(*all_marker_sets))

    print(f"Common markers across all subjects: {len(common_markers_all)}")
    for m in common_markers_all:
        print(f"   - {m}")

    # Filter subjects to those with consistent channel count
    channel_counts = [s["n_channels"] for s in subjects_data]
    most_common_channels = max(set(channel_counts), key=channel_counts.count)

    filtered_subjects = [
        s for s in subjects_data if s["n_channels"] == most_common_channels
    ]
    print(f"Subjects with {most_common_channels} channels: {len(filtered_subjects)}")

    if not filtered_subjects:
        print("No subjects with consistent channel count!")
        return None

    # Stack data for common markers
    n_subjects = len(filtered_subjects)
    n_markers = len(common_markers_all)
    n_channels = most_common_channels

    topos_orig_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_cbramod_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_totem_all = np.zeros((n_subjects, n_markers, n_channels))

    for i, subj in enumerate(filtered_subjects):
        for j, marker in enumerate(common_markers_all):
            marker_idx = subj["marker_names"].index(marker)
            topos_orig_all[i, j, :] = subj["topos_original"][marker_idx]
            topos_cbramod_all[i, j, :] = subj["topos_cbramod"][marker_idx]
            topos_totem_all[i, j, :] = subj["topos_totem"][marker_idx]

    return {
        "topos_orig": topos_orig_all,
        "topos_cbramod": topos_cbramod_all,
        "topos_totem": topos_totem_all,
        "marker_names": common_markers_all,
        "n_subjects": n_subjects,
        "n_channels": n_channels,
        "subject_ids": [s["subject_id"] for s in filtered_subjects],
    }


def create_control_rs_three_way_comparison_grid(output_dir, biomarker_filter=None):
    """Create grid plot: Original | CBraMod Reconstructed | TOTEM Reconstructed.

    All three columns share the same colormap scale per row.

    Args:
        output_dir: Directory to save the output plot
        biomarker_filter: Optional list of biomarker names to include. If None, uses all available.
    """

    print("=" * 60)
    print("Creating Control RS Three-Way Comparison Grid")
    print("=" * 60)

    # Define data directories
    orig_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data"
    cbramod_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/CBraMod/MARKERS/computed_data"
    totem_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data"

    # Load data
    data = load_control_rs_data_three_sources(
        orig_base_dir, cbramod_base_dir, totem_base_dir
    )

    if data is None:
        print("Failed to load data. Exiting.")
        return None

    # Get marker names and filter if specified
    marker_names = data["marker_names"]

    # Exclude absolute power markers (keep only normalized/relative ones)
    excluded_markers = [
        "alpha_power_spectralpower",
        "beta_power_spectralpower",
        "delta_power_spectralpower",
        "gamma_power_spectralpower",
        "theta_power_spectralpower",
    ]

    if biomarker_filter:
        # Filter to specified biomarkers that exist in data
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in biomarker_filter
            if m in marker_names and m not in excluded_markers
        ]
    else:
        # Use all available markers except excluded ones
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in marker_names
            if m not in excluded_markers
        ]

    if not available_biomarkers:
        print("No biomarkers available after filtering. Exiting.")
        return None

    print(f"Plotting {len(available_biomarkers)} biomarkers")

    # Compute mean across subjects
    topos_orig_mean = np.mean(data["topos_orig"], axis=0)  # (n_markers, n_channels)
    topos_cbramod_mean = np.mean(
        data["topos_cbramod"], axis=0
    )  # (n_markers, n_channels)
    topos_totem_mean = np.mean(data["topos_totem"], axis=0)  # (n_markers, n_channels)

    # Set up montage
    info, sphere, outlines = _setup_montage_and_sphere(
        data["n_channels"], topos_orig_mean
    )

    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_cols = 3  # Original, CBraMod, TOTEM

    fig, axes = plt.subplots(
        n_biomarkers, n_cols, figsize=(15, max(12, n_biomarkers * 2))
    )

    # Handle single row case
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)

    # Add column titles
    col_titles = ["Original", "Reconstructed \n (TOTEM)", "Reconstructed \n (CBraMod)"]
    for col, title in enumerate(col_titles):
        axes[0, col].text(
            0.5,
            1.15,
            title,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=24,
        )

    # Plot each biomarker
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        marker_idx = marker_names.index(marker_name)

        # Add row label
        axes[row, 0].text(
            -0.25,
            0.5,
            display_name,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=18,
            rotation=0,
        )

        # Get data for this marker
        orig_data = topos_orig_mean[marker_idx]
        cbramod_data = topos_cbramod_mean[marker_idx]
        totem_data = topos_totem_mean[marker_idx]

        # Compute shared vmin/vmax for all three columns
        vmin_shared = min(
            np.nanmin(orig_data), np.nanmin(cbramod_data), np.nanmin(totem_data)
        )
        vmax_shared = max(
            np.nanmax(orig_data), np.nanmax(cbramod_data), np.nanmax(totem_data)
        )

        # Plot Original (column 0)
        im_orig, _ = mne.viz.plot_topomap(
            orig_data,
            info,
            axes=axes[row, 0],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 0].set_title("")

        # Plot CBraMod Reconstructed (column 2)
        im_cbramod, _ = mne.viz.plot_topomap(
            cbramod_data,
            info,
            axes=axes[row, 2],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 2].set_title("")

        # Plot TOTEM Reconstructed (column 1)
        im_totem, _ = mne.viz.plot_topomap(
            totem_data,
            info,
            axes=axes[row, 1],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 1].set_title("")

        # Add colorbar for the row (on rightmost column)
        cbar = plt.colorbar(im_totem, ax=axes[row, 2], shrink=0.7, aspect=15, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.15)

    # Add subtitle with subject count
    # fig.suptitle(f'Control RS: Original vs CBraMod vs TOTEM (N={data["n_subjects"]} subjects)',
    #             fontsize=20, y=1.02)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, "control_rs_three_way_comparison_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
    return output_path


def create_control_rs_three_way_comparison_grid_selected(output_dir):
    """Create three-way grid plot with selected biomarkers."""

    selected_biomarkers = [
        "alpha_relative_spectralpower",
        "beta_relative_spectralpower",
        "delta_relative_spectralpower",
        "gamma_relative_spectralpower",
        "theta_relative_spectralpower",
        "pe_theta_permutationentropy",
        "spectral_entropy_spectralpower",
        "kolmogorov_complexity_kolmogorovcomplexity",
        "wsmi_theta_symbolicmutualinformation",
    ]

    return create_control_rs_three_way_comparison_grid(
        output_dir, biomarker_filter=selected_biomarkers
    )


def load_control_rs_data_four_sources(
    orig_base_dir, cbramod_base_dir, totem_base_dir, labram_base_dir
):
    """Load topographic data for Control RS subjects from original, CBraMod, TOTEM, and LaBram paths."""

    print("=" * 60)
    print("Loading Control RS Data for Four-Way Comparison")
    print("=" * 60)
    print(f"Original data: {orig_base_dir}")
    print(f"CBraMod reconstructed: {cbramod_base_dir}")
    print(f"TOTEM reconstructed: {totem_base_dir}")
    print(f"LaBram reconstructed: {labram_base_dir}")

    for base_dir, name in [
        (orig_base_dir, "Original"),
        (cbramod_base_dir, "CBraMod"),
        (totem_base_dir, "TOTEM"),
        (labram_base_dir, "LaBram"),
    ]:
        if not op.exists(base_dir):
            print(f"   Error: {name} directory not found: {base_dir}")
            return None

    # Find all subjects in original directory
    subject_dirs = [d for d in os.listdir(orig_base_dir) if d.startswith("sub-")]
    print(f"Found {len(subject_dirs)} subject directories in original data")

    subjects_data = []

    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.replace("sub-", "")

        # Build file paths
        orig_file = op.join(
            orig_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "orig",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        cbramod_file = op.join(
            cbramod_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        totem_file = op.join(
            totem_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )
        labram_file = op.join(
            labram_base_dir,
            f"sub-{subject_id}",
            "ses-01",
            "recon",
            f"topos_sub-{subject_id}_ses-01.npz",
        )

        # Check if all files exist
        if not op.exists(orig_file):
            print(f"   Skipping {subject_id}: original file not found")
            continue
        if not op.exists(cbramod_file):
            print(f"   Skipping {subject_id}: CBraMod file not found")
            continue
        if not op.exists(totem_file):
            print(f"   Skipping {subject_id}: TOTEM file not found")
            continue
        if not op.exists(labram_file):
            print(f"   Skipping {subject_id}: LaBram file not found")
            continue

        try:
            # Load data
            orig_data = np.load(orig_file)
            cbramod_data = np.load(cbramod_file)
            totem_data = np.load(totem_file)
            labram_data = np.load(labram_file)

            # Get marker names
            orig_markers = sorted(orig_data.files)
            cbramod_markers = sorted(cbramod_data.files)
            totem_markers = sorted(totem_data.files)
            labram_markers = sorted(labram_data.files)

            # Find common markers across all four
            common_markers = sorted(
                set(orig_markers)
                & set(cbramod_markers)
                & set(totem_markers)
                & set(labram_markers)
            )

            if not common_markers:
                print(f"   Skipping {subject_id}: no common markers")
                continue

            # Extract data for common markers
            topos_orig = np.array([orig_data[m] for m in common_markers])
            topos_cbramod = np.array([cbramod_data[m] for m in common_markers])
            topos_totem = np.array([totem_data[m] for m in common_markers])
            topos_labram = np.array([labram_data[m] for m in common_markers])

            # Validate shapes
            if not (
                topos_orig.shape
                == topos_cbramod.shape
                == topos_totem.shape
                == topos_labram.shape
            ):
                print(f"   Skipping {subject_id}: shape mismatch")
                continue

            subjects_data.append(
                {
                    "subject_id": subject_id,
                    "topos_original": topos_orig,
                    "topos_cbramod": topos_cbramod,
                    "topos_totem": topos_totem,
                    "topos_labram": topos_labram,
                    "marker_names": common_markers,
                    "n_channels": topos_orig.shape[1],
                }
            )

        except Exception as e:
            print(f"   Error loading {subject_id}: {e}")
            continue

    if not subjects_data:
        print("No valid subjects found!")
        return None

    print(f"Successfully loaded {len(subjects_data)} subjects")

    # Find common markers across all subjects
    all_marker_sets = [set(s["marker_names"]) for s in subjects_data]
    common_markers_all = sorted(set.intersection(*all_marker_sets))

    print(f"Common markers across all subjects: {len(common_markers_all)}")
    for m in common_markers_all:
        print(f"   - {m}")

    # Filter subjects to those with consistent channel count
    channel_counts = [s["n_channels"] for s in subjects_data]
    most_common_channels = max(set(channel_counts), key=channel_counts.count)

    filtered_subjects = [
        s for s in subjects_data if s["n_channels"] == most_common_channels
    ]
    print(f"Subjects with {most_common_channels} channels: {len(filtered_subjects)}")

    if not filtered_subjects:
        print("No subjects with consistent channel count!")
        return None

    # Stack data for common markers
    n_subjects = len(filtered_subjects)
    n_markers = len(common_markers_all)
    n_channels = most_common_channels

    topos_orig_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_cbramod_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_totem_all = np.zeros((n_subjects, n_markers, n_channels))
    topos_labram_all = np.zeros((n_subjects, n_markers, n_channels))

    for i, subj in enumerate(filtered_subjects):
        for j, marker in enumerate(common_markers_all):
            marker_idx = subj["marker_names"].index(marker)
            topos_orig_all[i, j, :] = subj["topos_original"][marker_idx]
            topos_cbramod_all[i, j, :] = subj["topos_cbramod"][marker_idx]
            topos_totem_all[i, j, :] = subj["topos_totem"][marker_idx]
            topos_labram_all[i, j, :] = subj["topos_labram"][marker_idx]

    return {
        "topos_orig": topos_orig_all,
        "topos_cbramod": topos_cbramod_all,
        "topos_totem": topos_totem_all,
        "topos_labram": topos_labram_all,
        "marker_names": common_markers_all,
        "n_subjects": n_subjects,
        "n_channels": n_channels,
        "subject_ids": [s["subject_id"] for s in filtered_subjects],
    }


def create_control_rs_four_way_comparison_grid(output_dir, biomarker_filter=None):
    """Create grid plot: Original | CBraMod Reconstructed | TOTEM Reconstructed | LaBram Reconstructed.

    All four columns share the same colormap scale per row.

    Args:
        output_dir: Directory to save the output plot
        biomarker_filter: Optional list of biomarker names to include. If None, uses all available.
    """

    print("=" * 60)
    print("Creating Control RS Four-Way Comparison Grid")
    print("=" * 60)

    # Define data directories
    orig_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data"
    cbramod_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/CBraMod/MARKERS/computed_data"
    totem_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/computed_data"
    labram_base_dir = "/data/project/eeg_foundation/src/doc_benchmark/results/LaBram/MARKERS/computed_data"

    # Load data
    data = load_control_rs_data_four_sources(
        orig_base_dir, cbramod_base_dir, totem_base_dir, labram_base_dir
    )

    if data is None:
        print("Failed to load data. Exiting.")
        return None

    # Get marker names and filter if specified
    marker_names = data["marker_names"]

    # Exclude absolute power markers (keep only normalized/relative ones)
    excluded_markers = [
        "alpha_power_spectralpower",
        "beta_power_spectralpower",
        "delta_power_spectralpower",
        "gamma_power_spectralpower",
        "theta_power_spectralpower",
    ]

    if biomarker_filter:
        # Filter to specified biomarkers that exist in data
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in biomarker_filter
            if m in marker_names and m not in excluded_markers
        ]
    else:
        # Use all available markers except excluded ones
        available_biomarkers = [
            (m, MARKER_DISPLAY_NAMES.get(m, m))
            for m in marker_names
            if m not in excluded_markers
        ]

    if not available_biomarkers:
        print("No biomarkers available after filtering. Exiting.")
        return None

    print(f"Plotting {len(available_biomarkers)} biomarkers")

    # Compute mean across subjects
    topos_orig_mean = np.mean(data["topos_orig"], axis=0)  # (n_markers, n_channels)
    topos_cbramod_mean = np.mean(
        data["topos_cbramod"], axis=0
    )  # (n_markers, n_channels)
    topos_totem_mean = np.mean(data["topos_totem"], axis=0)  # (n_markers, n_channels)
    topos_labram_mean = np.mean(data["topos_labram"], axis=0)  # (n_markers, n_channels)

    # Set up montage
    info, sphere, outlines = _setup_montage_and_sphere(
        data["n_channels"], topos_orig_mean
    )

    # Create figure
    n_biomarkers = len(available_biomarkers)
    n_cols = 4  # Original, CBraMod, TOTEM, LaBram

    fig, axes = plt.subplots(
        n_biomarkers, n_cols, figsize=(20, max(12, n_biomarkers * 2))
    )

    # Handle single row case
    if n_biomarkers == 1:
        axes = axes.reshape(1, -1)

    # Add column titles
    col_titles = [
        "Original",
        "Reconstructed (TOTEM)",
        "Reconstructed (CBraMod)",
        "Reconstructed (LaBram)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].text(
            0.5,
            1.15,
            title,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
        )

    # Plot each biomarker
    for row, (marker_name, display_name) in enumerate(available_biomarkers):
        marker_idx = marker_names.index(marker_name)

        # Add row label
        axes[row, 0].text(
            -0.25,
            0.5,
            display_name,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=22,
            rotation=0,
        )

        # Get data for this marker
        orig_data = topos_orig_mean[marker_idx]
        cbramod_data = topos_cbramod_mean[marker_idx]
        totem_data = topos_totem_mean[marker_idx]
        labram_data = topos_labram_mean[marker_idx]

        # Compute shared vmin/vmax for all four columns
        vmin_shared = min(
            np.nanmin(orig_data),
            np.nanmin(cbramod_data),
            np.nanmin(totem_data),
            np.nanmin(labram_data),
        )
        vmax_shared = max(
            np.nanmax(orig_data),
            np.nanmax(cbramod_data),
            np.nanmax(totem_data),
            np.nanmax(labram_data),
        )

        # Plot Original (column 0)
        im_orig, _ = mne.viz.plot_topomap(
            orig_data,
            info,
            axes=axes[row, 0],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 0].set_title("")

        # Plot CBraMod Reconstructed (column 2)
        im_cbramod, _ = mne.viz.plot_topomap(
            cbramod_data,
            info,
            axes=axes[row, 2],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 2].set_title("")

        # Plot TOTEM Reconstructed (column 2)
        im_totem, _ = mne.viz.plot_topomap(
            totem_data,
            info,
            axes=axes[row, 1],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 1].set_title("")

        # Plot LaBram Reconstructed (column 3)
        im_labram, _ = mne.viz.plot_topomap(
            labram_data,
            info,
            axes=axes[row, 3],
            vlim=(vmin_shared, vmax_shared),
            cmap="viridis",
            show=False,
            sphere=sphere,
            outlines=outlines,
            extrapolate="local",
            res=256,
            sensors=False,
            contours=0,
        )
        axes[row, 3].set_title("")

        # Add colorbar for the row (on rightmost column)
        cbar = plt.colorbar(im_labram, ax=axes[row, 3], shrink=0.7, aspect=15, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.12)

    # Add subtitle with subject count
    # fig.suptitle(f'Control RS: Original vs CBraMod vs TOTEM vs LaBram (N={data["n_subjects"]} subjects)',
    #              fontsize=18, y=1.02)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = op.join(output_dir, "control_rs_four_way_comparison_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
    return output_path


def create_control_rs_four_way_comparison_grid_selected(output_dir):
    """Create four-way grid plot with selected biomarkers."""

    selected_biomarkers = [
        "alpha_relative_spectralpower",
        "beta_relative_spectralpower",
        "delta_relative_spectralpower",
        "gamma_relative_spectralpower",
        "theta_relative_spectralpower",
        "pe_theta_permutationentropy",
        "spectral_entropy_spectralpower",
        "kolmogorov_complexity_kolmogorovcomplexity",
        "wsmi_theta_symbolicmutualinformation",
    ]

    return create_control_rs_four_way_comparison_grid(
        output_dir, biomarker_filter=selected_biomarkers
    )


def main():
    """Main function to run the Control RS CBraMod comparison analysis."""

    parser = argparse.ArgumentParser(
        description="Create Control RS topographic comparison plots for CBraMod"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/data/project/eeg_foundation/src/doc_benchmark/results/global_analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Only plot selected biomarkers (9 common ones)",
    )
    parser.add_argument(
        "--three-way",
        action="store_true",
        help="Create three-way comparison: Original vs CBraMod vs TOTEM",
    )
    parser.add_argument(
        "--four-way",
        action="store_true",
        help="Create four-way comparison: Original vs CBraMod vs TOTEM vs LaBram",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all plots (CBraMod comparison, three-way, and four-way comparison)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Control RS CBraMod Comparison Analysis")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    output_paths = []

    if args.all:
        # Create all plots
        print("\n--- Creating CBraMod comparison plot ---")
        path1 = create_control_rs_comparison_grid(args.output_dir)
        if path1:
            output_paths.append(path1)

        print("\n--- Creating three-way comparison plot ---")
        path2 = create_control_rs_three_way_comparison_grid(args.output_dir)
        if path2:
            output_paths.append(path2)

        print("\n--- Creating four-way comparison plot ---")
        path3 = create_control_rs_four_way_comparison_grid(args.output_dir)
        if path3:
            output_paths.append(path3)
    elif args.four_way:
        if args.selected_only:
            output_path = create_control_rs_four_way_comparison_grid_selected(
                args.output_dir
            )
        else:
            output_path = create_control_rs_four_way_comparison_grid(args.output_dir)
        if output_path:
            output_paths.append(output_path)
    elif args.three_way:
        if args.selected_only:
            output_path = create_control_rs_three_way_comparison_grid_selected(
                args.output_dir
            )
        else:
            output_path = create_control_rs_three_way_comparison_grid(args.output_dir)
        if output_path:
            output_paths.append(output_path)
    else:
        if args.selected_only:
            output_path = create_control_rs_comparison_grid_selected(args.output_dir)
        else:
            output_path = create_control_rs_comparison_grid(args.output_dir)
        if output_path:
            output_paths.append(output_path)

    if output_paths:
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        for p in output_paths:
            print(f"Output saved to: {p}")
        print("=" * 80)
    else:
        print("\nAnalysis failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
