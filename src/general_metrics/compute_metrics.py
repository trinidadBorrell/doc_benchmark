#!/usr/bin/env python3
"""
General Metrics: Original vs Reconstructed EEG Comparison
=========================================================
Computes per-subject reconstruction quality metrics:
  - Time-domain Pearson correlation (channel-averaged)
  - Time-domain MAPE (Mean Absolute Percentage Error)
  - FFT-domain Pearson correlation
  - FFT-domain MAPE

Produces:
  - metrics.json   (per-subject + aggregate statistics)
  - correlation_by_subject.png
  - mape_by_subject.png
  - fft_correlation_by_subject.png
  - fft_mape_by_subject.png

Usage (standalone):
    python compute_metrics.py \
        --base_dirs /path/to/recon_data \
        --original_data_dir /path/to/orig1 /path/to/orig2 \
        --output_dir /path/to/output

Called by pipeline.py Phase A.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Suppress MNE info messages
import mne
mne.set_log_level("WARNING")

# Plotting style
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _align(a: np.ndarray, b: np.ndarray):
    """Trim two arrays to the minimum common shape along all axes."""
    slices = tuple(slice(0, min(sa, sb)) for sa, sb in zip(a.shape, b.shape))
    return a[slices], b[slices]


def _vectorized_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r for each row pair using vectorized ops.

    Parameters
    ----------
    a, b : (N, T)  – N signals of length T

    Returns
    -------
    (N,) array of correlation coefficients
    """
    a_m = a - a.mean(axis=-1, keepdims=True)
    b_m = b - b.mean(axis=-1, keepdims=True)
    num = (a_m * b_m).sum(axis=-1)
    denom = np.sqrt((a_m ** 2).sum(axis=-1) * (b_m ** 2).sum(axis=-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = num / (denom + 1e-30)
    return r


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Channel-averaged Pearson correlation between two epoch arrays.

    Parameters
    ----------
    a, b : np.ndarray  (n_epochs, n_channels, n_timepoints)

    Returns
    -------
    float  Mean Pearson r across channels and epochs.
    """
    a, b = _align(a, b)
    # Flatten (epochs, channels) → (N, T) for vectorized correlation
    n_ep, n_ch, n_t = a.shape
    a_flat = a.reshape(n_ep * n_ch, n_t)
    b_flat = b.reshape(n_ep * n_ch, n_t)
    r = _vectorized_corr(a_flat, b_flat)
    mask = np.isfinite(r)
    return float(np.mean(r[mask])) if mask.any() else float("nan")


def mape(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Absolute Percentage Error between two epoch arrays.

    MAPE = mean(|a - b| / (|a| + eps))  averaged over channels & epochs.
    """
    eps = 1e-10
    a, b = _align(a, b)
    denom = np.abs(a) + eps
    # Per-channel-epoch MAPE: mean over time, then mean over everything
    val = np.mean(np.abs(a - b) / denom, axis=-1)  # (n_epochs, n_channels)
    mask = np.isfinite(val)
    return float(np.mean(val[mask])) if mask.any() else float("nan")


def fft_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation in the FFT magnitude domain."""
    a, b = _align(a, b)
    n_ep, n_ch, n_t = a.shape
    # Batch FFT: (n_epochs, n_channels, n_freq)
    fa = np.abs(np.fft.rfft(a, axis=-1))
    fb = np.abs(np.fft.rfft(b, axis=-1))
    n_f = fa.shape[-1]
    r = _vectorized_corr(fa.reshape(n_ep * n_ch, n_f),
                         fb.reshape(n_ep * n_ch, n_f))
    mask = np.isfinite(r)
    return float(np.mean(r[mask])) if mask.any() else float("nan")


def fft_mape(a: np.ndarray, b: np.ndarray) -> float:
    """MAPE in the FFT magnitude domain."""
    eps = 1e-10
    a, b = _align(a, b)
    fa = np.abs(np.fft.rfft(a, axis=-1))
    fb = np.abs(np.fft.rfft(b, axis=-1))
    denom = fa + eps
    val = np.mean(np.abs(fa - fb) / denom, axis=-1)  # (n_epochs, n_channels)
    mask = np.isfinite(val)
    return float(np.mean(val[mask])) if mask.any() else float("nan")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def find_original_file(subject_id: str, session_id: str,
                       original_data_dirs: list) -> Path | None:
    """Search through original data directories to find the original .fif file."""
    for orig_dir in original_data_dirs:
        orig_dir = Path(orig_dir)
        ses_dir = orig_dir / f"sub-{subject_id}" / f"ses-{session_id}"

        # Check eeg/ sub-directory first (nice_epochs style)
        eeg_dir = ses_dir / "eeg"
        if eeg_dir.exists():
            candidates = list(eeg_dir.glob(f"sub-{subject_id}_ses-{session_id}_task-*_epo.fif"))
            if candidates:
                return candidates[0]

        # Flat session directory (fifdata style)
        if ses_dir.exists():
            # Prefer *_epo_original.fif
            candidates = list(ses_dir.glob("*_epo_original.fif"))
            if candidates:
                return candidates[0]
            candidates = list(ses_dir.glob("*_epo.fif"))
            if candidates:
                return candidates[0]
    return None


def find_reconstructed_file(subject_id: str, session_id: str,
                            recon_dir: Path) -> Path | None:
    """Find the reconstructed .fif file for a subject/session."""
    ses_dir = recon_dir / f"sub-{subject_id}" / f"ses-{session_id}"
    if not ses_dir.exists():
        return None

    # Directories to search: session dir first, then eeg/ subdirectory
    search_dirs = [ses_dir]
    eeg_dir = ses_dir / "eeg"
    if eeg_dir.exists():
        search_dirs.append(eeg_dir)

    for search_dir in search_dirs:
        # Pattern: *_reconstructed_epo.fif (CBraMod/LaBram)
        candidates = list(search_dir.glob("*_reconstructed_epo.fif"))
        if candidates:
            return candidates[0]
        # Pattern: *_epo_reconstructed.fif (NeuroLM style)
        candidates = list(search_dir.glob("*_epo_reconstructed.fif"))
        if candidates:
            return candidates[0]
        # Pattern: *_epo_recon.fif (standard)
        candidates = list(search_dir.glob("*_epo_recon.fif"))
        if candidates:
            return candidates[0]

    # Pattern: recon/*.fif
    recon_subdir = ses_dir / "recon"
    if recon_subdir.exists():
        candidates = list(recon_subdir.glob("*.fif"))
        if candidates:
            return candidates[0]
    return None


def discover_subjects_sessions(recon_dir: Path):
    """Yield (subject_id, session_id) pairs from a reconstructed data directory."""
    for sub_dir in sorted(recon_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name.replace("sub-", "")
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if ses_dir.is_dir():
                session_id = ses_dir.name.replace("ses-", "")
                yield subject_id, session_id


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_plot(labels, values, title, ylabel, output_path, color="steelblue"):
    """Simple per-subject bar chart with mean line."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)

    mean_val = float(np.nanmean(values))
    ax.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_all_metrics(base_dirs, original_data_dirs, output_dir):
    """
    Compute reconstruction quality metrics for every subject/session found
    under *base_dirs* (reconstructed data) by comparing with originals found
    under *original_data_dirs*.

    Results are saved as JSON + per-metric bar plots in *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_metrics = {}   # key = "sub-{id}_ses-{ses}"
    datadir_metrics = {}   # aggregate per base_dir

    for recon_dir_str in base_dirs:
        recon_dir = Path(recon_dir_str)
        if not recon_dir.exists():
            print(f"WARNING: recon directory does not exist: {recon_dir}")
            continue

        dir_corrs, dir_mapes, dir_fft_corrs, dir_fft_mapes = [], [], [], []
        pairs = list(discover_subjects_sessions(recon_dir))
        print(f"\nProcessing {len(pairs)} subject/session pairs from {recon_dir}")

        for subject_id, session_id in pairs:
            key = f"sub-{subject_id}_ses-{session_id}"

            recon_file = find_reconstructed_file(subject_id, session_id, recon_dir)
            orig_file = find_original_file(subject_id, session_id, original_data_dirs)

            if recon_file is None:
                print(f"  {key}: no reconstructed file found, skipping")
                continue
            if orig_file is None:
                print(f"  {key}: no original file found, skipping")
                continue

            try:
                orig_epochs = mne.read_epochs(str(orig_file), preload=True, verbose=False)
                recon_epochs = mne.read_epochs(str(recon_file), preload=True, verbose=False)

                orig_data = orig_epochs.get_data().astype(np.float32)
                recon_data = recon_epochs.get_data().astype(np.float32)

                corr_val = pearson_correlation(orig_data, recon_data)
                mape_val = mape(orig_data, recon_data)
                fft_corr_val = fft_correlation(orig_data, recon_data)
                fft_mape_val = fft_mape(orig_data, recon_data)

                subject_metrics[key] = {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "correlation": corr_val,
                    "mape": mape_val,
                    "fft_correlation": fft_corr_val,
                    "fft_mape": fft_mape_val,
                    "n_epochs_orig": int(orig_data.shape[0]),
                    "n_epochs_recon": int(recon_data.shape[0]),
                    "n_channels_orig": int(orig_data.shape[1]),
                    "n_channels_recon": int(recon_data.shape[1]),
                }

                dir_corrs.append(corr_val)
                dir_mapes.append(mape_val)
                dir_fft_corrs.append(fft_corr_val)
                dir_fft_mapes.append(fft_mape_val)

                print(f"  {key}: corr={corr_val:.4f}  mape={mape_val:.4f}  "
                      f"fft_corr={fft_corr_val:.4f}  fft_mape={fft_mape_val:.4f}")

            except Exception as e:
                print(f"  {key}: ERROR - {e}")
                subject_metrics[key] = {"error": str(e)}

        if dir_corrs:
            datadir_metrics[str(recon_dir)] = {
                "n_subjects": len(dir_corrs),
                "mean_correlation": float(np.nanmean(dir_corrs)),
                "std_correlation": float(np.nanstd(dir_corrs)),
                "mean_mape": float(np.nanmean(dir_mapes)),
                "std_mape": float(np.nanstd(dir_mapes)),
                "mean_fft_correlation": float(np.nanmean(dir_fft_corrs)),
                "std_fft_correlation": float(np.nanstd(dir_fft_corrs)),
                "mean_fft_mape": float(np.nanmean(dir_fft_mapes)),
                "std_fft_mape": float(np.nanstd(dir_fft_mapes)),
            }

    # ---- Save JSON --------------------------------------------------------
    results = {
        "datadir_metrics": datadir_metrics,
        "subject_metrics": subject_metrics,
    }
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics JSON: {json_path}")

    # ---- Generate plots ---------------------------------------------------
    valid = {k: v for k, v in subject_metrics.items() if "error" not in v}
    if not valid:
        print("No valid subject metrics to plot.")
        return results

    labels = sorted(valid.keys())
    corrs = [valid[k]["correlation"] for k in labels]
    mapes_vals = [valid[k]["mape"] for k in labels]
    fft_corrs = [valid[k]["fft_correlation"] for k in labels]
    fft_mapes_vals = [valid[k]["fft_mape"] for k in labels]

    _bar_plot(labels, corrs,
              "Time-domain Pearson Correlation (Original vs Reconstructed)",
              "Correlation (r)", output_dir / "correlation_by_subject.png",
              color="steelblue")

    _bar_plot(labels, mapes_vals,
              "Time-domain MAPE (Original vs Reconstructed)",
              "MAPE", output_dir / "mape_by_subject.png",
              color="darkorange")

    _bar_plot(labels, fft_corrs,
              "FFT Pearson Correlation (Original vs Reconstructed)",
              "Correlation (r)", output_dir / "fft_correlation_by_subject.png",
              color="teal")

    _bar_plot(labels, fft_mapes_vals,
              "FFT MAPE (Original vs Reconstructed)",
              "MAPE", output_dir / "fft_mape_by_subject.png",
              color="indianred")

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERAL METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects analyzed: {len(valid)}")
    print(f"Time-domain correlation: {np.nanmean(corrs):.4f} ± {np.nanstd(corrs):.4f}")
    print(f"Time-domain MAPE:       {np.nanmean(mapes_vals):.4f} ± {np.nanstd(mapes_vals):.4f}")
    print(f"FFT correlation:        {np.nanmean(fft_corrs):.4f} ± {np.nanstd(fft_corrs):.4f}")
    print(f"FFT MAPE:               {np.nanmean(fft_mapes_vals):.4f} ± {np.nanstd(fft_mapes_vals):.4f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute general reconstruction quality metrics (correlation, MAPE) "
                    "between original and reconstructed EEG epochs."
    )
    parser.add_argument("--base_dirs", nargs="+", required=True,
                        help="Directories containing reconstructed data (sub-*/ses-*/ structure)")
    parser.add_argument("--original_data_dir", nargs="+", default=[],
                        help="Directories containing original data for comparison")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save metrics JSON and plots")
    args = parser.parse_args()

    if not args.original_data_dir:
        print("ERROR: --original_data_dir is required for computing reconstruction metrics.")
        sys.exit(1)

    compute_all_metrics(args.base_dirs, args.original_data_dir, args.output_dir)


if __name__ == "__main__":
    main()
