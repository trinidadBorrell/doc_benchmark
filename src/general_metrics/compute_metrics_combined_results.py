"""
Compute combined general metrics across models.

Reads metrics.json from each model directory and produces:
  1) FFT plot: boxplot+swarmplot of fft_correlation (left) and fft_mape (right)
  2) Normal plot: boxplot+swarmplot of correlation (left) and mape (right)

Usage:
    python compute_metrics_combined_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Aesthetics ──────────────────────────────────────────────────────────────
COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": False,
        "legend.fontsize": 14,
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(
    "/data/project/eeg_foundation/data/benchmark_results/new_results"
)
OUTPUT_DIR = BASE_DIR / "combined_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_models(base_dir: Path) -> pd.DataFrame:
    """Load subject_metrics from every model that has a metrics.json."""
    rows = []
    for model_dir in sorted(base_dir.iterdir()):
        metrics_path = (
            model_dir / "doc_patients" / "GENERAL_METRICS" / "metrics.json"
        )
        if not metrics_path.is_file():
            continue
        model_name = model_dir.name
        with open(metrics_path) as f:
            data = json.load(f)
        for entry in data.get("subject_metrics", {}).values():
            rows.append(
                {
                    "model": model_name,
                    "correlation": entry["correlation"],
                    "mape": entry["mape"],
                    "fft_correlation": entry["fft_correlation"],
                    "fft_mape": entry["fft_mape"],
                }
            )
    return pd.DataFrame(rows)


def make_combined_plot(
    df: pd.DataFrame,
    corr_col: str,
    mape_col: str,
    title_prefix: str,
    filename: str,
) -> None:
    """Create a figure with two subplots (correlation | MAPE) as boxplots + swarmplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    models = sorted(df["model"].unique())
    palette = dict(zip(models, sns.color_palette("Set2", n_colors=len(models))))

    # ── Left subplot: correlation ───────────────────────────────────────────
    ax = axes[0]
    sns.boxplot(
        data=df,
        x="model",
        y=corr_col,
        ax=ax,
        hue="model",
        palette=palette,
        fliersize=0,
        legend=False,
    )
    sns.swarmplot(
        data=df,
        x="model",
        y=corr_col,
        ax=ax,
        hue="model",
        palette=palette,
        alpha=0.5,
        size=3,
        legend=False,
    )
    ax.set_title(f"{title_prefix} – Correlation")
    ax.set_xlabel("Model")
    ax.set_ylabel("Correlation")

    # ── Right subplot: MAPE ─────────────────────────────────────────────────
    ax = axes[1]
    sns.boxplot(
        data=df,
        x="model",
        y=mape_col,
        ax=ax,
        hue="model",
        palette=palette,
        fliersize=0,
        legend=False,
    )
    sns.swarmplot(
        data=df,
        x="model",
        y=mape_col,
        ax=ax,
        hue="model",
        palette=palette,
        alpha=0.5,
        size=3,
        legend=False,
    )
    ax.set_title(f"{title_prefix} – MAPE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAPE")

    fig.suptitle(title_prefix, fontsize=20, y=1.02)
    fig.tight_layout()
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    df = load_all_models(BASE_DIR)
    if df.empty:
        print("No metrics.json files found – nothing to plot.")
        return

    print(f"Loaded {len(df)} subject entries across models: "
          f"{df['model'].unique().tolist()}")

    # Plot 1: FFT correlation + FFT MAPE
    make_combined_plot(
        df,
        corr_col="fft_correlation",
        mape_col="fft_mape",
        title_prefix="FFT Metrics",
        filename="fft_metrics_combined.png",
    )

    # Plot 2: Normal correlation + MAPE
    make_combined_plot(
        df,
        corr_col="correlation",
        mape_col="mape",
        title_prefix="Signal Metrics",
        filename="signal_metrics_combined.png",
    )


if __name__ == "__main__":
    main()
