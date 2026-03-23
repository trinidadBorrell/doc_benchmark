"""Embedding comparison analysis: R²/noise-ceiling, RSA, CKA, and dimensionality metrics.

==================================================
Embedding Comparison — Components 1, 2, 3, 4
==================================================

This script compares two embedding spaces for DoC patient classification:

* **Domain-knowledge (DK) embeddings**: hand-crafted NICE markers from
  ``baseline_stable_20210128_scalars.csv`` (reduction A).
* **Foundation model (FM) embeddings**: pre-computed embeddings from
  NeuroLM, TOTEM, CBraMod, LaBram (``sub-*/ses-*/embedding.npy``).

The central question: do the most important markers for DoC prediction
align between the two spaces?

Components
----------
1. **Noise-ceiling-corrected R²**: Ridge regression from FM embeddings to
   each DK marker, normalized by inter-subject noise ceiling (Monte Carlo
   split-half Spearman-Brown). Produces per-model detail plots and an
   all-model comparison.
2. **RSA (Representational Similarity Analysis)**: Build pairwise RDMs for
   DK and FM spaces, compare upper triangles via Spearman ρ with bootstrap
   CIs (95%) and permutation test. Also reports correlation with label RDM.
3. **CKA** (Centered Kernel Alignment): linear CKA between DK and FM
   representations, with bootstrap CIs and label alignment.
4. **Dimensionality Metrics**: Eigenspectrum power-law exponent α,
   participation ratio (PR), effective rank, and threshold-based
   dimensionality (n_80, n_90, n_95, n_99) per embedding space, all with
   bootstrap CIs.

Usage
-----
::

    python embedding_comparison.py \\
        --results-dir /data/.../new_results \\
        --marker-csv /data/.../baseline_stable_20210128_scalars.csv \\
        --patient-labels /data/.../patient_labels.csv \\
        --output-dir /data/.../embedding_comparison \\
        --foundation-models NeuroLM CBraMod TOTEM LaBram \\
        --marker-reduction A \\
        --n-cv-folds 5 \\
        --random-state 42 \\
        --pca-variance 0.95

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import logging
import os
import os.path as op
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

_EMBED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_DK_COLORS = ["#9467bd", "#8c564b", "#e377c2"]


def _short_marker_name(full_path: str) -> str:
    """``nice/marker/PowerSpectralDensity/deltan`` → ``PowerSpectralDensity_deltan``."""
    parts = full_path.split("/")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return full_path


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_domain_knowledge_embeddings(
    marker_csv: str, reduction_letter: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load domain-knowledge (NICE marker) embeddings.

    Returns
    -------
    X_dk : np.ndarray, shape (N, P)
    subject_keys : list of str  — e.g. "AA079_ses-02"
    marker_names : list of str
    """
    if reduction_letter not in REDUCTION_MAP:
        raise ValueError(
            f"Unknown reduction {reduction_letter!r}. "
            f"Choose from {list(REDUCTION_MAP)}."
        )
    reduction_str = REDUCTION_MAP[reduction_letter]
    df = pd.read_csv(marker_csv, sep=";")
    df_filt = df[df["Reduction"] == reduction_str].copy()
    if df_filt.empty:
        raise ValueError(
            f"No rows for Reduction={reduction_str!r} in {marker_csv}"
        )

    meta_cols = {"Subject", "Reduction", "Label"}
    marker_names = [
        c
        for c in df_filt.columns
        if c not in meta_cols
        and not str(c).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df_filt[c])
    ]

    rows_X = []
    subject_keys = []
    for _, row in df_filt.iterrows():
        raw = str(row["Subject"])
        parts = raw.rsplit("_", 1)
        if len(parts) != 2:
            continue
        subj_id, sesnum = parts
        try:
            key = f"{subj_id}_ses-{int(sesnum):02d}"
        except ValueError:
            continue
        subject_keys.append(key)
        rows_X.append(row[marker_names].values.astype(float))

    X_dk = np.array(rows_X)
    log.info(
        "Loaded DK embeddings: %d subjects × %d markers", X_dk.shape[0], X_dk.shape[1]
    )
    return X_dk, subject_keys, list(marker_names)


def _load_foundation_embeddings(
    pooled_embeddings_dir: str,
) -> Tuple[np.ndarray, List[str]]:
    """Walk sub-*/ses-*/ to load foundation model embeddings.

    Looks for ``embedding.npy`` or ``embedding.npz`` (key 'embedding').

    Returns
    -------
    X_fm : np.ndarray, shape (N, D)
    subject_keys : list of str
    """
    rows = []
    keys = []
    for subj_dir in sorted(os.listdir(pooled_embeddings_dir)):
        if not subj_dir.startswith("sub-"):
            continue
        subj_path = op.join(pooled_embeddings_dir, subj_dir)
        if not op.isdir(subj_path):
            continue
        subj_id = subj_dir[4:]  # strip "sub-"
        for ses_dir in sorted(os.listdir(subj_path)):
            if not ses_dir.startswith("ses-"):
                continue
            ses_path = op.join(subj_path, ses_dir)
            emb = None
            for fname in ["embedding.npy", "embedding.npz"]:
                fpath = op.join(ses_path, fname)
                if op.isfile(fpath):
                    if fname.endswith(".npy"):
                        emb = np.load(fpath)
                    else:
                        d = np.load(fpath)
                        emb = d[d.files[0]]
                    break
            if emb is None:
                continue
            key = f"{subj_id}_{ses_dir}"
            keys.append(key)
            rows.append(emb.flatten())

    if not rows:
        raise RuntimeError(f"No embeddings found in {pooled_embeddings_dir}")

    X_fm = np.array(rows)
    log.info(
        "Loaded FM embeddings: %d subjects × %d dims", X_fm.shape[0], X_fm.shape[1]
    )
    return X_fm, keys


def _load_patient_labels(labels_file: str) -> Dict[str, int]:
    """Load binary VS/MCS labels.

    Returns
    -------
    dict  subject_session_key -> 0 (VS) or 1 (MCS)
    """
    df = pd.read_csv(labels_file)
    labels = {}
    for _, row in df.iterrows():
        state = row.get("diagnostic_crs_final", "")
        if pd.isna(state) or state == "n/a":
            continue
        if state == "UWS":
            label = 0
        elif state in ("MCS+", "MCS-"):
            label = 1
        else:
            continue
        subj = str(row["subject"])
        ses = f"ses-{int(row['session']):02d}"
        key = f"{subj}_{ses}"
        labels[key] = label
    return labels


def _match_subjects(
    dk_keys: List[str],
    fm_keys: List[str],
    labels_dict: Dict[str, int],
) -> Tuple[List[int], List[int], np.ndarray, List[str]]:
    """Compute intersection and return aligned indices.

    Returns
    -------
    dk_idx : list of int
    fm_idx : list of int
    y      : np.ndarray of int labels
    matched_keys : list of str
    """
    common = sorted(set(dk_keys) & set(fm_keys) & set(labels_dict.keys()))
    if len(common) < 20:
        log.warning(
            "Only %d subjects in DK ∩ FM ∩ labels intersection (< 20). "
            "RSA/CKA results may be unreliable.",
            len(common),
        )
    else:
        log.info("Subject intersection: %d", len(common))

    dk_map = {k: i for i, k in enumerate(dk_keys)}
    fm_map = {k: i for i, k in enumerate(fm_keys)}

    dk_idx = [dk_map[k] for k in common]
    fm_idx = [fm_map[k] for k in common]
    y = np.array([labels_dict[k] for k in common])
    return dk_idx, fm_idx, y, common


# ═══════════════════════════════════════════════════════════════════════════════
# Ridge probe for FM → marker weights (Component 1 support)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_ridge_probe(
    X_fm: np.ndarray,
    X_dk: np.ndarray,
    marker_names: List[str],
    output_dir: str,
    fm_model_name: str,
    random_state: int = 42,
) -> np.ndarray:
    """Fit multi-output Ridge regression: FM embeddings → DK markers.

    Saves coefficient matrix W (D × P) and per-marker R².

    Returns
    -------
    W : np.ndarray (D, P) — Ridge coefficient matrix
    """
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415
    from sklearn.metrics import r2_score  # noqa: PLC0415
    from sklearn.multioutput import MultiOutputRegressor  # noqa: PLC0415

    os.makedirs(output_dir, exist_ok=True)

    # Impute NaNs in X_dk
    col_means = np.nanmean(X_dk, axis=0)
    nan_mask = np.isnan(X_dk)
    X_dk_clean = X_dk.copy()
    X_dk_clean[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler_fm = StandardScaler()
    X_fm_sc = scaler_fm.fit_transform(X_fm)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge = RidgeCV(alphas=alphas, cv=5, scoring="r2")

    mor = MultiOutputRegressor(ridge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mor.fit(X_fm_sc, X_dk_clean)

    # Collect weights: (D, P)
    W = np.column_stack([est.coef_ for est in mor.estimators_])  # (D, P)

    # R² per marker
    Y_pred = mor.predict(X_fm_sc)
    r2_per_marker = [
        float(r2_score(X_dk_clean[:, i], Y_pred[:, i]))
        for i in range(X_dk_clean.shape[1])
    ]

    # Save
    np.save(op.join(output_dir, f"ridge_weights_{fm_model_name}.npy"), W)
    summary = {
        marker: {"r2": r2, "alpha": float(mor.estimators_[i].alpha_)}
        for i, (marker, r2) in enumerate(zip(marker_names, r2_per_marker))
    }
    with open(op.join(output_dir, f"ridge_probe_summary_{fm_model_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Ridge probe fitted. Mean R² = %.3f (n_markers=%d)",
        np.mean(r2_per_marker),
        len(marker_names),
    )
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2 — RSA (Representational Similarity Analysis)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rdm(X: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Compute N×N Representational Dissimilarity Matrix.

    Parameters
    ----------
    X      : (N, D) embedding matrix
    metric : distance metric — 'correlation' or 'euclidean'

    Returns
    -------
    rdm : (N, N) symmetric distance matrix (zeros on diagonal)
    """
    return squareform(pdist(X, metric=metric))


def run_rsa_analysis(
    X_dk: np.ndarray,
    X_fm: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    fm_model_name: str,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    distance_metric: str = "correlation",
    random_state: int = 42,
) -> Dict:
    """Run RSA between DK and FM representation spaces.

    Builds pairwise RDMs, compares upper triangles via Spearman ρ.
    Reports bootstrap CI and permutation p-value.

    Parameters
    ----------
    X_dk           : (N, P) domain-knowledge features (may contain NaN)
    X_fm           : (N, D) foundation model embeddings
    y              : (N,) binary labels
    output_dir     : where to save JSON results
    fm_model_name  : e.g. "NeuroLM"
    n_bootstrap    : bootstrap iterations for CI
    n_permutations : permutation iterations for p-value
    distance_metric: 'correlation' or 'euclidean'
    random_state   : seed

    Returns
    -------
    dict with rho, ci_lower, ci_upper, p_perm, rho_dk_label, rho_fm_label
    """
    os.makedirs(output_dir, exist_ok=True)

    # Impute NaNs in DK
    col_means = np.nanmean(X_dk, axis=0)
    nan_mask = np.isnan(X_dk)
    X_dk_clean = X_dk.copy()
    X_dk_clean[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    n = X_dk_clean.shape[0]
    if n < 10:
        log.warning("RSA: only %d subjects — results unreliable.", n)

    rdm_dk = compute_rdm(X_dk_clean, distance_metric)
    rdm_fm = compute_rdm(X_fm, distance_metric)

    tri_idx = np.triu_indices(n, k=1)
    tri_dk = rdm_dk[tri_idx]
    tri_fm = rdm_fm[tri_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(tri_dk, tri_fm)
    rho = float(rho) if not np.isnan(rho) else 0.0

    # Bootstrap CI (resample subjects, recompute RDMs)
    rng = np.random.RandomState(random_state)
    boot_rhos = []
    for _ in range(n_bootstrap):
        samp = rng.choice(n, n, replace=True)
        rdm_dk_b = compute_rdm(X_dk_clean[samp], distance_metric)
        rdm_fm_b = compute_rdm(X_fm[samp], distance_metric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_b, _ = spearmanr(rdm_dk_b[tri_idx], rdm_fm_b[tri_idx])
        boot_rhos.append(float(r_b) if not np.isnan(r_b) else 0.0)

    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))

    # Permutation test: permute FM subjects (permutes both rows and cols)
    perm_rhos = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        rdm_fm_perm = rdm_fm[np.ix_(perm, perm)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_p, _ = spearmanr(tri_dk, rdm_fm_perm[tri_idx])
        perm_rhos.append(float(r_p) if not np.isnan(r_p) else 0.0)
    p_perm = float(np.mean(np.array(perm_rhos) >= rho))

    # Label RDM: L[i,j] = 1 if classes differ (dissimilarity convention)
    label_rdm = (y[:, None] != y[None, :]).astype(float)
    tri_label = label_rdm[tri_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_dk_label, _ = spearmanr(tri_dk, tri_label)
        rho_fm_label, _ = spearmanr(tri_fm, tri_label)

    results = {
        "fm_model_name": fm_model_name,
        "rho": rho,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_perm": p_perm,
        "rho_dk_label": float(rho_dk_label) if not np.isnan(rho_dk_label) else 0.0,
        "rho_fm_label": float(rho_fm_label) if not np.isnan(rho_fm_label) else 0.0,
        "n_subjects": int(n),
        "n_pairs": int(len(tri_dk)),
        "distance_metric": distance_metric,
    }

    out_json = op.join(output_dir, f"rsa_results_{fm_model_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("RSA results saved: %s", out_json)

    plots_dir = op.join(op.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_rsa_results(tri_dk, tri_fm, results, fm_model_name, plots_dir)

    log.info(
        "RSA ρ=%.3f [%.3f, %.3f]  p_perm=%.3f — %s",
        rho, ci_lo, ci_hi, p_perm, fm_model_name,
    )
    return results, tri_dk, tri_fm


def _plot_rsa_results(
    tri_dk: np.ndarray,
    tri_fm: np.ndarray,
    results: Dict,
    fm_model_name: str,
    plots_dir: str,
):
    """Scatter plot of RDM upper triangles with Spearman ρ annotation."""
    rho = results["rho"]
    ci_lo = results["ci_lower"]
    ci_hi = results["ci_upper"]
    p_perm = results["p_perm"]
    metric = results["distance_metric"]

    # Subsample for readability
    n_pairs = len(tri_dk)
    if n_pairs > 5000:
        rng = np.random.RandomState(0)
        idx = rng.choice(n_pairs, 5000, replace=False)
        x, y = tri_dk[idx], tri_fm[idx]
    else:
        x, y = tri_dk, tri_fm

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.15, s=8, color="#1f77b4", rasterized=True)
    ax.set_xlabel(f"RDM distance — DK space ({metric})")
    ax.set_ylabel(f"RDM distance — FM space ({fm_model_name}, {metric})")
    ax.set_title(
        f"RSA: DK vs {fm_model_name}\n"
        f"Spearman ρ = {rho:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]  "
        f"p_perm = {p_perm:.3f}"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = op.join(plots_dir, f"rsa_scatter_{fm_model_name}")
    plt.savefig(f"{sp}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_rsa_summary(rsa_results_all: Dict[str, Dict], output_dir: str):
    """Save combined RSA summary for all foundation models."""
    summary_path = op.join(output_dir, "rsa_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rsa_results_all, f, indent=2)
    log.info("RSA summary saved: %s", summary_path)

    models = list(rsa_results_all.keys())
    rho_vals = [rsa_results_all[m]["rho"] for m in models]
    ci_lo = [rsa_results_all[m]["ci_lower"] for m in models]
    ci_hi = [rsa_results_all[m]["ci_upper"] for m in models]
    p_vals = [rsa_results_all[m]["p_perm"] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    x = np.arange(len(models))
    err_lo = [rho_vals[i] - ci_lo[i] for i in range(len(models))]
    err_hi = [ci_hi[i] - rho_vals[i] for i in range(len(models))]
    ax.bar(
        x, rho_vals,
        yerr=[err_lo, err_hi],
        color=[_EMBED_COLORS[i % len(_EMBED_COLORS)] for i in range(len(models))],
        alpha=0.85, capsize=8,
        error_kw={"elinewidth": 2, "capthick": 2},
    )
  #  for xi, p, rho in zip(x, p_vals, rho_vals):
  #      if p < 0.05:
  #          ax.text(xi, rho + 0.02, "*", ha="center", va="bottom", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Spearman ρ (RDM upper triangle)")
    ax.set_title(
        "RSA(DK, FM) per Foundation Model"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plots_dir = op.join(op.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(op.join(plots_dir, "rsa_barplot_all.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_rsa_scatter_all(
    tri_per_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rsa_results_all: Dict[str, Dict],
    plots_dir: str,
):
    """Multi-panel RSA scatter: one subplot per FM model, shared DK axis label."""
    models = list(tri_per_model.keys())
    if not models:
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    metric = rsa_results_all[models[0]].get("distance_metric", "correlation")

    for ax, fm_name, color in zip(axes, models, _EMBED_COLORS):
        tri_dk, tri_fm = tri_per_model[fm_name]
        res = rsa_results_all[fm_name]
        rho = res["rho"]
        ci_lo = res["ci_lower"]
        ci_hi = res["ci_upper"]
        p_val = res["p_perm"]

        n_pairs = len(tri_dk)
        if n_pairs > 5000:
            rng = np.random.RandomState(0)
            idx = rng.choice(n_pairs, 5000, replace=False)
            x, y = tri_dk[idx], tri_fm[idx]
        else:
            x, y = tri_dk, tri_fm

        ax.scatter(x, y, alpha=0.12, s=6, color=color, rasterized=True)
        ax.set_xlabel(f"DK RDM ({metric})", fontsize=9)
        ax.set_ylabel(f"{fm_name} RDM", fontsize=9)
        ax.set_title(
            f"{fm_name}\nρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_val:.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle("RSA scatter — DK vs all FM models", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(op.join(plots_dir, "rsa_scatter_all.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Component 3 — Linear CKA
# ═══════════════════════════════════════════════════════════════════════════════


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment.

    Parameters
    ----------
    X : (N, P)  — first representation
    Y : (N, Q)  — second representation

    Returns
    -------
    float in [0, 1]
    """
    n = X.shape[0]
    if n < 10:
        log.warning("CKA: n=%d is very small.", n)

    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K = X @ X.T  # (N, N)
    L = Y @ Y.T  # (N, N)

    # Double-center
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    num = np.trace(Kc @ Lc)
    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Lc, "fro")
    if denom < 1e-10:
        return 0.0
    return float(num / denom)


def compute_cka_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """CKA with bootstrap confidence intervals.

    Returns
    -------
    cka : float
    ci_lower : float (2.5th percentile)
    ci_upper : float (97.5th percentile)
    """
    cka = compute_linear_cka(X, Y)
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    boot_ckas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_ckas.append(compute_linear_cka(X[idx], Y[idx]))
    ci_lower = float(np.percentile(boot_ckas, 2.5))
    ci_upper = float(np.percentile(boot_ckas, 97.5))
    return cka, ci_lower, ci_upper


def _label_kernel(y: np.ndarray) -> np.ndarray:
    """Binary RDM from labels: K[i,j] = 1 if y[i] == y[j], else 0."""
    y = y.reshape(-1, 1)
    return (y == y.T).astype(float)


def run_cka_analysis(
    X_dk: np.ndarray,
    X_fm: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    fm_model_name: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Compute linear CKA between DK and FM representations.

    Also computes CKA of each space with the label kernel.

    Saves results JSON and bar plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    if X_dk.shape[0] < 10:
        log.warning(
            "CKA: only %d subjects — results unreliable.", X_dk.shape[0]
        )
        if X_dk.shape[0] < 100:
            log.info(
                "Using unbiased CKA estimator is recommended for n < 100 "
                "(not yet implemented here; using standard estimator)."
            )

    # Replace NaNs with column means
    dk_col_means = np.nanmean(X_dk, axis=0)
    nan_mask_dk = np.isnan(X_dk)
    X_dk_clean = X_dk.copy()
    X_dk_clean[nan_mask_dk] = np.take(dk_col_means, np.where(nan_mask_dk)[1])

    cka, ci_lo, ci_hi = compute_cka_bootstrap(
        X_dk_clean, X_fm, n_bootstrap=n_bootstrap, random_state=random_state
    )

    # CKA with label kernel
    L = _label_kernel(y)
    cka_dk_label = compute_linear_cka(X_dk_clean, L)
    cka_fm_label = compute_linear_cka(X_fm, L)

    results = {
        "cka_linear": float(cka),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "cka_dk_label": float(cka_dk_label),
        "cka_fm_label": float(cka_fm_label),
        "n_subjects": int(X_dk.shape[0]),
        "n_dk_dims": int(X_dk.shape[1]),
        "n_fm_dims": int(X_fm.shape[1]),
        "fm_model_name": fm_model_name,
    }

    out_file = op.join(output_dir, f"cka_results_{fm_model_name}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("CKA results saved: %s", out_file)

    # Bar plot
    plots_dir = op.join(op.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_cka_bar(results, fm_model_name, plots_dir)

    return results


def _plot_cka_bar(results: Dict, fm_model_name: str, plots_dir: str):
    cka = results["cka_linear"]
    ci_lo = results["ci_lower"]
    ci_hi = results["ci_upper"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        [fm_model_name],
        [cka],
        yerr=[[cka - ci_lo], [ci_hi - cka]],
        color="#1f77b4",
        alpha=0.85,
        capsize=8,
        error_kw={"elinewidth": 2, "capthick": 2},
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Linear CKA")
    ax.set_title(f"CKA(DK, FM)\n{fm_model_name}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = op.join(plots_dir, f"cka_barplot_{fm_model_name}")
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_cka_summary(cka_results_all: Dict[str, Dict], output_dir: str):
    """Save combined CKA summary for all foundation models."""
    summary_path = op.join(output_dir, "cka_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cka_results_all, f, indent=2)
    log.info("CKA summary saved: %s", summary_path)

    # Summary bar plot (all models)
    models = list(cka_results_all.keys())
    cka_vals = [cka_results_all[m]["cka_linear"] for m in models]
    ci_lo = [cka_results_all[m]["ci_lower"] for m in models]
    ci_hi = [cka_results_all[m]["ci_upper"] for m in models]
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    x = np.arange(len(models))
    err_lo = [cka_vals[i] - ci_lo[i] for i in range(len(models))]
    err_hi = [ci_hi[i] - cka_vals[i] for i in range(len(models))]
    ax.bar(
        x, cka_vals,
        yerr=[err_lo, err_hi],
        color=[_EMBED_COLORS[i % len(_EMBED_COLORS)] for i in range(len(models))],
        alpha=0.85, capsize=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Linear CKA")
    ax.set_title("CKA(DK, FM) per Foundation Model")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plots_dir = op.join(op.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(op.join(plots_dir, "cka_barplot_all.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Component 4 — Dimensionality Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Center X and return sorted non-zero eigenvalues of the covariance matrix.

    Uses SVD for numerical stability: λ_i = s_i² / (N-1).
    Returns at most N-1 eigenvalues (rank constraint with N subjects).
    """
    X_c = X - X.mean(axis=0)
    n = X_c.shape[0]
    _, s, _ = np.linalg.svd(X_c, full_matrices=False)
    ev = s ** 2 / max(n - 1, 1)
    return ev[ev > 1e-10]


def _dim_metrics_from_eigenvalues(
    eigenvalues: np.ndarray, ambient_dim: int, n_subjects: int
) -> Dict:
    """Compute all scalar dimensionality metrics from an eigenvalue array."""
    k = len(eigenvalues)
    if k == 0:
        return {}

    lam_norm = eigenvalues / eigenvalues.sum()

    # Power-law exponent α via log-log OLS: log(λ_i) = log(C) - α·log(i)
    k_vals = np.arange(1, k + 1, dtype=float)
    if k > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, _ = np.polyfit(np.log(k_vals), np.log(eigenvalues), 1)
            corr = np.corrcoef(np.log(k_vals), np.log(eigenvalues))[0, 1]
        alpha = float(-slope)
        r2_fit = float(corr ** 2)
    else:
        alpha = float("nan")
        r2_fit = float("nan")

    # Threshold-based dimensionality n_q
    ev_curve = np.cumsum(eigenvalues) / eigenvalues.sum()
    n_80 = int(np.searchsorted(ev_curve, 0.80) + 1)
    n_90 = int(np.searchsorted(ev_curve, 0.90) + 1)
    n_95 = int(np.searchsorted(ev_curve, 0.95) + 1)
    n_99 = int(np.searchsorted(ev_curve, 0.99) + 1)

    # Participation Ratio: sum(λ)² / sum(λ²)
    PR = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())

    # Effective Rank: exp(H(p)) where H is Shannon entropy of normalized eigenvalues
    eff_rank = float(np.exp(-np.sum(lam_norm * np.log(lam_norm + 1e-12))))

    return {
        "n_subjects": int(n_subjects),
        "ambient_dim": int(ambient_dim),
        "n_eigenvalues": int(k),
        "power_law_alpha": alpha,
        "power_law_r2": r2_fit,
        "n_80": int(n_80),
        "n_90": int(n_90),
        "n_95": int(n_95),
        "n_99": int(n_99),
        "participation_ratio": PR,
        "effective_rank": eff_rank,
    }


def run_dimensionality_analysis(
    X_dk: np.ndarray,
    X_fm: np.ndarray,
    output_dir: str,
    fm_model_name: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Compute eigenspectrum-based dimensionality metrics for DK and FM spaces.

    Metrics (per space)
    -------------------
    - Power-law exponent α (log-log eigenspectrum fit) + R²_fit
    - n_80, n_90, n_95, n_99 — threshold-based dimensionality
    - Participation Ratio (PR)
    - Effective Rank

    All scalar metrics are accompanied by 95% bootstrap CIs (resample subjects).

    Parameters
    ----------
    X_dk          : (N, P) DK features (may contain NaN)
    X_fm          : (N, D) FM embeddings
    output_dir    : directory for JSON output
    fm_model_name : e.g. "NeuroLM"
    n_bootstrap   : bootstrap iterations (1000 recommended)
    random_state  : seed

    Returns
    -------
    dict with keys "DK" and "FM", each containing all metrics + CIs
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    # Impute NaNs in DK
    col_means = np.nanmean(X_dk, axis=0)
    nan_mask = np.isnan(X_dk)
    X_dk_clean = X_dk.copy()
    X_dk_clean[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    n_dk, d_dk = X_dk_clean.shape
    n_fm, d_fm = X_fm.shape

    dk_eigs = _compute_eigenvalues(X_dk_clean)
    fm_eigs = _compute_eigenvalues(X_fm)

    dk_base = _dim_metrics_from_eigenvalues(dk_eigs, d_dk, n_dk)
    fm_base = _dim_metrics_from_eigenvalues(fm_eigs, d_fm, n_fm)

    # Bootstrap CIs for PR, EffRank, n_90, n_95
    def _bootstrap(X: np.ndarray) -> Dict:
        boot_PR, boot_ER, boot_n90, boot_n95 = [], [], [], []
        for _ in range(n_bootstrap):
            idx_b = rng.choice(X.shape[0], X.shape[0], replace=True)
            ev_b = _compute_eigenvalues(X[idx_b])
            if len(ev_b) == 0:
                continue
            m = _dim_metrics_from_eigenvalues(ev_b, X.shape[1], X.shape[0])
            boot_PR.append(m["participation_ratio"])
            boot_ER.append(m["effective_rank"])
            boot_n90.append(m["n_90"])
            boot_n95.append(m["n_95"])
        def _ci(arr):
            if not arr:
                return [float("nan"), float("nan")]
            return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        return {
            "PR_ci": _ci(boot_PR),
            "EffRank_ci": _ci(boot_ER),
            "n_90_ci": [int(v) if not np.isnan(v) else 0 for v in _ci(boot_n90)],
            "n_95_ci": [int(v) if not np.isnan(v) else 0 for v in _ci(boot_n95)],
        }

    dk_ci = _bootstrap(X_dk_clean)
    fm_ci = _bootstrap(X_fm)

    dk_metrics = {**dk_base, **dk_ci}
    fm_metrics = {**fm_base, **fm_ci}

    results = {
        "fm_model_name": fm_model_name,
        "DK": dk_metrics,
        "FM": fm_metrics,
    }

    out_json = op.join(output_dir, f"dimensionality_{fm_model_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Dimensionality analysis saved: %s", out_json)

    # Plots
    dk_ev = np.cumsum(dk_eigs) / dk_eigs.sum() if len(dk_eigs) > 0 else np.array([])
    fm_ev = np.cumsum(fm_eigs) / fm_eigs.sum() if len(fm_eigs) > 0 else np.array([])
    plots_dir = op.join(op.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_eigenspectrum(dk_eigs, fm_eigs, fm_model_name, plots_dir)
    _plot_ev_curve(dk_ev, fm_ev, fm_model_name, plots_dir)

    log.info(
        "Dimensionality — DK: α=%.2f PR=%.1f EffRank=%.1f n_95=%d | "
        "FM: α=%.2f PR=%.1f EffRank=%.1f n_95=%d",
        dk_metrics.get("power_law_alpha", float("nan")),
        dk_metrics.get("participation_ratio", float("nan")),
        dk_metrics.get("effective_rank", float("nan")),
        dk_metrics.get("n_95", 0),
        fm_metrics.get("power_law_alpha", float("nan")),
        fm_metrics.get("participation_ratio", float("nan")),
        fm_metrics.get("effective_rank", float("nan")),
        fm_metrics.get("n_95", 0),
    )
    return results, dk_eigs, fm_eigs


def _plot_eigenspectrum(
    dk_eigs: np.ndarray,
    fm_eigs: np.ndarray,
    fm_model_name: str,
    plots_dir: str,
):
    """Log-log eigenspectrum plot for DK and FM spaces with OLS fit lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for eigs, label, color in [
        (dk_eigs, "DK (domain-knowledge)", "#9467bd"),
        (fm_eigs, f"FM ({fm_model_name})", "#1f77b4"),
    ]:
        if len(eigs) == 0:
            continue
        k = np.arange(1, len(eigs) + 1, dtype=float)
        ax.plot(
            np.log(k), np.log(eigs), ".", markersize=4, alpha=0.6,
            color=color, label=label,
        )
        if len(eigs) > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(np.log(k), np.log(eigs), 1)
            ax.plot(
                np.log(k), slope * np.log(k) + intercept, "-",
                color=color, linewidth=1.8, alpha=0.9,
                label=f"  fit α={-slope:.2f}",
            )

    ax.set_xlabel("log(rank $i$)")
    ax.set_ylabel("log(eigenvalue $\\lambda_i$)")
    ax.set_title(f"Eigenspectrum (log-log) — DK vs {fm_model_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = op.join(plots_dir, f"eigenspectrum_{fm_model_name}")
    plt.savefig(f"{sp}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_ev_curve(
    dk_ev: np.ndarray,
    fm_ev: np.ndarray,
    fm_model_name: str,
    plots_dir: str,
):
    """Cumulative explained variance curves for DK and FM spaces."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ev, label, color in [
        (dk_ev, "DK (domain-knowledge)", "#9467bd"),
        (fm_ev, f"FM ({fm_model_name})", "#1f77b4"),
    ]:
        if len(ev) == 0:
            continue
        k = np.arange(1, len(ev) + 1)
        ax.plot(k, ev, "-", color=color, linewidth=2, label=label)

    for thresh, ls in [(0.90, "--"), (0.95, ":")]:
        ax.axhline(
            thresh, color="gray", linestyle=ls, linewidth=0.9, alpha=0.6,
            label=f"{int(thresh * 100)}%",
        )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Explained Variance Curve — DK vs {fm_model_name}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = op.join(plots_dir, f"ev_curve_{fm_model_name}")
    plt.savefig(f"{sp}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_eigenspectrum_all(
    eigs_per_model: Dict[str, Dict[str, np.ndarray]],
    plots_dir: str,
):
    """Log-log eigenspectrum: DK (first match) + all FM models on one axes."""
    if not eigs_per_model:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    models = list(eigs_per_model.keys())

    # DK: use first model's DK as reference (all DK spectra are nearly identical)
    dk_eigs = eigs_per_model[models[0]]["DK"]
    if len(dk_eigs) > 0:
        k = np.arange(1, len(dk_eigs) + 1, dtype=float)
        ax.plot(np.log(k), np.log(dk_eigs), ".", ms=4, alpha=0.5, color="#9467bd", label="DK")
        if len(dk_eigs) > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(np.log(k), np.log(dk_eigs), 1)
            ax.plot(
                np.log(k), slope * np.log(k) + intercept, "-",
                color="#9467bd", linewidth=1.8, label=f"DK fit  α={-slope:.2f}",
            )

    for i, model in enumerate(models):
        fm_eigs = eigs_per_model[model]["FM"]
        if len(fm_eigs) == 0:
            continue
        color = _EMBED_COLORS[i % len(_EMBED_COLORS)]
        k = np.arange(1, len(fm_eigs) + 1, dtype=float)
        ax.plot(np.log(k), np.log(fm_eigs), ".", ms=4, alpha=0.5, color=color, label=model)
        if len(fm_eigs) > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(np.log(k), np.log(fm_eigs), 1)
            ax.plot(
                np.log(k), slope * np.log(k) + intercept, "-",
                color=color, linewidth=1.8, label=f"{model} α={-slope:.2f}",
            )

    ax.set_xlabel("log(rank $i$)")
    ax.set_ylabel("log(eigenvalue $\\lambda_i$)")
    ax.set_title("Eigenspectrum (log-log) — DK vs all FM models")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(op.join(plots_dir, "eigenspectrum_all.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_ev_curve_all(
    eigs_per_model: Dict[str, Dict[str, np.ndarray]],
    plots_dir: str,
):
    """Cumulative explained-variance curves: DK (first match) + all FM models."""
    if not eigs_per_model:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    models = list(eigs_per_model.keys())

    # DK reference
    dk_eigs = eigs_per_model[models[0]]["DK"]
    if len(dk_eigs) > 0:
        dk_ev = np.cumsum(dk_eigs) / dk_eigs.sum()
        ax.plot(np.arange(1, len(dk_ev) + 1), dk_ev, "-", color="#9467bd", linewidth=2.5, label="DK")

    for i, model in enumerate(models):
        fm_eigs = eigs_per_model[model]["FM"]
        if len(fm_eigs) == 0:
            continue
        fm_ev = np.cumsum(fm_eigs) / fm_eigs.sum()
        color = _EMBED_COLORS[i % len(_EMBED_COLORS)]
        ax.plot(np.arange(1, len(fm_ev) + 1), fm_ev, "-", color=color, linewidth=2, label=model)

    for thresh, ls in [(0.90, "--"), (0.95, ":")]:
        ax.axhline(thresh, color="gray", linestyle=ls, linewidth=0.9, alpha=0.6,
                   label=f"{int(thresh * 100)}%")

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Explained Variance Curves — DK vs all FM models")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(op.join(plots_dir, "ev_curve_all.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _flatten_dim_metrics(metrics_dict: Dict) -> Dict:
    """Flatten CI list fields (e.g. PR_ci: [lo, hi]) into _lower/_upper keys."""
    flat = {}
    for k, v in metrics_dict.items():
        if k.endswith("_ci") and isinstance(v, list) and len(v) == 2:
            flat[f"{k[:-3]}_ci_lower"] = v[0]
            flat[f"{k[:-3]}_ci_upper"] = v[1]
        else:
            flat[k] = v
    return flat


def save_dimensionality_csv(
    dim_results_all: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Save dimensionality metrics as a wide CSV.

    Rows   : individual metrics (point estimates + CI bounds)
    Columns: DK (from first model's match) + one column per FM model

    Parameters
    ----------
    dim_results_all : dict[model_name, results_dict]
        As returned by run_dimensionality_analysis (JSON-serialisable part).
    output_dir : where to write ``dimensionality_all_models.csv``

    Returns
    -------
    str  path to the saved CSV
    """
    models = list(dim_results_all.keys())
    if not models:
        return ""

    METRICS = [
        "n_subjects", "ambient_dim", "n_eigenvalues",
        "power_law_alpha", "power_law_r2",
        "n_80", "n_90", "n_90_ci_lower", "n_90_ci_upper",
        "n_95", "n_95_ci_lower", "n_95_ci_upper",
        "n_99",
        "participation_ratio", "PR_ci_lower", "PR_ci_upper",
        "effective_rank", "EffRank_ci_lower", "EffRank_ci_upper",
    ]

    data: Dict[str, Dict] = {}

    # DK column: use first model's DK (subject intersection is largest overlap)
    dk_flat = _flatten_dim_metrics(dim_results_all[models[0]].get("DK", {}))
    data["DK"] = {m: dk_flat.get(m, float("nan")) for m in METRICS}

    # One FM column per model
    for model in models:
        fm_flat = _flatten_dim_metrics(dim_results_all[model].get("FM", {}))
        data[model] = {m: fm_flat.get(m, float("nan")) for m in METRICS}

    df = pd.DataFrame(data, index=METRICS)
    df.index.name = "metric"

    os.makedirs(output_dir, exist_ok=True)
    csv_path = op.join(output_dir, "dimensionality_all_models.csv")
    df.to_csv(csv_path)
    log.info("Dimensionality CSV saved: %s", csv_path)
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════


def build_summary_table(
    marker_names: List[str],
    output_dir: str,
    fm_model_name: str,
    ridge_probe_dir: Optional[str] = None,
):
    """Build per-marker summary CSV with Ridge probe R² values."""
    ridge_r2: Dict[str, float] = {}
    if ridge_probe_dir:
        ridge_path = op.join(ridge_probe_dir, f"ridge_probe_summary_{fm_model_name}.json")
        if op.isfile(ridge_path):
            with open(ridge_path) as f:
                ridge_data = json.load(f)
            ridge_r2 = {m: v.get("r2", float("nan")) for m, v in ridge_data.items()}

    rows = [
        {"marker_name": mn, "ridge_r2": ridge_r2.get(mn, float("nan"))}
        for mn in marker_names
    ]
    df = pd.DataFrame(rows).sort_values("ridge_r2", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = op.join(output_dir, f"marker_summary_{fm_model_name}.csv")
    df.to_csv(csv_path, index=False)
    log.info("Marker summary saved: %s", csv_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=(
            "EEG embedding comparison: R²/noise-ceiling (C1), RSA (C2), "
            "CKA (C3), dimensionality metrics (C4)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root results directory containing NeuroLM/, TOTEM/, etc.",
    )
    parser.add_argument(
        "--marker-csv",
        required=True,
        help="Path to semicolon-delimited baseline scalars CSV.",
    )
    parser.add_argument(
        "--patient-labels",
        required=True,
        help="Path to patient_labels CSV with diagnostic_crs_final column.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--foundation-models",
        nargs="+",
        default=["NeuroLM", "CBraMod", "TOTEM", "LaBram"],
        help="Foundation model names (match subdirs in --results-dir).",
    )
    parser.add_argument(
        "--marker-reduction",
        default="A",
        choices=list(REDUCTION_MAP.keys()),
        help="Reduction letter A/B/C/D (default: A).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--skip-ridge-r2",
        action="store_true",
        help="Skip Component 1 plots (noise-ceiling-corrected R² comparison).",
    )
    parser.add_argument(
        "--skip-rsa",
        action="store_true",
        help="Skip RSA analysis (Component 2).",
    )
    parser.add_argument(
        "--skip-cka",
        action="store_true",
        help="Skip CKA analysis (Component 3).",
    )
    parser.add_argument(
        "--skip-dimensionality",
        action="store_true",
        help="Skip dimensionality metrics (Component 4).",
    )
    parser.add_argument(
        "--n-bootstrap-cka",
        type=int,
        default=1000,
        help="Bootstrap iterations for CKA CI (default: 1000).",
    )
    parser.add_argument(
        "--n-bootstrap-rsa",
        type=int,
        default=1000,
        help="Bootstrap iterations for RSA CI (default: 1000).",
    )
    parser.add_argument(
        "--n-permutations-rsa",
        type=int,
        default=1000,
        help="Permutation iterations for RSA p-value (default: 1000).",
    )
    parser.add_argument(
        "--n-bootstrap-dimensionality",
        type=int,
        default=1000,
        help="Bootstrap iterations for dimensionality CIs (default: 1000).",
    )
    parser.add_argument(
        "--rsa-distance",
        default="correlation",
        choices=["correlation", "euclidean"],
        help="Distance metric for RDMs (default: correlation).",
    )

    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("EMBEDDING COMPARISON ANALYSIS", flush=True)
    print(f"  results_dir      : {args.results_dir}", flush=True)
    print(f"  marker_csv       : {args.marker_csv}", flush=True)
    print(f"  patient_labels   : {args.patient_labels}", flush=True)
    print(f"  output_dir       : {args.output_dir}", flush=True)
    print(f"  foundation_models: {args.foundation_models}", flush=True)
    print(f"  marker_reduction : {args.marker_reduction}", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load DK embeddings ─────────────────────────────────────────────────
    X_dk_all, dk_keys_all, marker_names = _load_domain_knowledge_embeddings(
        args.marker_csv, args.marker_reduction
    )

    # ── Load patient labels ────────────────────────────────────────────────
    labels_dict = _load_patient_labels(args.patient_labels)
    log.info("Loaded %d labeled subject-sessions.", len(labels_dict))

    rsa_results_all: Dict[str, Dict] = {}
    cka_results_all: Dict[str, Dict] = {}
    rsa_tri_per_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    dim_results_all: Dict[str, Dict] = {}
    dim_eigs_per_model: Dict[str, Dict[str, np.ndarray]] = {}

    for fm_name in args.foundation_models:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  Foundation model: {fm_name}", flush=True)
        print(f"{'─' * 60}", flush=True)

        # Path to foundation model embeddings
        fm_emb_dir = op.join(
            args.results_dir, fm_name,
            "doc_patients/MLP_EMBEDDING/pooled_embeddings",
        )
        if not op.isdir(fm_emb_dir):
            log.warning("FM dir not found: %s — skipping.", fm_emb_dir)
            continue

        try:
            X_fm_all, fm_keys_all = _load_foundation_embeddings(fm_emb_dir)
        except RuntimeError as exc:
            log.warning("Could not load FM embeddings for %s: %s", fm_name, exc)
            continue

        # ── Subject matching ───────────────────────────────────────────────
        dk_idx, fm_idx, y, matched_keys = _match_subjects(
            dk_keys_all, fm_keys_all, labels_dict
        )
        if len(matched_keys) < 5:
            log.error(
                "Only %d subjects in intersection for %s — skipping.",
                len(matched_keys),
                fm_name,
            )
            continue

        X_dk = X_dk_all[dk_idx]
        X_fm = X_fm_all[fm_idx]
        log.info(
            "Matched %d subjects: X_dk=%s, X_fm=%s",
            len(matched_keys),
            X_dk.shape,
            X_fm.shape,
        )

        # Output subdirs (one shared dir per component, all models)
        c1_dir = op.join(args.output_dir, "component1")
        c2_dir = op.join(args.output_dir, "component2_rsa")
        c3_dir = op.join(args.output_dir, "component3_cka")
        c4_dir = op.join(args.output_dir, "component4_dimensionality")
        summary_dir = op.join(args.output_dir, "summary")
        for d in [c1_dir, c2_dir, c3_dir, c4_dir, summary_dir]:
            os.makedirs(d, exist_ok=True)

        # ── Ridge probe (Component 1 support — weights + R² per marker) ───
        ridge_weights_path = op.join(c1_dir, f"ridge_weights_{fm_name}.npy")
        if op.isfile(ridge_weights_path):
            log.info("Loaded cached Ridge weights: %s", ridge_weights_path)
        else:
            log.info("Fitting Ridge probe for %s ...", fm_name)
            fit_ridge_probe(
                X_fm, X_dk, marker_names, c1_dir, fm_name,
                random_state=args.random_state,
            )

        # ── Component 2: RSA ───────────────────────────────────────────────
        if not args.skip_rsa:
            log.info("Running RSA for %s ...", fm_name)
            try:
                rsa_result, tri_dk, tri_fm = run_rsa_analysis(
                    X_dk=X_dk,
                    X_fm=X_fm,
                    y=y,
                    output_dir=c2_dir,
                    fm_model_name=fm_name,
                    n_bootstrap=args.n_bootstrap_rsa,
                    n_permutations=args.n_permutations_rsa,
                    distance_metric=args.rsa_distance,
                    random_state=args.random_state,
                )
                rsa_results_all[fm_name] = rsa_result
                rsa_tri_per_model[fm_name] = (tri_dk, tri_fm)
            except Exception as exc:
                log.error("RSA failed for %s: %s", fm_name, exc, exc_info=True)

        # ── Component 3: CKA ──────────────────────────────────────────────
        if not args.skip_cka:
            log.info("Running CKA for %s ...", fm_name)
            try:
                cka_result = run_cka_analysis(
                    X_dk=X_dk,
                    X_fm=X_fm,
                    y=y,
                    output_dir=c3_dir,
                    fm_model_name=fm_name,
                    n_bootstrap=args.n_bootstrap_cka,
                    random_state=args.random_state,
                )
                cka_results_all[fm_name] = cka_result
            except Exception as exc:
                log.error("CKA failed for %s: %s", fm_name, exc, exc_info=True)

        # ── Component 4: Dimensionality Metrics ───────────────────────────
        if not args.skip_dimensionality:
            log.info("Running dimensionality analysis for %s ...", fm_name)
            try:
                dim_result, dk_eigs, fm_eigs = run_dimensionality_analysis(
                    X_dk=X_dk,
                    X_fm=X_fm,
                    output_dir=c4_dir,
                    fm_model_name=fm_name,
                    n_bootstrap=args.n_bootstrap_dimensionality,
                    random_state=args.random_state,
                )
                dim_results_all[fm_name] = dim_result
                dim_eigs_per_model[fm_name] = {"DK": dk_eigs, "FM": fm_eigs}
            except Exception as exc:
                log.error(
                    "Dimensionality analysis failed for %s: %s", fm_name, exc,
                    exc_info=True,
                )

        # ── Summary table ─────────────────────────────────────────────────
        build_summary_table(
            marker_names=marker_names,
            output_dir=summary_dir,
            fm_model_name=fm_name,
            ridge_probe_dir=c1_dir,
        )

    # ── Cross-model summaries ──────────────────────────────────────────────
    plots_dir = op.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if rsa_results_all and not args.skip_rsa:
        save_rsa_summary(rsa_results_all, c2_dir)
        if len(rsa_tri_per_model) > 1:
            _plot_rsa_scatter_all(rsa_tri_per_model, rsa_results_all, plots_dir)

    if cka_results_all and not args.skip_cka:
        save_cka_summary(cka_results_all, c3_dir)

    if dim_results_all and not args.skip_dimensionality:
        save_dimensionality_csv(dim_results_all, c4_dir)
        _plot_eigenspectrum_all(dim_eigs_per_model, plots_dir)
        _plot_ev_curve_all(dim_eigs_per_model, plots_dir)

    # ── Component 1: Noise-ceiling-corrected R² comparison ────────────────
    if not args.skip_ridge_r2:
        log.info("Running Component 1: noise-ceiling R² comparison ...")
        c1_plots_dir = op.join(args.output_dir, "component1")
        os.makedirs(c1_plots_dir, exist_ok=True)
        try:
            from feature_importance_comparison import (  # noqa: PLC0415
                plot_comparison as plot_r2_comparison,
            )
            plot_r2_comparison(
                results_dir=args.results_dir,
                marker_csv=args.marker_csv,
                marker_reduction=args.marker_reduction,
                output_dir=c1_plots_dir,
            )
        except Exception as exc:
            log.error("Component 1 (R² comparison) failed: %s", exc, exc_info=True)

    print("\n" + "=" * 70, flush=True)
    print("EMBEDDING COMPARISON COMPLETE", flush=True)
    print(f"  Output directory: {args.output_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
