"""Run nested CV classification on pooled embeddings concatenated with DK markers.

This script combines per-session foundation model pooled embeddings (dimension D)
with domain-knowledge markers from baseline scalars CSV (all available markers by
default, or evoked-only subset via ``--evoked-only``) to obtain D+M features per
subject/session, then reuses the same classification pipeline as
``mlp_embedding_classifier.py`` (MLP, Random Forest, Kernel Ridge).

Common session pool
-------------------
Before any model is run, the intersection of sessions across all FM models,
DK markers, and the labels file is computed.  All models are then evaluated on
this common pool using **identical** nested CV folds (StratifiedGroupKFold outer,
StratifiedGroupKFold inner) so that performance comparisons are fair.

Shifted (permutation) variants
--------------------------------
Three additional "null" conditions permute the FM and/or DK feature blocks
**within each fold** (train and test independently, with different seeds) to
break the feature-label association for the permuted modality:

- ``shift_fm``   : FM block permuted, DK block intact
- ``shift_dk``   : DK block permuted, FM block intact
- ``shift_both`` : both blocks permuted (with independent permutations)

Output layout per foundation model:
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED/
        {target}/nested_cv/{classifier_name}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_FM_SHIFTED/
        {target}/nested_cv/{classifier_name}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_DK_SHIFTED/
        {target}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_BOTH_SHIFTED/
        {target}/...
"""

import argparse
from datetime import datetime
import joblib
import json
import os
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP
    from .cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
        load_cv_splits,
    )
    from .kernel_ridge_classifier import KernelRidgeClassifier
except ImportError:
    from mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP
    from cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
        load_cv_splits,
    )
    from kernel_ridge_classifier import KernelRidgeClassifier


DEFAULT_RESULTS_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
DEFAULT_MARKER_CSV = (
    "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
)
DEFAULT_PATIENT_LABELS = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)
DEFAULT_PATIENT_LABELS_FULL = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)

DEFAULT_POOLED_SUBPATH = op.join("doc_patients", "MLP_EMBEDDING", "pooled_embeddings")
TARGET_CHOICES = ["crs", "etiology", "cs_6m", "cs_1y", "cs_2y"]

# Evoked markers (TimeLocked, WindowDecoding, CNV) + SymbolicMutualInformation
EVOKED_MARKER_PREFIXES = (
    "nice/marker/TimeLocked",
    "nice/marker/WindowDecoding",
    "nice/marker/ContingentNegativeVariation",
    "nice/marker/SymbolicMutualInformation",
    "nice/marker/PermutationEntropy",
    "nice/marker/KolmogorovComplexity",
    "nice/marker/PowerSpectralDensity",
)

SHIFT_MODES = ("shift_fm", "shift_dk", "shift_both")


def discover_foundation_models(results_root, pooled_subpath):
    """Return model directory names that contain a pooled embeddings folder."""
    if not op.isdir(results_root):
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    model_names = []
    for name in sorted(os.listdir(results_root)):
        candidate = op.join(results_root, name, pooled_subpath)
        if op.isdir(candidate):
            model_names.append(name)
    return model_names


class DKCombinedEmbeddingClassifier(EmbeddingClassifier):
    """Embedding classifier with marker concatenation: X = [FM || DK].

    After ``load_embeddings`` is called, the instance attributes
    ``n_fm_features`` and ``n_dk_features`` record the block sizes so that
    downstream permutation utilities can split the combined feature matrix.
    """

    def __init__(
        self,
        *args,
        marker_csv,
        marker_reduction,
        expected_marker_dim=None,
        evoked_only=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pooled_embeddings_dir = self.data_dir
        self.marker_csv = marker_csv
        self.marker_reduction = marker_reduction
        self.expected_marker_dim = expected_marker_dim
        self.evoked_only = evoked_only
        self._markers_dict = None
        self._marker_names = None
        self.n_fm_features: int | None = None
        self.n_dk_features: int | None = None

    def _load_markers_once(self):
        if self._markers_dict is None or self._marker_names is None:
            markers_dict, marker_names = self.load_markers_from_csv(
                self.marker_csv, self.marker_reduction
            )

            if self.evoked_only:
                keep_idx = [
                    i
                    for i, name in enumerate(marker_names)
                    if name.startswith(EVOKED_MARKER_PREFIXES)
                ]
                marker_names = [marker_names[i] for i in keep_idx]
                markers_dict = {key: vec[keep_idx] for key, vec in markers_dict.items()}
                print(
                    f"   Filtered to {len(marker_names)} evoked markers",
                    flush=True,
                )
            else:
                print(
                    f"   Using all {len(marker_names)} DK markers",
                    flush=True,
                )

            self._markers_dict = markers_dict
            self._marker_names = marker_names

            if (
                self.expected_marker_dim is not None
                and len(self._marker_names) != self.expected_marker_dim
            ):
                print(
                    "Warning: expected "
                    f"{self.expected_marker_dim} marker dimensions, got "
                    f"{len(self._marker_names)}.",
                    flush=True,
                )
        return self._markers_dict, self._marker_names

    def load_embeddings(self, embedding_suffix="_embedding.npy"):
        """Load pooled embeddings and append marker vectors per session key.

        Sets ``self.n_fm_features`` and ``self.n_dk_features`` for use by
        permutation utilities after this method is called.
        """
        base_embeddings = super().load_embeddings(embedding_suffix=embedding_suffix)
        markers_dict, marker_names = self._load_markers_once()

        combined = {}
        n_missing_markers = 0
        n_nan_rows = 0

        fm_dim: int | None = None
        dk_dim: int | None = None

        for key, emb in sorted(base_embeddings.items()):
            marker_vec = markers_dict.get(key)
            if marker_vec is None:
                n_missing_markers += 1
                continue

            emb_vec = np.asarray(emb, dtype=float).reshape(-1)
            marker_vec = np.asarray(marker_vec, dtype=float).reshape(-1)

            if np.isnan(emb_vec).any() or np.isnan(marker_vec).any():
                n_nan_rows += 1
                continue

            if fm_dim is None:
                fm_dim = int(emb_vec.shape[0])
                dk_dim = int(marker_vec.shape[0])

            combined[key] = np.concatenate([emb_vec, marker_vec], axis=0)

        if not combined:
            raise ValueError(
                "No combined embeddings available after intersecting pooled "
                "embeddings with marker CSV."
            )

        self.n_fm_features = fm_dim
        self.n_dk_features = dk_dim

        first_key = next(iter(combined))
        total_dim = int(combined[first_key].shape[0])
        print(
            "   Combined embeddings ready: "
            f"{len(combined)} sessions, fm_dim={fm_dim}, marker_dim={len(marker_names)}, "
            f"total_dim={total_dim}, missing_markers={n_missing_markers}, "
            f"nan_rows={n_nan_rows}",
            flush=True,
        )
        return combined


# ===========================================================================
# Shifted (permutation) CV utilities
# ===========================================================================


def _permute_block(X, n_fm_features, shift_mode, rng):
    """Return a copy of X with FM and/or DK feature blocks permuted row-wise.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Combined feature matrix [FM || DK].
    n_fm_features : int
        Number of FM features (columns 0 … n_fm_features-1).
    shift_mode : str
        One of ``shift_fm``, ``shift_dk``, ``shift_both``.
    rng : np.random.RandomState
        Source of randomness.  Caller is responsible for seeding it in a
        way that is independent between train and test permutations.

    Returns
    -------
    X_perm : ndarray, same shape as X (copy).
    """
    X_out = X.copy()
    n = X.shape[0]

    if shift_mode in ("shift_fm", "shift_both"):
        perm = rng.permutation(n)
        X_out[:, :n_fm_features] = X[perm, :n_fm_features]

    if shift_mode in ("shift_dk", "shift_both"):
        perm = rng.permutation(n)
        X_out[:, n_fm_features:] = X[perm, n_fm_features:]

    return X_out


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _get_classifier_and_grid(classifier_name, random_state):
    """Return (estimator, param_grid) for MLP, RF, or KR."""
    if classifier_name == "mlp":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=500, early_stopping=True, random_state=random_state
                    ),
                ),
            ]
        )
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 1e-2],
        }
    elif classifier_name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=1
        )
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
    elif classifier_name == "kernel_ridge":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kr", KernelRidgeClassifier(kernel="rbf")),
            ]
        )
        param_grid = {
            "kr__alpha": [0.01, 0.1, 1.0, 10.0],
            "kr__gamma": [0.001, 0.01, 0.1, 1.0],
        }
    elif classifier_name == "svm":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_grid = {
            "svc__C": [0.01, 0.1, 1.0, 10.0],
            "svc__kernel": ["rbf", "linear"],
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return estimator, param_grid


def _predict(estimator, X_test, classifier_name):
    """Return (y_pred, y_score) for any of the three classifiers."""
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
        y_score = proba[:, 1] if proba.ndim == 2 else np.asarray(proba).reshape(-1)
    elif classifier_name == "kernel_ridge":
        if isinstance(estimator, Pipeline):
            raw = estimator.named_steps["kr"].decision_function(
                estimator.named_steps["scaler"].transform(X_test)
            )
        else:
            raw = estimator.decision_function(X_test)
        y_score = np.clip(np.asarray(raw).reshape(-1), 0.0, 1.0)
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X_test)).reshape(-1)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        y_score = np.asarray(y_pred).astype(float)
    return np.asarray(y_pred).astype(int), np.asarray(y_score).astype(float)


def run_shifted_nested_cv(
    X,
    y,
    groups,
    n_fm_features,
    shift_mode,
    cv_folds,
    random_state,
    classifiers=("mlp", "random_forest", "kernel_ridge", "svm"),
    output_dir=None,
    session_keys=None,
):
    """Run nested CV with within-fold permutation of FM and/or DK blocks.

    For each outer fold:
      1. Permute the specified block(s) of X_train and X_test **independently**
         using seeds derived from ``random_state + fold_idx * 100 ± block_offset``.
      2. Train on permuted X_train, evaluate on permuted X_test.

    Parameters
    ----------
    X : ndarray, shape (n_sessions, n_features)
        Combined [FM || DK] feature matrix ordered to match *cv_folds*.
    y, groups : ndarrays of shape (n_sessions,)
    n_fm_features : int
        Column split point between FM and DK blocks.
    shift_mode : str
        One of ``shift_fm``, ``shift_dk``, ``shift_both``.
    cv_folds : list of fold dicts
        From :func:`cv_utils.generate_nested_cv_folds`.
    random_state : int
    classifiers : sequence of str
    output_dir : str or None
        If provided, fitted models are saved as
        ``{output_dir}/nested_cv/{clf_name}/fold_{N:02d}_model.joblib``.
    session_keys : list of str or None
        Ordered session keys aligned with rows of *X* (i.e. ``common_sessions``).
        Used to populate the ``subjects`` accumulator for downstream plotting.

    Returns
    -------
    tuple (fold_results, accum_preds)
        fold_results : dict mapping classifier_name → list of fold-metric dicts.
        accum_preds  : dict mapping classifier_name → dict with keys
                       ``y_true``, ``y_pred``, ``y_proba``, ``subjects``
                       (lists of per-fold arrays / session key lists).
    """
    results = {clf: [] for clf in classifiers}
    accum_preds = {
        clf: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
        for clf in classifiers
    }

    for fold_idx, fold in enumerate(cv_folds):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold.get("inner_splits")

        # Independent RNGs for train/test permutations.
        rng_train = np.random.RandomState(random_state + fold_idx * 100)
        rng_test = np.random.RandomState(random_state + fold_idx * 100 + 1)

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        X_train = _permute_block(X_train_raw, n_fm_features, shift_mode, rng_train)
        X_test = _permute_block(X_test_raw, n_fm_features, shift_mode, rng_test)

        for clf_name in classifiers:
            estimator, param_grid = _get_classifier_and_grid(clf_name, random_state)
            cv_for_gs = inner_splits if inner_splits is not None else 3

            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=cv_for_gs,
                n_jobs=1,
                refit=True,
            )
            gs.fit(X_train, y_train)

            y_pred, y_score = _predict(gs.best_estimator_, X_test, clf_name)
            auc = _safe_auc(y_test, y_score)

            # Save fitted model if output_dir provided.
            if output_dir is not None:
                clf_dir = op.join(output_dir, "nested_cv", clf_name)
                os.makedirs(clf_dir, exist_ok=True)
                joblib.dump(
                    gs.best_estimator_,
                    op.join(clf_dir, f"fold_{fold_idx:02d}_model.joblib"),
                )

            # Accumulate predictions for downstream micro/macro metrics and plots.
            accum_preds[clf_name]["y_true"].append(y_test)
            accum_preds[clf_name]["y_pred"].append(y_pred)
            accum_preds[clf_name]["y_proba"].append(y_score)
            if session_keys is not None:
                accum_preds[clf_name]["subjects"].extend(
                    [session_keys[i] for i in test_idx]
                )

            results[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "auc": auc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(g_train))),
                    "best_params": gs.best_params_,
                    "shift_mode": shift_mode,
                }
            )

    return results, accum_preds


def _summarize_fold_metrics(fold_rows):
    """Aggregate fold-level metric dicts into mean ± std summary."""
    auc_vals = [r["auc"] for r in fold_rows if r["auc"] is not None]
    bal_vals = [r["balanced_accuracy"] for r in fold_rows]
    return {
        "n_folds": len(fold_rows),
        "balanced_accuracy_mean": float(np.mean(bal_vals)),
        "balanced_accuracy_std": float(np.std(bal_vals)),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
        "auc_std": float(np.std(auc_vals)) if auc_vals else None,
    }


# ===========================================================================
# Model runners
# ===========================================================================


def _validate_requested_models(requested_models, discovered_models):
    if requested_models is None:
        return discovered_models

    discovered_set = set(discovered_models)
    missing = [m for m in requested_models if m not in discovered_set]
    if missing:
        raise ValueError(
            "Requested FM model(s) not found with pooled embeddings path: "
            f"{missing}. Available: {discovered_models}"
        )
    return requested_models


def _run_target_for_model(
    model_name,
    pooled_dir,
    args,
    target,
    precomputed_splits,
    common_sessions,
):
    """Run original (non-shifted) DK+FM classification for one model/target."""
    out_base = op.join(
        args.results_root,
        model_name,
        "doc_patients",
        "EMBEDDING_DK_COMBINED",
        target,
    )
    os.makedirs(out_base, exist_ok=True)

    classifier = DKCombinedEmbeddingClassifier(
        data_dir=pooled_dir,
        patient_labels_file=args.patient_labels,
        output_dir=out_base,
        random_state=args.random_state,
        full_cv=True,
        n_cv_folds=args.n_cv_folds,
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        expected_marker_dim=args.expected_marker_dim,
        evoked_only=args.evoked_only,
    )

    print("=" * 80, flush=True)
    print(f"FM model: {model_name} | target: {target} | mode: original", flush=True)
    print(f"Input pooled embeddings: {pooled_dir}", flush=True)
    print(f"Output directory: {out_base}", flush=True)
    print("=" * 80, flush=True)

    cv_kwargs = dict(
        precomputed_splits=precomputed_splits,
        common_sessions=common_sessions,
    )

    if target == "crs":
        classifier.run_full_cv(target="crs", **cv_kwargs)
    else:
        classifier.run_full_cv(
            target=target,
            labels_file=args.patient_labels_full,
            binary_outcome=args.binary_outcome,
            death_binary=args.death_binary,
            **cv_kwargs,
        )

    # Return n_fm_features for the shifted runs.
    return classifier.n_fm_features


def _run_shifted_target_for_model(
    model_name,
    pooled_dir,
    args,
    target,
    shift_mode,
    precomputed_splits,
    common_sessions,
    n_fm_features,
):
    """Run one shifted variant for one model/target using precomputed splits."""
    suffix_map = {
        "shift_fm": "EMBEDDING_DK_COMBINED_FM_SHIFTED",
        "shift_dk": "EMBEDDING_DK_COMBINED_DK_SHIFTED",
        "shift_both": "EMBEDDING_DK_COMBINED_BOTH_SHIFTED",
    }
    out_base = op.join(
        args.results_root,
        model_name,
        "doc_patients",
        suffix_map[shift_mode],
        target,
    )
    os.makedirs(out_base, exist_ok=True)

    print("=" * 80, flush=True)
    print(
        f"FM model: {model_name} | target: {target} | mode: {shift_mode}",
        flush=True,
    )
    print(f"Output directory: {out_base}", flush=True)
    print("=" * 80, flush=True)

    # Build combined feature matrix for common_sessions.
    classifier = DKCombinedEmbeddingClassifier(
        data_dir=pooled_dir,
        patient_labels_file=args.patient_labels,
        output_dir=out_base,
        random_state=args.random_state,
        full_cv=True,
        n_cv_folds=args.n_cv_folds,
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        expected_marker_dim=args.expected_marker_dim,
        evoked_only=args.evoked_only,
    )

    # collect_data returns (X, y, subjects); use common_sessions ordering.
    X_all, y_all, subjects_all = classifier.collect_data(target="crs")

    session_to_idx = {s: i for i, s in enumerate(subjects_all)}
    missing = [s for s in common_sessions if s not in session_to_idx]
    if missing:
        raise ValueError(
            f"[{model_name}/{shift_mode}] {len(missing)} common sessions missing: "
            f"{missing[:3]}"
        )
    keep = [session_to_idx[s] for s in common_sessions]
    X = X_all[keep]
    y = y_all[keep]
    groups = np.array([common_sessions[i].split("_ses-")[0] for i in range(len(keep))])

    if n_fm_features is None:
        # Fall back: load embeddings to determine block size.
        _ = classifier.load_embeddings()
        n_fm_features = classifier.n_fm_features

    fold_results, accum_preds = run_shifted_nested_cv(
        X=X,
        y=y,
        groups=groups,
        n_fm_features=n_fm_features,
        shift_mode=shift_mode,
        cv_folds=precomputed_splits,
        random_state=args.random_state,
        output_dir=out_base,
        session_keys=common_sessions,
    )

    _clf_display = {
        "mlp": "MLP",
        "random_forest": "Random Forest",
        "kernel_ridge": "Kernel Ridge",
        "svm": "SVM",
    }

    # Save per-classifier results: CSV + JSON + same plots as original variant.
    for clf_name, fold_rows in fold_results.items():
        clf_dir = op.join(out_base, "nested_cv", clf_name)
        os.makedirs(clf_dir, exist_ok=True)

        for row in fold_rows:
            row["model"] = model_name
            row["shift_mode"] = shift_mode
            row["classifier"] = clf_name
            row["target"] = target
            row["best_params"] = json.dumps(row["best_params"], sort_keys=True)

        pd.DataFrame(fold_rows).to_csv(
            op.join(clf_dir, "fold_metrics.csv"), index=False
        )
        summary = _summarize_fold_metrics(fold_rows)
        summary.update(
            {
                "model": model_name,
                "shift_mode": shift_mode,
                "classifier": clf_name,
                "target": target,
            }
        )
        with open(op.join(clf_dir, "summary_metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        auc_str = (
            f"{summary['auc_mean']:.3f}" if summary["auc_mean"] is not None else "N/A"
        )
        print(
            f"  [{model_name}|{shift_mode}|{clf_name}] "
            f"bal_acc={summary['balanced_accuracy_mean']:.3f} "
            f"auc={auc_str}",
            flush=True,
        )

        # Generate plots using accumulated predictions (mirrors original variant).
        clf_accum = accum_preds[clf_name]
        if clf_accum["y_true"]:
            y_true_all = np.concatenate(clf_accum["y_true"])
            y_pred_all = np.concatenate(clf_accum["y_pred"])
            y_proba_all = np.concatenate(clf_accum["y_proba"])
            subjects_all = clf_accum["subjects"]

            plot_results = classifier._compute_test_metrics(
                y_true_all, y_pred_all, y_proba_all, subjects_all
            )
            plot_results.update(
                {
                    "n_samples": int(len(y_true_all)),
                    "n_features": int(X.shape[1]),
                    "n_subjects": int(
                        len(np.unique([s.split("_ses-")[0] for s in subjects_all]))
                        if subjects_all
                        else 0
                    ),
                    "full_cv": True,
                    "n_folds": len(fold_rows),
                    "shift_mode": shift_mode,
                }
            )
            macro_results = classifier._compute_macro_average(
                clf_accum["y_true"],
                clf_accum["y_pred"],
                clf_accum["y_proba"],
            )
            plot_results["macro_average"] = macro_results

            display_name = f"{_clf_display.get(clf_name, clf_name)} [{shift_mode}]"
            classifier._save_results(
                plot_results, clf_dir, model=None, model_type=clf_name
            )
            classifier._plot_results(plot_results, display_name, clf_dir)
            classifier._plot_macro_vs_micro(plot_results, display_name, clf_dir)

    return fold_results


# ===========================================================================
# Comparison plot
# ===========================================================================


def plot_shifted_comparison(all_results, output_dir):
    """Bar plot comparing original vs shifted performance.

    Parameters
    ----------
    all_results : dict
        ``{model_name: {condition: {clf_name: summary_dict}}}``
        where *condition* is one of ``original``, ``shift_fm``, ``shift_dk``,
        ``shift_both``.
    output_dir : str
        Directory where PNG files are saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = ["original", "shift_fm", "shift_dk", "shift_both"]
    classifiers = ["mlp", "random_forest", "kernel_ridge", "svm"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 120

    for clf_name in classifiers:
        models = sorted(all_results.keys())
        n_models = len(models)
        n_cond = len(conditions)
        x = np.arange(n_models)
        width = 0.8 / n_cond
        offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))
        for ci, cond in enumerate(conditions):
            auc_vals = []
            for model in models:
                cond_data = all_results.get(model, {}).get(cond, {})
                clf_data = cond_data.get(clf_name, {})
                auc_vals.append(clf_data.get("auc_mean"))

            y_plot = [v if v is not None else 0.0 for v in auc_vals]
            ax.bar(x + offsets[ci], y_plot, width, label=cond)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("AUC (mean across folds)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
        ax.set_title(f"Original vs Shifted — {clf_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_file = op.join(output_dir, f"shifted_comparison_{clf_name}.png")
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate pooled FM embeddings with domain-knowledge marker "
            "embeddings and run nested CV classification (MLP/RF/KR).  "
            "All models share a common session pool and identical fold "
            "assignments for fair comparison.  Optional shifted (permuted) "
            "variants test the contribution of each modality."
        )
    )

    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing {FM_model}/doc_patients/MLP_EMBEDDING/pooled_embeddings",
    )
    parser.add_argument(
        "--pooled-subpath",
        default=DEFAULT_POOLED_SUBPATH,
        help="Relative path under each model directory where pooled embeddings live",
    )
    parser.add_argument(
        "--fm-models",
        nargs="+",
        default=None,
        help="Optional list of FM models to run (default: auto-discover all)",
    )
    parser.add_argument(
        "--feature-predicted",
        nargs="+",
        default=["crs"],
        choices=TARGET_CHOICES,
        help="Prediction target(s). Results are written under this folder name.",
    )

    parser.add_argument("--marker-csv", default=DEFAULT_MARKER_CSV)
    parser.add_argument(
        "--marker-reduction",
        choices=sorted(REDUCTION_MAP.keys()),
        default="A",
        help="Marker reduction letter from baseline CSV",
    )
    parser.add_argument(
        "--expected-marker-dim",
        type=int,
        default=None,
        help="Expected number of marker dimensions (warns if different)",
    )
    parser.add_argument(
        "--evoked-only",
        action="store_true",
        default=False,
        help="Only use evoked markers (TimeLocked, WindowDecoding, CNV, etc.)",
    )

    parser.add_argument("--patient-labels", default=DEFAULT_PATIENT_LABELS)
    parser.add_argument("--patient-labels-full", default=DEFAULT_PATIENT_LABELS_FULL)

    parser.add_argument(
        "--binary-outcome",
        action="store_true",
        default=False,
        help="For cs_* targets: map to binary VS vs MCS",
    )
    parser.add_argument(
        "--death-binary",
        action="store_true",
        default=False,
        help="For cs_* targets: map to DEATH vs NON_DEATH",
    )

    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--shift-modes",
        nargs="+",
        default=list(SHIFT_MODES),
        choices=list(SHIFT_MODES),
        help=(
            "Shifted (permutation) variants to run. "
            "Default: all three (shift_fm shift_dk shift_both). "
            "Pass an empty list to skip shifted variants."
        ),
    )
    parser.add_argument(
        "--no-shifts",
        action="store_true",
        default=False,
        help="Skip all shifted variants (equivalent to --shift-modes with no args).",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Path to a pre-computed common_cv_splits.json. "
            "When provided, the stored splits are used instead of generating new ones."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help="Path to save the generated common CV splits JSON.",
    )

    args = parser.parse_args()

    if args.binary_outcome and args.death_binary:
        raise ValueError("Choose at most one of --binary-outcome or --death-binary.")

    shift_modes_to_run = [] if args.no_shifts else args.shift_modes

    discovered_models = discover_foundation_models(
        args.results_root, args.pooled_subpath
    )
    selected_models = _validate_requested_models(args.fm_models, discovered_models)
    if not selected_models:
        raise ValueError(
            "No foundation models found. Check --results-root and --pooled-subpath."
        )

    print(f"Discovered FM models: {discovered_models}", flush=True)
    print(f"Selected FM models: {selected_models}", flush=True)
    print(f"Targets: {args.feature_predicted}", flush=True)
    print(f"Shift modes: {shift_modes_to_run}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Build common session pool (intersection across ALL FM models
    # + DK markers + labels).
    # ------------------------------------------------------------------
    print("\nBuilding common session pool ...", flush=True)

    # Load CRS labels as a simple dict for cv_utils.
    _ldf = pd.read_csv(args.patient_labels)
    labels_dict = {}
    for _, row in _ldf.iterrows():
        state = row.get("diagnostic_crs_final", "")
        if pd.isna(state) or state == "n/a":
            continue
        if state == "UWS":
            lbl = 0
        elif state in ("MCS+", "MCS-"):
            lbl = 1
        else:
            continue
        try:
            subj = str(row["subject"])
            ses = f"ses-{int(row['session']):02d}"
        except Exception:
            continue
        labels_dict[f"{subj}_{ses}"] = lbl

    def _make_combined_loader(
        pooled_dir,
        marker_csv,
        marker_reduction,
        evoked_only,
        expected_marker_dim,
        random_state,
        patient_labels,
    ):
        def _loader():
            tmp = DKCombinedEmbeddingClassifier(
                data_dir=pooled_dir,
                patient_labels_file=patient_labels,
                output_dir="/tmp",
                random_state=random_state,
                full_cv=True,
                marker_csv=marker_csv,
                marker_reduction=marker_reduction,
                expected_marker_dim=expected_marker_dim,
                evoked_only=evoked_only,
            )
            return tmp.load_embeddings()

        return _loader

    source_loaders = {
        model_name: _make_combined_loader(
            pooled_dir=op.join(args.results_root, model_name, args.pooled_subpath),
            marker_csv=args.marker_csv,
            marker_reduction=args.marker_reduction,
            evoked_only=args.evoked_only,
            expected_marker_dim=args.expected_marker_dim,
            random_state=args.random_state,
            patient_labels=args.patient_labels,
        )
        for model_name in selected_models
    }

    common_sessions = build_common_session_pool(source_loaders, labels_dict)
    print(
        f"Common pool: {len(common_sessions)} sessions, "
        f"{len({k.split('_ses-')[0] for k in common_sessions})} subjects\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Generate or load nested CV splits from the common pool.
    # ------------------------------------------------------------------
    if args.splits_file and op.isfile(args.splits_file):
        print(f"Loading pre-computed splits from {args.splits_file}", flush=True)
        precomputed_splits, common_sessions, labels_dict = load_cv_splits(
            args.splits_file
        )
    else:
        print("Generating nested CV splits from common pool ...", flush=True)
        precomputed_splits = generate_nested_cv_folds(
            common_sessions=common_sessions,
            labels=labels_dict,
            n_outer=args.n_cv_folds,
            random_state=args.random_state,
        )
        if args.save_splits_to:
            save_cv_splits(
                folds=precomputed_splits,
                common_sessions=common_sessions,
                labels=labels_dict,
                path=args.save_splits_to,
            )
            print(f"Splits saved to {args.save_splits_to}", flush=True)

    if not (args.splits_file and op.isfile(args.splits_file)):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_tag = "_".join(sorted(args.feature_predicted))
        splits_dir = op.join(args.results_root, "cv_splits")
        os.makedirs(splits_dir, exist_ok=True)
        default_splits_path = op.join(splits_dir, f"dk_combined_{target_tag}_{ts}.json")
        save_cv_splits(
            folds=precomputed_splits,
            common_sessions=common_sessions,
            labels=labels_dict,
            path=default_splits_path,
        )
        print(f"Common splits saved to: {default_splits_path}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Run original + shifted variants for each model × target.
    # ------------------------------------------------------------------
    run_log = {
        "results_root": args.results_root,
        "pooled_subpath": args.pooled_subpath,
        "marker_csv": args.marker_csv,
        "marker_reduction": args.marker_reduction,
        "n_common_sessions": len(common_sessions),
        "n_outer_folds": len(precomputed_splits),
        "models": {},
    }

    # Accumulate results for comparison plots.
    all_plot_results: dict[str, dict] = {}

    for model_name in selected_models:
        pooled_dir = op.join(args.results_root, model_name, args.pooled_subpath)
        run_log["models"][model_name] = {}
        all_plot_results[model_name] = {}

        for target in args.feature_predicted:
            # ---- Original ----
            try:
                n_fm_features = _run_target_for_model(
                    model_name=model_name,
                    pooled_dir=pooled_dir,
                    args=args,
                    target=target,
                    precomputed_splits=precomputed_splits,
                    common_sessions=common_sessions,
                )
                run_log["models"][model_name][target] = {"original": "ok"}
            except Exception as exc:
                print(
                    f"[FAILED] original model={model_name}, target={target}: {exc}",
                    flush=True,
                )
                run_log["models"][model_name][target] = {
                    "original": {"status": "failed", "error": str(exc)}
                }
                n_fm_features = None

            # ---- Shifted variants ----
            for shift_mode in shift_modes_to_run:
                try:
                    fold_results = _run_shifted_target_for_model(
                        model_name=model_name,
                        pooled_dir=pooled_dir,
                        args=args,
                        target=target,
                        shift_mode=shift_mode,
                        precomputed_splits=precomputed_splits,
                        common_sessions=common_sessions,
                        n_fm_features=n_fm_features,
                    )
                    # Summarise for plots.
                    if model_name not in all_plot_results:
                        all_plot_results[model_name] = {}
                    all_plot_results[model_name][shift_mode] = {
                        clf: _summarize_fold_metrics(rows)
                        for clf, rows in fold_results.items()
                    }
                    run_log["models"][model_name][target][shift_mode] = "ok"
                except Exception as exc:
                    print(
                        f"[FAILED] {shift_mode} model={model_name}, target={target}: {exc}",
                        flush=True,
                    )
                    run_log["models"][model_name][target][shift_mode] = {
                        "status": "failed",
                        "error": str(exc),
                    }

    # ------------------------------------------------------------------
    # Step 4: Comparison plots.
    # ------------------------------------------------------------------
    if all_plot_results:
        plot_dir = op.join(args.results_root, "dk_combined_shift_comparison")
        plot_shifted_comparison(all_plot_results, plot_dir)

    summary_file = op.join(args.results_root, "dk_embedding_combined_run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run summary written to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
