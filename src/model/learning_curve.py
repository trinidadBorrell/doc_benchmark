"""Learning curve benchmark for CRS prediction from embeddings.

This script runs a CRS-only nested cross-validation benchmark across multiple
embedding sources and classifiers. For each embedding type, it evaluates four
classifiers (SVM, MLP, Kernel Ridge, Random Forest) on progressively larger
subsets defined by a target number of sessions.

Key design principle
--------------------
All embedding types (foundation models + domain-knowledge baseline) are
evaluated on the **exact same** session subsets and **identical** nested CV
fold assignments at every budget point.

Steps performed in main():
  1. Load all embeddings for every source.
  2. Compute the **common session pool**: sessions present in every source
     AND in the CRS labels file.
  3. For each session budget:
       a. Subsample from the common pool once (class-balanced, subject-level)
          using seed = random_state + budget.
       b. Generate one set of nested CV folds from that subsample
          (StratifiedGroupKFold outer, StratifiedGroupKFold inner).
       c. Save the subsample + folds to
          LEARNING_CURVE/budget_{N}/common_cv_splits.json.
  4. Each parallel job receives the pre-computed (sessions, folds) dict and
     evaluates its own embedding type on those identical splits.

Outputs are saved under:
    /data/project/eeg_foundation/data/benchmark_results/new_results/LEARNING_CURVE

Directory layout:
    LEARNING_CURVE/budget_{N}/common_cv_splits.json  (shared splits)
    LEARNING_CURVE/{embedding_type}/{classifier_name}/
        - fold_metrics.csv
        - summary_metrics.csv
        - summary_metrics.json

And in LEARNING_CURVE root:
    - auc_learning_curve_svm.png
    - auc_learning_curve_mlp.png
    - auc_learning_curve_kernel_ridge.png
    - auc_learning_curve_random_forest.png
"""

import argparse
import json
import os
import os.path as op
from datetime import datetime
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .kernel_ridge_classifier import KernelRidgeClassifier
    from .cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
    )
except ImportError:
    from kernel_ridge_classifier import KernelRidgeClassifier
    from cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
    )


DEFAULT_RESULTS_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
DEFAULT_OUTPUT_DIR = op.join(DEFAULT_RESULTS_ROOT, "LEARNING_CURVE")
DEFAULT_PATIENT_LABELS = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)
DEFAULT_MARKER_CSV = (
    "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
)

DEFAULT_EMBEDDING_MODELS = ["CBraMod", "NeuroLM", "TOTEM", "LaBram"]
#SESSION_BUDGETS = [25, 40, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]
SESSION_BUDGETS = np.arange(20, 200, 5)

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}


def _safe_auc(y_true, y_score):
    """Return AUC or None if a fold has a single class."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def load_crs_labels(labels_file):
    """Load CRS labels as binary classes: VS=0, MCS=1."""
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

        try:
            subject = str(row["subject"])
            session = f"ses-{int(row['session']):02d}"
        except Exception:
            continue

        labels[f"{subject}_{session}"] = label

    if not labels:
        raise ValueError(f"No valid CRS labels found in {labels_file}")

    return labels


def load_pooled_embeddings(pooled_dir):
    """Load pooled embeddings from sub-*/ses-*/embedding.npz or embedding.npy."""
    if not op.isdir(pooled_dir):
        raise FileNotFoundError(f"Embedding directory not found: {pooled_dir}")

    embeddings = {}
    for subject_dir in sorted(os.listdir(pooled_dir)):
        if not subject_dir.startswith("sub-"):
            continue
        subject_id = subject_dir.replace("sub-", "")
        subject_path = op.join(pooled_dir, subject_dir)
        if not op.isdir(subject_path):
            continue

        for session_dir in sorted(os.listdir(subject_path)):
            if not session_dir.startswith("ses-"):
                continue
            session_path = op.join(subject_path, session_dir)
            if not op.isdir(session_path):
                continue

            emb = None
            npz_path = op.join(session_path, "embedding.npz")
            npy_path = op.join(session_path, "embedding.npy")
            if op.isfile(npz_path):
                npz = np.load(npz_path)
                emb_key = "embedding" if "embedding" in npz else npz.files[0]
                emb = np.asarray(npz[emb_key]).reshape(-1)
            elif op.isfile(npy_path):
                emb = np.asarray(np.load(npy_path)).reshape(-1)

            if emb is None:
                continue

            key = f"{subject_id}_{session_dir}"
            embeddings[key] = emb

    if not embeddings:
        raise ValueError(f"No embeddings found in {pooled_dir}")
    return embeddings


def load_domain_knowledge_embeddings(marker_csv, reduction_letter="A"):
    """Load domain-knowledge marker vectors keyed by subject_session."""
    if reduction_letter not in REDUCTION_MAP:
        raise ValueError(f"Unknown reduction letter: {reduction_letter}")

    reduction = REDUCTION_MAP[reduction_letter]

    # Support both comma- and semicolon-separated files.
    try:
        df = pd.read_csv(marker_csv, sep=",")
        if "Reduction" not in df.columns:
            raise ValueError("Missing Reduction column with comma separator")
    except Exception:
        df = pd.read_csv(marker_csv, sep=";")

    if "Reduction" not in df.columns or "Subject" not in df.columns:
        raise ValueError(f"Marker CSV is missing required columns: {marker_csv}")

    df = df[df["Reduction"] == reduction].copy()
    if df.empty:
        raise ValueError(f"No marker rows found for reduction {reduction}")

    meta_cols = {"Subject", "Reduction", "Label"}
    marker_cols = [
        col
        for col in df.columns
        if col not in meta_cols
        and not str(col).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not marker_cols:
        raise ValueError("No numeric marker columns found in marker CSV")

    embeddings = {}
    for _, row in df.iterrows():
        raw_subject = str(row["Subject"])
        parts = raw_subject.rsplit("_", 1)
        if len(parts) != 2:
            continue
        subject_id, session_num = parts
        try:
            key = f"{subject_id}_ses-{int(session_num):02d}"
        except ValueError:
            continue

        vec = row[marker_cols].values.astype(float)
        if np.isnan(vec).any():
            continue
        embeddings[key] = vec

    if not embeddings:
        raise ValueError("No valid domain-knowledge embeddings found after filtering")

    return embeddings


def build_embedding_source_specs(results_root, marker_csv, reduction_letter):
    """Create lightweight source specs to avoid large object pickling."""
    specs = {}

    for model_name in DEFAULT_EMBEDDING_MODELS:
        pooled_dir = op.join(
            results_root,
            model_name,
            "doc_patients",
            "MLP_EMBEDDING",
            "pooled_embeddings",
        )
        if op.isdir(pooled_dir):
            specs[model_name] = {"kind": "foundation", "path": pooled_dir}

    specs["Domain_Knowledge"] = {
        "kind": "domain_knowledge",
        "marker_csv": marker_csv,
        "reduction_letter": reduction_letter,
    }
    return specs


def load_embeddings_from_spec(spec):
    """Load embeddings dict based on source spec."""
    kind = spec["kind"]
    if kind == "foundation":
        return load_pooled_embeddings(spec["path"])
    if kind == "domain_knowledge":
        return load_domain_knowledge_embeddings(
            marker_csv=spec["marker_csv"],
            reduction_letter=spec["reduction_letter"],
        )
    raise ValueError(f"Unknown source kind: {kind}")


def build_dataset_for_sessions(embeddings_dict, labels_dict, session_keys):
    """Build X, y, groups arrays for an explicit ordered list of session keys.

    Parameters
    ----------
    embeddings_dict:
        Mapping session_key → embedding vector.
    labels_dict:
        Mapping session_key → integer label.
    session_keys:
        Ordered list of session keys (defines row order in X).

    Returns
    -------
    X : ndarray, shape (n_sessions, n_features)
    y : ndarray, shape (n_sessions,)
    groups : ndarray of subject IDs, shape (n_sessions,)
    """
    missing = [k for k in session_keys if k not in embeddings_dict]
    if missing:
        raise ValueError(
            f"{len(missing)} sessions from the common pool are missing in this "
            f"embedding source: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    vecs = [np.asarray(embeddings_dict[k]).reshape(-1) for k in session_keys]
    first_shape = vecs[0].shape
    bad = [k for k, v in zip(session_keys, vecs) if v.shape != first_shape]
    if bad:
        raise ValueError(
            f"Inconsistent embedding shapes for {len(bad)} sessions: {bad[:3]}"
        )

    X = np.array(vecs)
    y = np.array([int(labels_dict[k]) for k in session_keys])
    groups = np.array([k.split("_ses-")[0] for k in session_keys])
    return X, y, groups


def sample_sessions_balanced(session_keys, labels, n_sessions, random_state):
    """Sample an approximately half/half class-balanced session subset.

    For budget N, this selects floor(N/2) sessions from class 0 (VS/UWS) and
    the remaining sessions from class 1 (MCS). Selection is random but
    reproducible via random_state.
    """
    if n_sessions <= 0:
        raise ValueError(f"n_sessions must be > 0, got {n_sessions}")

    rng = np.random.RandomState(random_state)
    class0 = [k for k in session_keys if int(labels[k]) == 0]
    class1 = [k for k in session_keys if int(labels[k]) == 1]

    n_class0 = n_sessions // 2
    n_class1 = n_sessions - n_class0

    if len(class0) < n_class0 or len(class1) < n_class1:
        raise ValueError(
            "Cannot sample class-balanced subset for budget "
            f"{n_sessions}: available class0={len(class0)}, class1={len(class1)}, "
            f"requested class0={n_class0}, class1={n_class1}."
        )

    chosen0 = list(rng.choice(class0, size=n_class0, replace=False))
    chosen1 = list(rng.choice(class1, size=n_class1, replace=False))
    selected = chosen0 + chosen1
    rng.shuffle(selected)
    return selected


def get_classifier_and_grid(classifier_name, random_state):
    """Return estimator and parameter grid for a classifier."""
    if classifier_name == "svm":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_grid = {
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", 0.01, 0.1],
            "svm__kernel": ["rbf"],
        }
    elif classifier_name == "mlp":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=500,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 1e-2],
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
    elif classifier_name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3],
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return estimator, param_grid


def predict_scores(estimator, X_test, classifier_name):
    """Return hard predictions and probability-like scores for AUC."""
    y_pred = estimator.predict(X_test)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
        if proba.ndim == 2:
            y_score = proba[:, 1]
        else:
            y_score = np.asarray(proba).reshape(-1)
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


def run_nested_cv_single_setting(
    X,
    y,
    groups,
    classifier_name,
    random_state,
    cv_folds,
):
    """Run nested CV with pre-computed folds and return fold-level metric rows.

    Parameters
    ----------
    X, y, groups:
        Feature matrix, labels, and subject groups for the budget subset.
        Row order must match the session ordering used when generating cv_folds.
    classifier_name:
        One of ``svm``, ``mlp``, ``kernel_ridge``, ``random_forest``.
    random_state:
        Seed passed to the classifier.
    cv_folds:
        List of fold dicts as returned by
        :func:`cv_utils.generate_nested_cv_folds`.
    """
    rows = []
    estimator, param_grid = get_classifier_and_grid(classifier_name, random_state)

    for fold_idx, fold in enumerate(cv_folds, start=1):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold["inner_splits"]

        # Leakage guard — should never trigger given cv_utils checks, but kept
        # as a runtime safety net.
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        overlap = train_subjects & test_subjects
        if overlap:
            raise ValueError(
                f"Fold {fold_idx}: subject leakage detected: {sorted(overlap)}"
            )

        X_train = X[train_idx]
        y_train = y[train_idx]
        g_train = groups[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_splits,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred, y_score = predict_scores(best_estimator, X_test, classifier_name)

        auc = _safe_auc(y_test, y_score)
        row = {
            "fold": fold_idx,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "auc": auc,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "n_train_sessions": int(len(train_idx)),
            "n_test_sessions": int(len(test_idx)),
            "n_train_subjects": int(len(np.unique(g_train))),
            "n_test_subjects": int(len(np.unique(groups[test_idx]))),
            "best_params": grid.best_params_,
        }
        rows.append(row)

    return rows


def summarize_fold_rows(rows, n_sessions_requested, n_subjects_real, n_sessions_real):
    """Aggregate fold rows into mean/std summary."""
    auc_vals = [r["auc"] for r in rows if r["auc"] is not None]
    bal_vals = [r["balanced_accuracy"] for r in rows]
    prec_vals = [r["precision"] for r in rows]
    rec_vals = [r["recall"] for r in rows]

    summary = {
        "n_sessions_requested": int(n_sessions_requested),
        "n_subjects_real": int(n_subjects_real),
        "n_sessions_real": int(n_sessions_real),
        "n_folds": int(len(rows)),
        "balanced_accuracy_mean": float(np.mean(bal_vals)),
        "balanced_accuracy_std": float(np.std(bal_vals)),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
        "auc_std": float(np.std(auc_vals)) if auc_vals else None,
        "precision_mean": float(np.mean(prec_vals)),
        "precision_std": float(np.std(prec_vals)),
        "recall_mean": float(np.mean(rec_vals)),
        "recall_std": float(np.std(rec_vals)),
    }
    return summary


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _now():
    return datetime.now().strftime("%H:%M:%S")


def run_embedding_job(
    embedding_name,
    source_spec,
    labels,
    budget_splits,
    output_dir,
    random_state,
):
    """Run all classifiers for one embedding type using pre-computed splits.

    Parameters
    ----------
    embedding_name:
        Human-readable name for this embedding source (used in output paths
        and log messages).
    source_spec:
        Dict describing where to load embeddings from (passed to
        :func:`load_embeddings_from_spec`).
    labels:
        Full CRS labels dict (session_key → int).
    budget_splits:
        Dict mapping budget (int) → ``{"sessions": [...], "folds": [...],
        "n_subjects": int, "n_sessions": int}``.  Generated in ``main()``
        from the common session pool; identical for every embedding type.
    output_dir:
        Root output directory (LEARNING_CURVE/).
    random_state:
        Seed passed to classifiers.
    """
    print(f"[{_now()}] [{embedding_name}] Loading embeddings ...", flush=True)
    embeddings = load_embeddings_from_spec(source_spec)

    classifiers = ["svm", "mlp", "kernel_ridge", "random_forest"]
    embedding_auc_curve = {clf: [] for clf in classifiers}

    clf_fold_rows = {clf: [] for clf in classifiers}
    clf_summary_rows = {clf: [] for clf in classifiers}

    for clf in classifiers:
        ensure_dir(op.join(output_dir, embedding_name, clf))

    for budget in sorted(budget_splits.keys()):
        bdata = budget_splits[budget]
        budget_sessions = bdata["sessions"]
        cv_folds = bdata["folds"]
        n_subjects_real = bdata["n_subjects"]
        n_sessions_real = bdata["n_sessions"]

        # Build this model's feature matrix for the common budget sessions.
        X_sub, y_sub, g_sub = build_dataset_for_sessions(
            embeddings, labels, budget_sessions
        )
        print(
            f"[{_now()}] [{embedding_name}] budget={budget}: "
            f"sessions={n_sessions_real}, features={X_sub.shape[1]}, "
            f"subjects={n_subjects_real}",
            flush=True,
        )

        for classifier_name in classifiers:
            print(
                f"[{_now()}] [{embedding_name}] {classifier_name} | sessions={budget}",
                flush=True,
            )
            t0 = time.time()

            fold_rows = run_nested_cv_single_setting(
                X=X_sub,
                y=y_sub,
                groups=g_sub,
                classifier_name=classifier_name,
                random_state=random_state,
                cv_folds=cv_folds,
            )

            for row in fold_rows:
                row["n_sessions_requested"] = int(budget)
                row["n_subjects_real"] = int(n_subjects_real)
                row["n_sessions_real"] = int(n_sessions_real)
                row["embedding_type"] = embedding_name
                row["classifier"] = classifier_name
                row["best_params"] = json.dumps(row["best_params"], sort_keys=True)

            clf_fold_rows[classifier_name].extend(fold_rows)

            summary = summarize_fold_rows(
                rows=fold_rows,
                n_sessions_requested=budget,
                n_subjects_real=n_subjects_real,
                n_sessions_real=n_sessions_real,
            )
            summary["embedding_type"] = embedding_name
            summary["classifier"] = classifier_name
            clf_summary_rows[classifier_name].append(summary)

            # Incremental checkpoint writes.
            classifier_dir = op.join(output_dir, embedding_name, classifier_name)
            pd.DataFrame(clf_fold_rows[classifier_name]).to_csv(
                op.join(classifier_dir, "fold_metrics.csv"), index=False
            )
            pd.DataFrame(clf_summary_rows[classifier_name]).to_csv(
                op.join(classifier_dir, "summary_metrics.csv"), index=False
            )
            with open(op.join(classifier_dir, "summary_metrics.json"), "w") as f:
                json.dump(clf_summary_rows[classifier_name], f, indent=2)

            elapsed = time.time() - t0
            auc_msg = (
                f"{summary['auc_mean']:.3f}"
                if summary["auc_mean"] is not None
                else "N/A"
            )
            print(
                f"[{_now()}] [{embedding_name}] {classifier_name} | sessions={budget} "
                f"DONE in {elapsed:.1f}s "
                f"(bal_acc={summary['balanced_accuracy_mean']:.3f}, auc={auc_msg})",
                flush=True,
            )

    # Final writes and collect results.
    for classifier_name in classifiers:
        classifier_dir = op.join(output_dir, embedding_name, classifier_name)
        pd.DataFrame(clf_fold_rows[classifier_name]).to_csv(
            op.join(classifier_dir, "fold_metrics.csv"), index=False
        )
        pd.DataFrame(clf_summary_rows[classifier_name]).to_csv(
            op.join(classifier_dir, "summary_metrics.csv"), index=False
        )
        with open(op.join(classifier_dir, "summary_metrics.json"), "w") as f:
            json.dump(clf_summary_rows[classifier_name], f, indent=2)

        embedding_auc_curve[classifier_name] = clf_summary_rows[classifier_name]

    return embedding_name, embedding_auc_curve


def plot_auc_curves(all_results, output_dir):
    """Plot AUC learning curves by classifier with one curve per embedding type."""
    classifiers = ["svm", "mlp", "kernel_ridge", "random_forest"]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 120

    for classifier_name in classifiers:
        fig, ax = plt.subplots(figsize=(9, 6))
        all_xticks = set()

        for embedding_name in sorted(all_results):
            summary_rows = all_results[embedding_name].get(classifier_name, [])
            if not summary_rows:
                continue

            summary_rows = sorted(summary_rows, key=lambda r: r["n_subjects_real"])
            x = [r["n_subjects_real"] for r in summary_rows]
            y = [r["auc_mean"] for r in summary_rows]
            yerr = [
                r["auc_std"] if r["auc_std"] is not None else 0.0 for r in summary_rows
            ]

            valid = [(xi, yi, ei) for xi, yi, ei in zip(x, y, yerr) if yi is not None]
            if not valid:
                continue
            x_valid = [v[0] for v in valid]
            y_valid = [v[1] for v in valid]
            e_valid = [v[2] for v in valid]
            all_xticks.update(x_valid)

            ax.plot(x_valid, y_valid, marker="o", linewidth=2, label=embedding_name)
            ax.fill_between(
                x_valid,
                np.array(y_valid) - np.array(e_valid),
                np.array(y_valid) + np.array(e_valid),
                alpha=0.15,
            )

        ax.set_title(f"AUC Learning Curve - {classifier_name}")
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("AUC")
        if all_xticks:
            ax.set_xticks(sorted(all_xticks))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        out_file = op.join(output_dir, f"auc_learning_curve_{classifier_name}.png")
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested-CV learning curves for CRS prediction across embeddings"
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels-file", default=DEFAULT_PATIENT_LABELS)
    parser.add_argument("--marker-csv", default=DEFAULT_MARKER_CSV)
    parser.add_argument(
        "--marker-reduction",
        default="A",
        choices=["A", "B", "C", "D"],
        help="Reduction used for Domain_Knowledge markers",
    )
    parser.add_argument(
        "--session-budgets",
        nargs="+",
        type=int,
        default=SESSION_BUDGETS,
        help="Target session counts for learning curve subsets",
    )
    parser.add_argument(
        "--n-outer-folds",
        type=int,
        default=5,
        help="Number of outer folds for StratifiedGroupKFold",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs across embedding types (default: number of embeddings)",
    )
    parser.add_argument(
        "--parallel-backend",
        default="threading",
        choices=["threading", "loky"],
        help="Joblib backend. threading avoids large pickle overhead.",
    )
    parser.add_argument(
        "--parallel-verbose",
        type=int,
        default=10,
        help="Joblib verbosity level for progress messages",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("=" * 80, flush=True)
    print("LEARNING CURVE BENCHMARK (CRS only)", flush=True)
    print("=" * 80, flush=True)
    print(f"Results root: {args.results_root}", flush=True)
    print(f"Output dir:   {args.output_dir}", flush=True)

    labels = load_crs_labels(args.labels_file)
    print(f"Loaded CRS labels: {len(labels)} sessions", flush=True)

    embedding_source_specs = build_embedding_source_specs(
        results_root=args.results_root,
        marker_csv=args.marker_csv,
        reduction_letter=args.marker_reduction,
    )

    if not embedding_source_specs:
        raise ValueError("No embedding sources available")

    source_names = sorted(embedding_source_specs.keys())
    print(f"Embedding types: {source_names}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Build the common session pool (intersection across ALL sources
    # and the labels file).  Every model will be evaluated on this pool.
    # ------------------------------------------------------------------
    print("\nBuilding common session pool ...", flush=True)
    source_loaders = {
        name: (lambda spec=spec: load_embeddings_from_spec(spec))
        for name, spec in embedding_source_specs.items()
    }
    common_sessions = build_common_session_pool(source_loaders, labels, verbose=True)
    print(
        f"Common pool: {len(common_sessions)} sessions, "
        f"{len({k.split('_ses-')[0] for k in common_sessions})} subjects\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2: For each budget, subsample the common pool once and generate
    # nested CV folds.  The resulting (sessions, folds) dict is shared by
    # all embedding types.
    # ------------------------------------------------------------------
    session_budgets = sorted(set(args.session_budgets))
    budget_splits: dict[int, dict] = {}

    print("Pre-computing per-budget common splits ...", flush=True)
    for budget in session_budgets:
        if budget > len(common_sessions):
            print(
                f"  budget={budget}: skipped (common pool has only "
                f"{len(common_sessions)} sessions)",
                flush=True,
            )
            continue

        # (a) Subsample from the common pool — identical seed for every model.
        budget_sessions = sample_sessions_balanced(
            session_keys=common_sessions,
            labels=labels,
            n_sessions=budget,
            random_state=args.random_state + budget,
        )
        n_subjects = len({k.split("_ses-")[0] for k in budget_sessions})
        n_sessions = len(budget_sessions)

        # (b) Generate nested CV folds from this subsample.
        budget_folds = generate_nested_cv_folds(
            common_sessions=budget_sessions,
            labels=labels,
            n_outer=args.n_outer_folds,
            random_state=args.random_state,
        )

        budget_splits[budget] = {
            "sessions": budget_sessions,
            "folds": budget_folds,
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
        }

        # (c) Persist for reproducibility.
        splits_dir = op.join(args.output_dir, f"budget_{budget}")
        ensure_dir(splits_dir)
        save_cv_splits(
            folds=budget_folds,
            common_sessions=budget_sessions,
            labels=labels,
            path=op.join(splits_dir, "common_cv_splits.json"),
        )
        print(
            f"  budget={budget}: {n_sessions} sessions, {n_subjects} subjects, "
            f"{len(budget_folds)} outer folds — saved to {splits_dir}/",
            flush=True,
        )

    if not budget_splits:
        raise ValueError(
            "No valid budgets after filtering against the common pool size "
            f"({len(common_sessions)} sessions)."
        )

    # ------------------------------------------------------------------
    # Step 3: Evaluate each embedding type in parallel using the shared splits.
    # ------------------------------------------------------------------
    n_jobs = args.n_jobs if args.n_jobs is not None else len(source_names)
    n_jobs = max(1, min(n_jobs, len(source_names)))
    print(f"\nParallel jobs: {n_jobs}", flush=True)

    run_fn = delayed(run_embedding_job)
    job_outputs = Parallel(
        n_jobs=n_jobs,
        backend=args.parallel_backend,
        verbose=args.parallel_verbose,
    )(
        run_fn(
            embedding_name=name,
            source_spec=embedding_source_specs[name],
            labels=labels,
            budget_splits=budget_splits,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
        for name in source_names
    )

    all_results = {name: curve_dict for name, curve_dict in job_outputs}

    plot_auc_curves(
        all_results=all_results,
        output_dir=args.output_dir,
    )

    global_json = op.join(args.output_dir, "learning_curve_overview.json")
    with open(global_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("Finished learning curve benchmark.", flush=True)
    print(f"Outputs saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
