"""Shared cross-validation utilities for fair multi-model comparison.

All scripts that benchmark FM embeddings alongside domain-knowledge baselines
import from here so that:
  - Nested CV fold logic is defined in one place.
  - Common session pools are computed identically everywhere.
  - Leakage and integrity checks are applied uniformly.

Typical usage
-------------
1. Build a common session pool (intersection across all embedding sources + labels):

       common_sessions = build_common_session_pool(source_loaders, labels)

2. Generate one set of nested CV folds from that pool:

       folds = generate_nested_cv_folds(
           common_sessions, labels, n_outer=5, random_state=42
       )

3. Optionally persist / reload for reproducibility:

       save_cv_splits(folds, common_sessions, labels, path)
       folds, common_sessions, labels = load_cv_splits(path)

4. Pass (common_sessions, folds) to each model's evaluation function so
   every model trains and tests on identical session subsets and fold
   assignments.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A fold dict contains numpy arrays for indices and lists for subjects.
# Inner splits are stored as lists of (train_idx, val_idx) pairs relative
# to the outer training set (i.e. indices into X[train_idx]).
FoldDict = (
    dict  # keys: train_idx, test_idx, inner_splits, train_subjects, test_subjects
)


# ---------------------------------------------------------------------------
# Leakage and integrity checks
# ---------------------------------------------------------------------------


def check_no_subject_leakage(
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label: str = "",
) -> None:
    """Raise ValueError if any subject appears in both train and test sets.

    Parameters
    ----------
    groups:
        1-D array of subject IDs aligned with the full dataset.
    train_idx, test_idx:
        Index arrays into *groups*.
    label:
        Human-readable context for the error message (e.g. "outer fold 3").
    """
    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    overlap = train_subjects & test_subjects
    if overlap:
        prefix = f"{label}: " if label else ""
        raise ValueError(
            f"{prefix}Subject leakage detected — subjects appear in both train "
            f"and test: {sorted(overlap)}"
        )


def check_subject_session_integrity(
    groups: np.ndarray,
    fold_assignments: np.ndarray,
    label: str = "",
) -> None:
    """Raise ValueError if any subject's sessions are split across folds.

    Parameters
    ----------
    groups:
        1-D array of subject IDs (one entry per session).
    fold_assignments:
        1-D integer array of the same length; fold_assignments[i] is the
        outer fold index assigned to session i.
    label:
        Human-readable context for the error message.
    """
    subject_folds: dict[str, set] = defaultdict(set)
    for session_idx, fold_id in enumerate(fold_assignments):
        subject_folds[groups[session_idx]].add(int(fold_id))

    split_subjects = {
        s: sorted(folds) for s, folds in subject_folds.items() if len(folds) > 1
    }
    if split_subjects:
        prefix = f"{label}: " if label else ""
        raise ValueError(
            f"{prefix}Subject-session integrity violated — the following subjects "
            f"have sessions in multiple folds: {split_subjects}"
        )


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def generate_nested_cv_folds(
    common_sessions: list[str],
    labels: dict[str, int],
    n_outer: int = 5,
    random_state: int = 42,
) -> list[FoldDict]:
    """Generate nested stratified group-CV folds for a common session pool.

    Outer CV: ``StratifiedGroupKFold(n_splits=n_outer, shuffle=True)``.
    Inner CV: ``StratifiedGroupKFold(n_splits=min(n_outer-1, …), shuffle=True)``
              applied to each outer training set.

    Both outer and inner folds are checked for subject leakage and for
    subject-session integrity (all sessions of a subject in the same fold).

    Parameters
    ----------
    common_sessions:
        Ordered list of session keys (e.g. ``"AA079_ses-02"``).
    labels:
        Mapping session_key → integer class label.
    n_outer:
        Desired number of outer folds.  Actual splits may be fewer if
        class sizes or group counts are limiting.
    random_state:
        Seed for both outer and inner splitters.

    Returns
    -------
    List of fold dicts, one per outer fold::

        {
            "train_idx":      np.ndarray,  # indices into common_sessions
            "test_idx":       np.ndarray,
            "train_subjects": list[str],
            "test_subjects":  list[str],
            "inner_splits":   list of (inner_train_idx, inner_val_idx)
                              # indices relative to X[train_idx]
        }
    """
    # Lazy import to avoid circular deps when this module is used standalone.
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([labels[k] for k in common_sessions])
    groups = np.array([k.split("_ses-")[0] for k in common_sessions])

    unique_groups = np.unique(groups)
    class_counts = np.bincount(y, minlength=2)

    effective_outer = int(min(n_outer, int(np.min(class_counts)), len(unique_groups)))
    if effective_outer < 2:
        raise ValueError(
            f"Cannot create ≥2 outer folds: class_counts={class_counts.tolist()}, "
            f"n_unique_subjects={len(unique_groups)}, n_outer={n_outer}"
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=effective_outer,
        shuffle=True,
        random_state=random_state,
    )

    # Build fold assignment array for integrity check.
    fold_assignment = np.full(len(common_sessions), -1, dtype=int)

    folds: list[FoldDict] = []
    X_placeholder = np.zeros((len(common_sessions), 1))

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_placeholder, y, groups=groups)
    ):
        # --- outer leakage check ---
        check_no_subject_leakage(
            groups, train_idx, test_idx, label=f"outer fold {fold_idx}"
        )
        fold_assignment[test_idx] = fold_idx

        g_train = groups[train_idx]
        y_train = y[train_idx]

        # --- inner CV setup ---
        inner_n_groups = len(np.unique(g_train))
        inner_class_counts = np.bincount(y_train, minlength=2)
        inner_n_splits = int(
            min(
                effective_outer - 1,
                inner_n_groups,
                int(np.min(inner_class_counts)),
            )
        )
        if inner_n_splits < 2:
            raise ValueError(
                f"Outer fold {fold_idx}: cannot configure inner CV with ≥2 folds. "
                f"inner_class_counts={inner_class_counts.tolist()}, "
                f"n_inner_groups={inner_n_groups}"
            )

        inner_cv = StratifiedGroupKFold(
            n_splits=inner_n_splits,
            shuffle=True,
            random_state=random_state,
        )

        X_train_placeholder = np.zeros((len(train_idx), 1))
        inner_splits = []
        for inner_fold_idx, (inner_train, inner_val) in enumerate(
            inner_cv.split(X_train_placeholder, y_train, groups=g_train)
        ):
            # Inner indices are relative to X[train_idx].
            check_no_subject_leakage(
                g_train,
                inner_train,
                inner_val,
                label=f"outer fold {fold_idx} / inner fold {inner_fold_idx}",
            )
            inner_splits.append((inner_train, inner_val))

        folds.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_subjects": sorted(set(g_train.tolist())),
                "test_subjects": sorted(set(groups[test_idx].tolist())),
                "inner_splits": inner_splits,
            }
        )

    # --- subject-session integrity check across all outer folds ---
    check_subject_session_integrity(groups, fold_assignment, label="outer CV")

    return folds


# ---------------------------------------------------------------------------
# Common session pool
# ---------------------------------------------------------------------------


def build_common_session_pool(
    source_loaders: dict[str, Callable[[], dict[str, object]]],
    labels: dict[str, int],
    verbose: bool = True,
) -> list[str]:
    """Return sessions present in ALL embedding sources AND in labels.

    Parameters
    ----------
    source_loaders:
        Mapping name → zero-argument callable that returns a dict
        ``{session_key: embedding}``.  The callable is invoked once per
        source; results are not cached here.
    labels:
        Mapping session_key → integer class label.
    verbose:
        Print per-source counts and final common count.

    Returns
    -------
    Sorted list of session keys in the intersection.
    """
    common: set[str] | None = None

    for name, loader in source_loaders.items():
        embeddings = loader()
        available = set(embeddings.keys()) & set(labels.keys())
        if verbose:
            print(
                f"  [{name}] available sessions with labels: {len(available)}",
                flush=True,
            )
        if common is None:
            common = available
        else:
            common &= available
        if verbose:
            print(
                f"  [{name}] running intersection: {len(common)}",
                flush=True,
            )

    if not common:
        raise ValueError(
            "Common session pool is empty — no sessions shared across all "
            "embedding sources and labels."
        )

    result = sorted(common)
    if verbose:
        print(
            f"  Common session pool: {len(result)} sessions "
            f"({len({k.split('_ses-')[0] for k in result})} subjects)",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _fold_to_serialisable(fold: FoldDict) -> dict:
    """Convert numpy arrays in a fold dict to plain Python lists."""
    return {
        "train_idx": fold["train_idx"].tolist(),
        "test_idx": fold["test_idx"].tolist(),
        "train_subjects": fold["train_subjects"],
        "test_subjects": fold["test_subjects"],
        "inner_splits": [[it.tolist(), iv.tolist()] for it, iv in fold["inner_splits"]],
    }


def _fold_from_serialisable(d: dict) -> FoldDict:
    """Reconstruct numpy arrays from a deserialised fold dict."""
    return {
        "train_idx": np.array(d["train_idx"], dtype=int),
        "test_idx": np.array(d["test_idx"], dtype=int),
        "train_subjects": d["train_subjects"],
        "test_subjects": d["test_subjects"],
        "inner_splits": [
            (np.array(it, dtype=int), np.array(iv, dtype=int))
            for it, iv in d["inner_splits"]
        ],
    }


def save_cv_splits(
    folds: list[FoldDict],
    common_sessions: list[str],
    labels: dict[str, int],
    path: str,
) -> None:
    """Serialise folds + session metadata to a JSON file.

    Parameters
    ----------
    folds:
        Output of :func:`generate_nested_cv_folds`.
    common_sessions:
        Ordered session list used to generate the folds.
    labels:
        Full labels dict (only common_sessions entries are stored).
    path:
        Output file path (will be created / overwritten).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "common_sessions": common_sessions,
        "labels": {k: int(labels[k]) for k in common_sessions},
        "groups": [k.split("_ses-")[0] for k in common_sessions],
        "folds": [_fold_to_serialisable(f) for f in folds],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_cv_splits(
    path: str,
) -> tuple[list[FoldDict], list[str], dict[str, int]]:
    """Load folds + session metadata from a JSON file.

    Returns
    -------
    folds:
        List of fold dicts with numpy arrays restored.
    common_sessions:
        Ordered session list.
    labels:
        Mapping session_key → integer label (for the common sessions only).
    """
    with open(path) as fh:
        payload = json.load(fh)

    folds = [_fold_from_serialisable(d) for d in payload["folds"]]
    common_sessions: list[str] = payload["common_sessions"]
    labels: dict[str, int] = {k: int(v) for k, v in payload["labels"].items()}
    return folds, common_sessions, labels
