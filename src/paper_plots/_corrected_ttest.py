from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as _stats


# ── Test ─────────────────────────────────────────────────────────────────────


def corrected_resampled_ttest(
    diffs: list[float] | np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[float, float]:
    """
    Parameters
    ----------
    diffs : array-like
        Per-fold differences (e.g. ``aucs_a[i] - aucs_b[i]``).
    n_train, n_test : int
        Per-fold training- and test-set sizes. The correction term is
        ``n_test / n_train``; only the ratio matters.

    Returns
    -------
    (t_stat, p_two_sided)
        ``(np.nan, np.nan)`` if fewer than 2 paired samples or zero variance.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2 or n_train <= 0 or n_test <= 0:
        return np.nan, np.nan
    mean_d = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))
    if var_d <= 0:
        return np.nan, np.nan
    corrected_var = var_d * (1.0 / n + n_test / n_train)
    t_stat = mean_d / np.sqrt(corrected_var)
    p_two = 2.0 * float(_stats.t.sf(abs(t_stat), df=n - 1))
    return float(t_stat), p_two


def corrected_one_tailed(
    diffs: list[float] | np.ndarray,
    n_train: int,
    n_test: int,
    h1: str = "a>b",
) -> tuple[float, float]:
    """One-tailed variant. ``h1='a>b'`` tests mean(diffs) > 0."""
    t, p_two = corrected_resampled_ttest(diffs, n_train, n_test)
    if np.isnan(t):
        return np.nan, np.nan
    if h1 == "a>b":
        p = p_two / 2 if t > 0 else 1.0 - p_two / 2
    else:
        p = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p)


# ── Per-fold pairing ─────────────────────────────────────────────────────────


def _per_fold_dict(path: Path) -> dict[tuple[int, int], float] | None:
    """Read ``classification_results.json`` and return a dict keyed by
    ``(repeat, fold_within_repeat)`` → ``auc_score``.

    Some files store ``fold`` as a global running index across repeats and
    others as a within-repeat index. We normalise by ``fold % n_folds`` where
    ``n_folds`` is read from the JSON's top-level field; if missing, we infer
    it as ``max(fold) - min(fold) + 1`` per repeat (best effort).
    """
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    pf = d.get("macro_average", {}).get("per_fold_metrics")
    if not pf:
        return None
    n_folds = d.get("n_folds")
    if not n_folds:
        seen = {}
        for entry in pf:
            r = entry.get("repeat", 0)
            seen.setdefault(r, set()).add(entry.get("fold", 0))
        n_folds = max((len(v) for v in seen.values()), default=1) or 1
    out: dict[tuple[int, int], float] = {}
    for entry in pf:
        auc = entry.get("auc_score")
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        r = int(entry.get("repeat", 0))
        f = int(entry.get("fold", 0)) % int(n_folds)
        out[(r, f)] = float(auc)
    return out or None


def paired_per_fold_aucs(
    path_a: Path,
    path_b: Path,
) -> tuple[list[float], list[float]] | None:
    """Return ``(aucs_a, aucs_b)`` paired by ``(repeat, fold)``.

    Both files are normalised to within-repeat fold indexing so files using
    flat fold counters and within-repeat fold counters can still be paired.
    Returns ``None`` if either file is missing or has no overlap of keys.
    """
    da = _per_fold_dict(path_a)
    db = _per_fold_dict(path_b)
    if da is None or db is None:
        return None
    common = sorted(set(da.keys()) & set(db.keys()))
    if not common:
        return None
    return [da[k] for k in common], [db[k] for k in common]


# ── n_train / n_test from manifest ───────────────────────────────────────────


def load_n_train_test_from_manifest(manifest_path: Path) -> tuple[int, int] | None:
    """Read median per-fold ``(n_train, n_test)`` from a ``cv_split_manifest``."""
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    n_train_lst, n_test_lst = [], []
    for rep in d.get("repeats", []):
        for fold in rep.get("folds", []):
            ti = fold.get("train_idx")
            te = fold.get("test_idx")
            if ti is not None and te is not None:
                n_train_lst.append(len(ti))
                n_test_lst.append(len(te))
    if not n_train_lst:
        return None
    return int(np.median(n_train_lst)), int(np.median(n_test_lst))


def infer_n_train_test(
    classification_results_path: Path,
    manifest_path: Path | None = None,
    fallback_total: int | None = None,
) -> tuple[int, int]:
    """Best-effort ``(n_train, n_test)`` for the corrected test correction.

    Resolution order:

    1. ``manifest_path`` (if provided and exists) — median per-fold sizes.
    2. ``n_folds`` from the classification JSON + ``fallback_total`` — use
       ``n_test = total / k``, ``n_train = total - n_test``.
    3. ``n_folds`` only — return ``(k-1, 1)`` so the ratio
       ``n_test/n_train = 1/(k-1)`` is correct (only the ratio enters the
       formula).
    """
    if manifest_path is not None:
        nt = load_n_train_test_from_manifest(manifest_path)
        if nt is not None:
            return nt

    n_folds = 5
    if classification_results_path.is_file():
        try:
            with classification_results_path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            n_folds = int(d.get("n_folds") or 5)
        except (OSError, json.JSONDecodeError):
            pass

    if fallback_total is not None and fallback_total > n_folds:
        n_test = max(1, fallback_total // n_folds)
        n_train = max(1, fallback_total - n_test)
        return n_train, n_test

    return n_folds - 1, 1
