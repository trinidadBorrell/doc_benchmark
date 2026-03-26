"""Run nested CV classification on pooled embeddings concatenated with DK markers.

This script combines per-session foundation model pooled embeddings (dimension D)
with domain-knowledge markers from baseline scalars CSV (all available markers by
default, or evoked-only subset via ``--evoked-only``) to obtain D+M features per
subject/session, then reuses the same classification pipeline as
``mlp_embedding_classifier.py`` (MLP, Random Forest, Kernel Ridge).

Output layout per foundation model:
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED/
        {feature_predicted}/nested_cv/{classifier_name}/...
"""

import argparse
import json
import os
import os.path as op

import numpy as np

try:
    from mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP
except ImportError:  # pragma: no cover
    from .mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP


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
    """Embedding classifier with marker concatenation: X = [embedding || markers]."""

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
        # data_dir already points to pre-pooled embeddings (embedding.npz),
        # so use it as the cache directory so the parent finds them via path (A).
        self.pooled_embeddings_dir = self.data_dir
        self.marker_csv = marker_csv
        self.marker_reduction = marker_reduction
        self.expected_marker_dim = expected_marker_dim
        self.evoked_only = evoked_only
        self._markers_dict = None
        self._marker_names = None

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
                markers_dict = {
                    key: vec[keep_idx] for key, vec in markers_dict.items()
                }
                print(
                    f"   Filtered to {len(marker_names)} evoked markers: "
                    f"{marker_names}",
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
        """Load pooled embeddings and append marker vectors per session key."""
        base_embeddings = super().load_embeddings(embedding_suffix=embedding_suffix)
        markers_dict, marker_names = self._load_markers_once()

        combined = {}
        n_missing_markers = 0
        n_nan_rows = 0

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

            combined[key] = np.concatenate([emb_vec, marker_vec], axis=0)

        if not combined:
            raise ValueError(
                "No combined embeddings available after intersecting pooled "
                "embeddings with marker CSV."
            )

        first_key = next(iter(combined))
        total_dim = int(combined[first_key].shape[0])
        print(
            "   Combined embeddings ready: "
            f"{len(combined)} sessions, marker_dim={len(marker_names)}, "
            f"total_dim={total_dim}, missing_markers={n_missing_markers}, "
            f"nan_rows={n_nan_rows}",
            flush=True,
        )
        return combined


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


def _run_target_for_model(model_name, pooled_dir, args, target):
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
    print(f"FM model: {model_name} | target: {target}", flush=True)
    print(f"Input pooled embeddings: {pooled_dir}", flush=True)
    print(f"Output directory: {out_base}", flush=True)
    print("=" * 80, flush=True)

    if target == "crs":
        classifier.run_full_cv(target="crs")
    else:
        classifier.run_full_cv(
            target=target,
            labels_file=args.patient_labels_full,
            binary_outcome=args.binary_outcome,
            death_binary=args.death_binary,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate pooled FM embeddings with domain-knowledge marker "
            "embeddings and run nested CV classification (MLP/RF/KR)."
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
        help="Only use evoked markers (TimeLocked, WindowDecoding, CNV, etc.) "
        "instead of all available DK markers",
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

    args = parser.parse_args()

    if args.binary_outcome and args.death_binary:
        raise ValueError("Choose at most one of --binary-outcome or --death-binary.")

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

    run_log = {
        "results_root": args.results_root,
        "pooled_subpath": args.pooled_subpath,
        "marker_csv": args.marker_csv,
        "marker_reduction": args.marker_reduction,
        "models": {},
    }

    for model_name in selected_models:
        pooled_dir = op.join(args.results_root, model_name, args.pooled_subpath)
        run_log["models"][model_name] = {}
        for target in args.feature_predicted:
            try:
                _run_target_for_model(model_name, pooled_dir, args, target)
                run_log["models"][model_name][target] = {"status": "ok"}
            except Exception as exc:  # continue to next run
                print(
                    f"Run failed for model={model_name}, target={target}: {exc}",
                    flush=True,
                )
                run_log["models"][model_name][target] = {
                    "status": "failed",
                    "error": str(exc),
                }

    summary_file = op.join(args.results_root, "dk_embedding_combined_run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run summary written to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
