#!/usr/bin/env python3
"""
Inter-Model Results Aggregation Script

Aggregates decoder results and/or re-runs MLP embedding classification
across multiple foundation models (CBraMod, LaBram, NeuroLM).

Subject filtering pipeline:
  1. For each model, collect the set of subjects present in its
     final_layer_embeddings directory (sub-{ID} folders).
  2. Take the intersection of those per-model sets → "common subjects".
  3. Optionally further restrict to a reference directory when
     --unique-subjects-values is passed.
  4. The resulting common subject set is used for BOTH the decoder
     aggregation and the MLP embedding classification.
  5. Any subject that fails or is missing at any stage is reported
     explicitly (subject ID + model + reason).

Modes:
  --decoder-only          Re-aggregate existing decoder results only
  --mlp-embedding-only    Re-run MLP embedding classification only
  (default)               Run both: decoder + MLP

Data-leakage sanity check:
  Before training the MLP a dry-run of the split assignments is
  performed (GroupShuffleSplit).  Subject IDs are verified to be
  disjoint across train / val / test partitions.  Any leakage is
  reported immediately with the offending subject IDs.

Output structure:
  {output_root}/DECODER/intermodel-{ts}/{Model}/data/...
  {output_root}/DECODER/intermodel-{ts}/{Model}/plots/...
  {output_root}/MLP-CLASSIFIER/intermodel-{ts}/{Model}/...

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Prevent MKL/OpenBLAS segfaults when running as a subprocess
try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_SUBJECT_DIR = (
    "/data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata"
)
DEFAULT_RESULTS_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
DEFAULT_METADATA_DIR = "/data/project/eeg_foundation/data/metadata"
MODELS = ["CBraMod", "LaBram", "NeuroLM"]

# Embedding directory mapping per model
EMBEDDING_DIR_MAP = {
    "CBraMod": "/data/project/eeg_foundation/data/CbraMod/final_layer_embeddings",
    "LaBram": "/data/project/eeg_foundation/data/LaBram/results_DoC_lg/final_layer_embeddings",
    "NeuroLM": "/data/project/eeg_foundation/data/NeuroLM-output/final_layer_embeddings",
}


# ======================================================================
# Utility helpers (mirrored from decoder.py)
# ======================================================================


def _truncate_to_common_length(timeseries_list):
    """Truncate list of 1-D arrays to the minimum common length."""
    if not timeseries_list:
        return np.array([])
    lengths = [len(ts) for ts in timeseries_list]
    min_len = min(lengths)
    if min_len != max(lengths):
        print(
            f"  [truncation] Timeseries lengths vary "
            f"({min(lengths)}-{max(lengths)}), truncating to {min_len}"
        )
    return np.array([ts[:min_len] for ts in timeseries_list])


def add_stimulus_lines(ax, times):
    """Add vertical lines at stimulus onsets."""
    stimulus_times = [0.0, 0.15, 0.3, 0.45, 0.6]
    for i, st in enumerate(stimulus_times):
        if times[0] <= st <= times[-1]:
            label = "Stimuli" if i == 0 else None
            ax.axvline(
                st,
                color="darkgray",
                linestyle="--",
                alpha=1,
                linewidth=1.5,
                label=label,
            )


def run_permutation_test(all_scores, n_permutations=1000, seed=42):
    """Permutation test: mean AUC vs chance (0.5)."""
    np.random.seed(seed)
    observed_mean = np.mean(all_scores)
    mean_auc_per_subject = np.mean(all_scores, axis=1)
    t_statistic, p_value_ttest = stats.ttest_1samp(
        mean_auc_per_subject, 0.5, alternative="greater"
    )
    null_means = []
    centered_scores = all_scores - mean_auc_per_subject[:, np.newaxis] + 0.5
    for _ in range(n_permutations):
        boot_idx = np.random.choice(
            len(centered_scores), size=len(centered_scores), replace=True
        )
        null_means.append(np.mean(centered_scores[boot_idx]))
    null_means = np.array(null_means)
    p_value_perm = float(np.mean(null_means >= observed_mean))
    cohens_d = (
        (observed_mean - 0.5) / np.std(mean_auc_per_subject)
        if np.std(mean_auc_per_subject) > 0
        else 0
    )
    return {
        "observed_mean": float(observed_mean),
        "p_value_permutation": p_value_perm,
        "p_value_ttest": float(p_value_ttest),
        "t_statistic": float(t_statistic),
        "cohens_d": float(cohens_d),
        "null_mean": float(np.mean(null_means)),
        "null_std": float(np.std(null_means)),
        "n_permutations": n_permutations,
        "n_subjects": int(all_scores.shape[0]),
        "significant_permutation": bool(p_value_perm < 0.05),
        "significant_ttest": bool(p_value_ttest < 0.05),
    }


def run_wilcoxon_per_timepoint(all_scores, alternative="greater"):
    """Wilcoxon signed-rank test per timepoint vs 0.5."""
    n_subjects, n_timepoints = all_scores.shape
    p_values = np.zeros(n_timepoints)
    test_statistics = np.zeros(n_timepoints)
    for t in range(n_timepoints):
        col = all_scores[:, t]
        if np.all(col == col[0]):
            p_values[t] = 1.0
            test_statistics[t] = 0.0
        else:
            try:
                stat, pval = stats.wilcoxon(col - 0.5, alternative=alternative)
                p_values[t] = pval
                test_statistics[t] = stat
            except Exception:
                p_values[t] = 1.0
                test_statistics[t] = 0.0
    return {
        "p_values": p_values,
        "test_statistics": test_statistics,
        "significant_mask": p_values < 0.05,
        "n_subjects": n_subjects,
        "n_timepoints": n_timepoints,
    }


# ======================================================================
# Subject list helpers
# ======================================================================


def get_reference_subjects(subject_dir: str) -> set:
    """Return set of subject IDs (e.g. 'AA048') from *subject_dir*.

    Expects folders named ``sub-{ID}``.
    """
    subject_dir = Path(subject_dir)
    if not subject_dir.exists():
        print(f"WARNING: Subject reference directory does not exist: {subject_dir}")
        return set()
    subjects = set()
    for entry in subject_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("sub-"):
            subjects.add(entry.name.replace("sub-", ""))
    print(f"Reference subjects loaded: {len(subjects)} from {subject_dir}")
    return subjects


# ======================================================================
# DECODER aggregation (restricted subjects)
# ======================================================================


def aggregate_decoder_for_model(
    model: str,
    results_root: str,
    allowed_subjects: set,
    patient_labels_path: str,
    output_dir: Path,
):
    """Load individual decoder results for *model*, keep only
    subjects in *allowed_subjects*, aggregate, save data & plots.

    Returns True on success, False on skip/failure.
    """
    decoder_base = Path(results_root) / model / "doc_patients" / "DECODER"
    if not decoder_base.exists():
        print(
            f"  WARNING [{model}]: DECODER directory not found at "
            f"{decoder_base} – skipping."
        )
        return False

    # ------------------------------------------------------------------
    # 1. Load individual results filtering by allowed subjects
    # ------------------------------------------------------------------
    all_results = []
    subjects_sessions = []
    times = None

    for sub_dir in sorted(decoder_base.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name.replace("sub-", "")
        if subject_id not in allowed_subjects:
            continue

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            session_id = ses_dir.name.replace("ses-", "")

            results_path = ses_dir / "data" / "decoding_results.pkl"
            if not results_path.exists():
                continue
            try:
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if times is None:
                    times_path = ses_dir / "data" / "times.npy"
                    if times_path.exists():
                        times = np.load(times_path)
                    elif (
                        "overall" in results
                        and "mean_scores_time" in results["overall"]
                    ):
                        n_t = len(results["overall"]["mean_scores_time"])
                        times = np.linspace(-0.2, 0.8, n_t)
                all_results.append(results)
                subjects_sessions.append((subject_id, session_id))
            except Exception as e:
                print(
                    f"  WARNING [{model}]: Error loading "
                    f"sub-{subject_id}/ses-{session_id}: {e}"
                )

    if not all_results:
        print(
            f"  WARNING [{model}]: No matching decoder results found "
            f"for the specified subjects – skipping."
        )
        return False

    n_requested = len(allowed_subjects)
    n_found_subjects = len({s for s, _ in subjects_sessions})
    missing = allowed_subjects - {s for s, _ in subjects_sessions}
    print(
        f"  [{model}] Found {len(all_results)} sessions from "
        f"{n_found_subjects}/{n_requested} requested subjects."
    )
    if missing:
        print(
            f"  WARNING [{model}]: {len(missing)} subjects missing: "
            f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
        )

    # ------------------------------------------------------------------
    # 2. Build patient state map
    # ------------------------------------------------------------------
    subject_state_map = {}
    if patient_labels_path and Path(patient_labels_path).exists():
        try:
            df = pd.read_csv(patient_labels_path, dtype={"session": str})
            for _, row in df.iterrows():
                session_str = str(row["session"]).zfill(2)
                key = f"{row['subject']}_{session_str}"
                subject_state_map[key] = row["state"]
        except Exception as e:
            print(f"  WARNING [{model}]: Could not load patient labels: {e}")

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    aggregated = {
        "overall": {
            "all_mean_aucs": [],
            "all_mean_scores_time": [],
            "all_mean_scores_time_VS": [],
            "all_mean_scores_time_MCS": [],
            "all_mean_scores_time_dict": {},
            "MCS": {},
            "VS": {},
        },
        "trial_types": {},
        "has_task_events": False,
    }

    keys_VS, keys_MCS = [], []
    for i, results in enumerate(all_results):
        if "overall" not in results:
            continue
        aggregated["overall"]["all_mean_aucs"].append(results["overall"]["mean_auc"])
        aggregated["overall"]["all_mean_scores_time"].append(
            results["overall"]["mean_scores_time"]
        )
        sid, sesid = subjects_sessions[i]
        key = f"{sid}_{sesid}"
        aggregated["overall"]["all_mean_scores_time_dict"][key] = results["overall"][
            "mean_scores_time"
        ]
        if subject_state_map:
            state = subject_state_map.get(key, "UNKNOWN")
            if state in ["VS", "UWS"]:
                aggregated["overall"]["all_mean_scores_time_VS"].append(
                    results["overall"]["mean_scores_time"]
                )
                aggregated["overall"]["VS"][key] = results["overall"][
                    "mean_scores_time"
                ]
                keys_VS.append(key)
            elif state in ["MCS", "MCS-", "MCS+"]:
                aggregated["overall"]["all_mean_scores_time_MCS"].append(
                    results["overall"]["mean_scores_time"]
                )
                aggregated["overall"]["MCS"][key] = results["overall"][
                    "mean_scores_time"
                ]
                keys_MCS.append(key)

    if not aggregated["overall"]["all_mean_aucs"]:
        print(f"  WARNING [{model}]: No overall results to aggregate.")
        return False

    # Overall statistics
    aggregated["overall"]["mean_auc"] = np.mean(aggregated["overall"]["all_mean_aucs"])
    aggregated["overall"]["std_auc"] = np.std(aggregated["overall"]["all_mean_aucs"])
    all_ts = _truncate_to_common_length(aggregated["overall"]["all_mean_scores_time"])
    canonical_len = all_ts.shape[1]
    if times is not None and len(times) > canonical_len:
        times = times[:canonical_len]
    aggregated["overall"]["mean_scores_time"] = np.mean(all_ts, axis=0)
    aggregated["overall"]["std_scores_time"] = np.std(all_ts, axis=0)
    aggregated["overall"]["sem_scores_time"] = aggregated["overall"][
        "std_scores_time"
    ] / np.sqrt(len(all_ts))

    if aggregated["overall"]["all_mean_scores_time_VS"]:
        vs_ts = _truncate_to_common_length(
            aggregated["overall"]["all_mean_scores_time_VS"]
        )
        vs_ts = vs_ts[:, :canonical_len]
        aggregated["overall"]["mean_scores_time_VS"] = np.mean(vs_ts, axis=0)
        aggregated["overall"]["std_scores_time_VS"] = np.std(vs_ts, axis=0)
        aggregated["overall"]["n_subjects_VS"] = len(vs_ts)

    if aggregated["overall"]["all_mean_scores_time_MCS"]:
        mcs_ts = _truncate_to_common_length(
            aggregated["overall"]["all_mean_scores_time_MCS"]
        )
        mcs_ts = mcs_ts[:, :canonical_len]
        aggregated["overall"]["mean_scores_time_MCS"] = np.mean(mcs_ts, axis=0)
        aggregated["overall"]["std_scores_time_MCS"] = np.std(mcs_ts, axis=0)
        aggregated["overall"]["n_subjects_MCS"] = len(mcs_ts)

    # Trial-type aggregation
    trial_types = ["LSGS", "LSGD", "LDGD", "LDGS"]
    for tt in trial_types:
        aggregated["trial_types"][tt] = {
            "all_mean_aucs": [],
            "all_mean_scores_time": [],
            "all_mean_scores_time_VS": [],
            "all_mean_scores_time_MCS": [],
        }
        for i, results in enumerate(all_results):
            if "trial_types" in results and tt in results["trial_types"]:
                aggregated["trial_types"][tt]["all_mean_aucs"].append(
                    results["trial_types"][tt]["mean_auc"]
                )
                aggregated["trial_types"][tt]["all_mean_scores_time"].append(
                    results["trial_types"][tt]["mean_scores_time"]
                )
                if subject_state_map:
                    sid, sesid = subjects_sessions[i]
                    key = f"{sid}_{sesid}"
                    state = subject_state_map.get(key, "UNKNOWN")
                    if state in ["VS", "UWS"]:
                        aggregated["trial_types"][tt]["all_mean_scores_time_VS"].append(
                            results["trial_types"][tt]["mean_scores_time"]
                        )
                    elif state in ["MCS", "MCS-", "MCS+"]:
                        aggregated["trial_types"][tt][
                            "all_mean_scores_time_MCS"
                        ].append(results["trial_types"][tt]["mean_scores_time"])

        if aggregated["trial_types"][tt]["all_mean_aucs"]:
            tt_data = aggregated["trial_types"][tt]
            tt_data["mean_auc"] = np.mean(tt_data["all_mean_aucs"])
            tt_data["std_auc"] = np.std(tt_data["all_mean_aucs"])
            tt_ts = _truncate_to_common_length(tt_data["all_mean_scores_time"])
            tt_ts = tt_ts[:, :canonical_len]
            tt_data["mean_scores_time"] = np.mean(tt_ts, axis=0)
            tt_data["std_scores_time"] = np.std(tt_ts, axis=0)
            tt_data["sem_scores_time"] = tt_data["std_scores_time"] / np.sqrt(
                len(tt_ts)
            )
            tt_data["n_subjects_sessions"] = len(tt_ts)

            if tt_data["all_mean_scores_time_VS"]:
                vs = _truncate_to_common_length(tt_data["all_mean_scores_time_VS"])
                vs = vs[:, :canonical_len]
                tt_data["mean_scores_time_VS"] = np.mean(vs, axis=0)
                tt_data["std_scores_time_VS"] = np.std(vs, axis=0)
                tt_data["n_subjects_VS"] = len(vs)
            if tt_data["all_mean_scores_time_MCS"]:
                mcs = _truncate_to_common_length(tt_data["all_mean_scores_time_MCS"])
                mcs = mcs[:, :canonical_len]
                tt_data["mean_scores_time_MCS"] = np.mean(mcs, axis=0)
                tt_data["std_scores_time_MCS"] = np.std(mcs, axis=0)
                tt_data["n_subjects_MCS"] = len(mcs)

            # Permutation test
            perm = run_permutation_test(tt_ts)
            tt_data["permutation_test"] = perm
            print(
                f"    {tt}: AUC={perm['observed_mean']:.4f}  "
                f"p_perm={perm['p_value_permutation']:.4f}  "
                f"p_ttest={perm['p_value_ttest']:.4f}  "
                f"d={perm['cohens_d']:.4f}"
            )

    for results in all_results:
        if results.get("has_task_events", False):
            aggregated["has_task_events"] = True
            break

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    model_dir = output_dir / model
    data_dir = model_dir / "data"
    plots_dir = model_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pickle
    with open(data_dir / "aggregated_results.pkl", "wb") as f:
        pickle.dump(aggregated, f)

    if times is not None:
        np.save(data_dir / "times.npy", times)

    # Numpy timeseries
    np.save(data_dir / "all_mean_scores_time.npy", all_ts)
    if aggregated["overall"]["all_mean_scores_time_VS"]:
        np.save(
            data_dir / "all_mean_scores_time_VS.npy",
            _truncate_to_common_length(
                aggregated["overall"]["all_mean_scores_time_VS"]
            ),
        )
    if aggregated["overall"]["all_mean_scores_time_MCS"]:
        np.save(
            data_dir / "all_mean_scores_time_MCS.npy",
            _truncate_to_common_length(
                aggregated["overall"]["all_mean_scores_time_MCS"]
            ),
        )

    for tt in trial_types:
        td = aggregated["trial_types"][tt]
        if td["all_mean_scores_time"]:
            np.save(
                data_dir / f"all_mean_scores_time_{tt}.npy",
                _truncate_to_common_length(td["all_mean_scores_time"]),
            )
        if td.get("all_mean_scores_time_VS"):
            np.save(
                data_dir / f"all_mean_scores_time_VS_{tt}.npy",
                _truncate_to_common_length(td["all_mean_scores_time_VS"]),
            )
        if td.get("all_mean_scores_time_MCS"):
            np.save(
                data_dir / f"all_mean_scores_time_MCS_{tt}.npy",
                _truncate_to_common_length(td["all_mean_scores_time_MCS"]),
            )

    # Permutation test JSON
    perm_json = {}
    for tt in trial_types:
        if "permutation_test" in aggregated["trial_types"].get(tt, {}):
            pt = aggregated["trial_types"][tt]["permutation_test"]
            perm_json[tt] = {k: v for k, v in pt.items()}
    with open(data_dir / "permutation_test_results.json", "w") as f:
        json.dump(perm_json, f, indent=2)

    # JSON summary
    json_results = {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "intermodel_restricted_decoder_aggregation",
        "model": model,
        "n_subjects_sessions": len(subjects_sessions),
        "n_unique_subjects": n_found_subjects,
        "n_requested_subjects": n_requested,
        "missing_subjects": sorted(missing),
        "subjects_sessions": [
            {"subject_id": s, "session_id": ses} for s, ses in subjects_sessions
        ],
        "overall_classification": {},
        "trial_type_classification": {},
        "has_task_events": aggregated.get("has_task_events", False),
    }
    if "mean_auc" in aggregated["overall"]:
        json_results["overall_classification"] = {
            "mean_auc": float(aggregated["overall"]["mean_auc"]),
            "std_auc": float(aggregated["overall"]["std_auc"]),
            "individual_mean_aucs": [
                float(x) for x in aggregated["overall"]["all_mean_aucs"]
            ],
        }
        if "n_subjects_VS" in aggregated["overall"]:
            json_results["overall_classification"]["n_VS"] = int(
                aggregated["overall"]["n_subjects_VS"]
            )
        if "n_subjects_MCS" in aggregated["overall"]:
            json_results["overall_classification"]["n_MCS"] = int(
                aggregated["overall"]["n_subjects_MCS"]
            )
    for tt in trial_types:
        td = aggregated["trial_types"].get(tt, {})
        if "mean_auc" in td:
            json_results["trial_type_classification"][tt] = {
                "mean_auc": float(td["mean_auc"]),
                "std_auc": float(td["std_auc"]),
                "n_subjects_sessions": int(td["n_subjects_sessions"]),
            }
            if "permutation_test" in td:
                json_results["trial_type_classification"][tt]["permutation_test"] = {
                    k: v for k, v in td["permutation_test"].items()
                }
    with open(data_dir / "aggregated_summary.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Text summary
    _save_text_summary(aggregated, subjects_sessions, data_dir, model)

    # ------------------------------------------------------------------
    # 5. Wilcoxon per-timepoint analysis
    # ------------------------------------------------------------------
    _run_wilcoxon_and_save(aggregated, times, data_dir)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    _create_decoder_plots(aggregated, times, plots_dir, model)

    print(f"  [{model}] Decoder aggregation complete → {model_dir}")
    return True


def _save_text_summary(aggregated, subjects_sessions, data_dir, model):
    """Write a human-readable text summary."""
    with open(data_dir / "aggregated_summary.txt", "w") as f:
        f.write(f"Intermodel Restricted Decoder Aggregation – {model}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Subjects-sessions: {len(subjects_sessions)}\n\n")
        if "mean_auc" in aggregated["overall"]:
            f.write(
                f"Overall AUC: {aggregated['overall']['mean_auc']:.4f} "
                f"± {aggregated['overall']['std_auc']:.4f}\n"
            )
        for tt in ["LSGS", "LSGD", "LDGD", "LDGS"]:
            td = aggregated["trial_types"].get(tt, {})
            if "mean_auc" in td:
                f.write(
                    f"{tt}: {td['mean_auc']:.4f} ± {td['std_auc']:.4f} "
                    f"(n={td['n_subjects_sessions']})\n"
                )
        f.write("\nSubjects-sessions:\n")
        for s, ses in subjects_sessions:
            f.write(f"  sub-{s} ses-{ses}\n")


def _run_wilcoxon_and_save(aggregated, times, data_dir):
    """Run Wilcoxon per-timepoint and save CSV files."""
    # Canonical length from times (already truncated in caller)
    c_len = len(times) if times is not None else None

    # Overall
    if aggregated["overall"].get("all_mean_scores_time"):
        all_ts = _truncate_to_common_length(
            aggregated["overall"]["all_mean_scores_time"]
        )
        if c_len is not None:
            all_ts = all_ts[:, :c_len]
        wres = run_wilcoxon_per_timepoint(all_ts)
        t_arr = (
            times[: wres["n_timepoints"]]
            if times is not None
            else (np.arange(wres["n_timepoints"]))
        )
        df = pd.DataFrame(
            {
                "timepoint": np.arange(wres["n_timepoints"]),
                "time_seconds": t_arr,
                "p_value": wres["p_values"],
                "test_statistic": wres["test_statistics"],
                "significant": wres["significant_mask"],
            }
        )
        df.to_csv(data_dir / "overall_wilcoxon_per_timepoint.csv", index=False)
        sig_pct = 100 * wres["significant_mask"].mean()
        print(f"    Overall: {sig_pct:.1f}% significant timepoints")

    # Trial types
    for tt in ["LSGS", "LSGD", "LDGD", "LDGS"]:
        td = aggregated["trial_types"].get(tt, {})
        if td.get("all_mean_scores_time"):
            tt_ts = _truncate_to_common_length(td["all_mean_scores_time"])
            if c_len is not None:
                tt_ts = tt_ts[:, :c_len]
            wres = run_wilcoxon_per_timepoint(tt_ts)
            t_arr = (
                times[: wres["n_timepoints"]]
                if times is not None
                else (np.arange(wres["n_timepoints"]))
            )
            df = pd.DataFrame(
                {
                    "timepoint": np.arange(wres["n_timepoints"]),
                    "time_seconds": t_arr,
                    "p_value": wres["p_values"],
                    "test_statistic": wres["test_statistics"],
                    "significant": wres["significant_mask"],
                }
            )
            df.to_csv(data_dir / f"{tt}_wilcoxon_per_timepoint.csv", index=False)


def _create_decoder_plots(aggregated, times, plots_dir, model):
    """Create aggregated decoder plots (overall + trial-type)."""
    if times is None:
        print(f"  WARNING [{model}]: No times array – skipping plots.")
        return

    # --- Overall classification ---
    if "overall" in aggregated and "mean_scores_time" in aggregated["overall"]:
        mean_sc = aggregated["overall"]["mean_scores_time"]
        std_sc = aggregated["overall"]["std_scores_time"]
        n_sess = len(aggregated["overall"]["all_mean_aucs"])
        t = times[: len(mean_sc)]

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        ax.plot(t, mean_sc, "b-", linewidth=3, label=f"Mean AUC (n={n_sess})")
        ax.fill_between(
            t,
            mean_sc - std_sc,
            mean_sc + std_sc,
            alpha=0.3,
            color="blue",
            label=r"$\sigma$",
        )
        add_stimulus_lines(ax, t)
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylabel("AUC", fontsize=20)
        ax.set_title(f"{model} – Overall Decoding (restricted subjects)", fontsize=22)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        peak_idx = np.argmax(mean_sc)
        ax.text(
            0.02,
            0.98,
            f"Peak: AUC={mean_sc[peak_idx]:.3f} at t={t[peak_idx]:.3f}s",
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "aggregated_overall_classification.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Small version
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        trim = min(20, len(mean_sc) - 1)
        t_s = t[:-trim] if trim > 0 else t
        ms_s = mean_sc[:-trim] if trim > 0 else mean_sc
        ss_s = std_sc[:-trim] if trim > 0 else std_sc
        ax.plot(t_s, ms_s, "b-", linewidth=3)
        ax.fill_between(t_s, ms_s - ss_s, ms_s + ss_s, alpha=0.3, color="blue")
        add_stimulus_lines(ax, t_s)
        ax.set_xlabel("Time (s)", fontsize=18)
        ax.set_ylabel("AUC-ROC", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        plt.tight_layout()
        plt.savefig(
            plots_dir / "aggregated_overall_classification_small.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --- Trial-type combined plot ---
    trial_types = ["LSGS", "LSGD", "LDGD", "LDGS"]
    colors = ["blue", "red", "green", "orange"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    has_any = False
    for i, tt in enumerate(trial_types):
        td = aggregated["trial_types"].get(tt, {})
        if "mean_scores_time" not in td:
            continue
        has_any = True
        m = td["mean_scores_time"]
        s = td["std_scores_time"]
        t = times[: len(m)]
        ax.plot(
            t,
            m,
            color=colors[i],
            linewidth=2.5,
            label=f"{tt}: {td['mean_auc']:.3f} ± {td['std_auc']:.3f}",
        )
        ax.fill_between(t, m - s, m + s, alpha=0.1, color=colors[i])
    if has_any:
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Time (s)", fontsize=24)
        ax.set_ylabel("AUC", fontsize=24)
        ax.set_title(
            f"{model} – Decoding per Trial Type (restricted subjects)", fontsize=24
        )
        ax.legend(fontsize=18, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        plt.tight_layout()
        plt.savefig(
            plots_dir / "aggregated_trial_type_classification.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()

    # Small version
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, tt in enumerate(trial_types):
        td = aggregated["trial_types"].get(tt, {})
        if "mean_scores_time" not in td:
            continue
        m = td["mean_scores_time"]
        s = td["std_scores_time"]
        t = times[: len(m)]
        ax.plot(t, m, color=colors[i], linewidth=2.5, label=tt)
        ax.fill_between(t, m - s, m + s, alpha=0.1, color=colors[i])
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("AUC-ROC", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    plt.tight_layout()
    plt.savefig(
        plots_dir / "aggregated_trial_type_classification_small.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --- By subject type (VS / MCS) ---
    if aggregated["overall"].get("all_mean_scores_time_VS") and aggregated[
        "overall"
    ].get("all_mean_scores_time_MCS"):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for idx, (label, key, color) in enumerate(
            [
                ("VS/UWS", "VS", "red"),
                ("MCS", "MCS", "orange"),
            ]
        ):
            ax = axes[idx]
            m_key = f"mean_scores_time_{key}"
            s_key = f"std_scores_time_{key}"
            n_key = f"n_subjects_{key}"
            if m_key not in aggregated["overall"]:
                ax.set_visible(False)
                continue
            m = aggregated["overall"][m_key]
            s = aggregated["overall"][s_key]
            n = aggregated["overall"][n_key]
            t = times[: len(m)]
            ax.plot(t, m, color=color, linewidth=2, label=f"Mean AUC (n={n})")
            ax.fill_between(t, m - s, m + s, alpha=0.3, color=color, label=r"$\sigma$")
            ax.axhline(0.5, color="k", linestyle="--", alpha=0.7)
            add_stimulus_lines(ax, t)
            ax.set_xlabel("Time (s)", fontsize=20)
            ax.set_ylabel("AUC", fontsize=20)
            ax.set_title(f"{label} (n={n})", fontsize=22)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.4, 1.0])
        plt.suptitle(
            f"{model} – Decoder by Subject Type (restricted subjects)", fontsize=22
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "aggregated_by_subject_type.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


# ======================================================================
# MLP EMBEDDING classification (restricted subjects)
# ======================================================================


def run_mlp_for_model(
    model: str,
    allowed_subjects: set,
    patient_labels_path: str,
    output_dir: Path,
    embedding_dir_map: dict,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    patience: int = 100,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    """Run MLP embedding classification for *model* restricted to
    *allowed_subjects*.  Returns True on success.
    """
    # Locate embedding directory
    emb_dir = embedding_dir_map.get(model)
    if emb_dir is None or not Path(emb_dir).exists():
        print(
            f"  WARNING [{model}]: Embedding directory not found "
            f"({emb_dir}) – skipping MLP."
        )
        return False

    emb_dir = Path(emb_dir)
    model_out = output_dir / model
    model_out.mkdir(parents=True, exist_ok=True)

    # Find the classifier module on sys.path
    src_dir = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_dir / "model"))

    from mlp_embedding_classifier import EmbeddingMLPClassifier

    # ------------------------------------------------------------------
    # Subclass to inject subject filtering
    # ------------------------------------------------------------------
    class FilteredEmbeddingMLPClassifier(EmbeddingMLPClassifier):
        """Override collect_data to restrict subjects."""

        def __init__(self, *args, allowed_subjects_set=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._allowed = allowed_subjects_set or set()

        def collect_data(self, embedding_suffix="_embedding.npy"):
            """Collect data, filtering by allowed subjects, tracking failures."""
            labels_dict, _ = self.load_patient_labels()
            embeddings = self.load_embeddings(embedding_suffix)

            X_list, y_list, subjects_list = [], [], []
            # Per-subject failure tracking (cleared each call)
            self._skipped_no_label = []
            self._skipped_shape_mismatch = []

            for key, emb in sorted(embeddings.items()):
                # key is like "AA048_ses-01"
                subj_id = key.split("_ses-")[0] if "_ses-" in key else key
                if subj_id not in self._allowed:
                    continue
                if key not in labels_dict:
                    self._skipped_no_label.append(key)
                    continue
                if X_list and emb.shape != X_list[0].shape:
                    self._skipped_shape_mismatch.append(
                        f"{key} (got {emb.shape}, expected {X_list[0].shape})"
                    )
                    continue
                X_list.append(emb)
                y_list.append(labels_dict[key])
                subjects_list.append(key)

            # Report per-subject skips explicitly
            if self._skipped_no_label:
                print(
                    f"   [SKIP – no patient label]  "
                    f"{len(self._skipped_no_label)} entries: "
                    f"{self._skipped_no_label}",
                    flush=True,
                )
            if self._skipped_shape_mismatch:
                print(
                    f"   [SKIP – shape mismatch]  "
                    f"{len(self._skipped_shape_mismatch)} entries: "
                    f"{self._skipped_shape_mismatch}",
                    flush=True,
                )

            if not X_list:
                raise ValueError(
                    f"No subjects matched between allowed list, "
                    f"embeddings and labels for {model}!"
                )

            self.X = np.array(X_list)
            self.y = np.array(y_list)
            self.subjects = subjects_list
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            self.class_names = self.label_encoder.classes_

            print(
                f"   Dataset (filtered): {self.X.shape[0]} samples x "
                f"{self.X.shape[1]} features",
                flush=True,
            )
            print(f"   Classes: {list(self.class_names)}", flush=True)
            unique, counts = np.unique(self.y, return_counts=True)
            for cls, cnt in zip(unique, counts):
                print(f"      {cls}: {cnt}", flush=True)

            return self.X, self.y_encoded, self.subjects

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    try:
        clf = FilteredEmbeddingMLPClassifier(
            data_dir=str(emb_dir),
            patient_labels_file=patient_labels_path,
            output_dir=str(model_out),
            random_state=random_state,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            patience=patience,
            allowed_subjects_set=allowed_subjects,
        )
        clf.run_classification(test_size=test_size, val_size=val_size)
        failures = {
            "no_label": getattr(clf, "_skipped_no_label", []),
            "shape_mismatch": getattr(clf, "_skipped_shape_mismatch", []),
        }
        print(f"  [{model}] MLP classification complete → {model_out}")
        return True, failures
    except Exception as e:
        print(f"  WARNING [{model}]: MLP classification failed: {e}")
        return False, {"exception": str(e)}


# ======================================================================
# Data-leakage sanity check (dry-run, no array loading)
# ======================================================================


def _check_group_leakage(
    model: str,
    emb_dir: Path,
    allowed_subjects: set,
    patient_labels_path: str,
    mlp_test_size: float = 0.2,
    mlp_val_size: float = 0.2,
    mlp_random_state: int = 42,
) -> bool:
    """Dry-run group splits and verify zero subject-level data leakage.

    Scans embedding directory filenames (no arrays loaded) to build the
    list of valid subject-session keys, then simulates:

    * MLP  – two nested :class:`GroupShuffleSplit` (outer test / inner val)

    For every partition pair the function checks that the sets of subject
    IDs are disjoint.  Results are printed with explicit subject lists
    when a violation is found.

    Parameters
    ----------
    model :
        Model name, used only for console labels.
    emb_dir :
        Root embedding directory for this model.
    allowed_subjects :
        Intersection subject set (no filtering needed inside).
    patient_labels_path :
        CSV with ``subject``, ``session``, ``diagnostic_crs_final``.
    mlp_test_size / mlp_val_size / mlp_random_state :
        Must match the values used when actually training the MLP.

    Returns
    -------
    bool
        True if all checks pass (no leakage found).
    """
    from sklearn.model_selection import GroupShuffleSplit

    # ── 1. Build labels set (keys that have a VS/MCS label) ──────────
    df = pd.read_csv(patient_labels_path)
    labels_set: set[str] = set()
    for _, row in df.iterrows():
        state = row["diagnostic_crs_final"]
        if pd.isna(state) or state == "n/a":
            continue
        if state not in ("UWS", "VS", "MCS", "MCS+", "MCS-"):
            continue
        session = f"ses-{int(row['session']):02d}"
        labels_set.add(f"{row['subject']}_{session}")

    # ── 2. Collect keys by scanning filenames (no np.load) ───────────
    keys: list[str] = []
    for sub_dir in sorted(emb_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        subject_id = sub_dir.name[4:]
        if subject_id not in allowed_subjects:
            continue
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue
            has_emb = any(
                f.suffix == ".npy"
                and "embedding" in f.name
                and "_metadata" not in f.name
                for f in ses_dir.iterdir()
            )
            if not has_emb:
                continue
            key = f"{subject_id}_{ses_dir.name}"
            if key in labels_set:
                keys.append(key)

    if len(keys) < 3:
        print(
            f"  [{model}] leakage check: only {len(keys)} valid sessions "
            f"— skipped (need ≥3)"
        )
        return True

    groups = np.array([k.split("_ses-")[0] for k in keys])
    unique_subj = np.unique(groups)
    n_subj, n_sess = len(unique_subj), len(keys)
    X0, y0 = np.zeros((n_sess, 1)), np.zeros(n_sess)

    print(f"  [{model}] leakage check — {n_sess} sessions, {n_subj} unique subjects")
    all_ok = True

    # ── 3. MLP split (two nested GroupShuffleSplit) ───────────────────
    try:
        tv_idx, te_idx = next(
            GroupShuffleSplit(
                1, test_size=mlp_test_size, random_state=mlp_random_state
            ).split(X0, y0, groups=groups)
        )
        g_tv = groups[tv_idx]
        tr_idx_i, va_idx_i = next(
            GroupShuffleSplit(
                1, test_size=mlp_val_size, random_state=mlp_random_state
            ).split(X0[tv_idx], y0[tv_idx], groups=g_tv)
        )
        g_tr = set(g_tv[tr_idx_i])
        g_va = set(g_tv[va_idx_i])
        g_te = set(groups[te_idx])

        pairs = {
            "train ∩ val": g_tr & g_va,
            "train ∩ test": g_tr & g_te,
            "val   ∩ test": g_va & g_te,
        }
        mlp_ok = not any(pairs.values())
        tag = "✓ no leakage" if mlp_ok else "✗ LEAKAGE DETECTED"
        print(f"    MLP (GroupShuffleSplit): {tag}")
        print(
            f"      train={len(g_tr)} subj | val={len(g_va)} subj "
            f"| test={len(g_te)} subj"
        )
        for label, overlap in pairs.items():
            if overlap:
                print(f"      [LEAK] {label}: {sorted(overlap)}")
                all_ok = False
    except Exception as exc:
        print(f"    MLP split check error: {exc}")
        all_ok = False

    return all_ok


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Inter-model results aggregation with subject filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decoder aggregation only (restricted subjects)
  python intermodel_results.py --decoder-only

  # MLP only
  python intermodel_results.py --mlp-embedding-only

  # Both decoder + MLP (default)
  python intermodel_results.py

  # Further restrict by a reference directory
  python intermodel_results.py --unique-subjects-values \\
      --subject-dir /my/custom/subject/dir
        """,
    )

    parser.add_argument(
        "--unique-subjects-values",
        action="store_true",
        help="Restrict to subjects present in the reference directory "
        "(default: fifdata DOC directory)",
    )
    parser.add_argument(
        "--subject-dir",
        type=str,
        default=DEFAULT_SUBJECT_DIR,
        help=f"Directory whose sub-{{ID}} folders define the subject list "
        f"(default: {DEFAULT_SUBJECT_DIR})",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root of benchmark results (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root (default: {results-root})",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=DEFAULT_METADATA_DIR,
        help=f"Metadata directory (default: {DEFAULT_METADATA_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help=f"Models to process (default: {MODELS})",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--decoder-only", action="store_true", help="Only re-aggregate decoder results"
    )
    mode_group.add_argument(
        "--mlp-embedding-only",
        action="store_true",
        help="Only re-run MLP embedding classification",
    )

    # MLP hyper-parameters
    parser.add_argument("--mlp-n-epochs", type=int, default=500)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-patience", type=int, default=100)
    parser.add_argument("--mlp-test-size", type=float, default=0.2)
    parser.add_argument("--mlp-val-size", type=float, default=0.2)
    parser.add_argument("--mlp-random-state", type=int, default=42)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve subject list: intersection of all models' embedding dirs
    # ------------------------------------------------------------------
    emb_map = dict(EMBEDDING_DIR_MAP)
    model_subject_sets = {}
    for model in args.models:
        emb_dir = emb_map.get(model)
        if emb_dir is None or not Path(emb_dir).exists():
            print(f"WARNING: Embedding directory for {model} not found: {emb_dir}")
            model_subject_sets[model] = set()
            continue
        model_subject_sets[model] = set(
            d.name.replace("sub-", "")
            for d in Path(emb_dir).iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        print(
            f"{model}: {len(model_subject_sets[model])} subjects found in embeddings dir."
        )

    # Find intersection
    if args.unique_subjects_values:
        # Optionally restrict to reference dir as well
        ref_subjects = get_reference_subjects(args.subject_dir)
        if not ref_subjects:
            print("ERROR: No subjects found in reference directory. Aborting.")
            sys.exit(1)
        intersect_subjects = (
            set.intersection(*(model_subject_sets[m] for m in args.models))
            & ref_subjects
        )
        print(f"Intersected with reference dir: {len(intersect_subjects)} subjects.")
    else:
        intersect_subjects = set.intersection(
            *(model_subject_sets[m] for m in args.models)
        )

    if not intersect_subjects:
        print(
            "ERROR: No common subjects found across all models' embedding dirs. Aborting."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Detailed per-model set analysis: show what each model contributes
    # and which subjects were excluded because they're missing from at
    # least one other model's embedding directory.
    # ------------------------------------------------------------------
    print(f"\nPer-model subject set analysis (embedding dirs):")
    for m in args.models:
        s = model_subject_sets[m]
        excluded = s - intersect_subjects  # present here but absent in ≥1 other model
        line = f"  {m:12s}: {len(s):3d} subjects in embedding dir"
        if excluded:
            line += (
                f", {len(excluded)} excluded from intersection"
                f" (absent in ≥1 other model):\n"
                f"    {sorted(excluded)}"
            )
        else:
            line += ", all subjects are in the intersection"
        print(line)
    print(
        f"\n  → INTERSECTION: {len(intersect_subjects)} common subjects across all models"
    )
    preview = sorted(intersect_subjects)
    if len(preview) <= 15:
        print(f"    {preview}")
    else:
        print(f"    first 15: {preview[:15]} ...")

    allowed_subjects = intersect_subjects

    output_root = (
        Path(args.output_root) if args.output_root else Path(args.results_root)
    )
    patient_labels_path = str(
        Path(args.metadata_dir) / "patient_labels_with_controls.csv"
    )

    run_decoder = not args.mlp_embedding_only
    run_mlp = not args.decoder_only

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("INTER-MODEL RESULTS AGGREGATION")
    print("=" * 70)
    print(f"Date       : {datetime.now().isoformat()}")
    print(f"Models     : {args.models}")
    print(f"Decoder    : {'Yes' if run_decoder else 'No'}")
    print(f"MLP        : {'Yes' if run_mlp else 'No'}")
    print(f"Subjects   : {len(allowed_subjects)} (intersection across all models)")
    print(f"Output root: {output_root}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Data-leakage sanity check (dry-run, no training)
    # Runs for every model whose MLP will be trained.
    # ------------------------------------------------------------------
    if run_mlp:
        print(f"\n{'=' * 60}")
        print("DATA-LEAKAGE SANITY CHECK")
        print(f"{'=' * 60}")
        leakage_all_ok = True
        for model in args.models:
            emb_dir_check = Path(emb_map.get(model, ""))
            if not emb_dir_check.exists():
                print(f"  [{model}] embedding dir missing — skipping check")
                continue
            ok = _check_group_leakage(
                model=model,
                emb_dir=emb_dir_check,
                allowed_subjects=allowed_subjects,
                patient_labels_path=patient_labels_path,
                mlp_test_size=args.mlp_test_size,
                mlp_val_size=args.mlp_val_size,
                mlp_random_state=args.mlp_random_state,
            )
            if not ok:
                leakage_all_ok = False
        if leakage_all_ok:
            print(f"\n  ✓ All leakage checks passed — proceeding with training.")
        else:
            print(f"\n  ✗ WARNING: Data leakage detected in one or more models!")
            print(f"    Results should NOT be trusted. Check split parameters.")

    # ------------------------------------------------------------------
    # DECODER
    # ------------------------------------------------------------------
    if run_decoder:
        decoder_output = output_root / "DECODER" / f"intermodel-{timestamp}"
        decoder_output.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print("DECODER AGGREGATION (restricted subjects)")
        print(f"{'=' * 60}")

        decoder_summary = {}
        decoder_missing = {}
        for model in args.models:
            print(f"\n--- {model} ---")
            subj_set = allowed_subjects
            ok = aggregate_decoder_for_model(
                model=model,
                results_root=args.results_root,
                allowed_subjects=subj_set,
                patient_labels_path=patient_labels_path,
                output_dir=decoder_output,
            )
            decoder_summary[model] = "OK" if ok else "SKIPPED"
            # Report subjects in intersection that have no decoder results dir
            model_decoder_dir = (
                Path(args.results_root) / model / "doc_patients" / "DECODER"
            )
            if model_decoder_dir.exists():
                found_subjects = set(
                    d.name.replace("sub-", "")
                    for d in model_decoder_dir.glob("sub-*")
                    if d.is_dir()
                )
            else:
                found_subjects = set()
                print(
                    f"  [WARNING] {model}: DECODER directory missing: "
                    f"{model_decoder_dir}"
                )
            missing = subj_set - found_subjects
            decoder_missing[model] = sorted(missing)
            if missing:
                print(
                    f"  [WARNING] {model}: {len(missing)} intersection subjects "
                    f"have no decoder results: {sorted(missing)}"
                )

        # Save cross-model summary
        with open(decoder_output / "intermodel_summary.json", "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "models": decoder_summary,
                    "n_requested_subjects": len(allowed_subjects),
                    "subject_source": str(args.subject_dir),
                    "missing_subjects_per_model": decoder_missing,
                },
                f,
                indent=2,
            )

        print(f"\nDecoder summary:")
        for m, s in decoder_summary.items():
            print(f"  {m}: {s}")
        for m, missing in decoder_missing.items():
            if missing:
                print(
                    f"  [WARNING] {m}: {len(missing)} intersection subjects "
                    f"missing decoder results: {missing}"
                )

    # ------------------------------------------------------------------
    # MLP CLASSIFIER
    # ------------------------------------------------------------------
    if run_mlp:
        mlp_output = output_root / "MLP-CLASSIFIER" / f"intermodel-{timestamp}"
        mlp_output.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print("MLP EMBEDDING CLASSIFICATION (restricted subjects)")
        print(f"{'=' * 60}")

        mlp_summary = {}
        mlp_failed_subjects = {}
        for model in args.models:
            print(f"\n--- {model} ---")
            subj_set = allowed_subjects
            ok, failures = run_mlp_for_model(
                model=model,
                allowed_subjects=subj_set,
                patient_labels_path=patient_labels_path,
                output_dir=mlp_output,
                embedding_dir_map=emb_map,
                n_epochs=args.mlp_n_epochs,
                lr=args.mlp_lr,
                batch_size=args.mlp_batch_size,
                weight_decay=args.mlp_weight_decay,
                patience=args.mlp_patience,
                random_state=args.mlp_random_state,
                test_size=args.mlp_test_size,
                val_size=args.mlp_val_size,
            )
            mlp_summary[model] = "OK" if ok else "SKIPPED/FAILED"
            mlp_failed_subjects[model] = failures
            # Explicit per-failure-type reporting
            no_lbl = failures.get("no_label", [])
            shape_mm = failures.get("shape_mismatch", [])
            exc = failures.get("exception")
            if exc:
                print(f"  [ERROR] {model}: {exc}")
            if no_lbl:
                print(
                    f"  [WARNING] {model}: {len(no_lbl)} entries skipped "
                    f"(no patient label): {no_lbl}"
                )
            if shape_mm:
                print(
                    f"  [WARNING] {model}: {len(shape_mm)} entries skipped "
                    f"(shape mismatch): {shape_mm}"
                )

        with open(mlp_output / "intermodel_summary.json", "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "models": mlp_summary,
                    "n_common_subjects": len(allowed_subjects),
                    "common_subjects": sorted(allowed_subjects),
                    "failures_per_model": mlp_failed_subjects,
                },
                f,
                indent=2,
            )

        print(f"\nMLP summary:")
        for m, s in mlp_summary.items():
            print(f"  {m}: {s}")
        for m, failures in mlp_failed_subjects.items():
            total = len(failures.get("no_label", [])) + len(
                failures.get("shape_mismatch", [])
            )
            if total:
                print(
                    f"  [WARNING] {m}: {total} entries failed in MLP "
                    f"(no_label={len(failures.get('no_label', []))}, "
                    f"shape_mismatch={len(failures.get('shape_mismatch', []))})"
                )
            if failures.get("exception"):
                print(f"  [ERROR]   {m}: classification raised exception")

    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("INTER-MODEL AGGREGATION COMPLETE")
    print(f"{'=' * 70}")


# ======================================================================
# Fallback helpers when no filtering is active
# ======================================================================


def _get_all_decoder_subjects(results_root, model):
    """Return set of all subject IDs with decoder results for *model*."""
    base = Path(results_root) / model / "doc_patients" / "DECODER"
    if not base.exists():
        return set()
    return {d.name.replace("sub-", "") for d in base.glob("sub-*") if d.is_dir()}


def _get_all_embedding_subjects(emb_map, model):
    """Return set of all subject IDs with embeddings for *model*."""
    emb_dir = emb_map.get(model)
    if emb_dir is None or not Path(emb_dir).exists():
        return set()
    return {
        d.replace("sub-", "")
        for d in os.listdir(emb_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(emb_dir, d))
    }


if __name__ == "__main__":
    main()
