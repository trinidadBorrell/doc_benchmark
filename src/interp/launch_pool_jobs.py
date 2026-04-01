"""Generate and optionally submit HTCondor jobs for parallel linear probing.

Three stages, each producing its own ``.submit`` file:

1. **pool** — Split subjects into N chunks, one job per chunk.
   Each job runs ``linear_probing.py --pool-only --subjects <chunk>``.

2. **analysis** — One job per (model, layer) combination.
   Each job runs regression + classification for a single layer.
   Number of jobs = number of layers in the model config.

3. **aggregate** — Single job that reads all per-layer JSONs and
   produces summary files + plots (``--aggregate-only``).

Usage
-----
::

    # Pool (dry-run):
    python launch_pool_jobs.py --model CbraMod --stage pool --n-jobs 10

    # Analysis (one job per layer, auto-detected):
    python launch_pool_jobs.py --model CbraMod --stage analysis

    # Aggregate (single job):
    python launch_pool_jobs.py --model CbraMod --stage aggregate

    # All three at once (submit sequentially yourself):
    python launch_pool_jobs.py --model CbraMod --stage all --n-jobs 10

    # Actually submit any of the above:
    python launch_pool_jobs.py --model CbraMod --stage pool --n-jobs 10 --submit

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import math
import os
import os.path as op
import subprocess
import sys
import textwrap

# ---- defaults (mirrors linear_probing.py) --------------------------------

DEFAULT_DATA_DIR = "/data/project/eeg_foundation/data"
DEFAULT_OUTPUT_DIR = (
    "/data/project/eeg_foundation/data/benchmark_results/new_results/LINEAR_PROBING"
)
DEFAULT_PATIENT_LABELS = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)
DEFAULT_MARKER_CSV = (
    "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
)
LOG_DIR = "/data/project/eeg_foundation/logs"
SCRIPT_DIR = op.dirname(op.abspath(__file__))
LINEAR_PROBING = op.join(SCRIPT_DIR, "linear_probing.py")
RUN_POOL_SH = op.join(SCRIPT_DIR, "run_pool.sh")
RUN_ANALYSIS_SH = op.join(SCRIPT_DIR, "run_analysis.sh")

MODEL_DATA_SUBDIRS = {
    "CbraMod": "CbraMod/multi_layer_embeddings",
    "NeuroLM": "NeuroLM/multi_layer_embeddings",
}

# Layer keys per model (must match MODEL_LAYER_KEYS in linear_probing.py)
MODEL_LAYERS = {
    "CbraMod": ["patch_emb", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"],
    "NeuroLM": ["gpt_0", "gpt_3", "gpt_6", "gpt_9", "gpt_11"],
}


# ---- helpers -------------------------------------------------------------


def discover_subjects(data_dir, model):
    """Return sorted list of sub-* directory names for a model."""
    subdir = MODEL_DATA_SUBDIRS.get(model)
    if subdir is None:
        raise ValueError(f"Unknown model {model!r}")
    emb_dir = op.join(data_dir, subdir)
    if not op.isdir(emb_dir):
        raise FileNotFoundError(f"Embedding dir not found: {emb_dir}")
    subjects = sorted(
        d
        for d in os.listdir(emb_dir)
        if d.startswith("sub-") and op.isdir(op.join(emb_dir, d))
    )
    return subjects


def chunk_list(lst, n_chunks):
    """Split *lst* into *n_chunks* roughly-equal sublists."""
    k = math.ceil(len(lst) / n_chunks)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def _condor_header(model, tag, request_cpus, request_memory, max_runtime=86400):
    """Common HTCondor submit header."""
    return textwrap.dedent(f"""\
    # Auto-generated HTCondor submit file — {tag}
    # Model: {model}
    requirements      = (Arch == "ppc64le")

    universe          = vanilla
    getenv            = True

    request_cpus      = {request_cpus}
    request_memory    = {request_memory}

    +MaxRuntime       = {max_runtime}

    transfer_executable = False
    should_transfer_files = NO
    stream_output     = True
    stream_error      = True

    """)


# ---- submit-file builders ------------------------------------------------


def build_pool_submit(
    model, chunks, data_dir, output_dir, request_cpus, request_memory
):
    """Submit file: parallel pooling (one job per subject chunk)."""
    total = sum(len(c) for c in chunks)
    header = _condor_header(
        model,
        f"pool ({len(chunks)} jobs, {total} subjects)",
        request_cpus,
        request_memory,
    )
    header += textwrap.dedent(f"""\
    executable        = /bin/bash
    arguments         = {RUN_POOL_SH} --model $(model) --subjects $(subjects) --data_dir $(data_dir) --output_dir $(output_dir)

    log    = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).log
    output = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).out
    error  = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).err

    queue model,subjects,data_dir,output_dir from (
    """)

    rows = []
    for chunk in chunks:
        subjects_joined = "+".join(chunk)
        rows.append(f"    {model}, {subjects_joined}, {data_dir}, {output_dir}")

    return header + "\n".join(rows) + "\n)\n"


def build_analysis_submit(
    model,
    layers,
    data_dir,
    output_dir,
    patient_labels,
    marker_csv,
    request_cpus,
    request_memory,
    splits_file="",
):
    """Submit file: one job per layer (regression + classification)."""
    header = _condor_header(
        model,
        f"analysis ({len(layers)} layers)",
        request_cpus,
        request_memory,
        max_runtime=172800,
    )
    splits_arg = " --splits_file $(splits_file)" if splits_file else ""
    header += textwrap.dedent(f"""\
    executable        = /bin/bash
    arguments         = {RUN_ANALYSIS_SH} --model $(model) --layer $(layer) --data_dir $(data_dir) --output_dir $(output_dir) --patient_labels $(patient_labels) --marker_csv $(marker_csv){splits_arg}

    log    = {LOG_DIR}/analysis_{model}_$(Cluster).$(Process).log
    output = {LOG_DIR}/analysis_{model}_$(Cluster).$(Process).out
    error  = {LOG_DIR}/analysis_{model}_$(Cluster).$(Process).err

    queue model,layer,data_dir,output_dir,patient_labels,marker_csv{"" if not splits_file else ",splits_file"} from (
    """)

    rows = []
    for layer in layers:
        row = (
            f"    {model}, {layer}, {data_dir}, {output_dir}, "
            f"{patient_labels}, {marker_csv}"
        )
        if splits_file:
            row += f", {splits_file}"
        rows.append(row)

    return header + "\n".join(rows) + "\n)\n"


def build_aggregate_submit(
    model,
    data_dir,
    output_dir,
    patient_labels,
    marker_csv,
    request_cpus,
    request_memory,
):
    """Submit file: single aggregation job."""
    header = _condor_header(
        model, "aggregate (1 job)", request_cpus, request_memory, max_runtime=3600
    )
    header += textwrap.dedent(f"""\
    executable        = /bin/bash
    arguments         = {RUN_ANALYSIS_SH} --model $(model) --layer all --data_dir $(data_dir) --output_dir $(output_dir) --patient_labels $(patient_labels) --marker_csv $(marker_csv) --extra --aggregate-only

    log    = {LOG_DIR}/aggregate_{model}_$(Cluster).$(Process).log
    output = {LOG_DIR}/aggregate_{model}_$(Cluster).$(Process).out
    error  = {LOG_DIR}/aggregate_{model}_$(Cluster).$(Process).err

    queue model,data_dir,output_dir,patient_labels,marker_csv from (
        {model}, {data_dir}, {output_dir}, {patient_labels}, {marker_csv}
    )
    """)
    return header


# ---- main ----------------------------------------------------------------


def _write_and_maybe_submit(submit_text, submit_path, do_submit):
    """Write submit file and optionally condor_submit it."""
    os.makedirs(op.dirname(submit_path), exist_ok=True)
    with open(submit_path, "w") as f:
        f.write(submit_text)
    print(f"  Written: {submit_path}")

    if do_submit:
        result = subprocess.run(
            ["condor_submit", submit_path],
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"  condor_submit failed:\n{result.stderr}", file=sys.stderr)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTCondor jobs for parallel linear probing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Workflow
        -------
        1. pool     — parallel pooling (N jobs, split by subjects)
        2. analysis — parallel regression + classification (1 job per layer)
        3. aggregate — single job to merge per-layer JSONs into summaries + plots

        Examples
        --------
        python launch_pool_jobs.py --model CbraMod --stage pool --n-jobs 5 --submit
        python launch_pool_jobs.py --model CbraMod --stage analysis --submit
        python launch_pool_jobs.py --model CbraMod --stage aggregate --submit
        python launch_pool_jobs.py --model CbraMod --stage all --n-jobs 5 --submit
        """),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_DATA_SUBDIRS.keys()),
        help="Foundation model.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pool", "analysis", "aggregate", "all"],
        help="Which stage(s) to generate submit files for.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of pooling jobs (ignored for analysis/aggregate).",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--patient-labels",
        default=DEFAULT_PATIENT_LABELS,
    )
    parser.add_argument(
        "--marker-csv",
        default=DEFAULT_MARKER_CSV,
    )
    parser.add_argument(
        "--request-cpus",
        type=int,
        default=4,
        help="CPUs per job (default: 4).",
    )
    parser.add_argument(
        "--request-memory",
        default="16GB",
    )
    parser.add_argument(
        "--splits-file",
        default="",
        help="Path to pre-computed CV splits JSON (forwarded to linear_probing.py analysis jobs).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit to HTCondor (default: dry-run).",
    )

    args = parser.parse_args()
    stages = ["pool", "analysis", "aggregate"] if args.stage == "all" else [args.stage]
    model = args.model
    layers = MODEL_LAYERS[model]
    submit_paths = []

    # ---- pool ----
    if "pool" in stages:
        subjects = discover_subjects(args.data_dir, model)
        n_jobs = min(args.n_jobs, len(subjects))
        chunks = chunk_list(subjects, n_jobs)
        print(
            f"[pool] {len(subjects)} subjects -> {len(chunks)} jobs "
            f"({len(chunks[0])} subjects/job)"
        )

        text = build_pool_submit(
            model,
            chunks,
            args.data_dir,
            args.output_dir,
            args.request_cpus,
            args.request_memory,
        )
        path = op.join(LOG_DIR, f"pool_{model}_{len(chunks)}jobs.submit")
        _write_and_maybe_submit(text, path, args.submit)
        submit_paths.append(path)

    # ---- analysis ----
    if "analysis" in stages:
        print(f"[analysis] {model}: {len(layers)} layers -> {len(layers)} jobs")
        for lyr in layers:
            print(f"  - {lyr}")

        text = build_analysis_submit(
            model,
            layers,
            args.data_dir,
            args.output_dir,
            args.patient_labels,
            args.marker_csv,
            args.request_cpus,
            args.request_memory,
            splits_file=args.splits_file,
        )
        path = op.join(LOG_DIR, f"analysis_{model}_{len(layers)}layers.submit")
        _write_and_maybe_submit(text, path, args.submit)
        submit_paths.append(path)

    # ---- aggregate ----
    if "aggregate" in stages:
        print(f"[aggregate] {model}: 1 job")

        text = build_aggregate_submit(
            model,
            args.data_dir,
            args.output_dir,
            args.patient_labels,
            args.marker_csv,
            args.request_cpus,
            args.request_memory,
        )
        path = op.join(LOG_DIR, f"aggregate_{model}.submit")
        _write_and_maybe_submit(text, path, args.submit)
        submit_paths.append(path)

    # ---- summary ----
    if not args.submit:
        print("\nDry-run mode. To submit:")
        for p in submit_paths:
            print(f"  condor_submit {p}")


if __name__ == "__main__":
    main()
