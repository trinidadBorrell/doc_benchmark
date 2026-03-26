"""Generate and optionally submit HTCondor jobs for parallel embedding pooling.

Splits the subject list into N chunks and creates one HTCondor job per chunk.
Each job runs ``linear_probing.py --pool-only --subjects <chunk>``.

After all pooling jobs finish, run the analysis step once::

    python linear_probing.py --skip-regression   # classification only
    python linear_probing.py                      # full analysis

Usage
-----
::

    # Preview the generated submit file (dry-run, default):
    python launch_pool_jobs.py --model CbraMod --n-jobs 10

    # Actually submit:
    python launch_pool_jobs.py --model CbraMod --n-jobs 10 --submit

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
    "/data/project/eeg_foundation/data/benchmark_results"
    "/new_results/LINEAR_PROBING"
)
LOG_DIR = "/data/project/eeg_foundation/logs"
SCRIPT_DIR = op.dirname(op.abspath(__file__))
LINEAR_PROBING = op.join(SCRIPT_DIR, "linear_probing.py")

MODEL_DATA_SUBDIRS = {
    "CbraMod": "CbraMod/multi_layer_embeddings",
    "NeuroLM": "NeuroLM/multi_layer_embeddings",
}


def discover_subjects(data_dir, model):
    """Return sorted list of sub-* directory names for a model."""
    subdir = MODEL_DATA_SUBDIRS.get(model)
    if subdir is None:
        raise ValueError(f"Unknown model {model!r}")
    emb_dir = op.join(data_dir, subdir)
    if not op.isdir(emb_dir):
        raise FileNotFoundError(f"Embedding dir not found: {emb_dir}")
    subjects = sorted(
        d for d in os.listdir(emb_dir)
        if d.startswith("sub-") and op.isdir(op.join(emb_dir, d))
    )
    return subjects


def chunk_list(lst, n_chunks):
    """Split *lst* into *n_chunks* roughly-equal sublists."""
    k = math.ceil(len(lst) / n_chunks)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def build_submit_file(
    model,
    chunks,
    data_dir,
    output_dir,
    request_cpus,
    request_memory,
):
    """Return the text of an HTCondor submit file.

    Uses the ``run_pool.sh`` wrapper (same pattern as ``run_embeddings.sh``)
    so that conda activation and argument passing work correctly on the
    worker nodes.
    """
    run_pool_sh = op.join(SCRIPT_DIR, "run_pool.sh")
    total = sum(len(c) for c in chunks)

    header = textwrap.dedent(f"""\
    # Auto-generated HTCondor submit file for parallel pooling
    # Model: {model}  |  Jobs: {len(chunks)}  |  Total subjects: {total}
    requirements      = (Arch == "ppc64le")

    universe          = vanilla
    getenv            = True

    request_cpus      = {request_cpus}
    request_memory    = {request_memory}

    +MaxRuntime       = 86400

    executable        = /bin/bash
    arguments         = {run_pool_sh} --model $(model) --subjects $(subjects) --data_dir $(data_dir) --output_dir $(output_dir)
    transfer_executable = False

    log    = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).log
    output = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).out
    error  = {LOG_DIR}/pool_{model}_$(Cluster).$(Process).err
    stream_output     = True
    stream_error      = True
    should_transfer_files = NO

    queue model,subjects,data_dir,output_dir from (
    """)

    rows = []
    for i, chunk in enumerate(chunks):
        # Use '+' as subject separator (comma is HTCondor's column delimiter)
        subjects_joined = "+".join(chunk)
        rows.append(f"    {model}, {subjects_joined}, {data_dir}, {output_dir}")

    footer = ")\n"

    return header + "\n".join(rows) + "\n" + footer


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTCondor jobs for parallel embedding pooling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_DATA_SUBDIRS.keys()),
        help="Foundation model to pool.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel HTCondor jobs (default: 10).",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for pooled embeddings.",
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
        help="Memory per job (default: 16GB).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit to HTCondor (default: dry-run).",
    )

    args = parser.parse_args()

    # Discover subjects
    subjects = discover_subjects(args.data_dir, args.model)
    print(f"Found {len(subjects)} subjects for {args.model}")

    n_jobs = min(args.n_jobs, len(subjects))
    chunks = chunk_list(subjects, n_jobs)
    print(f"Splitting into {len(chunks)} jobs "
          f"({len(chunks[0])} subjects/job)")

    # Build submit file
    submit_text = build_submit_file(
        model=args.model,
        chunks=chunks,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        request_cpus=args.request_cpus,
        request_memory=args.request_memory,
    )

    submit_path = op.join(
        LOG_DIR, f"pool_{args.model}_{len(chunks)}jobs.submit"
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(submit_path, "w") as f:
        f.write(submit_text)
    print(f"Submit file written: {submit_path}")

    if args.submit:
        print("Submitting to HTCondor ...")
        result = subprocess.run(
            ["condor_submit", submit_path],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"condor_submit failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nDry-run mode. To submit:\n")
        print(f"  condor_submit {submit_path}")
        print(f"\n  or re-run with --submit")
        print(f"\nAfter all pooling jobs finish, run the analysis:")
        print(f"  python {LINEAR_PROBING}")


if __name__ == "__main__":
    main()
