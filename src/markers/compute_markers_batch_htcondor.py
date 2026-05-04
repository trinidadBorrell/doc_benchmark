#!/usr/bin/env python3
"""Batch process markers with Junifer using HTCondor for parallel computation.

This script:
1. Discovers all .fif files in the specified folder
2. Creates batches of N subject-sessions (default 40)
3. For each batch, creates a YAML config and submits to HTCondor via junifer queue
4. Monitors job completion
5. Processes results with compute_data.py and compute_scalars.py
6. Saves results to doc_benchmark/results/MARKERS/sub-{ID}_{type}/

Usage:
    python compute_markers_batch_htcondor.py --fif_folder /path/to/fif/files \
        [--batch-size 40] [--skip-clustering] [--keep-h5]
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from ruamel.yaml import YAML
from pathlib import Path


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def find_all_fif_files(fif_folder, task="rs"):
    """Find all .fif files and organize by subject-session-type"""
    fif_path = Path(fif_folder)
    if not fif_path.exists():
        raise FileNotFoundError(f"Folder not found: {fif_folder}")

    fif_files = list(fif_path.glob("**/*.fif"))
    if not fif_files:
        raise ValueError(f"No .fif files found in {fif_folder}")

    # Organize files by (subject, session, type)
    organized = []
    for fif_file in fif_files:
        filename = fif_file.name
        # Extract: sub-AA048_ses-01_task-lg_acq-01_epo_original.fif
        match = re.match(
            rf"(sub-[^_]+)_(ses-\d+)_task-{task}_acq-\d+_epo_(original|recon)\.fif",
            filename,
        )
        if match:
            subject = match.group(1)
            session = match.group(2)
            file_type = match.group(3)
            organized.append(
                {
                    "subject": subject,
                    "session": session,
                    "type": file_type,
                    "path": fif_file,
                }
            )

    return organized


def create_individual_yaml(template_yaml, item_info, output_h5_file, yaml_dir, logger):
    """Create individual YAML configuration for one subject-session

    Args:
        template_yaml: Path to template YAML file
        item_info: Dict with 'subject', 'session', 'type', 'path'
        output_h5_file: Path where H5 file should be saved
        yaml_dir: Directory to save YAML file
        logger: Logger instance

    Returns:
        Path to created YAML file
    """
    # Read template using ruamel.yaml to preserve exact format
    ryaml = YAML()
    ryaml.preserve_quotes = True
    ryaml.default_flow_style = False

    with open(template_yaml, "r") as f:
        config = ryaml.load(f)

    subject = item_info["subject"]
    session = item_info["session"]
    file_type = item_info["type"]
    fif_path = item_info["path"]

    # Create element ID
    element_id = f"{subject}_{session}_{file_type}"

    # Update datagrabber to point directly to this file
    config["datagrabber"]["patterns"]["EEG"]["pattern"] = str(fif_path.name)
    config["datagrabber"]["datadir"] = str(fif_path.parent)
    config["datagrabber"]["replacements"] = []

    # Remove elements section entirely - process single file directly
    if "elements" in config:
        del config["elements"]

    # Remove queue section - we'll create our own HTCondor submit files
    if "queue" in config:
        del config["queue"]

    # Update storage
    config["storage"]["uri"] = str(output_h5_file)
    config["storage"]["single_output"] = True

    # Write YAML using ruamel which preserves the exact structure
    yaml_file = yaml_dir / f"{element_id}.yaml"
    with open(yaml_file, "w") as f:
        ryaml.dump(config, f)

    return yaml_file


def create_htcondor_submit_file(yaml_file, venv_path, logger):
    """Create custom HTCondor submit file for a junifer run job

    Returns:
        Path to created submit file
    """
    logger.info(f"Creating HTCondor submit file for: {yaml_file}")

    yaml_path = Path(yaml_file)
    job_name = yaml_path.stem
    job_dir = yaml_path.parent / "htcondor_jobs" / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create a wrapper script to properly activate conda and run junifer
    wrapper_script = job_dir / "run_job.sh"

    # Extract the H5 file path to create parent directories
    h5_file_path = None
    with open(yaml_file, "r") as f:
        ryaml = YAML()
        config = ryaml.load(f)
        if "storage" in config and "uri" in config["storage"]:
            h5_file_path = config["storage"]["uri"]

    wrapper_content = f"""#!/usr/bin/env zsh
# Wrapper script to run junifer with conda

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh

# Activate base environment
conda activate base

# Create parent directory for H5 file
mkdir -p "$(dirname "{h5_file_path or ""}")"

# Run junifer
junifer run {yaml_file} --verbose info
"""

    with open(wrapper_script, "w") as f:
        f.write(wrapper_content)

    # Make wrapper executable
    import os

    os.chmod(wrapper_script, 0o755)

    # Create submit file content
    submit_content = f"""# HTCondor submit file for {job_name}

universe = vanilla
executable = {wrapper_script}
arguments = 

# Resource requirements
request_cpus = 1
request_memory = 8GB
request_disk = 2GB

# Logs
log = {job_dir}/job.log
output = {job_dir}/job.out
error = {job_dir}/job.err

# Environment
environment = "HOME={os.environ.get("HOME", "")}"

# Transfer files
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# Job completion
notification = Never

queue 1
"""

    submit_file = job_dir / f"{job_name}.submit"
    with open(submit_file, "w") as f:
        f.write(submit_content)

    logger.info(f"Created submit file: {submit_file}")
    return submit_file


def submit_htcondor_job(submit_file, logger):
    """Submit HTCondor job using condor_submit

    Returns:
        (success, cluster_id)
    """
    logger.info(f"Submitting HTCondor job: {submit_file}")

    try:
        result = subprocess.run(
            ["condor_submit", str(submit_file)], capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.error(f"condor_submit failed: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            return False, None

        logger.info("HTCondor job submitted successfully")
        logger.info(result.stdout.strip())

        # Extract cluster ID from output
        # Output format: "1 job(s) submitted to cluster 12345."
        import re

        match = re.search(r"submitted to cluster (\d+)", result.stdout)
        cluster_id = match.group(1) if match else None

        return True, cluster_id

    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        return False, None


def monitor_htcondor_jobs(
    cluster_ids, job_names, logger, check_interval=30, yaml_dir=None
):
    """Monitor multiple HTCondor jobs

    Args:
        cluster_ids: List of cluster IDs to monitor
        job_names: List of job names (for logging)
        logger: Logger instance
        check_interval: Seconds between status checks
        yaml_dir: Directory containing YAML files (used to find job logs)

    Returns:
        Dictionary mapping job names to success status
    """
    logger.info(f"Monitoring {len(cluster_ids)} HTCondor jobs...")
    logger.info(f"Checking every {check_interval} seconds...")

    job_status = {name: "running" for name in job_names}

    while True:
        running_count = 0
        completed_count = 0
        failed_count = 0

        for cluster_id, job_name in zip(cluster_ids, job_names):
            if job_status[job_name] != "running":
                if job_status[job_name] == "completed":
                    completed_count += 1
                else:
                    failed_count += 1
                continue

            # Check job status using condor_q
            result = subprocess.run(
                ["condor_q", cluster_id, "-af", "JobStatus"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Job no longer in queue - check if it completed successfully
                # Query condor_history
                history_result = subprocess.run(
                    ["condor_history", cluster_id, "-af", "ExitCode"],
                    capture_output=True,
                    text=True,
                )

                if history_result.returncode == 0 and history_result.stdout.strip():
                    exit_code = history_result.stdout.strip()
                    logger.debug(
                        f"Exit code for {job_name}: '{exit_code}' (repr: {repr(exit_code)})"
                    )
                    if exit_code == "0":
                        job_status[job_name] = "completed"
                        completed_count += 1
                        logger.info(f"✅ {job_name} completed successfully")
                    else:
                        job_status[job_name] = "failed"
                        failed_count += 1
                        logger.error(f"❌ {job_name} failed with exit code {exit_code}")
                else:
                    # condor_history doesn't have the job - check job log as fallback
                    logger.warning(
                        f"Could not get exit code from condor_history for {job_name}: returncode={history_result.returncode}, stdout='{history_result.stdout}', stderr='{history_result.stderr}'"
                    )

                    # Try to check job log file
                    job_completed = None  # None means unknown
                    if yaml_dir:
                        job_log = (
                            Path(yaml_dir) / "htcondor_jobs" / job_name / "job.log"
                        )
                        if job_log.exists():
                            try:
                                with open(job_log, "r") as f:
                                    log_content = f.read()
                                    # Check for "Job terminated" and "return value 0"
                                    # HTCondor log format: "Normal termination (return value 0)"
                                    if "return value 0" in log_content:
                                        logger.info(
                                            f"Found successful completion in job log for {job_name}"
                                        )
                                        job_completed = True
                                    elif re.search(r"return value [1-9]", log_content):
                                        # Job terminated with non-zero exit code
                                        logger.info(
                                            f"Found failed completion in job log for {job_name}"
                                        )
                                        job_completed = False
                                    elif "Job terminated" in log_content:
                                        # Job terminated but couldn't parse return value - check H5 file
                                        logger.warning(
                                            f"Job terminated but unclear return value for {job_name}, checking output file"
                                        )
                                        job_completed = None
                                    else:
                                        logger.warning(
                                            f"Job log exists but no clear completion status for {job_name}"
                                        )
                            except Exception as e:
                                logger.warning(
                                    f"Could not read job log for {job_name}: {e}"
                                )
                        else:
                            logger.warning(f"Job log not found at {job_log}")

                    # If still unknown, check if the H5 output file exists as final fallback
                    if job_completed is None and yaml_dir:
                        yaml_file = Path(yaml_dir) / f"{job_name}.yaml"
                        if yaml_file.exists():
                            try:
                                ryaml_check = YAML()
                                with open(yaml_file, "r") as f:
                                    config_check = ryaml_check.load(f)
                                    h5_path = config_check.get("storage", {}).get("uri")
                                    if h5_path and Path(h5_path).exists():
                                        logger.info(
                                            f"H5 output file exists for {job_name}, assuming success"
                                        )
                                        job_completed = True
                            except Exception as e:
                                logger.warning(
                                    f"Could not check H5 file for {job_name}: {e}"
                                )

                    if job_completed is True:
                        job_status[job_name] = "completed"
                        completed_count += 1
                        logger.info(
                            f"✅ {job_name} completed successfully (verified from log/output)"
                        )
                    elif job_completed is False:
                        job_status[job_name] = "failed"
                        failed_count += 1
                        logger.error(f"❌ {job_name} failed (non-zero exit code)")
                    else:
                        # job_completed is None - truly unknown
                        job_status[job_name] = "failed"
                        failed_count += 1
                        logger.error(
                            f"❌ {job_name} failed (could not verify completion)"
                        )
            else:
                running_count += 1

        logger.info(
            f"Progress: {completed_count} completed, {failed_count} failed, {running_count} running"
        )

        if running_count == 0:
            break

        time.sleep(check_interval)

    return {name: (status == "completed") for name, status in job_status.items()}


def monitor_multiple_dags_old(dag_files, logger, check_interval=30):
    """OLD Monitor multiple DAG completions - DEPRECATED

    Args:
        dag_files: List of DAG file paths
        logger: Logger instance
        check_interval: Seconds between checks

    Returns:
        Dict mapping dag_file to success status (True/False)
    """
    logger.info(f"Monitoring {len(dag_files)} DAG jobs...")
    logger.info(f"Checking every {check_interval} seconds...")

    status = {dag: "running" for dag in dag_files}

    while True:
        completed = 0
        failed = 0
        running = 0

        for dag_file in dag_files:
            if status[dag_file] in ["success", "failed"]:
                if status[dag_file] == "success":
                    completed += 1
                else:
                    failed += 1
                continue

            dagman_out = Path(dag_file).with_suffix(".dag.dagman.out")

            if not dagman_out.exists():
                running += 1
                continue

            # Read last lines
            try:
                with open(dagman_out, "r") as f:
                    lines = f.readlines()
            except:
                running += 1
                continue

            # Check for completion
            for line in reversed(lines[-50:]):
                if "EXITING WITH STATUS 0" in line:
                    status[dag_file] = "success"
                    completed += 1
                    logger.info(f"✅ {Path(dag_file).stem} completed")
                    break
                elif "EXITING WITH STATUS" in line:
                    status[dag_file] = "failed"
                    failed += 1
                    logger.error(f"❌ {Path(dag_file).stem} failed")
                    break
            else:
                running += 1

        logger.info(
            f"Progress: {completed} completed, {failed} failed, {running} running"
        )

        if running == 0:
            # All done
            break

        time.sleep(check_interval)

    # Return results
    results = {}
    for dag_file in dag_files:
        results[dag_file] = status[dag_file] == "success"

    return results


def process_individual_result(
    h5_file, item_info, output_base_dir, skip_clustering, task, logger
):
    """Process results from individual H5 file using compute_data.py and compute_scalars.py"""

    if not h5_file.exists():
        logger.error(f"H5 file not found: {h5_file}")
        return False

    # Import processing scripts
    compute_data_script = Path(__file__).parent / "report" / "compute_data.py"
    compute_scalars_script = Path(__file__).parent / "compute_scalars.py"

    if not compute_data_script.exists():
        logger.error(f"compute_data.py not found at {compute_data_script}")
        return False

    if not compute_scalars_script.exists():
        logger.error(f"compute_scalars.py not found at {compute_scalars_script}")
        return False

    subject = item_info["subject"]
    session = item_info["session"]
    file_type = item_info["type"]
    fif_file = item_info["path"]

    # Create output directory matching compute_markers_with_junifer.py format
    subject_id = subject.replace("sub-", "")
    subject_output_dir = output_base_dir / f"sub-{subject_id}_{file_type}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Run compute_data.py
    logger.info(f"Running compute_data.py for {subject}_{session}_{file_type}")
    cmd = [
        sys.executable,
        str(compute_data_script),
        "--subject_id",
        subject_id,
        "--h5_file",
        str(h5_file),
        "--fif_file",
        str(fif_file),
        "--output_dir",
        str(subject_output_dir),
        "--task",
        task,
    ]

    if skip_clustering:
        cmd.append("--skip-clustering")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"compute_data.py failed: {result.stderr}")
        return False

    logger.info("✅ compute_data.py completed")

    # Run compute_scalars.py
    scalars_file = subject_output_dir / f"scalars_{subject_id}.npz"
    cmd = [
        sys.executable,
        str(compute_scalars_script),
        "--h5_file",
        str(h5_file),
        "--output_file",
        str(scalars_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"compute_scalars.py failed: {result.stderr}")
        return False

    logger.info("✅ compute_scalars.py completed")
    return True


def main():
    """Main processing loop"""
    parser = argparse.ArgumentParser(
        description="Batch process markers with Junifer using HTCondor"
    )
    parser.add_argument(
        "--fif_folder", required=True, help="Folder containing .fif files to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of subject-sessions to process per batch (default: 20)",
    )
    parser.add_argument(
        "--template-yaml",
        default="input/icm_non-aggregated_full_markers_jobs_resting_state.yaml",
        help="Path to template YAML file",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Base results directory (default: ../../results/MARKERS relative to script)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip cluster permutation tests to speed up computation",
    )
    parser.add_argument(
        "--task",
        default="rs",
        choices=["lg", "rs"],
        help="Paradigm task: 'lg' (Local-Global) or 'rs' (Resting State)",
    )
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="Keep the batch H5 files after processing",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between job status checks (default: 60)",
    )

    args = parser.parse_args()

    logger = setup_logging()

    # Setup paths
    script_dir = Path(__file__).parent
    template_yaml = script_dir / args.template_yaml

    if not template_yaml.exists():
        logger.error(f"Template YAML not found: {template_yaml}")
        sys.exit(1)

    if args.results_dir:
        base_results_dir = Path(args.results_dir)
    else:
        base_results_dir = (
            script_dir.parent.parent / "results" / "new_results" / "MARKERS"
        )

    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Create batch working directory
    batch_work_dir = script_dir / "batch_jobs"
    batch_work_dir.mkdir(parents=True, exist_ok=True)

    # Create H5 output directory in the main results location
    h5_output_dir = base_results_dir / "h5_files"
    h5_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BATCH MARKER COMPUTATION WITH JUNIFER + HTCONDOR")
    logger.info("=" * 80)
    logger.info(f"FIF folder: {args.fif_folder}")
    logger.info(f"Template YAML: {template_yaml}")
    logger.info(f"Results directory: {base_results_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip clustering: {args.skip_clustering}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Keep H5 files: {args.keep_h5}")
    logger.info("=" * 80)

    # Find all FIF files
    try:
        all_files = find_all_fif_files(args.fif_folder, args.task)
        logger.info(f"Found {len(all_files)} .fif files")
    except Exception as e:
        logger.error(f"Error finding .fif files: {e}")
        sys.exit(1)

    # Filter out files where finished.txt already exists
    files_to_process = []
    skipped_count = 0

    for item in all_files:
        subject = item["subject"]
        session = item["session"]

        # Check for finished.txt marker
        finish_marker = os.path.join(h5_output_dir, subject, session, "finished.txt")
        if os.path.exists(finish_marker):
            logger.info(f"⏭️  Skipping {subject}/{session} - finished.txt marker found")
            skipped_count += 1
        else:
            files_to_process.append(item)

    logger.info(f"Skipped {skipped_count} files with existing finished.txt markers")
    logger.info(f"Will process {len(files_to_process)} .fif files")

    # If no files to process, exit early
    if len(files_to_process) == 0:
        logger.info(
            "✅ All files already have finished.txt markers - nothing to process"
        )
        sys.exit(0)

    # Keep track of total for summary
    total_files = len(all_files)

    # Process in batches
    total_batches = (len(files_to_process) + args.batch_size - 1) // args.batch_size
    logger.info(f"Will process in {total_batches} batch(es)")

    success_count = 0
    failed_batches = []

    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(files_to_process))
        batch_items = files_to_process[start_idx:end_idx]

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"BATCH {batch_idx + 1}/{total_batches}")
        logger.info(
            f"Processing items {start_idx + 1}-{end_idx} of {len(files_to_process)}"
        )
        logger.info("=" * 80)

        # Create batch YAML directory
        batch_yaml_dir = batch_work_dir / f"batch_{batch_idx:03d}_yamls"
        batch_yaml_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create individual YAMLs for each item
            logger.info(f"Creating {len(batch_items)} individual YAML files...")
            yaml_files = []
            h5_files = []

            for item in batch_items:
                element_id = f"{item['subject']}_{item['session']}_{item['type']}"

                # Create hierarchical directory structure: sub-{ID}/ses-{num}/
                subject_dir = h5_output_dir / item["subject"]
                session_dir = subject_dir / item["session"]
                session_dir.mkdir(parents=True, exist_ok=True)

                h5_file = session_dir / f"{item['type']}.h5"

                yaml_file = create_individual_yaml(
                    template_yaml, item, h5_file, batch_yaml_dir, logger
                )
                yaml_files.append(yaml_file)
                h5_files.append(h5_file)

            logger.info(f"Created {len(yaml_files)} YAML files")

            # Create HTCondor submit files and submit jobs
            logger.info(f"Submitting {len(yaml_files)} jobs to HTCondor...")
            cluster_ids = []
            job_names = []
            venv_path = script_dir.parent.parent / ".venv_n"  # Path to venv

            for yaml_file in yaml_files:
                # Create submit file
                submit_file = create_htcondor_submit_file(yaml_file, venv_path, logger)

                # Submit to HTCondor
                success, cluster_id = submit_htcondor_job(submit_file, logger)
                if not success:
                    logger.error(f"❌ Failed to submit {yaml_file.name}")
                else:
                    cluster_ids.append(cluster_id)
                    job_names.append(yaml_file.stem)

            logger.info(f"Successfully submitted {len(cluster_ids)} jobs")

            if len(cluster_ids) == 0:
                logger.error(f"❌ No jobs submitted for batch {batch_idx + 1}")
                failed_batches.append(batch_idx + 1)
                continue

            # Monitor all jobs
            logger.info(f"Waiting for {len(cluster_ids)} jobs to complete...")
            job_results = monitor_htcondor_jobs(
                cluster_ids,
                job_names,
                logger,
                args.check_interval,
                yaml_dir=batch_yaml_dir,
            )

            # Process results for successful jobs
            logger.info(f"Processing results for batch {batch_idx + 1}...")
            batch_success_count = 0

            for h5_file, item in zip(h5_files, batch_items):
                job_name = f"{item['subject']}_{item['session']}_{item['type']}"
                if not job_results.get(job_name, False):
                    logger.warning(f"Skipping {h5_file.name} - job failed")
                    continue

                # Process this result
                if process_individual_result(
                    h5_file,
                    item,
                    base_results_dir,
                    args.skip_clustering,
                    args.task,
                    logger,
                ):
                    batch_success_count += 1

                    # Cleanup H5 file if not keeping
                    if not args.keep_h5 and h5_file.exists():
                        h5_file.unlink()

            success_count += batch_success_count
            logger.info(
                f"✅ Batch {batch_idx + 1}: {batch_success_count}/{len(batch_items)} completed successfully"
            )

            # Cleanup YAML files
            import shutil

            if batch_yaml_dir.exists():
                shutil.rmtree(batch_yaml_dir)

        except Exception as e:
            logger.error(f"❌ Error processing batch {batch_idx + 1}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            failed_batches.append(batch_idx + 1)

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Skipped (finished.txt exists): {skipped_count}")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed batches: {len(failed_batches)}")

    if failed_batches:
        logger.warning(f"Failed batch numbers: {', '.join(map(str, failed_batches))}")

    logger.info("=" * 80)

    if success_count == len(files_to_process):
        logger.info("✅ All files processed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some batches failed processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
