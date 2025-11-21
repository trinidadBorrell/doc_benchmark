#!/usr/bin/env python3
"""Compute markers with Junifer for multiple subjects and generate reports.

This script:
1. Finds all .fif files in a specified folder (supports original/recon files)
2. For each subject, runs junifer to compute markers (creates .h5 file)
3. Processes the .h5 file with compute_data.py to generate reports
4. Computes 120 scalars from the .h5 file with compute_scalars.py
5. Saves results to doc_benchmark/results/MARKERS/sub-{ID}_{type}/
6. Optionally keeps or deletes .h5 files after processing

Usage:
    python compute_markers_with_junifer.py --fif_folder /path/to/fif/files \
        [--skip-clustering] [--keep-h5] [--test-mode]

Examples:
    # Test with original/recon files, keep .h5, stop after first successful run
    python compute_markers_with_junifer.py \
        --fif_folder /data/project/eeg_foundation/data/test_data/fif_data_compare \
        --keep-h5 --test-mode
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import yaml
import re


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def find_fif_files(fif_folder):
    """Find all .fif files in the specified folder"""
    fif_path = Path(fif_folder)
    if not fif_path.exists():
        raise FileNotFoundError(f"Folder not found: {fif_folder}")
    
    fif_files = list(fif_path.glob("**/*.fif"))
    if not fif_files:
        raise ValueError(f"No .fif files found in {fif_folder}")
    
    return fif_files


def extract_subject_id(fif_file):
    """Extract subject ID and type from .fif filename
    
    Expects filenames like: sub-001_ses-01_task-lg_acq-01_epo_original.fif
    or sub-001_ses-01_task-lg_acq-01_epo_recon.fif
    
    Returns:
        tuple: (subject_id, file_type) where file_type is 'original' or 'recon' or None
    """
    filename = Path(fif_file).name
    
    # Try to extract subject ID using regex
    match = re.search(r'sub-([A-Za-z0-9]+)', filename)
    subject_id = match.group(1) if match else Path(fif_file).stem
    
    # Extract file type (original or recon)
    file_type = None
    if '_original' in filename:
        file_type = 'original'
    elif '_recon' in filename:
        file_type = 'recon'
    
    return subject_id, file_type


def create_subject_yaml(template_yaml, fif_file, output_h5, logger):
    """Create a subject-specific YAML configuration"""
    # Read the template YAML as text
    with open(template_yaml, 'r') as f:
        yaml_text = f.read()
    
    # Get the paths we need
    fif_path = Path(fif_file).absolute()
    
    # For junifer's PatternDataGrabber, pattern must be relative to datadir
    # Set datadir to the parent directory and pattern to just the filename
    datadir = str(fif_path.parent)
    pattern = fif_path.name  # Just the filename, relative to datadir
    
    # Do simple string replacements instead of parsing/dumping
    # This preserves the exact YAML format without type conversion issues
    yaml_text = yaml_text.replace(
        'uri: icm_complete_features.h5',
        f'uri: {output_h5}'
    )
    
    # Update the datadir line - replace any existing datadir path
    yaml_text = re.sub(
        r'datadir: "[^"]*"',
        f'datadir: "{datadir}"',
        yaml_text
    )
    
    # Update the pattern line - use just the filename (relative pattern)
    yaml_text = re.sub(
        r'pattern: "[^"]*\.fif"',
        f'pattern: "{pattern}"',
        yaml_text
    )
    
    # Create temporary YAML file
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_yaml.write(yaml_text)
    temp_yaml.close()
    
    logger.info(f"Created temporary YAML: {temp_yaml.name}")
    return temp_yaml.name


def run_junifer(yaml_file, logger):
    """Run junifer with the specified YAML configuration"""
    logger.info(f"Running junifer with config: {yaml_file}")
    logger.info("(This may take several minutes... streaming output below)")
    try:
        # Stream output in real-time instead of buffering
        process = subprocess.Popen(
            ["junifer", "run", yaml_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output line by line
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[junifer] {line}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Junifer computation completed")
            return True
        else:
            logger.error(f"❌ Junifer failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Junifer failed with exception: {e}")
        return False


def run_compute_data(h5_file, fif_file, subject_id, output_dir, skip_clustering, task, logger):
    """Run compute_data.py from prediction-report on the generated .h5 file"""
    logger.info(f"Running compute_data.py for subject {subject_id}")
    
    # Path to compute_data.py in the report folder
    compute_data_script = Path(__file__).parent / "report" / "compute_data.py"
    
    if not compute_data_script.exists():
        logger.error(f"compute_data.py not found at {compute_data_script}")
        return False
    
    # Prepare command
    cmd = [
        sys.executable,
        str(compute_data_script),
        "--subject_id", subject_id,
        "--h5_file", str(h5_file),
        "--fif_file", str(fif_file),
        "--output_dir", str(output_dir),
        "--task", task
    ]
    
    if skip_clustering:
        cmd.append("--skip-clustering")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[compute_data] {line}")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Report computation completed")
            return True
        else:
            logger.error(f"❌ compute_data.py failed with return code {return_code}")
            return False
    except Exception as e:
        logger.error(f"compute_data.py failed with exception: {e}")
        return False


def run_compute_scalars(h5_file, subject_id, output_dir, logger):
    """Run compute_scalars.py to extract 120 scalars from the H5 file"""
    logger.info(f"Running compute_scalars.py for subject {subject_id}")
    
    # Path to compute_scalars.py
    compute_scalars_script = Path(__file__).parent / "compute_scalars.py"
    
    if not compute_scalars_script.exists():
        logger.error(f"compute_scalars.py not found at {compute_scalars_script}")
        return False
    
    # Output scalars file
    scalars_file = output_dir / f"scalars_{subject_id}.npz"
    
    # Prepare command
    cmd = [
        sys.executable,
        str(compute_scalars_script),
        "--h5_file", str(h5_file),
        "--output_file", str(scalars_file)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[compute_scalars] {line}")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Scalars computation completed")
            return True
        else:
            logger.error(f"❌ compute_scalars.py failed with return code {return_code}")
            return False
    except Exception as e:
        logger.error(f"compute_scalars.py failed with exception: {e}")
        return False


def main():
    """Main processing loop"""
    parser = argparse.ArgumentParser(
        description="Compute markers with Junifer for multiple subjects"
    )
    parser.add_argument(
        "--fif_folder",
        required=True,
        help="Folder containing .fif files to process"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip cluster permutation tests to speed up computation"
    )
    parser.add_argument(
        "--task",
        default="lg",
        choices=["lg", "rs"],
        help="Paradigm task: 'lg' (Local-Global) or 'rs' (Resting State)"
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Base results directory (default: ../../results/MARKERS relative to script)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Direct output directory (bypasses automatic sub-{ID}_{type} subdirectory creation)"
    )
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="Keep the .h5 file after processing (default: delete after processing)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Stop after successfully processing one file (for testing)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine paths
    script_dir = Path(__file__).parent
    template_yaml = script_dir / "input/icm_non-aggregated_full_markers.yaml"
    
    # Direct output mode (for pipeline integration)
    use_direct_output = args.output_dir is not None
    
    if use_direct_output:
        # Use output directory directly without creating subdirectories
        base_results_dir = Path(args.output_dir)
        base_results_dir.mkdir(parents=True, exist_ok=True)
    elif args.results_dir:
        base_results_dir = Path(args.results_dir)
        base_results_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_results_dir = script_dir.parent.parent / "results" / "new_results" / "MARKERS"
        base_results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("MARKER COMPUTATION WITH JUNIFER")
    logger.info("="*80)
    logger.info(f"FIF folder: {args.fif_folder}")
    logger.info(f"Template YAML: {template_yaml}")
    logger.info(f"Results directory: {base_results_dir}")
    logger.info(f"Skip clustering: {args.skip_clustering}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Keep H5 files: {args.keep_h5}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info("="*80)
    
    if not template_yaml.exists():
        logger.error(f"Template YAML not found: {template_yaml}")
        sys.exit(1)
    
    # Find all .fif files
    try:
        fif_files = find_fif_files(args.fif_folder)
        logger.info(f"Found {len(fif_files)} .fif files to process")
    except Exception as e:
        logger.error(f"Error finding .fif files: {e}")
        sys.exit(1)
    
    # Process each subject
    success_count = 0
    failed_subjects = []
    
    for idx, fif_file in enumerate(fif_files, 1):
        logger.info("")
        logger.info("="*80)
        logger.info(f"Processing file {idx}/{len(fif_files)}: {fif_file.name}")
        logger.info("="*80)
        
        subject_id, file_type = extract_subject_id(fif_file)
        logger.info(f"Subject ID: {subject_id}")
        if file_type:
            logger.info(f"File type: {file_type}")
        
        # Create subject-specific output directory
        if use_direct_output:
            # Use base_results_dir directly (pipeline mode)
            subject_output_dir = base_results_dir
            h5_filename = "icm_complete_features.h5"
        else:
            # Create subdirectories based on subject ID and file type (standalone mode)
            if file_type:
                subject_output_dir = base_results_dir / f"sub-{subject_id}_{file_type}"
                h5_filename = f"junifer_markers_{subject_id}_{file_type}.h5"
            else:
                subject_output_dir = base_results_dir / f"sub-{subject_id}"
                h5_filename = f"junifer_markers_{subject_id}.h5"
            
            subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save H5 file in subject directory
        h5_file = subject_output_dir / h5_filename
        
        # If H5 file already exists, remove it (junifer will create fresh)
        if h5_file.exists():
            h5_file.unlink()
            logger.info(f"Removed existing H5 file: {h5_file}")
        
        temp_yaml = None
        try:
            # Step 1: Create subject-specific YAML
            temp_yaml = create_subject_yaml(template_yaml, fif_file, h5_file, logger)
            
            # Step 2: Run junifer
            if not run_junifer(temp_yaml, logger):
                logger.error(f"❌ Failed to run junifer for subject {subject_id}")
                failed_subjects.append(subject_id)
                continue
            
            # Step 3: Run compute_data.py (generates reports)
            if not run_compute_data(
                h5_file, fif_file, subject_id, subject_output_dir,
                args.skip_clustering, args.task, logger
            ):
                logger.error(f"❌ Failed to run compute_data.py for subject {subject_id}")
                failed_subjects.append(subject_id)
                continue
            
            # Step 4: Compute scalars from H5 file
            if not run_compute_scalars(h5_file, subject_id, subject_output_dir, logger):
                logger.error(f"❌ Failed to run compute_scalars.py for subject {subject_id}")
                failed_subjects.append(subject_id)
                continue
            
            success_count += 1
            file_desc = f"{subject_id} ({file_type})" if file_type else subject_id
            logger.info(f"✅ Successfully processed subject {file_desc}")
            logger.info(f"✅ Results saved to: {subject_output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error processing subject {subject_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_subjects.append(subject_id)
        
        finally:
            # Cleanup temporary YAML
            if temp_yaml and Path(temp_yaml).exists():
                Path(temp_yaml).unlink()
                logger.debug(f"Removed temporary YAML: {temp_yaml}")
            
            # Handle H5 file based on keep-h5 flag
            if h5_file.exists():
                if args.keep_h5:
                    logger.info(f"H5 file saved at: {h5_file}")
                else:
                    h5_file.unlink()
                    logger.info(f"H5 file deleted (use --keep-h5 to preserve): {h5_file}")
        
        # If in test mode, stop after first successful processing
        if args.test_mode and success_count > 0:
            logger.info("")
            logger.info("="*80)
            logger.info("TEST MODE: Stopping after first successful run")
            file_desc = f"{subject_id} ({file_type})" if file_type else subject_id
            logger.info(f"Successfully processed: {file_desc}")
            logger.info(f"Results directory: {subject_output_dir}")
            logger.info("="*80)
            break
    
    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total subjects: {len(fif_files)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {len(failed_subjects)}")
    
    if failed_subjects:
        logger.warning(f"Failed subjects: {', '.join(failed_subjects)}")
    
    logger.info("="*80)
    
    if success_count == len(fif_files):
        logger.info("✅ All subjects processed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some subjects failed processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
