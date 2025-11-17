#!/usr/bin/env python3
"""
Compute markers with Junifer for multiple subjects and generate reports.

This script:
1. Finds all .fif files in a specified folder
2. For each subject, runs junifer to compute markers
3. Immediately processes the resulting .h5 file with compute_data.py
4. Saves results to doc_benchmark/results/MARKERS/sub-{ID}/

Usage:
    python compute_markers_with_junifer.py --fif_folder /path/to/fif/files [--skip-clustering]
"""

import argparse
import logging
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
    """Extract subject ID from .fif filename
    
    Expects filenames like: sub-001_ses-01_task-lg_acq-01_epo.fif
    or similar patterns with sub-XXX
    """
    filename = Path(fif_file).name
    # Try to extract subject ID using regex
    match = re.search(r'sub-([A-Za-z0-9]+)', filename)
    if match:
        return match.group(1)
    else:
        # Fallback: use the filename stem
        return Path(fif_file).stem


def create_subject_yaml(template_yaml, fif_file, output_h5, logger):
    """Create a subject-specific YAML configuration"""
    with open(template_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the pattern to point to the specific .fif file
    fif_path = Path(fif_file).absolute()
    config['datagrabber']['patterns']['EEG']['pattern'] = str(fif_path)
    config['datagrabber']['datadir'] = str(fif_path.parent)
    
    # Update storage to use the specified output .h5 file
    config['storage']['uri'] = str(output_h5)
    
    # Create temporary YAML file
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_yaml, default_flow_style=False)
    temp_yaml.close()
    
    logger.info(f"Created temporary YAML: {temp_yaml.name}")
    return temp_yaml.name


def install_junifer(logger):
    """Install junifer packages if needed"""
    logger.info("Checking junifer installation...")
    try:
        subprocess.run(
            ["pip", "install", "junifer_eeg", "junifer"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Junifer packages installed/verified")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install junifer: {e.stderr}")
        raise


def run_junifer(yaml_file, logger):
    """Run junifer with the specified YAML configuration"""
    logger.info(f"Running junifer with config: {yaml_file}")
    try:
        result = subprocess.run(
            ["junifer", "run", yaml_file],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Junifer computation completed")
        logger.debug(f"Junifer output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Junifer failed: {e.stderr}")
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
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Report computation completed")
        logger.debug(f"compute_data.py output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"compute_data.py failed: {e.stderr}")
        logger.error(f"stdout: {e.stdout}")
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
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine paths
    script_dir = Path(__file__).parent
    template_yaml = script_dir / "icm_complete_individual_markers.yaml"
    
    if args.results_dir:
        base_results_dir = Path(args.results_dir)
    else:
        base_results_dir = script_dir.parent.parent.parent / "results" / "MARKERS"
    
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("MARKER COMPUTATION WITH JUNIFER")
    logger.info("="*80)
    logger.info(f"FIF folder: {args.fif_folder}")
    logger.info(f"Template YAML: {template_yaml}")
    logger.info(f"Results directory: {base_results_dir}")
    logger.info(f"Skip clustering: {args.skip_clustering}")
    logger.info(f"Task: {args.task}")
    logger.info("="*80)
    
    if not template_yaml.exists():
        logger.error(f"Template YAML not found: {template_yaml}")
        sys.exit(1)
    
    # Install junifer
    try:
        install_junifer(logger)
    except Exception as e:
        logger.error(f"Failed to install junifer: {e}")
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
        
        subject_id = extract_subject_id(fif_file)
        logger.info(f"Subject ID: {subject_id}")
        
        # Create subject-specific output directory
        subject_output_dir = base_results_dir / f"sub-{subject_id}"
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary .h5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_h5:
            h5_file = Path(temp_h5.name)
        
        temp_yaml = None
        try:
            # Step 1: Create subject-specific YAML
            temp_yaml = create_subject_yaml(template_yaml, fif_file, h5_file, logger)
            
            # Step 2: Run junifer
            if not run_junifer(temp_yaml, logger):
                logger.error(f"❌ Failed to run junifer for subject {subject_id}")
                failed_subjects.append(subject_id)
                continue
            
            # Step 3: Run compute_data.py
            if not run_compute_data(
                h5_file, fif_file, subject_id, subject_output_dir,
                args.skip_clustering, args.task, logger
            ):
                logger.error(f"❌ Failed to run compute_data.py for subject {subject_id}")
                failed_subjects.append(subject_id)
                continue
            
            success_count += 1
            logger.info(f"✅ Successfully processed subject {subject_id}")
            logger.info(f"✅ Results saved to: {subject_output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error processing subject {subject_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_subjects.append(subject_id)
        
        finally:
            # Cleanup temporary files
            if temp_yaml and Path(temp_yaml).exists():
                Path(temp_yaml).unlink()
                logger.debug(f"Removed temporary YAML: {temp_yaml}")
            
            if h5_file.exists():
                h5_file.unlink()
                logger.debug(f"Removed temporary H5: {h5_file}")
    
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
