#!/usr/bin/env python3
"""
Sequential script to run scalars and topographies computation for subjects.

This script processes H5 files to compute scalars and topographies for each subject/session.
It can run sequentially or with optional parallelization within the interactive job.

Usage:
    python run_topos_scalars_subjects.py [--parallelize] [--skip-existing] [--subject ID] [--session NUM]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import multiprocessing as mp
from tqdm import tqdm

# Import compute functions directly
try:
    from compute_scalars import compute_scalars_from_h5
    from compute_topographies import compute_topographies_from_h5
except ImportError as e:
    print(f"Error importing compute functions: {e}")
    print("Make sure this script is run from the markers directory or the directory is in PYTHONPATH")
    sys.exit(1)


# Configure logging
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_worker_logging(log_level: str = 'INFO') -> None:
    """Setup logging for worker processes (simpler to avoid conflicts)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def discover_subjects_sessions(h5_files_dir: Path, logger: logging.Logger) -> List[Tuple[str, str]]:
    """Discover all subject/session combinations with H5 files."""
    subjects_sessions = []
    
    logger.info(f"🔍 Scanning H5 files directory: {h5_files_dir}")
    
    for sub_dir in sorted(h5_files_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        
        subject_id = sub_dir.name.replace("sub-", "")
        logger.debug(f"Found subject directory: {subject_id}")
        
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            
            session_num = ses_dir.name.replace("ses-", "")
            
            original_h5 = ses_dir / "original.h5"
            recon_h5 = ses_dir / "recon.h5"
            
            if original_h5.exists() and recon_h5.exists():
                subjects_sessions.append((subject_id, session_num))
                logger.debug(f"  ✓ Found session {session_num} with both H5 files")
            else:
                logger.debug(f"  ✗ Session {session_num} missing H5 files")
    
    logger.info(f"📊 Found {len(subjects_sessions)} subject/session combinations with H5 files")
    return subjects_sessions


def check_outputs_already_exist(subject_id: str, session_num: str, base_dir: Path, logger: logging.Logger) -> bool:
    """Check if all output files already exist for this subject/session."""
    computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
    
    # Check for original files
    orig_scalars = computed_data_dir / "orig" / f"scalars_sub-{subject_id}_ses-{session_num}.npz"
    orig_topos = computed_data_dir / "orig" / f"topos_sub-{subject_id}_ses-{session_num}.npz"
    
    # Check for recon files  
    recon_scalars = computed_data_dir / "recon" / f"scalars_sub-{subject_id}_ses-{session_num}.npz"
    recon_topos = computed_data_dir / "recon" / f"topos_sub-{subject_id}_ses-{session_num}.npz"
    
    # Check all files exist and are non-empty
    files_exist = [
        orig_scalars.exists() and orig_scalars.stat().st_size > 0,
        orig_topos.exists() and orig_topos.stat().st_size > 0,
        recon_scalars.exists() and recon_scalars.stat().st_size > 0,
        recon_topos.exists() and recon_topos.stat().st_size > 0
    ]
    
    if all(files_exist):
        logger.info(f"⏭️  Skipping sub-{subject_id}/ses-{session_num} - outputs already exist")
        return True
    else:
        missing = []
        if not files_exist[0]: missing.append("original scalars")
        if not files_exist[1]: missing.append("original topographies")
        if not files_exist[2]: missing.append("recon scalars")
        if not files_exist[3]: missing.append("recon topographies")
        
        if len(missing) < 4:
            logger.warning(f"⚠️  Partial results for sub-{subject_id}/ses-{session_num} - missing: {', '.join(missing)}")
        return False


def process_single_h5_file(args: Tuple[str, str, str, str, Path, str]) -> Dict[str, Any]:
    """Process a single H5 file (original or recon) for scalars and topographies.
    
    Args:
        subject_id: Subject ID
        session_num: Session number
        file_type: 'original' or 'recon'
        h5_file_path: Path to H5 file
        output_dir: Output directory for this file type
        log_level: Logging level
        
    Returns:
        Dictionary with results
    """
    subject_id, session_num, file_type, h5_file_path, output_dir, log_level = args
    
    # Setup logging for this process (simpler for multiprocessing)
    setup_worker_logging(log_level)
    logger = logging.getLogger(__name__)
    
    result = {
        'subject_id': subject_id,
        'session_num': session_num,
        'file_type': file_type,
        'success': False,
        'error': None,
        'scalars_file': None,
        'topos_file': None
    }
    
    try:
        # Verify H5 file exists
        if not h5_file_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        scalars_output = output_dir / f"scalars_sub-{subject_id}_ses-{session_num}.npz"
        topos_output = output_dir / f"topos_sub-{subject_id}_ses-{session_num}.npz"
        
        logger.info(f"🚀 Processing {file_type} H5 for sub-{subject_id}/ses-{session_num}")
        
        # Compute scalars
        logger.info(f"📊 Computing scalars...")
        compute_scalars_from_h5(h5_file_path, scalars_output, logger)
        result['scalars_file'] = str(scalars_output)
        logger.info(f"✅ Scalars saved to: {scalars_output}")
        
        # Compute topographies
        logger.info(f"🗺️  Computing topographies...")
        compute_topographies_from_h5(h5_file_path, topos_output, logger)
        result['topos_file'] = str(topos_output)
        logger.info(f"✅ Topographies saved to: {topos_output}")
        
        # Verify output files were created and have content
        if not scalars_output.exists() or scalars_output.stat().st_size == 0:
            raise RuntimeError(f"Scalars output file not created or empty: {scalars_output}")
        if not topos_output.exists() or topos_output.stat().st_size == 0:
            raise RuntimeError(f"Topographies output file not created or empty: {topos_output}")
        
        result['success'] = True
        logger.info(f"🎉 Completed {file_type} for sub-{subject_id}/ses-{session_num}")
        logger.info(f"   Scalars: {scalars_output.stat().st_size} bytes")
        logger.info(f"   Topographies: {topos_output.stat().st_size} bytes")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Failed to process {file_type} for sub-{subject_id}/ses-{session_num}: {e}")
    
    return result


def process_subject_session_sequential(subject_id: str, session_num: str, base_dir: Path, log_level: str, logger: logging.Logger) -> Dict[str, Any]:
    """Process a single subject/session sequentially."""
    logger.info(f"📝 Processing sub-{subject_id}/ses-{session_num} sequentially")
    
    # Define paths
    h5_files_dir = base_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session_num}"
    computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
    
    original_h5 = h5_files_dir / "original.h5"
    recon_h5 = h5_files_dir / "recon.h5"
    
    orig_output_dir = computed_data_dir / "orig"
    recon_output_dir = computed_data_dir / "recon"
    
    results = []
    
    # Process original
    orig_args = (subject_id, session_num, "original", original_h5, orig_output_dir, log_level)
    results.append(process_single_h5_file(orig_args))
    
    # Process recon
    recon_args = (subject_id, session_num, "recon", recon_h5, recon_output_dir, log_level)
    results.append(process_single_h5_file(recon_args))
    
    return results


def process_all_parallel(subjects_sessions: List[Tuple[str, str]], base_dir: Path, log_level: str, n_workers: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process all subjects/sessions in parallel."""
    logger.info(f"🚀 Processing {len(subjects_sessions)} subject/sessions with {n_workers} workers")
    
    # Prepare all arguments (each H5 file is a separate task)
    all_args = []
    for subject_id, session_num in subjects_sessions:
        h5_files_dir = base_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session_num}"
        computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
        
        original_h5 = h5_files_dir / "original.h5"
        recon_h5 = h5_files_dir / "recon.h5"
        
        orig_output_dir = computed_data_dir / "orig"
        recon_output_dir = computed_data_dir / "recon"
        
        all_args.append((subject_id, session_num, "original", original_h5, orig_output_dir, log_level))
        all_args.append((subject_id, session_num, "recon", recon_h5, recon_output_dir, log_level))
    
    # Process in parallel
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_h5_file, all_args),
            total=len(all_args),
            desc="Processing H5 files"
        ))
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sequential scalars and topographies computation for subjects"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS"),
        help="Base directory for MARKERS results"
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Run in parallel using multiprocessing"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count, but recommend 2-4 for large files)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects that already have complete outputs"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process only this subject ID (e.g., 'AA078')"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process only this session (e.g., '01')"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running computation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    base_dir = args.base_dir
    h5_files_dir = base_dir / "h5_files"
    
    if not h5_files_dir.exists():
        logger.error(f"❌ H5 files directory not found: {h5_files_dir}")
        sys.exit(1)
    
    logger.info(f"{'='*80}")
    logger.info("🧠🗺️  SCALARS & TOPOGRAPHIES SEQUENTIAL PROCESSOR")
    logger.info(f"{'='*80}")
    logger.info(f"📂 Base directory: {base_dir}")
    logger.info(f"📁 H5 directory: {h5_files_dir}")
    logger.info(f"🔄 Parallelize: {args.parallelize}")
    if args.parallelize:
        n_workers = args.n_workers or (mp.cpu_count() - 1)
        logger.info(f"⚙️  Workers: {n_workers} (recommended: 2-4 for large files)")
    logger.info(f"⏭️  Skip existing: {args.skip_existing}")
    logger.info(f"🔍 Dry run: {args.dry_run}")
    logger.info(f"🎯 Subject filter: {args.subject}")
    logger.info(f"🎯 Session filter: {args.session}")
    logger.info(f"{'='*80}\n")
    
    # Discover subjects and sessions
    logger.info("🔍 Discovering subjects and sessions...")
    subjects_sessions = discover_subjects_sessions(h5_files_dir, logger)
    
    # Filter by subject/session if specified
    if args.subject or args.session:
        logger.info(f"🎯 Filtering by subject: {args.subject}, session: {args.session}")
        filtered = []
        for subj_id, sess_num in subjects_sessions:
            if args.subject and subj_id != args.subject:
                continue
            if args.session and sess_num != args.session:
                continue
            filtered.append((subj_id, sess_num))
        subjects_sessions = filtered
        logger.info(f"📊 After filtering: {len(subjects_sessions)} subject/session combinations")
    
    if not subjects_sessions:
        logger.error("❌ No subjects/sessions found with H5 files!")
        sys.exit(1)
    
    # Check for existing results
    if args.skip_existing:
        logger.info("⏭️  Checking for existing results...")
        to_process = []
        skipped = []
        for subject_id, session_num in subjects_sessions:
            if check_outputs_already_exist(subject_id, session_num, base_dir, logger):
                skipped.append((subject_id, session_num))
            else:
                to_process.append((subject_id, session_num))
        
        logger.info(f"📊 Summary: {len(skipped)} subjects already have results, {len(to_process)} to process")
        subjects_sessions = to_process
    
    if not subjects_sessions:
        logger.info("✅ All subjects already have complete results. Nothing to do.")
        return
    
    if args.dry_run:
        logger.info("🔍 Dry run mode - showing what would be processed:")
        for subject_id, session_num in subjects_sessions:
            logger.info(f"  Would process: sub-{subject_id}/ses-{session_num}")
            logger.info(f"    Original H5: {h5_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / 'original.h5'}")
            logger.info(f"    Recon H5: {h5_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / 'recon.h5'}")
        logger.info(f"\n📊 Total: {len(subjects_sessions)} subject/session combinations")
        return
    
    logger.info(f"\n🚀 Starting computation for {len(subjects_sessions)} subject/session combinations\n")
    
    start_time = time.time()
    all_results = []
    
    if args.parallelize:
        n_workers = args.n_workers or mp.cpu_count()
        all_results = process_all_parallel(subjects_sessions, base_dir, args.log_level, n_workers, logger)
    else:
        # Sequential processing
        for i, (subject_id, session_num) in enumerate(subjects_sessions, 1):
            logger.info(f"📝 [{i}/{len(subjects_sessions)}] Processing sub-{subject_id}/ses-{session_num}...")
            results = process_subject_session_sequential(subject_id, session_num, base_dir, args.log_level, logger)
            all_results.extend(results)
    
    end_time = time.time()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 PROCESSING SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    logger.info(f"📝 Total files processed: {len(all_results)}")
    logger.info(f"✅ Successful: {len(successful)}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    
    if failed:
        logger.error(f"\n❌ Failed processes:")
        for result in failed:
            logger.error(f"  sub-{result['subject_id']}/ses-{result['session_num']} ({result['file_type']}): {result['error']}")
    
    logger.info(f"\n📂 Expected outputs in: {base_dir}/computed_data/sub-*/ses-*/")
    logger.info(f"{'='*80}\n")
    
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
