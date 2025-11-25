#!/usr/bin/env python3
"""
Sequential script to run compute_data and generate_plots for subjects.

This script processes H5 + FIF files to compute analysis data and generate reports.
It can run sequentially or with optional parallelization for compute_data step.

Usage:
    python run_compute_data_plots_subjects.py [--parallelize] [--skip-existing] [--skip-plots] [--subject ID] [--session NUM]
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
# Add script directory and report directory to Python path for imports
script_dir = Path(__file__).parent
report_dir = script_dir / "report"

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(report_dir) not in sys.path:
    sys.path.insert(0, str(report_dir))

try:
    from report.compute_data import main as compute_data_main
    from report.generate_plots import main as generate_plots_main
except ImportError as e:
    print(f"Error importing report functions: {e}")
    print(f"Script directory: {script_dir}")
    print(f"Report directory: {report_dir}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    print("Make sure this script is run from the markers directory and report_modules is in report/")
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


def discover_subjects_sessions(h5_files_dir: Path, fif_files_dir: Path, logger: logging.Logger) -> List[Tuple[str, str]]:
    """Discover all subject/session combinations with both H5 and FIF files."""
    subjects_sessions = []
    
    logger.info(f"🔍 Scanning H5 files directory: {h5_files_dir}")
    logger.info(f"🔍 Scanning FIF files directory: {fif_files_dir}")
    
    # Get all H5 subject/sessions
    h5_combinations = set()
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
                h5_combinations.add((subject_id, session_num))
                logger.debug(f"  ✓ Found session {session_num} with both H5 files")
            else:
                logger.debug(f"  ✗ Session {session_num} missing H5 files")
    
    # Check which have corresponding FIF files (more flexible discovery)
    for subject_id, session_num in sorted(h5_combinations):
        fif_dir = fif_files_dir / f"sub-{subject_id}" / f"ses-{session_num}"
        
        if not fif_dir.exists():
            logger.warning(f"✗ Skipping sub-{subject_id}/ses-{session_num} - FIF directory not found: {fif_dir}")
            continue
        
        # Look for FIF files with flexible naming patterns
        original_fif_patterns = [
            f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_original.fif",
            f"sub-{subject_id}_ses-{session_num}_task-lg_epo_original.fif",
            f"sub-{subject_id}_ses-{session_num}_epo_original.fif",
            f"*original*.fif"
        ]
        
        recon_fif_patterns = [
            f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_recon.fif",
            f"sub-{subject_id}_ses-{session_num}_task-lg_epo_recon.fif",
            f"sub-{subject_id}_ses-{session_num}_epo_recon.fif",
            f"*recon*.fif"
        ]
        
        original_fif = None
        recon_fif = None
        
        # Find original FIF
        for pattern in original_fif_patterns:
            matches = list(fif_dir.glob(pattern))
            if matches:
                original_fif = matches[0]
                break
        
        # Find recon FIF
        for pattern in recon_fif_patterns:
            matches = list(fif_dir.glob(pattern))
            if matches:
                recon_fif = matches[0]
                break
        
        if original_fif and recon_fif:
            subjects_sessions.append((subject_id, session_num))
            logger.info(f"✓ Found: sub-{subject_id}/ses-{session_num} (H5 + FIF)")
            logger.debug(f"    Original FIF: {original_fif.name}")
            logger.debug(f"    Recon FIF: {recon_fif.name}")
        else:
            logger.warning(f"✗ Skipping sub-{subject_id}/ses-{session_num} - missing FIF files")
            if not original_fif:
                logger.warning(f"    No original FIF found in {fif_dir}")
            if not recon_fif:
                logger.warning(f"    No recon FIF found in {fif_dir}")
    
    logger.info(f"📊 Found {len(subjects_sessions)} subject/session combinations with complete data")
    return subjects_sessions


def get_expected_pickle_files() -> List[str]:
    """Get list of expected pickle files from compute_data.py."""
    return [
        "diagnostic_gfp.pkl",
        "local_effect_gfp.pkl", 
        "local_effect_contrast.pkl",
        "local_cluster_test.pkl",
        "global_effect_gfp.pkl",
        "global_effect_contrast.pkl", 
        "global_cluster_test.pkl",
        "cnv_computed_data.pkl",
        "spectral_bands_normalized.pkl",
        "spectral_absolute_power.pkl",
        "spectral_summaries.pkl",
        "wsmi_bands_topo.pkl",
        "mutual_info_topo.pkl",
        "permutation_entropy_bands.pkl",
        "kolmogorov_complexity.pkl",
        # info_theory_*.pkl files are dynamic, check separately
        "prediction_results.pkl"
    ]


def check_computed_data_already_done(subject_id: str, session_num: str, base_dir: Path, logger: logging.Logger) -> bool:
    """Check if all computed data pickle files already exist for this subject/session."""
    computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
    
    # Check for original and recon directories
    orig_dir = computed_data_dir / "orig"
    recon_dir = computed_data_dir / "recon"
    
    if not orig_dir.exists() or not recon_dir.exists():
        return False
    
    expected_files = get_expected_pickle_files()
    
    # Check original files
    orig_files_exist = []
    for pickle_file in expected_files:
        orig_path = orig_dir / pickle_file
        exists = orig_path.exists() and orig_path.stat().st_size > 0
        orig_files_exist.append(exists)
        if not exists:
            logger.debug(f"    Missing original: {pickle_file}")
    
    # Check recon files  
    recon_files_exist = []
    for pickle_file in expected_files:
        recon_path = recon_dir / pickle_file
        exists = recon_path.exists() and recon_path.stat().st_size > 0
        recon_files_exist.append(exists)
        if not exists:
            logger.debug(f"    Missing recon: {pickle_file}")
    
    # Check for info_theory_*.pkl files (dynamic)
    orig_info_theory = list(orig_dir.glob("info_theory_*.pkl"))
    recon_info_theory = list(recon_dir.glob("info_theory_*.pkl"))
    
    if len(orig_info_theory) == 0 or len(recon_info_theory) == 0:
        logger.debug(f"    Missing info_theory_*.pkl files")
        return False
    
    # Check if all expected files exist (allow prediction_results.pkl to be missing)
    orig_complete = all(orig_files_exist[:-1])  # Exclude prediction_results.pkl
    recon_complete = all(recon_files_exist[:-1])  # Exclude prediction_results.pkl
    
    if orig_complete and recon_complete:
        logger.info(f"⏭️  Skipping sub-{subject_id}/ses-{session_num} - computed data already exists")
        return True
    else:
        missing_orig = sum(1 for x in orig_files_exist[:-1] if not x)
        missing_recon = sum(1 for x in recon_files_exist[:-1] if not x)
        if missing_orig < len(expected_files)-1 or missing_recon < len(expected_files)-1:
            logger.warning(f"⚠️  Partial computed data for sub-{subject_id}/ses-{session_num} - missing {missing_orig} original, {missing_recon} recon files")
        return False


def check_report_already_done(subject_id: str, session_num: str, base_dir: Path, logger: logging.Logger) -> bool:
    """Check if HTML report already exists for this subject/session."""
    reports_dir = base_dir / "reports" / f"sub-{subject_id}" / f"ses-{session_num}"
    report_file = reports_dir / f"sub-{subject_id}_ses-{session_num}_report_comparison.html"
    
    if report_file.exists() and report_file.stat().st_size > 0:
        logger.info(f"⏭️  Skipping sub-{subject_id}/ses-{session_num} - report already exists")
        return True
    else:
        return False


def process_compute_data_single(args: Tuple[str, str, str, Path, Path, Path, bool, str]) -> Dict[str, Any]:
    """Process compute_data for a single subject/session (original or recon).
    
    Args:
        subject_id: Subject ID
        session_num: Session number  
        file_type: 'original' or 'recon'
        base_dir: Base directory
        h5_file: Path to H5 file
        fif_file: Path to FIF file
        output_dir: Output directory for computed data
        skip_clustering: Whether to skip clustering
        log_level: Logging level
        
    Returns:
        Dictionary with results
    """
    subject_id, session_num, file_type, base_dir, h5_file, fif_file, output_dir, skip_clustering, log_level = args
    
    # Setup logging for this process
    setup_worker_logging(log_level)
    logger = logging.getLogger(__name__)
    
    result = {
        'subject_id': subject_id,
        'session_num': session_num,
        'file_type': file_type,
        'success': False,
        'error': None,
        'step': 'compute_data'
    }
    
    try:
        # Verify input files exist
        if not h5_file.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file}")
        if not fif_file.exists():
            raise FileNotFoundError(f"FIF file not found: {fif_file}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Computing data for {file_type} - sub-{subject_id}/ses-{session_num}")
        
        # Prepare sys.argv for compute_data_main
        original_argv = sys.argv.copy()
        sys.argv = [
            'compute_data.py',
            '--subject_id', subject_id,
            '--h5_file', str(h5_file),
            '--fif_file', str(fif_file),
            '--output_dir', str(output_dir),
            '--task', 'lg'
        ]
        if skip_clustering:
            sys.argv.append('--skip-clustering')
        
        try:
            compute_data_main()
            result['success'] = True
            logger.info(f"✅ Computed data for {file_type} - sub-{subject_id}/ses-{session_num}")
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Failed to compute data for {file_type} - sub-{subject_id}/ses-{session_num}: {e}")
    
    return result


def process_generate_plots_single(args: Tuple[str, str, Path, Path, Path, Path, bool, str]) -> Dict[str, Any]:
    """Process generate_plots for a single subject/session.
    
    Args:
        subject_id: Subject ID
        session_num: Session number
        base_dir: Base directory
        h5_file: Path to H5 file  
        fif_file: Path to FIF file
        data_dir_original: Original computed data directory
        data_dir_recon: Reconstructed computed data directory
        skip_clustering: Whether to skip clustering
        log_level: Logging level
        
    Returns:
        Dictionary with results
    """
    subject_id, session_num, base_dir, h5_file, fif_file, data_dir_original, data_dir_recon, skip_clustering, log_level = args
    
    # Setup logging for this process
    setup_worker_logging(log_level)
    logger = logging.getLogger(__name__)
    
    result = {
        'subject_id': subject_id,
        'session_num': session_num,
        'success': False,
        'error': None,
        'step': 'generate_plots'
    }
    
    try:
        # Verify input files exist
        if not h5_file.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file}")
        if not fif_file.exists():
            raise FileNotFoundError(f"FIF file not found: {fif_file}")
        if not data_dir_original.exists():
            raise FileNotFoundError(f"Original data directory not found: {data_dir_original}")
        if not data_dir_recon.exists():
            raise FileNotFoundError(f"Reconstructed data directory not found: {data_dir_recon}")
        
        # Create output directory
        reports_dir = base_dir / "reports" / f"sub-{subject_id}" / f"ses-{session_num}"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Generating plots for sub-{subject_id}/ses-{session_num}")
        
        # Prepare sys.argv for generate_plots_main
        original_argv = sys.argv.copy()
        sys.argv = [
            'generate_plots.py',
            '--subject_id', subject_id,
            '--session', session_num,
            '--h5_file', str(h5_file),
            '--fif_file', str(fif_file),
            '--data_dir_original', str(data_dir_original),
            '--data_dir_recon', str(data_dir_recon),
            '--output_dir', str(reports_dir)
        ]
        if skip_clustering:
            sys.argv.append('--skip-clustering')
        
        try:
            generate_plots_main()
            result['success'] = True
            logger.info(f"✅ Generated plots for sub-{subject_id}/ses-{session_num}")
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Failed to generate plots for sub-{subject_id}/ses-{session_num}: {e}")
    
    return result


def process_subject_session_sequential(subject_id: str, session_num: str, base_dir: Path, fif_files_dir: Path, skip_clustering: bool, skip_plots: bool, log_level: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process a single subject/session sequentially."""
    logger.info(f"📝 Processing sub-{subject_id}/ses-{session_num} sequentially")
    
    results = []
    
    # Define paths
    h5_files_dir = base_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session_num}"
    computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
    
    original_h5 = h5_files_dir / "original.h5"
    recon_h5 = h5_files_dir / "recon.h5"
    
    fif_dir = fif_files_dir / f"sub-{subject_id}" / f"ses-{session_num}"
    original_fif = fif_dir / f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_original.fif"
    recon_fif = fif_dir / f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_recon.fif"
    
    orig_output_dir = computed_data_dir / "orig"
    recon_output_dir = computed_data_dir / "recon"
    
    # Step 1: Compute data for original
    orig_args = (subject_id, session_num, "original", base_dir, original_h5, original_fif, orig_output_dir, skip_clustering, log_level)
    results.append(process_compute_data_single(orig_args))
    
    # Step 2: Compute data for recon
    recon_args = (subject_id, session_num, "recon", base_dir, recon_h5, recon_fif, recon_output_dir, skip_clustering, log_level)
    results.append(process_compute_data_single(recon_args))
    
    # Step 3: Generate plots (only if both compute_data succeeded and not skipping plots)
    if not skip_plots and all(r['success'] for r in results):
        plots_args = (subject_id, session_num, base_dir, original_h5, original_fif, orig_output_dir, recon_output_dir, skip_clustering, log_level)
        results.append(process_generate_plots_single(plots_args))
    elif skip_plots:
        logger.info(f"⏭️  Skipping plot generation for sub-{subject_id}/ses-{session_num}")
    else:
        logger.error(f"❌ Skipping plot generation for sub-{subject_id}/ses-{session_num} - compute_data failed")
    
    return results


def process_all_compute_data_parallel(subjects_sessions: List[Tuple[str, str]], base_dir: Path, fif_files_dir: Path, skip_clustering: bool, log_level: str, n_workers: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process compute_data for all subjects/sessions in parallel."""
    logger.info(f"🚀 Processing compute_data for {len(subjects_sessions)} subject/sessions with {n_workers} workers")
    
    # Prepare all arguments (each H5 file is a separate task)
    all_args = []
    for subject_id, session_num in subjects_sessions:
        h5_files_dir = base_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session_num}"
        computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
        
        original_h5 = h5_files_dir / "original.h5"
        recon_h5 = h5_files_dir / "recon.h5"
        
        fif_dir = fif_files_dir / f"sub-{subject_id}" / f"ses-{session_num}"
        original_fif = fif_dir / f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_original.fif"
        recon_fif = fif_dir / f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_recon.fif"
        
        orig_output_dir = computed_data_dir / "orig"
        recon_output_dir = computed_data_dir / "recon"
        
        all_args.append((subject_id, session_num, "original", base_dir, original_h5, original_fif, orig_output_dir, skip_clustering, log_level))
        all_args.append((subject_id, session_num, "recon", base_dir, recon_h5, recon_fif, recon_output_dir, skip_clustering, log_level))
    
    # Process in parallel
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_compute_data_single, all_args),
            total=len(all_args),
            desc="Computing data"
        ))
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sequential compute_data and generate_plots processor for subjects"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS"),
        help="Base directory for MARKERS results"
    )
    parser.add_argument(
        "--fif-dir",
        type=Path,
        default=Path("/data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata"),
        help="Directory containing FIF files"
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Run compute_data in parallel using multiprocessing"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1, recommended: 2-4 for large files)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects that already have complete computed data"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (only run compute_data)"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering analysis to speed up computation"
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
    fif_files_dir = args.fif_dir
    
    if not h5_files_dir.exists():
        logger.error(f"❌ H5 files directory not found: {h5_files_dir}")
        sys.exit(1)
    
    if not fif_files_dir.exists():
        logger.error(f"❌ FIF files directory not found: {fif_files_dir}")
        sys.exit(1)
    
    logger.info(f"{'='*80}")
    logger.info("🧠📊 COMPUTE_DATA & PLOTS SEQUENTIAL PROCESSOR")
    logger.info(f"{'='*80}")
    logger.info(f"📂 Base directory: {base_dir}")
    logger.info(f"📁 H5 directory: {h5_files_dir}")
    logger.info(f"📁 FIF directory: {fif_files_dir}")
    logger.info(f"🔄 Parallelize: {args.parallelize}")
    if args.parallelize:
        n_workers = args.n_workers or (mp.cpu_count() - 1)
        logger.info(f"⚙️  Workers: {n_workers} (recommended: 2-4 for large files)")
        logger.warning(f"⚠️  MEMORY WARNING: Each worker may use 8-16GB RAM. Monitor memory usage closely!")
    logger.info(f"⏭️  Skip existing: {args.skip_existing}")
    logger.info(f"⏭️  Skip plots: {args.skip_plots}")
    logger.info(f"⏭️  Skip clustering: {args.skip_clustering}")
    logger.info(f"🔍 Dry run: {args.dry_run}")
    logger.info(f"🎯 Subject filter: {args.subject}")
    logger.info(f"🎯 Session filter: {args.session}")
    logger.info(f"{'='*80}\n")
    
    # Discover subjects and sessions with both H5 and FIF files
    logger.info("🔍 Discovering subjects and sessions with complete data...")
    subjects_sessions = discover_subjects_sessions(h5_files_dir, fif_files_dir, logger)
    
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
        logger.error("❌ No subjects/sessions found with complete H5 and FIF data!")
        sys.exit(1)
    
    # Check for existing computed data
    if args.skip_existing:
        logger.info("⏭️  Checking for existing computed data...")
        to_process = []
        skipped = []
        for subject_id, session_num in subjects_sessions:
            if check_computed_data_already_done(subject_id, session_num, base_dir, logger):
                skipped.append((subject_id, session_num))
            else:
                to_process.append((subject_id, session_num))
        
        logger.info(f"📊 Summary: {len(skipped)} subjects already have computed data, {len(to_process)} to process")
        subjects_sessions = to_process
    
    if not subjects_sessions:
        logger.info("✅ All subjects already have complete computed data. Nothing to do.")
        return
    
    if args.dry_run:
        logger.info("🔍 Dry run mode - showing what would be processed:")
        for subject_id, session_num in subjects_sessions:
            logger.info(f"  Would process: sub-{subject_id}/ses-{session_num}")
            logger.info(f"    Original H5: {h5_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / 'original.h5'}")
            logger.info(f"    Recon H5: {h5_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / 'recon.h5'}")
            logger.info(f"    Original FIF: {fif_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / f'sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_original.fif'}")
            logger.info(f"    Recon FIF: {fif_files_dir / f'sub-{subject_id}' / f'ses-{session_num}' / f'sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_recon.fif'}")
            if not args.skip_plots:
                logger.info(f"    Report output: {base_dir / 'reports' / f'sub-{subject_id}' / f'ses-{session_num}' / f'sub-{subject_id}_ses-{session_num}_report_comparison.html'}")
        logger.info(f"\n📊 Total: {len(subjects_sessions)} subject/session combinations")
        return
    
    logger.info(f"\n🚀 Starting computation for {len(subjects_sessions)} subject/session combinations\n")
    
    start_time = time.time()
    all_results = []
    
    if args.parallelize:
        n_workers = args.n_workers or (mp.cpu_count() - 1)
        # Step 1: Parallel compute_data
        compute_results = process_all_compute_data_parallel(subjects_sessions, base_dir, fif_files_dir, args.skip_clustering, args.log_level, n_workers, logger)
        all_results.extend(compute_results)
        
        # Step 2: Sequential generate_plots (only if compute_data succeeded and not skipping plots)
        if not args.skip_plots:
            logger.info("📊 Starting plot generation (sequential)...")
            for subject_id, session_num in subjects_sessions:
                # Check if both original and recon compute_data succeeded
                orig_success = any(r['success'] and r['file_type'] == 'original' and r['subject_id'] == subject_id and r['session_num'] == session_num for r in compute_results)
                recon_success = any(r['success'] and r['file_type'] == 'recon' and r['subject_id'] == subject_id and r['session_num'] == session_num for r in compute_results)
                
                if orig_success and recon_success:
                    # Generate plots
                    h5_files_dir = base_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session_num}"
                    computed_data_dir = base_dir / "computed_data" / f"sub-{subject_id}" / f"ses-{session_num}"
                    
                    original_h5 = h5_files_dir / "original.h5"
                    fif_dir = fif_files_dir / f"sub-{subject_id}" / f"ses-{session_num}"
                    original_fif = fif_dir / f"sub-{subject_id}_ses-{session_num}_task-lg_acq-01_epo_original.fif"
                    
                    orig_output_dir = computed_data_dir / "orig"
                    recon_output_dir = computed_data_dir / "recon"
                    
                    plots_args = (subject_id, session_num, base_dir, original_h5, original_fif, orig_output_dir, recon_output_dir, args.skip_clustering, args.log_level)
                    plot_result = process_generate_plots_single(plots_args)
                    all_results.append(plot_result)
    else:
        # Sequential processing
        for i, (subject_id, session_num) in enumerate(subjects_sessions, 1):
            logger.info(f"📝 [{i}/{len(subjects_sessions)}] Processing sub-{subject_id}/ses-{session_num}...")
            results = process_subject_session_sequential(subject_id, session_num, base_dir, fif_files_dir, args.skip_clustering, args.skip_plots, args.log_level, logger)
            all_results.extend(results)
    
    end_time = time.time()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 PROCESSING SUMMARY")
    logger.info(f"{'='*80}")
    
    compute_results = [r for r in all_results if r['step'] == 'compute_data']
    plot_results = [r for r in all_results if r['step'] == 'generate_plots']
    
    compute_successful = [r for r in compute_results if r['success']]
    compute_failed = [r for r in compute_results if not r['success']]
    plot_successful = [r for r in plot_results if r['success']]
    plot_failed = [r for r in plot_results if not r['success']]
    
    logger.info(f"📝 Total compute_data tasks: {len(compute_results)}")
    logger.info(f"✅ Compute_data successful: {len(compute_successful)}")
    logger.info(f"❌ Compute_data failed: {len(compute_failed)}")
    
    if not args.skip_plots:
        logger.info(f"📊 Total plot generation tasks: {len(plot_results)}")
        logger.info(f"✅ Plot generation successful: {len(plot_successful)}")
        logger.info(f"❌ Plot generation failed: {len(plot_failed)}")
    
    logger.info(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    
    if compute_failed:
        logger.error(f"\n❌ Failed compute_data tasks:")
        for result in compute_failed:
            logger.error(f"  sub-{result['subject_id']}/ses-{result['session_num']} ({result['file_type']}): {result['error']}")
    
    if plot_failed:
        logger.error(f"\n❌ Failed plot generation tasks:")
        for result in plot_failed:
            logger.error(f"  sub-{result['subject_id']}/ses-{result['session_num']}: {result['error']}")
    
    logger.info(f"\n📂 Expected outputs:")
    logger.info(f"  - Computed data: {base_dir}/computed_data/sub-*/ses-*/orig/ and recon/")
    logger.info(f"  - Reports: {base_dir}/reports/sub-*/ses-*/")
    logger.info(f"{'='*80}\n")
    
    if compute_failed or (plot_failed and not args.skip_plots):
        sys.exit(1)


if __name__ == "__main__":
    main()
