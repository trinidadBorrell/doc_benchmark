#!/usr/bin/env python3
"""
Dummy Model Pipeline Integration

This script integrates the dummy model with the existing DOC benchmark pipeline.
It creates reconstructed .fif files that can be processed by the standard pipeline,
allowing for direct comparison with other reconstruction methods.

The dummy model:
1. Reads original .fif files
2. Creates reconstructed data by replacing time series with temporal means
3. Saves reconstructed .fif files with proper naming convention
4. Integrates with the existing pipeline for marker computation and analysis

Usage:
    python dummy_pipeline.py --subject AD023 --data-dir /path/to/data
    python dummy_pipeline.py --all --data-dir /path/to/data
    python dummy_pipeline.py --subjects AD023,YC260 --data-dir /path/to/data

Authors: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import random

from dummy_model import DummyModel


class DummyPipeline:
    """
    Pipeline integration for the dummy model.
    
    This class handles the integration of the dummy model with the existing
    DOC benchmark pipeline structure, creating reconstructed .fif files that
    can be processed by the standard pipeline.
    """
    
    def __init__(self, data_dir: str, verbose: bool = False, dry_run: bool = False):
        """
        Initialize the dummy pipeline.
        
        Parameters
        ----------
        data_dir : str
            Path to the raw data directory
        verbose : bool
            Enable verbose logging
        dry_run : bool
            Show what would be done without executing
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.dry_run = dry_run
        
        # Setup logging
        self._setup_logging()
        
        # Validate directories
        self._validate_directories()
        
        # Initialize dummy model
        self.dummy_model = DummyModel(verbose=verbose)
        
        # Track progress
        self.completed_subjects = []
        self.failed_subjects = []
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Dummy Pipeline initialized")
    
    def _validate_directories(self):
        """Validate that required directories exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def discover_subjects(self) -> List[Tuple[str, str]]:
        """
        Discover all subjects with available sessions that have original .fif files.
        
        Returns
        -------
        List[Tuple[str, str]]
            List of (subject_id, session) tuples
        """
        subjects = []
        
        for subject_dir in self.data_dir.glob("sub-*"):
            if not subject_dir.is_dir():
                continue
                
            subject_id = subject_dir.name.replace("sub-", "")
            
            # Find all sessions for this subject
            for session_dir in subject_dir.glob("ses-*"):
                if not session_dir.is_dir():
                    continue
                    
                session = session_dir.name
                
                # Check if original files exist
                original_files = list(session_dir.glob("*_original.fif"))
                
                if original_files:
                    subjects.append((subject_id, session))
                    self.logger.debug(f"Found subject {subject_id}/{session} with {len(original_files)} original file(s)")
        
        self.logger.info(f"Discovered {len(subjects)} subject/session pairs")
        return subjects
    
    def resolve_subjects(self, subject_args: List[str]) -> List[Tuple[str, str]]:
        """
        Resolve subject selection arguments to actual subject/session pairs.
        
        Parameters
        ----------
        subject_args : List[str]
            Subject selection arguments
            
        Returns
        -------
        List[Tuple[str, str]]
            List of (subject_id, session) tuples
        """
        all_subjects = self.discover_subjects()
        
        if not all_subjects:
            raise ValueError("No subjects found in data directory")
            
        resolved = []
        
        for arg in subject_args:
            if arg == "ALL":
                resolved.extend(all_subjects)
            elif arg.startswith("RANDOM:"):
                count = int(arg.split(":")[1])
                shuffled = random.sample(all_subjects, min(count, len(all_subjects)))
                resolved.extend(shuffled)
            else:
                # Handle specific subject ID
                matching = [(s, ses) for s, ses in all_subjects if s == arg]
                if not matching:
                    self.logger.warning(f"No sessions found for subject {arg}")
                else:
                    resolved.extend(matching)
        
        # Remove duplicates and sort
        resolved = list(set(resolved))
        resolved.sort()
        
        return resolved
    
    def is_subject_complete(self, subject_id: str, session: str) -> bool:
        """
        Check if a subject/session already has reconstructed files.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
            
        Returns
        -------
        bool
            True if reconstructed files exist, False otherwise
        """
        subject_dir = self.data_dir / f"sub-{subject_id}" / session
        recon_files = list(subject_dir.glob("*_recon.fif"))
        return len(recon_files) > 0
    
    def process_subject(self, subject_id: str, session: str, output_dir: str, 
                       force_recompute: bool = False, skip_smi: bool = True, 
                       cv: int = 5, n_jobs: int = 1) -> bool:
        """
        Process a single subject through the dummy model ON-THE-FLY.
        
        This processes everything in memory without saving heavy .fif files.
        Only markers, features, and decoder results are saved.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        output_dir : str
            Directory to save results
        force_recompute : bool
            Whether to recompute even if results already exist
        skip_smi : bool
            Whether to skip SymbolicMutualInformation computation
        cv : int
            Number of cross-validation folds for decoding
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            output_dir = Path(output_dir)
            
            # Check if already complete (check for feature files instead of .fif files)
            if not force_recompute:
                subject_output_dir = output_dir / f"sub-{subject_id}" / session
                features_dir = subject_output_dir / "features_variable"
                required_files = [
                    features_dir / "scalars_original.npy",
                    features_dir / "scalars_reconstructed.npy",
                    features_dir / "topos_original.npy", 
                    features_dir / "topos_reconstructed.npy"
                ]
                
                if all(f.exists() for f in required_files):
                    self.logger.info(f"⏭️  Skipping {subject_id}/{session} - already has results")
                    return True
            
            subject_dir = self.data_dir / f"sub-{subject_id}" / session
            
            # Find original files
            original_files = list(subject_dir.glob("*_original.fif"))
            
            if not original_files:
                self.logger.error(f"No original .fif files found for {subject_id}/{session}")
                return False
            
            if len(original_files) > 1:
                self.logger.warning(f"Multiple original files found for {subject_id}/{session}, using first one")
            
            original_file = original_files[0]
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would process {subject_id}/{session} with {original_file.name}")
                return True
            
            # Process using the dummy model's on-the-fly method
            success = self.dummy_model.process_subject(
                subject_id, session, self.data_dir, output_dir,
                skip_smi=skip_smi, cv=cv, n_jobs=n_jobs
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {subject_id}/{session}: {e}")
            return False
    
    def run_pipeline(self, subjects: List[Tuple[str, str]], output_dir: str, 
                    force_recompute: bool = False, skip_smi: bool = True, 
                    cv: int = 5, n_jobs: int = 1) -> dict:
        """
        Run the dummy model pipeline for multiple subjects ON-THE-FLY.
        
        Parameters
        ----------
        subjects : List[Tuple[str, str]]
            List of (subject_id, session) tuples to process
        output_dir : str
            Directory to save results
        force_recompute : bool
            Whether to recompute even if results already exist
        skip_smi : bool
            Whether to skip SymbolicMutualInformation computation
        cv : int
            Number of cross-validation folds for decoding
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        dict
            Summary of pipeline execution
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("DUMMY MODEL PIPELINE START")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Subjects to process: {len(subjects)}")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual processing")
            return {"status": "dry_run", "subjects": subjects}
        
        # Process subjects sequentially
        for i, (subject_id, session) in enumerate(subjects, 1):
            self.logger.info(f"Progress: {i}/{len(subjects)}")
            success = self.process_subject(
                subject_id, session, output_dir, force_recompute, 
                skip_smi, cv, n_jobs
            )
            
            # Track completed/failed subjects
            if success:
                self.completed_subjects.append((subject_id, session))
            else:
                self.failed_subjects.append((subject_id, session))
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("DUMMY MODEL PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Completed: {len(self.completed_subjects)}")
        self.logger.info(f"Failed: {len(self.failed_subjects)}")
        
        if self.failed_subjects:
            self.logger.error(f"Failed subjects: {self.failed_subjects}")
        
        # Provide instructions for next steps
        if self.completed_subjects:
            self.logger.info("")
            self.logger.info("🎉 Dummy model processing completed ON-THE-FLY!")
            self.logger.info("📋 Results generated:")
            self.logger.info("   ✓ Markers computed for original and dummy reconstructed data")
            self.logger.info("   ✓ Features extracted and saved as .npy files")
            self.logger.info("   ✓ Decoder analysis completed (original vs dummy)")
            self.logger.info("   ✓ No heavy .fif files were saved - only essential results")
            self.logger.info("")
            self.logger.info("📁 Output structure:")
            self.logger.info(f"   {output_dir}/sub-<ID>/<session>/")
            self.logger.info("   ├── markers_variable/")
            self.logger.info("   │   ├── markers_original.hdf5")
            self.logger.info("   │   └── markers_reconstructed.hdf5")
            self.logger.info("   ├── features_variable/")
            self.logger.info("   │   ├── scalars_original.npy")
            self.logger.info("   │   ├── scalars_reconstructed.npy")
            self.logger.info("   │   ├── topos_original.npy")
            self.logger.info("   │   └── topos_reconstructed.npy")
            self.logger.info("   └── decoder/")
            self.logger.info("       ├── decoding_results.pkl")
            self.logger.info("       ├── decoding_summary.json")
            self.logger.info("       ├── times.npy")
            self.logger.info("       └── decoding_plot.png")
        
        return {
            "status": "complete",
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "completed": self.completed_subjects,
            "failed": self.failed_subjects
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dummy Model Pipeline Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single subject
    python dummy_pipeline.py --subject AD023 --data-dir /path/to/data --output-dir /path/to/results
    
    # Process multiple subjects
    python dummy_pipeline.py --subjects AD023,YC260,BC031 --data-dir /path/to/data --output-dir /path/to/results
    
    # Process all subjects
    python dummy_pipeline.py --all --data-dir /path/to/data --output-dir /path/to/results
    
    # Process random subset with custom parameters
    python dummy_pipeline.py --random 5 --data-dir /path/to/data --output-dir /path/to/results --cv 10 --n-jobs 4
    
    # Force recompute existing results
    python dummy_pipeline.py --all --data-dir /path/to/data --output-dir /path/to/results --force-recompute
        """
    )
    
    # Subject selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subject",
        type=str,
        help="Process single subject ID"
    )
    group.add_argument(
        "--subjects",
        type=str,
        help="Process comma-separated list of subject IDs"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all available subjects"
    )
    group.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Process N random subjects"
    )
    
    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the raw data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--skip-smi",
        action="store_true",
        default=True,
        help="Skip SymbolicMutualInformation computation (default: True)"
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=10,
        help="Number of cross-validation folds for decoding (default: 5)"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute even if reconstructed files already exist"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_arguments()
    
    try:
        # Initialize pipeline
        pipeline = DummyPipeline(
            data_dir=args.data_dir,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        
        # Resolve subjects to process
        if args.subject:
            subject_args = [args.subject]
        elif args.subjects:
            subject_args = args.subjects.split(",")
        elif args.all:
            subject_args = ["ALL"]
        elif args.random:
            subject_args = [f"RANDOM:{args.random}"]
        
        subjects = pipeline.resolve_subjects(subject_args)
        
        if not subjects:
            pipeline.logger.error("No subjects to process")
            sys.exit(1)
        
        pipeline.logger.info(f"Selected {len(subjects)} subjects for processing")
        if pipeline.verbose:
            for subject_id, session in subjects:
                pipeline.logger.debug(f"  - {subject_id}/{session}")
        
        # Run pipeline
        results = pipeline.run_pipeline(
            subjects, 
            output_dir=args.output_dir,
            force_recompute=args.force_recompute,
            skip_smi=args.skip_smi,
            cv=args.cv,
            n_jobs=args.n_jobs
        )
        
        # Exit with error code if any subjects failed
        if results["status"] != "dry_run" and pipeline.failed_subjects:
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
