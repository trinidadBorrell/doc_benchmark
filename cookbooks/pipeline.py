#!/usr/bin/env python3
"""
Pipeline to benchmark data

A comprehensive Python pipeline for processing EEG data through three main phases:
1. Markers: Compute EEG markers and features from raw data
2. Analysis: Compare markers and perform individual subject analysis
3. Models: Train extra trees for classification 


Usage:
    python pipeline.py --subject AD023 --metadata-dir /path/to/metadata --data-dir /path/to/fifdata                 # Process one subject
    python pipeline.py --subjects AD023,YC260 --metadata-dir /path/to/metadata --data-dir /path/to/fifdata             # Process specific subjects  
    python pipeline.py --all --metadata-dir /path/to/metadata --data-dir /path/to/fifdata                              # Process all subjects
    python pipeline.py --random 5 --metadata-dir /path/to/metadata --data-dir /path/to/fifdata                         # Process 5 random subjects
    python pipeline.py --subject AD023 --skip-models --metadata-dir /path/to/metadata --data-dir /path/to/fifdata      # Skip model training
    python pipeline.py --all --markers-only --metadata-dir /path/to/metadata --data-dir /path/to/fifdata               # Only compute markers
    python pipeline.py --analysis-only --metadata-dir /path/to/metadata --data-dir /path/to/fifdata                    # Only run analysis on existing results

     python pipeline.py --subject AD023 --metadata-dir /Users/trinidad.borrell/Documents/Work/PhD/Proyects/nice/doc_a_data/metadata --data-dir /Users/trinidad.borrell/Documents/Work/PhD/data/TOTEM/zero_shot_data/fifdata/fifdata
Authors: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import sys
import subprocess
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class Pipeline:
    """Main pipeline class for benchmark data analysis."""
    
    def __init__(self, 
                 data_dir: str,
                 results_dir: str,
                 src_dir: str,
                 metadata_dir: str,
                 batch_size: int = 1,
                 verbose: bool = False,
                 dry_run: bool = False):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        data_dir : str
            Path to the raw data directory
        results_dir : str
            Path to save results
        src_dir : str
            Path to the source code directory
        batch_size : int
            Number of parallel processes
        verbose : bool
            Enable verbose logging
        dry_run : bool
            Show what would be done without executing
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.src_dir = Path(src_dir)
        self.metadata_dir = Path(metadata_dir)
        self.batch_size = min(batch_size, max(1, multiprocessing.cpu_count() - 1))
        self.verbose = verbose
        self.dry_run = dry_run
        
        # Setup logging
        self._setup_logging()
        
        # Validate directories
        self._validate_directories()
        
        # Track progress
        self.completed_subjects = []
        self.failed_subjects = []
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline log file: {log_file}")
        
    def _validate_directories(self):
        """Validate that required directories exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.src_dir}")
            
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required source files
        required_files = [
            "markers/compute_doc_forest_markers_variable.py",
            "markers/compute_doc_forest_features_variable.py",
            "model/extratrees.py",
            "analysis/compare_markers.py",
            "analysis/global_analysis.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.src_dir / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            raise FileNotFoundError(f"Missing required source files: {missing_files}")
    
    def discover_subjects(self) -> List[Tuple[str, str]]:
        """
        Discover all subjects with available sessions.
        
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
                
                # Check if both original and reconstructed files exist
                original_files = list(session_dir.glob("*_original.fif"))
                recon_files = list(session_dir.glob("*_recon.fif"))
                
                if original_files and recon_files:
                    subjects.append((subject_id, session))
                    
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
    
    def run_markers_phase(self, subject_id: str, session: str) -> bool:
        """
        Run the markers computation phase for a subject/session.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            subject_dir = self.data_dir / f"sub-{subject_id}" / session
            results_dir = self.results_dir / "SUBJECTS" / f"sub-{subject_id}" / session
            
            # Create output directories
            markers_dir = results_dir / "markers_variable"
            features_dir = results_dir / "features_variable"
            markers_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(parents=True, exist_ok=True)
            
            # Find data files
            original_file = next(subject_dir.glob("*_original.fif"), None)
            recon_file = next(subject_dir.glob("*_recon.fif"), None)
            
            if not original_file or not recon_file:
                self.logger.error(f"Missing data files for {subject_id}/{session}")
                return False
                
            self.logger.info(f"Computing markers for {subject_id}/{session}")
            
            # Process original data
            self.logger.info(f"  Step 1: Computing markers from {original_file.name}")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_markers_variable.py"),
                str(original_file),
                "--output", str(markers_dir / "markers_original.hdf5"),
                "--plot"
            ]):
                return False
                
            self.logger.info(f"  Step 2: Computing features from markers_original.hdf5")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_features_variable.py"),
                str(markers_dir / "markers_original.hdf5"),
                "--output-scalars", str(features_dir / "scalars_original.npy"),
                "--output-topos", str(features_dir / "topos_original.npy"),
                "--plot"
            ]):
                return False
            
            # Process reconstructed data
            self.logger.info(f"  Step 3: Computing markers from {recon_file.name}")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_markers_variable.py"),
                str(recon_file),
                "--output", str(markers_dir / "markers_reconstructed.hdf5"),
                "--plot"
            ]):
                return False
                
            self.logger.info(f"  Step 4: Computing features from markers_reconstructed.hdf5")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_features_variable.py"),
                str(markers_dir / "markers_reconstructed.hdf5"),
                "--output-scalars", str(features_dir / "scalars_reconstructed.npy"),
                "--output-topos", str(features_dir / "topos_reconstructed.npy"),
                "--plot"
            ]):
                return False
            
            self.logger.info(f"✓ Markers completed for {subject_id}/{session}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Markers failed for {subject_id}/{session}: {e}")
            return False
    
    def run_models_phase(self) -> bool:
        """
        Run the model training phase across all processed subjects.
        This should be called after all subjects have completed markers and analysis phases.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("Running global models training")
            
            # Create global models output directory
            models_output_dir = self.results_dir / "EXTRATREES" / f"models_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            models_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get patient labels path from metadata directory
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            
            if not patient_labels_path.exists():
                # Fallback to old path structure
                patient_labels_path = self.src_dir.parent / "metadata" / "patient_labels_with_controls.csv"
                
            if not patient_labels_path.exists():
                self.logger.error(f"Patient labels file not found at: {patient_labels_path}")
                return False
            
            self.logger.info(f"Using patient labels: {patient_labels_path}")
            
            # Train models for different configurations
            configurations = [
                ("scalar", "original"),
                ("scalar", "reconstructed"),
                ("topo", "original"), 
                ("topo", "reconstructed")
            ]
            
            success = True
            for marker_type, data_origin in configurations:
                self.logger.info(f"Training {marker_type} {data_origin} model...")
                
                config_output_dir = models_output_dir / f"{marker_type}_{data_origin}"
                
                if not self._run_command([
                    "python", str(self.src_dir / "model/extratrees.py"),
                    "--data-dir", str(self.results_dir),
                    "--patient-labels", str(patient_labels_path),
                    "--output-dir", str(config_output_dir),
                    "--marker-type", marker_type,
                    "--data-origin", data_origin
                ]):
                    self.logger.error(f"Failed to train {marker_type} {data_origin} model")
                    success = False
                else:
                    self.logger.info(f"✓ {marker_type} {data_origin} model completed")
            
            if success:
                self.logger.info("✓ Global models training completed")
            else:
                self.logger.error("✗ Some models failed to train")
                
            return success
            
        except Exception as e:
            self.logger.error(f"✗ Global models training failed: {e}")
            return False
    
    def run_analysis_phase(self, subject_id: str, session: str) -> bool:
        """
        Run the analysis phase for a subject/session.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            results_dir = self.results_dir / "SUBJECTS" / f"sub-{subject_id}" / session 
            compare_dir = results_dir / "compare_markers"
            compare_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Running analysis for {subject_id}/{session}")
            
            # Run marker comparison
            if not self._run_command([
                "python", str(self.src_dir / "analysis/compare_markers.py"),
                str(results_dir),
                "--output", str(compare_dir)
            ]):
                return False
            
            self.logger.info(f"✓ Analysis completed for {subject_id}/{session}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Analysis failed for {subject_id}/{session}: {e}")
            return False
    
    def run_global_analysis(self) -> bool:
        """
        Run global analysis across all processed subjects.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("Running global analysis")
            
            patient_labels_path = self.metadata_dir.parent / "metadata" / "patient_labels_with_controls.csv"
            global_output_dir = self.results_dir / "GLOBAL" / f"global_results_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            global_output_dir.mkdir(parents=True, exist_ok=True)
            results_dir = self.results_dir / "SUBJECTS"
            
            if not self._run_command([
                "python", str(self.src_dir / "analysis/global_analysis.py"),
                "--results-dir", str(results_dir),
                "--output-dir", str(global_output_dir),
                "--patient-labels", str(patient_labels_path)
            ]):
                return False
            
            self.logger.info("✓ Global analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Global analysis failed: {e}")
            return False
    
    def process_subject(self, subject_id: str, session: str, 
                       skip_markers: bool = False,
                       skip_models: bool = False, 
                       skip_analysis: bool = False) -> bool:
        """
        Process a single subject through individual phases (markers and analysis).
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        skip_markers : bool
            Skip markers computation phase
        skip_models : bool
            Skip models training phase 
        skip_analysis : bool
            Skip analysis phase
            
        Returns
        -------
        bool
            True if all requested phases completed successfully
        """
        self.logger.info(f"Processing {subject_id}/{session}")
        
        success = True
        
        # Phase 1: Markers
        if not skip_markers:
            if not self.run_markers_phase(subject_id, session):
                success = False
        else:
            self.logger.info(f"Skipping markers for {subject_id}/{session}")
        
        # Phase 2: Analysis  
        if not skip_analysis and success:
            if not self.run_analysis_phase(subject_id, session):
                success = False
        else:
            if skip_analysis:
                self.logger.info(f"Skipping analysis for {subject_id}/{session}")
        
        if success:
            self.completed_subjects.append((subject_id, session))
            self.logger.info(f"✓ Completed {subject_id}/{session}")
        else:
            self.failed_subjects.append((subject_id, session))
            self.logger.error(f"✗ Failed {subject_id}/{session}")
            
        return success
    
    def run_pipeline(self, subjects: List[Tuple[str, str]],
                    skip_markers: bool = False,
                    skip_models: bool = False,
                    skip_analysis: bool = False,
                    skip_global: bool = False) -> dict:
        """
        Run the complete pipeline for multiple subjects.
        
        Pipeline order:
        1. Individual subject processing (markers + analysis)
        2. Global analysis
        3. Model training

        Parameters
        ----------
        subjects : List[Tuple[str, str]]
            List of (subject_id, session) tuples to process
        skip_markers : bool
            Skip markers computation phase
        skip_models : bool
            Skip global models training phase
        skip_analysis : bool
            Skip individual analysis phase
        skip_global : bool
            Skip global analysis phase
            
        Returns
        -------
        dict
            Summary of pipeline execution
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE START")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Subjects to process: {len(subjects)}")
        self.logger.info(f"Parallel processes: {self.batch_size}")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual processing")
            return {"status": "dry_run", "subjects": subjects}
        
        # Process subjects
        if self.batch_size > 1:
            self._run_parallel(subjects, skip_markers, skip_models, skip_analysis)
        else:
            self._run_sequential(subjects, skip_markers, skip_models, skip_analysis)

        # Phase 2: Global Analysis  
        if not skip_global and self.completed_subjects:
            self.run_global_analysis()
        
        # Phase 3: Models Training
        if not skip_models and self.completed_subjects:
            self.logger.info("Starting global models training phase...")
            models_success = self.run_models_phase()
            if not models_success:
                self.logger.error("Global models training failed")
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Completed: {len(self.completed_subjects)}")
        self.logger.info(f"Failed: {len(self.failed_subjects)}")
        
        if self.failed_subjects:
            self.logger.error(f"Failed subjects: {self.failed_subjects}")
        
        return {
            "status": "complete",
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "completed": self.completed_subjects,
            "failed": self.failed_subjects
        }
    
    def _run_parallel(self, subjects: List[Tuple[str, str]],
                     skip_markers: bool, skip_models: bool, skip_analysis: bool):
        """Run processing in parallel."""
        self.logger.info(f"Running parallel processing with {self.batch_size} processes")
        
        with ProcessPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all jobs
            future_to_subject = {
                executor.submit(self.process_subject, subj_id, session, 
                              skip_markers, skip_models, skip_analysis): (subj_id, session)
                for subj_id, session in subjects
            }
            
            # Collect results
            for future in as_completed(future_to_subject):
                subject_id, session = future_to_subject[future]
                try:
                    success = future.result()
                    if success:
                        self.completed_subjects.append((subject_id, session))
                    else:
                        self.failed_subjects.append((subject_id, session))
                except Exception as e:
                    self.logger.error(f"Exception processing {subject_id}/{session}: {e}")
                    self.failed_subjects.append((subject_id, session))
    
    def _run_sequential(self, subjects: List[Tuple[str, str]],
                       skip_markers: bool, skip_models: bool, skip_analysis: bool):
        """Run processing sequentially."""
        self.logger.info("Running sequential processing")
        
        for i, (subject_id, session) in enumerate(subjects, 1):
            self.logger.info(f"Progress: {i}/{len(subjects)}")
            self.process_subject(subject_id, session, skip_markers, skip_models, skip_analysis)
    
    def _run_command(self, cmd: List[str]) -> bool:
        """
        Run a command and return success status.
        
        Parameters
        ----------
        cmd : List[str]
            Command to execute
            
        Returns
        -------
        bool
            True if command succeeded, False otherwise
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: {' '.join(cmd)}")
            return True
            
        try:
            self.logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Always show stdout for better debugging, not just in verbose mode
            if result.stdout.strip():
                self.logger.info(f"STDOUT: {result.stdout}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}")
            self.logger.error(f"Exit code: {e.returncode}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Benchmark Data Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline.py --subject AD023
    python pipeline.py --subjects AD023,YC260,BC031  
    python pipeline.py --all --skip-models
    python pipeline.py --random 5 --batch-size 4
    python pipeline.py --analysis-only --all
        """
    )

    # Set default paths relative to script location
    # Default root is doc_benchmark
    project_root = str(Path(__file__).parent.parent)
    
    # Subject selection (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument('--subject', 
                              help='Process single subject')
    subject_group.add_argument('--subjects',
                              help='Process specific subjects (comma-separated)')
    subject_group.add_argument('--all', action='store_true',
                              help='Process all available subjects')
    subject_group.add_argument('--random', type=int, metavar='N',
                              help='Process N random subjects')
    
    # Phase control
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument('--markers-only', action='store_true',
                            help='Only run markers computation')
    phase_group.add_argument('--analysis-only', action='store_true',
                            help='Only run analysis')
    phase_group.add_argument('--global-only', action='store_true',
                            help='Only run global analysis')
    phase_group.add_argument('--models-only', action='store_true',
                            help='Only run model training')
    
    # Individual phase skipping
    parser.add_argument('--skip-markers', action='store_true',
                       help='Skip markers computation phase')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis phase')
    parser.add_argument('--skip-global', action='store_true',
                       help='Skip global analysis')
    parser.add_argument('--skip-models', action='store_true', 
                       help='Skip model training phase')
    
    # Configuration
    parser.add_argument('--data-dir', 
                        required=True,
                       help='Path to data directory')
    parser.add_argument('--results-dir',
                       default=f'{project_root}/results',
                       help='Path to results directory')
    parser.add_argument('--src-dir',
                       default=f'{project_root}/src',
                       help='Path to source code directory')
    parser.add_argument('--metadata-dir',
                       required=True,
                       help='Path to metadata directory')

    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of parallel processes')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Resolve subject arguments
    subject_args = []
    if args.subject:
        subject_args = [args.subject]
    elif args.subjects:
        subject_args = args.subjects.split(',')
    elif args.all:
        subject_args = ["ALL"]
    elif args.random:
        subject_args = [f"RANDOM:{args.random}"]
    
    # Handle phase-only modes
    skip_markers = args.skip_markers or args.models_only or args.analysis_only or args.global_only
    skip_models = args.skip_models or args.markers_only or args.analysis_only or args.global_only  
    skip_analysis = args.skip_analysis or args.markers_only or args.models_only or args.global_only
    skip_global = args.skip_global or args.markers_only or args.models_only or args.analysis_only
    
   # try:
        # Create pipeline
    pipeline = Pipeline(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            src_dir=args.src_dir,
            metadata_dir=args.metadata_dir,
            batch_size=args.batch_size,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        
        # Handle global-only mode
    if args.global_only:
            pipeline.run_global_analysis()
            return
        
        # Handle models-only mode  
    if args.models_only:
            pipeline.run_models_phase()
            return

        # Resolve subjects
    subjects = pipeline.resolve_subjects(subject_args)
        
    if not subjects:
            print("No subjects to process")
            return
            
        # Run pipeline
    results = pipeline.run_pipeline(
            subjects=subjects,
            skip_markers=skip_markers,
            skip_models=skip_models, 
            skip_analysis=skip_analysis,
            skip_global=skip_global
        )
        
            
   # except Exception as e:
   #     print(f"Pipeline failed: {e}")
   #     sys.exit(1)


if __name__ == "__main__":
    main()
