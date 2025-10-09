#!/usr/bin/env python3
"""
Pipeline to benchmark data

A comprehensive Python pipeline for processing EEG data through three main phases:
1. Markers: Compute EEG markers and features from raw data
2. Analysis: Compare markers and perform individual subject analysis
3. Models: Train extra trees for binary classification (VS vs MCS) 


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
                 dry_run: bool = False,
                 skip_smi: bool = False):
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
        metadata_dir : str
            Path to metadata directory
        batch_size : int
            Number of parallel processes
        verbose : bool
            Enable verbose logging
        dry_run : bool
            Show what would be done without executing
        skip_smi : bool
            Skip SymbolicMutualInformation computation (recommended for large datasets)
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.src_dir = Path(src_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Determine optimal batch size
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.batch_size = min(batch_size, max_workers)
        
        self.verbose = verbose
        self.dry_run = dry_run
        self.skip_smi = skip_smi
        
        # Setup logging (must come before using self.logger)
        self._setup_logging()
        
        # Log parallelization info
        if self.batch_size > 1:
            self.logger.info(f"🚀 Parallel processing enabled: {self.batch_size} workers (CPU cores: {multiprocessing.cpu_count()})")
        else:
            self.logger.info(f"⚙️  Sequential processing (use --batch-size N for parallel processing)")
        
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
            "analysis/individual_analysis.py",
            "analysis/global_analysis.py",
            "analysis/statistical_analysis.py",
            "decoder/decoder.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.src_dir / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            raise FileNotFoundError(f"Missing required source files: {missing_files}")
    
    def is_subject_complete(self, subject_id: str, session: str) -> bool:
        """
        Check if a subject/session is already complete based on required files and folders.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
            
        Returns
        -------
        bool
            True if subject/session is complete, False otherwise
        """
        results_dir = self.results_dir / "SUBJECTS" / f"sub-{subject_id}" / session
        
        # Check if the main result directory exists
        if not results_dir.exists():
            return False
        
        # Check features_variable folder and required .npy files
        features_dir = results_dir / "features_variable"
        required_npy_files = [
            'scalars_reconstructed.npy',
            'topos_reconstructed.npy', 
            'topos_original.npy',
            'scalars_original.npy'
        ]
        
        if not features_dir.exists():
            return False
            
        for npy_file in required_npy_files:
            if not (features_dir / npy_file).exists():
                return False
        
        # Check markers_variable folder and required .hdf5 files
        # Both files should exist for a complete result
        markers_dir = results_dir / "markers_variable"
        required_hdf5_files = ['markers_original.hdf5', 'markers_reconstructed.hdf5']
        
        if not markers_dir.exists():
            return False
            
        # Check that both hdf5 files exist (pipeline creates both)
        for hdf5_file in required_hdf5_files:
            if not (markers_dir / hdf5_file).exists():
                return False
        
        # Check individual_analysis folder and required subfolders/files
        analysis_dir = results_dir / "individual_analysis"
        required_analysis_items = [
            'analysis_summary.json',
            'global_field_power',
            'scalars', 
            'timeseries_error',
            'topography'
        ]
        
        if not analysis_dir.exists():
            return False
            
        for item in required_analysis_items:
            item_path = analysis_dir / item
            if not item_path.exists():
                return False
            
            # For subfolders, check if they contain plots
            if item_path.is_dir():
                plots_dir = item_path / "plots"
                if not plots_dir.exists():
                    return False
                # Check if there are any plot files (png, jpg, etc.)
                plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg")) + list(plots_dir.glob("*.jpeg"))
                if not plot_files:
                    return False
        
        return True

    def discover_subjects(self, force_recompute: bool = False) -> List[Tuple[str, str]]:
        """
        Discover all subjects with available sessions.
        
        Parameters
        ----------
        force_recompute : bool
            If True, include subjects even if they are already complete
            
        Returns
        -------
        List[Tuple[str, str]]
            List of (subject_id, session) tuples
        """
        subjects = []
        skipped_complete = []
        
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
                    # Check if subject is already complete
                    if not force_recompute and self.is_subject_complete(subject_id, session):
                        skipped_complete.append((subject_id, session))
                        self.logger.info(f"⏭️  Skipping {subject_id}/{session} - already complete")
                    else:
                        subjects.append((subject_id, session))
        
        if skipped_complete:
            self.logger.info(f"📋 Skipped {len(skipped_complete)} already completed subjects")
            if self.verbose:
                for subject_id, session in skipped_complete:
                    self.logger.debug(f"   - {subject_id}/{session}")
                    
        return subjects
    
    def resolve_subjects(self, subject_args: List[str], force_recompute: bool = False) -> List[Tuple[str, str]]:
        """
        Resolve subject selection arguments to actual subject/session pairs.
        
        Parameters
        ----------
        subject_args : List[str]
            Subject selection arguments
        force_recompute : bool
            If True, include subjects even if they are already complete
            
        Returns
        -------
        List[Tuple[str, str]]
            List of (subject_id, session) tuples
        """
        all_subjects = self.discover_subjects(force_recompute=force_recompute)
        
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
            cmd = [
                "python", str(self.src_dir / "markers/compute_doc_forest_markers_variable.py"),
                str(original_file),
                "--output", str(markers_dir / "markers_original.hdf5"),
                "--plot"
            ]
            if self.skip_smi:
                cmd.append("--skip-smi")
            if not self._run_command(cmd):
                return False
                
            self.logger.info("  Step 2: Computing features from markers_original.hdf5")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_features_variable.py"),
                str(markers_dir / "markers_original.hdf5"),
                "--output-scalars", str(features_dir / "scalars_original.npy"),
                "--output-topos", str(features_dir / "topos_original.npy")
            ]):
                return False
            
            # Process reconstructed data
            self.logger.info(f"  Step 3: Computing markers from {recon_file.name}")
            cmd = [
                "python", str(self.src_dir / "markers/compute_doc_forest_markers_variable.py"),
                str(recon_file),
                "--output", str(markers_dir / "markers_reconstructed.hdf5"),
                "--plot"
            ]
            if self.skip_smi:
                cmd.append("--skip-smi")
            if not self._run_command(cmd):
                return False
                
            self.logger.info("  Step 4: Computing features from markers_reconstructed.hdf5")
            if not self._run_command([
                "python", str(self.src_dir / "markers/compute_doc_forest_features_variable.py"),
                str(markers_dir / "markers_reconstructed.hdf5"),
                "--output-scalars", str(features_dir / "scalars_reconstructed.npy"),
                "--output-topos", str(features_dir / "topos_reconstructed.npy")
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
            
            # First check if we have enough successful subjects
            subjects_dir = self.results_dir / "SUBJECTS"
            if not subjects_dir.exists():
                self.logger.error("No subjects directory found")
                return False
                
            # Count subjects with complete data
            complete_subjects = []
            for subject_dir in subjects_dir.glob("sub-*"):
                if not subject_dir.is_dir():
                    continue
                for session_dir in subject_dir.glob("ses-*"):
                    if not session_dir.is_dir():
                        continue
                    features_dir = session_dir / "features_variable"
                    if (features_dir.exists() and 
                        (features_dir / "scalars_original.npy").exists() and
                        (features_dir / "scalars_reconstructed.npy").exists()):
                        complete_subjects.append(f"{subject_dir.name}/{session_dir.name}")
            
            self.logger.info(f"Found {len(complete_subjects)} subjects with complete data")
            if len(complete_subjects) < 2:
                self.logger.error("Need at least 2 subjects with complete data for model training")
                return False
            
            # Create global models output directory
            models_output_dir = self.results_dir / "EXTRATREES" / f"models_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            models_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get patient labels path from metadata directory
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            
            if not patient_labels_path.exists():
                self.logger.error(f"Patient labels file not found at: {patient_labels_path}")
                self.logger.error(f"Please check that the file exists and the --metadata-dir path is correct")
                self.logger.info(f"Looking for: patient_labels_with_controls.csv")
                self.logger.info(f"In directory: {self.metadata_dir}")
                return False
            
            self.logger.info(f"Using patient labels: {patient_labels_path}")
            
            # Train cross-data models for different marker types
            marker_types = ["scalar", "topo"]
            
            success = True
            for marker_type in marker_types:
                self.logger.info(f"Training cross-data {marker_type} models...")
                
                config_output_dir = models_output_dir / f"{marker_type}"
                
                # Use the new cross-data classification which trains both models and tests on both test sets
                if not self._run_command([
                    "python", str(self.src_dir / "model/extratrees.py"),
                    "--data-dir", str(self.results_dir / "SUBJECTS"),
                    "--patient-labels", str(patient_labels_path),
                    "--output-dir", str(config_output_dir),
                    "--marker-type", marker_type,
                    "--cross-data"  # This tells the script to use CrossDataClassifier
                ]):
                    self.logger.error(f"Failed to train cross-data {marker_type} models")
                    success = False
                else:
                    self.logger.info(f"✓ Cross-data {marker_type} models completed")
                    self.logger.info(f"   - Model A (original) trained and tested on both test sets")
                    self.logger.info(f"   - Model B (reconstructed) trained and tested on both test sets")
            
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
            analysis_dir = results_dir / "individual_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if required feature files exist
            features_dir = results_dir / "features_variable"
            required_files = [
                features_dir / "scalars_original.npy",
                features_dir / "scalars_reconstructed.npy",
                features_dir / "topos_original.npy",
                features_dir / "topos_reconstructed.npy"
            ]
            
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                self.logger.warning(f"Missing feature files for {subject_id}/{session}: {[f.name for f in missing_files]}")
                self.logger.warning(f"Skipping analysis for {subject_id}/{session}")
                return False
            
            self.logger.info(f"Running analysis for {subject_id}/{session}")
            
            # Get the fif data directory for GFP analysis
            fif_data_dir = self.data_dir / f"sub-{subject_id}" / session
            
            # Run individual analysis (replaces compare_markers.py)
            if not self._run_command([
                "python", str(self.src_dir / "analysis/individual_analysis.py"),
                str(results_dir),
                str(fif_data_dir),
                "--output", str(analysis_dir)
            ]):
                return False
            
            self.logger.info(f"✓ Analysis completed for {subject_id}/{session}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Analysis failed for {subject_id}/{session}: {e}")
            return False
    
    def run_decoder_phase(self) -> bool:
        """
        Run decoder phase across all processed subjects.
        This decodes original vs reconstructed EEG data.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("Running decoder analysis")
            
            # Create decoder output directory
            decoder_output_dir = self.results_dir / "DECODER" / f"decoder_results_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            decoder_output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Decoder output directory: {decoder_output_dir}")
            
            # Run decoder on all available subjects for robust population-level analysis
            # Memory optimization: decoder now uses float32 and n_jobs=1
            if not self._run_command([
                "python", str(self.src_dir / "decoder/decoder.py"),
                "--main_path", str(self.data_dir),
                "--output_dir", str(decoder_output_dir),
                "--cv", "3",
                "--n_jobs", "1",  # Avoid memory multiplication from parallelization
                "--verbose"
            ]):
                return False
            
            self.logger.info("✓ Decoder analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Decoder analysis failed: {e}")
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
            
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            
            if not patient_labels_path.exists():
                self.logger.error(f"Patient labels file not found at: {patient_labels_path}")
                self.logger.error(f"Please check that the file exists and the --metadata-dir path is correct")
                return False
            
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
    
    def run_statistical_analysis(self) -> bool:
        """
        Run statistical analysis across all processed subjects.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("Running statistical analysis")
            
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            
            if not patient_labels_path.exists():
                self.logger.error(f"Patient labels file not found at: {patient_labels_path}")
                self.logger.error(f"Please check that the file exists and the --metadata-dir path is correct")
                return False
            
            stats_output_dir = self.results_dir / "STATISTICS" / f"statistics_results_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            stats_output_dir.mkdir(parents=True, exist_ok=True)
            results_dir = self.results_dir / "SUBJECTS"
            
            if not self._run_command([
                "python", str(self.src_dir / "analysis/statistical_analysis.py"),
                "--results-dir", str(results_dir),
                "--output-dir", str(stats_output_dir),
                "--patient-labels", str(patient_labels_path)
            ]):
                return False
            
            self.logger.info("✓ Statistical analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Statistical analysis failed: {e}")
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
        
        # Don't modify instance variables here - will be handled by caller
        # This is important for parallel processing
        if success:
            self.logger.info(f"✓ Completed {subject_id}/{session}")
        else:
            self.logger.error(f"✗ Failed {subject_id}/{session}")
            
        return success
    
    def run_pipeline(self, subjects: List[Tuple[str, str]],
                    skip_markers: bool = False,
                    skip_models: bool = False,
                    skip_analysis: bool = False,
                    skip_decoder: bool = False,
                    skip_global: bool = False,
                    skip_statistics: bool = False) -> dict:
        """
        Run the complete pipeline for multiple subjects.
        
        Pipeline order:
        1. Individual subject processing (markers + analysis)
        2. Decoder analysis
        3. Global analysis
        4. Statistical analysis
        5. Model training

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
        skip_decoder : bool
            Skip decoder analysis phase
        skip_global : bool
            Skip global analysis phase
        skip_statistics : bool
            Skip statistical analysis phase
            
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
        
        # Phase 1: Process subjects
        if self.batch_size > 1:
            self._run_parallel(subjects, skip_markers, skip_models, skip_analysis)
        else:
            self._run_sequential(subjects, skip_markers, skip_models, skip_analysis)

        # Phase 2: Decoder Analysis
        if not skip_decoder and self.completed_subjects:
            self.logger.info("Starting decoder analysis phase...")
            decoder_success = self.run_decoder_phase()
            if not decoder_success:
                self.logger.error("Decoder analysis failed")

        # Phase 3: Global Analysis  
        if not skip_global and self.completed_subjects:
            self.run_global_analysis()
        
        # Phase 4: Statistical Analysis (after global)
        if not skip_statistics and self.completed_subjects:
            self.logger.info("Starting statistical analysis phase...")
            stats_success = self.run_statistical_analysis()
            if not stats_success:
                self.logger.error("Statistical analysis failed")
        
        # Phase 5: Models Training (run even if some subjects failed, as long as we have enough data)
        if not skip_models:
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
            success = self.process_subject(subject_id, session, skip_markers, skip_models, skip_analysis)
            
            # Track completed/failed subjects
            if success:
                self.completed_subjects.append((subject_id, session))
            else:
                self.failed_subjects.append((subject_id, session))
    
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
            
            # Use Popen with stderr redirected to stdout to avoid deadlock
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, universal_newlines=True, bufsize=1)
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(f">> {output.strip()}")
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
                
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
                            help='Only run analysis (markers + individual analysis)')
    phase_group.add_argument('--individual-analysis-only', action='store_true',
                            help='Only run individual analysis (requires existing markers)')
    phase_group.add_argument('--decoder-only', action='store_true',
                            help='Only run decoder analysis')
    phase_group.add_argument('--global-only', action='store_true',
                            help='Only run global analysis')
    phase_group.add_argument('--models-only', action='store_true',
                            help='Only run model training')
    phase_group.add_argument('--statistics-only', action='store_true',
                            help='Only run statistical analysis')
    
    # Individual phase skipping
    parser.add_argument('--skip-markers', action='store_true',
                       help='Skip markers computation phase')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis phase')
    parser.add_argument('--skip-decoder', action='store_true',
                       help='Skip decoder analysis phase')
    parser.add_argument('--skip-global', action='store_true',
                       help='Skip global analysis')
    parser.add_argument('--skip-statistics', action='store_true',
                       help='Skip statistical analysis phase')
    parser.add_argument('--skip-models', action='store_true', 
                       help='Skip model training phase')
    parser.add_argument('--skip-smi', action='store_true',
                       help='Skip SymbolicMutualInformation computation (recommended for large datasets)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of already completed subjects')
    
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

    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of parallel processes (default: 4)')
    
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
    skip_markers = (args.skip_markers or args.models_only or args.decoder_only or 
                   args.global_only or args.statistics_only or args.individual_analysis_only)
    skip_models = (args.skip_models or args.markers_only or args.analysis_only or 
                  args.decoder_only or args.global_only or args.statistics_only or 
                  args.individual_analysis_only)
    skip_analysis = (args.skip_analysis or args.markers_only or args.models_only or 
                    args.decoder_only or args.global_only or args.statistics_only)
    skip_decoder = (args.skip_decoder or args.markers_only or args.models_only or 
                   args.analysis_only or args.global_only or args.statistics_only or 
                   args.individual_analysis_only)
    skip_global = (args.skip_global or args.markers_only or args.models_only or 
                  args.analysis_only or args.decoder_only or args.individual_analysis_only)
    skip_statistics = (args.skip_statistics or args.markers_only or args.models_only or 
                      args.analysis_only or args.decoder_only or args.individual_analysis_only)
    
    # Create pipeline
    pipeline = Pipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        src_dir=args.src_dir,
        metadata_dir=args.metadata_dir,
        batch_size=args.batch_size,
        verbose=args.verbose,
        dry_run=args.dry_run,
        skip_smi=args.skip_smi
    )
    
    # Handle statistics-only mode
    if args.statistics_only:
        pipeline.run_statistical_analysis()
        return
    
    # Handle decoder-only mode
    if args.decoder_only:
        pipeline.run_decoder_phase()
        return
    
    # Handle global-only mode
    if args.global_only:
        pipeline.run_global_analysis()
        return
    
    # Handle models-only mode  
    if args.models_only:
        pipeline.run_models_phase()
        return

    # Resolve subjects
    subjects = pipeline.resolve_subjects(subject_args, force_recompute=args.force_recompute)
    
    if not subjects:
        print("No subjects to process")
        return
        
    # Run pipeline
    pipeline.run_pipeline(
        subjects=subjects,
        skip_markers=skip_markers,
        skip_models=skip_models, 
        skip_analysis=skip_analysis,
        skip_decoder=skip_decoder,
        skip_global=skip_global,
        skip_statistics=skip_statistics
    )


if __name__ == "__main__":
    main()
