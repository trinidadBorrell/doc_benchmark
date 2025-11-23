#!/usr/bin/env python3
"""
New Pipeline for EEG Foundation Benchmark

Comprehensive pipeline orchestrating decoder, markers, and model training phases.
Reads from: main_path/sub-{num}/ses-{num}/(orig|recon)/*.fif

Supports 3 settings:
1. DoC local-global: patient mode, lg task
2. Control local-global: control mode, lg task  
3. Control resting-state: control mode, rs task

Pipeline phases:
A. DECODER: decoder.py → analysis.py → viz.py → results/new_results/DECODER/
B. MARKERS: compute_markers_with_junifer.py → compute_data.py → generate_plots.py → 
            compute_scalars.py → compute_topographies.py → results/new_results/MARKERS/
C. MODEL: support_vector_machine.py → results/new_results/MODEL/ (only for cases 1 & 2)

Usage:
    # DoC local-global
    python pipeline.py --main-path /data/doc --metadata-dir /metadata --mode patient --task lg --all
    
    # Control local-global
    python pipeline.py --main-path /data/control_lg --metadata-dir /metadata --mode control --task lg --all
    
    # Control resting-state
    python pipeline.py --main-path /data/control_rs --metadata-dir /metadata --mode control --task rs --all

More examples:
    # Run all phases
    python pipeline.py \
    --main-path /data/project/eeg_foundation/data/test_data/fif_data_doc/ \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all

    # Skip decoder, run only markers and model
    python pipeline.py \
    --main-path /data/project/eeg_foundation/data/test_data/fif_data_doc/ \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all --skip-decoder

    # Run only markers phase
    python pipeline.py \
    --main-path /data/project/eeg_foundation/data/test_data/fif_data_doc/ \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all --skip-decoder --skip-model

    # Control resting-state (no trial type filtering in decoder, MODEL skipped automatically)
    python pipeline.py \
    --main-path /data/project/eeg_foundation/data/control/control_rs/ \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode control --task rs --all
    
Authors: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import sys
import subprocess
import logging
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class Pipeline:
    """Main pipeline class for EEG Foundation benchmark."""
    
    def __init__(self, 
                 main_path: str,
                 results_dir: str,
                 src_dir: str,
                 metadata_dir: str,
                 mode: str = 'patient',
                 task: str = 'lg',
                 batch_size: int = 1,
                 verbose: bool = False,
                 dry_run: bool = False,
                 # Decoder parameters
                 decoder_cv: int = 3,
                 decoder_n_jobs: int = 1,
                 # Markers parameters
                 markers_skip_clustering: bool = False,
                 markers_keep_h5: bool = False,
                 # Model parameters
                 model_marker_type: str = 'scalar'):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        main_path : str
            Path to main data directory with sub-{num}/ses-{num}/(orig|recon)/ structure
        results_dir : str
            Path to save results (will create new_results subdirectories)
        src_dir : str
            Path to the source code directory
        metadata_dir : str
            Path to metadata directory
        mode : str
            Analysis mode: 'patient' or 'control' (for decoder and model)
        task : str
            Task paradigm: 'lg' (local-global) or 'rs' (resting-state)
        batch_size : int
            Number of parallel processes for subject-level tasks
        verbose : bool
            Enable verbose logging
        dry_run : bool
            Show what would be done without executing
        decoder_cv : int
            Cross-validation folds for decoder
        decoder_n_jobs : int
            Parallel jobs for decoder
        markers_skip_clustering : bool
            Skip cluster permutation tests in markers
        markers_keep_h5 : bool
            Keep H5 files after markers computation
        model_marker_type : str
            Marker type for model training: 'scalar' or 'topo'
        """
        self.main_path = Path(main_path)
        self.results_dir = Path(results_dir)
        self.src_dir = Path(src_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Core settings
        self.mode = mode  # 'patient' or 'control'
        self.task = task  # 'lg' or 'rs'
        
        # Determine optimal batch size
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.batch_size = min(batch_size, max_workers)
        
        self.verbose = verbose
        self.dry_run = dry_run
        
        # Component parameters
        self.decoder_cv = decoder_cv
        self.decoder_n_jobs = decoder_n_jobs
        self.markers_skip_clustering = markers_skip_clustering
        self.markers_keep_h5 = markers_keep_h5
        self.model_marker_type = model_marker_type
        
        # Setup logging (must come before using self.logger)
        self._setup_logging()
        
        # Log parallelization info
        if self.batch_size > 1:
            self.logger.info(f"🚀 Parallel processing enabled: {self.batch_size} workers (CPU cores: {multiprocessing.cpu_count()})")
        else:
            self.logger.info(f"⚙️  Sequential processing (use --batch-size N for parallel processing)")
        
        # Validate directories and configuration
        self._validate_directories()
        self._validate_configuration()
        
        # Track progress
        self.completed_subjects = []
        self.failed_subjects = []
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.results_dir / "new_results" / "logs"
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
        if not self.main_path.exists():
            raise FileNotFoundError(f"Main data directory not found: {self.main_path}")
            
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.src_dir}")
            
        # Create results directory if it doesn't exist
        (self.results_dir / "new_results").mkdir(parents=True, exist_ok=True)
        
        # Check for required source files
        required_files = [
            "decoder/decoder.py",
            "decoder/analysis/analysis.py",
            "decoder/analysis/viz.py",
            "markers/compute_markers_batch_htcondor.py",
            "markers/report/compute_data.py",
            "markers/report/generate_plots.py",
            "markers/compute_scalars.py",
            "markers/compute_topographies.py",
            "model/support_vector_machine.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.src_dir / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            raise FileNotFoundError(f"Missing required source files: {missing_files}")
    
    def _validate_configuration(self):
        """Validate pipeline configuration parameters."""
        if self.mode not in ['patient', 'control']:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'patient' or 'control'")
        
        if self.task not in ['lg', 'rs']:
            raise ValueError(f"Invalid task '{self.task}'. Must be 'lg' or 'rs'")
        
        # Check YAML templates exist
        yaml_template = self._get_yaml_template()
        if not yaml_template.exists():
            raise FileNotFoundError(f"YAML template not found: {yaml_template}")
        
        self.logger.info(f"Configuration: mode={self.mode}, task={self.task}")
        self.logger.info(f"YAML template: {yaml_template}")
    
    def _get_yaml_template(self) -> Path:
        """Get appropriate YAML template based on task."""
        if self.task == 'lg':
            return self.src_dir / "markers" / "input" / "icm_complete_individual_markers_local_global.yaml"
        else:  # rs
            return self.src_dir / "markers" / "input" / "icm_complete_individual_markers_resting_state.yaml"
    
    def discover_subjects(self, force_recompute: bool = False) -> List[Tuple[str, str]]:
        """
        Discover all subjects with available sessions from main_path.
        Supports two directory structures:
        1. main_path/sub-{num}/ses-{num}/orig/*.fif and .../recon/*.fif
        2. main_path/sub-{num}/ses-{num}/*_original.fif and *_recon.fif
        
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
        
        for subject_dir in self.main_path.glob("sub-*"):
            if not subject_dir.is_dir():
                continue
                
            subject_id = subject_dir.name.replace("sub-", "")
            
            # Find all sessions for this subject
            for session_dir in subject_dir.glob("ses-*"):
                if not session_dir.is_dir():
                    continue
                    
                session = session_dir.name.replace("ses-", "")
                
                # Check two possible structures:
                # Structure 1: orig/recon subdirectories
                orig_dir = session_dir / "orig"
                recon_dir = session_dir / "recon"
                
                has_orig_dir = orig_dir.exists() and list(orig_dir.glob("*.fif"))
                has_recon_dir = recon_dir.exists() and list(recon_dir.glob("*.fif"))
                
                # Structure 2: files with _original/_recon suffixes in session dir
                has_orig_files = list(session_dir.glob("*_original.fif"))
                has_recon_files = list(session_dir.glob("*_recon.fif"))
                
                if has_orig_dir or has_recon_dir or has_orig_files or has_recon_files:
                    subjects.append((subject_id, session))
                    if has_orig_dir or has_recon_dir:
                        self.logger.debug(f"Found: sub-{subject_id}/ses-{session} (orig/:{has_orig_dir}, recon/:{has_recon_dir})")
                    else:
                        self.logger.debug(f"Found: sub-{subject_id}/ses-{session} (*_original:{bool(has_orig_files)}, *_recon:{bool(has_recon_files)})")
        
        subjects.sort()
        self.logger.info(f"📋 Discovered {len(subjects)} subject/session pairs")
        return subjects

    def is_subject_complete(self, subject_id: str, session: str) -> bool:
        """
        Check if a subject/session has completed all processing stages.
        
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
        # Check decoder results
        decoder_dir = self.results_dir / "new_results" / "DECODER" / f"sub-{subject_id}" / f"ses-{session}"
        decoder_complete = (decoder_dir / "data" / "decoding_results.pkl").exists()
        
        # Check markers results (both orig and recon)
        markers_dir = self.results_dir / "new_results" / "MARKERS"
        orig_scalars = (markers_dir / f"sub-{subject_id}_ses-{session}_orig" / f"scalars_{subject_id}_ses-{session}_orig.npz").exists()
        recon_scalars = (markers_dir / f"sub-{subject_id}_ses-{session}_recon" / f"scalars_{subject_id}_ses-{session}_recon.npz").exists()
        
        return decoder_complete and orig_scalars and recon_scalars
        
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
            raise ValueError("No subjects found in main_path directory")
            
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

    
    def run_decoder_phase(self) -> bool:
        """
        Run decoder phase for all subjects: decoder.py → analysis.py → viz.py
        Saves results to new_results/DECODER/
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE A: DECODER")
            self.logger.info("=" * 60)
            
            decoder_output_dir = self.results_dir / "new_results" / "DECODER"
            decoder_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Run decoder.py
            self.logger.info("Step A1: Running decoder.py...")
            cmd = [
                "python", str(self.src_dir / "decoder" / "decoder.py"),
                "--main_path", str(self.main_path),
                "--output_dir", str(decoder_output_dir),
                "--cv", str(self.decoder_cv),
                "--n_jobs", str(self.decoder_n_jobs),
                "--mode", self.mode
            ]
            
            # Add patient labels if in patient mode
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            if patient_labels_path.exists():
                cmd.extend(["--patient-labels", str(patient_labels_path)])
            
            # For resting-state, don't filter trial types (no stratification)
            if self.task == 'rs':
                self.logger.info("Resting-state: disabling trial type filtering")
                # decoder.py will not filter trial types when --filter_trial_types is not provided
            
            if self.verbose:
                cmd.append("--verbose")
            
            if not self._run_command(cmd):
                self.logger.error("decoder.py failed")
                return False
            
            # Find the most recent decoder results directory
            decoder_dirs = sorted(decoder_output_dir.glob("decoding-*"))
            if not decoder_dirs:
                self.logger.error("No decoder results directory found")
                return False
            
            latest_decoder_dir = decoder_dirs[-1]
            self.logger.info(f"Using decoder results from: {latest_decoder_dir}")
            
            # Step 2: Run analysis.py
            self.logger.info("Step A2: Running analysis.py...")
            cmd = [
                "python", str(self.src_dir / "decoder" / "analysis" / "analysis.py"),
                "--results-dir", str(latest_decoder_dir)
            ]
            
            if not self._run_command(cmd):
                self.logger.error("analysis.py failed")
                return False
            
            # Step 3: Run viz.py
            self.logger.info("Step A3: Running viz.py...")
            # Map mode: patient -> DOC, control -> control
            viz_mode = "DOC" if self.mode == "patient" else "control"
            cmd = [
                "python", str(self.src_dir / "decoder" / "analysis" / "viz.py"),
                "--results-dir", str(latest_decoder_dir),
                "--mode", viz_mode
            ]
            
            if not self._run_command(cmd):
                self.logger.error("viz.py failed")
                return False
            
            self.logger.info("✓ DECODER phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ DECODER phase failed: {e}")
            return False
    
    def run_markers_phase_for_subject(self, subject_id: str, session: str, file_type: str) -> bool:
        """
        Run markers phase for a single subject/session/file_type.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., '001')
        session : str
            Session identifier (e.g., '01')
        file_type : str
            'orig' or 'recon'
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
            
            # Try Structure 1: orig/recon subdirectories
            fif_folder = session_dir / file_type
            fif_files = []
            
            if fif_folder.exists():
                fif_files = list(fif_folder.glob("*.fif"))
            
            # Try Structure 2: files with _original/_recon suffixes
            if not fif_files:
                suffix = "_original.fif" if file_type == "orig" else "_recon.fif"
                fif_files = list(session_dir.glob(f"*{suffix}"))
                # For structure 2, the fif_folder is the session dir itself
                if fif_files:
                    fif_folder = session_dir
            
            if not fif_files:
                self.logger.warning(f"No .fif files found for {subject_id}/ses-{session}/{file_type}")
                return False
            
            fif_file = fif_files[0]  # Use first .fif file found
            
            # Output directory: MARKERS/sub-{ID}/ses-{num}/orig/ or MARKERS/sub-{ID}/ses-{num}/recon/
            subject_dir = self.results_dir / "new_results" / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            subject_markers_dir = session_dir_out / file_type  # orig or recon
            subject_markers_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if results already exist (check all required outputs)
            scalars_file = subject_markers_dir / f"scalars_{subject_id}_ses-{session}_{file_type}.npz"
            topos_file = subject_markers_dir / f"topos_{subject_id}_ses-{session}_{file_type}.npz"
            h5_file = subject_markers_dir / "icm_complete_features.h5"
            
            # Check for key output files to determine if processing is complete
            if scalars_file.exists() and topos_file.exists():
                self.logger.info(f"⏭️  Skipping {subject_id}/ses-{session}/{file_type} - results already exist")
                return True
            
            self.logger.info(f"Processing {subject_id}/ses-{session}/{file_type}")
            
            # Step 1: compute_markers_with_junifer.py (ALWAYS keep H5 until after report generation)
            self.logger.info("  Step B1: compute_markers_with_junifer.py...")
            
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_markers_with_junifer.py"),
                "--fif_folder", str(fif_folder),
                "--task", self.task,
                "--output-dir", str(subject_markers_dir),  # Use direct output directory
                "--skip-clustering",  # Always skip clustering (default for speed)
                "--keep-h5"  # ALWAYS keep H5 file - we'll delete it later after report generation
            ]
            
            if not self._run_command(cmd):
                self.logger.error("compute_markers_with_junifer.py failed")
                return False
            
            # Check if H5 file was created
            if not h5_file.exists():
                self.logger.error(f"H5 file not found: {h5_file}")
                return False
            
            # Step 2: compute_data.py
            self.logger.info("  Step B2: compute_data.py...")
            cmd = [
                "python", str(self.src_dir / "markers" / "report" / "compute_data.py"),
                "--subject_id", subject_id,
                "--h5_file", str(h5_file),
                "--fif_file", str(fif_file),
                "--output_dir", str(subject_markers_dir),
                "--task", self.task
            ]
            
            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")
            
            if not self._run_command(cmd):
                self.logger.error("compute_data.py failed")
                return False
            
            # Step 3: compute_scalars.py
            self.logger.info("  Step B3: compute_scalars.py...")
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_scalars.py"),
                "--h5_file", str(h5_file),
                "--output_file", str(scalars_file)
            ]
            
            if not self._run_command(cmd):
                self.logger.error("compute_scalars.py failed")
                return False
            
            # Step 4: compute_topographies.py
            self.logger.info("  Step B4: compute_topographies.py...")
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_topographies.py"),
                "--h5_file", str(h5_file),
                "--output_file", str(topos_file)
            ]
            
            if not self._run_command(cmd):
                self.logger.error("compute_topographies.py failed")
                return False
            
            # Verify all outputs were created
            if not scalars_file.exists():
                self.logger.error(f"Scalars file not created: {scalars_file}")
                return False
            if not topos_file.exists():
                self.logger.error(f"Topographies file not created: {topos_file}")
                return False
            
            self.logger.info(f"✓ Markers completed for {subject_id}/ses-{session}/{file_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Markers failed for {subject_id}/ses-{session}/{file_type}: {e}")
            return False
    
    def run_markers_phase_for_all_subjects(self, subjects: List[Tuple[str, str]]) -> bool:
        """
        Run markers phase for all subjects using BATCH HTCondor processing.
        
        NEW BATCH WORKFLOW:
        1. Run compute_markers_batch_htcondor.py ONCE for ALL subject/sessions (generates all H5 files)
        2. For each subject/session: compute_data → compute_scalars → compute_topographies → report
        3. After EVERYTHING is done: cleanup H5 files (if not keeping them)
        
        Parameters
        ----------
        subjects : List[Tuple[str, str]]
            List of (subject_id, session) tuples
            
        Returns
        -------
        bool
            True if all successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE B: MARKERS (BATCH HTCondor)")
            self.logger.info("=" * 60)
            self.logger.info("Batch workflow: batch markers → compute_data → compute_scalars → compute_topographies → reports → cleanup")
            self.logger.info("=" * 60)
            
            markers_dir = self.results_dir / "new_results" / "MARKERS"
            markers_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Run batch HTCondor processing for ALL subjects (generates ALL H5 files)
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: BATCH MARKER COMPUTATION (HTCondor)")
            self.logger.info("=" * 60)
            self.logger.info(f"Processing {len(subjects)} subject/session pairs in batch...")
            
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_markers_batch_htcondor.py"),
                "--fif_folder", str(self.main_path),
                "--results-dir", str(markers_dir),
                "--task", self.task,
                "--batch-size", "40",  # Process 40 jobs in parallel on HTCondor
                "--keep-h5"  # ALWAYS keep H5 until after all processing
            ]
            
            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")
            
            if not self._run_command(cmd):
                self.logger.error("Batch marker computation failed")
                return False
            
            self.logger.info("✓ Batch marker computation completed")
            
            # Step 2: Process each subject/session (compute_data, compute_scalars, compute_topographies, reports)
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("STEP 2: PROCESSING MARKER RESULTS")
            self.logger.info("=" * 60)
            
            success_count = 0
            failed_subjects = []
            
            for subject_id, session in subjects:
                self.logger.info(f"\nProcessing results for {subject_id}/ses-{session}...")
                
                # Check if already complete
                if self._is_subject_complete(subject_id, session):
                    self.logger.info(f"⏭️  Skipping {subject_id}/ses-{session} - all outputs already exist")
                    success_count += 1
                    continue
                
                subject_success = True
                
                # Process both orig and recon
                for file_type in ["original", "recon"]:
                    self.logger.info(f"  Processing {file_type} data...")
                    if not self._process_markers_for_file_type(subject_id, session, file_type):
                        self.logger.error(f"Failed to process {file_type} data for {subject_id}/ses-{session}")
                        subject_success = False
                        failed_subjects.append((subject_id, session, file_type))
                
                # Generate report if both succeeded
                if subject_success:
                    self.logger.info(f"  Generating report...")
                    if not self._generate_report_for_subject(subject_id, session):
                        self.logger.warning(f"Report generation failed for {subject_id}/ses-{session} (continuing anyway)")
                    
                    success_count += 1
                    self.logger.info(f"✓ Completed {subject_id}/ses-{session}")
                else:
                    self.logger.error(f"✗ Skipping report for {subject_id}/ses-{session} due to processing failures")
            
            # Step 3: Cleanup ALL H5 files (only if not keeping them)
            if not self.markers_keep_h5:
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info("STEP 3: CLEANING UP H5 FILES")
                self.logger.info("=" * 60)
                self._cleanup_all_h5_files(markers_dir)
            else:
                self.logger.info("\nKeeping H5 files (--keep-h5 flag set)")
            
            # Summary
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("MARKERS PHASE SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"Successfully processed: {success_count}/{len(subjects)} subjects")
            
            if failed_subjects:
                self.logger.warning(f"Failed processing for {len(failed_subjects)} subject/session/file_type combinations:")
                for subj_id, sess, ftype in failed_subjects:
                    self.logger.warning(f"  - {subj_id}/ses-{sess}/{ftype}")
            
            self.logger.info("✓ MARKERS phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ MARKERS phase failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _is_subject_complete(self, subject_id: str, session: str) -> bool:
        """
        Check if a subject/session has been completely processed.
        
        Verifies that ALL required outputs exist:
        - orig: scalars, topos, pkl files
        - recon: scalars, topos, pkl files
        - HTML report
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
            
        Returns
        -------
        bool
            True if all outputs exist, False otherwise
        """
        try:
            subject_dir = self.results_dir / "new_results" / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            original_dir = session_dir_out / "original"
            recon_dir = session_dir_out / "recon"
            reports_dir = session_dir_out / "reports"
            
            # Check original outputs
            original_scalars = original_dir / f"scalars_{subject_id}_ses-{session}_original.npz"
            original_topos = original_dir / f"topos_{subject_id}_ses-{session}_original.npz"
            
            # Check recon outputs
            recon_scalars = recon_dir / f"scalars_{subject_id}_ses-{session}_recon.npz"
            recon_topos = recon_dir / f"topos_{subject_id}_ses-{session}_recon.npz"
            
            # Check report output
            report_file = reports_dir / f"sub-{subject_id}_ses-{session}_report_comparison.html"
            
            # Check for at least some pkl files (from compute_data.py)
            original_pkl_files = list(original_dir.glob("*.pkl")) if original_dir.exists() else []
            recon_pkl_files = list(recon_dir.glob("*.pkl")) if recon_dir.exists() else []
            
            # All required files must exist
            required_files = [
                original_scalars,
                original_topos,
                recon_scalars,
                recon_topos,
                report_file
            ]
            
            all_exist = all(f.exists() for f in required_files)
            has_pkl_files = len(original_pkl_files) > 0 and len(recon_pkl_files) > 0
            
            if all_exist and has_pkl_files:
                self.logger.debug(f"All outputs verified for {subject_id}/ses-{session}")
                return True
            else:
                # Log what's missing for debugging
                missing = []
                if not original_scalars.exists():
                    missing.append("original scalars")
                if not original_topos.exists():
                    missing.append("original topos")
                if not recon_scalars.exists():
                    missing.append("recon scalars")
                if not recon_topos.exists():
                    missing.append("recon topos")
                if not report_file.exists():
                    missing.append("HTML report")
                if len(original_pkl_files) == 0:
                    missing.append("original pkl files")
                if len(recon_pkl_files) == 0:
                    missing.append("recon pkl files")
                
                if missing:
                    self.logger.debug(f"Missing outputs for {subject_id}/ses-{session}: {', '.join(missing)}")
                
                return False
                
        except Exception as e:
            self.logger.debug(f"Error checking completion status for {subject_id}/ses-{session}: {e}")
            return False
    
    def _generate_report_for_subject(self, subject_id: str, session: str) -> bool:
        """
        Generate HTML report for a subject/session (combines orig and recon data).
        
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
            subject_dir = self.results_dir / "new_results" / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            original_dir = session_dir_out / "original"
            recon_dir = session_dir_out / "recon"
            
            # Find FIF files - support both structures
            session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
            
            # Try Structure 1: orig/recon subdirectories
            orig_fif_folder = session_dir / "orig"
            recon_fif_folder = session_dir / "recon"
            
            orig_fif = list(orig_fif_folder.glob("*.fif"))[0] if orig_fif_folder.exists() and list(orig_fif_folder.glob("*.fif")) else None
            recon_fif = list(recon_fif_folder.glob("*.fif"))[0] if recon_fif_folder.exists() and list(recon_fif_folder.glob("*.fif")) else None
            
            # Try Structure 2: files with _original/_recon suffixes
            if not orig_fif:
                orig_files = list(session_dir.glob("*_original.fif"))
                orig_fif = orig_files[0] if orig_files else None
            
            if not recon_fif:
                recon_files = list(session_dir.glob("*_recon.fif"))
                recon_fif = recon_files[0] if recon_files else None
            
            # Check prerequisites
            if not (orig_fif and recon_fif):
                self.logger.error(f"Missing FIF files for {subject_id}/ses-{session}")
                return False
            
            if not (original_dir.exists() and recon_dir.exists()):
                self.logger.error(f"Missing marker directories for {subject_id}/ses-{session}")
                return False
            
            # Find H5 files (from batch processing location)
            markers_dir = self.results_dir / "new_results" / "MARKERS"
            orig_h5 = markers_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session}" / "original.h5"
            recon_h5 = markers_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session}" / "recon.h5"
            
            if not (orig_h5.exists() and recon_h5.exists()):
                self.logger.error(f"Missing H5 files for {subject_id}/ses-{session}")
                self.logger.error(f"  orig H5: {orig_h5} - exists: {orig_h5.exists()}")
                self.logger.error(f"  recon H5: {recon_h5} - exists: {recon_h5.exists()}")
                return False
            
            # Reports go in MARKERS/sub-{ID}/ses-{num}/reports/
            output_dir = session_dir_out / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "python", str(self.src_dir / "markers" / "report" / "generate_plots.py"),
                "--subject_id", subject_id,
                "--session", session,
                "--h5_file", str(orig_h5),  # Can use either, script needs both via data_dir
                "--fif_file", str(orig_fif),
                "--data_dir_original", str(original_dir),
                "--data_dir_recon", str(recon_dir),
                "--output_dir", str(output_dir)
            ]
            
            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")
            
            # Run and check success
            if self._run_command(cmd):
                # Verify report was created
                report_file = output_dir / f"sub-{subject_id}_ses-{session}_report_comparison.html"
                if report_file.exists():
                    self.logger.info(f"✓ Report generated: {report_file}")
                    return True
                else:
                    self.logger.warning(f"Report command succeeded but file not found: {report_file}")
                    return False
            else:
                self.logger.error(f"Report generation command failed for {subject_id}/ses-{session}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during report generation for {subject_id}/ses-{session}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _cleanup_h5_files(self, subject_id: str, session: str) -> None:
        """
        Delete H5 files after all processing is complete.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        """
        try:
            subject_dir = self.results_dir / "new_results" / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            
            h5_files = [
                session_dir_out / "orig" / "icm_complete_features.h5",
                session_dir_out / "recon" / "icm_complete_features.h5"
            ]
            
            for h5_file in h5_files:
                if h5_file.exists():
                    try:
                        h5_file.unlink()
                        self.logger.info(f"  Deleted: {h5_file}")
                    except Exception as e:
                        self.logger.warning(f"  Could not delete {h5_file}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Exception during H5 cleanup for {subject_id}/ses-{session}: {e}")
    
    def _cleanup_all_h5_files(self, markers_dir: Path) -> None:
        """
        Delete ALL H5 files in the markers directory after all processing is complete.
        
        Parameters
        ----------
        markers_dir : Path
            Path to MARKERS directory
        """
        try:
            h5_files = list(markers_dir.glob("h5_files/**/*.h5"))
            
            deleted_count = 0
            for h5_file in h5_files:
                try:
                    h5_file.unlink()
                    deleted_count += 1
                    self.logger.debug(f"  Deleted: {h5_file}")
                except Exception as e:
                    self.logger.warning(f"  Could not delete {h5_file}: {e}")
            
            self.logger.info(f"✓ Deleted {deleted_count} H5 files")
            
            # Remove empty h5_files directory
            h5_dir = markers_dir / "h5_files"
            if h5_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(h5_dir)
                    self.logger.info(f"✓ Removed h5_files directory")
                except Exception as e:
                    self.logger.warning(f"Could not remove h5_files directory: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Exception during H5 cleanup: {e}")
    
    def _process_markers_for_file_type(self, subject_id: str, session: str, file_type: str) -> bool:
        """
        Process markers for a single subject/session/file_type.
        Runs: compute_data → compute_scalars → compute_topographies
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        file_type : str
            'original' or 'recon'
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            markers_dir = self.results_dir / "new_results" / "MARKERS"
            
            # H5 file location (from batch processing)
            h5_file = markers_dir / "h5_files" / f"sub-{subject_id}" / f"ses-{session}" / f"{file_type}.h5"
            
            if not h5_file.exists():
                self.logger.error(f"H5 file not found: {h5_file}")
                return False
            
            # Output directory
            subject_dir = markers_dir / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            output_dir = session_dir_out / file_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find FIF file
            session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
            
            # Try different file structures
            if file_type == "original":
                fif_folder = session_dir / "orig"
                if not fif_folder.exists():
                    # Try _original suffix
                    fif_files = list(session_dir.glob("*_original.fif"))
                    if fif_files:
                        fif_file = fif_files[0]
                    else:
                        self.logger.error(f"No original FIF file found for {subject_id}/ses-{session}")
                        return False
                else:
                    fif_files = list(fif_folder.glob("*.fif"))
                    if not fif_files:
                        self.logger.error(f"No FIF files in {fif_folder}")
                        return False
                    fif_file = fif_files[0]
            else:  # recon
                fif_folder = session_dir / "recon"
                if not fif_folder.exists():
                    # Try _recon suffix
                    fif_files = list(session_dir.glob("*_recon.fif"))
                    if fif_files:
                        fif_file = fif_files[0]
                    else:
                        self.logger.error(f"No recon FIF file found for {subject_id}/ses-{session}")
                        return False
                else:
                    fif_files = list(fif_folder.glob("*.fif"))
                    if not fif_files:
                        self.logger.error(f"No FIF files in {fif_folder}")
                        return False
                    fif_file = fif_files[0]
            
            # Step 1: compute_data.py
            self.logger.debug(f"    Running compute_data.py...")
            cmd = [
                "python", str(self.src_dir / "markers" / "report" / "compute_data.py"),
                "--subject_id", subject_id,
                "--h5_file", str(h5_file),
                "--fif_file", str(fif_file),
                "--output_dir", str(output_dir),
                "--task", self.task
            ]
            
            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")
            
            if not self._run_command(cmd):
                self.logger.error("compute_data.py failed")
                return False
            
            # Step 2: compute_scalars.py
            self.logger.debug(f"    Running compute_scalars.py...")
            scalars_file = output_dir / f"scalars_{subject_id}_ses-{session}_{file_type}.npz"
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_scalars.py"),
                "--h5_file", str(h5_file),
                "--output_file", str(scalars_file)
            ]
            
            if not self._run_command(cmd):
                self.logger.error("compute_scalars.py failed")
                return False
            
            # Step 3: compute_topographies.py
            self.logger.debug(f"    Running compute_topographies.py...")
            topos_file = output_dir / f"topos_{subject_id}_ses-{session}_{file_type}.npz"
            cmd = [
                "python", str(self.src_dir / "markers" / "compute_topographies.py"),
                "--h5_file", str(h5_file),
                "--output_file", str(topos_file)
            ]
            
            if not self._run_command(cmd):
                self.logger.error("compute_topographies.py failed")
                return False
            
            # Verify outputs
            if not scalars_file.exists() or not topos_file.exists():
                self.logger.error(f"Missing output files for {subject_id}/ses-{session}/{file_type}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing markers for {subject_id}/ses-{session}/{file_type}: {e}")
            return False
    
    def run_model_phase(self) -> bool:
        """
        Run model training phase (only for cases 1 and 2, not resting-state).
        Uses support_vector_machine.py for VS vs MCS classification.
        Saves results to new_results/MODEL/
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Skip model phase for resting-state
            if self.task == 'rs':
                self.logger.info("Skipping MODEL phase for resting-state data")
                return True
            
            self.logger.info("=" * 60)
            self.logger.info("PHASE C: MODEL")
            self.logger.info("=" * 60)
            
            model_output_dir = self.results_dir / "new_results" / "MODEL"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check patient labels file
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            if not patient_labels_path.exists():
                self.logger.error(f"Patient labels file not found: {patient_labels_path}")
                return False
            
            markers_dir = self.results_dir / "new_results" / "MARKERS"
            
            # Run SVM with cross-data classification
            self.logger.info(f"Training SVM classifier ({self.model_marker_type} markers)...")
            cmd = [
                "python", str(self.src_dir / "model" / "support_vector_machine.py"),
                "--data-dir", str(markers_dir),
                "--patient-labels", str(patient_labels_path),
                "--marker-type", self.model_marker_type,
                "--output-dir", str(model_output_dir / self.model_marker_type),
                "--cross-data"  # Enable cross-data classification
            ]
            
            if not self._run_command(cmd):
                self.logger.error("support_vector_machine.py failed")
                return False
            
            self.logger.info("✓ MODEL phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ MODEL phase failed: {e}")
            return False
    
    def run_pipeline(self, subjects: List[Tuple[str, str]], 
                    skip_decoder: bool = False,
                    skip_markers: bool = False,
                    skip_model: bool = False) -> dict:
        """
        Run the complete pipeline for the specified subjects.
        
        Pipeline flow:
        A. DECODER phase: decoder.py → analysis.py → viz.py
        B. MARKERS phase: For each subject, both orig and recon:
           - compute_markers_with_junifer.py
           - compute_data.py
           - compute_scalars.py
           - compute_topographies.py
           - generate_plots.py (combines orig and recon)
        C. MODEL phase: support_vector_machine.py (skip for resting-state)
        
        Parameters
        ----------
        subjects : List[Tuple[str, str]]
            List of (subject_id, session) tuples to process
        skip_decoder : bool
            Skip DECODER phase
        skip_markers : bool
            Skip MARKERS phase
        skip_model : bool
            Skip MODEL phase
            
        Returns
        -------
        dict
            Summary of pipeline execution
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE START")
        self.logger.info("=" * 70)
        self.logger.info(f"Configuration: mode={self.mode}, task={self.task}")
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Subjects to process: {len(subjects)}")
        self.logger.info(f"Subjects: {subjects}")
        self.logger.info("=" * 70)
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual processing")
            return {"status": "dry_run", "subjects": subjects}
        
        results = {
            "decoder": False if not skip_decoder else "skipped",
            "markers": False if not skip_markers else "skipped",
            "model": False if not skip_model else "skipped"
        }
        
        # Phase A: DECODER
        if not skip_decoder:
            self.logger.info("\n" + "=" * 70)
            if self.run_decoder_phase():
                results["decoder"] = True
            else:
                self.logger.error("DECODER phase failed - continuing with MARKERS phase")
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE A: DECODER - SKIPPED")
        
        # Phase B: MARKERS
        if not skip_markers:
            self.logger.info("\n" + "=" * 70)
            if self.run_markers_phase_for_all_subjects(subjects):
                results["markers"] = True
            else:
                self.logger.error("MARKERS phase failed - continuing with MODEL phase")
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE B: MARKERS - SKIPPED")
        
        # Phase C: MODEL (only for lg task, not rs)
        if not skip_model and self.task != 'rs':
            self.logger.info("\n" + "=" * 70)
            if self.run_model_phase():
                results["model"] = True
            else:
                self.logger.error("MODEL phase failed")
        elif skip_model:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE C: MODEL - SKIPPED")
        # else: task == 'rs', model phase skipped automatically
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Results:")
        decoder_status = '✓ Success' if results['decoder'] is True else ('⏸ Skipped' if results['decoder'] == 'skipped' else '✗ Failed')
        markers_status = '✓ Success' if results['markers'] is True else ('⏸ Skipped' if results['markers'] == 'skipped' else '✗ Failed')
        model_status = '✓ Success' if results['model'] is True else ('⏸ Skipped' if results['model'] == 'skipped' else '✗ Failed')
        
        self.logger.info(f"  - DECODER: {decoder_status}")
        self.logger.info(f"  - MARKERS: {markers_status}")
        if self.task != 'rs':
            self.logger.info(f"  - MODEL: {model_status}")
        else:
            self.logger.info(f"  - MODEL: Skipped (resting-state)")
        self.logger.info("=" * 70)
        
        return {
            "status": "complete",
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "results": results,
            "subjects": subjects
        }
    
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
        description="EEG Foundation Benchmark Pipeline - Supports 3 settings: DoC local-global, Control local-global, Control resting-state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # DoC local-global
    python pipeline.py --main-path /data/doc --metadata-dir /metadata --mode patient --task lg --all
    
    # Control local-global
    python pipeline.py --main-path /data/control_lg --metadata-dir /metadata --mode control --task lg --all
    
    # Control resting-state
    python pipeline.py --main-path /data/control_rs --metadata-dir /metadata --mode control --task rs --all
        """
    )

    # Set default paths relative to script location
    project_root = str(Path(__file__).parent.parent)
    
    # Core configuration
    parser.add_argument('--main-path', required=True,
                       help='Path to main data directory with sub-{num}/ses-{num}/(orig|recon)/ structure')
    parser.add_argument('--metadata-dir', required=True,
                       help='Path to metadata directory (must contain patient_labels_with_controls.csv)')
    parser.add_argument('--mode', choices=['patient', 'control'], required=True,
                       help='Analysis mode: patient (DoC) or control')
    parser.add_argument('--task', choices=['lg', 'rs'], required=True,
                       help='Task paradigm: lg (local-global) or rs (resting-state)')
    
    # Subject selection (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument('--subject', 
                              help='Process single subject ID')
    subject_group.add_argument('--subjects',
                              help='Process specific subjects (comma-separated IDs)')
    subject_group.add_argument('--all', action='store_true',
                              help='Process all available subjects')
    subject_group.add_argument('--random', type=int, metavar='N',
                              help='Process N random subjects')
    
    # Optional configuration
    parser.add_argument('--results-dir',
                       default=f'{project_root}/results',
                       help='Path to results directory (default: ./results)')
    parser.add_argument('--src-dir',
                       default=f'{project_root}/src',
                       help='Path to source code directory (default: ./src)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of parallel processes (default: 1, sequential)')
    
    # Decoder parameters
    parser.add_argument('--decoder-cv', type=int, default=3,
                       help='Cross-validation folds for decoder (default: 3)')
    parser.add_argument('--decoder-n-jobs', type=int, default=1,
                       help='Parallel jobs for decoder (default: 1)')
    
    # Markers parameters
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip cluster permutation tests in markers computation')
    parser.add_argument('--keep-h5', action='store_true',
                       help='Keep H5 files after markers computation')
    
    # Model parameters
    parser.add_argument('--model-marker-type', choices=['scalar', 'topo'], default='scalar',
                       help='Marker type for model training (default: scalar)')
    
    # Phase skipping
    parser.add_argument('--skip-decoder', action='store_true',
                       help='Skip DECODER phase')
    parser.add_argument('--skip-markers', action='store_true',
                       help='Skip MARKERS phase')
    parser.add_argument('--skip-model', action='store_true',
                       help='Skip MODEL phase')
    
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
    
    # Create pipeline
    pipeline = Pipeline(
        main_path=args.main_path,
        results_dir=args.results_dir,
        src_dir=args.src_dir,
        metadata_dir=args.metadata_dir,
        mode=args.mode,
        task=args.task,
        batch_size=args.batch_size,
        verbose=args.verbose,
        dry_run=args.dry_run,
        decoder_cv=args.decoder_cv,
        decoder_n_jobs=args.decoder_n_jobs,
        markers_skip_clustering=args.skip_clustering,
        markers_keep_h5=args.keep_h5,
        model_marker_type=args.model_marker_type
    )
    
    # Resolve subjects
    subjects = pipeline.resolve_subjects(subject_args)
    
    if not subjects:
        print("No subjects to process")
        return
    
    # Run pipeline
    result = pipeline.run_pipeline(
        subjects,
        skip_decoder=args.skip_decoder,
        skip_markers=args.skip_markers,
        skip_model=args.skip_model
    )
    
    # Exit with appropriate code
    if result.get("status") == "complete":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
