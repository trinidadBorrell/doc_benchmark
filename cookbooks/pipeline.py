#!/usr/bin/env python3
"""
Unified Pipeline for EEG Foundation Benchmark

Comprehensive pipeline orchestrating five independent analysis phases for
benchmarking EEG foundation models on Disorders of Consciousness (DoC) data.
Supports multiple data structures (CBraMod, TOTEM, LaBram, standard BIDS).

Supported Data Structures
--------------------------
1. Standard:     main_path/sub-{id}/ses-{num}/(orig|recon)/*.fif
2. Suffix-based: main_path/sub-{id}/ses-{num}/*_original.fif and *_recon.fif
3. CBraMod:      main_path/sub-{id}/ses-{num}/sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif
4. BIDS:         main_path/sub-{id}/ses-{num}/eeg/*.fif
5. Single-file:  main_path/sub-{id}/ses-{num}/*.fif  (reconstructed-only fallback)

Pipeline Phases
---------------
A. GENERAL_METRICS  compute_metrics.py          → MAPE & Pearson correlation
B. FM_EMBEDDING    fm_embedding_classifier.py → MLP/RF/KR on embeddings
                    Supports three sub-modes via CLI flags forwarded to the script:
                    • default             — binary VS vs MCS classification
                    • --marker-regression — Ridge regression: embeddings → scalar markers
                    • --full-metric-prediction — 5 clinical targets (crs, etiology,
                                                 cs_6m, cs_1y, cs_2y)
C. DECODER          decoder.py → analysis.py → viz.py
D. MARKERS          compute_markers_with_junifer.py → compute_scalars.py
                    → compute_topographies.py  (parallel per subject)
E. MODEL            support_vector_machine.py   → SVM classification (VS vs MCS)

Usage Examples
--------------
# CBraMod DoC patients — all phases
python cookbooks/pipeline.py \\
    --main-path /data/CbraMod/recon_data_inference \\
    --metadata-dir /data/metadata \\
    --mode patient --task lg --data-source CBraMod \\
    --results-subdir CBraMod/doc_patients --all

# Standard DoC local-global — all phases
python cookbooks/pipeline.py \\
    --main-path /data/doc --metadata-dir /data/metadata \\
    --mode patient --task lg --all

# Control resting-state — skip MODEL phase (auto-skipped for rs)
python cookbooks/pipeline.py \\
    --main-path /data/control_rs --metadata-dir /data/metadata \\
    --mode control --task rs --all

# Only markers phase
python cookbooks/pipeline.py \\
    --main-path /data --metadata-dir /data/metadata \\
    --mode patient --task lg --all --markers-only

# FM embedding phase: binary VS/MCS with 5-fold CV
python cookbooks/pipeline.py \\
    --main-path /data --metadata-dir /data/metadata \\
    --mode patient --task lg --all --fm-embedding-only \\
    --embedding-data-dir /data/embeddings/totem \\
    --emb-full-cv

# MODEL phase: SVM with baseline CSV and full CV
python cookbooks/pipeline.py \\
    --main-path /data --metadata-dir /data/metadata \\
    --mode patient --task lg --all --model-only \\
    --model-baseline-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \\
    --model-full-cv --model-use-subject-intersection

# Skip general metrics and decoder, run everything else
python cookbooks/pipeline.py \\
    --main-path /data --metadata-dir /data/metadata \\
    --mode patient --task lg --all \\
    --skip-general-metrics --skip-decoder

Phase-Specific Flags
--------------------
--general-metrics-only   Only GENERAL_METRICS
--fm-embedding-only     Only FM_EMBEDDING
--decoder-only           Only DECODER
--markers-only           Only MARKERS
--model-only             Only MODEL

--skip-general-metrics   Skip GENERAL_METRICS
--skip-fm-embedding     Skip FM_EMBEDDING
--skip-decoder           Skip DECODER
--skip-markers           Skip MARKERS
--skip-model             Skip MODEL
"""

import argparse
import os
import sys
import subprocess
import logging
import random
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing


class Pipeline:
    """Main pipeline class for EEG Foundation benchmark."""

    def __init__(
        self,
        main_path: str,
        results_dir: str,
        src_dir: str,
        metadata_dir: str,
        mode: str = "patient",
        task: str = "lg",
        batch_size: int = 1,
        verbose: bool = False,
        dry_run: bool = False,
        # Decoder parameters
        decoder_cv: int = 3,
        decoder_n_jobs: int = 14,
        # Markers parameters
        markers_skip_clustering: bool = False,
        markers_keep_h5: bool = False,
        # Model parameters
        model_marker_type: str = "scalar",
        model_orig_data_dir: str = None,
        model_baseline_csv: str = None,
        model_use_subject_intersection: bool = False,
        model_full_cv: bool = False,
        # MLP Embedding parameters
        embedding_data_dir: str = None,
        mlp_n_epochs: int = 500,
        mlp_lr: float = 1e-3,
        mlp_batch_size: int = 32,
        emb_full_cv: bool = False,
        emb_use_subject_intersection: bool = False,
        emb_all_embedding_dirs: list = None,
        emb_marker_regression: bool = False,
        emb_marker_csv: str = None,
        emb_marker_reduction: str = "A",
        emb_full_metric_prediction: bool = False,
        emb_patient_labels_full: str = None,
        # New parameters for CBraMod and flexible data handling
        data_source: str = "auto",
        results_subdir: str = None,
        original_data_paths: list = None,
        file_pattern: str = None,
        save_time: bool = False,
    ):
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
        model_orig_data_dir : str, optional
            Path to a separate directory tree with the original (non-reconstructed)
            marker files (sub-{ID}/ses-{N}/orig/scalars_*.npz).  When set, the
            standard MARKERS output dir is used for recon and this path for orig.
        embedding_data_dir : str
            Path to pre-computed embeddings directory (for FM embedding phase)
        mlp_n_epochs : int
            Maximum training epochs for FM embedding classifier
        mlp_lr : float
            Learning rate for FM embedding classifier
        mlp_batch_size : int
            Batch size for FM embedding classifier
        data_source : str
            Data source identifier: 'auto', 'CBraMod', 'TOTEM', 'LaBram', 'standard'
        results_subdir : str
            Subdirectory within results for output (e.g., 'CBraMod/doc_patients')
        original_data_paths : list of str
            Paths to original data if separate from main_path (for comparison metrics).
            Multiple paths are searched in order until a match is found.
        file_pattern : str
            Custom file pattern to match (e.g., '*_vqnsp_reconstructed_epo.fif')
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
        self.model_orig_data_dir = model_orig_data_dir
        self.model_baseline_csv = model_baseline_csv
        self.model_use_subject_intersection = model_use_subject_intersection
        self.model_full_cv = model_full_cv

        # MLP Embedding parameters
        self.embedding_data_dir = (
            Path(embedding_data_dir) if embedding_data_dir else None
        )
        self.mlp_n_epochs = mlp_n_epochs
        self.mlp_lr = mlp_lr
        self.mlp_batch_size = mlp_batch_size
        self.emb_full_cv = emb_full_cv
        self.emb_use_subject_intersection = emb_use_subject_intersection
        self.emb_all_embedding_dirs = emb_all_embedding_dirs or []
        self.emb_marker_regression = emb_marker_regression
        self.emb_marker_csv = emb_marker_csv
        self.emb_marker_reduction = emb_marker_reduction
        self.emb_full_metric_prediction = emb_full_metric_prediction
        self.emb_patient_labels_full = emb_patient_labels_full

        # New CBraMod and flexible data handling parameters
        self.data_source = data_source
        self.results_subdir = results_subdir
        self.original_data_paths = (
            [Path(p) for p in original_data_paths] if original_data_paths else []
        )
        self.file_pattern = file_pattern

        # Auto-detect data source if set to 'auto'
        if self.data_source == "auto":
            self.data_source = self._detect_data_source()

        # Setup logging (must come before using self.logger)
        self._setup_logging()

        # Log parallelization info
        if self.batch_size > 1:
            self.logger.info(
                f"🚀 Parallel processing enabled: {self.batch_size} workers (CPU cores: {multiprocessing.cpu_count()})"
            )
        else:
            self.logger.info(
                f"⚙️  Sequential processing (use --batch-size N for parallel processing)"
            )

        # Validate directories and configuration
        self._validate_directories()
        self._validate_configuration()

        # Timing (thread-safe for parallel markers)
        self.save_time = save_time
        self._timing_records: list = []
        self._timing_path: Optional[Path] = None  # set at run_pipeline() start
        self._timing_lock = threading.Lock()

        # Track progress
        self.completed_subjects = []
        self.failed_subjects = []

    def _detect_data_source(self) -> str:
        """Auto-detect data source based on directory structure and file patterns."""
        # If a custom file_pattern is provided, check recursively (incl. eeg/ subdir)
        if self.file_pattern:
            matched = list(self.main_path.glob(f"**/{self.file_pattern}"))
            if matched:
                return "bids"

        # Check for CBraMod pattern
        cbramod_files = list(
            self.main_path.glob("**/sub-*_ses-*_vqnsp_reconstructed_epo.fif")
        )
        if cbramod_files:
            return "CBraMod"

        # Check for standard structure with orig/recon subdirs
        orig_dirs = list(self.main_path.glob("**/orig"))
        recon_dirs = list(self.main_path.glob("**/recon"))
        if orig_dirs or recon_dirs:
            return "standard"

        # Check for suffix-based structure
        original_files = list(self.main_path.glob("**/*_original.fif"))
        recon_files = list(self.main_path.glob("**/*_recon.fif"))
        if original_files or recon_files:
            return "suffix"

        # Check for BIDS-style eeg/ subdirectory with _reconstructed suffix
        bids_recon_files = list(self.main_path.glob("**/eeg/*_reconstructed.fif"))
        if bids_recon_files:
            return "bids"

        # Check for any .fif inside eeg/ subdirectory (BIDS-style)
        bids_fif_files = list(self.main_path.glob("**/eeg/*.fif"))
        if bids_fif_files:
            return "bids"

        # Default to standard
        return "standard"

    def _find_original_for_subject(
        self, subject_id: str, session: str
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Search through original_data_paths to find original FIF file for a subject/session.

        Tries each path in order, looking for:
        1. BIDS-like: sub-{id}/ses-{num}/eeg/sub-{id}_ses-{num}_task-*_epo.fif
        2. Direct with _original suffix: sub-{id}/ses-{num}/*_original.fif
        3. Any .fif in session dir: sub-{id}/ses-{num}/*.fif

        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier

        Returns
        -------
        Tuple[Optional[Path], Optional[Path]]
            (fif_file, fif_folder) or (None, None) if not found
        """
        for orig_path in self.original_data_paths:
            session_dir = orig_path / f"sub-{subject_id}" / f"ses-{session}"
            if not session_dir.exists():
                continue

            # Check BIDS-like: ses-{num}/eeg/sub-*_task-*_epo.fif
            eeg_dir = session_dir / "eeg"
            if eeg_dir.exists():
                fif_files = list(
                    eeg_dir.glob(f"sub-{subject_id}_ses-{session}_task-*_epo.fif")
                )
                if not fif_files:
                    fif_files = list(eeg_dir.glob("*.fif"))
                if fif_files:
                    return fif_files[0], eeg_dir

            # Check _original suffix
            fif_files = list(session_dir.glob("*_original.fif"))
            if fif_files:
                return fif_files[0], session_dir

            # Check any .fif
            fif_files = list(session_dir.glob("*.fif"))
            if fif_files:
                return fif_files[0], session_dir

        return None, None

    def _get_output_base_dir(self) -> Path:
        """Get the base output directory based on results_subdir setting."""
        if self.results_subdir:
            return self.results_dir / self.results_subdir
        else:
            return self.results_dir / "new_results"

    def _setup_logging(self):
        """Setup logging configuration."""
        output_base = self._get_output_base_dir()
        log_dir = output_base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure logging
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
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

        # Check for required source files. Paper-relevant code lives under
        # src/{model,interp,paper_plots}/; the legacy phases (GENERAL_METRICS,
        # DECODER, the per-subject HTML report) live under src/misc/ and are
        # reached through dedicated --skip / --*-only flags. DK marker
        # extraction (paper §3.2) is performed with the upstream `nice`
        # library (https://github.com/nice-tools/nice) -- the historical
        # in-tree wrappers under src/markers/ are not redistributed and are
        # only checked when the MARKERS phase is explicitly requested.
        required_files = [
            "misc/decoder/decoder.py",
            "misc/decoder/analysis/analysis.py",
            "misc/decoder/analysis/viz.py",
            "misc/markers_report/compute_data.py",
            "misc/markers_report/generate_plots.py",
            "model/support_vector_machine.py",
            "model/fm_embedding_classifier.py",
        ]

        missing_files = []
        for file_path in required_files:
            if not (self.src_dir / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            raise FileNotFoundError(f"Missing required source files: {missing_files}")

    def _validate_configuration(self):
        """Validate pipeline configuration parameters."""
        if self.mode not in ["patient", "control"]:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be 'patient' or 'control'"
            )

        if self.task not in ["lg", "rs"]:
            raise ValueError(f"Invalid task '{self.task}'. Must be 'lg' or 'rs'")

        # Check YAML templates exist
        yaml_template = self._get_yaml_template()
        if not yaml_template.exists():
            raise FileNotFoundError(f"YAML template not found: {yaml_template}")

        self.logger.info(f"Configuration: mode={self.mode}, task={self.task}")
        self.logger.info(f"YAML template: {yaml_template}")

    def _get_yaml_template(self) -> Path:
        """Get appropriate YAML template based on task."""
        if self.task == "lg":
            return (
                self.src_dir
                / "markers"
                / "input"
                / "icm_complete_individual_markers_local_global.yaml"
            )
        else:  # rs
            return (
                self.src_dir
                / "markers"
                / "input"
                / "icm_complete_individual_markers_resting_state.yaml"
            )

    def discover_subjects(self, force_recompute: bool = False) -> List[Tuple[str, str]]:
        """
        Discover all subjects with available sessions from main_path.
        Supports multiple directory structures:
        1. main_path/sub-{num}/ses-{num}/orig/*.fif and .../recon/*.fif
        2. main_path/sub-{num}/ses-{num}/*_original.fif and *_recon.fif
        3. CBraMod: main_path/sub-{id}/ses-{num}/sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif
        4. Single-file mode: any .fif file in session directory

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

                # Check multiple possible structures:
                # Structure 1: orig/recon subdirectories
                orig_dir = session_dir / "orig"
                recon_dir = session_dir / "recon"

                has_orig_dir = orig_dir.exists() and list(orig_dir.glob("*.fif"))
                has_recon_dir = recon_dir.exists() and list(recon_dir.glob("*.fif"))

                # Structure 2: files with _original/_recon suffixes in session dir
                has_orig_files = list(session_dir.glob("*_original.fif"))
                has_recon_files = list(session_dir.glob("*_recon.fif"))

                # Structure 3: CBraMod pattern (vqnsp_reconstructed)
                has_cbramod_files = list(
                    session_dir.glob("*_vqnsp_reconstructed_epo.fif")
                )

                # Structure 4: Custom file pattern (also search in eeg/ subdir)
                has_custom_pattern = []
                if self.file_pattern:
                    has_custom_pattern = list(session_dir.glob(self.file_pattern))
                    # Also search inside eeg/ subdirectory (BIDS-style)
                    eeg_subdir = session_dir / "eeg"
                    if eeg_subdir.exists():
                        has_custom_pattern.extend(
                            list(eeg_subdir.glob(self.file_pattern))
                        )

                # Structure 5: BIDS-style eeg/ subdirectory
                eeg_subdir = session_dir / "eeg"
                has_eeg_fif = (
                    list(eeg_subdir.glob("*.fif")) if eeg_subdir.exists() else []
                )

                # Structure 6: Any .fif file in session directory (fallback)
                has_any_fif = list(session_dir.glob("*.fif"))

                if (
                    has_orig_dir
                    or has_recon_dir
                    or has_orig_files
                    or has_recon_files
                    or has_cbramod_files
                    or has_custom_pattern
                    or has_eeg_fif
                    or has_any_fif
                ):
                    subjects.append((subject_id, session))
                    if has_cbramod_files:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (CBraMod reconstructed)"
                        )
                    elif has_orig_dir or has_recon_dir:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (orig/:{has_orig_dir}, recon/:{has_recon_dir})"
                        )
                    elif has_orig_files or has_recon_files:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (*_original:{bool(has_orig_files)}, *_recon:{bool(has_recon_files)})"
                        )
                    elif has_custom_pattern:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (custom pattern: {len(has_custom_pattern)} files)"
                        )
                    elif has_eeg_fif:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (BIDS eeg/: {len(has_eeg_fif)} files)"
                        )
                    else:
                        self.logger.debug(
                            f"Found: sub-{subject_id}/ses-{session} (fif files: {len(has_any_fif)})"
                        )

        subjects.sort()
        self.logger.info(
            f"📋 Discovered {len(subjects)} subject/session pairs (data_source: {self.data_source})"
        )
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
        decoder_dir = (
            self.results_dir
            / "new_results"
            / "DECODER"
            / f"sub-{subject_id}"
            / f"ses-{session}"
        )
        decoder_complete = (decoder_dir / "data" / "decoding_results.pkl").exists()

        # Check markers results (both orig and recon)
        markers_dir = self.results_dir / "new_results" / "MARKERS"
        orig_scalars = (
            markers_dir
            / f"sub-{subject_id}_ses-{session}_orig"
            / f"scalars_{subject_id}_ses-{session}_orig.npz"
        ).exists()
        recon_scalars = (
            markers_dir
            / f"sub-{subject_id}_ses-{session}_recon"
            / f"scalars_{subject_id}_ses-{session}_recon.npz"
        ).exists()

        return decoder_complete and orig_scalars and recon_scalars

    def resolve_subjects(
        self, subject_args: List[str], force_recompute: bool = False
    ) -> List[Tuple[str, str]]:
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

    def _get_fif_files_for_subject(
        self, subject_id: str, session: str
    ) -> Dict[str, Optional[Path]]:
        """
        Get FIF file paths for a subject/session based on data source.

        Returns
        -------
        dict
            Dictionary with 'original' and 'reconstructed' keys, values are Path or None
        """
        session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
        # Also check BIDS-style eeg/ subdirectory
        eeg_subdir = session_dir / "eeg"
        search_dirs = [session_dir]
        if eeg_subdir.exists():
            search_dirs.append(eeg_subdir)
        result = {"original": None, "reconstructed": None}

        # Helper: search session_dir and eeg/ subdir for a glob pattern
        def _find_fif(pattern: str) -> Optional[Path]:
            for sdir in search_dirs:
                matches = list(sdir.glob(pattern))
                if matches:
                    return matches[0]
            return None

        # If a custom file_pattern is set, try it first for reconstructed data
        if self.file_pattern:
            result["reconstructed"] = _find_fif(self.file_pattern)
            if self.original_data_paths:
                orig_fif, _ = self._find_original_for_subject(subject_id, session)
                if orig_fif:
                    result["original"] = orig_fif
            if result["reconstructed"]:
                return result

        if self.data_source == "CBraMod":
            # CBraMod pattern: sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif
            result["reconstructed"] = _find_fif("*_vqnsp_reconstructed_epo.fif")
            # Check for original data in separate paths if provided
            if self.original_data_paths:
                orig_fif, _ = self._find_original_for_subject(subject_id, session)
                if orig_fif:
                    result["original"] = orig_fif
        elif self.data_source == "standard":
            # Standard structure: orig/ and recon/ subdirs
            orig_dir = session_dir / "orig"
            recon_dir = session_dir / "recon"
            if orig_dir.exists():
                orig_files = list(orig_dir.glob("*.fif"))
                if orig_files:
                    result["original"] = orig_files[0]
            if recon_dir.exists():
                recon_files = list(recon_dir.glob("*.fif"))
                if recon_files:
                    result["reconstructed"] = recon_files[0]
            # Fallback: check eeg/ subdir for any .fif
            if (
                not result["original"]
                and not result["reconstructed"]
                and eeg_subdir.exists()
            ):
                any_fif = list(eeg_subdir.glob("*.fif"))
                if any_fif:
                    result["reconstructed"] = any_fif[0]
        elif self.data_source == "bids":
            # BIDS-like: files in eeg/ subdirectory
            result["reconstructed"] = (
                _find_fif("*_epo_reconstructed.fif")
                or _find_fif("*_reconstructed_epo.fif")
                or _find_fif("*_epo_recon.fif")
                or _find_fif("*_reconstructed.fif")
            )
            # Check for original data in separate paths if provided
            if self.original_data_paths:
                orig_fif, _ = self._find_original_for_subject(subject_id, session)
                if orig_fif:
                    result["original"] = orig_fif
            # Fallback: look for original in eeg/ subdir
            if not result["original"]:
                result["original"] = _find_fif("*_epo.fif")
        else:  # suffix or auto-detected
            # Suffix-based: *_original.fif and *_recon.fif (search session dir + eeg/ subdir)
            for sdir in search_dirs:
                if not result["original"]:
                    orig_files = list(sdir.glob("*_original.fif"))
                    if orig_files:
                        result["original"] = orig_files[0]
                if not result["reconstructed"]:
                    recon_files = list(sdir.glob("*_recon.fif"))
                    if not recon_files:
                        recon_files = list(sdir.glob("*_reconstructed.fif"))
                    if recon_files:
                        result["reconstructed"] = recon_files[0]
            # Check for original data in separate paths
            if not result["original"] and self.original_data_paths:
                orig_fif, _ = self._find_original_for_subject(subject_id, session)
                if orig_fif:
                    result["original"] = orig_fif
            # Fallback to any .fif file in session dir or eeg/ subdir
            if not result["original"] and not result["reconstructed"]:
                for sdir in search_dirs:
                    any_fif = list(sdir.glob("*.fif"))
                    if any_fif:
                        result["reconstructed"] = any_fif[0]
                        break

        return result

    def run_general_metrics_phase(self) -> bool:
        """
        Run general metrics phase: compute_metrics.py → Correlation/MAPE analysis
        Saves results to output_base/GENERAL_METRICS/

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE A: GENERAL METRICS")
            self.logger.info("=" * 60)

            output_base = self._get_output_base_dir()
            metrics_output_dir = output_base / "GENERAL_METRICS"
            metrics_output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare base directories for compute_metrics.py
            base_dirs = [str(self.main_path)]

            self.logger.info(f"Computing metrics for directories: {base_dirs}")

            cmd = (
                [
                    sys.executable,
                    str(self.src_dir / "misc" / "general_metrics" / "compute_metrics.py"),
                    "--base_dirs",
                ]
                + base_dirs
                + ["--output_dir", str(metrics_output_dir)]
            )

            # Add original data paths if provided (for comparison with reconstructed data)
            if self.original_data_paths:
                existing_paths = [
                    str(p) for p in self.original_data_paths if p.exists()
                ]
                if existing_paths:
                    cmd.extend(["--original_data_dir"] + existing_paths)
                    self.logger.info(
                        f"Using original data directories: {existing_paths}"
                    )
                else:
                    self.logger.warning(
                        "All original_data_paths do not exist, skipping original data comparison"
                    )

            if not self._run_command(cmd):
                self.logger.error("compute_metrics.py failed")
                return False

            self.logger.info("✓ GENERAL METRICS phase completed")
            return True

        except Exception as e:
            self.logger.error(f"✗ GENERAL METRICS phase failed: {e}")
            return False

    def run_fm_embedding_phase(self) -> bool:
        """
        Run FM embedding classification phase (Phase B).

        Invokes ``fm_embedding_classifier.py`` with the configured options.
        The sub-mode is determined by which flags are set on the Pipeline
        instance (defaults to binary VS vs MCS):

        - Standard binary classification (default):
          trains MLP, Random Forest, and Kernel Ridge on pre-computed
          embeddings for VS vs MCS discrimination.

        - Marker regression (``--marker-regression`` / not yet wired into
          pipeline CLI but callable programmatically via the classifier):
          fits a Ridge regressor per scalar marker to predict it from
          embeddings; results in ``regressor_results/``.

        - Full metric prediction (``--full-metric-prediction``):
          runs the same three models for five clinical targets
          (crs, etiology, cs_6m, cs_1y, cs_2y); results under
          ``{output_dir}/{target}/``.

        Requires pre-computed ``*_embedding.npy`` or ``*_embedding.npz``
        files under ``sub-{ID}/ses-{NUM}/`` in the embedding data directory.
        Saves results to ``output_base/MLP_EMBEDDING/``.

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE B: MLP EMBEDDING CLASSIFIER")
            self.logger.info("=" * 60)

            output_base = self._get_output_base_dir()
            mlp_output_dir = output_base / "MLP_EMBEDDING"
            mlp_output_dir.mkdir(parents=True, exist_ok=True)

            # Determine embedding data directory
            if self.embedding_data_dir:
                emb_dir = self.embedding_data_dir
            else:
                # Auto-detect: check for final_layer_embedding sibling of main_path
                candidate = self.main_path.parent / "final_layer_embeddings"
                if candidate.exists():
                    emb_dir = candidate
                    self.logger.info(f"Auto-detected embedding directory: {emb_dir}")
                else:
                    emb_dir = self.main_path

            # Validate that embedding files actually exist (support both singular and plural suffix, .npy and .npz)
            embedding_files = (
                list(emb_dir.glob("**/sub-*/**/*_embedding.npy"))
                + list(emb_dir.glob("**/sub-*/**/*_embeddings.npy"))
                + list(emb_dir.glob("**/sub-*/**/*_embedding.npz"))
                + list(emb_dir.glob("**/sub-*/**/*_embeddings.npz"))
            )
            # Exclude metadata files
            embedding_files = [
                f for f in embedding_files if not f.name.endswith("_metadata.npy")
            ]
            if not embedding_files:
                self.logger.warning(
                    f"No embedding files (*_embedding.npy/npz / *_embeddings.npy/npz) found in {emb_dir}"
                )
                self.logger.warning(
                    "MLP Embedding phase requires pre-computed .npy or .npz embeddings."
                )
                self.logger.warning(
                    "Skipping MLP EMBEDDING phase. To generate embeddings, run the foundation model's embedding extraction first."
                )
                return True  # Not a failure, just no data available
            self.logger.info(
                f"Found {len(embedding_files)} embedding files in {emb_dir}"
            )

            # Check patient labels
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            if not patient_labels_path.exists():
                self.logger.error(
                    f"Patient labels file not found: {patient_labels_path}"
                )
                return False

            cmd = [
                sys.executable,
                str(self.src_dir / "model" / "fm_embedding_classifier.py"),
                "--data-dir",
                str(emb_dir),
                "--patient-labels",
                str(patient_labels_path),
                "--output-dir",
                str(mlp_output_dir),
                "--n-epochs",
                str(self.mlp_n_epochs),
                "--lr",
                str(self.mlp_lr),
                "--batch-size",
                str(self.mlp_batch_size),
            ]

            if self.emb_full_cv:
                cmd.append("--full-cv")

            if self.emb_use_subject_intersection and self.emb_all_embedding_dirs:
                cmd.append("--use-subject-intersection")
                cmd.append("--embedding-dirs")
                cmd.extend(self.emb_all_embedding_dirs)

            if self.emb_marker_regression:
                if not self.emb_marker_csv:
                    self.logger.error(
                        "--emb-marker-regression requires --emb-marker-csv"
                    )
                    return False
                cmd.extend([
                    "--marker-regression",
                    "--marker-csv", self.emb_marker_csv,
                    "--marker-reduction", self.emb_marker_reduction,
                ])

            if self.emb_full_metric_prediction:
                if not self.emb_patient_labels_full:
                    self.logger.error(
                        "--emb-full-metric-prediction requires --emb-patient-labels-full"
                    )
                    return False
                cmd.extend([
                    "--full-metric-prediction",
                    "--patient-labels-full", self.emb_patient_labels_full,
                ])

            if not self._run_command(cmd):
                self.logger.error("fm_embedding_classifier.py failed")
                return False

            self.logger.info("✓ MLP EMBEDDING phase completed")
            return True

        except Exception as e:
            self.logger.error(f"✗ MLP EMBEDDING phase failed: {e}")
            return False

    def run_decoder_phase(self) -> bool:
        """
        Run decoder phase for all subjects: decoder.py → analysis.py → viz.py
        Saves results to output_base/DECODER/

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE C: DECODER")
            self.logger.info("=" * 60)

            output_base = self._get_output_base_dir()
            decoder_output_dir = output_base / "DECODER"
            decoder_output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Run decoder.py
            self.logger.info("Step C1: Running decoder.py...")
            cmd = [
                sys.executable,
                str(self.src_dir / "misc" / "decoder" / "decoder.py"),
                "--main_path",
                str(self.main_path),
                "--output_dir",
                str(decoder_output_dir),
                "--cv",
                str(self.decoder_cv),
                "--n_jobs",
                str(self.decoder_n_jobs),
                "--mode",
                self.mode,
            ]

            # Add original data paths if specified (for CBraMod/LaBram)
            # Pass all paths after a single --original-data-path flag so nargs='+' captures them all
            if self.original_data_paths:
                cmd.append("--original-data-path")
                for orig_path in self.original_data_paths:
                    cmd.append(str(orig_path))

            # Add patient labels if in patient mode
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            if patient_labels_path.exists():
                cmd.extend(["--patient-labels", str(patient_labels_path)])

            # For resting-state, don't filter trial types (no stratification)
            if self.task == "rs":
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
            self.logger.info("Step C2: Running analysis.py...")
            cmd = [
                sys.executable,
                str(self.src_dir / "misc" / "decoder" / "analysis" / "analysis.py"),
                "--results-dir",
                str(latest_decoder_dir),
            ]

            if not self._run_command(cmd):
                self.logger.error("analysis.py failed")
                return False

            # Step 3: Run viz.py
            self.logger.info("Step C3: Running viz.py...")
            # Map mode: patient -> DOC, control -> control
            viz_mode = "DOC" if self.mode == "patient" else "control"
            cmd = [
                sys.executable,
                str(self.src_dir / "misc" / "decoder" / "analysis" / "viz.py"),
                "--results-dir",
                str(latest_decoder_dir),
                "--mode",
                viz_mode,
            ]

            if not self._run_command(cmd):
                self.logger.error("viz.py failed")
                return False

            self.logger.info("✓ DECODER phase completed")
            return True

        except Exception as e:
            self.logger.error(f"✗ DECODER phase failed: {e}")
            return False

    def run_markers_phase_for_subject(
        self, subject_id: str, session: str, file_type: str
    ) -> bool:
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
            # Determine the correct data path based on file_type and data_source
            if file_type in ["orig", "original"] and self.original_data_paths:
                # Use separate original data paths (e.g., for CBraMod)
                orig_fif, orig_folder = self._find_original_for_subject(
                    subject_id, session
                )
                if orig_fif:
                    fif_files = [orig_fif]
                    fif_folder = orig_folder
                else:
                    fif_files = []
                    fif_folder = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
            else:
                session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
                eeg_subdir = session_dir / "eeg"
                fif_files = []
                fif_folder = session_dir

                # For reconstructed data: check for _vqnsp_reconstructed_epo.fif pattern (CBraMod/LaBram style)
                if file_type in ["recon", "reconstructed"]:
                    recon_pattern = (
                        f"sub-{subject_id}_ses-{session}_vqnsp_reconstructed_epo.fif"
                    )
                    fif_files = list(session_dir.glob(recon_pattern))
                    if not fif_files and eeg_subdir.exists():
                        fif_files = list(eeg_subdir.glob(recon_pattern))
                    if not fif_files:
                        fif_files = list(session_dir.glob("*_reconstructed_epo.fif"))
                    if not fif_files and eeg_subdir.exists():
                        fif_files = list(eeg_subdir.glob("*_reconstructed_epo.fif"))
                    # Also try *_epo_reconstructed.fif (NeuroLM style)
                    if not fif_files:
                        fif_files = list(session_dir.glob("*_epo_reconstructed.fif"))
                    if not fif_files and eeg_subdir.exists():
                        fif_files = list(eeg_subdir.glob("*_epo_reconstructed.fif"))
                    if fif_files and eeg_subdir.exists():
                        fif_folder = eeg_subdir

                # Try Structure 1: orig/recon subdirectories
                if not fif_files:
                    subdir = session_dir / file_type
                    if subdir.exists():
                        fif_files = list(subdir.glob("*.fif"))
                        if fif_files:
                            fif_folder = subdir

                # Try Structure 2: files with _original/_recon suffixes
                if not fif_files:
                    suffix = (
                        "_original.fif"
                        if file_type in ["orig", "original"]
                        else "_recon.fif"
                    )
                    fif_files = list(session_dir.glob(f"*{suffix}"))
                    if not fif_files and eeg_subdir.exists():
                        fif_files = list(eeg_subdir.glob(f"*{suffix}"))
                    if fif_files:
                        fif_folder = (
                            session_dir
                            if fif_files[0].parent == session_dir
                            else eeg_subdir
                        )

            if not fif_files:
                self.logger.warning(
                    f"No .fif files found for {subject_id}/ses-{session}/{file_type}"
                )
                return False

            fif_file = fif_files[0]  # Use first .fif file found

            # Output directory: {output_base}/MARKERS/sub-{ID}/ses-{num}/orig/ or recon/
            output_base = self._get_output_base_dir()
            subject_dir = output_base / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            subject_markers_dir = session_dir_out / file_type  # orig or recon
            subject_markers_dir.mkdir(parents=True, exist_ok=True)

            # Check if results already exist (check all required outputs)
            scalars_file = (
                subject_markers_dir
                / f"scalars_{subject_id}_ses-{session}_{file_type}.npz"
            )
            topos_file = (
                subject_markers_dir
                / f"topos_{subject_id}_ses-{session}_{file_type}.npz"
            )
            h5_file = subject_markers_dir / "icm_complete_features.h5"

            # Check for key output files to determine if processing is complete
            if scalars_file.exists() and topos_file.exists():
                self.logger.info(
                    f"⏭️  Skipping {subject_id}/ses-{session}/{file_type} - results already exist"
                )
                return True

            self.logger.info(f"Processing {subject_id}/ses-{session}/{file_type}")

            # Step 1: compute_markers_with_junifer.py (ALWAYS keep H5 until after report generation)
            self.logger.info("  Step D1: compute_markers_with_junifer.py...")
            _t0 = datetime.now()

            cmd = [
                sys.executable,
                str(self.src_dir / "markers" / "compute_markers_with_junifer.py"),
                "--fif_folder",
                str(fif_folder),
                "--task",
                self.task,
                "--output-dir",
                str(subject_markers_dir),  # Use direct output directory
                "--skip-clustering",  # Always skip clustering (default for speed)
                "--keep-h5",  # ALWAYS keep H5 file - we'll delete it later after report generation
            ]

            if not self._run_command(cmd):
                self.logger.warning(
                    "compute_markers_with_junifer.py reported errors (internal compute_data/scalars may have failed)"
                )
                # Don't abort — the H5 file may still have been created by junifer

            _elapsed = (datetime.now() - _t0).total_seconds()
            self.logger.info(f"  Step D1 (junifer) took {_elapsed:.1f}s")
            self._record_timing(
                "MARKERS", "D1_junifer", _elapsed, subject_id, session, file_type
            )
            self._flush_timing_csv()

            # Check if H5 file was created (this is the essential output of Step D1)
            if not h5_file.exists():
                self.logger.error(f"H5 file not found: {h5_file} — junifer failed")
                return False

            # Step 2: compute_data.py (report generation — non-critical for scalars/topos)
            self.logger.info("  Step D2: compute_data.py...")
            _t0 = datetime.now()
            cmd = [
                sys.executable,
                str(self.src_dir / "misc" / "markers_report" / "compute_data.py"),
                "--subject_id",
                subject_id,
                "--h5_file",
                str(h5_file),
                "--fif_file",
                str(fif_file),
                "--output_dir",
                str(subject_markers_dir),
                "--task",
                self.task,
            ]

            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")

            if not self._run_command(cmd):
                self.logger.warning(
                    "compute_data.py failed (report generation) — continuing with scalars/topos"
                )

            _elapsed = (datetime.now() - _t0).total_seconds()
            self.logger.info(f"  Step D2 (compute_data) took {_elapsed:.1f}s")
            self._record_timing(
                "MARKERS", "D2_compute_data", _elapsed, subject_id, session, file_type
            )
            self._flush_timing_csv()

            # Steps D3 + D4: compute_scalars and compute_topographies (run concurrently)
            # Both read the H5 file independently, so they can safely overlap.
            self.logger.info(
                "  Steps D3+D4: compute_scalars.py & compute_topographies.py (parallel)..."
            )

            def _run_d3_scalars():
                _t0 = datetime.now()
                cmd = [
                    sys.executable,
                    str(self.src_dir / "markers" / "compute_scalars.py"),
                    "--h5_file",
                    str(h5_file),
                    "--output_file",
                    str(scalars_file),
                ]
                ok = self._run_command(cmd)
                _elapsed = (datetime.now() - _t0).total_seconds()
                self.logger.info(f"  Step D3 (compute_scalars) took {_elapsed:.1f}s")
                self._record_timing(
                    "MARKERS",
                    "D3_compute_scalars",
                    _elapsed,
                    subject_id,
                    session,
                    file_type,
                )
                self._flush_timing_csv()
                return ok

            def _run_d4_topos():
                _t0 = datetime.now()
                cmd = [
                    sys.executable,
                    str(self.src_dir / "markers" / "compute_topographies.py"),
                    "--h5_file",
                    str(h5_file),
                    "--output_file",
                    str(topos_file),
                ]
                ok = self._run_command(cmd)
                _elapsed = (datetime.now() - _t0).total_seconds()
                self.logger.info(
                    f"  Step D4 (compute_topographies) took {_elapsed:.1f}s"
                )
                self._record_timing(
                    "MARKERS",
                    "D4_compute_topographies",
                    _elapsed,
                    subject_id,
                    session,
                    file_type,
                )
                self._flush_timing_csv()
                return ok

            with ThreadPoolExecutor(max_workers=2) as step_executor:
                d3_future = step_executor.submit(_run_d3_scalars)
                d4_future = step_executor.submit(_run_d4_topos)
                d3_ok = d3_future.result()
                d4_ok = d4_future.result()

            if not d3_ok:
                self.logger.error("compute_scalars.py failed")
                return False
            if not d4_ok:
                self.logger.error("compute_topographies.py failed")
                return False

            # Verify all outputs were created
            if not scalars_file.exists():
                self.logger.error(f"Scalars file not created: {scalars_file}")
                return False
            if not topos_file.exists():
                self.logger.error(f"Topographies file not created: {topos_file}")
                return False

            self.logger.info(
                f"✓ Markers completed for {subject_id}/ses-{session}/{file_type}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"✗ Markers failed for {subject_id}/ses-{session}/{file_type}: {e}"
            )
            return False

    def _process_single_subject_markers(
        self,
        subject_id: str,
        session: str,
        markers_dir: Path,
        idx: int,
        total: int,
    ) -> Tuple[str, str, bool, list, bool]:
        """
        Process all file types for a single subject. Thread-safe.

        Returns
        -------
        Tuple[str, str, bool, list, bool]
            (subject_id, session, success, failed_list, skipped)
            skipped is True when another job already holds the processing lock.
        """
        self.logger.info(f"\n[{idx}/{total}] Processing {subject_id}/ses-{session}...")

        # Atomic lock file to prevent duplicate processing across HTCondor jobs
        lock_file = (
            markers_dir / f"sub-{subject_id}" / f"ses-{session}" / "processing.lock"
        )
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            self.logger.info(
                f"Skipping {subject_id}/ses-{session} - already being processed by another job"
            )
            return (subject_id, session, True, [], True)

        subject_success = True
        failed = []

        # Determine which file types to process based on data availability
        file_types_to_process = []

        # Check for original data (search through all original_data_paths)
        if self.original_data_paths:
            orig_fif, _ = self._find_original_for_subject(subject_id, session)
            if orig_fif:
                file_types_to_process.append("original")
        else:
            # Check standard original data locations
            session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
            if (session_dir / "orig").exists() or list(
                session_dir.glob("*_original.fif")
            ):
                file_types_to_process.append("original")

        # Check for reconstructed data
        session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"
        sessions_dir_eeg = (
            self.main_path / f"sub-{subject_id}" / f"ses-{session}" / "eeg"
        )
        # Check for _vqnsp_reconstructed_epo.fif pattern (CBraMod/LaBram style)
        if list(session_dir.glob("*_reconstructed_epo.fif")):
            file_types_to_process.append("recon")
        elif list(sessions_dir_eeg.glob("*_epo_reconstructed.fif")):
            file_types_to_process.append("recon")
        # Also check standard recon patterns
        elif (session_dir / "recon").exists() or list(session_dir.glob("*_recon.fif")):
            file_types_to_process.append("recon")

        if not file_types_to_process:
            self.logger.warning(
                f"No data files found for {subject_id}/ses-{session}, skipping"
            )
            return (subject_id, session, True, [], False)

        self.logger.info(f"  Processing file types: {file_types_to_process}")

        # Process each available file type
        for file_type in file_types_to_process:
            self.logger.info(f"  Processing {file_type} data...")
            if not self.run_markers_phase_for_subject(subject_id, session, file_type):
                self.logger.error(
                    f"Failed to process {file_type} data for {subject_id}/ses-{session}"
                )
                subject_success = False
                failed.append((subject_id, session, file_type))

        # Flush timing CSV after every subject so progress is preserved on crash
        self._flush_timing_csv()

        # Generate report if processing succeeded (only if both original and recon exist)
        if subject_success and len(file_types_to_process) == 2:
            self.logger.info("  Generating report...")
            if not self._generate_report_for_subject(subject_id, session):
                self.logger.warning(
                    f"Report generation failed for {subject_id}/ses-{session} (continuing anyway)"
                )

        if subject_success:
            # Create finished.txt marker after all processing is complete
            self._create_finish_marker(subject_id, session)
            # Lock file no longer needed — finished.txt is the permanent marker
            lock_file.unlink(missing_ok=True)
            self.logger.info(f"✓ Completed {subject_id}/ses-{session}")
        else:
            # Clean up lock so the subject can be retried
            lock_file.unlink(missing_ok=True)
            self.logger.error(
                f"✗ Skipping report for {subject_id}/ses-{session} due to processing failures"
            )

        return (subject_id, session, subject_success, failed, False)

    def run_markers_phase_for_all_subjects(
        self, subjects: List[Tuple[str, str]]
    ) -> bool:
        """
        Run markers phase for all subjects using batch processing.

        BATCH WORKFLOW:
        1. Run compute_markers_with_junifer.py ONCE for ALL subject/sessions (generates all H5 files)
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
            self.logger.info("PHASE D: MARKERS (Individual Processing)")
            self.logger.info("=" * 60)
            self.logger.info(
                "Workflow: junifer markers → compute_scalars → compute_topographies → reports"
            )
            self.logger.info("=" * 60)

            output_base = self._get_output_base_dir()
            markers_dir = output_base / "MARKERS"
            markers_dir.mkdir(parents=True, exist_ok=True)

            # Filter out subjects that already have finished.txt
            subjects_to_process = []
            already_complete = []

            for subject_id, session in subjects:
                finish_marker = (
                    markers_dir
                    / f"sub-{subject_id}"
                    / f"ses-{session}"
                    / "finished.txt"
                )
                if finish_marker.exists():
                    self.logger.info(
                        f"⏭️  Skipping {subject_id}/ses-{session} - finished.txt marker found"
                    )
                    already_complete.append((subject_id, session))
                else:
                    subjects_to_process.append((subject_id, session))

            self.logger.info(
                f"Found {len(already_complete)} subject/session pairs already complete"
            )
            self.logger.info(
                f"Will process {len(subjects_to_process)} subject/session pairs"
            )

            # If all subjects are already complete, skip processing
            if len(subjects_to_process) == 0:
                self.logger.info(
                    "✓ All subjects already have finished.txt markers - skipping processing"
                )
                return True

            # Process each subject/session individually
            # This allows proper handling of CBraMod where original and reconstructed data are in different locations
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("PROCESSING MARKERS FOR EACH SUBJECT")
            self.logger.info("=" * 60)

            success_count = len(already_complete)  # Count already complete subjects
            failed_subjects = []
            skipped_count = 0
            total = len(subjects_to_process)
            completed_count = 0
            completed_lock = threading.Lock()

            if self.batch_size > 1:
                # Parallel processing with ThreadPoolExecutor
                self.logger.info(
                    f"Processing {total} subjects with {self.batch_size} parallel workers"
                )
                with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    futures = {}
                    for idx, (subject_id, session) in enumerate(subjects_to_process, 1):
                        future = executor.submit(
                            self._process_single_subject_markers,
                            subject_id,
                            session,
                            markers_dir,
                            idx,
                            total,
                        )
                        futures[future] = (subject_id, session)

                    for future in as_completed(futures):
                        subj_id, sess, success, failed, skipped = future.result()
                        with completed_lock:
                            completed_count += 1
                            if skipped:
                                skipped_count += 1
                                status = "SKIPPED (other job)"
                            elif success:
                                success_count += 1
                                status = "OK"
                            else:
                                failed_subjects.extend(failed)
                                status = "FAILED"
                            self.logger.info(
                                f">>> Progress: {completed_count}/{total} done | "
                                f"{subj_id}/ses-{sess}: {status} | "
                                f"succeeded: {success_count}, failed: {len(failed_subjects)}, "
                                f"skipped: {skipped_count}"
                            )
                        self._flush_timing_csv()
            else:
                # Sequential processing
                for idx, (subject_id, session) in enumerate(subjects_to_process, 1):
                    subj_id, sess, success, failed, skipped = (
                        self._process_single_subject_markers(
                            subject_id, session, markers_dir, idx, total
                        )
                    )
                    completed_count += 1
                    if skipped:
                        skipped_count += 1
                        status = "SKIPPED (other job)"
                    elif success:
                        success_count += 1
                        status = "OK"
                    else:
                        failed_subjects.extend(failed)
                        status = "FAILED"
                    self.logger.info(
                        f">>> Progress: {completed_count}/{total} done | "
                        f"{subj_id}/ses-{sess}: {status} | "
                        f"succeeded: {success_count}, failed: {len(failed_subjects)}, "
                        f"skipped: {skipped_count}"
                    )
                    self._flush_timing_csv()

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
            self.logger.info(
                f"Successfully processed: {success_count}/{len(subjects)} subjects"
            )
            if skipped_count > 0:
                self.logger.info(f"Skipped (claimed by other jobs): {skipped_count}")

            if failed_subjects:
                self.logger.warning(
                    f"Failed processing for {len(failed_subjects)} subject/session/file_type combinations:"
                )
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
            output_base = self._get_output_base_dir()
            subject_dir = output_base / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            original_dir = session_dir_out / "original"
            recon_dir = session_dir_out / "recon"
            reports_dir = session_dir_out / "reports"

            # Check original outputs
            original_scalars = (
                original_dir / f"scalars_{subject_id}_ses-{session}_original.npz"
            )
            original_topos = (
                original_dir / f"topos_{subject_id}_ses-{session}_original.npz"
            )

            # Check recon outputs
            recon_scalars = recon_dir / f"scalars_{subject_id}_ses-{session}_recon.npz"
            recon_topos = recon_dir / f"topos_{subject_id}_ses-{session}_recon.npz"

            # Check report output
            report_file = (
                reports_dir / f"sub-{subject_id}_ses-{session}_report_comparison.html"
            )

            # Check for at least some pkl files (from compute_data.py)
            original_pkl_files = (
                list(original_dir.glob("*.pkl")) if original_dir.exists() else []
            )
            recon_pkl_files = (
                list(recon_dir.glob("*.pkl")) if recon_dir.exists() else []
            )

            # All required files must exist
            required_files = [
                original_scalars,
                original_topos,
                recon_scalars,
                recon_topos,
                report_file,
            ]

            all_exist = all(f.exists() for f in required_files)
            has_pkl_files = len(original_pkl_files) > 0 and len(recon_pkl_files) > 0

            if all_exist and has_pkl_files:
                self.logger.debug(
                    f"All outputs verified for {subject_id}/ses-{session}"
                )
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
                    self.logger.debug(
                        f"Missing outputs for {subject_id}/ses-{session}: {', '.join(missing)}"
                    )

                return False

        except Exception as e:
            self.logger.debug(
                f"Error checking completion status for {subject_id}/ses-{session}: {e}"
            )
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
            output_base = self._get_output_base_dir()
            subject_dir = output_base / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"
            original_dir = session_dir_out / "original"
            recon_dir = session_dir_out / "recon"

            # Find FIF files - support multiple structures including CBraMod
            session_dir = self.main_path / f"sub-{subject_id}" / f"ses-{session}"

            orig_fif = None
            recon_fif = None

            # For CBraMod: original data is in separate location(s)
            if self.original_data_paths:
                orig_fif, _ = self._find_original_for_subject(subject_id, session)

            # Check for _vqnsp_reconstructed_epo.fif pattern (CBraMod/LaBram style)
            recon_files = list(
                session_dir.glob(
                    f"sub-{subject_id}_ses-{session}_vqnsp_reconstructed_epo.fif"
                )
            )
            if not recon_files:
                recon_files = list(session_dir.glob("*_reconstructed_epo.fif"))
            recon_fif = recon_files[0] if recon_files else None

            # Try Structure 1: orig/recon subdirectories
            if not orig_fif:
                orig_fif_folder = session_dir / "orig"
                if orig_fif_folder.exists():
                    orig_files = list(orig_fif_folder.glob("*.fif"))
                    orig_fif = orig_files[0] if orig_files else None

            if not recon_fif:
                recon_fif_folder = session_dir / "recon"
                if recon_fif_folder.exists():
                    recon_files = list(recon_fif_folder.glob("*.fif"))
                    recon_fif = recon_files[0] if recon_files else None

            # Try Structure 2: files with _original/_recon suffixes
            if not orig_fif:
                orig_files = list(session_dir.glob("*_original.fif"))
                orig_fif = orig_files[0] if orig_files else None

            if not recon_fif:
                recon_files = list(session_dir.glob("*_recon.fif"))
                recon_fif = recon_files[0] if recon_files else None

            # Check prerequisites - for CBraMod, original might be optional
            if not recon_fif:
                self.logger.error(
                    f"Missing reconstructed FIF file for {subject_id}/ses-{session}"
                )
                return False

            if not (original_dir.exists() or recon_dir.exists()):
                self.logger.error(
                    f"Missing marker directories for {subject_id}/ses-{session}"
                )
                return False

            # Find H5 files (from individual processing location)
            markers_dir = output_base / "MARKERS"
            orig_h5 = original_dir / "icm_complete_features.h5"
            recon_h5 = recon_dir / "icm_complete_features.h5"

            if not (orig_h5.exists() and recon_h5.exists()):
                self.logger.error(f"Missing H5 files for {subject_id}/ses-{session}")
                self.logger.error(f"  orig H5: {orig_h5} - exists: {orig_h5.exists()}")
                self.logger.error(
                    f"  recon H5: {recon_h5} - exists: {recon_h5.exists()}"
                )
                return False

            # Reports go in MARKERS/sub-{ID}/ses-{num}/reports/
            output_dir = session_dir_out / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(self.src_dir / "misc" / "markers_report" / "generate_plots.py"),
                "--subject_id",
                subject_id,
                "--session",
                session,
                "--h5_file",
                str(orig_h5),  # Can use either, script needs both via data_dir
                "--fif_file",
                str(orig_fif),
                "--data_dir_original",
                str(original_dir),
                "--data_dir_recon",
                str(recon_dir),
                "--output_dir",
                str(output_dir),
            ]

            if self.markers_skip_clustering:
                cmd.append("--skip-clustering")

            # Run and check success
            if self._run_command(cmd):
                # Verify report was created
                report_file = (
                    output_dir
                    / f"sub-{subject_id}_ses-{session}_report_comparison.html"
                )
                if report_file.exists():
                    self.logger.info(f"✓ Report generated: {report_file}")
                    return True
                else:
                    self.logger.warning(
                        f"Report command succeeded but file not found: {report_file}"
                    )
                    return False
            else:
                self.logger.error(
                    f"Report generation command failed for {subject_id}/ses-{session}"
                )
                return False

        except Exception as e:
            self.logger.error(
                f"Exception during report generation for {subject_id}/ses-{session}: {e}"
            )
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
            output_base = self._get_output_base_dir()
            subject_dir = output_base / "MARKERS" / f"sub-{subject_id}"
            session_dir_out = subject_dir / f"ses-{session}"

            h5_files = [
                session_dir_out / "original" / "icm_complete_features.h5",
                session_dir_out / "recon" / "icm_complete_features.h5",
            ]

            for h5_file in h5_files:
                if h5_file.exists():
                    try:
                        h5_file.unlink()
                        self.logger.info(f"  Deleted: {h5_file}")
                    except Exception as e:
                        self.logger.warning(f"  Could not delete {h5_file}: {e}")

        except Exception as e:
            self.logger.warning(
                f"Exception during H5 cleanup for {subject_id}/ses-{session}: {e}"
            )

    def _create_finish_marker(self, subject_id: str, session: str) -> None:
        """
        Create a finished.txt marker file after all processing is complete for a subject/session.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        """
        try:
            output_base = self._get_output_base_dir()
            markers_dir = output_base / "MARKERS"
            # Store finish marker in subject session directory
            finish_marker = (
                markers_dir / f"sub-{subject_id}" / f"ses-{session}" / "finished.txt"
            )

            # Create parent directory if needed
            finish_marker.parent.mkdir(parents=True, exist_ok=True)

            # Write marker file with timestamp
            with open(finish_marker, "w") as f:
                f.write(
                    f"Marker processing completed at {datetime.now().isoformat()}\n"
                )

            self.logger.info(f"  ✓ Created finish marker: {finish_marker}")

        except Exception as e:
            self.logger.warning(
                f"Could not create finish marker for {subject_id}/ses-{session}: {e}"
            )

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

    def run_model_phase(self) -> bool:
        """
        Run model training phase (only for cases 1 and 2, not resting-state).
        Uses support_vector_machine.py for VS vs MCS classification.
        Saves results to output_base/MODEL/

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Skip model phase for resting-state
            if self.task == "rs":
                self.logger.info("Skipping MODEL phase for resting-state data")
                return True

            self.logger.info("=" * 60)
            self.logger.info("PHASE E: MODEL (SVM)")
            self.logger.info("=" * 60)

            output_base = self._get_output_base_dir()
            model_output_dir = output_base / "MODEL"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Check patient labels file
            patient_labels_path = self.metadata_dir / "patient_labels_with_controls.csv"
            if not patient_labels_path.exists():
                self.logger.error(
                    f"Patient labels file not found: {patient_labels_path}"
                )
                return False

            markers_dir = output_base / "MARKERS"

            # Run SVM with cross-data classification
            self.logger.info(
                f"Training SVM classifier ({self.model_marker_type} markers)..."
            )
            cmd = [
                sys.executable,
                str(self.src_dir / "model" / "support_vector_machine.py"),
                "--data-dir",
                str(markers_dir),
                "--patient-labels",
                str(patient_labels_path),
                "--marker-type",
                self.model_marker_type,
                "--output-dir",
                str(model_output_dir / self.model_marker_type),
                "--cross-data",  # Enable cross-data classification
            ]

            if self.model_orig_data_dir:
                cmd.extend(["--orig-data-dir", str(self.model_orig_data_dir)])

            if self.model_baseline_csv:
                cmd.extend(["--baseline-csv", str(self.model_baseline_csv)])

            if self.model_use_subject_intersection:
                cmd.append("--use-subject-intersection")

            if self.model_full_cv:
                cmd.append("--full-cv")

            if not self._run_command(cmd):
                self.logger.error("support_vector_machine.py failed")
                return False

            self.logger.info("✓ MODEL phase completed")
            return True

        except Exception as e:
            self.logger.error(f"✗ MODEL phase failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # NeurIPS paper-evaluation phases (representational audit, §3.2)
    #
    # These four phases mirror the 7-step evaluation framework of the paper:
    #   F. Probing            → §3.2 layer-wise R² + per-layer CRS AUC (steps B & C)
    #   G. Combined DK        → §3.2 FM ⊕ DK concatenation (step D)
    #   H. Residualisation    → §3.2 fold-internal residualisation (step E)
    #   I. MKNN alignment     → §3.2 + Appendix MKNN(k) and k-sweep (steps F & G)
    # Step A (Utility nested CV across 6 tasks) reuses the existing
    # FM_EMBEDDING phase. Each helper subprocesses into the canonical entry
    # point under src/{interp,model}/ with sys.executable.
    # ──────────────────────────────────────────────────────────────────────

    def _paper_results_root(self) -> "Path":
        """Where the per-FM result trees live (parent of {FM}/doc_patients/)."""
        from pathlib import Path

        if self.results_subdir:
            return Path(self.results_dir) / Path(self.results_subdir).parts[0]
        return Path(self.results_dir)

    def run_probing_phase(self) -> bool:
        """Phase F: layer-wise linear probing (R² + per-layer CRS AUC).

        Wraps ``src/interp/linear_probing.py``. Produces
        ``LINEAR_PROBING/regression/{layer}/{FM}/summary.json`` and
        ``LINEAR_PROBING/classification/{layer}/{FM}/...``.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE F: LAYER-WISE LINEAR PROBING")
            self.logger.info("=" * 60)
            output_dir = self._paper_results_root() / "LINEAR_PROBING"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(self.src_dir / "interp" / "linear_probing.py"),
                "--output-dir", str(output_dir),
            ]
            if self.embedding_data_dir:
                cmd.extend(["--data-dir", str(self.embedding_data_dir)])
            if self.emb_marker_csv:
                cmd.extend(["--marker-csv", str(self.emb_marker_csv)])
            if self.emb_patient_labels_full:
                cmd.extend(["--patient-labels", str(self.emb_patient_labels_full)])
            if not self._run_command(cmd):
                self.logger.error("linear_probing.py failed")
                return False
            self.logger.info("✓ PROBING phase completed")
            return True
        except Exception as e:
            self.logger.error(f"✗ PROBING phase failed: {e}")
            return False

    def run_fm_plus_dk_phase(self) -> bool:
        """Phase G: FM ⊕ DK concatenation classification.

        Wraps ``src/model/fm_plus_dk_classifier.py``. Produces
        ``EMBEDDING_DK_COMBINED/{target}/...``.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE G: FM + DK COMBINED CLASSIFICATION")
            self.logger.info("=" * 60)
            cmd = [
                sys.executable,
                str(self.src_dir / "model" / "fm_plus_dk_classifier.py"),
                "--results-root", str(self._paper_results_root()),
            ]
            if self.emb_marker_csv:
                cmd.extend(["--marker-csv", str(self.emb_marker_csv)])
            if self.emb_patient_labels_full:
                cmd.extend(["--patient-labels", str(self.emb_patient_labels_full)])
            if not self._run_command(cmd):
                self.logger.error("fm_plus_dk_classifier.py failed")
                return False
            self.logger.info("✓ FM_PLUS_DK phase completed")
            return True
        except Exception as e:
            self.logger.error(f"✗ FM_PLUS_DK phase failed: {e}")
            return False

    def run_residualise_phase(self) -> bool:
        """Phase H: fold-internal linear residualisation (no leakage).

        Wraps ``src/interp/res_no_leakage/fold_internal_residualisation.py``.
        Produces ``RES_NO_LEAKAGE/{target}/...``.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE H: FOLD-INTERNAL RESIDUALISATION")
            self.logger.info("=" * 60)
            output_dir = self._paper_results_root() / "RES_NO_LEAKAGE"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(
                    self.src_dir
                    / "interp"
                    / "res_no_leakage"
                    / "fold_internal_residualisation.py"
                ),
                "--results-root", str(self._paper_results_root()),
                "--output-dir", str(output_dir),
            ]
            if self.emb_marker_csv:
                cmd.extend(["--marker-csv", str(self.emb_marker_csv)])
            if self.emb_patient_labels_full:
                cmd.extend(["--patient-labels", str(self.emb_patient_labels_full)])
            if not self._run_command(cmd):
                self.logger.error("fold_internal_residualisation.py failed")
                return False
            self.logger.info("✓ RESIDUALISE phase completed")
            return True
        except Exception as e:
            self.logger.error(f"✗ RESIDUALISE phase failed: {e}")
            return False

    def run_mknn_phase(self) -> bool:
        """Phase I: MKNN(DK, FM) alignment + permutation null + k-sweep.

        Wraps ``src/model/embedding_comparison.py`` and runs Component 5 only,
        twice: once at the canonical k=20 (paper Table 1, output to
        ``EMBEDDING_COMPARISON/component5_mknn/``) and once across k ∈ [10, 40]
        for the appendix k-sweep ablation
        (``EMBEDDING_COMPARISON/component5_mknn_ksweep/``).
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("PHASE I: MKNN ALIGNMENT")
            self.logger.info("=" * 60)
            output_dir = self._paper_results_root() / "EMBEDDING_COMPARISON"
            output_dir.mkdir(parents=True, exist_ok=True)

            base_cmd = [
                sys.executable,
                str(self.src_dir / "model" / "embedding_comparison.py"),
                "--results-dir", str(self._paper_results_root()),
                "--output-dir", str(output_dir),
                "--skip-ridge-r2",
                "--skip-rsa",
                "--skip-cka",
                "--skip-dimensionality",
                "--skip-pwcca",
            ]
            if self.emb_marker_csv:
                base_cmd.extend(["--marker-csv", str(self.emb_marker_csv)])
            if self.emb_patient_labels_full:
                base_cmd.extend(["--patient-labels", str(self.emb_patient_labels_full)])

            # 1) Canonical k=20 (paper Table 1).
            cmd_k20 = base_cmd + [
                "--mknn-k", "20",
                "--mknn-subdir", "component5_mknn",
            ]
            self.logger.info("MKNN: running canonical k=20")
            if not self._run_command(cmd_k20):
                self.logger.error("embedding_comparison.py (k=20) failed")
                return False

            # 2) k-sweep for the Appendix ablation (k = 10..40 step 1).
            ksweep = [str(k) for k in range(10, 41)]
            cmd_sweep = base_cmd + [
                "--mknn-k", *ksweep,
                "--mknn-subdir", "component5_mknn_ksweep",
            ]
            self.logger.info("MKNN: running k-sweep over k ∈ [10, 40]")
            if not self._run_command(cmd_sweep):
                self.logger.error("embedding_comparison.py (k-sweep) failed")
                return False

            self.logger.info("✓ MKNN phase completed")
            return True
        except Exception as e:
            self.logger.error(f"✗ MKNN phase failed: {e}")
            return False

    def run_pipeline(
        self,
        subjects: List[Tuple[str, str]],
        skip_general_metrics: bool = False,
        skip_fm_embedding: bool = False,
        skip_decoder: bool = False,
        skip_markers: bool = False,
        skip_model: bool = False,
        skip_probing: bool = True,
        skip_fm_plus_dk: bool = True,
        skip_residualise: bool = True,
        skip_mknn: bool = True,
    ) -> dict:
        """
        Run the complete pipeline for the specified subjects.

        Pipeline flow:
        A. GENERAL_METRICS phase: compute_metrics.py → Correlation/MAPE
        B. FM_EMBEDDING phase: fm_embedding_classifier.py → MLP on embeddings (VS vs MCS)
        C. DECODER phase: decoder.py → analysis.py → viz.py
        D. MARKERS phase: For each subject:
           - compute_markers_with_junifer.py → HDF5
           - compute_scalars.py → scalars.npz
           - compute_topographies.py → topos.npz
        E. MODEL phase: support_vector_machine.py (skip for resting-state)

        Parameters
        ----------
        subjects : List[Tuple[str, str]]
            List of (subject_id, session) tuples to process
        skip_general_metrics : bool
            Skip GENERAL_METRICS phase
        skip_fm_embedding : bool
            Skip FM_EMBEDDING phase
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
        output_base = self._get_output_base_dir()

        # Initialise timing CSV path so incremental flushes can start immediately
        if self.save_time:
            log_dir = output_base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._timing_path = (
                log_dir / f"timing_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.logger.info(f"Timing will be saved to: {self._timing_path}")
        else:
            self.logger.info("Timing CSV disabled (use --save-time to enable)")

        self.logger.info("=" * 70)
        self.logger.info("PIPELINE START")
        self.logger.info("=" * 70)
        self.logger.info(
            f"Configuration: mode={self.mode}, task={self.task}, data_source={self.data_source}"
        )
        self.logger.info(f"Output directory: {output_base}")
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Subjects to process: {len(subjects)}")
        self.logger.info("=" * 70)

        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual processing")
            return {"status": "dry_run", "subjects": subjects}

        results = {
            "general_metrics": False if not skip_general_metrics else "skipped",
            "fm_embedding": False if not skip_fm_embedding else "skipped",
            "decoder": False if not skip_decoder else "skipped",
            "markers": False if not skip_markers else "skipped",
            "model": False if not skip_model else "skipped",
            "probing": False if not skip_probing else "skipped",
            "fm_plus_dk": False if not skip_fm_plus_dk else "skipped",
            "residualise": False if not skip_residualise else "skipped",
            "mknn": False if not skip_mknn else "skipped",
        }

        # Phase A: GENERAL_METRICS
        if not skip_general_metrics:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_general_metrics_phase():
                results["general_metrics"] = True
            else:
                self.logger.error(
                    "GENERAL_METRICS phase failed - continuing with MLP EMBEDDING phase"
                )
            self._record_timing(
                "GENERAL_METRICS", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE A: GENERAL_METRICS - SKIPPED")

        # Phase B: FM_EMBEDDING
        if not skip_fm_embedding:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_fm_embedding_phase():
                results["fm_embedding"] = True
            else:
                self.logger.error(
                    "FM_EMBEDDING phase failed - continuing with DECODER phase"
                )
            self._record_timing(
                "FM_EMBEDDING", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE B: FM_EMBEDDING - SKIPPED")

        # Phase C: DECODER
        if not skip_decoder:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_decoder_phase():
                results["decoder"] = True
            else:
                self.logger.error(
                    "DECODER phase failed - continuing with MARKERS phase"
                )
            self._record_timing(
                "DECODER", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE C: DECODER - SKIPPED")

        # Phase D: MARKERS
        if not skip_markers:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_markers_phase_for_all_subjects(subjects):
                results["markers"] = True
            else:
                self.logger.error("MARKERS phase failed - continuing with MODEL phase")
            self._record_timing(
                "MARKERS", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE D: MARKERS - SKIPPED")

        # Phase E: MODEL (only for lg task, not rs)
        if not skip_model and self.task != "rs":
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_model_phase():
                results["model"] = True
            else:
                self.logger.error("MODEL phase failed")
            self._record_timing(
                "MODEL", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        elif skip_model:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE E: MODEL - SKIPPED")
        # else: task == 'rs', model phase skipped automatically

        # Phase F: PROBING (paper §3.2 layer-wise linear probing)
        if not skip_probing:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_probing_phase():
                results["probing"] = True
            else:
                self.logger.error("PROBING phase failed - continuing")
            self._record_timing(
                "PROBING", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE F: PROBING - SKIPPED")

        # Phase G: FM_PLUS_DK (paper §3.2 FM ⊕ DK concatenation)
        if not skip_fm_plus_dk:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_fm_plus_dk_phase():
                results["fm_plus_dk"] = True
            else:
                self.logger.error("FM_PLUS_DK phase failed - continuing")
            self._record_timing(
                "FM_PLUS_DK", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE G: FM_PLUS_DK - SKIPPED")

        # Phase H: RESIDUALISE (paper §3.2 fold-internal residualisation)
        if not skip_residualise:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_residualise_phase():
                results["residualise"] = True
            else:
                self.logger.error("RESIDUALISE phase failed - continuing")
            self._record_timing(
                "RESIDUALISE", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE H: RESIDUALISE - SKIPPED")

        # Phase I: MKNN (paper §3.2 + Appendix MKNN alignment)
        if not skip_mknn:
            self.logger.info("\n" + "=" * 70)
            _phase_t0 = datetime.now()
            if self.run_mknn_phase():
                results["mknn"] = True
            else:
                self.logger.error("MKNN phase failed - continuing")
            self._record_timing(
                "MKNN", "total", (datetime.now() - _phase_t0).total_seconds()
            )
            self._flush_timing_csv()
        else:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PHASE I: MKNN - SKIPPED")

        # Final timing flush (ensures file is complete)
        if self.save_time and self._timing_records:
            self._flush_timing_csv()
            self.logger.info(f"Timing saved to: {self._timing_path}")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Output directory: {output_base}")
        self.logger.info(f"Results:")

        gm_status = (
            "✓ Success"
            if results["general_metrics"] is True
            else (
                "⏸ Skipped" if results["general_metrics"] == "skipped" else "✗ Failed"
            )
        )
        fm_emb_status = (
            "✓ Success"
            if results["fm_embedding"] is True
            else ("⏸ Skipped" if results["fm_embedding"] == "skipped" else "✗ Failed")
        )
        decoder_status = (
            "✓ Success"
            if results["decoder"] is True
            else ("⏸ Skipped" if results["decoder"] == "skipped" else "✗ Failed")
        )
        markers_status = (
            "✓ Success"
            if results["markers"] is True
            else ("⏸ Skipped" if results["markers"] == "skipped" else "✗ Failed")
        )
        model_status = (
            "✓ Success"
            if results["model"] is True
            else ("⏸ Skipped" if results["model"] == "skipped" else "✗ Failed")
        )

        self.logger.info(f"  - GENERAL_METRICS: {gm_status}")
        self.logger.info(f"  - FM_EMBEDDING: {fm_emb_status}")
        self.logger.info(f"  - DECODER: {decoder_status}")
        self.logger.info(f"  - MARKERS: {markers_status}")
        if self.task != "rs":
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
            "subjects": subjects,
            "output_dir": str(output_base),
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
            # PYTHONUNBUFFERED=1 forces line-buffered output in child process
            # (otherwise Python block-buffers when stdout is a pipe, causing apparent hangs)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Prevent MKL/OpenBLAS threading crashes in subprocess
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                bufsize=1,
                env=env,
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
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

    def _record_timing(
        self,
        phase: str,
        step: str,
        duration_s: float,
        subject_id: str = "*",
        session: str = "*",
        file_type: str = "*",
    ) -> None:
        """Append a timing record (only when --save-time is active). Thread-safe."""
        if not self.save_time:
            return
        with self._timing_lock:
            self._timing_records.append(
                {
                    "phase": phase,
                    "step": step,
                    "subject_id": subject_id,
                    "session": session,
                    "file_type": file_type,
                    "duration_s": round(duration_s, 2),
                }
            )

    def _flush_timing_csv(self) -> None:
        """Overwrite the timing CSV with all records accumulated so far. Thread-safe."""
        if not self.save_time or not self._timing_path:
            return
        import csv

        with self._timing_lock:
            if not self._timing_records:
                return
            fieldnames = [
                "phase",
                "step",
                "subject_id",
                "session",
                "file_type",
                "duration_s",
            ]
            with open(self._timing_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._timing_records)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Unified EEG Foundation Benchmark Pipeline - Supports CBraMod, TOTEM, LaBram and standard data structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # CBraMod DoC patients — all phases
  python pipeline.py \\
      --main-path /data/CbraMod/recon_data_inference \\
      --metadata-dir /data/metadata \\
      --mode patient --task lg --data-source CBraMod \\
      --results-subdir CBraMod/doc_patients --all

  # Standard DoC local-global — all phases
  python pipeline.py \\
      --main-path /data/doc --metadata-dir /data/metadata \\
      --mode patient --task lg --all

  # Control resting-state
  python pipeline.py \\
      --main-path /data/control_rs --metadata-dir /data/metadata \\
      --mode control --task rs --all

  # Only markers phase
  python pipeline.py \\
      --main-path /data --metadata-dir /data/metadata \\
      --mode patient --task lg --all --markers-only

  # FM embedding phase only: binary VS/MCS with 5-fold CV
  python pipeline.py \\
      --main-path /data --metadata-dir /data/metadata \\
      --mode patient --task lg --all --fm-embedding-only \\
      --embedding-data-dir /data/embeddings/totem \\
      --emb-full-cv

  # MODEL phase only: SVM with baseline CSV and full CV
  python pipeline.py \\
      --main-path /data --metadata-dir /data/metadata \\
      --mode patient --task lg --all --model-only \\
      --model-baseline-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \\
      --model-full-cv --model-use-subject-intersection

  # Skip general metrics and decoder, run everything else
  python pipeline.py \\
      --main-path /data --metadata-dir /data/metadata \\
      --mode patient --task lg --all \\
      --skip-general-metrics --skip-decoder

  # Subject intersection across two foundation models (embedding phase)
  python pipeline.py \\
      --main-path /data --metadata-dir /data/metadata \\
      --mode patient --task lg --all --fm-embedding-only \\
      --emb-use-subject-intersection \\
      --emb-all-embedding-dirs TOTEM=/data/totem_emb CBraMod=/data/cbramod_emb
        """,
    )

    # Set default paths relative to script location
    project_root = str(Path(__file__).parent.parent)

    # Core configuration
    parser.add_argument(
        "--main-path",
        required=True,
        help="Path to main data directory with sub-{num}/ses-{num}/ structure",
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        help="Path to metadata directory (must contain patient_labels_with_controls.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["patient", "control"],
        required=True,
        help="Analysis mode: patient (DoC) or control",
    )
    parser.add_argument(
        "--task",
        choices=["lg", "rs"],
        required=True,
        help="Task paradigm: lg (local-global) or rs (resting-state)",
    )

    # Data source and output configuration
    parser.add_argument(
        "--data-source",
        choices=["auto", "CBraMod", "TOTEM", "LaBram", "standard", "suffix", "bids"],
        default="auto",
        help='Data source identifier (default: auto-detect). Use "bids" for BIDS-like eeg/ subdir structure.',
    )
    parser.add_argument(
        "--results-subdir",
        help="Subdirectory within results for output (e.g., CBraMod/doc_patients)",
    )
    parser.add_argument(
        "--original-data-path",
        nargs="+",
        dest="original_data_paths",
        help="Path(s) to original data if separate from main-path. Supports multiple paths searched in order.",
    )
    parser.add_argument(
        "--file-pattern",
        help="Custom file pattern to match (e.g., *_vqnsp_reconstructed_epo.fif)",
    )

    # Subject selection (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--subject", help="Process single subject ID")
    subject_group.add_argument(
        "--subjects", help="Process specific subjects (comma-separated IDs)"
    )
    subject_group.add_argument(
        "--all", action="store_true", help="Process all available subjects"
    )
    subject_group.add_argument(
        "--random", type=int, metavar="N", help="Process N random subjects"
    )

    # Optional configuration
    parser.add_argument(
        "--results-dir",
        default=f"{project_root}/results",
        help="Path to results directory (default: ./results)",
    )
    parser.add_argument(
        "--src-dir",
        default=f"{project_root}/src",
        help="Path to source code directory (default: ./src)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1, sequential)",
    )

    # Decoder parameters
    parser.add_argument(
        "--decoder-cv",
        type=int,
        default=3,
        help="Cross-validation folds for decoder (default: 3)",
    )
    parser.add_argument(
        "--decoder-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for decoder (default: 1)",
    )

    # Markers parameters
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip cluster permutation tests in markers computation",
    )
    parser.add_argument(
        "--keep-h5", action="store_true", help="Keep H5 files after markers computation"
    )

    # Model parameters
    parser.add_argument(
        "--model-marker-type",
        choices=["scalar", "topo"],
        default="scalar",
        help="Marker type for model training (default: scalar)",
    )
    parser.add_argument(
        "--model-orig-data-dir",
        default=None,
        help=(
            "Path to a separate directory tree containing the ORIGINAL "
            "(non-reconstructed) marker files "
            "(sub-{ID}/ses-{N}/orig/scalars_*.npz). "
            "When set, the pipeline MARKERS output dir is used for recon and "
            "this path is used for orig; only the intersection of "
            "subject/sessions found in both trees is loaded."
        ),
    )

    parser.add_argument(
        "--model-baseline-csv",
        default=None,
        help=(
            "Path to a baseline CSV file (semicolon-delimited) with "
            "pre-computed original scalar markers. When set, this CSV "
            "is used as the source of 'original' data instead of npz files."
        ),
    )
    parser.add_argument(
        "--model-use-subject-intersection",
        action="store_true",
        default=False,
        help=(
            "Restrict the SVM subject pool to the intersection of subjects "
            "available in both the original source (baseline CSV or "
            "orig-data-dir) and the reconstructed data."
        ),
    )
    parser.add_argument(
        "--model-full-cv",
        action="store_true",
        default=False,
        help=(
            "Run full 5-fold StratifiedGroupKFold cross-validation for the "
            "SVM MODEL phase instead of a single train/test split."
        ),
    )

    # MLP Embedding parameters
    parser.add_argument(
        "--embedding-data-dir",
        help="Path to pre-computed embeddings (sub-{ID}/ses-{NUM}/*_embeddings.npy)",
    )
    parser.add_argument(
        "--mlp-n-epochs",
        type=int,
        default=500,
        help="Max training epochs for FM embedding classifier (default: 500)",
    )
    parser.add_argument(
        "--mlp-lr",
        type=float,
        default=1e-3,
        help="Learning rate for FM embedding classifier (default: 0.001)",
    )
    parser.add_argument(
        "--mlp-batch-size",
        type=int,
        default=32,
        help="Batch size for FM embedding classifier (default: 32)",
    )
    parser.add_argument(
        "--emb-full-cv",
        action="store_true",
        default=False,
        help="Run 5-fold StratifiedGroupKFold CV for embedding classifiers",
    )
    parser.add_argument(
        "--emb-use-subject-intersection",
        action="store_true",
        default=False,
        help="Restrict embedding subjects to intersection across all foundation models",
    )
    parser.add_argument(
        "--emb-all-embedding-dirs",
        nargs="+",
        metavar="NAME=PATH",
        help="Embedding dirs for all models (e.g., TOTEM=/path CBraMod=/path)",
    )
    parser.add_argument(
        "--emb-marker-regression",
        action="store_true",
        default=False,
        help="Run Ridge marker regression (embeddings → scalar markers) in the MLP EMBEDDING phase",
    )
    parser.add_argument(
        "--emb-marker-csv",
        default=None,
        help=(
            "Path to baseline scalars CSV for marker regression "
            "(required with --emb-marker-regression)"
        ),
    )
    parser.add_argument(
        "--emb-marker-reduction",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Reduction variant for marker CSV: A=trim_mean80, B=std, C=gfp/trim_mean80, D=gfp/std (default: A)",
    )
    parser.add_argument(
        "--emb-full-metric-prediction",
        action="store_true",
        default=False,
        help="Run MLP+RF+KR for 5 clinical targets (crs, etiology, cs_6m, cs_1y, cs_2y) in the MLP EMBEDDING phase",
    )
    parser.add_argument(
        "--emb-patient-labels-full",
        default=None,
        help=(
            "Path to patient_labels.csv with all target columns "
            "(required with --emb-full-metric-prediction)"
        ),
    )

    # Phase skipping
    parser.add_argument(
        "--skip-general-metrics", action="store_true", help="Skip GENERAL_METRICS phase"
    )
    parser.add_argument(
        "--skip-fm-embedding", action="store_true", help="Skip FM_EMBEDDING phase"
    )
    parser.add_argument(
        "--skip-decoder", action="store_true", help="Skip DECODER phase"
    )
    parser.add_argument(
        "--skip-markers", action="store_true", help="Skip MARKERS phase"
    )
    parser.add_argument("--skip-model", action="store_true", help="Skip MODEL phase")

    # Paper-evaluation phases (off by default; enabled by --paper-eval or *-only)
    parser.add_argument(
        "--skip-probing", action="store_true", default=True,
        help="Skip layer-wise PROBING phase (default: skipped)",
    )
    parser.add_argument(
        "--skip-fm-plus-dk", action="store_true", default=True,
        help="Skip FM_PLUS_DK phase (default: skipped)",
    )
    parser.add_argument(
        "--skip-residualise", action="store_true", default=True,
        help="Skip RESIDUALISE phase (default: skipped)",
    )
    parser.add_argument(
        "--skip-mknn", action="store_true", default=True,
        help="Skip MKNN alignment phase (default: skipped)",
    )

    # Only run specific phases
    parser.add_argument(
        "--general-metrics-only",
        action="store_true",
        help="Only run GENERAL_METRICS phase",
    )
    parser.add_argument(
        "--fm-embedding-only", action="store_true", help="Only run FM_EMBEDDING phase"
    )
    parser.add_argument(
        "--markers-only",
        action="store_true",
        help="Only run MARKERS phase (HDF5 + scalars + topos)",
    )
    parser.add_argument(
        "--decoder-only", action="store_true", help="Only run DECODER phase"
    )
    parser.add_argument(
        "--model-only", action="store_true", help="Only run MODEL phase"
    )
    parser.add_argument(
        "--probing-only", action="store_true",
        help="Only run PROBING phase (paper §3.2 layer-wise R²/AUC)",
    )
    parser.add_argument(
        "--fm-plus-dk-only", action="store_true",
        help="Only run FM_PLUS_DK phase (paper §3.2 FM⊕DK concatenation)",
    )
    parser.add_argument(
        "--residualise-only", action="store_true",
        help="Only run RESIDUALISE phase (paper §3.2 fold-internal residualisation)",
    )
    parser.add_argument(
        "--mknn-only", action="store_true",
        help="Only run MKNN alignment phase (paper §3.2 + Appendix)",
    )
    parser.add_argument(
        "--paper-eval", action="store_true",
        help=(
            "End-to-end NeurIPS paper evaluation: runs the seven steps "
            "(FM_EMBEDDING + PROBING + FM_PLUS_DK + RESIDUALISE + MKNN). "
            "Skips GENERAL_METRICS, DECODER, MARKERS, MODEL."
        ),
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--save-time",
        action="store_true",
        default=False,
        help="Save per-step timing data to a CSV file in the logs directory",
    )

    args = parser.parse_args()

    # Handle *-only flags
    if args.general_metrics_only:
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_general_metrics = False
    elif args.fm_embedding_only:
        args.skip_general_metrics = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_fm_embedding = False
    elif args.markers_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_model = True
        args.skip_markers = False
    elif args.decoder_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_decoder = False
    elif args.model_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = False
    elif args.probing_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_probing = False
    elif args.fm_plus_dk_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_fm_plus_dk = False
    elif args.residualise_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_residualise = False
    elif args.mknn_only:
        args.skip_general_metrics = True
        args.skip_fm_embedding = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_mknn = False
    elif args.paper_eval:
        # End-to-end paper evaluation: A (utility nested-CV) + B/C (probing) +
        # D (combined DK) + E (residualisation) + F/G (MKNN + k-sweep).
        args.skip_general_metrics = True
        args.skip_decoder = True
        args.skip_markers = True
        args.skip_model = True
        args.skip_fm_embedding = False
        args.skip_probing = False
        args.skip_fm_plus_dk = False
        args.skip_residualise = False
        args.skip_mknn = False

    # Resolve subject arguments
    subject_args = []
    if args.subject:
        subject_args = [args.subject]
    elif args.subjects:
        subject_args = args.subjects.split(",")
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
        model_marker_type=args.model_marker_type,
        model_orig_data_dir=args.model_orig_data_dir,
        model_baseline_csv=args.model_baseline_csv,
        model_use_subject_intersection=args.model_use_subject_intersection,
        model_full_cv=args.model_full_cv,
        embedding_data_dir=args.embedding_data_dir,
        mlp_n_epochs=args.mlp_n_epochs,
        mlp_lr=args.mlp_lr,
        mlp_batch_size=args.mlp_batch_size,
        emb_full_cv=args.emb_full_cv,
        emb_use_subject_intersection=args.emb_use_subject_intersection,
        emb_all_embedding_dirs=args.emb_all_embedding_dirs,
        emb_marker_regression=args.emb_marker_regression,
        emb_marker_csv=args.emb_marker_csv,
        emb_marker_reduction=args.emb_marker_reduction,
        emb_full_metric_prediction=args.emb_full_metric_prediction,
        emb_patient_labels_full=args.emb_patient_labels_full,
        data_source=args.data_source,
        results_subdir=args.results_subdir,
        original_data_paths=args.original_data_paths,
        file_pattern=args.file_pattern,
        save_time=args.save_time,
    )

    # Resolve subjects
    subjects = pipeline.resolve_subjects(subject_args)

    if not subjects:
        print("No subjects to process")
        return

    # Run pipeline
    result = pipeline.run_pipeline(
        subjects,
        skip_general_metrics=args.skip_general_metrics,
        skip_fm_embedding=args.skip_fm_embedding,
        skip_decoder=args.skip_decoder,
        skip_markers=args.skip_markers,
        skip_model=args.skip_model,
        skip_probing=args.skip_probing,
        skip_fm_plus_dk=args.skip_fm_plus_dk,
        skip_residualise=args.skip_residualise,
        skip_mknn=args.skip_mknn,
    )

    # Exit with appropriate code
    if result.get("status") == "complete":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
