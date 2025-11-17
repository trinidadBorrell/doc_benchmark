"""
Computation modules for Next-ICM report generation.

This package contains all heavy computation logic separated from visualization.
"""

from .cnv import compute_cnv_analysis_data
from .connectivity import compute_connectivity_analysis_data
from .diagnostic import compute_diagnostic_data
from .information_theory import compute_information_theory_data
from .erp import compute_erp_analysis_data
from .spectral import compute_spectral_analysis_data

__all__ = [
    'compute_cnv_analysis_data',
    'compute_connectivity_analysis_data',
    'compute_diagnostic_data',
    'compute_information_theory_data',
    'compute_erp_analysis_data',
    'compute_spectral_analysis_data',
]
