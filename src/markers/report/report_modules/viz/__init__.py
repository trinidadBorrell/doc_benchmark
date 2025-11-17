"""
Visualization functions extracted from nice_ext for Next-ICM reports.

This module contains only the viz functions actually used by the report generator,
extracted from the nice_ext library to make the codebase self-contained.
"""

from .evokeds import (
    plot_gfp,
    plot_evoked_topomap,
    plot_evoked,
    plot_cluster_test,
    plot_ttest,
    plot_cnv,
)
from .topos import (
    plot_marker_topo,
    plot_markers_topos,
)
from .utils import (
    plot_bad_channels,
    render_preprocessing_summary,
    render_prediction_summary,
    get_stat_colormap,
)
from .reductions import trim_mean80
from .contrast import get_contrast

__all__ = [
    'plot_gfp',
    'plot_evoked_topomap',
    'plot_evoked',
    'plot_cluster_test',
    'plot_ttest',
    'plot_cnv',
    'plot_marker_topo',
    'plot_markers_topos',
    'plot_bad_channels',
    'render_preprocessing_summary',
    'render_prediction_summary',
    'get_stat_colormap',
    'trim_mean80',
    'get_contrast',
]
