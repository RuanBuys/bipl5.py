"""Payload layer: builds the plotly trace/layout dictionaries (mdsDisplays)
that bipl5's vendored JavaScript consumes. Ports of the R package's
``mdsDisplay_constructor.R``, ``PCAbiplot_Helper.R``,
``build_secondary_biplots.R`` and ``wrap_bipl5_helper.R``.
"""

from .build_one import build_one_mds_display, clean_linear_axes_coordinates, obtain_xhat
from .hover import hovertext_generator
from .mds_display import (
    mds_display_add_config,
    mds_display_add_layout,
    mds_display_add_traces,
    mds_display_new,
)

__all__ = [
    "build_one_mds_display",
    "clean_linear_axes_coordinates",
    "obtain_xhat",
    "hovertext_generator",
    "mds_display_new",
    "mds_display_add_traces",
    "mds_display_add_layout",
    "mds_display_add_config",
]
