"""Rendering layer: assembles the plotly figure and delivers it as HTML with
the vendored bipl5 JavaScript attached (the Python replacement for R's
htmlwidgets machinery).
"""

from .figure import figure_scaffolding
from .widget import Bipl5Widget

__all__ = ["figure_scaffolding", "Bipl5Widget"]
