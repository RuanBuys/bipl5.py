"""Internal ordination engine for bipl5 — a Python port of the parts of the
R package `biplotEZ <https://cran.r-project.org/package=biplotEZ>`_ (MIT,
Lubbe et al.) that bipl5 relies on.

The workflow mirrors the R pipeline::

    from bipl5.ordination import biplot, pca, fit_measures, axes_coordinates

    bp = biplot(data, center=True, scaled=False)
    bp = pca(bp, e_vects=(1, 2))          # or: bp = bp.pca(e_vects=(1, 2))
    bp = fit_measures(bp)
    ticks = axes_coordinates(bp)

Implemented methods: PCA (incl. correlation biplots), CVA (incl. the
``sample.opt`` low-dimension strategy), PCO with regression axes, and
regression biplots on user-supplied coordinates. PCO/regression spline axes
raise ``NotImplementedError`` for now.

This module is deliberately self-contained (NumPy/pandas only, no plotting)
so it can later grow into a full biplotEZ replication for static biplots.
"""

from .aesthetics import EZ_COL, GREY_07, axes, means, samples
from .base import EZBiplot, as_factor, biplot, indmat
from .calibration import axes_coordinates, calibrate_axis
from .cva import cva
from .fit_measures import (
    cumulative_adequacies,
    cumulative_axis_predictivities,
    fit_measures,
    fit_quality_string,
    marginal_axis_predictivities,
)
from .pca import pca
from .pco import euclidean_dist, extended_matching_coefficient, pco
from .regress import regress
from .splines import spline_axis, spline_axis_control

__all__ = [
    "EZBiplot",
    "biplot",
    "pca",
    "cva",
    "pco",
    "regress",
    "samples",
    "axes",
    "means",
    "fit_measures",
    "axes_coordinates",
    "calibrate_axis",
    "cumulative_axis_predictivities",
    "cumulative_adequacies",
    "marginal_axis_predictivities",
    "fit_quality_string",
    "euclidean_dist",
    "extended_matching_coefficient",
    "spline_axis",
    "spline_axis_control",
    "as_factor",
    "indmat",
    "EZ_COL",
    "GREY_07",
]
