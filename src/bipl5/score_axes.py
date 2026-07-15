"""Alves direct-reading errors — port of ``score_axes.R``.

Augments every mdsDisplay's observation hover tables with the
direct-reading diagnostic of Alves (2012): for observation *i* and variable
*j*, ``100 * |x_ij - xhat_ij| / s_j`` where ``xhat_ij`` is the value read
directly off the calibrated axis (the orthogonal projection onto the
displayed axis) and ``s_j`` is 1 for unscaled data or the column standard
deviation when the biplot was built with ``scale=True``.
"""

from __future__ import annotations

import warnings

import numpy as np

from .biplot import Biplot
from .display.build_one import obtain_xhat
from .display.hover import hovertext_generator

__all__ = ["score_axes"]


class _FakeRegress:
    """Minimal object routing ``obtain_xhat()`` down the interpolation path,
    mirroring R's ``direct_reading_values()`` fake ``regress`` object."""

    method = "regress"
    Lmat = None
    scaled = False
    center = False

    def __init__(self, Z, X):
        self.Z = np.asarray(Z, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.n, _ = self.Z.shape
        self.p = self.X.shape[1]


def _direct_reading_values(Z, X, z_axes) -> np.ndarray:
    return obtain_xhat(_FakeRegress(Z, X), z_axes=z_axes)


def _trace_is_data(trace: dict) -> bool:
    meta = trace.get("meta")
    if meta is None:
        return False
    if isinstance(meta, str):
        return meta == "data"
    return "data" in list(meta)


def score_axes(bp: Biplot, digits: int = 2) -> Biplot:
    """Add Alves reading errors to all hover tables (port of ``score_axes()``).

    Applied per mdsDisplay because the read-off value depends on the
    displayed dimension pair. Returns the modified :class:`Biplot`; spline
    displays are returned unchanged with a warning, matching R.
    """
    if bp.meta.get("spline"):
        warnings.warn(
            "score_axes() does not support spline axes; direct-reading "
            "errors are only defined for calibrated linear axes. "
            "Returning the biplot unchanged.",
            stacklevel=2,
        )
        return bp

    ez = bp.meta.get("x")
    if ez is None:
        raise ValueError(
            "score_axes() requires a Biplot created by scale_mds()."
        )

    raw_X = np.asarray(ez.X, dtype=float)
    p = raw_X.shape[1]
    s = np.ones(p)
    if ez.scaled and ez.sd is not None:
        s = np.asarray(ez.sd, dtype=float)

    codes, levels = bp.meta["group"]
    sample_pred = ez.sample_predictivity
    if sample_pred is None:
        sample_pred = ez.within_class_sample_predictivity

    out = bp._copy()
    out.displays = dict(bp.displays)
    for name in bp.meta["pc_info"]:
        bundle = bp.displays.get(name)
        if bundle is None:
            continue
        Z = bundle["Data"]["sample_coordinates"]
        z_axes = bundle["Data"]["axes_coordinates"]
        if Z is None or z_axes is None:
            continue

        pred = _direct_reading_values(Z, raw_X, z_axes)
        reading_error = 100 * np.abs(raw_X - pred) / s

        obj = {
            "Z": Z,
            "group_codes": codes,
            "group_levels": levels,
            "n": ez.n,
            "x": raw_X,
            "row_names": ez.row_names,
            "col_names": ez.col_names,
            "XHat": pred,
            "sample_predictivity": sample_pred,
            "reading_error": reading_error,
            "reading_error_digits": digits,
        }

        hover_by_level = {
            level: hovertext_generator(obj, i, "<br />")
            for i, level in enumerate(levels)
        }

        new_bundle = dict(bundle)
        new_bundle["mds"] = dict(bundle["mds"])
        traces = []
        for trace in bundle["mds"]["trace_data"]:
            if _trace_is_data(trace) and trace.get("name") in hover_by_level:
                trace = dict(trace)
                trace["hovertext"] = hover_by_level[trace["name"]]
            traces.append(trace)
        new_bundle["mds"]["trace_data"] = traces
        out.displays[name] = new_bundle

    out.meta["reading_errors"] = True
    return out
