"""Sample, axis and class-mean aesthetics — ports of ``biplotEZ::samples()``,
``biplotEZ::axes()`` and ``biplotEZ::means()`` defaults.

Only the aesthetic fields consumed by bipl5 are carried: colours, plotting
characters, tick counts and axis label/tick colours. R colour names used in
the defaults are stored as hex so they mean the same thing in plotly/CSS
(R's ``"green"`` and ``"purple"`` differ from the CSS colours of the same
name).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import EZBiplot

__all__ = ["samples", "axes", "means", "EZ_COL", "GREY_07"]

# biplotEZ's ez.col palette, as hex equivalents of the R colour names
# c("blue","green","gold","cyan","magenta","black","red","grey","purple","salmon")
EZ_COL = [
    "#0000FF",
    "#00FF00",
    "#FFD700",
    "#00FFFF",
    "#FF00FF",
    "#000000",
    "#FF0000",
    "#BEBEBE",
    "#A020F0",
    "#FA8072",
]

# grDevices::grey(0.7), the biplotEZ default axis colour
GREY_07 = "#B3B3B3"


def _recycle(values, length: int) -> list:
    """R-style cyclic recycling of a scalar or vector to ``length``."""
    if values is None:
        return [None] * length
    if isinstance(values, (str, int, float, np.integer, np.floating)):
        values = [values]
    values = list(values)
    return [values[i % len(values)] for i in range(length)]


def samples(
    bp: EZBiplot,
    which: Sequence[int] | None = None,
    col=None,
    pch=16,
    cex=1.0,
    opacity: float = 1.0,
) -> EZBiplot:
    """Set per-group sample aesthetics (port of ``samples.biplot`` defaults).

    ``col`` defaults to the ``ez.col`` palette and ``pch`` to ``16`` (solid
    circle), both recycled to the number of groups. ``which`` selects groups
    by 0-based index (default: all).
    """
    bp = bp._copy()
    g = bp.g
    if which is None:
        which = list(range(g))
    col = _recycle(EZ_COL if col is None else col, g)
    pch = _recycle(pch, g)
    cex = _recycle(cex, g)
    bp.samples = {
        "which": list(which),
        "col": col,
        "pch": pch,
        "cex": cex,
        "opacity": float(opacity),
    }
    return bp


def axes(
    bp: EZBiplot,
    which: Sequence[int] | None = None,
    col=GREY_07,
    lwd=1,
    lty=1,
    ticks=5,
    tick_col=None,
    tick_label=True,
    tick_label_col=None,
    ax_names: Sequence[str] | None = None,
    orthogx=0.0,
    orthogy=0.0,
) -> EZBiplot:
    """Set per-variable axis aesthetics (port of ``axes.biplot`` defaults).

    Defaults follow biplotEZ: axis colour ``grey(0.7)``, five tick marks per
    axis, and tick/tick-label colours inheriting from the axis colour. All
    vectors are recycled to the number of displayed axes. ``which`` selects
    variables by 0-based index (default: all).
    """
    bp = bp._copy()
    p = bp.p
    if which is None:
        which = list(range(p))
    which = [w for w in which if 0 <= w < p]
    k = len(which)

    col = _recycle(col, k)
    tick_col = col if tick_col is None else _recycle(tick_col, k)
    tick_label_col = (
        list(tick_col) if tick_label_col is None else _recycle(tick_label_col, k)
    )
    if ax_names is None:
        ax_names = [bp.col_names[w] for w in which]

    bp.axes = {
        "which": list(which),
        "col": col,
        "lwd": _recycle(lwd, k),
        "lty": _recycle(lty, k),
        "ticks": _recycle(ticks, k),
        "tick_col": tick_col,
        "tick_label": _recycle(tick_label, k),
        "tick_label_col": tick_label_col,
        "ax_names": list(ax_names),
        "orthogx": _recycle(orthogx, p),
        "orthogy": _recycle(orthogy, p),
    }
    return bp


def means(
    bp: EZBiplot,
    which: Sequence[int] | None = None,
    col=None,
    pch=15,
    cex=1.0,
) -> EZBiplot:
    """Set class-mean aesthetics (port of ``means.biplot`` defaults).

    ``col`` defaults to the ``ez.col`` entry of each class and ``pch`` to
    ``15`` (solid square). Also fills ``Zmeans`` from the group means of
    ``Z`` when a method has been applied but class means were not requested
    at fit time (bipl5 needs coordinates whenever mean aesthetics exist).
    """
    bp = bp._copy()
    g = bp.g
    if which is None:
        which = list(range(g))
    if col is None:
        col = [EZ_COL[w % len(EZ_COL)] for w in which]
    bp.means_aes = {
        "which": list(which),
        "col": _recycle(col, len(which)),
        "pch": _recycle(pch, len(which)),
        "cex": _recycle(cex, len(which)),
    }
    if bp.Zmeans is None and bp.Z is not None and bp.g > 1:
        Zmeans = np.vstack(
            [bp.Z[bp.group == i].mean(axis=0) for i in range(bp.g)]
        )
        bp.Zmeans = Zmeans
    return bp
