"""Geometry helpers for static calibrated biplot axes — port of
``wrap_bipl5_gg_helper.R`` and the ``draw_panel()`` helpers of
``geom_calibrated_axis.R`` from the R package.

Everything here is pure NumPy so it can be unit-tested without a plotting
backend. The matplotlib artist in :mod:`bipl5.static.calibrated_axis` calls
these at draw time, once the final plotting window is known.

The central idea (shared with the R package): ``axes_coordinates()`` already
returns exact calibrated marker positions, so instead of re-deriving the
calibration, the affine value-to-position map those markers imply is
*recovered*. Markers can then be regenerated at the spacing and phase that
biplotEZ chose, but across the whole visible axis — so tick marks always run
right up to the edge of the plot, whatever the final window turns out to be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PT_PER_MM",
    "AxisCalibration",
    "fold_angle",
    "axis_calibration",
    "curve_angles",
    "line_window_span",
    "format_tick_values",
    "extend_axis_values",
    "inside_window",
    "dodge_direction",
    "resolve_axis_aes",
    "r_linetype",
]

#: ggplot2's ``.pt``: points per millimetre, for converting R text sizes.
PT_PER_MM = 72.27 / 25.4


def fold_angle(angle):
    """Fold angles (radians) into the half-open interval ``(-pi/2, pi/2]``.

    Port of ``gg_fold_angle()``. Text written along a biplot axis is rotated
    by the folded angle — the undirected slope angle — so it never reads
    upside down. Vertical directions fold to ``pi/2``.
    """
    arr = np.asarray(angle, dtype=float)
    folded = np.arctan(np.tan(arr))
    folded = np.where(np.isfinite(folded), folded, np.pi / 2)
    folded = np.where(folded <= -np.pi / 2 + 1e-12, folded + np.pi, folded)
    if folded.ndim == 0:
        return float(folded)
    return folded


@dataclass(frozen=True)
class AxisCalibration:
    """The affine calibration ``v -> (x0 + v*dxdv, y0 + v*dydv)`` of one
    linear biplot axis, recovered by :func:`axis_calibration`.

    Attributes
    ----------
    x0, y0:
        Display position of the variable value ``0``.
    dxdv, dydv:
        Displacement in the biplot plane per unit of the variable.
    angle:
        Direction of increasing variable value, in radians.
    vref:
        A marker value supplied by biplotEZ, anchoring the phase of any
        regenerated markers.
    vstep:
        Spacing between successive biplotEZ markers (``nan`` if unknown).
    offset:
        Foot of the perpendicular from the origin onto the axis; ``(0, 0)``
        for an axis through the origin, non-zero for translated axes.
    """

    x0: float
    y0: float
    dxdv: float
    dydv: float
    angle: float
    vref: float
    vstep: float
    offset: tuple[float, float]

    def position(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Display coordinates of the given variable values."""
        v = np.asarray(values, dtype=float)
        return self.x0 + v * self.dxdv, self.y0 + v * self.dydv

    def predict(self, x, y) -> np.ndarray:
        """Variable value read off this axis at biplot coordinates
        ``(x, y)``: the value of the marker at the foot of the orthogonal
        projection. Broadcasts over arrays — e.g. a ``meshgrid`` for a
        heatmap of predicted values drawn beneath the biplot."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        len2 = self.dxdv**2 + self.dydv**2
        return ((x - self.x0) * self.dxdv + (y - self.y0) * self.dydv) / len2

    @property
    def direction(self) -> np.ndarray:
        """Displacement per unit of the variable, as a length-2 array."""
        return np.array([self.dxdv, self.dydv])

    @property
    def has_step(self) -> bool:
        return bool(np.isfinite(self.vstep) and self.vstep > 0)


def axis_calibration(ax) -> AxisCalibration | None:
    """Recover the affine calibration of one linear biplot axis.

    Port of ``gg_axis_calibration()``. ``ax`` is one element of
    :func:`bipl5.ordination.axes_coordinates`: a ``k x 3`` (or wider) array
    of marker x, y and variable value. The map is fitted by least squares
    over all markers, which keeps the direction stable when the stored
    coordinates carry rounding noise.

    Returns ``None`` when the calibration is degenerate: fewer than two
    finite markers, constant marker values, or a zero-length axis.
    """
    ax = np.asarray(ax, dtype=float)
    if ax.ndim != 2 or ax.shape[0] < 2 or ax.shape[1] < 3:
        return None

    px, py, vv = ax[:, 0], ax[:, 1], ax[:, 2]
    keep = np.isfinite(px) & np.isfinite(py) & np.isfinite(vv)
    px, py, vv = px[keep], py[keep], vv[keep]
    if vv.size < 2:
        return None

    vbar = vv.mean()
    vdev = vv - vbar
    denom = float(np.sum(vdev**2))
    if not np.isfinite(denom) or denom <= 0:
        return None

    dxdv = float(np.sum(vdev * (px - px.mean())) / denom)
    dydv = float(np.sum(vdev * (py - py.mean())) / denom)
    if not (np.isfinite(dxdv) and np.isfinite(dydv)):
        return None
    len2 = dxdv**2 + dydv**2
    if len2 <= 0:
        return None

    x0 = float(px.mean() - dxdv * vbar)
    y0 = float(py.mean() - dydv * vbar)

    steps = np.diff(np.unique(vv))
    vstep = float(np.median(steps)) if steps.size else math.nan
    if not np.isfinite(vstep) or vstep <= 0:
        vstep = math.nan

    proj = (x0 * dxdv + y0 * dydv) / len2
    off = np.array([x0 - proj * dxdv, y0 - proj * dydv])
    off[np.abs(off) < 1e-10] = 0.0

    return AxisCalibration(
        x0=x0,
        y0=y0,
        dxdv=dxdv,
        dydv=dydv,
        angle=float(math.atan2(dydv, dxdv)),
        vref=float(vv[0]),
        vstep=vstep,
        offset=(float(off[0]), float(off[1])),
    )


def curve_angles(px, py, vv) -> np.ndarray:
    """Local direction of increasing value along a curved (spline) axis.

    Port of ``gg_curve_angles()``: central differences, one-sided at the
    ends, each tangent oriented towards increasing variable value. Vertices
    without a usable tangent get ``nan``.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    vv = np.asarray(vv, dtype=float)
    n = px.size
    if n < 2:
        return np.full(n, np.nan)

    idx = np.arange(n)
    lo = np.maximum(idx - 1, 0)
    hi = np.minimum(idx + 1, n - 1)
    dx = px[hi] - px[lo]
    dy = py[hi] - py[lo]
    dv = vv[hi] - vv[lo]

    flip = np.isfinite(dv) & (dv < 0)
    dx = np.where(flip, -dx, dx)
    dy = np.where(flip, -dy, dy)

    with np.errstate(invalid="ignore"):
        ang = np.arctan2(dy, dx)
    bad = ~np.isfinite(dx) | ~np.isfinite(dy) | ((dx == 0) & (dy == 0))
    ang[bad] = np.nan
    return ang


def line_window_span(p0, u, xrange, yrange) -> tuple[float, float] | None:
    """Parameter interval of ``p0 + t*u`` that lies inside the window.

    Port of ``gg_line_window_span()``. Returns ``(tmin, tmax)``, or ``None``
    when the line misses the plotting rectangle. With ``p0`` the position of
    value 0 and ``u`` the per-unit displacement of a calibration, ``t`` *is*
    the variable value, so the span is the visible value range of the axis.
    """
    tol = 1e-12
    tmin, tmax = -math.inf, math.inf
    for p, d, lim in (
        (float(p0[0]), float(u[0]), xrange),
        (float(p0[1]), float(u[1]), yrange),
    ):
        lo, hi = float(min(lim)), float(max(lim))
        if abs(d) < tol:
            # constant in this coordinate: either always or never inside
            if p < lo or p > hi:
                return None
            continue
        t1, t2 = (lo - p) / d, (hi - p) / d
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))

    if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmax < tmin:
        return None
    return tmin, tmax


def _format_one(value: float, digits: int) -> str:
    r = round(float(value), digits) + 0.0  # "+ 0.0" turns -0.0 into 0.0
    if digits <= 0:
        return f"{r:.0f}"
    s = f"{r:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def format_tick_values(values, step: float = math.nan) -> list[str]:
    """Format calibrated marker values for printing.

    Port of ``gg_format_tick_values()``. Regenerated markers are floating
    point sums (``0.30000000000000004``); the number of decimals is derived
    from the marker spacing so neighbouring markers stay distinguishable,
    and trailing zeros are dropped.
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.size == 0:
        return []
    if np.isfinite(step) and step > 0:
        digits = max(0, min(10, math.ceil(-math.log10(step)) + 1))
    else:
        digits = 3
    return [_format_one(v, digits) for v in values]


def extend_axis_values(
    cal: AxisCalibration, xrange, yrange, max_marks: int = 1000
) -> np.ndarray | None:
    """Marker values spanning the whole visible part of a linear axis.

    Port of ``gg_extend_axis_marks()``: markers are emitted at
    ``vref + k*vstep``, keeping the spacing and phase biplotEZ chose, for
    every ``k`` whose marker falls inside the window. Returns ``None`` when
    the axis misses the window, the calibration has no step, or spanning
    the window would need more than ``max_marks`` markers (the caller then
    falls back to the supplied markers).
    """
    if not cal.has_step:
        return None
    span = line_window_span((cal.x0, cal.y0), cal.direction, xrange, yrange)
    if span is None:
        return None
    vref = cal.vref if np.isfinite(cal.vref) else 0.0
    kmin = math.ceil((span[0] - vref) / cal.vstep - 1e-9)
    kmax = math.floor((span[1] - vref) / cal.vstep + 1e-9)
    if kmax < kmin or kmax - kmin > max_marks:
        return None
    return vref + np.arange(kmin, kmax + 1) * cal.vstep


def inside_window(x, y, xrange, yrange, tol: float = 1e-9) -> np.ndarray:
    """Port of ``gg_inside_window()``: points inside the plotting window."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= min(xrange) - tol)
        & (x <= max(xrange) + tol)
        & (y >= min(yrange) - tol)
        & (y <= max(yrange) + tol)
    )


def dodge_direction(angle, offset=(0.0, 0.0)) -> np.ndarray:
    """Unit vector orthogonal to an axis, for ticks and marker values.

    Port of ``gg_dodge_direction()``. Translated axes dodge away from the
    origin (along their offset); axes through the origin dodge towards the
    upward normal of the folded angle. Marker values are drawn on the
    *opposite* side of this vector — below the axis, as biplotEZ does — and
    variable names on this side.
    """
    off = np.asarray(offset, dtype=float)
    length = float(np.hypot(off[0], off[1]))
    if length > 1e-10:
        return off / length
    fold = fold_angle(angle)
    return np.array([-math.sin(fold), math.cos(fold)])


def resolve_axis_aes(value, ez_value, default, n: int) -> list:
    """Resolve a per-axis aesthetic specification (port of ``gg_axis_aes()``).

    ``value`` may be a single value (recycled), one value per axis, ``None``
    (use ``default``), or the string ``"biplotEZ"`` to adopt the aesthetics
    stored on the ordination object by :func:`bipl5.ordination.axes`.
    ``default`` may itself be a per-axis list.
    """
    if value is None:
        value = default
    if isinstance(value, str) and value == "biplotEZ":
        value = ez_value
    if value is None or (not isinstance(value, str) and _is_empty(value)):
        value = default
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        value = [value]
    value = list(value)
    if not value:
        value = [default]
    return [value[i % len(value)] for i in range(n)]


def _is_empty(value) -> bool:
    try:
        return len(value) == 0
    except TypeError:
        return False


# R line types (names and lty integers) -> matplotlib dash styles
_R_LINETYPES = {
    "solid": "-",
    "dashed": (0, (4, 4)),
    "dotted": (0, (1, 3)),
    "dotdash": (0, (1, 3, 4, 3)),
    "longdash": (0, (7, 3)),
    "twodash": (0, (2, 2, 6, 2)),
    "blank": "None",
}
_R_LTY = ["blank", "solid", "dashed", "dotted", "dotdash", "longdash", "twodash"]


def r_linetype(lty):
    """Translate an R line type into a matplotlib ``linestyle``.

    Accepts R names (``"dashed"``), R ``lty`` integers (``2``) and anything
    matplotlib already understands (``"--"``, ``(0, (3, 1))``), which passes
    through unchanged.
    """
    if isinstance(lty, (bool, np.bool_)):
        return "-"
    if isinstance(lty, (int, np.integer)) or (
        isinstance(lty, (float, np.floating)) and float(lty).is_integer()
    ):
        lty = _R_LTY[int(lty) % len(_R_LTY)]
    if isinstance(lty, str) and lty in _R_LINETYPES:
        return _R_LINETYPES[lty]
    return lty
