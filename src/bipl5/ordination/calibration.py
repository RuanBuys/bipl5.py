"""Calibrated axis coordinates — ports of ``biplotEZ::axes_coordinates()``
and the internal ``.calibrate.axis()``.

Each biplot axis is a straight line through the display; calibration places
tick marks on it so that the orthogonal projection of a sample point onto
the axis can be read off in the original units of that variable.
"""

from __future__ import annotations

import numpy as np

from ..rcompat import pretty
from .aesthetics import axes as set_axes
from .base import EZBiplot

__all__ = ["axes_coordinates", "calibrate_axis"]


def calibrate_axis(
    j: int,
    Xhat: np.ndarray,
    means: np.ndarray,
    sd: np.ndarray,
    axes_rows: np.ndarray,
    ax_which: list[int],
    ax_tickvec: list[int],
    ax_orthogx: list[float],
    ax_orthogy: list[float],
) -> dict:
    """Port of biplotEZ's ``.calibrate.axis()`` for one axis.

    Tick values are chosen with :func:`~bipl5.rcompat.pretty` over the range
    of the predicted values for the variable, then extended one full range
    below and above (the extended ticks are trimmed to the plot area later
    by bipl5's display code). Tick positions along the axis direction are
    the standardized tick values ``(t - mean_j)/sd_j`` (offset by the
    orthogonal shift ``phi`` when non-zero axis offsets are used).

    Returns a dict with ``coords`` (k x 3: x, y, tick label in raw units),
    plus the line's intercept ``a``, slope ``b`` and vertical-line ``v``.
    """
    ax_num = ax_which[j]
    tick = ax_tickvec[j]
    ax_direction = axes_rows[ax_num, :]
    r = axes_rows.shape[1]

    ax_orthog = np.vstack([ax_orthogx, ax_orthogy])
    if ax_orthog.shape[0] < r:
        ax_orthog = np.vstack([ax_orthog, np.zeros(ax_orthog.shape[1])])
    denom = np.sum(axes_rows**2, axis=1)
    phi_vec = (axes_rows / denom[:, None]) @ ax_orthog[:, ax_num]

    std_labels = pretty(
        (float(np.min(Xhat[:, ax_num])), float(np.max(Xhat[:, ax_num]))), n=tick
    )
    span = float(np.max(std_labels) - np.min(std_labels))
    std_labels = np.concatenate([std_labels, std_labels - span, std_labels + span])
    interval = (std_labels - means[ax_num]) / sd[ax_num]
    axis_vals = np.unique(interval)  # sorted unique, like R's sort(unique(.))

    axis_points = np.empty((axis_vals.shape[0], r + 1))
    for i in range(r):
        axis_points[:, i] = ax_orthog[i, ax_num] + (
            axis_vals - phi_vec[ax_num]
        ) * ax_direction[i]
    axis_points[:, r] = axis_vals * sd[ax_num] + means[ax_num]

    dx = axis_points[0, 0] - axis_points[1, 0]
    dy = axis_points[0, 1] - axis_points[1, 1]
    slope = None if dx == 0 else dy / dx
    v = axis_points[0, 0] if slope is None or not np.isfinite(slope) else None
    if v is not None:
        slope = None
    intercept = (
        None if slope is None else axis_points[0, 1] - slope * axis_points[0, 0]
    )

    return {"coords": axis_points, "a": intercept, "b": slope, "v": v}


def axes_coordinates(x: EZBiplot) -> list[np.ndarray]:
    """Port of ``biplotEZ::axes_coordinates()``.

    Returns one ``k x 3`` array per displayed axis: tick x/y coordinates in
    the biplot plane and the tick label in the variable's raw units. Default
    axis aesthetics are applied first when none are set, as in R.

    Predicted values for tick ranges come from ``Z @ inv(Lmat)[e_idx, :]``
    when a square ``Lmat`` exists (PCA/CVA) and from the data matrix
    otherwise (PCO/regression), back-transformed to raw units.
    """
    if x.Z is None:
        raise ValueError("apply an ordination method before axes_coordinates().")
    if x.axes is None:
        x = set_axes(x)
    ax_aes = x.axes

    if x.Lmat is not None and x.Lmat.shape[0] == x.Lmat.shape[1]:
        Xhat = x.Z @ np.linalg.inv(x.Lmat)[x.e_idx, :]
    else:
        Xhat = x.X
    if x.scaled:
        Xhat = Xhat * x.sd
    if x.center:
        Xhat = Xhat + x.means

    if x.PCOaxes == "splines":
        from .splines import spline_axis

        control = dict(x.spline_control or {})
        rng = np.random.default_rng(control.get("seed"))
        return [
            spline_axis(
                j, x.Z, x.X, x.means, x.sd, control=control, rng=rng
            )
            for j in ax_aes["which"]
        ]

    if x.ax_one_unit is None:
        raise ValueError("this biplot has no linear axis directions.")

    z_axes = []
    for j in range(len(ax_aes["which"])):
        out = calibrate_axis(
            j,
            Xhat,
            x.means,
            x.sd,
            x.ax_one_unit,
            ax_aes["which"],
            ax_aes["ticks"],
            ax_aes["orthogx"],
            ax_aes["orthogy"],
        )
        z_axes.append(out["coords"])
    return z_axes
