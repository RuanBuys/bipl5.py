"""Assemble one complete mdsDisplay — port of ``wrap_bipl5_helper.R``
(``build_one_mdsDisplay()`` and the calibrated-tick post-processing) plus
``obtain_xhat()`` from ``biplotEZ_helper.R``.
"""

from __future__ import annotations

import numpy as np

from ..geometry import rotation_constructor
from ..ordination.aesthetics import axes as set_axes
from ..ordination.aesthetics import means as set_means
from ..ordination.calibration import axes_coordinates
from ..ordination.fit_measures import fit_quality_string
from ..printing import BiplotData, MdsDisplay
from ..rcompat import approx, pretty
from ..symbols import pch_to_plotly
from .builders import (
    add_tda,
    insert_class_means,
    insert_linear_axes,
    insert_polygon_ez,
    insert_vector_annots,
    insert_z_coo,
    plot_scaffolding_mds,
    slider_control,
)
from .mds_display import mds_display_new

__all__ = [
    "zero_to_near_zero",
    "extend_linear_axis_ticks",
    "keep_pretty_axis_ticks",
    "clean_linear_axes_coordinates",
    "obtain_xhat",
    "build_one_mds_display",
]


def zero_to_near_zero(z_axes: list, tol: float = 1e-12) -> list:
    """Snap tick labels within ``tol`` of zero to exactly zero."""
    out = []
    for ax in z_axes:
        if ax is None:
            out.append(ax)
            continue
        ax = np.array(ax, dtype=float, copy=True)
        if ax.shape[1] >= 3:
            ticks = ax[:, 2]
            ticks[np.abs(ticks) < tol] = 0.0
            ax[:, 2] = ticks
        out.append(ax)
    return out


def extend_linear_axis_ticks(z_axes: list, r: float, tol: float = 1e-10) -> list:
    """Port of ``extend_linear_axis_ticks()``: prolong evenly spaced ticks
    beyond the data range until they cross the bounding circle."""
    out = []
    for ax in z_axes:
        if ax is None:
            out.append(ax)
            continue
        ax = np.asarray(ax, dtype=float)
        if ax.shape[0] < 2 or ax.shape[1] < 3:
            out.append(ax)
            continue

        inside = ax[:, 0] ** 2 + ax[:, 1] ** 2 <= r**2 + tol
        if not np.all(inside):
            out.append(ax)
            continue

        step_xy = np.median(np.diff(ax[:, :2], axis=0), axis=0)
        step_tick = float(np.median(np.diff(ax[:, 2])))
        if (
            not np.all(np.isfinite(step_xy))
            or not np.isfinite(step_tick)
            or float(np.sqrt(np.sum(step_xy**2))) < tol
            or abs(step_tick) < tol
        ):
            out.append(ax)
            continue

        prepend = []
        current = ax[0].copy()
        while True:
            current = current.copy()
            current[:2] -= step_xy
            current[2] -= step_tick
            prepend.append(current)
            if float(np.sum(current[:2] ** 2)) > r**2 + tol:
                break

        append = []
        current = ax[-1].copy()
        while True:
            current = current.copy()
            current[:2] += step_xy
            current[2] += step_tick
            append.append(current)
            if float(np.sum(current[:2] ** 2)) > r**2 + tol:
                break

        out.append(np.vstack([*reversed(prepend), ax, *append]))
    return out


def keep_pretty_axis_ticks(z_axes: list, n: int = 8, tol: float = 1e-10) -> list:
    """Port of ``keep_pretty_axis_ticks()``: keep only ticks matching
    ``pretty(labels, n)``."""
    out = []
    for ax in z_axes:
        if ax is None:
            out.append(ax)
            continue
        ax = np.asarray(ax, dtype=float)
        if ax.shape[0] == 0 or ax.shape[1] < 3:
            out.append(ax)
            continue
        target = pretty((ax[:, 2].min(), ax[:, 2].max()), n=n)
        keep = np.array(
            [bool(np.any(np.abs(t - target) < tol)) for t in ax[:, 2]]
        )
        out.append(ax if not keep.any() else ax[keep, :])
    return out


def _has_two_inside(ax, radius: float, tol: float = 1e-10) -> bool:
    if ax is None:
        return False
    ax = np.asarray(ax, dtype=float)
    if ax.shape[0] < 2 or ax.shape[1] < 3:
        return False
    inside = ax[:, 0] ** 2 + ax[:, 1] ** 2 <= radius**2 + tol
    return int(inside.sum()) >= 2


def clean_linear_axes_coordinates(ez, z_axes: list | None = None) -> list:
    """Port of ``clean_linear_axes_coordinates()``: extend ticks to the
    bounding circle, thin to pretty values, and fall back to the extended
    ticks when thinning leaves fewer than two visible ticks."""
    if z_axes is None:
        z_axes = axes_coordinates(ez)
    radius = float(np.max(np.abs(np.asarray(ez.Z)))) * 1.2
    z_axes = zero_to_near_zero(z_axes)
    extended = extend_linear_axis_ticks(z_axes, r=radius)
    pretty_axes = keep_pretty_axis_ticks(extended, n=8)
    pretty_axes = extend_linear_axis_ticks(pretty_axes, r=radius)
    return [
        p if _has_two_inside(p, radius) else e
        for p, e in zip(pretty_axes, extended)
    ]


def obtain_xhat(ez, z_axes: list | None = None) -> np.ndarray:
    """Port of ``obtain_xhat()``: predicted values in raw data units.

    PCA/CVA invert the square ``Lmat``; regression and PCO biplots read the
    prediction off the calibrated axes by rotating each axis horizontal and
    interpolating tick labels at the samples' projections. (The R original
    only interpolates for ``regress``; routing PCO here too avoids R's
    double back-transformation of the already-restored data matrix.)
    """
    Z = np.asarray(ez.Z, dtype=float)
    if ez.method in ("regress", "pco") and z_axes is not None:
        p = ez.p
        n = ez.n
        xhat = np.full((n, p), np.nan)
        for i in range(p):
            ax = np.asarray(z_axes[i], dtype=float)
            m = (ax[1, 1] - ax[0, 1]) / (ax[1, 0] - ax[0, 0])
            rot = rotation_constructor(np.arctan(m))
            rot_z = Z @ rot
            rot_ax = ax[:, :2] @ rot
            xhat[:, i] = approx(rot_ax[:, 0], ax[:, 2], rot_z[:, 0])
        return xhat

    if ez.Lmat is not None and ez.Lmat.shape[0] == ez.Lmat.shape[1]:
        xhat = Z @ np.linalg.inv(ez.Lmat)[ez.e_idx, :]
    else:
        xhat = np.asarray(ez.X, dtype=float)
    if ez.scaled:
        xhat = xhat * ez.sd
    if ez.center:
        xhat = xhat + ez.means
    return xhat


def restore_raw_x(ez):
    """Port of ``scale_mds_restore_raw_x()``: un-scale/un-center ``X`` so
    hover tables show raw values."""
    ez = ez._copy()
    X = np.asarray(ez.X, dtype=float)
    if ez.scaled:
        X = X * ez.sd
    if ez.center:
        X = X + ez.means
    ez.X = X
    return ez


def build_one_mds_display(
    ez,
    group_codes,
    group_levels,
    color,
    symbol,
    x_ref,
    include_polygons: bool = False,
    dim_prefix: str = "PC",
    ax_pred: bool = True,
    vec_dis: bool = True,
    z_axes: list | None = None,
    fit_qual: str | None = None,
) -> dict:
    """Port of ``build_one_mdsDisplay()``: one full display bundle.

    Layers are added in the fixed order the vendored JS expects:
    scaffolding, polygons, observations, class means, calibrated linear
    axes, optional vector annotations, translated density axes, slider
    metadata, and the nested data object.

    Returns a dict bundle: ``{"mds", "fit_qual", "m", "shift", "Data"}``.
    """
    if ez.axes is None:
        ez = set_axes(ez)

    payl = mds_display_new()
    if fit_qual is None:
        fit_qual = fit_quality_string(ez.eigenvalues, ez.e_vects, dim_prefix)

    payl = plot_scaffolding_mds(
        payl,
        dpquality=fit_qual,
        basis=ez.e_vects,
        PC_toggle=True,
        ax_pred=ax_pred,
        TDA=True,
        vec_dis=vec_dis,
    )

    # Polygons (only valid in the primary pair's coordinate system)
    if include_polygons:
        alpha_bags = getattr(x_ref, "alpha_bags", None)
        if alpha_bags:
            payl = insert_polygon_ez(payl, alpha_bags, x_ref.alpha_bag_aes)
        conc_ellipses = getattr(x_ref, "conc_ellipses", None)
        if conc_ellipses:
            payl = insert_polygon_ez(
                payl, conc_ellipses, x_ref.conc_ellipse_aes, "Con. Ellipses"
            )

    if z_axes is None:
        z_axes = clean_linear_axes_coordinates(ez)
    xhat = obtain_xhat(ez, z_axes=z_axes)

    sample_pred = ez.sample_predictivity
    if sample_pred is None:
        sample_pred = ez.within_class_sample_predictivity

    obj = {
        "Z": ez.Z,
        "group_codes": group_codes,
        "group_levels": group_levels,
        "n": ez.n,
        "x": np.asarray(ez.X, dtype=float),
        "row_names": ez.row_names,
        "col_names": ez.col_names,
        "XHat": xhat,
        "sample_predictivity": sample_pred,
    }
    payl = insert_z_coo(payl, obj, p_ly_pch=symbol, col=color, visible=True)

    # Class means: coordinates from ez, aesthetics from x_ref
    if x_ref.class_means:
        if x_ref.means_aes is None:
            x_ref = set_means(x_ref)
        mean_symbol = pch_to_plotly(x_ref.means_aes["pch"])
        if ez.Zmeans is not None:
            zmeans = np.asarray(ez.Zmeans, dtype=float)
        else:
            zmeans = np.vstack(
                [
                    np.asarray(ez.Z)[np.asarray(group_codes) == i].mean(axis=0)
                    for i in range(len(group_levels))
                ]
            )
        payl = insert_class_means(
            payl, zmeans, mean_symbol, x_ref.means_aes["col"], group_levels
        )

    out = insert_linear_axes(payl, z_axes, ez)
    payl = out["mds"]

    if vec_dis:
        payl = insert_vector_annots(payl, ez.Vr, ez.Z, ez.col_names)

    tda_out = add_tda(
        mds=payl,
        z_axes=z_axes,
        ez=ez,
        Z=ez.Z,
        group_codes=group_codes,
        group_levels=group_levels,
        col=color,
    )

    bundle = {
        "mds": tda_out["mds"],
        "m": tda_out["m"],
        "shift": tda_out["shift"],
    }
    bundle = slider_control(bundle, n_inside=17, n_outside=4)
    bundle["fit_qual"] = fit_qual
    bundle["Data"] = BiplotData(
        {
            "sample_coordinates": np.asarray(ez.Z, dtype=float),
            "axes_coordinates": z_axes,
            "translated_axes_coordinates": tda_out["shift"],
        }
    )
    return MdsDisplay(bundle)
