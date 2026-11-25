"""Trace and annotation builders for mdsDisplays — port of
``build_secondary_biplots.R``.

Every function builds plain plotly-compatible dicts. Trace ``meta`` tags
("data", "axis", "ExpAx", "density", "FitPanel", …) are a contract with the
vendored JavaScript and must not change.
"""

from __future__ import annotations

import numpy as np

from ..geometry import (
    compute_density_inflation,
    ellipse_points,
    get_quads_axes,
    move_densities,
    move_lines,
    mvee,
    obtain_zhat,
    shorten_axes,
)
from ..ordination.fit_measures import (
    cumulative_adequacies,
    cumulative_axis_predictivities,
    marginal_axis_predictivities,
)
from ..rcompat import adjustcolor
from .hover import hovertext_generator
from .mds_display import mds_display_add_layout, mds_display_add_traces

__all__ = [
    "plot_scaffolding_mds",
    "insert_z_coo",
    "insert_class_means",
    "insert_polygon_ez",
    "add_axis_pred_traces",
    "add_axis_adeq_traces",
    "add_prop_variance_traces",
    "add_scree_traces",
    "insert_vector_annots",
    "add_tda",
    "insert_linear_axes",
    "fit_table_traces",
    "slider_control",
    "check_inside_circle",
]

_QUALITY_TITLE = "Overall quality and axis predictivities (cumulative)"


def plot_scaffolding_mds(
    mds: dict,
    dpquality,
    basis,
    PC_toggle: bool = True,
    ax_pred: bool = True,
    TDA: bool = True,
    vec_dis: bool = True,
) -> dict:
    """Port of ``plot_scaffolding_mdsDisplay()``: base layout for a payload."""
    layout = {
        "legend": {
            "tracegroupgap": 0,
            "xref": "container",
            "yref": "container",
            "x": 1,
            "y": 0.82,
            "groupclick": "toggleitem",
        },
        "xaxis": {
            "title": dpquality,
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
            "domain": [0, 1],
        },
        "yaxis": {
            "showticklabels": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
            "showgrid": False,
        },
        "xaxis2": {"domain": [0, 0.15], "zeroline": True},
        "yaxis2": {"zeroline": True, "side": "left", "position": 0},
        "xaxis3": {
            "domain": [0.65, 1],
            "zeroline": True,
            "showgrid": True,
            "anchor": "y3",
            "dtick": 1,
            "title": "Dimension of Subspace",
        },
        "yaxis3": {
            "zeroline": True,
            "anchor": "free",
            "side": "left",
            "position": 0.65,
            "showgrid": True,
            "domain": [0.15, 0.85],
            "layer": "below traces",
            "title": _QUALITY_TITLE,
        },
        "hoverlabel": {"font": {"family": "Courier New, monospace"}},
        "updatemenus": [
            {
                "y": 0.8,
                "type": "buttons",
                "x": 0,
                "pad": {"r": 0},
                "showactive": True,
                "active": -1,
                "buttons": [
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Axis Predictivity",
                        "name": "AxisStats",
                        "visible": ax_pred,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Translated Axes",
                        "name": "TransAxes",
                        "visible": TDA,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Vector Display",
                        "name": "vecload",
                        "visible": vec_dis,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Edit: Axes",
                        "name": "EditAxes",
                        "visible": False,
                        "execute": False,
                    },
                ],
            },
            {
                "type": "dropdown",
                "x": 0,
                "pad": {"r": 0},
                "visible": PC_toggle,
                "buttons": [
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "PC 1 & 2",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "histogram"],
                        "label": "PC 1 & 3",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "histogram"],
                        "label": "PC 2 & 3",
                    },
                ],
            },
        ],
    }
    return mds_display_add_layout(mds, layout)


def insert_z_coo(mds: dict, obj: dict, p_ly_pch, col, visible=True) -> dict:
    """Port of ``insert_Z_coo_mdsDisplay()``: one scatter trace per group."""
    levels = obj["group_levels"]
    codes = np.asarray(obj["group_codes"])
    Z = np.asarray(obj["Z"], dtype=float)
    n = int(obj["n"])

    traces = []
    for i, level in enumerate(levels):
        sel = np.where(codes == i)[0]
        traces.append(
            {
                "x": Z[sel, 0],
                "y": Z[sel, 1],
                "name": level,
                "type": "scatter",
                "mode": "markers",
                "hovertext": hovertext_generator(obj, i, "<br />"),
                "hoverinfo": "text+name",
                "customdata": (np.arange(1, n + 1))[sel],  # 1-based, like R
                "meta": ["data"],
                "xaxis": "x",
                "yaxis": "y",
                "visible": visible,
                "marker": {"symbol": p_ly_pch[i], "color": col[i], "opacity": 1},
                "legendgroup": "data",
                "legendgrouptitle": {"text": "<b>Data</b>"},
            }
        )
    return mds_display_add_traces(mds, traces)


def insert_class_means(mds: dict, Z, symbol, color, names) -> dict:
    """Port of ``insert_class_means_mdsDisplay()``."""
    Z = np.asarray(Z, dtype=float)
    traces = []
    for i in range(Z.shape[0]):
        traces.append(
            {
                "x": [float(Z[i, 0])],
                "y": [float(Z[i, 1])],
                "name": names[i] if names is not None else f"ClassMean_{i + 1}",
                "type": "scatter",
                "mode": "markers",
                "hovertext": "Class Mean",
                "hoverinfo": "text+name",
                "customdata": i,  # 0-based id, kept from R
                "meta": ["ClassMean"],
                "xaxis": "x",
                "yaxis": "y",
                "visible": True,
                "showlegend": False,
                "marker": {"symbol": symbol[i], "color": color[i]},
                "legendgroup": "ClassMean",
            }
        )
    return mds_display_add_traces(mds, traces)


def insert_polygon_ez(mds: dict, coors: dict, aes: dict, leg_group="Alpha Bags") -> dict:
    """Port of ``insert_polygon_EZ_mdsDisplay()``: alpha bags / ellipses."""
    traces = []
    for i, (name, xy) in enumerate(coors.items()):
        xy = np.asarray(xy, dtype=float)
        if leg_group != "Alpha Bags":
            center, A = mvee(xy)
            xy = ellipse_points(center, A, n_out=101)
        xy = np.vstack([xy, xy[:1]])  # close the polygon
        traces.append(
            {
                "x": xy[:, 0],
                "y": xy[:, 1],
                "mode": "lines",
                "type": "scatter",
                "line": {"color": aes["col"][i], "width": aes["lwd"][i]},
                "fill": "toself",
                "fillcolor": adjustcolor(aes["col"][i], aes["opacity"][i]),
                "legendrank": 2000,
                "name": name,
                "legendgroup": leg_group,
                "legendgrouptitle": {"text": f"<b>{leg_group}</b>"},
                "visible": True,
                "meta": ["polygon"],
                "xaxis": "x",
                "yaxis": "y",
            }
        )
    return mds_display_add_traces(mds, traces)


def add_axis_pred_traces(ez) -> list:
    """Port of ``add_axis_pred_mdsDisplay()``: cumulative predictivity lines."""
    pred = cumulative_axis_predictivities(ez)
    names = [*ez.col_names, "Weighted mean = Quality"]
    p = ez.p
    traces = []
    for i in range(pred.shape[0]):
        last = i == pred.shape[0] - 1
        traces.append(
            {
                "x": list(range(1, p + 1)),
                "y": pred[i, :],
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"dash": "solid" if last else "dashdot", "width": 3 if last else 2},
                "xaxis": "x3",
                "yaxis": "y3",
                "hoverinfo": "skip",
                "showlegend": True,
                "name": names[i],
                "visible": True,
                "meta": ["FitPanel", "Cum. Predictivity"],
                "legendgroup": "AxPred",
                "legendgrouptitle": {"text": "<b> Axis Predictivity <b>"},
            }
        )
    return traces


def add_axis_adeq_traces(ez) -> list:
    """Port of ``add_axis_adeq_mdsDisplay()``: cumulative adequacy lines."""
    adeq = cumulative_adequacies(ez)
    p = ez.p
    traces = []
    for i in range(adeq.shape[0]):
        traces.append(
            {
                "x": list(range(1, p + 1)),
                "y": adeq[i, :],
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"dash": "dashdot", "width": 2},
                "xaxis": "x3",
                "yaxis": "y3",
                "hoverinfo": "skip",
                "showlegend": True,
                "name": ez.col_names[i],
                "visible": True,
                "meta": ["FitPanel", "Cum. Adequacy"],
                "legendgroup": "AxPred",
                "legendgrouptitle": {"text": "<b> Axis Adequacy <b>"},
            }
        )
    return traces


def add_prop_variance_traces(ez, axis="x3", yaxis="y3", line_color="black") -> list:
    """Port of ``add_prop_variance_mdsDisplay()``: stacked variance bars."""
    eig = np.asarray(ez.eigenvalues, dtype=float)
    pcs = list(range(1, eig.size + 1))
    ind = eig / eig.sum()
    cum = np.cumsum(ind)

    preds = cumulative_axis_predictivities(ez)[: ez.p, :]
    V = ez.Lmat
    v_ii = np.diag(V @ np.diag(eig) @ V.T)
    w = v_ii / eig.sum()

    meta_key = "Variance Explained"
    traces = []
    for i in range(ez.p):
        traces.append(
            {
                "type": "bar",
                "x": pcs,
                "y": preds[i, :] * w[i],
                "name": ez.col_names[i],
                "xaxis": axis,
                "yaxis": yaxis,
                "meta": ["FitPanel", ez.col_names[i]],
                "textposition": "outside",
                "hoverinfo": "text",
                "visible": True,
                "hovertext": [
                    f"PC {pc}<br>Cumulative: {c:.2f}%" for pc, c in zip(pcs, cum)
                ],
                "legendgroup": "VarExplained",
                "legendgrouptitle": {"text": "<b> Variance Contribution <b>"},
                "marker": {"opacity": 0.7},
            }
        )
    traces.append(
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": pcs,
            "y": ind,
            "name": "% Individual",
            "xaxis": axis,
            "yaxis": yaxis,
            "meta": ["FitPanel", meta_key],
            "hoverinfo": "text",
            "visible": True,
            "text": [
                f"PC {pc}<br>Individual: {v:.2f}%" for pc, v in zip(pcs, ind)
            ],
            "line": {"color": line_color},
            "marker": {"color": line_color},
        }
    )
    return traces


def add_scree_traces(ez, axis="x3", yaxis="y3", line_color="black") -> list:
    """Port of ``add_scree_mdsDisplay()``."""
    eig = np.asarray(ez.eigenvalues, dtype=float)
    pcs = list(range(1, eig.size + 1))
    return [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": pcs,
            "y": eig,
            "name": "Eigenvalue",
            "xaxis": axis,
            "yaxis": yaxis,
            "meta": ["FitPanel", "Scree Plot"],
            "visible": True,
            "hoverinfo": "text",
            "text": [f"PC {pc}<br>Eigenvalue: {e:.4f}" for pc, e in zip(pcs, eig)],
            "line": {"color": line_color},
            "marker": {"color": line_color},
        }
    ]


def insert_vector_annots(mds: dict, V, Z, var_names) -> dict:
    """Port of ``insert_vector_annots_mdsDisplay()``: loading arrows."""
    V = np.asarray(V, dtype=float)
    Z = np.asarray(Z, dtype=float)
    alpha = 1.0
    if Z.size:
        norms = np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2)
        peak = float(np.max(norms))
        if np.isfinite(peak):
            alpha = peak

    annotations = []
    for i in range(V.shape[0]):
        annotations.append(
            {
                "x": 0,
                "y": 0,
                "ax": alpha * float(V[i, 0]),
                "ay": alpha * float(V[i, 1]),
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": var_names[i],
                "showarrow": True,
                "arrowside": "start",
                "visible": False,
                "meta": ["vecload"],
            }
        )
    return mds_display_add_layout(mds, {"annotations": annotations})


def check_inside_circle(ticks: list, r: float) -> list:
    """Port of ``check_inside_circle()``: drop ticks outside the circle."""
    out = []
    for ax in ticks:
        ax = np.asarray(ax, dtype=float)
        inside = ax[:, 0] ** 2 + ax[:, 1] ** 2 <= r**2
        out.append(ax[inside, :])
    return out


def insert_linear_axes(mds: dict, z_axes: list, ez) -> dict:
    """Port of ``insert_linear_axes_mdsDisplay()``: calibrated axis lines,
    tick annotations, axis-name labels and the bounding circle.
    """
    p = ez.p
    Z = np.asarray(ez.Z, dtype=float)
    radius = float(np.max(np.abs(Z))) * 1.2
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    z_axes = check_inside_circle(z_axes, radius)
    var_names = ez.col_names
    ax_aes = ez.axes

    grads = np.zeros(p)
    traces: list = []
    annotations: list = []

    for i in range(p):
        ax = np.asarray(z_axes[i], dtype=float)
        ax_name = f"<b>{var_names[i]}</b>"

        endp = ax[int(np.argmax(ax[:, 2])), :2]
        pos = "right"
        m = (ax[1, 1] - ax[0, 1]) / (ax[1, 0] - ax[0, 0])
        grads[i] = m
        angle = np.arctan(m)
        if endp[0] < 0:
            pos = "left"
            angle -= np.pi

        x_line = [radius * np.cos(np.arctan(m)), radius * np.cos(np.arctan(m) - np.pi)]
        y_line = [radius * np.sin(np.arctan(m)), radius * np.sin(np.arctan(m) - np.pi)]
        zhats = obtain_zhat(np.column_stack([x_line, y_line]), ax)

        trace = {
            "x": x_line,
            "y": y_line,
            "type": "scatter",
            "mode": "lines",
            "line": {"color": ax_aes["col"][i], "width": 1, "simplify": False},
            "name": var_names[i],
            "legendgroup": f"Ax{i + 1}",
            "meta": ["axis"],
            "xaxis": "x",
            "yaxis": "y",
            "customdata": zhats,
            "visible": True,
            "hoverinfo": "name",
        }
        if i == 0:
            trace["legendgrouptitle"] = {"text": "<b>Axes</b>"}
        traces.append(trace)

        ang_deg = float(-np.arctan(m) * 180 / np.pi)
        yshift = float(-12 * np.cos(np.arctan(m)))
        xshift = float(12 * np.sin(np.arctan(m)))

        for k in range(ax.shape[0]):
            annotations.append(
                {
                    "x": float(ax[k, 0]),
                    "y": float(ax[k, 1]),
                    "text": f"{ax[k, 2]:g}",
                    "showarrow": False,
                    "textangle": ang_deg,
                    "visible": True,
                    "yshift": yshift,
                    "xshift": xshift,
                    "meta": ["Ax"],
                    "xref": "x",
                    "yref": "y",
                    "customdata": i + 1,
                    "font": {"size": 10, "color": ax_aes["tick_label_col"][i]},
                }
            )
            annotations.append(
                {
                    "x": float(ax[k, 0]),
                    "y": float(ax[k, 1]),
                    "text": "&#124;",
                    "showarrow": False,
                    "textangle": ang_deg,
                    "visible": True,
                    "meta": ["Ax"],
                    "xref": "x",
                    "yref": "y",
                    "customdata": i + 1,
                    "font": {"size": 8, "color": ax_aes["tick_col"][i]},
                }
            )

        traces.append(
            {
                "x": [radius * np.cos(angle)],
                "y": [radius * np.sin(angle)],
                "text": ax_name,
                "type": "scatter",
                "mode": "text",
                "textposition": pos,
                "legendgroup": f"Ax{i + 1}",
                "showlegend": False,
                "textfont": {"size": 12, "color": "gray"},
                "meta": "axis",
                "xaxis": "x",
                "yaxis": "y",
                "visible": True,
            }
        )

    traces.append(
        {
            "x": circle[:, 0],
            "y": circle[:, 1],
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "green", "width": 0.6},
            "name": "OuterCircle",
            "showlegend": False,
            "meta": ["OuterCircle"],
            "xaxis": "x",
            "yaxis": "y",
            "visible": True,
            "hoverinfo": "none",
        }
    )

    mds = mds_display_add_traces(mds, traces)
    mds = mds_display_add_layout(mds, {"annotations": annotations})
    return {"mds": mds, "grads": grads, "radius": radius}


def add_tda(mds: dict, z_axes: list, ez, Z, group_codes, group_levels, col, inflate=1.0) -> dict:
    """Port of ``add_TDA_mdsDisplay()``: translated density axes.

    Shortens each calibrated axis to the data's bounding ellipse, translates
    it outward, and superimposes per-group kernel densities. All TDA traces
    start hidden; the vendored JS toggles them.
    """
    Z = np.asarray(Z, dtype=float)
    group = np.asarray([group_levels[c] for c in group_codes])

    r1 = np.ptp(np.asarray(ez.Z)[:, 0])
    r2 = np.ptp(np.asarray(ez.Z)[:, 1])
    length = float(np.sqrt(r1**2 + r2**2))
    dist = length / 8

    center, A = mvee(Z)
    elipcoords = ellipse_points(center, A, n_out=101)

    quads = get_quads_axes(z_axes)
    m = np.array([np.asarray(ax)[0, 1] / np.asarray(ax)[0, 0] for ax in z_axes])
    p = m.size

    endpoints = shorten_axes(z_axes, elipcoords)
    shift = move_lines(
        elipcoords, m, quads, dist, endpoints, swop=False, cols=ez.col_names
    )

    inflation = compute_density_inflation(
        Z=Z,
        m=m,
        endpoints=shift["ends"],
        group=group,
        target_height=dist * inflate,
    )
    dens_coors = move_densities(
        Z=Z,
        m=m,
        endpoints=shift["ends"],
        dist=shift["ShiftDist"],
        dinflation=inflation,
        group=group,
    )

    traces: list = []
    annotations: list = []
    var_names = ez.col_names
    ax_aes = ez.axes
    visible_axes = False

    for i in range(p):
        ends = np.asarray(shift["ends"][i], dtype=float)
        index2 = int(np.argmax(ends[:, 2]))
        lab2 = "&#11166;" if quads[i] in (1, 4) else "&#11164;"

        trace = {
            "x": ends[:, 0],
            "y": ends[:, 1],
            "type": "scatter",
            "mode": "lines",
            "line": {"color": ax_aes["col"][i], "width": 1, "simplify": False},
            "name": var_names[i],
            "legendgroup": f"ExpAx{i + 1}",
            "meta": "ExpAx",
            "xaxis": "x",
            "yaxis": "y",
            "customdata": ends[:, 2],
            "hoverinfo": "name",
            "visible": visible_axes,
        }
        if i == 0:
            trace["legendgrouptitle"] = {"text": "<b>Axes</b>"}
        traces.append(trace)

        ang = float(np.arctan(m[i]))
        ang_deg = -ang * 180 / np.pi
        xs1, ys1 = 12 * np.sin(ang), -12 * np.cos(ang)
        xs2, ys2 = 22 * np.sin(ang), -22 * np.cos(ang)

        for k in range(ends.shape[0]):
            annotations.append(
                {
                    "x": float(ends[k, 0]),
                    "y": float(ends[k, 1]),
                    "text": f"{ends[k, 2]:g}",
                    "showarrow": False,
                    "textangle": ang_deg,
                    "visible": visible_axes,
                    "yshift": ys1,
                    "xshift": xs1,
                    "meta": "ExpAx",
                    "xref": "x",
                    "yref": "y",
                    "customdata": i + 1,
                    "font": {"size": 10, "color": ax_aes["tick_label_col"][i]},
                }
            )
            annotations.append(
                {
                    "x": float(ends[k, 0]),
                    "y": float(ends[k, 1]),
                    "text": "&#124;",
                    "showarrow": False,
                    "textangle": ang_deg,
                    "visible": visible_axes,
                    "meta": "ExpAx",
                    "xref": "x",
                    "yref": "y",
                    "customdata": i + 1,
                    "font": {"size": 8, "color": ax_aes["tick_col"][i]},
                }
            )

        annotations.append(
            {
                "x": float(np.mean(ends[:, 0])),
                "y": float(np.mean(ends[:, 1])),
                "text": f"<b>{var_names[i]}</b>",
                "showarrow": False,
                "textangle": ang_deg,
                "visible": visible_axes,
                "yshift": ys2,
                "xshift": xs2,
                "meta": "ExpAx",
                "xref": "x",
                "yref": "y",
                "customdata": i + 1,
                "font": {"size": 12, "color": "gray"},
            }
        )
        annotations.append(
            {
                "x": float(ends[index2, 0]),
                "y": float(ends[index2, 1]),
                "text": lab2,
                "showarrow": False,
                "textangle": ang_deg,
                "visible": visible_axes,
                "meta": "ExpAx",
                "xref": "x",
                "yref": "y",
                "customdata": i + 1,
                "font": {"size": 18, "color": ax_aes["tick_label_col"][i]},
            }
        )

    seen_order = list(dict.fromkeys(group.tolist()))
    for gi, dens in enumerate(dens_coors):
        dens = np.asarray(dens, dtype=float)
        gname = seen_order[gi]
        gidx = group_levels.index(gname) if gname in group_levels else gi

        traces.append(
            {
                "x": [0],
                "y": [0],
                "type": "scatter",
                "mode": "lines",
                "line": {"dash": "dot", "color": col[gidx], "width": 0.95},
                "legendgroup": gname,
                "showlegend": True,
                "name": gname,
                "meta": "density",
                "xaxis": "x",
                "yaxis": "y",
                "hoverinfo": "skip",
                "customdata": ["legendentry"],
                "visible": False,
            }
        )
        for j in range(p):
            traces.append(
                {
                    "x": dens[:, 2 * j],
                    "y": dens[:, 2 * j + 1],
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"dash": "dot", "color": col[gidx], "width": 0.95},
                    "legendgroup": gname,
                    "showlegend": False,
                    "name": gname,
                    "meta": "density",
                    "xaxis": "x",
                    "yaxis": "y",
                    "hoverinfo": "skip",
                    "customdata": [f"ExpAx{j + 1}"],
                    "visible": visible_axes,
                }
            )

    mds = mds_display_add_traces(mds, traces)
    mds = mds_display_add_layout(mds, {"annotations": annotations})
    return {"mds": mds, "m": m, "shift": shift}


def fit_table_traces(
    ez, domain_x=(0.5, 1), domain_y=(0.15, 0.85), visible=True
) -> list:
    """Port of ``add_table_mdsDisplay()``: adequacy/predictivity table."""
    adequacy = np.round(np.sum(np.asarray(ez.Vr) ** 2, axis=1), 4)
    predictivity = np.round(marginal_axis_predictivities(ez), 4)
    return [
        {
            "type": "table",
            "domain": {"x": list(domain_x), "y": list(domain_y)},
            "header": {
                "values": ["Variable", "Adequacy", "Predictivity"],
                "align": "left",
            },
            "cells": {
                "values": [list(ez.col_names), adequacy, predictivity],
                "align": "left",
            },
            "meta": ["FitPanel", "Summary Table"],
            "showlegend": False,
            "visible": visible,
        }
    ]


def slider_control(bundle: dict, n_inside: int = 17, n_outside: int = 4) -> dict:
    """Port of ``slider_control_mdsDisplay()``: initial slider positions."""
    m = np.asarray(bundle["m"], dtype=float)
    ends = bundle["shift"]["ends"]
    dist = np.zeros(m.size)
    for i in range(m.size):
        e = np.asarray(ends[i], dtype=float)
        b = e[0, 1] - m[i] * e[0, 0]
        x_cross = -b / (1 / m[i] + m[i])
        y_cross = -1 / m[i] * x_cross
        dist[i] = np.sign(x_cross) * np.sqrt(x_cross**2 + y_cross**2)

    radius = float(np.max(np.abs(dist)))
    indiv_pos = dist / radius
    pos_shifted = np.round(indiv_pos * (n_inside - 1) / 2, 0)
    actual_pos = pos_shifted + (n_inside - 1) / 2 + 1 + n_outside / 2 - 1
    step_size = 2 * radius / n_inside

    bundle["mds"]["config"]["slider_info"] = {
        "slider_pos": actual_pos,
        "step_size": step_size,
    }
    return bundle
