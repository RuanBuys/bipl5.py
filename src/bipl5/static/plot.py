"""Static, publication-ready biplots with matplotlib — port of
``wrap_bipl5_gg.R`` (``wrap_bipl5_gg()`` / ``build_bipl5_gg()``).

:func:`plot_static` is the static counterpart of
:meth:`bipl5.Biplot.plot`. It accepts either a :class:`~bipl5.Biplot`
(``bp.plot_static()``) or a raw ordination object from
:mod:`bipl5.ordination` (``ez.plot_static()``, the analogue of calling
``wrap_bipl5_gg()`` on a biplotEZ object), and returns a
:class:`StaticBiplot` wrapping an ordinary matplotlib figure.

Layers are drawn in biplotEZ's order — calibrated axes, then alpha bags and
concentration ellipses, then samples, then class means — at the z-orders in
:data:`ZORDER`. Anything drawn below ``ZORDER["axes"]`` (heatmaps,
prediction regions, densities: ``ax.imshow(..., zorder=0.5)``) sits beneath
the whole biplot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import AutoMinorLocator

from ..display.build_one import zero_to_near_zero
from ..ordination import axes as set_axes
from ..ordination import axes_coordinates
from ..ordination import means as set_means
from ..ordination import samples as set_samples
from ..ordination.base import EZBiplot
from ..ordination.fit_measures import fit_quality_string, regression_fit_quality
from ..symbols import pch_to_plotly
from .calibrated_axis import AxisStyle, CalibratedAxes, CalibratedAxis
from .helpers import PT_PER_MM, r_linetype, resolve_axis_aes
from .markers import marker_kwargs

__all__ = ["plot_static", "StaticBiplot", "ZORDER", "THEMES"]

#: Drawing order of the biplot layers (biplotEZ's order). Put heatmaps and
#: other backgrounds below ``ZORDER["axes"]``.
ZORDER = {
    "axes": 1.0,
    "polygons": 1.5,
    "samples": 2.0,
    "sample_labels": 2.5,
    "means": 3.0,
}

_PREFIX = {"pca": "PC", "cva": "CV"}

# ggplot2 theme_grey/theme_bw text sizes for base_size = 11 (points)
_BASE = 11.0
_SMALL = 0.8 * _BASE


# ─────────────────────────────────────────────────────────────────────────────
# Context: everything the renderer needs, from either kind of input
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Context:
    ez: EZBiplot  # ordination of the drawn pair, X centred/scaled
    Z: np.ndarray
    z_axes: list
    curved: bool
    groups: list[dict]
    legend: list[dict]
    means: dict | None
    polygons: list[dict]
    dim_prefix: str
    caption: str | None
    title: str | None
    description: str
    extra: dict = field(default_factory=dict)


def _pick(values, w: int, j: int, g: int):
    values = list(values)
    return values[w] if len(values) == g else values[j % len(values)]


def _polygon_records(ez) -> list[dict]:
    """Port of ``polygon_frame()``: alpha bags and concentration ellipses
    drawn exactly as stored on the ordination object, with no refitting."""
    out = []
    for attr, aes_attr, kind in (
        ("alpha_bags", "alpha_bag_aes", "alpha_bags"),
        ("conc_ellipses", "conc_ellipse_aes", "conc_ellipses"),
    ):
        coords = getattr(ez, attr, None)
        if not coords:
            continue
        aes = getattr(ez, aes_attr, None) or {}
        items = coords.items() if isinstance(coords, dict) else enumerate(coords)
        for i, (name, xy) in enumerate(items):
            xy = np.asarray(xy, dtype=float)
            if xy.ndim != 2 or xy.shape[0] < 3:
                continue

            def get(key, default, i=i):
                vals = aes.get(key)
                if vals is None:
                    return default
                vals = list(vals) if not isinstance(vals, str) else [vals]
                return vals[i % len(vals)]

            out.append(
                {
                    "kind": kind,
                    "name": str(name),
                    "xy": xy[:, :2],
                    "colour": get("col", "black"),
                    "linewidth": float(get("lwd", 1)) * 0.75,
                    "linetype": get("lty", 1),
                    "alpha": float(get("opacity", 0.25)),
                }
            )
    return out


def _context_from_ez(ez: EZBiplot, z_axes=None, include_polygons: bool = True) -> _Context:
    """Port of ``gg_biplot_context()`` plus the sample, mean and polygon
    frame builders, for an ordination object."""
    if not isinstance(ez, EZBiplot):
        raise TypeError("expected an ordination object from bipl5.ordination.")
    if ez.Z is None or ez.method is None:
        raise ValueError(
            "plot_static() needs a fitted biplot. Apply pca(), cva(), "
            "regress() or pco() first."
        )
    Z = np.asarray(ez.Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 2 or ez.dim_biplot not in (None, 2):
        raise ValueError("plot_static() only accepts biplots of two dimensions.")

    if ez.samples is None:
        ez = set_samples(ez)
    if ez.axes is None:
        ez = set_axes(ez)
    if ez.method == "cva" and ez.class_means and ez.means_aes is None:
        ez = set_means(ez)

    curved = ez.PCOaxes == "splines"
    if z_axes is None:
        # axes_coordinates() must see the centred/scaled X (ez is unrestored)
        z_axes = axes_coordinates(ez)
        if curved:
            z_axes = zero_to_near_zero(z_axes)

    # sample groups (port of sample_point_frame)
    s = ez.samples
    codes, names, g = np.asarray(ez.group), list(ez.g_names), ez.g
    symbols = s.get("plotly_symbol") or s["pch"]
    groups = []
    for j, w in enumerate(s["which"]):
        idx = np.where(codes == w)[0]
        if idx.size:
            groups.append(
                {
                    "label": names[w],
                    "index": idx,
                    "color": _pick(s["col"], w, j, g),
                    "symbol": _pick(symbols, w, j, g),
                }
            )
    legend = [
        {
            "title": None,
            "kind": "group",
            "entries": [(gr["label"], gr["color"], gr["symbol"]) for gr in groups],
        }
    ]

    # class means (port of class_mean_frame)
    means = None
    if ez.class_means and ez.Zmeans is not None:
        m_aes = ez.means_aes or set_means(ez).means_aes
        Zm = np.asarray(ez.Zmeans, dtype=float)
        which = [w for w in m_aes.get("which", range(Zm.shape[0])) if w < Zm.shape[0]]
        Zm = Zm[which]
        means = {
            "x": Zm[:, 0],
            "y": Zm[:, 1],
            "labels": [names[w] if w < len(names) else str(w + 1) for w in which],
            "colors": list(m_aes["col"]),
            "symbols": list(m_aes.get("plotly_symbol") or pch_to_plotly(m_aes["pch"])),
        }

    prefix = _PREFIX.get(ez.method, "Dim")
    if ez.method in ("pca", "cva"):
        caption = fit_quality_string(ez.eigenvalues, ez.e_vects, dim_prefix=prefix)
    elif ez.method == "regress":
        caption = regression_fit_quality(ez.X, ez.Z, dim_prefix="Dim")
    else:
        caption = None

    return _Context(
        ez=ez,
        Z=Z,
        z_axes=list(z_axes),
        curved=curved,
        groups=groups,
        legend=legend,
        means=means,
        polygons=_polygon_records(ez) if include_polygons else [],
        dim_prefix=prefix,
        caption=caption or None,
        title=ez.title,
        description=f"{ez.method.upper()} biplot",
    )


def _resolve_display(bp, display) -> str:
    pc_info = bp.meta["pc_info"]
    available = [k for k in pc_info if k in bp.displays]
    if display is None:
        return available[0]
    if isinstance(display, str):
        if display in available:
            return display
        for k in available:
            if pc_info[k]["label"] == display:
                return k
    else:
        try:
            pcs = tuple(sorted(int(v) for v in display))
        except (TypeError, ValueError):
            pcs = None
        for k in available:
            if pcs is not None and tuple(pc_info[k]["pcs"]) == pcs:
                return k
    raise ValueError(
        f"{display!r} is not a display of this biplot. Available: "
        + ", ".join(f"{k} ({pc_info[k]['label']})" for k in available)
    )


def _processed(ez: EZBiplot) -> EZBiplot:
    """Undo ``restore_raw_x()``: the stored Biplot object keeps raw X for
    hover tables, but calibration needs the centred/scaled matrix."""
    out = ez._copy()
    out.X = (np.asarray(ez.X, dtype=float) - ez.means) / ez.sd
    return out


def _biplot_means(bp, ez_raw, ez, Z) -> dict | None:
    """Class means exactly as the interactive display shows them."""
    from ..format_samples import resolve_mean_aesthetics

    if not ez_raw.class_means:
        return None
    spec = resolve_mean_aesthetics(bp)
    if spec is not None:
        # format_samples() regroups the means; they follow the samples
        pts = [Z[spec["labels"] == lev].mean(axis=0) for lev in spec["levels"]]
        pts = np.vstack(pts)
        return {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "labels": spec["levels"],
            "colors": spec["colors"],
            "symbols": spec["symbols"],
        }

    m_aes = ez_raw.means_aes or set_means(ez_raw).means_aes
    if bp.biplot_type == "cva":
        codes, levels = np.asarray(ez_raw.group), list(ez_raw.g_names)
    else:
        codes, levels = bp.meta["group"]
        codes = np.asarray(codes)
    if ez.Zmeans is not None and np.asarray(ez.Zmeans).shape[0] == len(levels):
        Zm = np.asarray(ez.Zmeans, dtype=float)
    else:
        Zm = np.vstack([Z[codes == i].mean(axis=0) for i in range(len(levels))])
    which = [w for w in m_aes.get("which", range(Zm.shape[0])) if w < Zm.shape[0]]
    return {
        "x": Zm[which, 0],
        "y": Zm[which, 1],
        "labels": [levels[w] for w in which],
        "colors": list(m_aes["col"]),
        "symbols": list(m_aes.get("plotly_symbol") or pch_to_plotly(m_aes["pch"])),
    }


def _context_from_biplot(bp, display=None) -> _Context:
    from ..format_samples import resolve_sample_aesthetics

    name = _resolve_display(bp, display)
    info = bp.meta["pc_info"][name]
    pcs = tuple(info["pcs"])
    ez_raw = bp.meta["x"]

    if bp.biplot_type in ("pca", "cva") and tuple(sorted(ez_raw.e_vects)) != pcs:
        # an appended pair: re-run the ordination for it, as
        # append_mds_display() did, keeping the object's axis aesthetics
        ez = bp._refit_pair(pcs)
        ez.axes = ez_raw.axes
        primary = False
    else:
        ez = _processed(ez_raw)
        primary = True

    spline_axes = bp.meta.get("spline_axes") if ez.PCOaxes == "splines" else None
    ctx = _context_from_ez(ez, z_axes=spline_axes, include_polygons=primary)

    sample_aes = resolve_sample_aesthetics(bp)
    ctx.groups = sample_aes["groups"]
    ctx.legend = sample_aes["legend"]
    ctx.means = _biplot_means(bp, ez_raw, ctx.ez, ctx.Z)
    ctx.caption = bp.displays[name].get("fit_qual") or None
    ctx.title = ez_raw.title
    ctx.description = f"{bp.biplot_type.upper()} biplot, {info['label']}"
    ctx.extra["display"] = name
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Scaffolding: window, labels, themes
# ─────────────────────────────────────────────────────────────────────────────


def _expand(lo: float, hi: float, mult: float = 0.05) -> tuple[float, float]:
    span = hi - lo
    if span <= 0:
        return lo - 0.5, hi + 0.5
    return lo - mult * span, hi + mult * span


def _window(ctx, xlim, ylim, expand, square, exp_factor, show_means, show_polygons):
    """The plotting window: ggplot2's ``coord_equal(xlim, ylim, expand)``
    over the drawn layers (R bipl5), or biplotEZ's ``range(Z * exp.factor)``
    when ``exp_factor`` is given. Calibrated axes never widen it."""
    if exp_factor is not None:
        pts = ctx.Z * float(exp_factor)
        expand = False
    else:
        parts = [ctx.Z[g["index"]] for g in ctx.groups] or [ctx.Z]
        if show_means and ctx.means is not None and len(ctx.means["x"]):
            parts.append(np.column_stack([ctx.means["x"], ctx.means["y"]]))
        if show_polygons:
            parts.extend(p["xy"] for p in ctx.polygons)
        pts = np.vstack(parts)
    xr = (float(np.min(pts[:, 0])), float(np.max(pts[:, 0])))
    yr = (float(np.min(pts[:, 1])), float(np.max(pts[:, 1])))
    if xlim is not None:
        xr = (float(min(xlim)), float(max(xlim)))
    if ylim is not None:
        yr = (float(min(ylim)), float(max(ylim)))
    if expand:
        xr, yr = _expand(*xr), _expand(*yr)
    if square:
        # biplotEZ: pad the narrower range so the panel renders square
        half = max(xr[1] - xr[0], yr[1] - yr[0]) / 2
        cx, cy = sum(xr) / 2, sum(yr) / 2
        xr, yr = (cx - half, cx + half), (cy - half, cy + half)
    return xr, yr


def _dim_labels(ez, prefix: str, percentages: bool) -> list[str]:
    """Port of ``gg_dim_labels()``: e.g. ``["PC1 (73.0%)", "PC2 (22.9%)"]``."""
    basis = list(ez.e_vects) if ez.e_vects is not None else [1, 2]
    basis = (basis + [1, 2])[:2] if len(basis) < 2 else basis[:2]
    titles = [f"{prefix}{b}" for b in basis]
    eig = ez.eigenvalues
    if percentages and eig is not None:
        eig = np.asarray(eig, dtype=float)
        total = float(np.sum(eig))
        if total > 0 and all(1 <= b <= eig.size for b in basis):
            pct = [round(100 * float(eig[b - 1]) / total, 1) for b in basis]
            if all(np.isfinite(pct)):
                # R's format() gives both values a common number of decimals
                fmt = "{:.0f}" if all(p == round(p) for p in pct) else "{:.1f}"
                titles = [f"{t} ({fmt.format(p)}%)" for t, p in zip(titles, pct)]
    return titles


def _theme_bw(ax) -> dict:
    """ggplot2's ``theme_bw()``: grey20 border, grey92 grid, grey30 text."""
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#333333")
        sp.set_linewidth(0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.8)
    ax.grid(True, which="minor", color="#EBEBEB", linewidth=0.4)
    ax.tick_params(
        which="major", length=2.75, width=0.8, color="#333333",
        labelsize=_SMALL, labelcolor="#4D4D4D",
    )
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)
    return {"title_loc": "left", "axis_titles": True}


def _theme_minimal(ax) -> dict:
    """ggplot2's ``theme_minimal()``: grid only, no border or ticks."""
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.8)
    ax.grid(True, which="minor", color="#EBEBEB", linewidth=0.4)
    ax.tick_params(which="both", length=0, labelsize=_SMALL, labelcolor="#4D4D4D")
    ax.set_axisbelow(True)
    return {"title_loc": "left", "axis_titles": True}


def _theme_classic(ax) -> dict:
    """ggplot2's ``theme_classic()``: x and y axis lines, no grid."""
    ax.set_facecolor("white")
    for name, sp in ax.spines.items():
        sp.set_visible(name in ("left", "bottom"))
        sp.set_color("black")
        sp.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(
        which="major", length=2.75, width=0.8, color="black",
        labelsize=_SMALL, labelcolor="#4D4D4D",
    )
    return {"title_loc": "left", "axis_titles": True}


def _theme_biplotez(ax) -> dict:
    """biplotEZ's ``.theme_biplot()``: a plain framed panel — no scaffolding
    axes, ticks or grid — with a centred title."""
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(0.8)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return {"title_loc": "center", "axis_titles": False}


#: Built-in themes for :func:`plot_static`.
THEMES: dict[str, Callable] = {
    "bw": _theme_bw,
    "minimal": _theme_minimal,
    "classic": _theme_classic,
    "biplotEZ": _theme_biplotez,
}


def _apply_theme(ax, theme) -> dict:
    if theme is None:
        return {"title_loc": "left", "axis_titles": True}
    if callable(theme):
        out = theme(ax)
        return {"title_loc": "left", "axis_titles": True, **(out or {})}
    if theme not in THEMES:
        raise ValueError("theme must be one of: " + ", ".join(THEMES) + " (or a callable).")
    return THEMES[theme](ax)


def _xaxis_bottom(ax):
    """Bounding box below which the caption goes: the x-axis decorations
    when there are any, otherwise the panel."""

    def bbox(renderer):
        bb = ax.xaxis.get_tightbbox(renderer)
        if bb is None or bb.height <= 0:
            return ax.get_window_extent(renderer)
        return bb

    return bbox


def _add_titles(ax, title, subtitle, caption, loc):
    x = 0.0 if loc == "left" else 0.5
    out = []
    offset = 5.5
    if subtitle:
        out.append(
            ax.annotate(
                subtitle, xy=(x, 1), xycoords="axes fraction",
                xytext=(0, offset), textcoords="offset points",
                ha=loc, va="bottom", fontsize=_BASE,
            )
        )
        offset += _BASE * 1.4
    if title:
        out.append(
            ax.annotate(
                title, xy=(x, 1), xycoords="axes fraction",
                xytext=(0, offset), textcoords="offset points",
                ha=loc, va="bottom", fontsize=1.2 * _BASE,
            )
        )
    if caption:
        out.append(
            ax.annotate(
                caption, xy=(1, 0), xycoords=("axes fraction", _xaxis_bottom(ax)),
                xytext=(0, -5.5), textcoords="offset points",
                ha="right", va="top", fontsize=_SMALL,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The result object
# ─────────────────────────────────────────────────────────────────────────────


class StaticBiplot:
    """A rendered static biplot: a matplotlib figure plus handles to its
    layers, returned by :func:`plot_static`.

    Attributes
    ----------
    figure, ax:
        The matplotlib :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes`. Customise them freely; the
        calibrated axes re-lay themselves out on every draw, so changing
        ``ax.set_xlim()`` / ``ax.set_ylim()`` keeps the ticks running to the
        edge of the panel.
    calibrated_axes:
        The :class:`~bipl5.static.CalibratedAxes` artist.
    samples, class_means, polygons, sample_labels, legends, titles:
        Lists of the matplotlib artists of each layer.
    window:
        The ``(xlim, ylim)`` the plot was drawn with.
    """

    def __init__(self, figure, ax, calibrated_axes, layers: dict, context, window):
        self.figure = figure
        self.ax = ax
        self.calibrated_axes = calibrated_axes
        self.samples = layers.get("samples", [])
        self.class_means = layers.get("class_means", [])
        self.polygons = layers.get("polygons", [])
        self.sample_labels = layers.get("sample_labels", [])
        self.legends = layers.get("legends", [])
        self.titles = layers.get("titles", [])
        self.window = window
        self._context = context

    @property
    def fig(self):
        """Alias of :attr:`figure`."""
        return self.figure

    @property
    def calibrations(self) -> list:
        """The recovered affine calibration of every linear axis (``None``
        for spline axes): ``cal.position(values)`` maps variable values to
        biplot coordinates."""
        return [a.calibration for a in self.calibrated_axes.calibrated_axes]

    def save(self, path, *, dpi: int = 300, bbox_inches="tight", **kwargs):
        """Save the figure (``.pdf``, ``.svg``, ``.eps`` for vector output;
        ``.png``/``.tiff`` at ``dpi``). Extra keyword arguments go to
        :meth:`matplotlib.figure.Figure.savefig`. Returns ``path``."""
        self.figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        return path

    def show(self):
        """Show the figure (``matplotlib.pyplot.show()``)."""
        plt.show()

    def __repr__(self) -> str:
        n_axes = len(self.calibrated_axes.calibrated_axes)
        n = sum(len(line.get_xdata()) for line in self.samples)
        return (
            f"<bipl5 StaticBiplot: {self._context.description}, "
            f"{n_axes} calibrated axes, {n} samples>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The builder
# ─────────────────────────────────────────────────────────────────────────────


def _auto_figsize(xr, yr, has_legend: bool, n_title_lines: int, caption: bool):
    w, h = xr[1] - xr[0], yr[1] - yr[0]
    long_side = 5.0
    pw, ph = (long_side, long_side * h / w) if w >= h else (long_side * w / h, long_side)
    return (
        pw + 0.9 + (1.5 if has_legend else 0.0),
        ph + 0.8 + 0.3 * n_title_lines + (0.3 if caption else 0.0),
    )


def plot_static(
    x,
    display=None,
    *,
    ax=None,
    figsize: tuple[float, float] | None = None,
    axis_colour="black",
    tick_colour=None,
    text_colour=None,
    label_colour=None,
    axis_linewidth=0.5,
    axis_linetype="solid",
    text_size=7.4,
    label_size=11.0,
    halo_colour=None,
    halo_width=2.0,
    axis_labels: bool = True,
    axis_ticks: bool = True,
    axis_text: bool = True,
    tick_extend: bool = True,
    tick_length: float = 0.025,
    text_dodge: float = 0.03,
    label_dodge: float = 0.012,
    label_placement: str = "positive",
    tidy_text: bool = True,
    point_size: float = 5.0,
    point_alpha: float = 0.6,
    sample_labels=None,
    sample_label_size: float = 8.0,
    class_means: bool | None = None,
    mean_size: float = 8.0,
    alpha_bags: bool = True,
    conc_ellipses: bool = True,
    legend: bool | None = None,
    legend_title=None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    axis_percentages: bool = True,
    xlim=None,
    ylim=None,
    expand: bool = True,
    square: bool = False,
    exp_factor: float | None = None,
    clip: bool = True,
    theme="bw",
) -> StaticBiplot:
    """Draw a static, publication-ready biplot with matplotlib.

    Port of the R package's ``wrap_bipl5_gg()``. Sample points, class means,
    alpha bags and concentration ellipses are ordinary matplotlib artists;
    the calibrated axes are a :class:`~bipl5.static.CalibratedAxes` artist
    that reproduces the geometry of ``geom_calibrated_axis()``:

    - every axis is a line spanning the whole plotting window;
    - the markers are exactly the ones ``axes_coordinates()`` chose — same
      values, spacing and phase — but, with ``tick_extend=True``, they are
      regenerated from the recovered calibration along the **entire visible
      axis**, so the tick marks run all the way to the edge of the plot
      (``axes_coordinates()`` only covers the range of the fitted values);
    - marker values are rotated to the axis angle and printed below it;
    - each variable name is written along its axis, inside the panel, at
      the increasing end.

    Parameters
    ----------
    x:
        A :class:`~bipl5.Biplot` (from ``scale_mds()``) or a fitted
        ordination object from :mod:`bipl5.ordination`.
    display:
        Which dimension pair of a ``Biplot`` to draw: a display name
        (``"mdsDisplay_13"``), a pair (``(1, 3)``) or a dropdown label
        (``"PC 1 & 3"``). Default: the first display.
    ax:
        Draw into an existing matplotlib ``Axes`` (e.g. one panel of a
        multi-panel figure) instead of creating a new figure.
    figsize:
        Size of a newly created figure, in inches. Default: sized so the
        panel's longer side is 5 inches.
    axis_colour:
        Colour of the axis lines: one value, one per axis, or
        ``"biplotEZ"`` to use the colours set by
        :func:`bipl5.ordination.axes` (grey by default).
    tick_colour, text_colour, label_colour:
        Colours of the tick marks, marker values and variable names;
        ``None`` follows ``axis_colour``; ``"biplotEZ"`` as above.
    axis_linewidth, axis_linetype:
        Width (points) and type (``"solid"``, ``"dashed"``, R ``lty``
        codes, or matplotlib styles) of the axis lines.
    text_size, label_size:
        Font sizes (points) of the marker values and the variable names.
    halo_colour, halo_width:
        Outline the axis lines, tick marks, marker values and variable
        names in ``halo_colour``, ``halo_width`` points wide, so the axes
        stay legible over a heatmap drawn beneath the biplot. One value or
        one per axis; ``None`` (default) draws no outline.
    axis_labels, axis_ticks, axis_text:
        Whether to draw the variable names, the tick marks and the marker
        values.
    tick_extend:
        Regenerate markers along the whole visible axis (default). ``False``
        draws only the markers ``axes_coordinates()`` returned. Spline axes
        always draw their own markers.
    tick_length:
        Tick length, as a proportion of the smaller panel dimension.
    text_dodge, label_dodge:
        Distances of the marker values and the variable names from the axis,
        as proportions of the smaller panel dimension.
    label_placement:
        Where variable names go: ``"positive"`` (the increasing end of the
        axis), ``"negative"`` or ``"peripheral"`` (the end farther from
        the origin).
    tidy_text:
        Drop marker values that the panel border would cut or that collide
        with a variable name, and slide variable names into the panel.
    point_size, point_alpha:
        Marker size (points) and opacity of the samples.
    sample_labels:
        ``True`` labels every sample with its row name; a sequence gives
        one label per sample; ``None``/``False`` (default) draws none.
    sample_label_size:
        Font size (points) of the sample labels.
    class_means:
        Draw class means; ``None`` (default) follows the biplot
        (``show_class_means`` / always for CVA).
    mean_size:
        Marker size (points) of the class means.
    alpha_bags, conc_ellipses:
        Draw alpha bags / concentration ellipses when the object has them.
    legend:
        Show the sample legend; ``None`` shows it when there is more than
        one group.
    legend_title:
        Legend title (a list of two under dual stratification). Default:
        the ``format_samples()`` title, else ``"Group"``.
    title, subtitle, caption:
        Plot annotations. ``title`` defaults to the biplot's title;
        ``caption`` to the display-quality string the interactive biplot
        shows. Pass ``caption=""`` to drop it.
    axis_percentages:
        Append the percentage of variation explained to the titles of the
        two scaffolding axes (``"PC1 (73.0%)"``).
    xlim, ylim, expand:
        Plotting window, as ggplot2's ``coord_equal()``: by default the
        range of the samples (and means and polygons) expanded by 5% on each
        side; ``expand=False`` uses the limits exactly.
    square:
        Pad the narrower range so the panel is square (biplotEZ's
        ``pty = "s"``).
    exp_factor:
        biplotEZ's window instead: ``range(Z * exp_factor)``, unexpanded
        (biplotEZ uses 1.2).
    clip:
        Clip all layers to the panel.
    theme:
        ``"bw"`` (default; ggplot2's ``theme_bw()``), ``"minimal"``,
        ``"classic"``, ``"biplotEZ"`` (a framed panel with no scaffolding
        axes), ``None`` (matplotlib's defaults), or a callable that styles
        the ``Axes``.

    Returns
    -------
    StaticBiplot
        The figure, axes and layer artists. Save with
        ``.save("biplot.pdf")`` or ``.figure.savefig(...)``.
    """
    from ..biplot import Biplot

    if isinstance(x, Biplot):
        ctx = _context_from_biplot(x, display)
    elif isinstance(x, EZBiplot):
        if display is not None:
            raise ValueError("'display' only applies to Biplot objects.")
        ctx = _context_from_ez(x)
    else:
        raise TypeError(
            "plot_static() needs a bipl5 Biplot (from scale_mds()) or a fitted "
            "ordination object from bipl5.ordination."
        )

    # ── calibrated axis layer data (port of calibrated_axis_frame) ──────
    ax_aes = ctx.ez.axes
    n_ax = len(ctx.z_axes)
    names = list(ax_aes.get("ax_names") or [ctx.ez.col_names[w] for w in ax_aes["which"]])
    # biplotEZ's tick.label.cex = 0.6 and label.cex = 0.75, in points
    ez_text = 0.6 * 3.88 * PT_PER_MM
    ez_label = 0.75 * 3.88 * PT_PER_MM
    aes_axis = resolve_axis_aes(axis_colour, ax_aes.get("col"), "black", n_ax)
    aes_tick = resolve_axis_aes(tick_colour, ax_aes.get("tick_col"), aes_axis, n_ax)
    aes_text = resolve_axis_aes(text_colour, ax_aes.get("tick_label_col"), aes_axis, n_ax)
    aes_label = resolve_axis_aes(
        label_colour, ax_aes.get("label_col", ax_aes.get("col")), aes_axis, n_ax
    )
    aes_lw = resolve_axis_aes(
        axis_linewidth, [0.75 * float(v) for v in ax_aes.get("lwd", [1])], 0.5, n_ax
    )
    aes_lty = resolve_axis_aes(axis_linetype, ax_aes.get("lty"), "solid", n_ax)
    aes_tsize = resolve_axis_aes(text_size, [ez_text], 7.4, n_ax)
    aes_lsize = resolve_axis_aes(label_size, [ez_label], 11.0, n_ax)
    aes_halo = resolve_axis_aes(halo_colour, None, None, n_ax)
    aes_halo_w = resolve_axis_aes(halo_width, None, 2.0, n_ax)

    records = []
    for i, coords in enumerate(ctx.z_axes):
        style = AxisStyle(
            axis_colour=aes_axis[i],
            axis_linewidth=float(aes_lw[i]),
            axis_linetype=aes_lty[i],
            tick_colour=aes_tick[i],
            text_colour=aes_text[i],
            text_size=float(aes_tsize[i]),
            label_colour=aes_label[i],
            label_size=float(aes_lsize[i]),
            halo_colour=aes_halo[i],
            halo_width=float(aes_halo_w[i]),
        )
        records.append(
            CalibratedAxis.from_coordinates(
                coords, names[i] if i < len(names) else "", curved=ctx.curved, style=style
            )
        )

    show_means = ctx.means is not None if class_means is None else bool(class_means)
    if show_means and ctx.means is None:
        show_means = False
    polygons = [
        p
        for p in ctx.polygons
        if (p["kind"] == "alpha_bags" and alpha_bags)
        or (p["kind"] == "conc_ellipses" and conc_ellipses)
    ]
    xr, yr = _window(ctx, xlim, ylim, expand, square, exp_factor, show_means, bool(polygons))

    n_groups = len(ctx.groups)
    show_legend = (n_groups > 1) if legend is None else bool(legend)
    caption = ctx.caption if caption is None else (caption or None)
    title = ctx.title if title is None else title

    # ── figure ──────────────────────────────────────────────────────────
    if ax is None:
        if figsize is None:
            figsize = _auto_figsize(
                xr, yr, show_legend, int(bool(title)) + int(bool(subtitle)), bool(caption)
            )
        fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    theme_opts = _apply_theme(ax, theme)
    layers: dict[str, list] = {}

    # ── alpha bags and concentration ellipses ──────────────────────────
    layers["polygons"] = []
    for p in polygons:
        patch = Polygon(
            p["xy"],
            closed=True,
            facecolor=to_rgba(p["colour"], p["alpha"]),
            edgecolor=p["colour"],
            linewidth=p["linewidth"],
            linestyle=r_linetype(p["linetype"]),
            zorder=ZORDER["polygons"],
            clip_on=clip,
            label=p["name"],
        )
        ax.add_patch(patch)
        layers["polygons"].append(patch)

    # ── samples ────────────────────────────────────────────────────────
    layers["samples"] = []
    Z = ctx.Z
    for g in ctx.groups:
        pts = Z[g["index"]]
        (line,) = ax.plot(
            pts[:, 0],
            pts[:, 1],
            **marker_kwargs(g["symbol"], g["color"], point_size),
            alpha=point_alpha,
            zorder=ZORDER["samples"],
            clip_on=clip,
            label=g["label"],
        )
        layers["samples"].append(line)

    # ── sample labels ──────────────────────────────────────────────────
    layers["sample_labels"] = []
    if sample_labels is not None and sample_labels is not False:
        if sample_labels is True:
            label_text = list(ctx.ez.row_names)
        else:
            label_text = [str(s) for s in sample_labels]
            if len(label_text) != Z.shape[0]:
                raise ValueError(
                    f"sample_labels must be True or have one label per sample ({Z.shape[0]})."
                )
        shown = np.concatenate([g["index"] for g in ctx.groups]) if ctx.groups else []
        for i in np.sort(shown):
            layers["sample_labels"].append(
                ax.annotate(
                    label_text[i], (Z[i, 0], Z[i, 1]),
                    xytext=(0, -0.45 * point_size), textcoords="offset points",
                    ha="center", va="top", fontsize=sample_label_size,
                    zorder=ZORDER["sample_labels"], annotation_clip=clip,
                )
            )

    # ── class means ────────────────────────────────────────────────────
    layers["class_means"] = []
    if show_means:
        m = ctx.means
        for k in range(len(m["x"])):
            (line,) = ax.plot(
                [m["x"][k]],
                [m["y"][k]],
                **marker_kwargs(
                    m["symbols"][k % len(m["symbols"])],
                    m["colors"][k % len(m["colors"])],
                    mean_size,
                ),
                zorder=ZORDER["means"],
                clip_on=clip,
                label=f"{m['labels'][k]} (mean)",
            )
            layers["class_means"].append(line)

    # ── calibrated axes ────────────────────────────────────────────────
    artist = CalibratedAxes(
        records,
        axis_labels=axis_labels,
        axis_ticks=axis_ticks,
        axis_text=axis_text,
        tick_extend=tick_extend and not ctx.curved,
        tick_length=tick_length,
        text_dodge=text_dodge,
        label_dodge=label_dodge,
        label_placement=label_placement,
        tidy_text=tidy_text,
        clip=clip,
        zorder=ZORDER["axes"],
    )
    ax.add_artist(artist)

    # ── scaffolding ────────────────────────────────────────────────────
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal", adjustable="box")
    if theme_opts.get("axis_titles", True):
        xlab, ylab = _dim_labels(ctx.ez, ctx.dim_prefix, axis_percentages)
        ax.set_xlabel(xlab, fontsize=_BASE)
        ax.set_ylabel(ylab, fontsize=_BASE)
    layers["titles"] = _add_titles(
        ax, title, subtitle, caption, theme_opts.get("title_loc", "left")
    )

    layers["legends"] = []
    if show_legend and ctx.groups:
        layers["legends"] = _add_legends(
            ax, ctx.legend, legend_title, point_size, point_alpha
        )

    return StaticBiplot(fig, ax, artist, layers, ctx, (xr, yr))


def _add_legends(ax, sections, legend_title, point_size, point_alpha) -> list:
    """The sample legend, to the right of the panel like ggplot2's: one
    section, or two stacked sections under dual stratification."""
    if isinstance(legend_title, str) or legend_title is None:
        titles = [legend_title] * len(sections)
    else:
        titles = list(legend_title) + [None] * len(sections)
    legends = []
    n = len(sections)
    for k, sec in enumerate(sections):
        handles = [
            Line2D([], [], **marker_kwargs(symbol, color, point_size), alpha=point_alpha)
            for _, color, symbol in sec["entries"]
        ]
        labels = [str(lab) for lab, _, _ in sec["entries"]]
        sec_title = titles[k] or sec.get("title") or "Group"
        if n == 1:
            loc, anchor = "center left", (1.02, 0.5)
        elif k == 0:
            loc, anchor = "lower left", (1.02, 0.52)
        else:
            loc, anchor = "upper left", (1.02, 0.48)
        kwargs = dict(
            title=sec_title,
            loc=loc,
            bbox_to_anchor=anchor,
            frameon=False,
            fontsize=_SMALL,
            title_fontsize=_BASE,
            alignment="left",
            handletextpad=0.3,
            borderaxespad=0.0,
            handlelength=1.2,
        )
        if k == 0:
            leg = ax.legend(handles, labels, **kwargs)
        else:
            leg = Legend(ax, handles, labels, **kwargs)
            ax.add_artist(leg)
        legends.append(leg)
    return legends
