"""Calibrated (Gower) biplot axes for matplotlib — port of
``geom_calibrated_axis.R`` (``GeomCalibratedAxis``) from the R package.

:class:`CalibratedAxes` is a matplotlib :class:`~matplotlib.artist.Artist`
that lays its axes out **at draw time**, against whatever plotting window the
:class:`~matplotlib.axes.Axes` ends up with — the matplotlib analogue of a
ggplot2 ``draw_panel()`` method:

- each axis is a line spanning the whole panel (or a polyline for the
  spline axes of PCO biplots);
- tick marks are short segments orthogonal to the axis, **regenerated across
  the entire visible axis** from the calibration recovered from
  :func:`bipl5.ordination.axes_coordinates`, so they always reach the panel
  border — including after ``ax.set_xlim()``, a different figure size, or
  interactive zooming;
- marker values are rotated to the axis angle and dodged below the axis;
- the variable name is written alongside the axis just inside the panel at
  its increasing end (biplotEZ's ``label.dir = "Along"`` placement), slid
  inwards or flipped across the axis when it would run out of the panel.

The calibrated axes never take part in autoscaling, so — as in the R
package — they never inflate the plotting window.

Geometry assumes an equal aspect ratio (``ax.set_aspect("equal")``), which a
biplot needs anyway for orthogonal projection to read off values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from matplotlib import artist as martist
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .helpers import (
    AxisCalibration,
    axis_calibration,
    curve_angles,
    dodge_direction,
    extend_axis_values,
    fold_angle,
    format_tick_values,
    inside_window,
    line_window_span,
    r_linetype,
)

__all__ = [
    "AxisStyle",
    "CalibratedAxis",
    "CalibratedAxes",
    "add_calibrated_axes",
]

_PLACEMENTS = ("positive", "negative", "peripheral")


@dataclass
class AxisStyle:
    """Visual properties of one calibrated axis.

    Sizes are in points and line widths in points (matplotlib units).
    ``tick_colour``, ``text_colour``, ``label_colour`` and
    ``tick_linewidth`` left at ``None`` follow the axis line, mirroring the
    fallbacks of the R layer.
    """

    axis_colour: object = "black"
    axis_linewidth: float = 0.5
    axis_linetype: object = "solid"
    axis_alpha: float | None = None
    tick_colour: object = None
    tick_linewidth: float | None = None
    text_colour: object = None
    text_size: float = 7.4
    label_colour: object = None
    label_size: float = 11.0
    fontfamily: str | None = None
    label_fontstyle: str = "normal"
    label_fontweight: str = "normal"

    def resolved(self) -> "AxisStyle":
        """Port of ``gg_fill_axis_defaults()``: fill the inherited values."""
        return replace(
            self,
            tick_colour=self.axis_colour if self.tick_colour is None else self.tick_colour,
            tick_linewidth=(
                self.axis_linewidth if self.tick_linewidth is None else self.tick_linewidth
            ),
            text_colour=self.axis_colour if self.text_colour is None else self.text_colour,
            label_colour=(
                self.axis_colour if self.label_colour is None else self.label_colour
            ),
        )


@dataclass
class CalibratedAxis:
    """The layer data for one calibrated axis (one group of the R layer).

    Build these with :meth:`from_coordinates` from one element of
    :func:`bipl5.ordination.axes_coordinates`.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    value: np.ndarray
    tick: np.ndarray
    angle: np.ndarray
    label: list[str]
    calibration: AxisCalibration | None
    curved: bool = False
    style: AxisStyle = field(default_factory=AxisStyle)

    @classmethod
    def from_coordinates(
        cls,
        coords,
        name: str,
        curved: bool = False,
        style: AxisStyle | None = None,
    ) -> "CalibratedAxis | None":
        """Port of the per-axis branch of ``calibrated_axis_frame()``.

        ``coords`` is a ``k x 3`` array of marker x, y and value for a linear
        axis, or the ``k x 4`` curve of a spline axis whose fourth column
        flags the labelled ticks. Returns ``None`` for unusable input (too
        few markers, or a degenerate linear calibration).
        """
        ax = np.asarray(coords, dtype=float)
        if ax.ndim != 2 or ax.shape[0] < 2 or ax.shape[1] < 3:
            return None
        px, py, vv = ax[:, 0].copy(), ax[:, 1].copy(), ax[:, 2].copy()
        style = style or AxisStyle()

        if curved:
            angle = curve_angles(px, py, vv)
            if ax.shape[1] >= 4:
                tick = np.isfinite(ax[:, 3]) & (ax[:, 3] == 1)
            else:
                tick = np.ones(ax.shape[0], dtype=bool)
            tick &= np.isfinite(angle)
            steps = np.diff(np.unique(vv[tick]))
            vstep = float(np.median(steps)) if steps.size else math.nan
            label = [""] * ax.shape[0]
            for i, text in zip(np.where(tick)[0], format_tick_values(vv[tick], vstep)):
                label[i] = text
            return cls(name, px, py, vv, tick, angle, label, None, True, style)

        cal = axis_calibration(ax)
        if cal is None:
            return None
        angle = np.full(ax.shape[0], cal.angle)
        tick = np.ones(ax.shape[0], dtype=bool)
        label = format_tick_values(vv, cal.vstep)
        return cls(name, px, py, vv, tick, angle, label, cal, False, style)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.hypot(v[0], v[1]))
    return v / norm if norm > 0 else v


class CalibratedAxes(martist.Artist):
    """A matplotlib artist drawing calibrated biplot axes.

    Port of ``GeomCalibratedAxis``. Add it to an axes with
    ``ax.add_artist(...)`` (or use :func:`add_calibrated_axes`); the
    geometry is recomputed from the current view limits every time the
    figure is drawn.

    Parameters
    ----------
    axes:
        :class:`CalibratedAxis` records, one per biplot axis.
    axis_line, axis_labels, axis_ticks, axis_text:
        Whether to draw the axis lines, the variable names, the tick marks
        and the marker values.
    tick_extend:
        Regenerate markers across the whole visible axis from the
        recovered calibration (default), preserving biplotEZ's marker
        spacing and phase. ``False`` draws only the supplied markers.
        Ignored for curved axes.
    tick_length:
        Tick length as a proportion of the smaller panel dimension.
    text_dodge:
        Orthogonal distance from the axis to the centre of the marker
        values, as a proportion of the smaller panel dimension.
    label_dodge:
        Orthogonal gap between the axis and the variable name, as a
        proportion of the smaller panel dimension.
    label_placement:
        ``"positive"`` (the increasing end of the axis), ``"negative"``,
        or ``"peripheral"`` (the end farther from the origin).
    tidy_text:
        Drop marker values that would be cut by the panel border or that
        collide with a variable name, and keep variable names inside the
        panel. ``False`` draws everything and lets clipping decide.
    clip:
        Clip to the panel (default), like ggplot2's ``clip = "on"``.
    """

    def __init__(
        self,
        axes,
        *,
        axis_line: bool = True,
        axis_labels: bool = True,
        axis_ticks: bool = True,
        axis_text: bool = True,
        tick_extend: bool = True,
        tick_length: float = 0.025,
        text_dodge: float = 0.03,
        label_dodge: float = 0.012,
        label_placement: str = "positive",
        tidy_text: bool = True,
        clip: bool = True,
        zorder: float = 1.0,
    ):
        super().__init__()
        if label_placement not in _PLACEMENTS:
            raise ValueError(
                "label_placement must be one of: " + ", ".join(_PLACEMENTS)
            )
        self.calibrated_axes = [a for a in axes if a is not None]
        self.axis_line = axis_line
        self.axis_labels = axis_labels
        self.axis_ticks = axis_ticks
        self.axis_text = axis_text
        self.tick_extend = tick_extend
        self.tick_length = tick_length
        self.text_dodge = text_dodge
        self.label_dodge = label_dodge
        self.label_placement = label_placement
        self.tidy_text = tidy_text
        self.clip = clip
        self.set_zorder(zorder)
        # contents are clipped to the panel: keep them out of layout
        self.set_in_layout(False)
        self.last_render: dict = {}

    # ── geometry (backend-free; testable) ──────────────────────────────

    def compute_layout(self, xrange, yrange) -> dict:
        """Lay the axes out against a plotting window.

        Returns a dict with ``"lines"`` (per-axis polylines), ``"marks"``
        (tick positions, values, labels, angles and dodge directions)
        and ``"titles"`` (the end of each axis where its name goes).
        This is the window-dependent part of ``draw_panel()``; drawing
        itself only adds text measurement on top.
        """
        xr = (float(min(xrange)), float(max(xrange)))
        yr = (float(min(yrange)), float(max(yrange)))
        lines, marks, titles = [], [], []
        for i, axis in enumerate(self.calibrated_axes):
            line = self._axis_line(axis, xr, yr)
            if line is not None:
                lines.append({"axis": i, "x": line[0], "y": line[1]})
            marks.extend(self._axis_marks(i, axis, xr, yr))
            title = self._axis_title(axis, xr, yr)
            if title is not None:
                title["axis"] = i
                titles.append(title)
        return {
            "xrange": xr,
            "yrange": yr,
            "plot_whmin": min(xr[1] - xr[0], yr[1] - yr[0]),
            "lines": lines,
            "marks": marks,
            "titles": titles,
        }

    @staticmethod
    def _axis_line(axis, xr, yr):
        if axis.curved:
            return axis.x, axis.y
        cal = axis.calibration
        span = line_window_span((cal.x0, cal.y0), cal.direction, xr, yr)
        if span is None:
            return None
        xs, ys = cal.position(np.array(span))
        return xs, ys

    def _axis_marks(self, i, axis, xr, yr) -> list[dict]:
        """Port of ``gg_axis_mark_data()`` for one axis."""
        cal = axis.calibration
        values = None
        if (
            not axis.curved
            and self.tick_extend
            and cal is not None
            and cal.has_step
        ):
            # falls back to the supplied markers when the axis misses
            # the panel or would need an unreasonable number of marks
            values = extend_axis_values(cal, xr, yr)

        if values is not None:
            px, py = cal.position(values)
            labels = format_tick_values(values, cal.vstep)
            angles = np.full(values.size, cal.angle)
        else:
            sel = axis.tick
            px, py = axis.x[sel], axis.y[sel]
            values = axis.value[sel]
            labels = [lab for lab, s in zip(axis.label, sel) if s]
            angles = axis.angle[sel]

        inside = inside_window(px, py, xr, yr)
        offset = cal.offset if cal is not None else (0.0, 0.0)
        out = []
        for k in np.where(inside)[0]:
            out.append(
                {
                    "axis": i,
                    "x": float(px[k]),
                    "y": float(py[k]),
                    "value": float(values[k]),
                    "label": labels[k],
                    "angle": float(angles[k]),
                    "dodge": dodge_direction(angles[k], offset),
                }
            )
        return out

    def _axis_title(self, axis, xr, yr) -> dict | None:
        """Where the variable name goes: the exit point ``E`` of the
        axis, the outward direction ``out`` there, and the side ``n``
        (opposite the marker values) — biplotEZ's ``.axis_title_along``
        with bipl5's ``label_placement`` choices."""
        if not axis.name:
            return None
        if axis.curved:
            inside = inside_window(axis.x, axis.y, xr, yr) & np.isfinite(axis.angle)
            if not inside.any():
                return None
            idx = np.where(inside)[0]
            vals = axis.value[idx]
            has_value = np.isfinite(vals).any()
            if self.label_placement == "negative":
                pick = idx[int(np.nanargmin(vals))] if has_value else idx[0]
            elif self.label_placement == "peripheral":
                pick = idx[int(np.argmax(axis.x[idx] ** 2 + axis.y[idx] ** 2))]
            else:
                pick = idx[int(np.nanargmax(vals))] if has_value else idx[-1]
            ang = float(axis.angle[pick])
            point = np.array([axis.x[pick], axis.y[pick]])
            # local tangent, oriented towards increasing value
            out = np.array([math.cos(ang), math.sin(ang)])
            if self.label_placement == "negative" or (
                self.label_placement == "peripheral" and float(point @ out) < 0
            ):
                out = -out
            offset = (0.0, 0.0)
        else:
            cal = axis.calibration
            u = _unit(cal.direction)
            p0 = np.array([cal.x0, cal.y0])
            span = line_window_span(p0, u, xr, yr)
            if span is None:
                return None
            e_pos, e_neg = p0 + span[1] * u, p0 + span[0] * u
            if self.label_placement == "negative":
                point, out = e_neg, -u
            elif self.label_placement == "peripheral":
                if float(e_pos @ e_pos) >= float(e_neg @ e_neg):
                    point, out = e_pos, u
                else:
                    point, out = e_neg, -u
            else:
                point, out = e_pos, u
            ang = cal.angle
            offset = cal.offset

        fold = fold_angle(ang)
        reading = np.array([math.cos(fold), math.sin(fold)])
        up = np.array([-reading[1], reading[0]])
        side = dodge_direction(ang, offset)
        return {
            "point": point,
            "out": _unit(out),
            "side": side,
            "rotation": math.degrees(fold),
            # the text ends at the exit point when it reads outwards
            "ha": "right" if float(reading @ out) > 0 else "left",
            "va": "bottom" if float(side @ up) > 0 else "top",
            "label": axis.name,
        }

    # ── drawing ───────────────────────────────────────────────────────

    def _text(self, x, y, s, *, size, color, rotation, ha, va, style):
        t = Text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va=va,
        )
        if style.fontfamily:
            t.set_fontfamily(style.fontfamily)
        if style.axis_alpha is not None:
            t.set_alpha(style.axis_alpha)
        self._adopt(t)
        return t

    def _adopt(self, art):
        ax = self.axes
        art.set_figure(self.figure)
        art.set_transform(ax.transData)
        if self.clip:
            art.set_clip_on(True)
            art.set_clip_box(ax.bbox)
        else:
            art.set_clip_on(False)
        return art

    @staticmethod
    def _inside(bb, panel, tol=0.5) -> bool:
        return (
            bb.x0 >= panel.x0 - tol
            and bb.x1 <= panel.x1 + tol
            and bb.y0 >= panel.y0 - tol
            and bb.y1 <= panel.y1 + tol
        )

    def _slide_inside(self, text, renderer, panel, direction, max_shift):
        """Pixels to slide ``text`` along ``direction`` (display unit
        vector) to bring it inside ``panel``; ``None`` if impossible."""
        bb = text.get_window_extent(renderer)
        dx, dy = direction
        needs = []
        for overflow, comp in (
            (panel.x0 - bb.x0, dx),
            (bb.x1 - panel.x1, -dx),
            (panel.y0 - bb.y0, dy),
            (bb.y1 - panel.y1, -dy),
        ):
            if overflow > 0.5:
                if comp <= 1e-6:
                    return None
                needs.append(overflow / comp)
        shift = max(needs, default=0.0)
        if shift > max_shift:
            return None
        moved = Bbox.from_extents(
            bb.x0 + shift * dx,
            bb.y0 + shift * dy,
            bb.x1 + shift * dx,
            bb.y1 + shift * dy,
        )
        return shift if self._inside(moved, panel, tol=1.0) else None

    def _place_title(self, title, style, renderer, whmin, panel, placed):
        """Build the variable-name text, keeping it inside the panel and
        clear of the names already placed (``placed``: display bboxes).

        Candidates are the two sides of the axis (the side opposite the
        marker values first), each slid back along the axis — just far
        enough to fit inside the panel, then further in small steps. The
        best candidate is inside the panel, overlaps no earlier name (or the
        least), and needs the least sliding; switching sides costs about
        one line of text.
        """
        trans = self.axes.transData
        gap = self.label_dodge * whmin
        inset = 0.01 * whmin
        base = title["point"] - inset * title["out"]
        flip_va = {"bottom": "top", "top": "bottom"}
        max_shift = 0.45 * min(panel.width, panel.height)
        step = style.label_size * renderer.points_to_pixels(1.0) * 0.6

        best = None  # (key, text, position)
        for side_idx, (side, va) in enumerate(
            ((title["side"], title["va"]), (-title["side"], flip_va[title["va"]]))
        ):
            anchor = base + gap * side
            t = self._text(
                anchor[0],
                anchor[1],
                title["label"],
                size=style.label_size,
                color=style.label_colour,
                rotation=title["rotation"],
                ha=title["ha"],
                va=va,
                style=style,
            )
            t.set_fontstyle(style.label_fontstyle)
            t.set_fontweight(style.label_fontweight)
            if not self.tidy_text:
                return t
            # slide back along the axis (display space), into the panel
            d0 = trans.transform(anchor)
            d1 = trans.transform(anchor - title["out"] * whmin * 0.01)
            direction = _unit(d1 - d0)
            shift0 = self._slide_inside(t, renderer, panel, direction, max_shift)
            shifts = [0.0] if shift0 is None else np.arange(shift0, max_shift + 1e-9, step)
            for shift in shifts:
                pos = trans.inverted().transform(d0 + shift * direction)
                t.set_position((pos[0], pos[1]))
                bb = t.get_window_extent(renderer)
                overlap = 0.0
                for other in placed:
                    inter = Bbox.intersection(bb, other)
                    if inter is not None:
                        overlap += inter.width * inter.height
                key = (
                    not self._inside(bb, panel),
                    overlap > 1e-6,
                    overlap,
                    shift + side_idx * step * 2,
                )
                if best is None or key < best[0]:
                    best = (key, t, (pos[0], pos[1]))
                if key[:2] == (False, False):
                    break  # cleanest spot on this side; larger slides only cost
        key, t, pos = best
        if key[0]:
            return None  # cannot fit inside the panel: drop, never truncate
        t.set_position(pos)
        return t

    @martist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible() or self.axes is None:
            return
        ax = self.axes
        layout = self.compute_layout(ax.get_xlim(), ax.get_ylim())
        whmin = layout["plot_whmin"]
        panel = ax.bbox
        styles = [a.style.resolved() for a in self.calibrated_axes]

        renderer.open_group("calibrated_axes", gid=self.get_gid())
        artists = []

        if self.axis_line:
            for line in layout["lines"]:
                st = styles[line["axis"]]
                ln = Line2D(
                    line["x"],
                    line["y"],
                    color=st.axis_colour,
                    linewidth=st.axis_linewidth,
                    linestyle=r_linetype(st.axis_linetype),
                    alpha=st.axis_alpha,
                    solid_capstyle="butt",
                )
                artists.append(self._adopt(ln))

        marks = layout["marks"] if (self.axis_ticks or self.axis_text) else []
        if self.axis_ticks and marks:
            rtick = whmin * self.tick_length / 2
            segs, cols, lws = [], [], []
            for m in marks:
                st = styles[m["axis"]]
                d = m["dodge"] * rtick
                segs.append([(m["x"] + d[0], m["y"] + d[1]), (m["x"] - d[0], m["y"] - d[1])])
                cols.append(st.tick_colour)
                lws.append(st.tick_linewidth)
            lc = LineCollection(segs, colors=cols, linewidths=lws, capstyle="butt")
            if any(st.axis_alpha is not None for st in styles):
                lc.set_alpha(
                    [styles[m["axis"]].axis_alpha or 1.0 for m in marks]
                )
            artists.append(self._adopt(lc))

        title_texts = []
        if self.axis_labels:
            placed = []
            for title in layout["titles"]:
                t = self._place_title(
                    title, styles[title["axis"]], renderer, whmin, panel, placed
                )
                if t is None:
                    continue
                title_texts.append((title, t))
                placed.append(t.get_window_extent(renderer).padded(1.0))

        mark_texts, drawn_marks = [], []
        if self.axis_text:
            title_boxes = [
                t.get_window_extent(renderer).padded(1.0) for _, t in title_texts
            ] if self.tidy_text else []
            for m in marks:
                if not m["label"]:
                    continue
                st = styles[m["axis"]]
                pos = np.array([m["x"], m["y"]]) - m["dodge"] * whmin * self.text_dodge
                t = self._text(
                    pos[0],
                    pos[1],
                    m["label"],
                    size=st.text_size,
                    color=st.text_colour,
                    rotation=math.degrees(fold_angle(m["angle"])),
                    ha="center",
                    va="center",
                    style=st,
                )
                if self.tidy_text:
                    bb = t.get_window_extent(renderer)
                    if not self._inside(bb, panel) or any(
                        bb.overlaps(tb) for tb in title_boxes
                    ):
                        continue
                mark_texts.append(t)
                drawn_marks.append(m)

        for art in artists + mark_texts + [t for _, t in title_texts]:
            art.draw(renderer)
        renderer.close_group("calibrated_axes")

        self.last_render = {
            **layout,
            "drawn_marks": drawn_marks,
            "title_texts": [
                {
                    "axis": ti["axis"],
                    "label": t.get_text(),
                    "position": t.get_position(),
                    "bbox": t.get_window_extent(renderer),
                }
                for ti, t in title_texts
            ],
            "panel": panel.frozen(),
        }
        self.stale = False


def add_calibrated_axes(
    ax,
    z_axes,
    names,
    *,
    curved: bool = False,
    style: AxisStyle | list[AxisStyle] | None = None,
    **kwargs,
):
    """Draw calibrated biplot axes onto an existing matplotlib ``Axes``.

    The analogue of R's ``geom_calibrated_axis()`` for hand-built plots:

    >>> from bipl5.ordination import biplot, axes_coordinates
    >>> ez = biplot(data).pca()
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(ez.Z[:, 0], ez.Z[:, 1])
    >>> ax.set_aspect("equal")
    >>> add_calibrated_axes(ax, axes_coordinates(ez), ez.col_names)

    ``z_axes`` is the list returned by
    :func:`~bipl5.ordination.axes_coordinates`; ``names`` one variable name
    per axis; ``style`` an :class:`AxisStyle` (or one per axis). Remaining
    keyword arguments go to :class:`CalibratedAxes`. Returns the artist.
    """
    n = len(z_axes)
    if style is None or isinstance(style, AxisStyle):
        styles = [style or AxisStyle()] * n
    else:
        styles = list(style)
    records = [
        CalibratedAxis.from_coordinates(z, str(nm), curved=curved, style=st)
        for z, nm, st in zip(z_axes, names, styles)
    ]
    artist = CalibratedAxes(records, **kwargs)
    ax.add_artist(artist)
    return artist
