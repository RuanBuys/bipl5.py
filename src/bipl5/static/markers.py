"""Marker translation for static biplots.

bipl5 stores sample symbols as plotly symbol names (``"circle"``,
``"triangle-up-open"``, …) and biplotEZ-style objects carry base-R ``pch``
codes. Both are translated here to matplotlib markers, together with the fill
style needed to reproduce filled, open and line-only symbols.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..symbols import pch_to_plotly

__all__ = ["MarkerStyle", "marker_style", "marker_kwargs"]

# plotly base symbol -> matplotlib marker. Tuples are matplotlib's
# (numsides, style, angle) polygon markers: style 0 polygon, 1 star,
# 2 asterisk. Text markers ("$...$") cover the circled/boxed glyphs.
_BASE = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "cross": "P",
    "x": "X",
    "triangle-up": "^",
    "triangle-down": "v",
    "triangle-left": "<",
    "triangle-right": ">",
    "triangle-ne": (3, 0, -45),
    "triangle-se": (3, 0, -135),
    "triangle-sw": (3, 0, 135),
    "triangle-nw": (3, 0, 45),
    "pentagon": "p",
    "hexagon": "h",
    "hexagon2": "H",
    "octagon": "8",
    "star": "*",
    "hexagram": (6, 1, 0),
    "star-triangle-up": (3, 1, 0),
    "star-triangle-down": (3, 1, 180),
    "star-square": (4, 1, 45),
    "star-diamond": (4, 1, 0),
    "diamond-tall": "d",
    "diamond-wide": "D",
    "hourglass": "X",
    "bowtie": "X",
    "circle-cross": "$⊕$",
    "circle-x": "$⊗$",
    "square-cross": "$⊞$",
    "square-x": "$⊠$",
    "diamond-cross": "P",
    "diamond-x": "X",
    "cross-thin": "+",
    "x-thin": "x",
    "asterisk": (6, 2, 0),
    "hash": "+",
    "y-up": "2",
    "y-down": "1",
    "y-left": "3",
    "y-right": "4",
    "line-ew": "_",
    "line-ns": "|",
    "line-ne": (2, 2, -45),
    "line-nw": (2, 2, 45),
    "arrow-up": "^",
    "arrow-down": "v",
    "arrow-left": "<",
    "arrow-right": ">",
    "arrow-bar-up": "^",
    "arrow-bar-down": "v",
    "arrow-bar-left": "<",
    "arrow-bar-right": ">",
}

# markers matplotlib can only stroke (they have no interior to fill)
_LINE_ONLY = {"+", "x", "1", "2", "3", "4", "|", "_"}


@dataclass(frozen=True)
class MarkerStyle:
    """A matplotlib marker plus how it is painted.

    ``fill`` is ``"filled"`` (solid, no outline — like R ``pch = 16``),
    ``"open"`` (outline only — ``pch = 1``) or ``"line"`` (stroke-only
    glyphs such as ``+`` and ``x``).
    """

    marker: object
    fill: str


def marker_style(symbol) -> MarkerStyle:
    """Translate a plotly symbol name or base-R ``pch`` code.

    Unknown names fall back to a filled circle, plotly's own default.
    """
    if not isinstance(symbol, str):
        symbol = pch_to_plotly([symbol])[0]
    name = symbol.strip()
    is_open = False
    for suffix in ("-open-dot", "-open", "-dot"):
        if name.endswith(suffix):
            is_open = "open" in suffix
            name = name[: -len(suffix)]
            break
    marker = _BASE.get(name, "o")
    if (
        marker in _LINE_ONLY
        or (isinstance(marker, str) and marker.startswith("$"))
        or (isinstance(marker, tuple) and marker[1] == 2)
    ):
        return MarkerStyle(marker, "line")
    return MarkerStyle(marker, "open" if is_open else "filled")


def marker_kwargs(symbol, color, size: float, edge_width: float = 1.0) -> dict:
    """``Line2D`` keyword arguments drawing ``symbol`` in ``color``."""
    style = marker_style(symbol)
    out = {"marker": style.marker, "markersize": size, "linestyle": "none"}
    if style.fill == "filled":
        out.update(markerfacecolor=color, markeredgecolor="none", markeredgewidth=0)
    elif style.fill == "open":
        out.update(
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=edge_width
        )
    else:
        out.update(markeredgecolor=color, markerfacecolor=color, markeredgewidth=edge_width)
    return out
