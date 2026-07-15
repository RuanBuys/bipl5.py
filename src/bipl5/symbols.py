"""Plotting symbols and colours — port of ``Plotting_Symbols.R`` and
``colorpal()`` from ``Translate_Helpers.R``.
"""

from __future__ import annotations

__all__ = ["pch_to_plotly", "colorpal", "symbol_list", "PCH_TO_PLOTLY"]

# R base plotting character -> plotly symbol name (Plotting_Symbols.R)
PCH_TO_PLOTLY = {
    0: "square-open",
    1: "circle-open",
    2: "triangle-up-open",
    3: "cross-thin",
    4: "x-thin",
    5: "diamond-open",
    6: "triangle-down-open",
    7: "square-x-open",
    8: "asterisk",
    9: "diamond-cross-open",
    10: "circle-cross-open",
    11: "hourglass-open",
    12: "square-cross-open",
    13: "circle-x-open",
    14: "hash-open",
    15: "square",
    16: "circle",
    17: "triangle-up",
    18: "diamond",
    19: "circle",
    20: "circle",
    21: "circle",
    22: "square",
    23: "diamond",
    24: "triangle-up",
    25: "triangle-down",
}


def pch_to_plotly(pch) -> list[str]:
    """Convert base-R ``pch`` codes to plotly symbol names.

    Strings are assumed to already be plotly symbol names and pass through
    unchanged; unknown numeric codes fall back to ``"circle"`` (plotly's own
    default).
    """
    out = []
    for value in pch:
        if isinstance(value, str):
            out.append(value)
        else:
            out.append(PCH_TO_PLOTLY.get(int(value), "circle"))
    return out


def colorpal(number: int = 16) -> list[str]:
    """bipl5's default colour scale — port of ``colorpal()``.

    Sixteen distinct colours; ``number`` selects the first ``1..16``.
    """
    number = int(number)
    if number > 16:
        raise ValueError("Only 16 unique colors are available")
    if number < 1:
        raise ValueError("Enter integer between 1 and 16")
    pal = [
        "#396AB1",
        "#DA7C30",
        "#3E9651",
        "#CC2529",
        "#535154",
        "#6B4C9A",
        "#922428",
        "#948B3D",
        "#7293CB",
        "#FF974C",
        "#84BA5B",
        "#D35E60",
        "#808585",
        "#9067A7",
        "#AB6857",
        "#CCC210",
    ]
    return pal[:number]


_BASE_SYMBOLS = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "triangle-down", "triangle-left", "triangle-right",
    "triangle-ne", "triangle-se", "triangle-sw", "triangle-nw",
    "pentagon", "hexagon", "hexagon2", "octagon", "star", "hexagram",
    "star-triangle-up", "star-triangle-down", "star-square", "star-diamond",
    "diamond-tall", "diamond-wide", "hourglass", "bowtie",
    "circle-cross", "circle-x", "square-cross", "square-x",
    "diamond-cross", "diamond-x", "cross-thin", "x-thin", "asterisk",
    "hash", "y-up", "y-down", "y-left", "y-right",
    "line-ew", "line-ns", "line-ne", "line-nw",
    "arrow-up", "arrow-down", "arrow-left", "arrow-right",
    "arrow-bar-up", "arrow-bar-down", "arrow-bar-left", "arrow-bar-right",
]


def symbol_list() -> list[str]:
    """Available plotly marker symbols — Python analogue of ``Symbol_List()``.

    Returns the base symbol names; most accept the modifier suffixes
    ``-open``, ``-dot`` and ``-open-dot`` (e.g. ``"circle-open-dot"``).
    """
    return list(_BASE_SYMBOLS)
