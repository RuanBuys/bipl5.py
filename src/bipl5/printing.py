"""Tree-style printing — port of the R package's print methods
(``print.bipl5_biplot``, ``print.bipl5_mdsDisplay``, ``print.bipl5_data``,
``print.bipl5_fitmeasures``) with their locale-safe tree symbols and crayon
colour scheme.

Container classes here are thin ``dict`` subclasses: the payload structures
keep behaving exactly like the plain dicts the rest of the package builds,
but render as coloured tree diagrams at the REPL.
"""

from __future__ import annotations

import os
import sys

import numpy as np

__all__ = [
    "MdsDisplay",
    "BiplotData",
    "FitMeasures",
    "set_print_options",
    "tree_symbols",
]

# user overrides, mirroring R's options(bipl5.unicode = ...); None = auto
_OPTIONS: dict = {"unicode": None, "color": None}


def set_print_options(unicode: bool | None = None, color: bool | None = None):
    """Override tree-printing behaviour.

    ``unicode`` toggles box-drawing characters vs ASCII (default: inferred
    from the stdout encoding, like R's ``l10n_info()`` check). ``color``
    toggles ANSI colours (default: only when stdout is a terminal and
    ``NO_COLOR`` is unset). Pass ``None`` to restore auto-detection.
    """
    _OPTIONS["unicode"] = unicode
    _OPTIONS["color"] = color


def _use_unicode() -> bool:
    if _OPTIONS["unicode"] is not None:
        return _OPTIONS["unicode"]
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "").startswith("utf")


def _use_color() -> bool:
    if _OPTIONS["color"] is not None:
        return _OPTIONS["color"]
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def tree_symbols(unicode: bool | None = None) -> dict:
    """Port of ``tree_symbols()``: branch/indent strings for the locale."""
    if unicode is None:
        unicode = _use_unicode()
    if unicode:
        return {
            "branch": "├── ",
            "last": "└── ",
            "pipe": "│   ",
            "space": "    ",
        }
    return {"branch": "+-- ", "last": "`-- ", "pipe": "|   ", "space": "    "}


_ANSI = {"bold": "1", "cyan": "36", "green": "32", "yellow": "33", "silver": "90"}


def _painter(color: bool):
    if not color:
        return lambda text, *styles: text

    def paint(text: str, *styles: str) -> str:
        seq = ";".join(_ANSI[s] for s in styles)
        return f"\x1b[{seq}m{text}\x1b[0m"

    return paint


def dim_label(value) -> str:
    """Port of ``dim_label()``: compact shape suffix like ``"  [60 x 2]"``."""
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return f"  [{value.shape[0]} x {value.shape[1]}]"
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return f"  [{value.shape[0]}]"
    if isinstance(value, (list, tuple)):
        return f"  [{len(value)}]"
    return ""


def _ft_label(ft_key: str, prefix: str = "PC") -> str:
    digits = ft_key.replace("fit_table_", "")
    return f"{prefix} {digits[0]} & {digits[1]}"


# ── tree renderers ──────────────────────────────────────────────────────────

def format_data_subtree(data: dict, prefix: str, tree: dict, paint) -> list[str]:
    """Port of ``print_data_subtree()``."""
    lines = [
        prefix + tree["branch"] + paint("Data", "green")
        + paint(" <bipl5_data>", "silver")
    ]
    inner = prefix + tree["pipe"]
    dims = dim_label(data.get("sample_coordinates"))
    lines.append(inner + tree["branch"] + "sample_coordinates" + paint(dims, "silver"))
    n_ax = len(data.get("axes_coordinates") or [])
    lines.append(
        inner + tree["branch"] + "axes_coordinates"
        + paint(f"  [{n_ax} axes]", "silver")
    )
    lines.append(inner + tree["last"] + "translated_axes_coordinates")
    return lines


def format_fitmeasures_subtree(fm: dict, prefix: str, tree: dict, paint) -> list[str]:
    """Port of ``print_fitmeasures_subtree()``."""
    lines = []
    for name in ("CumPred", "CumAd", "VarExp", "Scree"):
        n_tr = len(fm.get(name) or [])
        lines.append(
            prefix + tree["branch"] + name + paint(f"  [{n_tr} traces]", "silver")
        )
    tables = [k for k in fm if k.startswith("fit_table_") and fm[k] is not None]
    for j, key in enumerate(tables):
        branch = tree["last"] if j == len(tables) - 1 else tree["branch"]
        lines.append(
            prefix + branch + paint(key, "green")
            + paint(f"  [{_ft_label(key)}]", "silver")
        )
    return lines


def format_display_lines(bundle: dict, tree: dict, paint) -> list[str]:
    """Port of ``print.bipl5_mdsDisplay``."""
    lines = [paint("bipl5_mdsDisplay", "bold", "cyan")]
    if bundle.get("fit_qual"):
        lines.append(paint(str(bundle["fit_qual"]), "silver"))
    lines.extend(format_data_subtree(bundle.get("Data") or {}, "", tree, paint))
    mds = bundle.get("mds") or {}
    n_traces = len(mds.get("trace_data") or [])
    lines.append(
        tree["branch"] + paint("trace_data", "green")
        + paint(f"  [{n_traces} traces]", "silver")
    )
    n_ann = len((mds.get("layout") or {}).get("annotations") or [])
    lines.append(
        tree["last"] + paint("annotations", "green")
        + paint(f"  [{n_ann} items]", "silver")
    )
    return lines


def format_biplot_lines(bp, tree: dict, paint) -> list[str]:
    """Port of ``print.bipl5_biplot``."""
    lines = [paint(f"bipl5_biplot [{bp.biplot_type.upper()}]", "bold")]

    pc_info = bp.meta["pc_info"]
    present = [name for name in pc_info if name in bp.displays]
    has_fm = bp.fit_measures is not None

    for j, name in enumerate(present):
        bundle = bp.displays[name]
        is_last = j == len(present) - 1 and not has_fm
        branch = tree["last"] if is_last else tree["branch"]
        pipe = tree["space"] if is_last else tree["pipe"]

        lines.append(
            branch
            + paint(f"{name} [{pc_info[name]['label']}]", "bold", "cyan")
            + paint(" <bipl5_mdsDisplay>", "silver")
        )
        lines.extend(
            format_data_subtree(bundle.get("Data") or {}, pipe, tree, paint)
        )
        mds = bundle.get("mds") or {}
        n_traces = len(mds.get("trace_data") or [])
        lines.append(
            pipe + tree["branch"] + paint("trace_data", "green")
            + paint(f"  [{n_traces} traces]", "silver")
        )
        n_ann = len((mds.get("layout") or {}).get("annotations") or [])
        lines.append(
            pipe + tree["last"] + paint("annotations", "green")
            + paint(f"  [{n_ann} items]", "silver")
        )

    if has_fm:
        lines.append(
            tree["last"] + paint("fit_measures", "bold", "yellow")
            + paint(" <bipl5_fitmeasures>", "silver")
        )
        lines.extend(
            format_fitmeasures_subtree(
                bp.fit_measures, tree["space"], tree, paint
            )
        )
    return lines


def render(lines: list[str]) -> str:
    return "\n".join(lines)


# ── container classes ───────────────────────────────────────────────────────

class BiplotData(dict):
    """The ``Data`` node of an mdsDisplay (port of ``bipl5_data``)."""

    def __repr__(self) -> str:
        tree, paint = tree_symbols(), _painter(_use_color())
        lines = [paint("bipl5_data", "bold", "green")]
        lines.append(
            tree["branch"] + "sample_coordinates"
            + paint(dim_label(self.get("sample_coordinates")), "silver")
        )
        n_ax = len(self.get("axes_coordinates") or [])
        lines.append(
            tree["branch"] + "axes_coordinates"
            + paint(f"  [{n_ax} axes]", "silver")
        )
        lines.append(tree["last"] + "translated_axes_coordinates")
        return render(lines)


class MdsDisplay(dict):
    """One display bundle of a Biplot (port of ``bipl5_mdsDisplay``)."""

    def __repr__(self) -> str:
        return render(
            format_display_lines(self, tree_symbols(), _painter(_use_color()))
        )


class FitMeasures(dict):
    """The fit-measure collection of a Biplot (port of ``bipl5_fitmeasures``)."""

    def __repr__(self) -> str:
        tree, paint = tree_symbols(), _painter(_use_color())
        lines = [paint("bipl5_fitmeasures", "bold", "yellow")]
        lines.extend(format_fitmeasures_subtree(self, "", tree, paint))
        return render(lines)
