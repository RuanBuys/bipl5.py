"""Hover-table generation — port of the hovertext part of
``PCAbiplot_Helper.R``.

Each observation's hover shows a fixed-width, pipe-delimited table of
Actual vs Pred(icted) values per variable (plus an optional Alves Error
column added by ``score_axes()``), rendered in a monospace hover font.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["hovertext_generator", "format_hover_values", "format_reading_error"]


def _decimals_needed(value: float, digits: int) -> int:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def format_hover_values(values: np.ndarray, digits: int = 4) -> list[list[str]]:
    """Row-wise number formatting like R's ``format(round(x, 4), trim=TRUE)``.

    R formats each row as a vector, giving every entry in a row the same
    number of decimal places (the most needed by any entry, after rounding).
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    out = []
    for row in values:
        rounded = [round(float(v), digits) for v in row]
        dec = max((_decimals_needed(v, digits) for v in rounded), default=0)
        out.append([f"{v:.{dec}f}" for v in rounded])
    return out


def format_reading_error(values: np.ndarray, digits: int = 2) -> list[list[str]]:
    """Format Alves reading errors as percentages; dashes for undefined."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    out = []
    for row in values:
        out.append(
            [f"{v:.{digits}f}%" if math.isfinite(v) else "-" for v in row]
        )
    return out


def _center_pad(value: str, width: int) -> str:
    pad = max(width - len(value), 0)
    left = pad // 2
    return " " * left + value + " " * (pad - left)


def hovertext_generator(obj: dict, i: int, linebreak: str = "\n") -> list[str]:
    """Port of ``hovertext_generator()``: hover strings for group level ``i``.

    ``obj`` mirrors the R list: ``Z``, ``group_codes``/``group_levels``,
    ``n``, ``x`` (raw data matrix), ``row_names``, ``col_names``, ``XHat``,
    ``sample_predictivity`` and optionally ``reading_error`` /
    ``reading_error_digits``.
    """
    codes = np.asarray(obj["group_codes"])
    sel = np.where(codes == i)[0]
    row_names = obj["row_names"]

    if obj.get("XHat") is None:
        return [row_names[j] for j in sel]
    if sel.size == 0:
        return []

    x = np.asarray(obj["x"], dtype=float)
    xhat = np.asarray(obj["XHat"], dtype=float)
    var_names = list(obj["col_names"])
    sample_pred = obj.get("sample_predictivity")

    actual_rows = format_hover_values(x[sel, :])
    pred_rows = format_hover_values(xhat[sel, :])
    name_width = max(len(v) for v in var_names) + 1

    reading_error = obj.get("reading_error")
    show_error = reading_error is not None
    if show_error:
        err_rows = format_reading_error(
            np.asarray(reading_error)[sel, :],
            digits=obj.get("reading_error_digits", 2),
        )

    out = []
    for pos, j in enumerate(sel):
        actual_row = actual_rows[pos]
        pred_row = pred_rows[pos]
        actual_width = max(len(s) for s in ["Actual", *actual_row]) + 2
        pred_width = max(len(s) for s in ["Pred", *pred_row]) + 2
        if show_error:
            err_row = err_rows[pos]
            err_width = max(len(s) for s in ["Error", *err_row]) + 2

        head1 = (
            "|"
            + " " * name_width
            + "|"
            + _center_pad("Actual", actual_width)
            + "|"
            + _center_pad("Pred", pred_width)
            + "|"
        )
        head2 = (
            "|:"
            + "-" * (name_width - 1)
            + "|:"
            + "-" * (actual_width - 2)
            + ":|:"
            + "-" * (pred_width - 2)
            + ":|"
        )
        if show_error:
            head1 += _center_pad("Error", err_width) + "|"
            head2 += ":" + "-" * (err_width - 2) + ":|"

        body = []
        for k, name in enumerate(var_names):
            line = (
                "|"
                + f"{name:<{name_width}}"
                + "|"
                + _center_pad(actual_row[k], actual_width)
                + "|"
                + _center_pad(pred_row[k], pred_width)
                + "|"
            )
            if show_error:
                line += _center_pad(err_row[k], err_width) + "|"
            body.append(line)

        table = "".join(part + linebreak for part in [head1, head2, *body])
        text = f"Observation: {row_names[j]}{linebreak}{linebreak}{table}"
        if sample_pred is not None and len(sample_pred) > j:
            text += (
                f"{linebreak}Sample predictivity: "
                f"{float(sample_pred[j]):.4f}"
            )
        out.append(text)
    return out
