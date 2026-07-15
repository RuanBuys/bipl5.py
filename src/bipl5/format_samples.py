"""Sample formatting — port of ``format_samples.R``.

``format_samples()`` rebuilds the sample-trace block inside every mdsDisplay
so observations are grouped by ``by`` and rendered with one trace per visual
class. A first call creates one legend section for the requested aesthetic
(``stratify="col"`` or ``"symbol"``); a second call with a *different*
grouping variable activates **dual stratification**: the observation layer is
split into hidden colour x symbol combination traces and the legend shows two
independent sections whose entries toggle across the other stratification.

Colour stratification also rebuilds the kernel densities on the translated
axes to follow the colour classes; symbol stratification leaves them alone.
The stored state is replayed by ``append_mds_display()`` so later displays
inherit the same formatting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .display.builders import add_tda, slider_control
from .symbols import (
    colorpal,
    pch_to_plotly_strict,
    plotly_to_pch,
    validate_symbols,
)

__all__ = ["format_samples"]


# ── small helpers ───────────────────────────────────────────────────────────

def _or(x, y):
    return y if x is None else x


def _has_meta(item: dict, key: str) -> bool:
    meta = item.get("meta")
    if meta is None:
        return False
    if isinstance(meta, str):
        return meta == key
    return key in list(meta)


def _as_group(values) -> dict:
    """R factor semantics for a grouping vector.

    pandas Categoricals keep their category order (with unused categories
    dropped, like ``droplevels()``); anything else gets levels in order of
    first appearance, matching ``factor(x, levels = unique(x))``.
    """
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        values = values.array
    if isinstance(values, pd.Categorical):
        values = values.remove_unused_categories()
        if values.isna().any():
            raise ValueError("Missing values are not supported in 'by'.")
        labels = np.asarray([str(v) for v in values])
        levels = [str(c) for c in values.categories]
        return {"labels": labels, "levels": levels}

    arr = list(values)
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in arr):
        raise ValueError("Missing values are not supported in 'by'.")
    labels = np.asarray([str(v) for v in arr])
    levels = list(dict.fromkeys(labels.tolist()))
    return {"labels": labels, "levels": levels}


def _groups_identical(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return False
    la, lb = a["labels"], b["labels"]
    return len(la) == len(lb) and bool(np.all(la == lb))


def _hcl_colors_dark3(k: int) -> list[str]:
    """Approximation of ``grDevices::hcl.colors(k, "Dark 3")`` for k > 16:
    a qualitative HCL palette with evenly spaced hues at C=80, L=60."""
    def hcl_to_hex(h_deg, c, l):
        u = c * math.cos(math.radians(h_deg))
        v = c * math.sin(math.radians(h_deg))
        y = 100 * ((l + 16) / 116) ** 3 if l > 8 else 100 * l / 903.3
        un, vn = 0.1978398, 0.4683363
        up = u / (13 * l) + un
        vp = v / (13 * l) + vn
        x = y * 9 * up / (4 * vp)
        z = y * (12 - 3 * up - 20 * vp) / (4 * vp)
        x, y2, z = x / 100, y / 100, z / 100
        r = 3.2404542 * x - 1.5371385 * y2 - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y2 + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y2 + 1.0572252 * z

        def enc(ch):
            ch = min(max(ch, 0.0), 1.0)
            ch = 1.055 * ch ** (1 / 2.4) - 0.055 if ch > 0.0031308 else 12.92 * ch
            return round(min(max(ch, 0.0), 1.0) * 255)

        return f"#{enc(r):02X}{enc(g):02X}{enc(b):02X}"

    hues = [360 * i * (k - 1) / (k * max(k - 1, 1)) for i in range(k)]
    return [hcl_to_hex(h, 80, 60) for h in hues]


# ── state management ────────────────────────────────────────────────────────

def _get_data(bp) -> pd.DataFrame:
    spec = bp.meta.get("spec")
    if spec is not None and getattr(spec, "data", None) is not None:
        return spec.data
    ez = bp.meta["x"]
    return pd.DataFrame(ez.raw_X, columns=ez.col_names)


def _default_aes(bp) -> dict:
    ez = bp.meta["x"]
    color = bp.meta.get("color")
    symbol = bp.meta.get("symbol")
    pch = (ez.samples or {}).get("pch")
    return {
        "color": color[0] if color else colorpal(1)[0],
        "symbol": symbol[0] if symbol else "circle",
        "pch_numeric": int(pch[0]) if pch else 19,
    }


def _current_group(bp, n: int) -> dict:
    codes, levels = bp.meta.get("group", (None, None))
    if codes is None or len(codes) != n:
        return {"labels": np.asarray(["Data"] * n), "levels": ["Data"]}
    labels = np.asarray([levels[c] for c in codes])
    used = [lev for lev in levels if lev in set(labels.tolist())]
    return {"labels": labels, "levels": used}


def _get_state(bp, n: int) -> dict:
    defaults = _default_aes(bp)
    current = bp.meta.get("sample_format")
    if isinstance(current, dict) and current.get("version") == 2:
        state = dict(current)
        state["defaults"] = defaults
        state.setdefault("order", [])
        return state
    return {
        "version": 2,
        "order": [],
        "color": None,
        "symbol": None,
        "defaults": defaults,
    }


def _primary_kind(state: dict) -> str | None:
    for kind in state["order"]:
        if kind in ("color", "symbol") and state.get(kind) is not None:
            return kind
    return None


def _section_order(state: dict) -> list[str]:
    return [
        kind
        for kind in state["order"]
        if kind in ("color", "symbol") and state.get(kind) is not None
    ]


def _has_dual(state: dict) -> bool:
    color, symbol = state.get("color"), state.get("symbol")
    if color is None or symbol is None:
        return False
    return not _groups_identical(color["group"], symbol["group"])


def _effective_spec(state: dict, kind: str, n: int) -> dict:
    spec = state.get(kind)
    if spec is not None:
        return spec
    group = {"labels": np.asarray(["Data"] * n), "levels": ["Data"]}
    base = {
        "group": group,
        "levels": ["Data"],
        "legend_title": "Data",
        "source": "default",
    }
    if kind == "color":
        return {**base, "values": [state["defaults"]["color"]]}
    return {
        **base,
        "values": [state["defaults"]["symbol"]],
        "pch_numeric": [state["defaults"]["pch_numeric"]],
    }


def _unified_spec(state: dict, n: int) -> dict:
    primary = _primary_kind(state) or "color"
    primary_spec = _effective_spec(state, primary, n)
    group = primary_spec["group"]
    k = len(group["levels"])

    color_spec, symbol_spec = state.get("color"), state.get("symbol")
    if color_spec is not None and _groups_identical(color_spec["group"], group):
        colors = color_spec["values"]
    else:
        colors = [state["defaults"]["color"]] * k
    if symbol_spec is not None and _groups_identical(symbol_spec["group"], group):
        symbols = symbol_spec["values"]
        pch_numeric = symbol_spec["pch_numeric"]
    else:
        symbols = [state["defaults"]["symbol"]] * k
        pch_numeric = [state["defaults"]["pch_numeric"]] * k

    return {
        "group": group,
        "levels": group["levels"],
        "colors": list(colors),
        "symbols": list(symbols),
        "pch_numeric": [int(v) for v in pch_numeric],
        "legend_title": primary_spec["legend_title"],
    }


# ── grouping / aesthetics resolution ────────────────────────────────────────

def _resolve_grouping(bp, state, stratify, by, data, n) -> dict:
    kind = "color" if stratify == "col" else "symbol"
    current_spec = state.get(kind)
    fallback = current_spec
    if fallback is None:
        primary = _primary_kind(state)
        fallback = state.get(primary) if primary else None

    if by is None:
        if fallback is not None:
            group = fallback["group"]
            legend_title = fallback["legend_title"]
        else:
            group = _current_group(bp, n)
            legend_title = "Data"
        return {
            "group": group,
            "levels": group["levels"],
            "source": "existing grouping",
            "legend_title": legend_title,
        }

    if isinstance(by, str):
        if by not in data.columns:
            raise ValueError(
                f"Column '{by}' was not found in the data stored by this "
                "object. Supply the grouping variable directly as a vector "
                "if it is not part of the dataset passed to init_biplot()."
            )
        group = _as_group(data[by])
        return {
            "group": group,
            "levels": group["levels"],
            "source": by,
            "legend_title": by,
        }

    try:
        length = len(by)
    except TypeError:
        length = -1
    if length != n:
        raise ValueError(
            f"'by' must resolve to a stored column name or a vector of "
            f"length {n}."
        )

    # a named pandas Series matching a stored column plays the role of R's
    # bare-symbol `by`: the name becomes the legend section title
    name = by.name if isinstance(by, pd.Series) else None
    if name is not None and str(name) in data.columns:
        source = legend_title = str(name)
    else:
        source, legend_title = "supplied vector", "Data"

    group = _as_group(by)
    return {
        "group": group,
        "levels": group["levels"],
        "source": source,
        "legend_title": legend_title,
    }


def _resolve_target_aes(stratify, group, col, pch) -> dict:
    k = len(group["levels"])
    if stratify == "col":
        if col is None:
            values = colorpal(k) if k <= 16 else _hcl_colors_dark3(k)
        else:
            values = list(col)
            if len(values) != k:
                raise ValueError(f"Expected {k} colours, got {len(values)}.")
        return {"values": values}

    if pch is None:
        raise ValueError("'pch' is required when stratify = 'symbol'.")
    pch = list(pch)
    if len(pch) != k:
        raise ValueError(f"Expected {k} plotting symbols, got {len(pch)}.")

    if all(isinstance(v, (int, np.integer, float, np.floating)) for v in pch):
        pch_numeric = [int(v) for v in pch]
        symbols = pch_to_plotly_strict(pch_numeric)
    else:
        symbols = [str(v) for v in pch]
        invalid = validate_symbols(symbols)
        if invalid:
            raise ValueError(
                "Invalid plotly symbols: " + ", ".join(invalid)
            )
        pch_numeric = plotly_to_pch(symbols)
    return {"values": symbols, "pch_numeric": pch_numeric}


def _update_state(state, stratify, group_info, aes_info) -> dict:
    kind = "color" if stratify == "col" else "symbol"
    spec = {
        "group": group_info["group"],
        "levels": group_info["levels"],
        "values": list(aes_info["values"]),
        "legend_title": group_info["legend_title"],
        "source": group_info["source"],
    }
    if kind == "symbol":
        spec["pch_numeric"] = list(aes_info["pch_numeric"])
    state = dict(state)
    state[kind] = spec
    if kind not in state["order"]:
        state["order"] = [*state["order"], kind]
    return state


# ── trace rebuilding ────────────────────────────────────────────────────────

def _collect_sample_points(traces: list[dict]) -> dict:
    obs, xs, ys, hovers = [], [], [], []
    for tr in traces:
        idx = np.atleast_1d(np.asarray(tr.get("customdata"))).astype(int)
        hover = tr.get("hovertext")
        hover = [hover] if isinstance(hover, str) else list(hover)
        if len(hover) == 1 and idx.size > 1:
            hover = hover * idx.size
        obs.append(idx)
        xs.append(np.atleast_1d(np.asarray(tr["x"], dtype=float)))
        ys.append(np.atleast_1d(np.asarray(tr["y"], dtype=float)))
        hovers.extend(hover)

    obs = np.concatenate(obs)
    order = np.argsort(obs, kind="stable")
    return {
        "obs_idx": obs[order],
        "x": np.concatenate(xs)[order],
        "y": np.concatenate(ys)[order],
        "hovertext": [hovers[i] for i in order],
    }


def _legend_title(title) -> dict:
    if not isinstance(title, str) or not title:
        title = "Data"
    return {"text": f"<b>{title}</b>"}


def _labels_at(group: dict, obs_idx: np.ndarray) -> np.ndarray:
    return group["labels"][obs_idx - 1]  # obs_idx is 1-based


def _build_unified_traces(points, unified, template) -> list[dict]:
    template_marker = template.get("marker") or {"opacity": 1}
    labels = _labels_at(unified["group"], points["obs_idx"])
    traces = []
    for i, lev in enumerate(unified["levels"]):
        sel = labels == lev
        marker = dict(template_marker)
        marker["color"] = unified["colors"][i]
        marker["symbol"] = unified["symbols"][i]
        traces.append(
            {
                "x": points["x"][sel],
                "y": points["y"][sel],
                "name": lev,
                "type": _or(template.get("type"), "scatter"),
                "mode": _or(template.get("mode"), "markers"),
                "hovertext": [
                    h for h, s in zip(points["hovertext"], sel) if s
                ],
                "hoverinfo": _or(template.get("hoverinfo"), "text+name"),
                "customdata": points["obs_idx"][sel],
                "meta": ["data", f"group:{lev}"],
                "xaxis": _or(template.get("xaxis"), "x"),
                "yaxis": _or(template.get("yaxis"), "y"),
                "visible": _or(template.get("visible"), True),
                "showlegend": True,
                "marker": marker,
                "legendgroup": "data",
                "legendgrouptitle": _legend_title(unified["legend_title"]),
            }
        )
    return traces


def _build_combo_traces(points, color_spec, symbol_spec, template) -> list[dict]:
    template_marker = template.get("marker") or {"opacity": 1}
    color_labels = _labels_at(color_spec["group"], points["obs_idx"])
    symbol_labels = _labels_at(symbol_spec["group"], points["obs_idx"])

    traces = []
    for i, color_level in enumerate(color_spec["levels"]):
        for j, symbol_level in enumerate(symbol_spec["levels"]):
            sel = (color_labels == color_level) & (symbol_labels == symbol_level)
            if not sel.any():
                continue
            marker = dict(template_marker)
            marker["color"] = color_spec["values"][i]
            marker["symbol"] = symbol_spec["values"][j]
            traces.append(
                {
                    "x": points["x"][sel],
                    "y": points["y"][sel],
                    "name": f"{color_level} | {symbol_level}",
                    "type": _or(template.get("type"), "scatter"),
                    "mode": _or(template.get("mode"), "markers"),
                    "hovertext": [
                        h for h, s in zip(points["hovertext"], sel) if s
                    ],
                    "hoverinfo": _or(template.get("hoverinfo"), "text+name"),
                    "customdata": points["obs_idx"][sel],
                    "meta": [
                        "data",
                        "sample-combo",
                        f"color:{color_level}",
                        f"symbol:{symbol_level}",
                    ],
                    "xaxis": _or(template.get("xaxis"), "x"),
                    "yaxis": _or(template.get("yaxis"), "y"),
                    "visible": _or(template.get("visible"), True),
                    "showlegend": False,
                    "marker": marker,
                    "legendgroup": "data",
                }
            )
    return traces


def _build_legend_traces(kind, spec, state, template) -> list[dict]:
    k = len(spec["levels"])
    if kind == "color":
        marker_symbol = [state["defaults"]["symbol"]] * k
        marker_color = spec["values"]
    else:
        marker_symbol = spec["values"]
        marker_color = ["black"] * k

    traces = []
    for i, lev in enumerate(spec["levels"]):
        trace = {
            "x": [None],
            "y": [None],
            "name": lev,
            "type": _or(template.get("type"), "scatter"),
            "mode": _or(template.get("mode"), "markers"),
            "hoverinfo": "skip",
            "showlegend": True,
            "visible": True,
            "marker": {
                "color": marker_color[i],
                "symbol": marker_symbol[i],
                "opacity": 1,
            },
            "meta": ["sample-legend", kind, f"{kind}:{lev}"],
            "xaxis": _or(template.get("xaxis"), "x"),
            "yaxis": _or(template.get("yaxis"), "y"),
            "legendgroup": f"sample-legend-{kind}",
        }
        if i == 0:
            trace["legendgrouptitle"] = _legend_title(spec["legend_title"])
        traces.append(trace)
    return traces


def _build_sample_traces(points, state, template) -> list[dict]:
    n = int(points["obs_idx"].max()) if points["obs_idx"].size else 0
    color_spec = _effective_spec(state, "color", n)
    symbol_spec = _effective_spec(state, "symbol", n)

    if _has_dual(state):
        out = []
        for kind in _section_order(state):
            spec = color_spec if kind == "color" else symbol_spec
            out.extend(_build_legend_traces(kind, spec, state, template))
        out.extend(
            _build_combo_traces(points, color_spec, symbol_spec, template)
        )
        return out

    return _build_unified_traces(points, _unified_spec(state, n), template)


def _mean_spec(state, n) -> dict:
    if _has_dual(state) and state.get("color") is not None:
        color = state["color"]
        return {
            "group": color["group"],
            "levels": color["levels"],
            "colors": color["values"],
            "symbols": [state["defaults"]["symbol"]] * len(color["levels"]),
        }
    unified = _unified_spec(state, n)
    return {
        "group": unified["group"],
        "levels": unified["levels"],
        "colors": unified["colors"],
        "symbols": unified["symbols"],
    }


def _build_mean_traces(points, mean_spec, template) -> list[dict]:
    template = template or {}
    template_marker = template.get("marker") or {"size": 10}
    labels = _labels_at(mean_spec["group"], points["obs_idx"])

    traces = []
    for i, lev in enumerate(mean_spec["levels"]):
        sel = labels == lev
        marker = dict(template_marker)
        marker["color"] = mean_spec["colors"][i]
        marker["symbol"] = mean_spec["symbols"][i]
        traces.append(
            {
                "x": [float(np.mean(points["x"][sel]))],
                "y": [float(np.mean(points["y"][sel]))],
                "name": lev,
                "type": _or(template.get("type"), "scatter"),
                "mode": _or(template.get("mode"), "markers"),
                "hovertext": "Class Mean",
                "hoverinfo": _or(template.get("hoverinfo"), "text+name"),
                "customdata": i,  # 0-based id, kept from R
                "meta": ["ClassMean"],
                "xaxis": _or(template.get("xaxis"), "x"),
                "yaxis": _or(template.get("yaxis"), "y"),
                "visible": _or(template.get("visible"), True),
                "showlegend": False,
                "marker": marker,
                "legendgroup": "ClassMean",
            }
        )
    return traces


def _replace_trace_blocks(
    traces, sample_idx, rebuilt_samples, mean_idx, rebuilt_means
) -> list[dict]:
    sample_first = min(sample_idx) if sample_idx else math.inf
    mean_first = min(mean_idx) if mean_idx else math.inf
    drop = set(sample_idx) | set(mean_idx)

    out: list[dict] = []
    for i, trace in enumerate(traces):
        if i == sample_first:
            out.extend(rebuilt_samples)
        if i == mean_first:
            out.extend(rebuilt_means)
        if i in drop:
            continue
        out.append(trace)
    return out


def _rebuild_tda_layer(bundle: dict, state: dict, ez) -> dict:
    color_spec = state.get("color")
    if color_spec is None or ez is None:
        return bundle
    z_axes = bundle["Data"].get("axes_coordinates")
    Z = bundle["Data"].get("sample_coordinates")
    if z_axes is None or Z is None:
        return bundle

    mds = bundle["mds"]
    traces = mds["trace_data"]
    keep = [
        tr
        for tr in traces
        if not (_has_meta(tr, "ExpAx") or _has_meta(tr, "density"))
    ]
    if len(keep) == len(traces):
        return bundle
    # copy every layer we (or add_tda/slider_control) mutate, so the input
    # Biplot keeps R's copy-on-modify semantics
    mds = dict(mds)
    mds["trace_data"] = keep
    mds["layout"] = dict(mds.get("layout") or {})
    mds["config"] = dict(mds.get("config") or {})
    annotations = mds["layout"].get("annotations")
    if annotations is not None:
        mds["layout"]["annotations"] = [
            ann for ann in annotations if not _has_meta(ann, "ExpAx")
        ]

    tda_ez = ez._copy()
    tda_ez.Z = np.asarray(Z, dtype=float)
    levels = color_spec["group"]["levels"]
    lookup = {lev: i for i, lev in enumerate(levels)}
    codes = np.asarray([lookup[v] for v in color_spec["group"]["labels"]])

    tda_out = add_tda(
        mds=mds,
        z_axes=z_axes,
        ez=tda_ez,
        Z=tda_ez.Z,
        group_codes=codes,
        group_levels=levels,
        col=color_spec["values"],
    )

    bundle = dict(bundle)
    bundle["mds"] = tda_out["mds"]
    bundle["m"] = tda_out["m"]
    bundle["shift"] = tda_out["shift"]
    bundle["Data"] = dict(bundle["Data"])
    bundle["Data"]["translated_axes_coordinates"] = tda_out["shift"]
    slider_control(
        {"mds": bundle["mds"], "m": tda_out["m"], "shift": tda_out["shift"]},
        n_inside=17,
        n_outside=4,
    )
    return bundle


def rebuild_mds_display(bundle, update_means, state, ez=None, rebuild_tda=False):
    """Port of ``format_samples_rebuild_mdsDisplay()`` for one display."""
    traces = bundle["mds"]["trace_data"]
    sample_idx = [i for i, tr in enumerate(traces) if _has_meta(tr, "data")]
    legend_idx = [
        i for i, tr in enumerate(traces) if _has_meta(tr, "sample-legend")
    ]
    if not sample_idx:
        return bundle

    points = _collect_sample_points([traces[i] for i in sample_idx])
    rebuilt_samples = _build_sample_traces(
        points, state, template=traces[sample_idx[0]]
    )

    mean_idx: list[int] = []
    rebuilt_means: list[dict] = []
    if update_means:
        spec = _mean_spec(state, len(points["obs_idx"]))
        mean_idx = [
            i for i, tr in enumerate(traces) if _has_meta(tr, "ClassMean")
        ]
        rebuilt_means = _build_mean_traces(
            points, spec, template=traces[mean_idx[0]] if mean_idx else None
        )

    bundle = dict(bundle)
    bundle["mds"] = dict(bundle["mds"])
    bundle["mds"]["trace_data"] = _replace_trace_blocks(
        traces, sample_idx + legend_idx, rebuilt_samples, mean_idx, rebuilt_means
    )

    if rebuild_tda:
        bundle = _rebuild_tda_layer(bundle, state, ez)
    return bundle


def should_update_means(bp, state) -> bool:
    return bool(bp.meta["x"].class_means) and bp.biplot_type != "cva"


def get_state(bp, n: int) -> dict:
    """Public wrapper used by ``append_mds_display()`` to replay state."""
    return _get_state(bp, n)


# ── metadata update ─────────────────────────────────────────────────────────

def _compat_meta(state, bp, n) -> dict:
    primary = _primary_kind(state)
    if primary is None:
        group = _current_group(bp, n)
        k = len(group["levels"])
        return {
            "group": group,
            "color": [state["defaults"]["color"]] * k,
            "symbol": [state["defaults"]["symbol"]] * k,
            "pch_numeric": [state["defaults"]["pch_numeric"]] * k,
            "legend_title": "Data",
        }

    primary_spec = state[primary]
    group = primary_spec["group"]
    k = len(group["levels"])

    if primary == "color":
        color = primary_spec["values"]
        symbol_spec = state.get("symbol")
        if symbol_spec is not None and _groups_identical(
            symbol_spec["group"], group
        ):
            symbol = symbol_spec["values"]
            pch_numeric = symbol_spec["pch_numeric"]
        else:
            symbol = [state["defaults"]["symbol"]] * k
            pch_numeric = [state["defaults"]["pch_numeric"]] * k
    else:
        symbol = primary_spec["values"]
        pch_numeric = primary_spec["pch_numeric"]
        color_spec = state.get("color")
        if color_spec is not None and _groups_identical(
            color_spec["group"], group
        ):
            color = color_spec["values"]
        else:
            color = [state["defaults"]["color"]] * k

    return {
        "group": group,
        "color": list(color),
        "symbol": list(symbol),
        "pch_numeric": [int(v) for v in pch_numeric],
        "legend_title": primary_spec["legend_title"],
    }


def _update_meta(bp, state):
    is_cva = bp.biplot_type == "cva"
    ez = bp.meta["x"]._copy()
    compat = _compat_meta(state, bp, ez.n)
    mean_spec = _mean_spec(state, ez.n)

    group = compat["group"]
    lookup = {lev: i for i, lev in enumerate(group["levels"])}
    codes = np.asarray([lookup[v] for v in group["labels"]])

    bp.meta["group"] = (codes, list(group["levels"]))
    bp.meta["color"] = compat["color"]
    bp.meta["symbol"] = compat["symbol"]
    bp.meta["sample_format"] = state

    if is_cva:
        # preserve the fitted CVA classes; only the display grouping changes
        if bp.meta.get("model_group") is None:
            bp.meta["model_group"] = (ez.group, list(ez.g_names))
        ez.display_group = (codes, list(group["levels"]))
    else:
        ez.group = codes
        ez.g_names = list(group["levels"])
        ez.g = len(group["levels"])

    samples = dict(ez.samples or {})
    samples["col"] = compat["color"]
    samples["pch"] = compat["pch_numeric"]
    samples["plotly_symbol"] = compat["symbol"]
    ez.samples = samples

    if should_update_means(bp, state):
        means_aes = dict(ez.means_aes or {})
        means_aes["col"] = mean_spec["colors"]
        means_aes["pch"] = [state["defaults"]["pch_numeric"]] * len(
            mean_spec["levels"]
        )
        means_aes["plotly_symbol"] = mean_spec["symbols"]
        ez.means_aes = means_aes

    bp.meta["x"] = ez
    return bp


# ── public entry point ──────────────────────────────────────────────────────

def format_samples(bp, stratify: str = "col", by=None, col=None, pch=None):
    """Reformat the sample layer of a Biplot (port of ``format_samples()``).

    Parameters
    ----------
    bp:
        A :class:`~bipl5.biplot.Biplot` created by ``scale_mds()``.
    stratify:
        ``"col"`` to change marker colours or ``"symbol"`` to change marker
        symbols.
    by:
        The grouping: a column name stored in the data passed to
        ``init_biplot()``, an array-like of length ``n`` (a named pandas
        Series whose name matches a stored column labels the legend section
        with that name), or ``None`` to reuse the current grouping.
    col:
        Optional list of colours, one per class (``stratify="col"``).
    pch:
        Plotting symbols, one per class (required for
        ``stratify="symbol"``): base-R numeric pch codes or plotly symbol
        names.

    Calling twice with different ``by`` variables activates dual
    stratification with independent colour and symbol legend sections.
    Colour calls also rebuild the translated-axis densities to follow the
    colour classes.
    """
    if stratify not in ("col", "symbol"):
        raise ValueError("stratify must be 'col' or 'symbol'.")

    data = _get_data(bp)
    n = bp.meta["x"].n
    state = _get_state(bp, n)

    group_info = _resolve_grouping(bp, state, stratify, by, data, n)
    aes_info = _resolve_target_aes(stratify, group_info["group"], col, pch)
    state = _update_state(state, stratify, group_info, aes_info)

    update_means = should_update_means(bp, state)

    out = bp._copy()
    out.displays = {
        name: rebuild_mds_display(
            bundle,
            update_means=update_means,
            state=state,
            ez=bp.meta["x"],
            rebuild_tda=(stratify == "col"),
        )
        for name, bundle in bp.displays.items()
    }
    return _update_meta(out, state)
