"""mdsDisplay container helpers — port of ``mdsDisplay_constructor.R``.

An mdsDisplay is a plain dict with plotly-compatible pieces:
``{"trace_data": [...], "layout": {"annotations": [...]}, "config": {...}}``.
"""

from __future__ import annotations

__all__ = [
    "mds_display_new",
    "mds_display_add_traces",
    "mds_display_add_layout",
    "mds_display_add_config",
]


def mds_display_new() -> dict:
    return {"trace_data": [], "layout": {"annotations": []}, "config": {}}


def mds_display_add_traces(mds: dict, traces: list) -> dict:
    mds["trace_data"] = list(mds.get("trace_data", [])) + list(traces)
    return mds


def _modify_list(base: dict, new: dict) -> dict:
    """R ``utils::modifyList()``: recursive dict update."""
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _modify_list(out[key], value)
        else:
            out[key] = value
    return out


def mds_display_add_layout(mds: dict, layout: dict) -> dict:
    current = mds.get("layout") or {}
    for name, value in layout.items():
        if name == "annotations":
            current["annotations"] = list(current.get("annotations", [])) + list(
                value or []
            )
        elif isinstance(value, dict) and isinstance(current.get(name), dict):
            current[name] = _modify_list(current[name], value)
        else:
            current[name] = value
    mds["layout"] = current
    return mds


def mds_display_add_config(mds: dict, config: dict) -> dict:
    mds["config"] = _modify_list(mds.get("config") or {}, config)
    return mds
