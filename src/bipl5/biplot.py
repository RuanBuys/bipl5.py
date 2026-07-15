"""The ``Biplot`` object and its verbs — port of the retained parts of
``wrap_bipl5.R``: constructors, ``plot()``, ``extract()``,
``append_mdsDisplay()``, ``remove_mdsDisplay()`` and ``overlay_fit()``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .display.build_one import build_one_mds_display, restore_raw_x
from .display.builders import fit_table_traces

__all__ = ["Biplot", "BiplotFit", "mds_display_name", "pair_label", "ft_name"]


# ── naming helpers ──────────────────────────────────────────────────────────

def mds_display_name(pcs) -> str:
    """``(1, 2) -> "mdsDisplay_12"`` — the canonical display key."""
    return f"mdsDisplay_{pcs[0]}{pcs[1]}"


def pair_label(pcs, prefix: str = "PC") -> str:
    """``(1, 3) -> "PC 1 & 3"`` — the user-facing dropdown label."""
    return f"{prefix} {pcs[0]} & {pcs[1]}"


def ft_name(pcs) -> str:
    """``(1, 2) -> "fit_table_12"`` — the fit-table storage key."""
    return f"fit_table_{pcs[0]}{pcs[1]}"


FIT_GRAPH_NAMES = ("CumPred", "CumAd", "VarExp", "Scree")


# ── fit display configuration (port of fit_display_config etc.) ────────────

def fit_display_config(mode: str = "panel") -> dict:
    return {
        "mode": mode,
        "panel": {
            "xaxis_domain": [0, 0.5],
            "xaxis3_domain": [0.65, 1],
            "yaxis3_domain": [0.15, 0.85],
            "yaxis3_position": 0.65,
            "yaxis3_side": "left",
            "table_domain_x": [0.5, 1],
            "table_domain_y": [0.15, 0.85],
            "slider_len": 0.5,
            "menu_pad_right": 0,
        },
        "overlay": {
            "xaxis_domain": [0, 1],
            "xaxis3_domain": [0, 1],
            "yaxis3_domain": [0.15, 0.85],
            "yaxis3_position": 0,
            "yaxis3_side": "left",
            "table_domain_x": [0, 1],
            "table_domain_y": [0.15, 0.85],
            "slider_len": 1,
            "menu_pad_right": 60,
        },
    }


class BiplotFit:
    """A single extracted fit graph (port of ``bipl5_fit``).

    ``plot()`` renders the stored traces as a standalone figure.
    """

    _LABELS = {
        "CumPred": ("Cumulative axis predictivity", "Dimension of Subspace"),
        "CumAd": ("Cumulative axis adequacy", "Dimension of Subspace"),
        "VarExp": ("Variance explained", "Principal component"),
        "Scree": ("Scree plot", "Principal component"),
    }

    def __init__(self, trace_data: list, fit_name: str | None = None):
        self.trace_data = trace_data
        self.fit_name = fit_name

    def plot(self):
        from .render.widget import Bipl5Widget

        title, x_title = self._LABELS.get(self.fit_name, (self.fit_name, ""))
        traces = []
        for tr in self.trace_data:
            tr = dict(tr)
            tr.pop("xaxis", None)
            tr.pop("yaxis", None)
            traces.append(tr)
        figure = {
            "data": traces,
            "layout": {
                "title": title,
                "xaxis": {"title": x_title, "dtick": 1},
                "barmode": "stack",
            },
        }
        return Bipl5Widget(figure, payload=None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<BiplotFit {self.fit_name}: {len(self.trace_data)} traces>"


class Biplot:
    """Port of the ``bipl5_biplot`` S3 object.

    Holds one or more mdsDisplay bundles, optional fit measures, and the
    metadata needed by ``plot()`` and the manipulation verbs. Created by
    :meth:`bipl5.BiplotSpec.scale_mds`.
    """

    def __init__(
        self,
        displays: dict[str, dict],
        fit_measures: dict | None,
        meta: dict,
        biplot_type: str = "pca",
    ):
        self.displays = displays
        self.fit_measures = fit_measures
        self.meta = meta
        self.biplot_type = biplot_type

    # dict-like access to displays, mirroring R's bp$mdsDisplay_12
    def __getitem__(self, key: str) -> dict:
        return self.displays[key]

    def _copy(self) -> "Biplot":
        new = Biplot(
            dict(self.displays),
            copy.copy(self.fit_measures) if self.fit_measures else None,
            copy.deepcopy(
                {k: v for k, v in self.meta.items() if k not in ("x", "spec")}
            ),
            self.biplot_type,
        )
        new.meta["x"] = self.meta["x"]
        if "spec" in self.meta:
            new.meta["spec"] = self.meta["spec"]
        return new

    # ── plotting ────────────────────────────────────────────────────────

    def _resolve_fit_display_mode(self, fit_display: str) -> str:
        if fit_display == "inherit":
            stored = (self.meta.get("plot_options") or {}).get(
                "fit_display", "panel"
            )
            fit_display = stored
        if fit_display not in ("panel", "overlay"):
            raise ValueError("fit_display must be 'inherit', 'panel' or 'overlay'.")
        if self.fit_measures is None:
            return "panel"
        return fit_display

    def plot(self, fit_display: str = "inherit"):
        """Port of ``plot.bipl5_biplot()``: build the interactive widget.

        Renders the first available mdsDisplay into a plotly figure and
        attaches the remaining displays plus fit measures as the JSON
        payload consumed by the vendored bipl5 JavaScript.
        """
        from .render.figure import figure_scaffolding
        from .render.widget import Bipl5Widget

        ez = self.meta["x"]
        pc_info = self.meta["pc_info"]
        has_fm = self.fit_measures is not None
        is_cva = self.biplot_type == "cva"
        is_reg = self.biplot_type == "reg"
        is_pco = self.biplot_type == "pco"
        is_spline = bool(self.meta.get("spline"))
        mode = self._resolve_fit_display_mode(fit_display)
        cfg = fit_display_config(mode)

        available = [k for k in pc_info if k in self.displays]
        pc_map = {k: pc_info[k]["label"] for k in available}
        ft_map = {k: pc_info[k]["ft_name"] for k in available}
        use_pc_toggle = len(available) > 1

        first_name = available[0]
        first = self.displays[first_name]

        dpquality = first["fit_qual"]
        if is_reg and self.meta.get("fit_quality_plotly"):
            dpquality = self.meta["fit_quality_plotly"]

        layout = figure_scaffolding(
            dpquality=dpquality,
            PC_toggle=use_pc_toggle,
            ax_pred=has_fm,
            TDA=not is_spline,
            vec_dis=not (is_cva or is_reg or is_pco),
            x_colnames=ez.col_names,
        )

        if use_pc_toggle:
            layout["updatemenus"][1]["buttons"] = [
                {
                    "method": "skip",
                    "args": ["type", "scatter" if i == 0 else "histogram"],
                    "label": pc_map[name],
                }
                for i, name in enumerate(available)
            ]

        data = list(first["mds"]["trace_data"])
        annotations = first["mds"]["layout"].get("annotations") or []
        if annotations:
            layout["annotations"] = annotations

        mds_for_js: dict[str, Any] = {}
        for name in available:
            label = pc_map[name]
            if name == first_name:
                entry: dict[str, Any] = {"config": first["mds"]["config"]}
                if has_fm:
                    entry["fit_table"] = self.fit_measures[ft_map[name]]
                mds_for_js[label] = entry
            else:
                js_payl = dict(self.displays[name]["mds"])
                if has_fm:
                    js_payl["fit_table"] = self.fit_measures[ft_map[name]]
                mds_for_js[label] = js_payl

        fm_mds = (
            {key: self.fit_measures[key] for key in FIT_GRAPH_NAMES}
            if has_fm
            else None
        )

        payload = {
            "p": ez.p,
            "cols": ez.axes["tick_label_col"],
            "class_mean_hover": False,
            "mdsDisplays": mds_for_js,
            "fm_mdsDisplay": fm_mds,
            "fitDisplay": cfg,
            "ax_slider": None,
            "initialPCKey": pc_map[first_name],
        }

        figure = {"data": data, "layout": layout}
        return Bipl5Widget(figure, payload)

    # ── verbs ───────────────────────────────────────────────────────────

    def overlay_fit(self, overlay: bool = True) -> "Biplot":
        """Port of ``overlay_fit()``: store the default fit-display mode."""
        if self.fit_measures is None:
            raise ValueError("overlay_fit() requires a biplot with fit measures.")
        out = self._copy()
        options = dict(out.meta.get("plot_options") or {})
        options["fit_display"] = "overlay" if overlay else "panel"
        out.meta["plot_options"] = options
        return out

    def extract(self, path: str):
        """Port of ``extract()`` with dotted-string paths.

        - ``extract("mdsDisplay_12")`` returns a plottable one-display Biplot
        - ``extract("mdsDisplay_12.Data.sample_coordinates")`` drills in
        - ``extract("fit_measures.CumPred")`` returns a :class:`BiplotFit`
        """
        parts = path.split(".") if isinstance(path, str) else list(path)
        if not parts:
            raise ValueError("extract() requires a non-empty path.")

        if parts[0] == "fit_measures":
            if len(parts) != 2 or parts[1] not in FIT_GRAPH_NAMES:
                raise ValueError(
                    "fit-measure paths must be one of: "
                    + ", ".join(f"fit_measures.{n}" for n in FIT_GRAPH_NAMES)
                )
            if self.fit_measures is None:
                raise ValueError("this biplot has no fit measures.")
            return BiplotFit(self.fit_measures[parts[1]], fit_name=parts[1])

        if len(parts) == 1:
            name = parts[0]
            if name in self.meta["pc_info"]:
                return self._subset([name])
            raise KeyError(f"'{name}' is not an mdsDisplay of this biplot.")

        result: Any = self.displays.get(parts[0])
        if result is None:
            raise KeyError(f"'{parts[0]}' is not an mdsDisplay of this biplot.")
        for field in parts[1:]:
            if isinstance(result, dict):
                if field not in result:
                    raise KeyError(f"Field '{field}' not found at this level.")
                result = result[field]
            else:
                result = getattr(result, field)
        return result

    def _subset(self, keep: list[str]) -> "Biplot":
        """Port of ``subset_biplot()``: keep only the named displays."""
        out = self._copy()
        out.displays = {k: self.displays[k] for k in keep}
        out.meta["pc_info"] = {
            k: v for k, v in self.meta["pc_info"].items() if k in keep
        }
        if self.fit_measures is not None:
            kept_ft = {v["ft_name"] for v in out.meta["pc_info"].values()}
            out.fit_measures = {
                k: v
                for k, v in self.fit_measures.items()
                if k in FIT_GRAPH_NAMES or k in kept_ft
            }
        return out

    def remove_mds_display(self, name: str) -> "Biplot":
        """Port of ``remove_mdsDisplay()``."""
        if self.biplot_type in ("reg", "pco"):
            raise ValueError(
                "remove_mds_display() is not supported for this biplot type."
            )
        all_names = list(self.meta["pc_info"])
        if name not in all_names:
            raise ValueError(
                f"'{name}' is not a valid mdsDisplay name. Must be one of: "
                + ", ".join(all_names)
            )
        keep = [k for k in all_names if k != name and k in self.displays]
        if not keep:
            raise ValueError("Cannot remove the last remaining mdsDisplay.")
        return self._subset(keep)

    def append_mds_display(self, eigenvectors) -> "Biplot":
        """Port of ``append_mdsDisplay()``: add a new dimension pair.

        Re-runs the stored ordination for the requested pair and builds a
        full display for it, reusing the object's aesthetics.
        """
        if self.biplot_type in ("reg", "pco"):
            raise ValueError(
                "append_mds_display() is not supported for this biplot type."
            )
        eigenvectors = tuple(int(v) for v in eigenvectors)
        if len(eigenvectors) != 2:
            raise ValueError("eigenvectors must be a pair of PC indices.")

        pcs = tuple(sorted(eigenvectors))
        ez = self.meta["x"]
        p = ez.p
        if any(v < 1 or v > p for v in pcs):
            raise ValueError(
                f"eigenvectors must be between 1 and {p} (the number of variables)."
            )
        if pcs[0] == pcs[1]:
            raise ValueError("eigenvectors must contain two different PC indices.")
        pname = mds_display_name(pcs)
        if pname in self.meta["pc_info"]:
            raise ValueError(
                f"{pname} already exists in this object. Existing mdsDisplays: "
                + ", ".join(self.meta["pc_info"])
            )

        from .ordination import biplot as ez_biplot
        from .ordination import cva as ez_cva
        from .ordination import fit_measures as ez_fit
        from .ordination import pca as ez_pca

        dim_prefix = self.meta.get("dim_prefix", "PC")
        is_cva = self.biplot_type == "cva"
        stored = self.meta.get("scale_mds", {})
        args = stored.get("args", {})

        spec = self.meta.get("spec")
        source = spec.analysis_data if spec is not None else ez.raw_X
        base = ez_biplot(source, center=ez.center, scaled=ez.scaled)
        if is_cva:
            new_ez = ez_cva(
                base,
                classes=stored.get("common", {}).get("classes"),
                e_vects=pcs,
                weighted_cva=args.get("weighted_cva", "weighted"),
            )
        else:
            new_ez = ez_pca(
                base,
                e_vects=pcs,
                correlation_biplot=bool(args.get("correlation_biplot", False)),
            )
        new_ez = ez_fit(new_ez)
        new_ez = restore_raw_x(new_ez)

        group_codes, group_levels = self.meta["group"]
        new_payl = build_one_mds_display(
            new_ez,
            group_codes=group_codes,
            group_levels=group_levels,
            color=self.meta["color"],
            symbol=self.meta["symbol"],
            x_ref=ez,
            include_polygons=False,
            dim_prefix=dim_prefix,
            ax_pred=not is_cva,
            vec_dis=not is_cva,
        )

        # replay any stored sample formatting so the new display matches
        if self.meta.get("sample_format") is not None:
            from .format_samples import (
                get_state,
                rebuild_mds_display,
                should_update_means,
            )

            state = get_state(self, ez.n)
            new_payl = rebuild_mds_display(
                new_payl,
                update_means=should_update_means(self, state),
                state=state,
                ez=self.meta["x"],
                rebuild_tda=state.get("color") is not None,
            )

        out = self._copy()
        out.displays = dict(self.displays)
        out.displays[pname] = new_payl

        if out.fit_measures is not None:
            out.fit_measures = dict(out.fit_measures)
            out.fit_measures[ft_name(pcs)] = fit_table_traces(new_ez)

        out.meta["pc_info"] = dict(self.meta["pc_info"])
        out.meta["pc_info"][pname] = {
            "pcs": pcs,
            "label": pair_label(pcs, prefix=dim_prefix),
            "ft_name": ft_name(pcs),
        }
        return out

    def format_samples(
        self, stratify: str = "col", by=None, col=None, pch=None
    ) -> "Biplot":
        from .format_samples import format_samples

        return format_samples(self, stratify=stratify, by=by, col=col, pch=pch)

    def score_axes(self, digits: int = 2) -> "Biplot":
        from .score_axes import score_axes

        return score_axes(self, digits=digits)

    # ── printing ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ez = self.meta["x"]
        lines = [
            f"bipl5 biplot [{self.biplot_type}] "
            f"(n={ez.n}, p={ez.p}, groups={len(self.meta['group'][1])})"
        ]
        quality = self.meta.get("fit_quality")
        if quality:
            lines.append(f"  {quality}")
        lines.append("  mdsDisplays:")
        for name, info in self.meta["pc_info"].items():
            marker = "*" if name in self.displays else " "
            lines.append(f"   {marker} {name}  ({info['label']})")
        if self.fit_measures is not None:
            graphs = ", ".join(FIT_GRAPH_NAMES)
            lines.append(f"  fit_measures: {graphs}")
        if self.meta.get("reading_errors"):
            lines.append("  score_axes: reading errors in hover tables")
        lines.append("  plot() renders the interactive biplot")
        return "\n".join(lines)
