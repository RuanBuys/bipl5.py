"""``init_biplot()`` and ``scale_mds()`` — port of ``init_biplot.R``.

``init_biplot()`` stores the data and preprocessing options in a
:class:`BiplotSpec`; ``scale_mds()`` runs the requested ordination through
the internal engine (``bipl5.ordination``) and compiles a fully formed
:class:`~bipl5.biplot.Biplot`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .biplot import Biplot, ft_name, mds_display_name, pair_label
from .display.build_one import (
    build_one_mds_display,
    clean_linear_axes_coordinates,
    restore_raw_x,
    zero_to_near_zero,
)
from .display.builders import (
    add_axis_adeq_traces,
    add_axis_pred_traces,
    add_prop_variance_traces,
    add_scree_traces,
    fit_table_traces,
)
from .ordination import (
    axes as ez_axes,
    axes_coordinates,
    biplot as ez_biplot,
    cva as ez_cva,
    fit_measures as ez_fit_measures,
    fit_quality_string,
    means as ez_means,
    pca as ez_pca,
    pco as ez_pco,
    regress as ez_regress,
    samples as ez_samples,
)
from .ordination.fit_measures import (
    regression_fit_quality,
    regression_fit_quality_tex,
)
from .printing import FitMeasures
from .symbols import pch_to_plotly

__all__ = ["init_biplot", "BiplotSpec"]

_TYPES = {
    "pca": "pca",
    "cva": "cva",
    "pco": "pco",
    "reg": "regress",
    "regress": "regress",
    "regression": "regress",
}


def _pull_arg(kwargs: dict, aliases: tuple[str, ...], default=None):
    """Port of ``scale_mds_pull_arg()``: resolve aliased arguments."""
    hits = [k for k in kwargs if k in aliases]
    if len(hits) > 1:
        raise ValueError("Please supply only one of: " + ", ".join(aliases) + ".")
    if not hits:
        return default
    return kwargs.pop(hits[0])


def _check_unused(kwargs: dict, type_: str) -> None:
    if kwargs:
        raise ValueError(
            f"Unsupported arguments for scale_mds(type='{type_}'): "
            + ", ".join(kwargs)
        )


class BiplotSpec:
    """Port of ``bipl5_spec``: raw data + preprocessing options."""

    def __init__(self, data, center: bool = True, scale: bool = False):
        if not isinstance(center, bool) or not isinstance(scale, bool):
            raise ValueError("center and scale must be True or False.")
        if isinstance(data, pd.DataFrame):
            frame = data
        else:
            arr = np.asarray(data)
            if arr.ndim != 2:
                raise ValueError("data must be a matrix or data frame.")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError("matrix inputs to init_biplot() must be numeric.")
            frame = pd.DataFrame(
                arr, columns=[f"V{j + 1}" for j in range(arr.shape[1])]
            )

        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise ValueError(
                "init_biplot() requires at least one numeric column "
                "for the biplot calculation."
            )

        self.data = frame
        self.analysis_data = numeric
        self.numeric_columns = [str(c) for c in numeric.columns]
        self.center = center
        self.scale = scale

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<bipl5 spec: {self.data.shape[0]} obs, "
            f"{len(self.numeric_columns)} numeric columns, "
            f"center={self.center}, scale={self.scale}>"
        )

    # ── scale_mds ───────────────────────────────────────────────────────

    def scale_mds(self, type: str = "pca", **kwargs) -> Biplot:
        """Port of ``scale_mds()``: run the ordination and compile a Biplot.

        ``type`` is one of ``"pca"``, ``"cva"``, ``"pco"`` or ``"regress"``.
        Common aliases: ``classes``, ``group_aes``, ``title``. Method
        arguments mirror the R package (``dimensions``/``eigenvectors``,
        ``show_class_means``/``show_group_means``, ``correlation_biplot``,
        ``weighted_cva``, ``Dmat``/``dist_mat``, ``dist_func``, ``Z`` …);
        for ``type="pco"`` remaining keyword arguments are forwarded to the
        distance function.
        """
        key = _TYPES.get(str(type).lower())
        if key is None:
            raise ValueError(
                f"Unsupported type '{type}'. Use one of: "
                "'pca', 'cva', 'pco', 'regress'."
            )
        kwargs = dict(kwargs)

        classes = _pull_arg(kwargs, ("classes",))
        group_aes = _pull_arg(kwargs, ("group_aes",))
        title = _pull_arg(kwargs, ("title",))
        common = {
            k: v
            for k, v in (
                ("classes", classes),
                ("group_aes", group_aes),
                ("title", title),
            )
            if v is not None
        }

        base = ez_biplot(
            self.analysis_data,
            classes=classes,
            group_aes=group_aes,
            center=self.center,
            scaled=self.scale,
            title=title,
        )

        builder = {
            "pca": _build_pca,
            "cva": _build_cva,
            "pco": _build_pco,
            "regress": _build_regress,
        }[key]
        ez, args = builder(base, kwargs, common)

        compiler = {
            "pca": _compile_pca,
            "cva": _compile_cva,
            "pco": _compile_pco,
            "regress": _compile_regress,
        }[key]
        bp = compiler(ez)

        bp.meta["spec"] = self
        bp.meta["scale_mds"] = {"type": key, "common": common, "args": args}
        return bp


def init_biplot(data, center: bool = True, scale: bool = False) -> BiplotSpec:
    """Create a bipl5 specification object (port of ``init_biplot()``).

    Stores the raw data and preprocessing options; run
    :meth:`BiplotSpec.scale_mds` to construct the biplot. Non-numeric
    DataFrame columns are retained for later formatting steps but excluded
    from the ordination.
    """
    return BiplotSpec(data, center=center, scale=scale)


# ── method builders (alias resolution, port of scale_mds_build_*) ─────────

_SHOW_ALIASES = (
    "show_class_means",
    "show_group_means",
)


def _build_pca(base, kwargs, common):
    dimensions = _pull_arg(kwargs, ("dimensions", "dim_biplot"))
    eigenvectors = _pull_arg(kwargs, ("eigenvectors", "e_vects"))
    show_class_means = _pull_arg(kwargs, _SHOW_ALIASES)
    correlation_biplot = _pull_arg(kwargs, ("correlation_biplot",))
    _check_unused(kwargs, "pca")

    call: dict[str, Any] = {}
    if dimensions is not None:
        call["dim_biplot"] = dimensions
    if eigenvectors is not None:
        call["e_vects"] = eigenvectors
    if show_class_means is not None:
        call["show_class_means"] = show_class_means
    if correlation_biplot is not None:
        call["correlation_biplot"] = correlation_biplot

    args = {
        k: v
        for k, v in (
            ("dimensions", dimensions),
            ("eigenvectors", eigenvectors),
            ("show_class_means", show_class_means),
            ("correlation_biplot", correlation_biplot),
        )
        if v is not None
    }
    return ez_pca(base, **call), args


def _build_cva(base, kwargs, common):
    if "classes" not in common:
        raise ValueError("scale_mds(type='cva') requires 'classes'.")
    dimensions = _pull_arg(kwargs, ("dimensions", "dim_biplot"))
    eigenvectors = _pull_arg(kwargs, ("eigenvectors", "e_vects"))
    weighted_cva = _pull_arg(kwargs, ("weighted_cva",))
    show_class_means = _pull_arg(kwargs, _SHOW_ALIASES)
    low_dim = _pull_arg(kwargs, ("low_dim",))
    _check_unused(kwargs, "cva")

    call: dict[str, Any] = {"classes": common["classes"]}
    if dimensions is not None:
        call["dim_biplot"] = dimensions
    if eigenvectors is not None:
        call["e_vects"] = eigenvectors
    if weighted_cva is not None:
        call["weighted_cva"] = weighted_cva
    if show_class_means is not None:
        call["show_class_means"] = show_class_means
    if low_dim is not None:
        call["low_dim"] = low_dim

    args = {
        k: v
        for k, v in (
            ("classes", common["classes"]),
            ("dimensions", dimensions),
            ("eigenvectors", eigenvectors),
            ("weighted_cva", weighted_cva),
            ("show_class_means", show_class_means),
            ("low_dim", low_dim),
        )
        if v is not None
    }
    return ez_cva(base, **call), args


def _build_pco(base, kwargs, common):
    Dmat = _pull_arg(kwargs, ("Dmat", "dist_mat"))
    dist_func = _pull_arg(kwargs, ("dist_func",))
    dist_func_cat = _pull_arg(kwargs, ("dist_func_cat",))
    dimensions = _pull_arg(kwargs, ("dimensions", "dim_biplot"))
    eigenvectors = _pull_arg(kwargs, ("eigenvectors", "e_vects"))
    show_class_means = _pull_arg(kwargs, _SHOW_ALIASES)
    axes = _pull_arg(kwargs, ("axes",))
    # remaining kwargs are forwarded to the distance function, as in R

    call: dict[str, Any] = dict(kwargs)
    kwargs.clear()
    if Dmat is not None:
        call["Dmat"] = Dmat
    if dist_func is not None:
        call["dist_func"] = dist_func
    if dist_func_cat is not None:
        call["dist_func_cat"] = dist_func_cat
    if dimensions is not None:
        call["dim_biplot"] = dimensions
    if eigenvectors is not None:
        call["e_vects"] = eigenvectors
    if show_class_means is not None:
        call["show_class_means"] = show_class_means
    if axes is not None:
        call["axes"] = axes

    args = {
        k: v
        for k, v in (
            ("Dmat", Dmat),
            ("dist_func", dist_func),
            ("dist_func_cat", dist_func_cat),
            ("dimensions", dimensions),
            ("eigenvectors", eigenvectors),
            ("show_class_means", show_class_means),
            ("axes", axes),
        )
        if v is not None
    }
    return ez_pco(base, **call), args


def _build_regress(base, kwargs, common):
    Z = _pull_arg(kwargs, ("Z", "z"))
    if Z is None:
        raise ValueError("scale_mds(type='regress') requires 'Z'.")
    show_group_means = _pull_arg(kwargs, _SHOW_ALIASES)
    axes = _pull_arg(kwargs, ("axes",))
    _check_unused(kwargs, "regress")

    call: dict[str, Any] = {}
    if show_group_means is not None:
        call["show_group_means"] = show_group_means
    if axes is not None:
        call["axes"] = axes

    args = {
        k: v
        for k, v in (
            ("Z", Z),
            ("show_group_means", show_group_means),
            ("axes", axes),
        )
        if v is not None
    }
    return ez_regress(base, Z, **call), args


# ── compile paths (port of scale_mds_compile_*_biplot) ─────────────────────

def _display_aes(ez):
    """Port of ``scale_mds_extract_display_aes()``."""
    color = ez.samples["col"]
    symbol = pch_to_plotly(ez.samples["pch"])
    codes, levels = ez.group, list(ez.g_names)
    if len(levels) == 1:
        codes = np.zeros(ez.n, dtype=int)
        levels = ["Data"]
    return color, symbol, (codes, levels)


def _prepare(ez, with_fit=True, with_means=False):
    if ez.samples is None:
        ez = ez_samples(ez)
    if ez.axes is None:
        ez = ez_axes(ez)
    if with_fit:
        ez = ez_fit_measures(ez)
    if with_means and ez.means_aes is None:
        ez = ez_means(ez)
    return ez


def _single_biplot(
    ez, bundle, fit_measures, pcs, dim_prefix, biplot_type, fit_quality,
    fit_quality_plotly=None, spline=False,
) -> Biplot:
    """Port of ``scale_mds_new_single_biplot()``."""
    color, symbol, group = _display_aes(ez)
    pname = mds_display_name(pcs)
    meta = {
        "x": ez,
        "color": color,
        "symbol": symbol,
        "group": group,
        "fit_quality": fit_quality,
        "pc_info": {
            pname: {
                "pcs": pcs,
                "label": pair_label(pcs, prefix=dim_prefix),
                "ft_name": ft_name(pcs),
            }
        },
        "dim_prefix": dim_prefix,
        "spline": spline,
    }
    if fit_quality_plotly:
        meta["fit_quality_plotly"] = fit_quality_plotly
    return Biplot({pname: bundle}, fit_measures, meta, biplot_type=biplot_type)


def _compile_pca(ez) -> Biplot:
    ez = _prepare(ez, with_fit=True)
    ez = restore_raw_x(ez)
    color, symbol, (codes, levels) = _display_aes(ez)
    pcs = tuple(sorted(ez.e_vects))

    bundle = build_one_mds_display(
        ez,
        group_codes=codes,
        group_levels=levels,
        color=color,
        symbol=symbol,
        x_ref=ez,
        include_polygons=True,
    )

    fit_measures = FitMeasures(
        {
            "CumPred": add_axis_pred_traces(ez),
            "CumAd": add_axis_adeq_traces(ez),
            "VarExp": add_prop_variance_traces(ez),
            "Scree": add_scree_traces(ez),
            ft_name(pcs): fit_table_traces(ez),
        }
    )

    return _single_biplot(
        ez,
        bundle,
        fit_measures,
        pcs,
        dim_prefix="PC",
        biplot_type="pca",
        fit_quality=fit_quality_string(ez.eigenvalues, ez.e_vects),
    )


def _compile_cva(ez) -> Biplot:
    ez = _prepare(ez, with_fit=True, with_means=True)
    ez = restore_raw_x(ez)
    color, symbol, (codes, levels) = _display_aes(ez)
    pcs = tuple(sorted(ez.e_vects))

    bundle = build_one_mds_display(
        ez,
        group_codes=codes,
        group_levels=levels,
        color=color,
        symbol=symbol,
        x_ref=ez,
        include_polygons=True,
        dim_prefix="CV",
        ax_pred=False,
        vec_dis=False,
    )

    return _single_biplot(
        ez,
        bundle,
        None,
        pcs,
        dim_prefix="CV",
        biplot_type="cva",
        fit_quality=fit_quality_string(ez.eigenvalues, ez.e_vects, dim_prefix="CV"),
    )


def _compile_pco(ez) -> Biplot:
    ez = _prepare(ez, with_fit=False)
    z_axes = zero_to_near_zero(axes_coordinates(ez))
    ez = restore_raw_x(ez)
    color, symbol, (codes, levels) = _display_aes(ez)
    pcs = (1, 2)

    z_axes = clean_linear_axes_coordinates(ez, z_axes)
    bundle = build_one_mds_display(
        ez,
        group_codes=codes,
        group_levels=levels,
        color=color,
        symbol=symbol,
        x_ref=ez,
        include_polygons=True,
        dim_prefix="Dim",
        ax_pred=False,
        vec_dis=False,
        z_axes=z_axes,
    )

    return _single_biplot(
        ez,
        bundle,
        None,
        pcs,
        dim_prefix="Dim",
        biplot_type="pco",
        fit_quality="",
        spline=False,
    )


def _compile_regress(ez) -> Biplot:
    ez = _prepare(ez, with_fit=False)
    z_axes = clean_linear_axes_coordinates(ez)
    pcs = (1, 2)
    fit_qual = regression_fit_quality(ez.X, ez.Z, dim_prefix="Dim")
    fit_qual_plotly = regression_fit_quality_tex(ez.X, ez.Z)

    ez = restore_raw_x(ez)
    color, symbol, (codes, levels) = _display_aes(ez)

    bundle = build_one_mds_display(
        ez,
        group_codes=codes,
        group_levels=levels,
        color=color,
        symbol=symbol,
        x_ref=ez,
        include_polygons=True,
        dim_prefix="Dim",
        ax_pred=False,
        vec_dis=False,
        z_axes=z_axes,
        fit_qual=fit_qual,
    )

    return _single_biplot(
        ez,
        bundle,
        None,
        pcs,
        dim_prefix="Dim",
        biplot_type="reg",
        fit_quality=fit_qual,
        fit_quality_plotly=fit_qual_plotly,
    )
