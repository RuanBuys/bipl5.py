"""Core biplot container — Python port of ``biplotEZ::biplot()``.

This module defines :class:`EZBiplot`, the equivalent of the list object of
class ``biplot`` created by ``biplotEZ::biplot()`` in R, and the
:func:`biplot` constructor. Method functions (:func:`~bipl5.ordination.pca`,
:func:`~bipl5.ordination.cva`, …) enrich a copy of this object exactly as the
corresponding biplotEZ generics do.

Conventions
-----------
- ``e_vects`` is **1-based** in the public API, matching R and the bipl5
  display vocabulary (``PC 1 & 2``). The 0-based version for indexing NumPy
  arrays is available as :attr:`EZBiplot.e_idx`.
- All other stored indices (``samples["which"]``, ``axes["which"]``) are
  0-based.
- When ``center=False`` the stored ``means`` are zeros and when
  ``scale=False`` the stored ``sd`` are ones, so that
  ``X == (raw_X - means) / sd`` holds unconditionally (same convention as
  biplotEZ's ``prcomp`` branch).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["EZBiplot", "biplot", "as_factor", "indmat"]


def as_factor(values) -> tuple[np.ndarray, list[str]]:
    """Replicate R ``factor()``: return integer codes and sorted level names.

    Existing pandas Categoricals keep their category order; anything else
    gets levels sorted like R's default (lexicographic for strings, numeric
    order for numbers). Level names are returned as strings, matching R's
    ``levels()`` which is always character.
    """
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        values = values.array
    if isinstance(values, pd.Categorical):
        codes = np.asarray(values.codes, dtype=int)
        if (codes < 0).any():
            raise ValueError("grouping variable contains missing values")
        return codes, [str(c) for c in values.categories]

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("grouping variable must be one-dimensional")
    levels = sorted(pd.unique(arr).tolist())
    lookup = {lev: i for i, lev in enumerate(levels)}
    codes = np.array([lookup[v] for v in arr.tolist()], dtype=int)
    return codes, [str(lev) for lev in levels]


def indmat(codes: np.ndarray, g: int | None = None) -> np.ndarray:
    """Port of biplotEZ's ``indmat()``: n x g one-hot indicator matrix."""
    codes = np.asarray(codes, dtype=int)
    if g is None:
        g = int(codes.max()) + 1
    out = np.zeros((codes.shape[0], g))
    out[np.arange(codes.shape[0]), codes] = 1.0
    return out


class EZBiplot:
    """Python equivalent of the biplotEZ ``biplot`` object.

    Instances are created by :func:`biplot` and enriched by the method
    functions. Attribute names mirror the R object's element names with
    snake_case (``raw.X`` → ``raw_X``, ``group.aes`` → ``group``,
    ``ax.one.unit`` → ``ax_one_unit``).
    """

    # --- populated by biplot() -------------------------------------------
    X: np.ndarray            # centered/scaled numeric matrix (n x p)
    raw_X: np.ndarray        # original numeric matrix
    Xcat: pd.DataFrame | None  # categorical columns (PCO), else None
    means: np.ndarray
    sd: np.ndarray
    center: bool
    scaled: bool
    n: int
    p: int
    col_names: list[str]
    row_names: list[str]
    group: np.ndarray        # integer codes, length n (R: group.aes factor)
    g_names: list[str]       # level names (R: g.names)
    g: int
    classes: Any             # raw classes vector as supplied, or None
    title: str | None

    # --- populated by method functions (None until set) ------------------
    method: str | None = None          # "pca" | "cva" | "pco" | "regress"
    Z: np.ndarray | None = None
    Lmat: np.ndarray | None = None
    Vr: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    ax_one_unit: np.ndarray | None = None
    e_vects: tuple[int, ...] | None = None   # 1-based
    dim_biplot: int | None = None
    class_means: bool = False
    Zmeans: np.ndarray | None = None
    PCOaxes: str | None = None         # "regression" | "splines" (PCO/regress)

    # CVA extras
    Xmeans: np.ndarray | None = None
    Gmat: np.ndarray | None = None
    Cmat: np.ndarray | None = None
    Bmat: np.ndarray | None = None
    Wmat: np.ndarray | None = None
    Mr: np.ndarray | None = None
    Mrr: np.ndarray | None = None
    Nmat: np.ndarray | None = None
    crit_opt: float | None = None
    low_dim: str | None = None

    # PCO extras
    DDmat: np.ndarray | None = None
    Ymat: np.ndarray | None = None
    Lambda: np.ndarray | None = None
    dist_func: Callable | None = None
    dist_func_cat: Callable | None = None

    # aesthetics (dicts; see aesthetics.py)
    samples: dict | None = None
    axes: dict | None = None
    means_aes: dict | None = None

    # fit measures (see fit_measures.py)
    quality: Any = None
    adequacy: np.ndarray | None = None
    axis_predictivity: np.ndarray | None = None
    sample_predictivity: np.ndarray | None = None
    class_predictivity: np.ndarray | None = None
    within_class_axis_predictivity: np.ndarray | None = None
    within_class_sample_predictivity: np.ndarray | None = None

    @property
    def e_idx(self) -> np.ndarray:
        """0-based array version of the 1-based ``e_vects``."""
        if self.e_vects is None:
            raise AttributeError("no method applied yet: e_vects is unset")
        return np.asarray(self.e_vects, dtype=int) - 1

    def _copy(self) -> "EZBiplot":
        """Shallow copy, mimicking R's copy-on-modify value semantics."""
        return copy.copy(self)

    # --- chaining sugar: bp.pca(...) === pca(bp, ...) --------------------
    def pca(self, **kwargs) -> "EZBiplot":
        from .pca import pca

        return pca(self, **kwargs)

    def cva(self, classes=None, **kwargs) -> "EZBiplot":
        from .cva import cva

        return cva(self, classes=classes, **kwargs)

    def pco(self, **kwargs) -> "EZBiplot":
        from .pco import pco

        return pco(self, **kwargs)

    def regress(self, Z, **kwargs) -> "EZBiplot":
        from .regress import regress

        return regress(self, Z, **kwargs)

    def with_samples(self, **kwargs) -> "EZBiplot":
        from .aesthetics import samples

        return samples(self, **kwargs)

    def with_axes(self, **kwargs) -> "EZBiplot":
        from .aesthetics import axes

        return axes(self, **kwargs)

    def with_means(self, **kwargs) -> "EZBiplot":
        from .aesthetics import means

        return means(self, **kwargs)

    def fit_measures(self) -> "EZBiplot":
        from .fit_measures import fit_measures

        return fit_measures(self)

    def axes_coordinates(self) -> list[np.ndarray]:
        from .calibration import axes_coordinates

        return axes_coordinates(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        method = self.method or "unfitted"
        return (
            f"<EZBiplot [{method}] n={self.n}, p={self.p}, g={self.g}, "
            f"center={self.center}, scaled={self.scaled}>"
        )


def _split_numeric(data) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame | None]:
    """Split input into a numeric matrix and (optional) categorical columns."""
    if isinstance(data, pd.DataFrame):
        numeric = data.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise ValueError("biplot() requires at least one numeric column.")
        categorical = data.drop(columns=numeric.columns)
        X = numeric.to_numpy(dtype=float)
        col_names = [str(c) for c in numeric.columns]
        index = data.index
        if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
            # default pandas index: use R-style 1-based row names
            row_names = [str(i + 1) for i in range(len(index))]
        else:
            row_names = [str(r) for r in index]
        Xcat = categorical if categorical.shape[1] > 0 else None
        return X, col_names, row_names, Xcat

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("biplot() expects a 2-D matrix or a DataFrame.")
    col_names = [f"V{j + 1}" for j in range(X.shape[1])]
    row_names = [str(i + 1) for i in range(X.shape[0])]
    return X, col_names, row_names, None


def biplot(
    data,
    classes: Sequence | None = None,
    group_aes: Sequence | None = None,
    center: bool = True,
    scaled: bool = False,
    title: str | None = None,
) -> EZBiplot:
    """Port of ``biplotEZ::biplot()`` for matrix/DataFrame input.

    Stores the (optionally centered and scaled) numeric matrix along with
    grouping information, ready for a method function such as
    :func:`~bipl5.ordination.pca`.

    Parameters
    ----------
    data:
        A 2-D numeric array or a DataFrame. DataFrame non-numeric columns are
        kept aside as ``Xcat`` (used by PCO's categorical distances).
    classes, group_aes:
        Grouping vectors of length ``n``; ``group_aes`` wins when both are
        given (matching R, where ``classes`` seeds ``group.aes``).
    center, scaled:
        Column centering/scaling flags. Scaling uses the sample standard
        deviation (``ddof=1``), like R's ``scale()``.
    """
    X, col_names, row_names, Xcat = _split_numeric(data)
    n, p = X.shape
    if np.isnan(X).any():
        raise ValueError(
            "biplot() does not support missing values in numeric columns."
        )

    means = X.mean(axis=0) if center else np.zeros(p)
    sd = X.std(axis=0, ddof=1) if scaled else np.ones(p)
    Xwork = (X - means) / sd

    grouping = group_aes if group_aes is not None else classes
    if grouping is not None:
        if len(grouping) != n:
            raise ValueError("grouping variable must have one value per row.")
        codes, levels = as_factor(grouping)
    else:
        codes, levels = np.zeros(n, dtype=int), ["1"]

    bp = EZBiplot()
    bp.X = Xwork
    bp.raw_X = X
    bp.Xcat = Xcat
    bp.means = means
    bp.sd = sd
    bp.center = bool(center)
    bp.scaled = bool(scaled)
    bp.n = n
    bp.p = p
    bp.col_names = col_names
    bp.row_names = row_names
    bp.group = codes
    bp.g_names = levels
    bp.g = len(levels)
    bp.classes = classes
    bp.title = title
    return bp
