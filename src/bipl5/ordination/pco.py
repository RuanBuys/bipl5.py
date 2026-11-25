"""PCO biplot method — Python port of ``biplotEZ::PCO()``.

Includes the default distance functions: Euclidean distance for numeric
data (R's ``stats::dist``) and the extended matching coefficient for
categorical data.
"""

from __future__ import annotations

import sys
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .base import EZBiplot, as_factor, indmat

__all__ = ["pco", "euclidean_dist", "extended_matching_coefficient"]


def euclidean_dist(X, **_ignored) -> np.ndarray:
    """Pairwise Euclidean distances (equivalent of ``stats::dist``)."""
    X = np.asarray(X, dtype=float)
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def extended_matching_coefficient(Xcat, **_ignored) -> np.ndarray:
    """Port of biplotEZ's ``extended.matching.coefficient()``.

    For each categorical variable, distance 1 is added for every pair of
    samples with mismatching levels; the final distance is the square root
    of the accumulated mismatch counts.
    """
    df = pd.DataFrame(Xcat)
    n = df.shape[0]
    Dsq = np.zeros((n, n))
    for col in df.columns:
        codes, levels = as_factor(df[col])
        Gk = indmat(codes, len(levels))
        Dsq += 1.0 - Gk @ Gk.T
    return np.sqrt(Dsq)


def _as_square_dist(D, n: int) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.shape != (n, n):
        raise ValueError(f"distance matrix must be {n} x {n}, got {D.shape}.")
    return D


def pco(
    bp: EZBiplot,
    Dmat=None,
    dist_func: Callable | None = None,
    dist_func_cat: Callable | None = None,
    dim_biplot: int = 2,
    e_vects: Sequence[int] | None = None,
    group_aes=None,
    show_class_means: bool = False,
    axes: str = "regression",
    spline_control: dict | None = None,
    **dist_kwargs,
) -> EZBiplot:
    """Append PCO (classical MDS) elements to a biplot object.

    The distance matrix comes from ``Dmat`` if given; otherwise numeric
    distances (default Euclidean) and categorical distances (default
    extended matching coefficient) are combined as
    ``D = sqrt(D_num^2 + D_cat^2)``, exactly as in ``PCO.biplot``. The
    double-centered matrix ``B = -0.5 J D^2 J`` is decomposed and sample
    coordinates are ``Z = V sqrt(Lambda)`` restricted to ``e_vects``
    (1-based).

    ``axes="regression"`` fits linear calibrated axes by regressing the
    (scaled) data on ``Z``; ``axes="splines"`` fits non-linear B-spline
    axes (port of biplotEZ 2.3's C++-backed optimizer — computation is
    deferred to ``axes_coordinates()``, and can be tuned/sped up via
    ``spline_control``, see
    :func:`~bipl5.ordination.splines.spline_axis_control`).

    Extra keyword arguments are forwarded to the distance function(s),
    mirroring ``...`` in R.
    """
    if axes not in ("regression", "splines"):
        raise ValueError("axes must be one of 'regression', 'splines'.")
    if dim_biplot not in (1, 2, 3):
        raise ValueError("Only 1D, 2D and 3D biplots")

    bp = bp._copy()
    X = bp.X
    n = bp.n
    p2 = 0 if bp.Xcat is None else bp.Xcat.shape[1]
    pp = (bp.p or 0) + p2
    if e_vects is None:
        e_vects = tuple(range(1, pp + 1))
    e_vects = tuple(int(v) for v in e_vects[:dim_biplot])
    e_idx = np.asarray(e_vects, dtype=int) - 1

    if group_aes is not None:
        bp.group, bp.g_names = as_factor(group_aes)
        bp.g = len(bp.g_names)

    if dist_func is None and X is not None:
        dist_func = euclidean_dist
    if dist_func_cat is None and bp.Xcat is not None:
        dist_func_cat = extended_matching_coefficient

    if Dmat is None:
        D_sq = None
        if dist_func is not None:
            D1 = _as_square_dist(dist_func(X, **dist_kwargs), n)
            D_sq = D1**2
        if dist_func_cat is not None and bp.Xcat is not None:
            D2 = _as_square_dist(dist_func_cat(bp.Xcat, **dist_kwargs), n)
            D_sq = D2**2 if D_sq is None else D_sq + D2**2
        Dmat = np.sqrt(D_sq)
    else:
        Dmat = _as_square_dist(Dmat, n)

    DDmat = -0.5 * Dmat**2
    centering = np.eye(n) - np.full((n, n), 1 / n)
    B = centering @ DDmat @ centering

    eigvals = np.linalg.eigvalsh(B)
    if np.any(eigvals < -1e-8 * max(1.0, float(np.abs(eigvals).max()))):
        warnings.warn("Your distances are not Euclidean embeddable.", stacklevel=2)

    _, d, vh = np.linalg.svd(B)
    Ymat = vh.T @ np.diag(np.sqrt(d))
    keep = d > np.sqrt(sys.float_info.epsilon)
    Lambda = np.diag(d[keep])
    Ymat = Ymat[:, keep]
    Z = Ymat[:, e_idx]

    bp.method = "pco"
    bp.Z = Z
    bp.Lmat = None
    bp.eigenvalues = d
    bp.e_vects = e_vects
    bp.dim_biplot = dim_biplot
    bp.DDmat = DDmat
    bp.dist_func = dist_func
    bp.dist_func_cat = dist_func_cat
    bp.Ymat = Ymat
    bp.Lambda = Lambda
    bp.PCOaxes = axes

    if axes == "regression" and X is not None:
        Mr = np.linalg.solve(Z.T @ Z, Z.T @ X)          # dim x p
        bp.ax_one_unit = Mr.T / np.sum(Mr.T**2, axis=1, keepdims=True)
    else:
        bp.ax_one_unit = None
    if axes == "splines":
        from .splines import spline_axis_control

        bp.spline_control = spline_axis_control(**(spline_control or {}))

    bp.class_means = False if bp.g == 1 else bool(show_class_means)
    if bp.class_means:
        G = indmat(bp.group, bp.g)
        bp.Zmeans = np.linalg.solve(G.T @ G, G.T @ Z)

    return bp
