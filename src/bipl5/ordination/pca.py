"""PCA biplot method — Python port of ``biplotEZ::PCA()``."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .base import EZBiplot, as_factor, biplot, indmat

__all__ = ["pca"]


def pca(
    bp: EZBiplot,
    dim_biplot: int = 2,
    e_vects: Sequence[int] | None = None,
    group_aes=None,
    show_class_means: bool = False,
    correlation_biplot: bool = False,
) -> EZBiplot:
    """Append PCA elements to a biplot object (port of ``PCA.biplot``).

    Follows biplotEZ exactly: the SVD of the centered/scaled matrix gives
    ``Lmat`` (the full right singular vectors), ``eigenvalues`` (squared
    singular values, i.e. eigenvalues of X'X), sample coordinates
    ``Z = X @ Vr`` and per-variable axis directions ``ax_one_unit`` with rows
    ``v_j / ||v_j||^2``. The ``correlation_biplot`` variant rescales ``Z``,
    ``Lmat`` and ``ax_one_unit`` by the singular values as in R.

    ``e_vects`` is 1-based (``(1, 2)`` selects the first two components).
    """
    if dim_biplot not in (1, 2, 3):
        raise ValueError("Only 1D, 2D and 3D biplots")
    if e_vects is None:
        e_vects = tuple(range(1, dim_biplot + 1))
    e_vects = tuple(int(v) for v in e_vects[:dim_biplot])
    if any(v < 1 or v > bp.p for v in e_vects):
        raise ValueError(f"e_vects entries must be between 1 and {bp.p}.")

    bp = bp._copy()
    if group_aes is not None:
        bp.group, bp.g_names = as_factor(group_aes)
        bp.g = len(bp.g_names)

    if not bp.center:
        warnings.warn(
            "PCA requires a centred datamatrix. Your data was centred before "
            "computation. Use center=True in the call to biplot()",
            stacklevel=2,
        )
        recentred = biplot(
            bp.raw_X, center=True, scaled=bp.scaled, title=bp.title
        )
        recentred.group, recentred.g_names, recentred.g = bp.group, bp.g_names, bp.g
        recentred.col_names, recentred.row_names = bp.col_names, bp.row_names
        recentred.classes, recentred.Xcat = bp.classes, bp.Xcat
        bp = recentred

    X = bp.X
    n = bp.n
    idx = np.asarray(e_vects, dtype=int) - 1

    _, d, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T                       # p x min(n, p), R's svd(X)$v
    Lmat = V
    Vr = V[:, idx]

    if correlation_biplot:
        lambda_r = d[idx] ** 2
        Z = np.sqrt(n - 1) * X @ Vr @ np.diag(1 / np.sqrt(lambda_r))
        Lmat = np.sqrt(n - 1) * Lmat @ np.diag(1 / np.sqrt(d**2))
        denom = np.diag(Vr @ np.diag(lambda_r) @ Vr.T)
        ax_one_unit = (np.sqrt(n - 1) / denom)[:, None] * (
            Vr @ np.diag(np.sqrt(lambda_r))
        )
    else:
        Z = X @ Vr
        ax_one_unit = Vr / np.sum(Vr**2, axis=1, keepdims=True)

    bp.method = "pca"
    bp.Z = Z
    bp.Lmat = Lmat
    bp.eigenvalues = d**2
    bp.ax_one_unit = ax_one_unit
    bp.e_vects = e_vects
    bp.Vr = Vr
    bp.dim_biplot = dim_biplot

    bp.class_means = False if bp.g == 1 else bool(show_class_means)
    if bp.class_means:
        G = indmat(bp.group, bp.g)
        Xmeans = np.linalg.solve(G.T @ G, G.T @ X)
        bp.Zmeans = Xmeans @ Lmat[:, idx]

    return bp
