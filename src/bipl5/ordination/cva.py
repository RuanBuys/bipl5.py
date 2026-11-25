"""CVA biplot method — Python port of ``biplotEZ::CVA()`` and ``CVAlowdim()``."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .base import EZBiplot, as_factor, indmat

__all__ = ["cva"]

_WEIGHT_OPTIONS = ("weighted", "unweightedI", "unweightedCent")


def _cva_low_dim(bp, M, low_dim, K, e_idx):
    """Port of ``CVAlowdim()`` for the ``"sample.opt"`` strategy.

    Adds dimensions to the canonical space when the number of classes minus
    one is smaller than the biplot dimension, by maximizing the sample
    predictivity of the extra dimension(s).
    """
    if low_dim != "sample.opt":
        raise NotImplementedError(
            "Only low_dim='sample.opt' is implemented; the "
            "'Bhattacharyya.dist' strategy is not yet ported."
        )
    first_dims = e_idx[:K]
    others = np.setdiff1d(np.arange(M.shape[1]), first_dims)

    Minv = np.linalg.inv(M)
    M_sup2 = Minv[others, :]
    # right singular vectors of the symmetric matrix M_sup2 M_sup2'
    _, _, vt = np.linalg.svd(M_sup2 @ M_sup2.T)
    fvek_opt = vt.T
    M_opt = np.hstack([M[:, first_dims], M[:, others] @ fvek_opt])

    Mr = M_opt[:, e_idx]
    Mrr = np.linalg.inv(M_opt)[e_idx, :]
    X_hat = bp.X @ Mr @ Mrr
    tsres = float(np.sum((bp.X - X_hat) ** 2) / np.sum(bp.X**2))
    return M_opt, tsres


def cva(
    bp: EZBiplot,
    classes=None,
    dim_biplot: int = 2,
    e_vects: Sequence[int] | None = None,
    weighted_cva: str = "weighted",
    show_class_means: bool = True,
    low_dim: str = "sample.opt",
) -> EZBiplot:
    """Append CVA elements to a biplot object (port of ``CVA.biplot``).

    Solves the two-sided eigenvalue problem of the between-class matrix
    ``B`` with respect to the within-class matrix ``W`` via
    ``W^(-1/2) B W^(-1/2)``, giving the canonical weight matrix
    ``M = W^(-1/2) V`` stored as ``Lmat``. Sample coordinates are
    ``Z = X @ M[:, e_vects]`` and class means ``Zmeans = Xbar @ M[:, e_vects]``.

    ``weighted_cva`` selects the class weighting ``Cmat`` used for ``B``:
    ``"weighted"`` (class sizes), ``"unweightedI"`` (identity) or
    ``"unweightedCent"`` (centering matrix). ``e_vects`` is 1-based.
    """
    if dim_biplot not in (1, 2, 3):
        raise ValueError("Only 1D, 2D and 3D biplots")
    if e_vects is None:
        e_vects = tuple(range(1, dim_biplot + 1))
    e_vects = tuple(int(v) for v in e_vects[:dim_biplot])
    if weighted_cva not in _WEIGHT_OPTIONS:
        raise ValueError(
            "Argument 'weighted_cva' must be one of "
            "'weighted', 'unweightedI', 'unweightedCent'"
        )

    if classes is None:
        classes = bp.classes
    if classes is None:
        raise ValueError("You have to specify the class variable in argument classes.")

    bp = bp._copy()
    codes, levels = as_factor(classes)
    bp.classes = classes
    if bp.g == 1:
        bp.group, bp.g_names, bp.g = codes, levels, len(levels)

    X = bp.X
    p = bp.p
    G = indmat(codes, len(levels))
    J = G.shape[1]
    K = min(p, J - 1)
    e_idx = np.asarray(e_vects, dtype=int) - 1

    N = G.T @ G
    X_bar = np.linalg.solve(N, G.T @ X)
    W = X.T @ X - X_bar.T @ N @ X_bar
    B = X_bar.T @ N @ X_bar

    u_w, d_w, vt_w = np.linalg.svd(W)
    if np.any(d_w < 1e-10):
        raise ValueError(
            "Your within class covariance matrix is approaching singularity."
        )
    W_minhalf = u_w @ np.diag(1 / np.sqrt(d_w)) @ vt_w

    if weighted_cva == "weighted":
        Cmat = N
    elif weighted_cva == "unweightedI":
        Cmat = np.eye(J)
    else:
        Cmat = np.eye(J) - np.full((J, J), 1 / J)

    _, d_c, vt_c = np.linalg.svd(W_minhalf @ X_bar.T @ Cmat @ X_bar @ W_minhalf)
    V = vt_c.T
    M = W_minhalf @ V

    bp.method = "cva"
    bp.crit_opt = None
    if K < dim_biplot:
        M_opt, tsres = _cva_low_dim(bp, M, low_dim, K, e_idx)
        bp.Lmat = M_opt
        bp.eigenvalues = None
        Mr = M_opt[:, e_idx]
        Minv = np.linalg.inv(M_opt)[e_idx, :]
        bp.crit_opt = tsres
        warnings.warn(
            "The dimension of the canonical space < dim_biplot; "
            f"{low_dim} method used for additional dimension(s).",
            stacklevel=2,
        )
    else:
        Mr = M[:, e_idx]
        Minv = np.linalg.inv(M)[e_idx, :]
        bp.Lmat = M
        bp.eigenvalues = d_c

    bp.Z = X @ Mr
    bp.ax_one_unit = Minv.T / np.sum(Minv.T**2, axis=1, keepdims=True)

    bp.Xmeans = X_bar
    bp.Gmat = G
    bp.Zmeans = X_bar @ Mr
    bp.e_vects = e_vects
    bp.Cmat = Cmat
    bp.Bmat = B
    bp.Wmat = W
    bp.Mrr = Minv
    bp.Mr = Mr
    bp.Nmat = N
    bp.class_means = bool(show_class_means)
    bp.dim_biplot = dim_biplot
    bp.low_dim = low_dim
    return bp
