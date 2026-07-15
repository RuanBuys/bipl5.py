"""Measures of fit — port of ``biplotEZ::fit.measures()`` plus the
cumulative predictivity/adequacy curves that bipl5 computes itself
(``ax_pred.R`` and ``FitPanel_funcs.R`` in the R package).
"""

from __future__ import annotations

import numpy as np

from .base import EZBiplot, indmat

__all__ = [
    "fit_measures",
    "cumulative_axis_predictivities",
    "cumulative_adequacies",
    "marginal_axis_predictivities",
    "fit_quality_string",
]


def _row_quad_diag(A: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """diag(A @ Minv @ A.T) without forming the full n x n product."""
    return np.einsum("ij,ij->i", A @ Minv, A)


def fit_measures(bp: EZBiplot) -> EZBiplot:
    """Augment a fitted biplot with measures of fit (port of ``fit.measures``).

    For PCA: overall ``quality``, per-variable ``adequacy`` and
    ``axis_predictivity``, and per-observation ``sample_predictivity``.

    For CVA: ``quality`` for both canonical and original variables,
    ``adequacy``, ``axis_predictivity`` and ``class_predictivity`` in the
    ``Cmat`` metric, plus ``within_class_axis_predictivity`` and
    ``within_class_sample_predictivity`` in the ``W^-1`` metric.

    Formulas follow Gardner-Lubbe, le Roux & Gower (2008) exactly as
    implemented in biplotEZ.
    """
    if bp.method == "pca":
        bp = bp._copy()
        e_idx = bp.e_idx
        eig = bp.eigenvalues
        bp.quality = float(eig[e_idx].sum() / eig.sum())
        bp.adequacy = np.sum(bp.Lmat[:, e_idx] ** 2, axis=1)
        Xhat = bp.Z @ bp.Lmat[:, e_idx].T
        bp.axis_predictivity = np.sum(Xhat**2, axis=0) / np.sum(bp.X**2, axis=0)
        bp.sample_predictivity = np.sum(Xhat**2, axis=1) / np.sum(bp.X**2, axis=1)
        return bp

    if bp.method == "cva":
        bp = bp._copy()
        e_idx = bp.e_idx
        Lmat_inv = np.linalg.inv(bp.Lmat)
        Xbar_hat = bp.Zmeans @ Lmat_inv[e_idx, :]

        u_c, d_c, vt_c = np.linalg.svd(bp.Cmat)
        C_half = u_c @ np.diag(np.sqrt(d_c)) @ vt_c
        G = bp.Gmat
        proj = G @ np.linalg.solve(G.T @ G, G.T)
        Xwithin = bp.X - proj @ bp.X
        Xwithin_hat = Xwithin @ bp.Lmat[:, e_idx] @ Lmat_inv[e_idx, :]

        quality = {}
        if bp.eigenvalues is not None:
            quality["canonical variables"] = float(
                bp.eigenvalues[e_idx].sum() / bp.eigenvalues.sum()
            )
        quality["original variables"] = float(
            np.trace(Xbar_hat.T @ bp.Cmat @ Xbar_hat)
            / np.trace(bp.Xmeans.T @ bp.Cmat @ bp.Xmeans)
        )
        bp.quality = quality

        bp.adequacy = np.sum(bp.Lmat[:, e_idx] ** 2, axis=1) / np.diag(
            bp.Lmat @ bp.Lmat.T
        )
        bp.axis_predictivity = np.diag(Xbar_hat.T @ bp.Cmat @ Xbar_hat) / np.diag(
            bp.Xmeans.T @ bp.Cmat @ bp.Xmeans
        )

        W_inv = np.linalg.inv(bp.Wmat)
        num = np.diag(C_half @ Xbar_hat @ W_inv @ Xbar_hat.T @ C_half)
        den = np.diag(C_half @ bp.Xmeans @ W_inv @ bp.Xmeans.T @ C_half)
        bp.class_predictivity = num / den

        bp.within_class_axis_predictivity = np.sum(
            Xwithin_hat**2, axis=0
        ) / np.sum(Xwithin**2, axis=0)
        bp.within_class_sample_predictivity = _row_quad_diag(
            Xwithin_hat, W_inv
        ) / _row_quad_diag(Xwithin, W_inv)
        return bp

    raise ValueError(
        "fit_measures() is implemented for PCA and CVA biplots; "
        f"got method={bp.method!r}."
    )


def cumulative_axis_predictivities(bp: EZBiplot) -> np.ndarray:
    """Port of bipl5's ``axis_predictivities_EZ()`` (``ax_pred.R``).

    Returns a ``(p + 1) x p`` matrix: row ``j`` traces variable ``j``'s
    cumulative axis predictivity as the subspace rank grows from 1 to ``p``;
    the final row is the overall quality at each rank. Rounding to three
    decimals matches the R implementation.
    """
    if bp.method != "pca" or bp.Lmat is None or bp.eigenvalues is None:
        raise ValueError("cumulative_axis_predictivities() requires a PCA biplot.")
    V = bp.Lmat
    eig = bp.eigenvalues
    p, n = bp.p, bp.n

    out = np.full((p + 1, p), np.nan)
    total = np.diag(V @ np.diag(eig) @ V.T)
    for i in range(1, min(p, n) + 1):
        Vi = V[:, :i]
        part = np.diag(Vi @ np.diag(eig[:i]) @ Vi.T)
        out[:p, i - 1] = np.round(part / total, 3)
        out[p, i - 1] = eig[:i].sum() / eig.sum()
    return out


def cumulative_adequacies(bp: EZBiplot) -> np.ndarray:
    """Port of bipl5's ``axis_adequacies()`` (``FitPanel_funcs.R``).

    ``p x p`` matrix of cumulative adequacies: entry ``(i, j)`` is the sum of
    squared loadings of variable ``i`` over the first ``j`` components.
    """
    if bp.Lmat is None:
        raise ValueError("cumulative_adequacies() requires a fitted PCA biplot.")
    return np.cumsum(bp.Lmat**2, axis=1)


def marginal_axis_predictivities(bp: EZBiplot) -> np.ndarray:
    """Port of bipl5's ``marginal_predictivities_EZ()`` (``FitPanel_funcs.R``).

    Per-variable predictivity of the displayed pair only (used in the
    summary fit table alongside adequacy).
    """
    if bp.method != "pca" or bp.Lmat is None or bp.eigenvalues is None:
        raise ValueError("marginal_axis_predictivities() requires a PCA biplot.")
    V = bp.Lmat
    eig = bp.eigenvalues
    Vr = bp.Vr
    lam_r = np.diag(eig[bp.e_idx])
    total = np.diag(V @ np.diag(eig) @ V.T)
    return np.diag(Vr @ lam_r @ Vr.T) / total


def fit_quality_string(
    eigenvalues, e_vects, dim_prefix: str = "PC", digits: int = 2
) -> str:
    """Port of bipl5's ``fit_quality()`` display string.

    Example: ``"Quality of display = 95.81% = 72.96% (PC1) + 22.85% (PC2)"``.
    Returns an empty string when eigenvalues are unavailable (as for CVA
    low-dimension fits), matching R.
    """
    if eigenvalues is None:
        return ""
    eig = np.asarray(eigenvalues, dtype=float)
    idx = np.asarray(e_vects, dtype=int) - 1
    total = eig.sum()

    def fmt(x: float) -> str:
        # R's paste0(round(x, digits)) drops trailing zeros ("75", "74.96")
        return f"{round(x, digits):g}"

    both = fmt((eig[idx[0]] + eig[idx[1]]) / total * 100)
    first = fmt(eig[idx[0]] / total * 100)
    second = fmt(eig[idx[1]] / total * 100)
    return (
        f"Quality of display = {both}% = {first}% ({dim_prefix}{e_vects[0]}) "
        f"+ {second}% ({dim_prefix}{e_vects[1]})"
    )
