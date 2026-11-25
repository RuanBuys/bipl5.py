"""Regression biplot method — Python port of ``biplotEZ::regress()``."""

from __future__ import annotations

import numpy as np

from .base import EZBiplot, as_factor, indmat

__all__ = ["regress"]


def regress(
    bp: EZBiplot,
    Z,
    group_aes=None,
    show_group_means: bool = True,
    axes: str = "regression",
) -> EZBiplot:
    """Append regression-biplot elements for user-supplied coordinates.

    ``Z`` is an ``n x dim`` matrix of display coordinates (for example from
    an external MDS). It is column-centered if needed, then linear axis
    directions are obtained by regressing the (centered/scaled) data on
    ``Z``: ``Vr = ((Z'Z)^-1 Z'X)'`` with ``ax_one_unit`` rows
    ``v_j / ||v_j||^2``, exactly as in ``regress.biplot``. Like R, ``Lmat``
    and ``eigenvalues`` are unset for this method.

    ``axes="splines"`` is not yet implemented in the Python port.
    """
    if axes == "splines":
        raise NotImplementedError(
            "Spline axes are not yet implemented in bipl5.py; "
            "use axes='regression'."
        )
    if axes != "regression":
        raise ValueError("axes must be one of 'regression', 'splines'.")

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be an n x dim matrix of coordinates.")
    dim_biplot = Z.shape[1]
    if dim_biplot not in (1, 2, 3):
        raise ValueError("Only 1D, 2D and 3D biplots")
    if Z.shape[0] != bp.n:
        raise ValueError("Z must have one row per observation.")

    bp = bp._copy()
    if group_aes is not None:
        bp.group, bp.g_names = as_factor(group_aes)
        bp.g = len(bp.g_names)

    if np.any(Z.mean(axis=0) != 0):
        Z = Z - Z.mean(axis=0)

    X = bp.X
    Lmat_reg = np.linalg.solve(Z.T @ Z, Z.T @ X)   # dim x p
    Vr = Lmat_reg.T
    bp.ax_one_unit = Vr / np.sum(Vr**2, axis=1, keepdims=True)
    bp.Vr = Vr
    bp.PCOaxes = axes

    bp.method = "regress"
    bp.Z = Z
    bp.Lmat = None
    bp.eigenvalues = None
    bp.e_vects = tuple(range(1, dim_biplot + 1))
    bp.dim_biplot = dim_biplot

    bp.class_means = False if bp.g == 1 else bool(show_group_means)
    if bp.class_means:
        G = indmat(bp.group, bp.g)
        bp.Zmeans = np.linalg.solve(G.T @ G, G.T @ Z)

    return bp
