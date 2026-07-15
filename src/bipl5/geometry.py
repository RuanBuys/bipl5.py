"""Geometry for translated density axes (TDA) — ports of the R package's
``Translate.R``, ``Translate_Helpers.R`` and the ellipse machinery that R
delegated to ``cluster::ellipsoidhull()``.

The TDA algorithm rotates each calibrated axis until horizontal, slides it
out of a bounding ellipse of the data (stacking axes so they never overlap),
and superimposes per-group kernel densities on each translated axis.
"""

from __future__ import annotations

import numpy as np

from .rcompat import density_r

__all__ = [
    "rotation_constructor",
    "translate",
    "move_lines",
    "compute_density_inflation",
    "move_densities",
    "get_quads_axes",
    "shorten_axes",
    "obtain_zhat",
    "mvee",
    "ellipse_points",
]


def rotation_constructor(angles) -> np.ndarray:
    """Port of ``RotationConstructor()``: horizontally stacked 2x2 rotations.

    Returns a ``2 x 2p`` matrix; block ``i`` rotates row-vectors by
    ``angles[i]`` via ``x @ R[:, 2i:2i+2]``.
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    mat = np.empty((2, 2 * angles.size))
    mat[0, 0::2] = np.cos(angles)
    mat[0, 1::2] = -np.sin(angles)
    mat[1, 0::2] = np.sin(angles)
    mat[1, 1::2] = np.cos(angles)
    return mat


def translate(elip, quadrant, other, d, endpoints, theta, swop=False):
    """Port of ``translate()``: shift a rotated axis out of the ellipse.

    Axes whose loading fell in quadrants 2/3 move below the ellipse, 1/4
    above (swapped when ``swop``), always keeping distance ``d`` from the
    ellipse and from previously placed axes (``other``).
    Returns ``{"distance": shifted_y, "ends": endpoints_in_original_space}``.
    """
    other = np.asarray(other, dtype=float) @ rotation_constructor(theta)
    endpoints = np.array(endpoints, dtype=float, copy=True)
    q1, q2 = (2, 3), (1, 4)
    if swop:
        q1, q2 = q2, q1

    others_y = other[:, 1]
    finite = others_y[np.isfinite(others_y)]
    if quadrant in q1:
        btm = float(np.min(elip[:, 1]))
        low = min([btm, *finite.tolist()])
        endpoints[:, 1] = low - d
    if quadrant in q2:
        top = float(np.max(elip[:, 1]))
        high = max([top, *finite.tolist()])
        endpoints[:, 1] = high + d

    shifted = float(endpoints[0, 1])
    true_endpoints = endpoints @ rotation_constructor(-theta)
    return {"distance": shifted, "ends": true_endpoints}


def move_lines(elip, m, quadrant, d, initial_ends, swop, cols):
    """Port of ``MoveLines()``: translate all axes out of the ellipse.

    Axes are processed in decreasing order of slope so consecutive axes
    stack without intersecting. ``initial_ends`` holds each axis's ``k x 3``
    tick matrix (x, y, label). Returns ``{"ShiftDist", "ends", "Axes"}``
    where ``ends`` carries the translated coordinates with labels retained.
    """
    m = np.asarray(m, dtype=float)
    p = m.size
    thetas = np.arctan(m)
    rot = rotation_constructor(thetas)

    dist_shifted = np.zeros(p)
    final_pos = np.full((1, 2), np.nan)
    final_enders: list = [None] * p
    axis_cols: list = [None] * p

    ordering = np.argsort(-m, kind="stable")
    for i in ordering:
        block = rot[:, 2 * i : 2 * i + 2]
        ends = np.asarray(initial_ends[i])[:, :2] @ block
        out = translate(
            (np.asarray(elip) @ rot)[:, 2 * i : 2 * i + 2],
            int(quadrant[i]),
            final_pos,
            d,
            ends,
            thetas[i],
            swop,
        )
        final_pos = np.vstack([final_pos, out["ends"]])
        dist_shifted[i] = out["distance"]
        final_enders[i] = np.column_stack(
            [out["ends"], np.asarray(initial_ends[i])[:, 2]]
        )
        axis_cols[i] = [cols[i]] * final_enders[i].shape[0]

    return {"ShiftDist": dist_shifted, "ends": final_enders, "Axes": axis_cols}


def compute_density_inflation(
    Z, m, endpoints, group, target_height, density_n: int = 128
) -> np.ndarray:
    """Port of ``compute_density_inflation()``.

    Scales each translated axis's tallest group density to
    ``target_height``, so densities occupy a fixed fraction of the axis
    spacing regardless of the data.
    """
    Z = np.asarray(Z, dtype=float)
    group = np.asarray(group)
    groups = list(dict.fromkeys(group.tolist()))  # R unique(): first-seen order
    m = np.asarray(m, dtype=float)
    p = m.size
    thetas = np.arctan(m)
    rot = rotation_constructor(thetas)
    rotated_z = Z @ rot
    target = np.broadcast_to(np.asarray(target_height, dtype=float), (p,)).copy()

    inflation = np.ones(p)
    for i in range(p):
        block = rot[:, 2 * i : 2 * i + 2]
        rotend = np.asarray(endpoints[i])[:, :2] @ block
        low, up = float(rotend[:, 0].min()), float(rotend[:, 0].max())
        peak = 0.0
        for g in groups:
            _, dens = density_r(
                rotated_z[group == g, 2 * i], from_=low, to=up, n=density_n
            )
            peak = max(peak, float(np.max(dens)))
        if np.isfinite(peak) and peak > 0:
            inflation[i] = target[i] / peak
    return inflation


def move_densities(
    Z, m, endpoints, dist, dinflation, group, density_n: int = 128
) -> list[np.ndarray]:
    """Port of ``MoveDensities()``: per-group densities on translated axes.

    For each axis, computes each group's kernel density along the rotated
    axis, lifts it to the translated position, and rotates it back. Returns
    one matrix per group with columns ``(x_axis1, y_axis1, x_axis2, ...)``.
    """
    Z = np.asarray(Z, dtype=float)
    group = np.asarray(group)
    groups = list(dict.fromkeys(group.tolist()))
    m = np.asarray(m, dtype=float)
    p = m.size
    dinflation = np.broadcast_to(np.asarray(dinflation, dtype=float), (p,)).copy()
    thetas = np.arctan(m)
    rot = rotation_constructor(thetas)
    rotated_z = Z @ rot

    density_per_group: list = [None] * len(groups)
    for i in range(p):
        block = rot[:, 2 * i : 2 * i + 2]
        rotend = np.asarray(endpoints[i])[:, :2] @ block
        low, up = float(rotend[:, 0].min()), float(rotend[:, 0].max())
        back = rotation_constructor(-thetas[i])
        for j, g in enumerate(groups):
            grid, dens = density_r(
                rotated_z[group == g, 2 * i], from_=low, to=up, n=density_n
            )
            coors = np.column_stack([grid, dens * dinflation[i] + dist[i]])
            rotated = coors @ back
            if density_per_group[j] is None:
                density_per_group[j] = rotated
            else:
                density_per_group[j] = np.hstack([density_per_group[j], rotated])
    return density_per_group


def get_quads_axes(z_axes) -> np.ndarray:
    """Port of ``get_quads_axes()``: quadrant of each axis's positive end."""
    quads = np.zeros(len(z_axes), dtype=int)
    for i, ax in enumerate(z_axes):
        ax = np.asarray(ax)
        k = int(np.argmax(ax[:, 2]))
        x, y = ax[k, 0], ax[k, 1]
        slope = y / x
        if slope > 0:
            quads[i] = 1 if x > 0 else 3
        elif slope < 0:
            quads[i] = 2 if x < 0 else 4
    return quads


def obtain_zhat(Z_ranges, z_axis) -> np.ndarray:
    """Port of ``obtain_zhat()``: interpolate tick values at two endpoints."""
    z_axis = np.asarray(z_axis, dtype=float)
    Z_ranges = np.asarray(Z_ranges, dtype=float)
    x0, x1 = z_axis[0, 0], z_axis[-1, 0]
    t0, t1 = z_axis[0, 2], z_axis[-1, 2]
    frac = (Z_ranges[:, 0] - x0) / (x1 - x0)
    return frac * (t1 - t0) + t0


def shorten_axes(z_axes, ellip) -> list[np.ndarray]:
    """Port of ``shorten_axes()``: trim calibrated axes to the ellipse.

    Rotates each axis horizontal, finds the ellipse's horizontal extent,
    keeps the ticks inside that extent plus the nearest tick outside on each
    side (so the axis starts and ends on a tick mark), and rotates back.
    """
    p = len(z_axes)
    gradient = np.array(
        [np.asarray(ax)[0, 1] / np.asarray(ax)[0, 0] for ax in z_axes]
    )
    thetas = np.arctan(gradient)
    out = []
    ellip = np.asarray(ellip, dtype=float)

    for i in range(p):
        ax = np.asarray(z_axes[i], dtype=float)
        rot = rotation_constructor(thetas[i])
        back = rotation_constructor(-thetas[i])

        rotated_axis = np.column_stack([ax[:, :2] @ rot, ax[:, 2]])
        rotated_ellip = ellip @ rot

        bounds = np.array(
            [
                [float(rotated_ellip[:, 0].min()), 0.0],
                [float(rotated_ellip[:, 0].max()), 0.0],
            ]
        )
        z_ranges = bounds @ back
        zhat_range = np.sort(obtain_zhat(z_ranges, ax))
        tick_vals = rotated_axis[:, 2]

        inside = np.where(
            (tick_vals >= zhat_range[0]) & (tick_vals <= zhat_range[1])
        )[0]
        lower = np.where(tick_vals < zhat_range[0])[0]
        upper = np.where(tick_vals > zhat_range[1])[0]

        keep = list(inside)
        if lower.size:
            keep.append(lower[np.argmax(tick_vals[lower])])
        if upper.size:
            keep.append(upper[np.argmin(tick_vals[upper])])
        keep = np.unique(np.asarray(keep, dtype=int))

        trimmed = rotated_axis[keep, :]
        out.append(np.column_stack([trimmed[:, :2] @ back, trimmed[:, 2]]))
    return out


def mvee(points, tol: float = 1e-7, max_iter: int = 10_000):
    """Minimum-volume enclosing ellipsoid via Khachiyan's algorithm.

    Python replacement for ``cluster::ellipsoidhull()``. Returns
    ``(center, A)`` where the ellipsoid is ``(x - c)' A (x - c) <= 1`` and
    every input point lies inside (up to ``tol``).
    """
    P = np.asarray(points, dtype=float)
    n, d = P.shape
    Q = np.column_stack([P, np.ones(n)]).T  # (d+1) x n
    u = np.full(n, 1 / n)

    for _ in range(max_iter):
        X = Q @ np.diag(u) @ Q.T
        M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv(X), Q)
        j = int(np.argmax(M))
        maximum = M[j]
        step = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    center = P.T @ u
    cov = P.T @ np.diag(u) @ P - np.outer(center, center)
    A = np.linalg.inv(cov) / d
    # Khachiyan is an approximation algorithm; rescale so the furthest point
    # lies exactly on the boundary, guaranteeing containment (and matching
    # the "smallest enclosing" semantics of cluster::ellipsoidhull).
    diff = P - center
    dist = np.einsum("ij,jk,ik->i", diff, A, diff)
    peak = float(dist.max())
    if peak > 0:
        A = A / peak
    return center, A


def ellipse_points(center, A, n_out: int = 101) -> np.ndarray:
    """Boundary points of the ellipse ``(x - c)' A (x - c) = 1``.

    Python replacement for ``cluster::predict.ellipsoid(..., n.out=101)``.
    """
    vals, vecs = np.linalg.eigh(np.asarray(A, dtype=float))
    radii = 1 / np.sqrt(vals)
    theta = np.linspace(0, 2 * np.pi, n_out)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    return np.asarray(center) + (circle * radii) @ vecs.T
