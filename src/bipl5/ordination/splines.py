"""Spline (non-linear) biplot axes — port of biplotEZ 2.3's
``biplot.spline.axis()`` (R driver in ``plot2D.R``) and its C++ kernel
``LnjTinyNew.cpp`` (the ``alfunc`` loss and the Nelder-Mead ``amoeba``
optimizer that replaced the original Fortran on GitHub; not yet on CRAN).

For each variable, a B-spline trajectory through the display space is fitted
so that the value read at each sample's nearest curve point approximates the
sample's actual value. The optimizer is multi-started from perturbations of
the best solution, exactly as in R.
"""

from __future__ import annotations

import numpy as np

from ..rcompat import pretty

__all__ = ["spline_axis_control", "spline_axis", "bs_basis", "alfunc"]


def spline_axis_control(
    tau: float = 0.5,
    nmu: int = 100,
    u: int = 2,
    v: int = 3,
    lambda_: float = 0.0,
    smallsigma: float = 0.01,
    bigsigma: float | None = None,
    gamma: int = 250,
    bigsigmaactivate: int | None = None,
    eps: float = 0.01,
    tiny: float = 1e-30,
    itmax: int = 10_000,
    ftol: float | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Port of ``biplot.spline.axis.control()`` with the R defaults.

    ``seed`` and ``verbose`` are Python extensions: the R implementation
    always uses the global RNG and always prints progress.
    """
    if bigsigma is None:
        bigsigma = smallsigma * 10
    if bigsigmaactivate is None:
        bigsigmaactivate = int(np.floor(gamma * 0.1))
    if ftol is None:
        ftol = 1.5 * np.finfo(float).eps

    if tau <= 0:
        raise ValueError("value of 'tau' must be > 0")
    if nmu <= 0:
        raise ValueError("value of 'nmu' must be > 0")
    if u < 0 or v < 0:
        raise ValueError("values of 'u' and 'v' must not be less than zero")
    if lambda_ < 0:
        raise ValueError("value of 'lambda_' must be >= 0")
    if smallsigma <= 0 or bigsigma <= 0:
        raise ValueError("sigma values must be > 0")
    if gamma <= 0 or bigsigmaactivate <= 0 or itmax <= 0:
        raise ValueError("gamma, bigsigmaactivate and itmax must be > 0")

    return {
        "tau": tau,
        "nmu": nmu,
        "u": u,
        "v": v,
        "lambda": lambda_,
        "smallsigma": smallsigma,
        "bigsigma": bigsigma,
        "gamma": gamma,
        "bigsigmaactivate": bigsigmaactivate,
        "eps": eps,
        "tiny": tiny,
        "itmax": itmax,
        "ftol": ftol,
        "seed": seed,
        "verbose": verbose,
    }


# ── B-spline basis (replacement for splines::bs) ────────────────────────────

def bs_basis(x, degree: int, interior_knots=None, df: int | None = None) -> np.ndarray:
    """B-spline basis like ``splines::bs(x, knots=, degree=, intercept=FALSE)``.

    Boundary knots sit at ``range(x)`` (the R default). With
    ``interior_knots=None`` and ``df`` given, ``df - degree`` interior knots
    are placed at quantiles of ``x`` — matching ``bs(x, df=, degree=)``.
    The first basis function is dropped (``intercept=FALSE``), giving
    ``len(interior_knots) + degree`` columns.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if interior_knots is None:
        n_interior = max((df or degree) - degree, 0)
        if n_interior > 0:
            probs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.quantile(x, probs)
        else:
            interior_knots = np.array([])
    interior_knots = np.sort(np.asarray(interior_knots, dtype=float))

    order = degree + 1
    knots = np.concatenate(
        [np.full(order, lo), interior_knots, np.full(order, hi)]
    )
    n_basis = len(interior_knots) + order

    # Cox-de Boor recursion
    basis = np.zeros((x.size, len(knots) - 1))
    for i in range(len(knots) - 1):
        basis[:, i] = (x >= knots[i]) & (x < knots[i + 1])
    # right-closed support at the upper boundary
    last = np.where(knots[:-1] < hi)[0]
    if last.size:
        basis[x == hi, last[-1]] = 1.0

    for d in range(1, order):
        new = np.zeros((x.size, len(knots) - 1 - d))
        for i in range(len(knots) - 1 - d):
            left_den = knots[i + d] - knots[i]
            right_den = knots[i + d + 1] - knots[i + 1]
            term = np.zeros(x.size)
            if left_den > 0:
                term = term + (x - knots[i]) / left_den * basis[:, i]
            if right_den > 0:
                term = term + (knots[i + d + 1] - x) / right_den * basis[:, i + 1]
            new[:, i] = term
        basis = new

    full = basis[:, :n_basis]
    return full[:, 1:]  # intercept = FALSE drops the first basis function


# ── loss and optimizer (port of LnjTinyNew.cpp) ─────────────────────────────

def alfunc(bvec, X, y, M, mu, lambda_, const1, const2) -> float:
    """Port of the C++ ``alfunc`` loss, vectorized.

    ``Z = M @ B`` is the curve; each sample contributes the squared
    difference between its value and the ``mu`` value at its nearest curve
    point, scaled by ``const1``; ``lambda_`` adds a second-difference
    smoothness penalty scaled by ``const2``.
    """
    uv = M.shape[1]
    B = np.asarray(bvec, dtype=float).reshape(2, uv).T  # column-major layout
    Z = M @ B  # nmu x 2

    d2 = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(Z**2, axis=1)[None, :]
        - 2.0 * (X @ Z.T)
    )
    closest = np.argmin(d2, axis=1)
    loss = float(np.sum((y - mu[closest]) ** 2)) / const1

    if lambda_ > 0 and Z.shape[0] > 2:
        second = Z[:-2] - 2 * Z[1:-1] + Z[2:]
        loss += lambda_ * float(np.sum(second**2)) / const2
    return loss


def _varset(bvec, tau: float) -> np.ndarray:
    """Port of ``varset()``: axis-aligned initial simplex."""
    ndim = bvec.size
    simplex = np.tile(bvec, (ndim + 1, 1))
    simplex[1:, :] += np.eye(ndim) * tau
    return simplex


def _amoeba(simplex, yvek, loss, ftol, itmax, tiny):
    """Port of the C++ ``amoeba`` (Numerical Recipes Nelder-Mead).

    Mutates ``simplex``/``yvek`` in place; returns ``(iterations, error)``
    with the best vertex swapped into row 0, exactly as the C++ does.
    """
    ndim = simplex.shape[1]
    if ndim > 20:
        return 0, 5
    iter_count = 0

    def amotry(psum, ihi, fac):
        fac1 = (1.0 - fac) / ndim
        fac2 = fac1 - fac
        ptry = psum * fac1 - simplex[ihi] * fac2
        ytry = loss(ptry)
        if ytry < yvek[ihi]:
            yvek[ihi] = ytry
            psum += ptry - simplex[ihi]
            simplex[ihi] = ptry
        return ytry

    psum = simplex.sum(axis=0)
    while True:
        order = np.argsort(yvek)
        ilo, ihi = order[0], order[-1]
        inhi = order[-2]

        rtol = 2.0 * abs(yvek[ihi] - yvek[ilo]) / (
            abs(yvek[ihi]) + abs(yvek[ilo]) + tiny
        )
        if rtol < ftol:
            yvek[0], yvek[ilo] = yvek[ilo], yvek[0]
            simplex[[0, ilo]] = simplex[[ilo, 0]]
            return iter_count, 0
        if iter_count >= itmax:
            return iter_count, 1

        iter_count += 2
        ytry = amotry(psum, ihi, -1.0)
        if ytry <= yvek[ilo]:
            amotry(psum, ihi, 2.0)
        elif ytry >= yvek[inhi]:
            ysave = yvek[ihi]
            ytry = amotry(psum, ihi, 0.5)
            if ytry >= ysave:
                for i in range(simplex.shape[0]):
                    if i != ilo:
                        simplex[i] = 0.5 * (simplex[i] + simplex[ilo])
                        yvek[i] = loss(simplex[i])
                iter_count += ndim
                psum = simplex.sum(axis=0)
        else:
            iter_count -= 1


def optimize_spline(bvec, X, y, M, mu, lambda_, const1, const2, tau, ftol, tiny, itmax):
    """Port of the C++ ``optimize_spline``: two amoeba stages, each from a
    fresh axis-aligned simplex around the current best coefficients."""

    def loss(b):
        return alfunc(b, X, y, M, mu, lambda_, const1, const2)

    best = np.asarray(bvec, dtype=float)
    iters = []
    erro = 0
    for _ in range(2):
        simplex = _varset(best, tau)
        yvek = np.array([loss(v) for v in simplex])
        it, erro = _amoeba(simplex, yvek, loss, ftol, itmax, tiny)
        iters.append(it)
        best = simplex[0].copy()
        best_loss = float(yvek[0])
        if erro != 0:
            break

    return {"LOSS": best_loss, "BVEC": best, "ITER": iters[-1], "ERRO": erro}


# ── the spline-axis driver (port of biplot.spline.axis) ─────────────────────

def spline_axis(j, X, Y, means, sd, control=None, rng=None) -> np.ndarray:
    """Fit the calibrated spline axis for variable ``j`` (0-based).

    Parameters mirror the R driver: ``X`` the n x 2 display coordinates,
    ``Y`` the centered/scaled data matrix, ``means``/``sd`` the original
    column transforms (biplotEZ 2.3 passes ``x$X`` here — the GitHub change
    away from ``raw.X``).

    Returns the ``nmu x 4`` axis matrix: curve x/y, tick value in raw units,
    and a 0/1 flag marking the labelled ("pretty") tick positions.
    """
    control = dict(spline_axis_control() if control is None else control)
    rng = np.random.default_rng(control.get("seed")) if rng is None else rng

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape

    if n > 103:  # R subsamples large datasets for tractability
        keep = rng.choice(n, size=103, replace=False)
        X = X[keep]
        Y = Y[keep]
        n = 103

    u, v = int(control["u"]), int(control["v"])
    if control["verbose"]:
        print(f"Calculating spline axis for variable {j + 1}", flush=True)

    ytilde = Y[:, j] * sd[j] + means[j]  # raw units
    y = Y[:, j]
    rangey = float(y.max() - y.min())
    mu = np.linspace(y.min() - 0.3 * rangey, y.max() + 0.3 * rangey, control["nmu"])
    markers = (pretty((ytilde.min(), ytilde.max())) - means[j]) / sd[j]
    mu = np.unique(np.concatenate([mu, markers]))

    if v > 0:
        probs = np.linspace(0, 1, v + 2)[1:-1]
        knots = np.quantile(y, probs)
        M = bs_basis(mu, degree=u, interior_knots=knots)
    else:
        M = bs_basis(mu, degree=u, df=u + v)
    # centre so the curve passes through the origin at the variable's mean
    M = M - M[np.argmin(np.abs(mu)), :]

    # start closest to the regression biplot's linear axis
    Breg = np.linalg.solve(X.T @ X, X.T @ y)  # length 2
    Zreg = np.outer(mu, Breg) / float(np.sum(Breg**2))
    Bmat = np.linalg.lstsq(M, Zreg, rcond=None)[0]  # (u+v) x 2
    bvec = Bmat.T.reshape(-1)  # column-major, like R's as.vector

    const1 = float(np.sum(y**2))
    const2 = float(np.sum(X**2)) / (n * p)

    def run(b):
        return optimize_spline(
            b, X, y, M, mu,
            control["lambda"], const1, const2,
            control["tau"], control["ftol"], control["tiny"],
            int(control["itmax"]),
        )

    gamma = int(control["gamma"])
    ndim = 2 * (u + v)
    best_values = [np.nan] * (gamma + 1)
    best_solutions = np.full((ndim, gamma + 1), np.nan)
    frequency = [0] * (gamma + 1)

    out = run(bvec)
    best_values[0] = out["LOSS"]
    best_solutions[:, 0] = out["BVEC"]
    frequency[0] = 1
    distinct = 1
    previous_best = None
    n_same_consecutively = 0

    for _ in range(1, gamma + 1):
        current = best_solutions[:, int(np.nanargmin(best_values[:distinct]))]
        sigma = (
            control["bigsigma"]
            if n_same_consecutively >= control["bigsigmaactivate"]
            else control["smallsigma"]
        )
        out = run(current + rng.normal(0.0, sigma, size=ndim))

        d2 = np.sum(
            (best_solutions[:, :distinct] - out["BVEC"][:, None]) ** 2, axis=0
        )
        hits = np.where(d2 < control["eps"])[0]
        if hits.size:
            idx = int(d2.argmin())
            frequency[idx] += 1
            if previous_best is not None and idx == previous_best:
                n_same_consecutively += 1
            else:
                previous_best = idx
                n_same_consecutively = 0
        else:
            best_values[distinct] = out["LOSS"]
            best_solutions[:, distinct] = out["BVEC"]
            frequency[distinct] = 1
            distinct += 1
            n_same_consecutively = 0

    winner = best_solutions[:, int(np.nanargmin(best_values[:distinct]))]
    curve = M @ winner.reshape(2, M.shape[1]).T

    flags = np.isin(mu, markers).astype(float)
    return np.column_stack([curve, mu * sd[j] + means[j], flags])
