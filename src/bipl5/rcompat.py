"""Ports of base-R utilities that bipl5 depends on for numerical parity.

Only functions whose R defaults materially shape bipl5 output live here.
Each port documents the R original it replicates.
"""

from __future__ import annotations

import math
import sys

import numpy as np

__all__ = ["pretty", "r_range"]


def r_range(x) -> tuple[float, float]:
    """Equivalent of R's ``range(x)`` for numeric input."""
    arr = np.asarray(x, dtype=float)
    return float(np.min(arr)), float(np.max(arr))


def pretty(
    x,
    n: int = 5,
    min_n: int | None = None,
    shrink_sml: float = 0.75,
    high_u_bias: float = 1.5,
    u5_bias: float | None = None,
) -> np.ndarray:
    """Port of R's ``pretty()`` (the ``R_pretty`` algorithm in pretty.c).

    Computes a sequence of about ``n`` "pretty" values (multiples of 1, 2 or
    5 times a power of 10) covering the range of ``x``. This drives the
    calibrated tick labels on every biplot axis, so it must match R —
    ``numpy``/``matplotlib`` tick locators choose different breakpoints.

    Parameters mirror R: ``n`` the desired number of intervals, ``min_n``
    the minimum number of intervals (R default ``n %/% 3``), and the bias
    parameters controlling the preference for 2, 5 and 10 as step factors.

    Degenerate zero-width ranges reproduce R's behaviour approximately; the
    calibration code only calls this with non-degenerate data ranges.
    """
    lo, up = r_range(x)
    if min_n is None:
        min_n = n // 3
    if u5_bias is None:
        u5_bias = 0.5 + 1.5 * high_u_bias
    h, h5 = high_u_bias, u5_bias
    eps = sys.float_info.epsilon
    rounding_eps = 1e-10

    dx = up - lo
    if dx == 0 and up == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(up))
        u = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        u *= max(1, n) * eps
        i_small = dx < cell * u * 3

    if i_small:
        cell = min(cell, 99.9)
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if n > 1:
            cell /= n
    cell = max(cell, 20 * sys.float_info.min)

    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if (2 * base) - cell < h * (cell - unit):
        unit = 2 * base
        if (5 * base) - cell < h5 * (cell - unit):
            unit = 5 * base
            if (10 * base) - cell < h * (cell - unit):
                unit = 10 * base

    ns = math.floor(lo / unit + rounding_eps)
    nu = math.ceil(up / unit - rounding_eps)
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1

    k = nu - ns
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2

    return np.arange(ns, nu + 1, dtype=float) * unit
