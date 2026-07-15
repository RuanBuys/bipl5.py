import numpy as np
import pytest

from bipl5.rcompat import pretty

# Golden values generated with R: pretty(x, n)
R_CASES = [
    ((4.3, 7.9), 5, [4, 5, 6, 7, 8]),
    ((0.0, 1.0), 5, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ((2.0, 10.5), 5, [2, 4, 6, 8, 10, 12]),
    ((-1.5, 3.2), 5, [-2, -1, 0, 1, 2, 3, 4]),
    ((0.0, 0.05), 5, [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]),
    ((100.0, 105.0), 5, [100, 101, 102, 103, 104, 105]),
]


@pytest.mark.parametrize("x, n, expected", R_CASES)
def test_pretty_matches_r(x, n, expected):
    np.testing.assert_allclose(pretty(x, n=n), expected, atol=1e-12)


def test_pretty_covers_range():
    rng = np.random.default_rng(7)
    for _ in range(200):
        lo, span = rng.normal(scale=50), abs(rng.normal(scale=10)) + 1e-6
        out = pretty((lo, lo + span))
        assert out[0] <= lo + 1e-9
        assert out[-1] >= lo + span - 1e-9
        steps = np.diff(out)
        np.testing.assert_allclose(steps, steps[0])  # evenly spaced
