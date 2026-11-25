import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data() -> pd.DataFrame:
    """Deterministic correlated data: 60 samples, 4 variables, 3 groups."""
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(60, 2))
    mixing = np.array(
        [
            [2.0, 0.3, -1.0, 0.5],
            [0.2, 1.5, 0.8, -0.7],
        ]
    )
    X = latent @ mixing + rng.normal(scale=0.4, size=(60, 4))
    X += np.array([10.0, -5.0, 3.0, 0.0])
    return pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])


@pytest.fixture
def groups() -> pd.Series:
    return pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 20, name="grp")


def align_signs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Flip columns of B so each best matches the sign of the same column in A.

    SVD/eigendecomposition column signs are arbitrary; comparisons between
    two computation routes must be sign-aligned first.
    """
    flipped = B.copy()
    for j in range(A.shape[1]):
        if np.dot(A[:, j], flipped[:, j]) < 0:
            flipped[:, j] = -flipped[:, j]
    return flipped
