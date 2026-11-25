import numpy as np
import pandas as pd
import pytest

from bipl5.ordination import biplot, cva, fit_measures


def test_cva_core_conventions(data, groups):
    bp = cva(biplot(data), classes=groups)

    # Canonical weights satisfy M' W M = I
    np.testing.assert_allclose(
        bp.Lmat.T @ bp.Wmat @ bp.Lmat, np.eye(bp.p), atol=1e-8
    )
    np.testing.assert_allclose(bp.Z, bp.X @ bp.Lmat[:, :2])
    # Class means live at the group means of Z
    for i in range(3):
        np.testing.assert_allclose(
            bp.Zmeans[i], bp.Z[bp.group == i].mean(axis=0), atol=1e-10
        )
    assert np.all(np.diff(bp.eigenvalues) <= 1e-10)
    # W + B partitions X'X
    np.testing.assert_allclose(bp.Wmat + bp.Bmat, bp.X.T @ bp.X, atol=1e-8)


def test_cva_requires_classes(data):
    with pytest.raises(ValueError, match="class variable"):
        cva(biplot(data))


def test_cva_weighting_options(data, groups):
    weighted = cva(biplot(data), classes=groups, weighted_cva="weighted")
    unweighted = cva(biplot(data), classes=groups, weighted_cva="unweightedI")
    assert not np.allclose(weighted.eigenvalues[:2], unweighted.eigenvalues[:2])
    with pytest.raises(ValueError, match="weighted_cva"):
        cva(biplot(data), classes=groups, weighted_cva="nope")


def test_cva_fit_measures(data, groups):
    bp = fit_measures(cva(biplot(data), classes=groups))
    assert set(bp.quality) == {"canonical variables", "original variables"}
    for key in ("canonical variables", "original variables"):
        assert 0 <= bp.quality[key] <= 1 + 1e-12
    for vec in (
        bp.adequacy,
        bp.axis_predictivity,
        bp.class_predictivity,
        bp.within_class_axis_predictivity,
        bp.within_class_sample_predictivity,
    ):
        assert np.all(vec >= -1e-12) and np.all(vec <= 1 + 1e-12)
    assert bp.class_predictivity.shape == (3,)
    assert bp.within_class_sample_predictivity.shape == (bp.n,)


def test_cva_two_groups_low_dim(data):
    two = pd.Series(["a"] * 30 + ["b"] * 30)
    with pytest.warns(UserWarning, match="canonical space"):
        bp = cva(biplot(data), classes=two)
    assert bp.Z.shape == (60, 2)
    assert bp.eigenvalues is None
    assert bp.crit_opt is not None
    assert bp.low_dim == "sample.opt"
    # the appended dimension still gives an invertible transformation
    assert np.linalg.matrix_rank(bp.Lmat) == bp.p


def test_cva_singular_within_matrix(data, groups):
    degenerate = data.copy()
    degenerate["dup"] = degenerate["alpha"]  # perfectly collinear column
    with pytest.raises(ValueError, match="singularity"):
        cva(biplot(degenerate), classes=groups)
