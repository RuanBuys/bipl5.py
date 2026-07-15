import numpy as np
import pytest

from bipl5.ordination import biplot, fit_measures, pca

from conftest import align_signs


def test_pca_core_conventions(data):
    bp = pca(biplot(data))

    # Z is the projection of the centered data onto the selected eigenvectors
    np.testing.assert_allclose(bp.Z, bp.X @ bp.Vr)
    # Lmat holds orthonormal right singular vectors
    np.testing.assert_allclose(bp.Lmat.T @ bp.Lmat, np.eye(bp.p), atol=1e-10)
    # eigenvalues are those of X'X: descending, summing to ||X||_F^2
    assert np.all(np.diff(bp.eigenvalues) <= 1e-10)
    np.testing.assert_allclose(bp.eigenvalues.sum(), np.sum(bp.X**2))
    # axis directions: rows v_j / ||v_j||^2
    expected = bp.Vr / np.sum(bp.Vr**2, axis=1, keepdims=True)
    np.testing.assert_allclose(bp.ax_one_unit, expected)


def test_pca_axis_calibration_identity(data):
    # A point mu * ax_one_unit[j] must predict exactly mu for variable j
    bp = pca(biplot(data))
    for j in range(bp.p):
        mu = 2.37
        point = mu * bp.ax_one_unit[j]
        np.testing.assert_allclose(point @ bp.Vr[j], mu)


def test_pca_e_vects_selection(data):
    bp = pca(biplot(data), e_vects=(1, 3))
    full = pca(biplot(data), e_vects=(1, 2, 3), dim_biplot=3)
    np.testing.assert_allclose(bp.Vr, full.Lmat[:, [0, 2]])
    assert bp.e_vects == (1, 3)
    np.testing.assert_array_equal(bp.e_idx, [0, 2])


def test_pca_correlation_biplot(data):
    bp = pca(biplot(data), correlation_biplot=True)
    n = bp.n
    # standardized scores: every column of Z has norm sqrt(n - 1)
    np.testing.assert_allclose(
        np.sum(bp.Z**2, axis=0), np.full(2, n - 1), rtol=1e-10
    )
    # ax_one_unit must differ from the regular biplot (bipl5's is_correlation)
    regular = pca(biplot(data))
    assert not np.allclose(bp.ax_one_unit, regular.ax_one_unit)
    # predicted (standardized) values agree between the two variants:
    # Z @ inv(Lmat)[e_idx, :] is basis-invariant
    pred_c = bp.Z @ np.linalg.inv(bp.Lmat)[bp.e_idx, :]
    pred_r = regular.Z @ np.linalg.inv(regular.Lmat)[regular.e_idx, :]
    np.testing.assert_allclose(pred_c, pred_r, atol=1e-10)


def test_pca_uncentered_warns_and_recenters(data):
    with pytest.warns(UserWarning, match="centred"):
        bp = pca(biplot(data, center=False))
    ref = pca(biplot(data, center=True))
    np.testing.assert_allclose(align_signs(ref.Z, bp.Z), ref.Z, atol=1e-10)
    assert bp.center is True


def test_pca_class_means(data, groups):
    bp = pca(biplot(data, group_aes=groups), show_class_means=True)
    assert bp.class_means is True
    assert bp.Zmeans.shape == (3, 2)
    for i in range(3):
        np.testing.assert_allclose(
            bp.Zmeans[i], bp.Z[bp.group == i].mean(axis=0), atol=1e-10
        )


def test_pca_single_group_never_shows_means(data):
    bp = pca(biplot(data), show_class_means=True)
    assert bp.class_means is False
    assert bp.Zmeans is None


def test_pca_fit_measures(data):
    bp = fit_measures(pca(biplot(data)))
    eig = bp.eigenvalues
    np.testing.assert_allclose(bp.quality, eig[:2].sum() / eig.sum())
    for vec in (bp.adequacy, bp.axis_predictivity, bp.sample_predictivity):
        assert np.all(vec >= -1e-12) and np.all(vec <= 1 + 1e-12)
    # adequacy: squared loadings of the displayed pair
    np.testing.assert_allclose(bp.adequacy, np.sum(bp.Vr**2, axis=1))


def test_pca_fit_measures_full_rank_is_perfect(data):
    bp = fit_measures(pca(biplot(data.iloc[:, :2])))  # p == dim_biplot == 2
    np.testing.assert_allclose(bp.quality, 1.0)
    np.testing.assert_allclose(bp.axis_predictivity, np.ones(2), atol=1e-10)
    np.testing.assert_allclose(bp.sample_predictivity, np.ones(bp.n), atol=1e-10)


def test_pca_invalid_e_vects(data):
    with pytest.raises(ValueError, match="between 1 and"):
        pca(biplot(data), e_vects=(0, 1))
    with pytest.raises(ValueError, match="between 1 and"):
        pca(biplot(data), e_vects=(1, 9))
