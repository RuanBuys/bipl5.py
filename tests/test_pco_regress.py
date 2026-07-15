import numpy as np
import pytest

from bipl5.ordination import biplot, euclidean_dist, pca, pco, regress

from conftest import align_signs


def test_pco_euclidean_recovers_pca(data):
    """Classical MDS on Euclidean distances of centered data == PCA scores."""
    ref = pca(biplot(data))
    bp = pco(biplot(data))
    np.testing.assert_allclose(align_signs(ref.Z, bp.Z), ref.Z, atol=1e-8)
    # nonzero eigenvalues of B = XX' equal those of X'X
    np.testing.assert_allclose(
        bp.eigenvalues[:4], ref.eigenvalues, atol=1e-8
    )
    # regression axis directions coincide with PCA's v_j / ||v_j||^2
    assert bp.ax_one_unit.shape == (4, 2)
    np.testing.assert_allclose(
        np.abs(bp.ax_one_unit), np.abs(ref.ax_one_unit), atol=1e-8
    )


def test_pco_dmat_equals_dist_func(data):
    D = euclidean_dist(biplot(data).X)
    via_dmat = pco(biplot(data), Dmat=D)
    via_func = pco(biplot(data))
    np.testing.assert_allclose(via_dmat.Z, via_func.Z, atol=1e-10)


def test_pco_non_euclidean_warns(data):
    D = euclidean_dist(biplot(data).X) ** 2  # squared distances: not embeddable
    with pytest.warns(UserWarning, match="Euclidean embeddable"):
        pco(biplot(data), Dmat=D)


def test_pco_spline_axes_not_implemented(data):
    with pytest.raises(NotImplementedError, match="[Ss]pline"):
        pco(biplot(data), axes="splines")


def test_pco_class_means(data, groups):
    bp = pco(biplot(data, group_aes=groups), show_class_means=True)
    assert bp.Zmeans.shape == (3, 2)
    for i in range(3):
        np.testing.assert_allclose(
            bp.Zmeans[i], bp.Z[bp.group == i].mean(axis=0), atol=1e-10
        )


def test_regress_on_pca_scores_recovers_loadings(data):
    ref = pca(biplot(data))
    bp = regress(biplot(data), Z=ref.Z)
    # (Z'Z)^-1 Z'X == Vr' when Z are the PCA scores of X
    np.testing.assert_allclose(bp.Vr, ref.Vr, atol=1e-10)
    np.testing.assert_allclose(bp.ax_one_unit, ref.ax_one_unit, atol=1e-10)
    assert bp.Lmat is None and bp.eigenvalues is None
    assert bp.method == "regress"
    assert bp.e_vects == (1, 2)


def test_regress_centers_z(data):
    ref = pca(biplot(data))
    shifted = ref.Z + np.array([3.0, -8.0])
    bp = regress(biplot(data), Z=shifted)
    np.testing.assert_allclose(bp.Z.mean(axis=0), [0, 0], atol=1e-10)
    np.testing.assert_allclose(bp.Z, ref.Z, atol=1e-10)


def test_regress_group_means(data, groups):
    ref = pca(biplot(data))
    bp = regress(biplot(data, group_aes=groups), Z=ref.Z)
    assert bp.class_means is True  # show_group_means defaults to True
    assert bp.Zmeans.shape == (3, 2)


def test_regress_dimension_checks(data):
    with pytest.raises(ValueError, match="one row per observation"):
        regress(biplot(data), Z=np.zeros((3, 2)))
