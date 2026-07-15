import numpy as np
import pytest

from bipl5.ordination import (
    EZ_COL,
    GREY_07,
    axes,
    axes_coordinates,
    biplot,
    cumulative_adequacies,
    cumulative_axis_predictivities,
    cva,
    fit_quality_string,
    marginal_axis_predictivities,
    means,
    pca,
    pco,
    samples,
)
from bipl5.rcompat import pretty


def test_axes_coordinates_shape_and_reading(data):
    bp = pca(biplot(data))
    z_axes = axes_coordinates(bp)
    assert len(z_axes) == bp.p

    for j, ax in enumerate(z_axes):
        assert ax.shape[1] == 3
        # ticks are collinear with the axis direction through the origin
        d = bp.ax_one_unit[j]
        cross = ax[:, 0] * d[1] - ax[:, 1] * d[0]
        np.testing.assert_allclose(cross, 0, atol=1e-10)
        # projecting a tick onto v_j and back-transforming returns its label
        read = (ax[:, :2] @ bp.Vr[j]) * bp.sd[j] + bp.means[j]
        np.testing.assert_allclose(read, ax[:, 2], atol=1e-8)
        # labels are evenly spaced and sorted
        steps = np.diff(ax[:, 2])
        np.testing.assert_allclose(steps, steps[0], atol=1e-9)


def test_axes_coordinates_contains_pretty_ticks(data):
    bp = pca(biplot(data))
    pred = bp.Z @ np.linalg.inv(bp.Lmat)[bp.e_idx, :] + bp.means
    z_axes = axes_coordinates(bp)
    for j, ax in enumerate(z_axes):
        expected = pretty(
            (pred[:, j].min(), pred[:, j].max()), n=5
        )  # biplotEZ default: 5 ticks
        assert set(np.round(expected, 9)).issubset(set(np.round(ax[:, 2], 9)))


def test_axes_coordinates_works_for_cva_and_scaled(data, groups):
    bp = cva(biplot(data, scaled=True), classes=groups)
    z_axes = axes_coordinates(bp)
    assert len(z_axes) == bp.p
    for ax in z_axes:
        assert np.all(np.isfinite(ax))


def test_samples_defaults(data, groups):
    bp = samples(pca(biplot(data, group_aes=groups)))
    assert bp.samples["col"] == EZ_COL[:3]
    assert bp.samples["pch"] == [16, 16, 16]
    assert bp.samples["which"] == [0, 1, 2]


def test_axes_defaults(data):
    bp = axes(pca(biplot(data)))
    assert bp.axes["col"] == [GREY_07] * 4
    assert bp.axes["ticks"] == [5] * 4
    assert bp.axes["tick_col"] == [GREY_07] * 4
    assert bp.axes["tick_label_col"] == [GREY_07] * 4
    assert bp.axes["ax_names"] == ["alpha", "beta", "gamma", "delta"]


def test_axes_recycling(data):
    bp = axes(pca(biplot(data)), col=["red", "blue"])
    assert bp.axes["col"] == ["red", "blue", "red", "blue"]


def test_means_defaults_fill_zmeans(data, groups):
    bp = means(pca(biplot(data, group_aes=groups)))
    assert bp.means_aes["pch"] == [15, 15, 15]
    assert bp.means_aes["col"] == EZ_COL[:3]
    assert bp.Zmeans is not None and bp.Zmeans.shape == (3, 2)


def test_fit_quality_string_formatting():
    s = fit_quality_string(np.array([8.0, 4.0, 2.0, 2.0]), (1, 2))
    assert s == "Quality of display = 75% = 50% (PC1) + 25% (PC2)"
    assert fit_quality_string(None, (1, 2)) == ""
    s2 = fit_quality_string(np.array([8.0, 4.0, 2.0, 2.0]), (1, 2), dim_prefix="CV")
    assert "(CV1)" in s2 and "(CV2)" in s2


def test_method_chaining(data, groups):
    result = (
        biplot(data, group_aes=groups)
        .pca(e_vects=(1, 2), show_class_means=True)
        .with_samples()
        .with_axes()
        .fit_measures()
    )
    assert result.method == "pca"
    assert result.samples is not None
    assert result.axes is not None
    assert result.quality is not None
    ticks = result.axes_coordinates()
    assert len(ticks) == 4


def test_cumulative_measures(data):
    bp = pca(biplot(data))
    cum_pred = cumulative_axis_predictivities(bp)
    assert cum_pred.shape == (5, 4)
    # cumulative in rank: rows non-decreasing, rank p is perfect
    assert np.all(np.diff(cum_pred, axis=1) >= -1e-9)
    np.testing.assert_allclose(cum_pred[:, -1], 1.0, atol=1e-9)

    cum_adeq = cumulative_adequacies(bp)
    np.testing.assert_allclose(cum_adeq[:, -1], 1.0, atol=1e-9)

    marg = marginal_axis_predictivities(bp)
    assert marg.shape == (4,)
    # marginal (PC1&2) predictivity equals the cumulative value at rank 2
    np.testing.assert_allclose(np.round(marg, 3), cum_pred[:4, 1], atol=1e-9)


def test_fit_measures_unsupported_method(data):
    bp = pco(biplot(data))
    with pytest.raises(ValueError, match="PCA and CVA"):
        bp.fit_measures()
