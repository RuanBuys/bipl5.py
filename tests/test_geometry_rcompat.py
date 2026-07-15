import numpy as np
import pytest

from bipl5.geometry import (
    ellipse_points,
    get_quads_axes,
    move_lines,
    mvee,
    obtain_zhat,
    rotation_constructor,
    shorten_axes,
)
from bipl5.rcompat import adjustcolor, approx, bw_nrd0, density_r


def test_rotation_constructor_roundtrip():
    angles = np.array([0.3, -1.1, 2.0])
    rot = rotation_constructor(angles)
    back = rotation_constructor(-angles)
    assert rot.shape == (2, 6)
    pts = np.random.default_rng(0).normal(size=(10, 2))
    for i in range(3):
        fwd = pts @ rot[:, 2 * i : 2 * i + 2]
        rev = fwd @ back[:, 2 * i : 2 * i + 2]
        np.testing.assert_allclose(rev, pts, atol=1e-12)


def test_mvee_contains_all_points():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(80, 2)) @ np.array([[2.0, 0.5], [0.0, 0.7]])
    center, A = mvee(pts)
    d = np.einsum("ij,jk,ik->i", pts - center, A, pts - center)
    assert np.all(d <= 1 + 1e-4)
    # the ellipse is tight: some point lies essentially on the boundary
    assert d.max() > 0.98


def test_ellipse_points_on_boundary():
    center = np.array([1.0, -2.0])
    A = np.diag([1 / 4.0, 1 / 9.0])  # radii 2 and 3
    pts = ellipse_points(center, A, n_out=51)
    d = np.einsum("ij,jk,ik->i", pts - center, A, pts - center)
    np.testing.assert_allclose(d, 1.0, atol=1e-10)


def test_quadrants():
    z_axes = [
        np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 5.0]]),     # up-right -> 1
        np.array([[-1.0, 1.0, 0.0], [-2.0, 2.0, 5.0]]),   # up-left  -> 2
        np.array([[-1.0, -1.0, 0.0], [-2.0, -2.0, 5.0]]), # down-left -> 3
        np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 5.0]]),   # down-right -> 4
    ]
    np.testing.assert_array_equal(get_quads_axes(z_axes), [1, 2, 3, 4])


def test_obtain_zhat_interpolates_linearly():
    ax = np.array([[0.0, 0.0, 10.0], [2.0, 0.0, 30.0]])
    ranges = np.array([[1.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(obtain_zhat(ranges, ax), [20.0, 40.0])


def test_shorten_axes_keeps_axis_on_line():
    ticks = np.linspace(-10, 10, 21)
    z_axes = [np.column_stack([ticks, 0.5 * ticks, ticks])]
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.column_stack([3 * np.cos(theta), 2 * np.sin(theta)])
    trimmed = shorten_axes(z_axes, ellipse)[0]
    assert trimmed.shape[0] < 21
    # still collinear with the original direction (skip the origin tick)
    nonzero = np.abs(trimmed[:, 0]) > 1e-12
    np.testing.assert_allclose(
        trimmed[nonzero, 1] / trimmed[nonzero, 0], 0.5, atol=1e-9
    )
    # labels preserved from the original tick set
    assert set(np.round(trimmed[:, 2], 9)).issubset(set(np.round(ticks, 9)))


def test_move_lines_translates_out_of_ellipse():
    ticks = np.linspace(-4, 4, 9)
    z_axes = [
        np.column_stack([ticks, 0.4 * ticks, ticks]),
        np.column_stack([ticks, -0.7 * ticks, ticks]),
    ]
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.column_stack([4 * np.cos(theta), 3 * np.sin(theta)])
    quads = get_quads_axes(z_axes)
    out = move_lines(ellipse, [0.4, -0.7], quads, 1.0, z_axes, False, ["a", "b"])
    assert len(out["ends"]) == 2
    for ends in out["ends"]:
        # every translated tick lies outside the ellipse
        d = ends[:, 0] ** 2 / 16 + ends[:, 1] ** 2 / 9
        assert np.all(d > 1.0)


def test_bw_nrd0_matches_r():
    # R: bw.nrd0(1:10) = 0.9 * min(sd(1:10), IQR(1:10)/1.34) * 10^(-1/5)
    #                  = 0.9 * 3.0276504 * 0.6309573 = 1.7192864
    assert bw_nrd0(np.arange(1, 11)) == pytest.approx(1.7192864, abs=1e-6)


def test_density_r_integrates_to_one():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    grid, dens = density_r(x, from_=-6, to=6, n=512)
    integral = np.trapezoid(dens, grid)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_approx_clamps_like_rule_2():
    y = approx([0, 1], [10, 20], [-1, 0.5, 2])
    np.testing.assert_allclose(y, [10, 15, 20])


def test_adjustcolor():
    assert adjustcolor("#FF0000", 0.5) == "#FF000080"
    assert adjustcolor("#FF0000", 1.0) == "#FF0000FF"
    with pytest.raises(ValueError):
        adjustcolor("red", 0.5)
