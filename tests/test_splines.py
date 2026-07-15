"""Spline axes for PCO biplots — port of biplotEZ 2.3's C++-backed
``biplot.spline.axis()`` (GitHub version, not yet on CRAN)."""

import numpy as np
import pandas as pd
import pytest

import bipl5
from bipl5.ordination import biplot, pco
from bipl5.ordination.splines import (
    _amoeba,
    _varset,
    alfunc,
    bs_basis,
    spline_axis,
    spline_axis_control,
)

TINY_CONTROL = {
    "gamma": 2,
    "bigsigmaactivate": 1,
    "nmu": 25,
    "itmax": 200,
    "seed": 1,
    "verbose": False,
}


@pytest.fixture
def curved() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    t = rng.uniform(-2, 2, size=40)
    return pd.DataFrame(
        {
            "a": t + rng.normal(scale=0.2, size=40),
            "b": t**2 + rng.normal(scale=0.3, size=40),
            "c": np.sin(t) + rng.normal(scale=0.2, size=40),
        }
    )


# ── basis and loss ──────────────────────────────────────────────────────────

def test_bs_basis_boundary_behaviour():
    x = np.linspace(0.0, 1.0, 41)
    M = bs_basis(x, degree=2, interior_knots=[0.3, 0.5, 0.7])
    assert M.shape == (41, 5)  # len(knots) + degree, intercept dropped
    assert np.all((M >= -1e-12) & (M <= 1 + 1e-12))
    # at the lower boundary only the dropped first basis is active
    np.testing.assert_allclose(M[0], 0.0, atol=1e-12)
    # at the upper boundary the last basis equals one
    np.testing.assert_allclose(M[-1, -1], 1.0)
    np.testing.assert_allclose(M[-1, :-1], 0.0, atol=1e-12)


def test_bs_basis_df_mode_matches_knot_mode():
    x = np.linspace(-1.0, 1.0, 30)
    by_df = bs_basis(x, degree=2, df=5)
    knots = np.quantile(x, np.linspace(0, 1, 5)[1:-1])
    by_knots = bs_basis(x, degree=2, interior_knots=knots)
    np.testing.assert_allclose(by_df, by_knots, atol=1e-12)


def test_alfunc_hand_computed():
    # curve rows: (0,0), (1,0), (2,0) for mu = 0, 1, 2
    M = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    bvec = np.array([1.0, 2.0, 0.0, 0.0])  # column-major B = [[1,0],[2,0]]
    mu = np.array([0.0, 1.0, 2.0])
    X = np.array([[1.1, 0.0]])  # nearest curve point is mu = 1
    y = np.array([3.0])
    const1 = 4.0
    # loss = (3 - 1)^2 / 4 = 1
    assert alfunc(bvec, X, y, M, mu, 0.0, const1, 1.0) == pytest.approx(1.0)
    # smoothness penalty: second difference of Z columns
    withpen = alfunc(bvec, X, y, M, mu, 2.0, const1, 1.0)
    z = M @ np.array([[1.0, 0.0], [2.0, 0.0]])
    pen = 2.0 * float(np.sum((z[0] - 2 * z[1] + z[2]) ** 2)) / 1.0
    assert withpen == pytest.approx(1.0 + pen)


def test_amoeba_minimizes_and_flags_dimension_limit():
    target = np.array([3.0, -1.0, 2.0])

    def loss(b):
        return float(np.sum((b - target) ** 2))

    simplex = _varset(np.zeros(3), tau=0.5)
    yvek = np.array([loss(v) for v in simplex])
    iters, erro = _amoeba(simplex, yvek, loss, ftol=1e-12, itmax=5000, tiny=1e-30)
    assert erro == 0
    np.testing.assert_allclose(simplex[0], target, atol=1e-4)
    assert yvek[0] == pytest.approx(0.0, abs=1e-8)

    big = np.zeros((22, 21))
    _, erro = _amoeba(big, np.zeros(22), loss, 1e-8, 100, 1e-30)
    assert erro == 5


# ── the axis driver ─────────────────────────────────────────────────────────

def test_spline_axis_output_contract(curved):
    ez = pco(biplot(curved), axes="splines", spline_control=TINY_CONTROL)
    ax = spline_axis(1, ez.Z, ez.X, ez.means, ez.sd, control=ez.spline_control)

    assert ax.shape[1] == 4
    flags = ax[:, 3]
    assert set(np.unique(flags)) <= {0.0, 1.0}
    # flagged rows carry the pretty tick values of the raw variable
    from bipl5.rcompat import pretty

    raw = curved["b"].to_numpy()
    expected = pretty((raw.min(), raw.max()))
    flagged = ax[flags == 1, 2]
    assert set(np.round(flagged, 9)) <= set(np.round(expected, 9))
    # the curve passes through the origin at the variable's mean
    mu = (ax[:, 2] - ez.means[1]) / ez.sd[1]
    at_mean = int(np.argmin(np.abs(mu)))
    np.testing.assert_allclose(ax[at_mean, :2], [0.0, 0.0], atol=1e-10)
    # tick values increase along the curve rows
    assert np.all(np.diff(ax[:, 2]) > 0)


def test_spline_axis_deterministic_with_seed(curved):
    ez = pco(biplot(curved), axes="splines", spline_control=TINY_CONTROL)
    one = spline_axis(0, ez.Z, ez.X, ez.means, ez.sd, control=ez.spline_control)
    two = spline_axis(0, ez.Z, ez.X, ez.means, ez.sd, control=ez.spline_control)
    np.testing.assert_allclose(one, two)


def test_spline_axis_subsamples_large_data():
    rng = np.random.default_rng(3)
    t = rng.uniform(-1, 1, size=150)
    frame = pd.DataFrame({"a": t, "b": t**2, "c": -t})
    ez = pco(biplot(frame), axes="splines", spline_control=TINY_CONTROL)
    ax = spline_axis(0, ez.Z, ez.X, ez.means, ez.sd, control=ez.spline_control)
    assert ax.shape[1] == 4 and np.all(np.isfinite(ax[:, :3]))


def test_control_validation():
    with pytest.raises(ValueError, match="tau"):
        spline_axis_control(tau=0.0)
    with pytest.raises(ValueError, match="lambda"):
        spline_axis_control(lambda_=-1)
    with pytest.raises(ValueError, match="gamma"):
        spline_axis_control(gamma=3)  # bigsigmaactivate floors to 0, like R
    ctrl = spline_axis_control(gamma=3, bigsigmaactivate=1)
    assert ctrl["bigsigma"] == pytest.approx(0.1)


# ── engine and pipeline wiring ──────────────────────────────────────────────

def test_pco_spline_engine_state(curved):
    ez = pco(biplot(curved), axes="splines", spline_control=TINY_CONTROL)
    assert ez.PCOaxes == "splines"
    assert ez.ax_one_unit is None  # only regression axes set directions
    assert ez.spline_control["nmu"] == 25
    z_axes = ez.axes_coordinates()
    assert len(z_axes) == 3
    assert all(ax.shape[1] == 4 for ax in z_axes)
    with pytest.raises(ValueError, match="regression"):
        pco(biplot(curved), axes="nope")


def test_scale_mds_spline_pipeline(curved):
    bp = bipl5.init_biplot(curved).scale_mds(
        "pco", axes="splines", spline_control=TINY_CONTROL
    )
    assert bp.biplot_type == "pco"
    assert bp.meta["spline"] is True
    assert bp.meta["fit_quality"] == ""

    d = bp.displays["mdsDisplay_12"]
    metas: dict[str, int] = {}
    for tr in d["mds"]["trace_data"]:
        key = tr["meta"] if isinstance(tr["meta"], str) else tr["meta"][0]
        metas[key] = metas.get(key, 0) + 1
    p = 3
    assert metas["data"] == 1
    assert metas["axis"] == 2 * p  # curve + name label per axis
    assert metas["circle"] == 1
    assert "ExpAx" not in metas and "density" not in metas
    assert "slider_info" not in d["mds"]["config"]
    # bare hover text (no Actual/Pred table for spline displays)
    assert "Actual" not in d["mds"]["trace_data"][0]["hovertext"][0]
    # curve traces carry gradients (NaN at ends) and rounded hover ticks
    curve = next(
        tr for tr in d["mds"]["trace_data"]
        if not isinstance(tr["meta"], str) and tr["meta"][0] == "axis"
    )
    grads = np.asarray(curve["customdata"], dtype=float)
    assert np.isnan(grads[0]) and np.isnan(grads[-1])

    widget = bp.plot()
    assert widget.js == "spline"
    assert widget.payload == {"p": 3}
    html = widget.to_html()
    assert ("bipl5SplineAttach" in html) is True
    assert ("bipl5Attach(" not in html) is True


def test_spline_verbs(curved):
    bp = bipl5.init_biplot(curved).scale_mds(
        "pco", axes="splines", spline_control=TINY_CONTROL
    )
    with pytest.warns(UserWarning, match="spline"):
        unchanged = bp.score_axes()
    assert unchanged is bp
    with pytest.raises(ValueError, match="not supported"):
        bp.append_mds_display((1, 3))

    # format_samples still restratifies the sample layer (no TDA to rebuild)
    frame = curved.copy()
    frame["grp"] = ["x", "y"] * 20
    bp2 = bipl5.init_biplot(frame).scale_mds(
        "pco", axes="splines", spline_control=TINY_CONTROL
    )
    out = bp2.format_samples(stratify="col", by="grp")
    names = [
        tr["name"]
        for tr in out.displays["mdsDisplay_12"]["mds"]["trace_data"]
        if not isinstance(tr["meta"], str) and tr["meta"][0] == "data"
    ]
    assert names == ["x", "y"]
