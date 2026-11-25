"""Static matplotlib biplots — port of the R package's
``test-wrap-bipl5-gg.R`` plus checks specific to the matplotlib renderer."""

import math
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import bipl5  # noqa: E402
from bipl5.ordination import GREY_07, axes, axes_coordinates, biplot, cva, pca, pco, regress  # noqa: E402
from bipl5.static import (  # noqa: E402
    AxisStyle,
    CalibratedAxes,
    CalibratedAxis,
    StaticBiplot,
    add_calibrated_axes,
    axis_calibration,
    marker_style,
    plot_static,
)
from bipl5.static.helpers import (  # noqa: E402
    curve_angles,
    extend_axis_values,
    fold_angle,
    format_tick_values,
    line_window_span,
    r_linetype,
    resolve_axis_aes,
)

TINY_SPLINE = {"gamma": 2, "bigsigmaactivate": 1, "nmu": 25, "seed": 1}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def frame(data, groups) -> pd.DataFrame:
    out = data.copy()
    out["grp"] = groups
    return out


@pytest.fixture
def pca_ez(data):
    return pca(biplot(data, scaled=True))


@pytest.fixture
def pca_bp(frame):
    return bipl5.init_biplot(frame, scale=True).scale_mds("pca", classes=frame["grp"])


def _draw(sb: StaticBiplot):
    sb.figure.canvas.draw()
    return sb.calibrated_axes.last_render


# ── input validation ────────────────────────────────────────────────────────


def test_plot_static_validates_its_input(data, pca_ez):
    with pytest.raises(TypeError, match="Biplot"):
        plot_static(data)
    with pytest.raises(ValueError, match="Apply pca\\(\\), cva\\(\\)"):
        plot_static(biplot(data))
    with pytest.raises(ValueError, match="two dimensions"):
        plot_static(pca(biplot(data), dim_biplot=3))
    with pytest.raises(ValueError, match="'display' only applies"):
        plot_static(pca_ez, display="mdsDisplay_12")


def test_builds_a_figure_with_a_calibrated_axis_layer(pca_ez):
    sb = pca_ez.plot_static()
    assert isinstance(sb, StaticBiplot)
    assert isinstance(sb.calibrated_axes, CalibratedAxes)
    assert sb.calibrated_axes in sb.ax.get_children()
    # a biplot is only readable when both display dimensions share a scale
    assert sb.ax.get_aspect() == 1.0
    render = _draw(sb)
    assert len(render["lines"]) == 4
    assert render["marks"]
    assert "StaticBiplot" in repr(sb)


# ── calibration recovery ───────────────────────────────────────────────────


def test_recovered_calibration_reproduces_the_axis_coordinates(pca_ez):
    for ax in axes_coordinates(axes(pca_ez)):
        cal = axis_calibration(ax)
        assert cal is not None
        x, y = cal.position(ax[:, 2])
        np.testing.assert_allclose(x, ax[:, 0], atol=1e-8)
        np.testing.assert_allclose(y, ax[:, 1], atol=1e-8)
        assert cal.offset == (0.0, 0.0)
        assert cal.vstep > 0


def test_axis_calibration_rejects_degenerate_axes():
    assert axis_calibration(np.zeros((1, 3))) is None
    assert axis_calibration(np.zeros((4, 2))) is None
    # constant marker values carry no calibration
    assert axis_calibration(np.column_stack([np.arange(4), np.arange(4), np.full(4, 2)])) is None
    # a zero-length axis is degenerate even when the values vary
    assert axis_calibration(np.column_stack([np.zeros(4), np.zeros(4), np.arange(4)])) is None


# ── the plotting window ────────────────────────────────────────────────────


def test_calibrated_axes_never_widen_the_plotting_window(pca_ez):
    z_axes = axes_coordinates(axes(pca_ez))
    axis_extent = max(np.abs(a[:, :2]).max() for a in z_axes)
    Z = pca_ez.Z

    sb = pca_ez.plot_static()
    (x0, x1), (y0, y1) = sb.ax.get_xlim(), sb.ax.get_ylim()

    # markers reach well beyond the samples, yet the window follows them
    assert axis_extent > np.abs(Z).max()
    assert max(abs(x0), abs(x1), abs(y0), abs(y1)) < axis_extent
    span_x, span_y = np.ptp(Z[:, 0]), np.ptp(Z[:, 1])
    np.testing.assert_allclose((x0, x1), (Z[:, 0].min() - 0.05 * span_x, Z[:, 0].max() + 0.05 * span_x))
    np.testing.assert_allclose((y0, y1), (Z[:, 1].min() - 0.05 * span_y, Z[:, 1].max() + 0.05 * span_y))
    _draw(sb)
    assert sb.ax.get_xlim() == (x0, x1)  # drawing never re-scales


def test_window_options(pca_ez):
    Z = pca_ez.Z
    exact = pca_ez.plot_static(xlim=(-2, 2), ylim=(-1, 1), expand=False)
    assert exact.ax.get_xlim() == (-2, 2) and exact.ax.get_ylim() == (-1, 1)

    sq = pca_ez.plot_static(square=True)
    (x0, x1), (y0, y1) = sq.ax.get_xlim(), sq.ax.get_ylim()
    assert math.isclose(x1 - x0, y1 - y0)

    # biplotEZ's window: range(Z * exp.factor), unexpanded
    ez_win = pca_ez.plot_static(exp_factor=1.2)
    np.testing.assert_allclose(ez_win.ax.get_xlim(), (1.2 * Z[:, 0].min(), 1.2 * Z[:, 0].max()))


# ── the markers ────────────────────────────────────────────────────────────


def _marks_by_axis(render):
    out = {}
    for m in render["marks"]:
        out.setdefault(m["axis"], []).append(m)
    return out


def test_drawn_markers_lie_on_the_calibration_grid_and_reach_the_edge(pca_ez):
    z_axes = axes_coordinates(axes(pca_ez))
    sb = pca_ez.plot_static()
    render = _draw(sb)
    xr, yr = render["xrange"], render["yrange"]

    for i, marks in _marks_by_axis(render).items():
        ax = z_axes[i]
        cal = axis_calibration(ax)
        values = np.array([float(m["label"]) for m in marks])
        x = np.array([m["x"] for m in marks])
        y = np.array([m["y"] for m in marks])

        # every marker sits exactly where the calibration puts it ...
        np.testing.assert_allclose(x, cal.x0 + values * cal.dxdv, atol=1e-8)
        np.testing.assert_allclose(y, cal.y0 + values * cal.dydv, atol=1e-8)
        # ... its value belongs to the marker grid biplotEZ chose ...
        offsets = (values - ax[0, 2]) / cal.vstep
        np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-8)
        # ... markers are confined to the window ...
        assert np.all((x >= xr[0] - 1e-8) & (x <= xr[1] + 1e-8))
        assert np.all((y >= yr[0] - 1e-8) & (y <= yr[1] + 1e-8))
        # ... and they run to both edges: one more step either way leaves it
        lo, hi = line_window_span((cal.x0, cal.y0), cal.direction, xr, yr)
        assert values.min() - cal.vstep < lo
        assert values.max() + cal.vstep > hi


def test_tick_extend_reaches_edges_the_supplied_markers_miss(pca_ez):
    # in a wide window biplotEZ's markers (pretty range +- one range) stop
    # well short of the border; regenerated markers do not
    window = dict(xlim=(-25, 25), ylim=(-25, 25), expand=False)
    extended = _draw(pca_ez.plot_static(**window))
    supplied = _draw(pca_ez.plot_static(tick_extend=False, **window))
    for i, marks in _marks_by_axis(extended).items():
        cal = axis_calibration(axes_coordinates(axes(pca_ez))[i])
        lo, hi = line_window_span((cal.x0, cal.y0), cal.direction, (-25, 25), (-25, 25))
        values = [m["value"] for m in marks]
        assert min(values) - cal.vstep < lo and max(values) + cal.vstep > hi
        short = [m["value"] for m in supplied["marks"] if m["axis"] == i]
        assert len(short) < len(values)
        assert min(short) - cal.vstep > lo or max(short) + cal.vstep < hi


def test_markers_follow_the_window_after_the_fact(pca_ez):
    sb = pca_ez.plot_static()
    sb.ax.set_xlim(0, 4)
    sb.ax.set_ylim(-0.5, 1.5)
    render = _draw(sb)
    assert render["xrange"] == (0, 4)
    for i, marks in _marks_by_axis(render).items():
        cal = sb.calibrations[i]
        values = np.array([m["value"] for m in marks])
        lo, hi = line_window_span((cal.x0, cal.y0), cal.direction, (0, 4), (-0.5, 1.5))
        assert values.min() - cal.vstep < lo and values.max() + cal.vstep > hi


def test_tick_extend_false_keeps_only_the_supplied_markers(pca_ez):
    z_axes = axes_coordinates(axes(pca_ez))
    sb = pca_ez.plot_static(tick_extend=False, xlim=(-6, 6), ylim=(-6, 6))
    render = _draw(sb)
    for i, marks in _marks_by_axis(render).items():
        supplied = np.round(z_axes[i][:, 2], 10)
        assert all(round(m["value"], 10) in supplied for m in marks)


def test_marker_extension_falls_back_when_it_cannot_run():
    # a marker grid so fine (median step 1e-6) that spanning the window
    # would need millions of markers: fall back to the supplied ones
    values = np.array([0, 1e-6, 2e-6, 3e-6, 1.0])
    coords = np.column_stack([values, np.zeros(5), values])
    rec = CalibratedAxis.from_coordinates(coords, "fine")
    assert rec.calibration.vstep == pytest.approx(1e-6)
    assert extend_axis_values(rec.calibration, (-3, 3), (-3, 3)) is None

    fig, ax = plt.subplots()
    artist = CalibratedAxes([rec])
    ax.add_artist(artist)
    layout = artist.compute_layout((-3, 3), (-3, 3))
    np.testing.assert_allclose(sorted(m["value"] for m in layout["marks"]), values)

    # an axis that misses the window contributes nothing
    far = artist.compute_layout((500, 501), (500, 501))
    assert far["marks"] == [] and far["lines"] == [] and far["titles"] == []


# ── aesthetics ─────────────────────────────────────────────────────────────


def _styles(sb):
    return [a.style.resolved() for a in sb.calibrated_axes.calibrated_axes]


def test_axis_aesthetics_follow_the_arguments_and_can_defer_to_biplotez(pca_ez):
    ez = axes(pca_ez, col="steelblue", tick_label_col="firebrick")

    default = _styles(ez.plot_static())
    assert {s.axis_colour for s in default} == {"black"}
    assert {s.text_colour for s in default} == {"black"}

    deferred = _styles(ez.plot_static(axis_colour="biplotEZ", text_colour="biplotEZ"))
    assert {s.axis_colour for s in deferred} == {"steelblue"}
    assert {s.text_colour for s in deferred} == {"firebrick"}

    # tick and label colours default to the axis colour
    custom = _styles(ez.plot_static(axis_colour="darkgreen"))
    assert {s.tick_colour for s in custom} == {"darkgreen"}
    assert {s.label_colour for s in custom} == {"darkgreen"}

    # the biplotEZ default axis colour is grey(0.7)
    assert {s.axis_colour for s in _styles(pca_ez.plot_static(axis_colour="biplotEZ"))} == {GREY_07}

    # one value per axis
    per_axis = _styles(pca_ez.plot_static(axis_colour=["red", "blue", "green", "black"]))
    assert [s.axis_colour for s in per_axis] == ["red", "blue", "green", "black"]


def _pixels_of(sb, rgb) -> int:
    sb.figure.set_dpi(150)
    sb.figure.canvas.draw()
    img = np.asarray(sb.figure.canvas.buffer_rgba())[..., :3].astype(int)
    return int(np.sum(np.all(np.abs(img - np.array(rgb)) <= 8, axis=-1)))


def test_axis_halo_outlines_the_calibrated_axes(pca_ez):
    green = (0, 255, 0)
    plain = pca_ez.plot_static(legend=False, caption="")
    assert {s.halo_colour for s in _styles(plain)} == {None}
    assert _pixels_of(plain, green) == 0

    haloed = pca_ez.plot_static(legend=False, caption="", halo_colour="#00ff00", halo_width=3)
    assert {s.halo_colour for s in _styles(haloed)} == {"#00ff00"}
    assert {s.halo_width for s in _styles(haloed)} == {3.0}
    assert _pixels_of(haloed, green) > 500

    # one value per axis; None leaves that axis plain
    per_axis = _styles(pca_ez.plot_static(halo_colour=["white", None, "black", None]))
    assert [s.halo_colour for s in per_axis] == ["white", None, "black", None]

    # the path effects: a wider stroke under lines, an outline around text
    style = AxisStyle(axis_linewidth=1.5, halo_colour="white", halo_width=2.5)
    stroke, normal = style.line_halo(1.5)
    assert stroke._gc["linewidth"] == pytest.approx(4.0)
    (outline,) = style.text_halo()
    assert outline._gc["linewidth"] == pytest.approx(2.5)
    assert AxisStyle().line_halo(1.0) is None and AxisStyle().text_halo() is None
    assert AxisStyle(halo_colour="k", halo_width=0).line_halo(1.0) is None


def test_axis_subsets_and_marker_density_follow_the_axes_call(pca_ez, data):
    sb = axes(pca_ez, which=[0, 2]).plot_static()
    names = [a.name for a in sb.calibrated_axes.calibrated_axes]
    assert names == [data.columns[0], data.columns[2]]

    coarse = axes_coordinates(axes(pca_ez, ticks=3))
    fine = axes_coordinates(axes(pca_ez, ticks=10))
    assert axis_calibration(coarse[0]).vstep > axis_calibration(fine[0]).vstep


# ── every biplot family ────────────────────────────────────────────────────


def test_renders_every_supported_biplot_family(data, groups):
    cv = cva(biplot(data), classes=groups).plot_static()
    _draw(cv)
    assert cv.ax.get_xlabel().startswith("CV1 (")
    assert len(cv.class_means) == 3

    Z = pca(biplot(data)).Z
    reg = regress(biplot(data), Z).plot_static()
    _draw(reg)
    assert reg.ax.get_xlabel() == "Dim1"
    assert reg.titles[-1].get_text().startswith("R^2_disp")

    pc = pco(biplot(data), axes="regression").plot_static()
    _draw(pc)
    assert pc.ax.get_xlabel().startswith("Dim1")


def test_spline_pco_biplots_are_drawn_as_curved_axes(data):
    ez = pco(biplot(data), axes="splines", spline_control=TINY_SPLINE)
    sb = ez.plot_static()
    artist = sb.calibrated_axes
    assert artist.tick_extend is False
    recs = artist.calibrated_axes
    assert all(r.curved for r in recs)
    # curve vertices outnumber the labelled markers, each with its own angle
    assert sum(r.x.size for r in recs) > sum(int(r.tick.sum()) for r in recs)
    assert len(np.unique(np.round(np.concatenate([r.angle for r in recs]), 4))) > 1
    render = _draw(sb)
    assert len(render["lines"]) == len(recs)


def test_spline_biplot_reuses_the_stored_curves(frame):
    bp = bipl5.init_biplot(frame).scale_mds("pco", axes="splines", spline_control=TINY_SPLINE)
    assert len(bp.meta["spline_axes"]) == 4
    sb = bp.plot_static()
    for rec, stored in zip(sb.calibrated_axes.calibrated_axes, bp.meta["spline_axes"]):
        np.testing.assert_allclose(rec.x, stored[:, 0])


# ── samples, legends, means and polygons ───────────────────────────────────


def test_group_aesthetics_reach_the_plot(data, groups, pca_ez):
    grouped = pca(biplot(data, group_aes=groups)).plot_static(legend_title="Species")
    assert len(grouped.samples) == 3
    assert grouped.legends[0].get_title().get_text() == "Species"
    assert grouped.calibrated_axes.zorder < grouped.samples[0].get_zorder()

    default = pca(biplot(data, group_aes=groups)).plot_static()
    assert default.legends[0].get_title().get_text() == "Group"

    ungrouped = pca_ez.plot_static()
    assert ungrouped.legends == []
    assert len(ungrouped.samples) == 1


def test_biplot_samples_follow_format_samples(pca_bp, frame):
    plain = pca_bp.plot_static()
    assert [s.get_label() for s in plain.samples] == ["a", "b", "c"]
    # colours as the interactive display shows them
    rgba = matplotlib.colors.to_rgba
    assert [rgba(s.get_markerfacecolor()) for s in plain.samples] == [
        rgba(c) for c in pca_bp.meta["color"]
    ]

    fmt = pca_bp.format_samples(stratify="col", by="grp")
    sb = fmt.plot_static()
    assert sb.legends[0].get_title().get_text() == "grp"
    assert [rgba(s.get_markerfacecolor()) for s in sb.samples] == [
        rgba(c) for c in bipl5.colorpal(3)
    ]

    size = np.where(frame["alpha"] > frame["alpha"].median(), "big", "small")
    dual = fmt.format_samples(stratify="symbol", by=size, pch=[16, 2])
    sb = dual.plot_static()
    assert len(sb.legends) == 2
    assert [leg.get_title().get_text() for leg in sb.legends] == ["grp", "Data"]
    # one artist per colour x symbol combination, covering every sample
    assert sum(len(s.get_xdata()) for s in sb.samples) == len(frame)
    assert {s.get_marker() for s in sb.samples} == {"o", "^"}
    assert any(rgba(s.get_markerfacecolor())[3] == 0 for s in sb.samples)  # open triangles
    sb = dual.plot_static(legend_title=["Group", "Size"])
    assert [leg.get_title().get_text() for leg in sb.legends] == ["Group", "Size"]


def test_class_means(frame, groups, data):
    cv = bipl5.init_biplot(frame).scale_mds("cva", classes=frame["grp"])
    sb = cv.plot_static()
    ez = cv.meta["x"]
    xy = np.array([[m.get_xdata()[0], m.get_ydata()[0]] for m in sb.class_means])
    np.testing.assert_allclose(xy, ez.Zmeans)
    assert cv.plot_static(class_means=False).class_means == []

    pc = bipl5.init_biplot(frame).scale_mds("pca", classes=frame["grp"], show_class_means=True)
    sb = pc.plot_static()
    assert len(sb.class_means) == 3
    # format_samples regroups the means with the samples
    two = np.where(np.arange(len(frame)) < 30, "first", "second")
    sb = pc.format_samples(by=two).plot_static()
    assert len(sb.class_means) == 2
    Z = pc.meta["x"].Z
    np.testing.assert_allclose(sb.class_means[0].get_xdata()[0], Z[:30, 0].mean())


def test_bags_and_ellipses_are_drawn_exactly_as_stored(data, groups):
    ez = pca(biplot(data, group_aes=groups))
    rng = np.random.default_rng(0)
    ez.alpha_bags = {"a": rng.normal(size=(6, 2)), "b": rng.normal(size=(5, 2))}
    ez.alpha_bag_aes = {"col": ["red", "blue"], "lwd": [1, 2], "lty": [1, 2], "opacity": [0.25, 0.5]}
    ez.conc_ellipses = {"a": rng.normal(size=(20, 2))}
    ez.conc_ellipse_aes = {"col": ["green"], "opacity": [0.1]}

    sb = ez.plot_static()
    assert len(sb.polygons) == 3
    np.testing.assert_allclose(sb.polygons[0].get_xy()[:6], ez.alpha_bags["a"])
    assert sb.polygons[1].get_facecolor() == matplotlib.colors.to_rgba("blue", 0.5)
    assert len(ez.plot_static(alpha_bags=False).polygons) == 1
    assert len(ez.plot_static(alpha_bags=False, conc_ellipses=False).polygons) == 0


def test_sample_labels(pca_ez):
    sb = pca_ez.plot_static(sample_labels=True)
    assert [t.get_text() for t in sb.sample_labels][:3] == ["1", "2", "3"]
    custom = [f"s{i}" for i in range(pca_ez.n)]
    assert pca_ez.plot_static(sample_labels=custom).sample_labels[5].get_text() == "s5"
    with pytest.raises(ValueError, match="one label per sample"):
        pca_ez.plot_static(sample_labels=["x"])
    assert pca_ez.plot_static().sample_labels == []


# ── scaffolding ────────────────────────────────────────────────────────────


def test_display_quality_is_the_caption_and_axes_carry_percentages(pca_bp):
    sb = pca_bp.plot_static()
    texts = [t.get_text() for t in sb.titles]
    assert texts[-1].startswith("Quality of display")
    assert sb.ax.get_xlabel().startswith("PC1 (") and sb.ax.get_xlabel().endswith("%)")

    assert pca_bp.plot_static(caption="").titles == []
    assert [t.get_text() for t in pca_bp.plot_static(caption="custom").titles] == ["custom"]
    assert pca_bp.plot_static(axis_percentages=False).ax.get_xlabel() == "PC1"

    sb = pca_bp.plot_static(title="T", subtitle="S", caption="")
    assert [t.get_text() for t in sb.titles] == ["S", "T"]


def test_themes(pca_bp):
    ez_theme = pca_bp.plot_static(theme="biplotEZ")
    assert ez_theme.ax.get_xlabel() == "" and list(ez_theme.ax.get_xticks()) == []
    for theme in ("bw", "minimal", "classic", None):
        _draw(pca_bp.plot_static(theme=theme))
    called = []
    pca_bp.plot_static(theme=lambda ax: called.append(ax))
    assert len(called) == 1
    with pytest.raises(ValueError, match="theme must be one of"):
        pca_bp.plot_static(theme="nope")


def test_display_selection(pca_bp):
    bp3 = pca_bp.append_mds_display((1, 3))
    first = bp3.plot_static()
    assert first.ax.get_xlabel().startswith("PC1") and first.ax.get_ylabel().startswith("PC2")
    for key in ("mdsDisplay_13", (1, 3), (3, 1), "PC 1 & 3"):
        sb = bp3.plot_static(key)
        assert sb.ax.get_ylabel().startswith("PC3")
        np.testing.assert_allclose(
            np.sort(np.concatenate([s.get_ydata() for s in sb.samples])),
            np.sort(bp3.displays["mdsDisplay_13"]["Data"]["sample_coordinates"][:, 1]),
        )
    with pytest.raises(ValueError, match="not a display"):
        bp3.plot_static("mdsDisplay_23")


def test_draw_into_existing_axes(pca_bp):
    fig, (a1, a2) = plt.subplots(1, 2)
    s1 = pca_bp.plot_static(ax=a1, legend=False)
    s2 = pca_bp.plot_static(ax=a2)
    assert s1.figure is fig and s2.ax is a2
    fig.canvas.draw()


def test_tidy_text_keeps_text_inside_the_panel(pca_bp):
    sb = pca_bp.plot_static(xlim=(0, 3), ylim=(-1, 1.5), expand=False)
    render = _draw(sb)
    panel = render["panel"]
    for t in render["title_texts"]:
        bb = t["bbox"]
        assert bb.x0 >= panel.x0 - 1 and bb.x1 <= panel.x1 + 1
        assert bb.y0 >= panel.y0 - 1 and bb.y1 <= panel.y1 + 1
    assert len(render["drawn_marks"]) <= len(render["marks"])

    untidy = _draw(pca_bp.plot_static(xlim=(0, 3), ylim=(-1, 1.5), expand=False, tidy_text=False))
    assert len(untidy["drawn_marks"]) == len(untidy["marks"])


def test_save_vector_and_raster_output(pca_bp, tmp_path):
    sb = pca_bp.plot_static()
    for ext in ("pdf", "svg", "png"):
        path = sb.save(tmp_path / f"biplot.{ext}")
        assert path.stat().st_size > 1000
    assert b"calibrated_axes" in (tmp_path / "biplot.svg").read_bytes()


def test_import_bipl5_does_not_import_matplotlib():
    import os
    from pathlib import Path

    code = "import sys, bipl5; print('matplotlib' in sys.modules)"
    src = str(Path(bipl5.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "False"


# ── the calibrated-axis layer on hand-built plots ──────────────────────────


def test_vertical_and_translated_axes():
    vertical = np.array([[0, -1, -1], [0, 0, 0], [0, 1, 1]], dtype=float)
    translated = vertical.copy()
    translated[:, 0] = 1.5

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    artist = add_calibrated_axes(ax, [vertical, translated], ["V", "W"])
    layout = artist.compute_layout((-2, 2), (-2, 2))

    xs = [line["x"] for line in layout["lines"]]
    np.testing.assert_allclose(xs[0], [0, 0])
    np.testing.assert_allclose(xs[1], [1.5, 1.5])
    # markers on the vertical axis span the whole window height
    values = sorted(m["value"] for m in layout["marks"] if m["axis"] == 0)
    assert values == [-2, -1, 0, 1, 2]
    # the translated axis dodges away from the origin
    w_marks = [m for m in layout["marks"] if m["axis"] == 1]
    np.testing.assert_allclose(w_marks[0]["dodge"], [1, 0])
    fig.canvas.draw()


def test_add_calibrated_axes_on_a_hand_built_plot(pca_ez):
    fig, ax = plt.subplots()
    ax.scatter(pca_ez.Z[:, 0], pca_ez.Z[:, 1])
    ax.set_aspect("equal")
    style = AxisStyle(axis_colour="grey", label_fontstyle="italic")
    artist = add_calibrated_axes(ax, axes_coordinates(pca_ez), pca_ez.col_names, style=style)
    fig.canvas.draw()
    assert artist.last_render["title_texts"]
    with pytest.raises(ValueError, match="label_placement"):
        CalibratedAxes([], label_placement="middle")


@pytest.mark.parametrize("placement", ["positive", "negative", "peripheral"])
def test_label_placement(pca_ez, placement):
    sb = pca_ez.plot_static(label_placement=placement)
    layout = sb.calibrated_axes.compute_layout(sb.ax.get_xlim(), sb.ax.get_ylim())
    for title in layout["titles"]:
        cal = sb.calibrations[title["axis"]]
        along = float(np.dot(title["out"], cal.direction))
        if placement == "positive":
            assert along > 0
        elif placement == "negative":
            assert along < 0
        else:
            assert float(np.dot(title["point"], title["out"])) > 0
    _draw(sb)


# ── pure helpers ───────────────────────────────────────────────────────────


def test_line_window_span_bounds_a_line_by_the_window():
    assert line_window_span((0, 0), (1, 0), (-2, 3), (-1, 1)) == (-2, 3)
    # a vertical line inside the window is bounded by the y range only
    assert line_window_span((0.5, 0), (0, 1), (-2, 3), (-1, 4)) == (-1, 4)
    # a vertical line outside the window misses it entirely
    assert line_window_span((9, 0), (0, 1), (-2, 3), (-1, 4)) is None


def test_marker_values_are_formatted_from_the_marker_spacing():
    assert format_tick_values([0, 0.1, 0.2, 0.30000000000000004], step=0.1) == ["0", "0.1", "0.2", "0.3"]
    assert format_tick_values([-10, 0, 10], step=10) == ["-10", "0", "10"]
    assert format_tick_values([-1e-17, 2.5], step=0.5) == ["0", "2.5"]
    assert format_tick_values([1e-5, 2e-5], step=1e-5) == ["0.00001", "0.00002"]
    assert format_tick_values([]) == []


def test_axis_angles_fold_into_the_half_open_interval():
    folded = fold_angle([0, math.pi / 4, 3 * math.pi / 4, math.pi, -math.pi / 3, -math.pi / 2])
    assert np.all((folded > -math.pi / 2 - 1e-8) & (folded <= math.pi / 2 + 1e-8))
    np.testing.assert_allclose(folded[:4], [0, math.pi / 4, -math.pi / 4, 0], atol=1e-12)
    np.testing.assert_allclose(folded[5], math.pi / 2)


def test_curve_angles_track_increasing_values():
    px, py = np.array([0, 1, 2, 3.0]), np.zeros(4)
    np.testing.assert_allclose(curve_angles(px, py, [0, 1, 2, 3]), np.zeros(4))
    flipped = curve_angles(px, py, [3, 2, 1, 0])
    np.testing.assert_allclose(np.cos(flipped), -np.ones(4))
    np.testing.assert_allclose(np.sin(flipped), np.zeros(4), atol=1e-12)


def test_resolve_axis_aes_and_linetypes():
    assert resolve_axis_aes(None, ["a"], "black", 3) == ["black"] * 3
    assert resolve_axis_aes("biplotEZ", ["a", "b"], "black", 3) == ["a", "b", "a"]
    assert resolve_axis_aes("biplotEZ", None, "black", 2) == ["black", "black"]
    assert resolve_axis_aes(["x", "y"], None, "k", 4) == ["x", "y", "x", "y"]
    assert resolve_axis_aes(None, None, ["p", "q"], 2) == ["p", "q"]
    assert r_linetype("solid") == "-"
    assert r_linetype(2) == r_linetype("dashed")
    assert r_linetype("--") == "--"


def test_marker_styles():
    assert marker_style("circle") == marker_style(16)
    assert marker_style("circle").fill == "filled"
    assert marker_style("triangle-up-open").marker == "^"
    assert marker_style("triangle-up-open").fill == "open"
    assert marker_style(1).fill == "open"  # pch 1: open circle
    assert marker_style(3).fill == "line"  # pch 3: plus
    assert marker_style("no-such-symbol").marker == "o"


def test_calibration_predict_reads_values_off_the_axis(pca_bp):
    from bipl5.display.build_one import obtain_xhat

    sb = pca_bp.plot_static()
    ez = pca_bp.meta["x"]
    xhat = obtain_xhat(ez)
    for j, cal in enumerate(sb.calibrations):
        np.testing.assert_allclose(cal.predict(ez.Z[:, 0], ez.Z[:, 1]), xhat[:, j], atol=1e-10)
    # broadcasts over grids, e.g. for heatmaps beneath the biplot
    gx, gy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3))
    assert sb.calibrations[0].predict(gx, gy).shape == (3, 4)


def test_appended_cva_pair(frame):
    rng = np.random.default_rng(3)
    four = frame.copy()
    four["grp"] = rng.choice(list("abcd"), size=len(frame))
    cv = bipl5.init_biplot(four).scale_mds("cva", classes=four["grp"]).append_mds_display((1, 3))
    sb = cv.plot_static((1, 3))
    assert sb.ax.get_xlabel().startswith("CV1") and sb.ax.get_ylabel().startswith("CV3")
    np.testing.assert_allclose(
        np.sort(np.concatenate([s.get_xdata() for s in sb.samples])),
        np.sort(cv.displays["mdsDisplay_13"]["Data"]["sample_coordinates"][:, 0]),
    )
    assert len(sb.class_means) == 4
    _draw(sb)
