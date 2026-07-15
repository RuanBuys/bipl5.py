"""Port of the behavioural contract in the R package's
``test-format-samples.R``."""

import numpy as np
import pandas as pd
import pytest

import bipl5


@pytest.fixture
def frame(data) -> pd.DataFrame:
    out = data.copy()
    out["species"] = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    out["band"] = ["low", "high"] * 30
    return out


@pytest.fixture
def bp(frame):
    return bipl5.init_biplot(frame).scale_mds("pca")


def _traces(b, name="mdsDisplay_12"):
    return b.displays[name]["mds"]["trace_data"]


def _by_meta(b, key, name="mdsDisplay_12"):
    out = []
    for tr in _traces(b, name):
        meta = tr["meta"]
        meta = [meta] if isinstance(meta, str) else list(meta)
        if key in meta:
            out.append(tr)
    return out


def test_splits_by_stored_column_name(bp):
    out = bp.format_samples(stratify="col", by="species")
    data_traces = _by_meta(out, "data")
    assert [tr["name"] for tr in data_traces] == ["a", "b", "c"]
    assert [tr["marker"]["color"] for tr in data_traces] == bipl5.colorpal(3)
    assert data_traces[0]["legendgrouptitle"] == {"text": "<b>species</b>"}
    # per-level metas allow the JS to target the traces
    assert data_traces[0]["meta"] == ["data", "group:a"]
    # membership: customdata is 1-based; species 'a' is rows 1..20
    assert sorted(data_traces[0]["customdata"].tolist()) == list(range(1, 21))
    # metadata updated for downstream verbs
    assert out.meta["group"][1] == ["a", "b", "c"]
    assert out.meta["color"] == bipl5.colorpal(3)


def test_accepts_supplied_vector(bp, frame):
    plain = bp.format_samples(stratify="col", by=list(frame["species"]))
    assert [t["name"] for t in _by_meta(plain, "data")] == ["a", "b", "c"]
    # unnamed vector keeps the generic legend title
    assert _by_meta(plain, "data")[0]["legendgrouptitle"] == {"text": "<b>Data</b>"}
    # a named Series matching a stored column labels the section
    named = bp.format_samples(stratify="col", by=frame["species"])
    assert _by_meta(named, "data")[0]["legendgrouptitle"] == {
        "text": "<b>species</b>"
    }


def test_custom_colours_and_validation(bp):
    out = bp.format_samples(
        stratify="col", by="species", col=["#111111", "#222222", "#333333"]
    )
    assert [t["marker"]["color"] for t in _by_meta(out, "data")] == [
        "#111111", "#222222", "#333333",
    ]
    with pytest.raises(ValueError, match="Expected 3 colours, got 2"):
        bp.format_samples(stratify="col", by="species", col=["#111111", "#222222"])


def test_symbol_stratification_requires_pch(bp):
    with pytest.raises(ValueError, match="'pch' is required"):
        bp.format_samples(stratify="symbol", by="species")
    with pytest.raises(ValueError, match="Expected 3 plotting symbols"):
        bp.format_samples(stratify="symbol", by="species", pch=[15])
    with pytest.raises(ValueError, match="Invalid plotly symbols"):
        bp.format_samples(
            stratify="symbol", by="species", pch=["circle", "nope", "square"]
        )
    with pytest.raises(ValueError, match="not supported"):
        bp.format_samples(stratify="symbol", by="species", pch=[15, 16, 99])


def test_unified_legend_when_groupings_match(bp):
    out = bp.format_samples(stratify="col", by="species").format_samples(
        stratify="symbol", by="species", pch=[15, 17, 18]
    )
    data_traces = _by_meta(out, "data")
    # one legend section, one trace per class, both aesthetics applied
    assert [t["name"] for t in data_traces] == ["a", "b", "c"]
    assert [t["marker"]["symbol"] for t in data_traces] == [
        "square", "triangle-up", "diamond",
    ]
    assert [t["marker"]["color"] for t in data_traces] == bipl5.colorpal(3)
    assert not _by_meta(out, "sample-legend")


def test_dual_stratification(bp):
    out = bp.format_samples(stratify="col", by="species").format_samples(
        stratify="symbol", by="band", pch=[15, 17]
    )
    legends = _by_meta(out, "sample-legend")
    combos = _by_meta(out, "sample-combo")

    # two legend sections: 3 colour entries + 2 symbol entries
    assert [t["name"] for t in legends] == ["a", "b", "c", "low", "high"]
    titles = [t.get("legendgrouptitle") for t in legends]
    assert titles[0] == {"text": "<b>species</b>"}
    assert titles[3] == {"text": "<b>band</b>"}
    assert titles[1] is None and titles[4] is None

    # hidden combination traces cover species x band
    assert sorted(t["name"] for t in combos) == sorted(
        f"{s} | {b}" for s in "abc" for b in ("low", "high")
    )
    assert all(t["showlegend"] is False for t in combos)
    assert all("data" in t["meta"] for t in combos)
    # combined membership is a partition of all observations
    all_idx = np.concatenate([t["customdata"] for t in combos])
    assert sorted(all_idx.tolist()) == list(range(1, 61))
    # meta keeps the primary (colour) grouping
    assert out.meta["group"][1] == ["a", "b", "c"]


def test_colour_call_rebuilds_tda_densities(bp):
    out = bp.format_samples(stratify="col", by="species")
    dens = _by_meta(out, "density")
    # 3 groups x (4 axes + 1 legend entry)
    assert len(dens) == 15
    names = {t["name"] for t in dens}
    assert names == {"a", "b", "c"}
    # slider metadata regenerated
    assert "slider_info" in out.displays["mdsDisplay_12"]["mds"]["config"]


def test_symbol_only_call_leaves_densities(bp):
    out = bp.format_samples(
        stratify="symbol", by="species", pch=[15, 17, 18]
    )
    dens = _by_meta(out, "density")
    assert len(dens) == 5  # unchanged: original single group
    assert {t["name"] for t in dens} == {"Data"}


def test_colour_after_symbol_rebuilds_densities(bp):
    out = bp.format_samples(
        stratify="symbol", by="species", pch=[15, 17, 18]
    ).format_samples(stratify="col", by="band")
    dens = _by_meta(out, "density")
    assert len(dens) == 10  # 2 band groups x (4 axes + 1 legend)
    assert {t["name"] for t in dens} == {"low", "high"}


def test_applies_to_all_displays_and_append_replays(bp):
    multi = bp.append_mds_display((1, 3)).format_samples(
        stratify="col", by="species"
    )
    for name in ("mdsDisplay_12", "mdsDisplay_13"):
        assert [t["name"] for t in _by_meta(multi, "data", name)] == [
            "a", "b", "c",
        ]

    # a display appended after formatting inherits the stored state
    later = multi.format_samples(
        stratify="symbol", by="band", pch=[15, 17]
    ).append_mds_display((2, 3))
    combos = _by_meta(later, "sample-combo", "mdsDisplay_23")
    legends = _by_meta(later, "sample-legend", "mdsDisplay_23")
    assert len(combos) == 6 and len(legends) == 5


def test_class_means_follow_colour_groups(frame):
    bp = bipl5.init_biplot(frame).scale_mds(
        "pca", classes=frame["species"], show_class_means=True
    )
    out = bp.format_samples(stratify="col", by="band")
    means = _by_meta(out, "ClassMean")
    assert [t["name"] for t in means] == ["low", "high"]
    assert [t["marker"]["color"] for t in means] == bipl5.colorpal(2)


def test_cva_classes_preserved(frame):
    bp = bipl5.init_biplot(frame).scale_mds("cva", classes=frame["species"])
    out = bp.format_samples(stratify="col", by="band")
    # display grouping changes ...
    assert [t["name"] for t in _by_meta(out, "data")] == ["low", "high"]
    assert out.meta["group"][1] == ["low", "high"]
    # ... but the fitted CVA classes are untouched
    assert list(out.meta["x"].g_names) == ["a", "b", "c"]
    assert out.meta["model_group"][1] == ["a", "b", "c"]
    # CVA class-mean traces are not rebuilt
    assert [t["name"] for t in _by_meta(out, "ClassMean")] == ["a", "b", "c"]


def test_errors_cleanly_for_unknown_column(bp):
    with pytest.raises(ValueError, match="Column 'nope' was not found"):
        bp.format_samples(stratify="col", by="nope")
    with pytest.raises(ValueError, match="length 60"):
        bp.format_samples(stratify="col", by=["a", "b"])
    with pytest.raises(ValueError, match="Missing values"):
        bp.format_samples(stratify="col", by=["a"] * 59 + [None])


def test_original_object_unchanged(bp):
    before = [t["name"] for t in _by_meta(bp, "data")]
    bp.format_samples(stratify="col", by="species")
    assert [t["name"] for t in _by_meta(bp, "data")] == before
    assert bp.meta.get("sample_format") is None
    assert bp.meta["group"][1] == ["Data"]


def test_score_axes_after_format_samples(bp):
    out = bp.format_samples(stratify="col", by="species").score_axes()
    for tr in _by_meta(out, "data"):
        assert "Error" in tr["hovertext"][0]


def test_dual_plot_renders(bp):
    out = bp.format_samples(stratify="col", by="species").format_samples(
        stratify="symbol", by="band", pch=[15, 17]
    )
    html = out.plot().to_html()
    assert ("bipl5Attach" in html) is True
    assert ("sample-legend" in html) is True
