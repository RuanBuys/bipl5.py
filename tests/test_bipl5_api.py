import numpy as np
import pandas as pd
import pytest

import bipl5


@pytest.fixture
def frame(data, groups) -> pd.DataFrame:
    out = data.copy()
    out["grp"] = groups
    return out


@pytest.fixture
def pca_bp(frame):
    return bipl5.init_biplot(frame).scale_mds("pca", classes=frame["grp"])


def test_init_biplot_validation():
    with pytest.raises(ValueError, match="numeric"):
        bipl5.init_biplot(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(ValueError, match="numeric column"):
        bipl5.init_biplot(pd.DataFrame({"s": ["x", "y"]}))
    spec = bipl5.init_biplot(pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]}))
    assert spec.numeric_columns == ["a"]
    assert list(spec.data.columns) == ["a", "s"]  # full frame retained


def test_scale_mds_type_and_arg_validation(frame):
    spec = bipl5.init_biplot(frame)
    with pytest.raises(ValueError, match="Unsupported type"):
        spec.scale_mds("tsne")
    with pytest.raises(ValueError, match="Unsupported arguments"):
        spec.scale_mds("pca", bananas=1)
    with pytest.raises(ValueError, match="only one of"):
        spec.scale_mds("pca", eigenvectors=(1, 2), e_vects=(1, 2))
    with pytest.raises(ValueError, match="requires 'classes'"):
        spec.scale_mds("cva")
    with pytest.raises(ValueError, match="requires 'Z'"):
        spec.scale_mds("regress")


def test_pca_biplot_structure(pca_bp):
    bp = pca_bp
    assert bp.biplot_type == "pca"
    assert list(bp.displays) == ["mdsDisplay_12"]
    assert set(bp.fit_measures) == {
        "CumPred", "CumAd", "VarExp", "Scree", "fit_table_12",
    }
    assert bp.meta["pc_info"]["mdsDisplay_12"]["label"] == "PC 1 & 2"
    assert bp.meta["group"][1] == ["a", "b", "c"]
    assert "Quality of display" in bp.meta["fit_quality"]

    d = bp.displays["mdsDisplay_12"]
    meta_counts: dict[str, int] = {}
    for tr in d["mds"]["trace_data"]:
        m = tr["meta"]
        key = m if isinstance(m, str) else m[0]
        meta_counts[key] = meta_counts.get(key, 0) + 1
    p, g = 4, 3
    assert meta_counts["data"] == g
    assert meta_counts["axis"] == 2 * p  # line + name label per axis
    assert meta_counts["OuterCircle"] == 1
    assert meta_counts["ExpAx"] == p
    assert meta_counts["density"] == g * (p + 1)  # legend entry + per-axis
    # TDA layers start hidden; calibrated axes visible
    for tr in d["mds"]["trace_data"]:
        key = tr["meta"] if isinstance(tr["meta"], str) else tr["meta"][0]
        if key in ("ExpAx", "density"):
            assert tr["visible"] is False
        if key in ("data", "axis", "OuterCircle"):
            assert tr["visible"] is True
    assert "slider_info" in d["mds"]["config"]
    assert d["Data"]["sample_coordinates"].shape == (60, 2)

    hover = d["mds"]["trace_data"][0]["hovertext"][0]
    assert "Actual" in hover and "Pred" in hover
    assert "Sample predictivity:" in hover
    # 1-based observation names from the default index
    assert "Observation: 1" in hover


def test_scale_mds_other_types(frame):
    spec = bipl5.init_biplot(frame)
    cva = spec.scale_mds("cva", classes=frame["grp"])
    assert cva.biplot_type == "cva"
    assert cva.fit_measures is None
    assert cva.meta["pc_info"]["mdsDisplay_12"]["label"] == "CV 1 & 2"
    # class-mean traces present (show_class_means defaults to True for CVA)
    metas = [
        tr["meta"][0]
        for tr in cva.displays["mdsDisplay_12"]["mds"]["trace_data"]
        if not isinstance(tr["meta"], str)
    ]
    assert "ClassMean" in metas

    pco = spec.scale_mds("pco")
    assert pco.biplot_type == "pco"
    assert pco.meta["fit_quality"] == ""

    Z = pco.displays["mdsDisplay_12"]["Data"]["sample_coordinates"]
    reg = spec.scale_mds("regress", Z=Z)
    assert reg.biplot_type == "reg"
    assert reg.meta["fit_quality"].startswith("R^2_disp")
    assert reg.meta["fit_quality_plotly"].startswith("\\(")

    with pytest.raises(NotImplementedError, match="[Ss]pline"):
        spec.scale_mds("pco", axes="splines")


def test_append_and_remove_display(pca_bp):
    bp = pca_bp.append_mds_display((1, 3))
    assert list(bp.displays) == ["mdsDisplay_12", "mdsDisplay_13"]
    assert "fit_table_13" in bp.fit_measures
    assert bp.meta["pc_info"]["mdsDisplay_13"]["label"] == "PC 1 & 3"
    # appended display uses the same aesthetics and trace structure
    first = bp.displays["mdsDisplay_12"]["mds"]["trace_data"]
    second = bp.displays["mdsDisplay_13"]["mds"]["trace_data"]
    assert len(first) == len(second)

    with pytest.raises(ValueError, match="already exists"):
        bp.append_mds_display((3, 1))
    with pytest.raises(ValueError, match="between 1 and"):
        bp.append_mds_display((1, 9))

    out = bp.remove_mds_display("mdsDisplay_13")
    assert list(out.displays) == ["mdsDisplay_12"]
    with pytest.raises(ValueError, match="last remaining"):
        out.remove_mds_display("mdsDisplay_12")
    with pytest.raises(ValueError, match="not a valid"):
        out.remove_mds_display("mdsDisplay_99")


def test_extract(pca_bp):
    coords = pca_bp.extract("mdsDisplay_12.Data.sample_coordinates")
    assert coords.shape == (60, 2)
    sub = pca_bp.extract("mdsDisplay_12")
    assert isinstance(sub, bipl5.Biplot)
    assert list(sub.displays) == ["mdsDisplay_12"]
    fit = pca_bp.extract("fit_measures.CumPred")
    assert isinstance(fit, bipl5.BiplotFit)
    assert len(fit.trace_data) == 5  # 4 variables + overall quality
    html = fit.plot().to_html()
    assert "Cumulative axis predictivity" in html

    with pytest.raises(KeyError):
        pca_bp.extract("mdsDisplay_99")
    with pytest.raises(ValueError, match="fit-measure paths"):
        pca_bp.extract("fit_measures.Nope")


def test_overlay_fit(pca_bp):
    out = pca_bp.overlay_fit(True)
    assert out.meta["plot_options"]["fit_display"] == "overlay"
    payload = out.plot().payload
    assert payload["fitDisplay"]["mode"] == "overlay"
    # explicit plot() argument overrides the stored default
    payload = out.plot(fit_display="panel").payload
    assert payload["fitDisplay"]["mode"] == "panel"

    cva = bipl5.init_biplot(pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)))).scale_mds("pca")
    cva.fit_measures = None
    with pytest.raises(ValueError, match="fit measures"):
        cva.overlay_fit()


def test_score_axes_adds_error_column(pca_bp):
    scored = pca_bp.score_axes()
    hover = scored.displays["mdsDisplay_12"]["mds"]["trace_data"][0]["hovertext"][0]
    assert "Error" in hover and "%" in hover
    assert scored.meta["reading_errors"] is True
    # original object untouched
    original = pca_bp.displays["mdsDisplay_12"]["mds"]["trace_data"][0]["hovertext"][0]
    assert "Error" not in original
    assert "score_axes" in repr(scored)


def test_repr(pca_bp):
    text = repr(pca_bp)
    assert "bipl5 biplot [pca]" in text
    assert "mdsDisplay_12" in text
    assert "PC 1 & 2" in text
