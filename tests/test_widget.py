import json

import numpy as np
import pandas as pd
import pytest

import bipl5
from bipl5.render.widget import PLOTLY_JS_VERSION, payload_json


@pytest.fixture
def widget(data, groups):
    frame = data.copy()
    frame["grp"] = groups
    bp = bipl5.init_biplot(frame).scale_mds("pca", classes=frame["grp"])
    return bp.append_mds_display((1, 3)).plot()


def test_payload_contract(widget):
    payload = widget.payload
    assert set(payload) == {
        "p", "cols", "class_mean_hover", "mdsDisplays",
        "fm_mdsDisplay", "fitDisplay", "ax_slider", "initialPCKey",
    }
    assert payload["p"] == 4
    assert payload["initialPCKey"] == "PC 1 & 2"
    assert set(payload["mdsDisplays"]) == {"PC 1 & 2", "PC 1 & 3"}
    # the first display ships only config + fit table; others the full bundle
    first = payload["mdsDisplays"]["PC 1 & 2"]
    assert set(first) == {"config", "fit_table"}
    second = payload["mdsDisplays"]["PC 1 & 3"]
    assert {"trace_data", "layout", "config", "fit_table"} <= set(second)
    assert set(payload["fm_mdsDisplay"]) == {"CumPred", "CumAd", "VarExp", "Scree"}


def test_payload_serializes(widget):
    text = payload_json(widget.payload)
    parsed = json.loads(text)  # payload contains no NaN, so valid JSON too
    assert parsed["p"] == 4


def test_figure_structure(widget):
    layout = widget.figure["layout"]
    assert layout["yaxis"]["scaleanchor"] == "x"
    assert layout["barmode"] == "stack"
    menus = layout["updatemenus"]
    assert [m.get("name") for m in menus[1:]] == [
        "PC_toggle", "Fit_toggle", "Slider_toggle",
    ]
    # PC dropdown trimmed to the available displays
    labels = [b["label"] for b in menus[1]["buttons"]]
    assert labels == ["PC 1 & 2", "PC 1 & 3"]
    assert len(layout["sliders"][0]["steps"]) == 21
    assert layout["annotations"]  # first display's annotations rendered


def test_to_html_cdn_and_embedded(widget):
    # NOTE: these HTML strings are megabytes; compare booleans so a failure
    # never sends huge operands through pytest's comparison repr.
    html = widget.to_html()
    assert ("bipl5Attach" in html) is True
    cdn_tag = f'src="https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"'
    assert (cdn_tag in html) is True
    assert html.startswith("<!DOCTYPE html>") is True

    fragment = widget.to_html(full_html=False)
    assert fragment.startswith("<!DOCTYPE html>") is False
    assert ("bipl5Attach" in fragment) is True

    embedded = widget.to_html(include_plotlyjs=True)
    assert (f"plotly.js v{PLOTLY_JS_VERSION}" in embedded) is True
    assert (cdn_tag in embedded) is False  # no CDN script tag when embedded


def test_save(tmp_path, widget):
    out = widget.save(tmp_path / "biplot.html")
    text = out.read_text(encoding="utf-8")
    assert ("bipl5Attach" in text) is True
    # plotly.js embedded by default so the saved file works offline
    assert (f"plotly.js v{PLOTLY_JS_VERSION}" in text) is True


def test_repr_html(widget):
    assert ("bipl5Attach" in widget._repr_html_()) is True


def test_numpy_payload_encoding():
    text = payload_json(
        {
            "a": np.array([1.5, 2.5]),
            "b": np.int64(3),
            "c": np.float64(0.5),
            "d": np.bool_(True),
        }
    )
    assert json.loads(text) == {"a": [1.5, 2.5], "b": 3, "c": 0.5, "d": True}
