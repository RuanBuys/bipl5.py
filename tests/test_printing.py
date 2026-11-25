"""Tree-style printing — port of the R package's print-method contract."""

import numpy as np
import pandas as pd
import pytest

import bipl5
from bipl5.printing import dim_label, tree_symbols


@pytest.fixture(autouse=True)
def fixed_print_options():
    """Deterministic output: unicode trees, no ANSI colours."""
    bipl5.set_print_options(unicode=True, color=False)
    yield
    bipl5.set_print_options()


@pytest.fixture
def bp(data, groups):
    frame = data.copy()
    frame["grp"] = groups
    return bipl5.init_biplot(frame).scale_mds("pca", classes=frame["grp"])


def test_biplot_tree(bp):
    lines = repr(bp.append_mds_display((1, 3))).splitlines()
    assert lines[0] == "bipl5_biplot [PCA]"
    assert lines[1] == "├── mdsDisplay_12 [PC 1 & 2] <bipl5_mdsDisplay>"
    assert lines[2] == "│   ├── Data <bipl5_data>"
    assert lines[3] == "│   │   ├── sample_coordinates  [60 x 2]"
    assert lines[4] == "│   │   ├── axes_coordinates  [4 axes]"
    assert lines[5] == "│   │   └── translated_axes_coordinates"
    assert lines[6].startswith("│   ├── trace_data  [")
    assert lines[7].startswith("│   └── annotations  [")
    assert "├── mdsDisplay_13 [PC 1 & 3] <bipl5_mdsDisplay>" in lines
    assert "└── fit_measures <bipl5_fitmeasures>" in lines
    # fit-measure children indented under the last branch
    assert "    ├── CumPred  [5 traces]" in lines
    assert "    ├── fit_table_12  [PC 1 & 2]" in lines
    assert "    └── fit_table_13  [PC 1 & 3]" in lines


def test_biplot_tree_without_fit_measures(bp, data, groups):
    frame = data.copy()
    frame["grp"] = groups
    cva = bipl5.init_biplot(frame).scale_mds("cva", classes=frame["grp"])
    lines = repr(cva).splitlines()
    assert lines[0] == "bipl5_biplot [CVA]"
    # the single display becomes the last branch when no fit measures exist
    assert lines[1] == "└── mdsDisplay_12 [CV 1 & 2] <bipl5_mdsDisplay>"
    assert lines[2] == "    ├── Data <bipl5_data>"
    assert "fit_measures" not in repr(cva)


def test_display_tree(bp):
    text = repr(bp.displays["mdsDisplay_12"])
    lines = text.splitlines()
    assert lines[0] == "bipl5_mdsDisplay"
    assert lines[1].startswith("Quality of display = ")
    assert lines[2] == "├── Data <bipl5_data>"
    assert lines[-1].startswith("└── annotations  [")
    assert isinstance(bp.displays["mdsDisplay_12"], bipl5.MdsDisplay)


def test_data_tree(bp):
    node = bp.extract("mdsDisplay_12.Data")
    assert isinstance(node, bipl5.BiplotData)
    lines = repr(node).splitlines()
    assert lines == [
        "bipl5_data",
        "├── sample_coordinates  [60 x 2]",
        "├── axes_coordinates  [4 axes]",
        "└── translated_axes_coordinates",
    ]


def test_fitmeasures_tree(bp):
    assert isinstance(bp.fit_measures, bipl5.FitMeasures)
    lines = repr(bp.fit_measures).splitlines()
    assert lines[0] == "bipl5_fitmeasures"
    assert lines[1] == "├── CumPred  [5 traces]"
    assert lines[-1] == "└── fit_table_12  [PC 1 & 2]"


def test_ascii_mode(bp):
    bipl5.set_print_options(unicode=False, color=False)
    text = repr(bp.extract("mdsDisplay_12.Data"))
    assert "+-- sample_coordinates" in text
    assert "`-- translated_axes_coordinates" in text
    assert "├" not in text


def test_color_mode(bp):
    bipl5.set_print_options(unicode=True, color=True)
    text = repr(bp.fit_measures)
    assert "\x1b[1;33mbipl5_fitmeasures\x1b[0m" in text
    assert "\x1b[90m" in text  # silver annotations


def test_tree_symbols():
    uni = tree_symbols(unicode=True)
    ascii_ = tree_symbols(unicode=False)
    assert uni["branch"] == "├── " and uni["pipe"] == "│   "
    assert ascii_["branch"] == "+-- " and ascii_["last"] == "`-- "
    assert uni["space"] == ascii_["space"] == "    "


def test_dim_label():
    assert dim_label(np.zeros((150, 2))) == "  [150 x 2]"
    assert dim_label(np.zeros(4)) == "  [4]"
    assert dim_label([1, 2, 3]) == "  [3]"
    assert dim_label(None) == ""


def test_types_survive_verbs(bp, data, groups):
    frame = data.copy()
    frame["grp"] = groups
    frame["band"] = ["low", "high"] * 30

    formatted = (
        bipl5.init_biplot(frame)
        .scale_mds("pca", classes=frame["grp"])
        .format_samples(stratify="col", by="band")
        .score_axes()
        .append_mds_display((1, 3))
    )
    for name in ("mdsDisplay_12", "mdsDisplay_13"):
        assert isinstance(formatted.displays[name], bipl5.MdsDisplay)
        assert isinstance(formatted.displays[name]["Data"], bipl5.BiplotData)
    assert isinstance(formatted.fit_measures, bipl5.FitMeasures)

    subset = formatted.extract("mdsDisplay_12")
    assert isinstance(subset.fit_measures, bipl5.FitMeasures)
    assert repr(subset).splitlines()[0] == "bipl5_biplot [PCA]"
