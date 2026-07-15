"""Figure-level plotly layout — port of ``plot_scaffolding()`` from
``biplotEZ_helper.R`` (the layout applied to the rendered figure, as opposed
to the payload-side scaffolding in ``display.builders``).
"""

from __future__ import annotations

__all__ = ["figure_scaffolding"]

_QUALITY_TITLE = "Overall quality and axis predictivities (cumulative)"


def figure_scaffolding(
    dpquality,
    PC_toggle: bool = True,
    ax_pred: bool = True,
    TDA: bool = True,
    vec_dis: bool = True,
    n: int = 21,
    x_colnames: list[str] | None = None,
) -> dict:
    """Build the base figure layout dict.

    Contains the main biplot axes (x/y), the prediction inset (x2/y2), the
    fit panel (x3/y3), the top-row buttons, the PC dropdown, the hidden
    fit-measure and axis-name dropdowns, and the hidden calibration slider
    with ``n`` steps — all driven at runtime by the vendored JS.
    """
    x_colnames = [str(c) for c in (x_colnames or [])]
    slider_steps = [
        {"label": "", "value": i + 1, "method": "skip", "args": []}
        for i in range(n)
    ]
    axis_name_buttons = [
        {"method": "skip", "label": name} for name in x_colnames
    ]

    return {
        "legend": {
            "tracegroupgap": 0,
            "xref": "container",
            "yref": "container",
            "x": 1,
            "y": 0.82,
            "groupclick": "toggleitem",
        },
        "xaxis": {
            "title": dpquality,
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
            "domain": [0, 1],
        },
        "yaxis": {
            "showticklabels": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
            "showgrid": False,
        },
        "xaxis2": {"domain": [0, 0.15], "zeroline": True},
        "yaxis2": {"zeroline": True, "side": "left", "position": 0},
        "xaxis3": {
            "domain": [0.65, 1],
            "zeroline": True,
            "showgrid": True,
            "anchor": "y3",
            "dtick": 1,
            "title": "Dimension of Subspace",
        },
        "yaxis3": {
            "zeroline": True,
            "anchor": "free",
            "side": "left",
            "position": 0.65,
            "showgrid": True,
            "domain": [0.15, 0.85],
            "layer": "below traces",
            "title": _QUALITY_TITLE,
            "range": [0, 1],
        },
        "hoverlabel": {"font": {"family": "Courier New, monospace"}},
        "barmode": "stack",
        "updatemenus": [
            {
                "y": 0.8,
                "type": "buttons",
                "x": 0,
                "pad": {"r": 0},
                "showactive": True,
                "active": -1,
                "buttons": [
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Measures of Fit",
                        "name": "AxisStats",
                        "visible": ax_pred,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Translated Axes",
                        "name": "TransAxes",
                        "visible": TDA,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Vector Display",
                        "name": "vecload",
                        "visible": vec_dis,
                        "execute": False,
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Edit: Axes",
                        "name": "EditAxes",
                        "visible": False,
                        "execute": False,
                    },
                ],
            },
            {
                "type": "dropdown",
                "x": 0,
                "pad": {"r": 0},
                "visible": PC_toggle,
                "name": "PC_toggle",
                "buttons": [
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "PC 1 & 2",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "histogram"],
                        "label": "PC 1 & 3",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "histogram"],
                        "label": "PC 2 & 3",
                    },
                ],
            },
            {
                "type": "dropdown",
                "x": 0.5,
                "visible": False,
                "name": "Fit_toggle",
                "xanchor": "left",
                "buttons": [
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Cum. Predictivity",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Cum. Adequacy",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Scree Plot",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "scatter"],
                        "label": "Variance Explained",
                    },
                    {
                        "method": "skip",
                        "args": ["type", "histogram"],
                        "label": "Summary Table",
                    },
                ],
            },
            {
                "type": "dropdown",
                "visible": False,
                "x": 0,
                "y": 0,
                "xanchor": "right",
                "yanchor": "bottom",
                "name": "Slider_toggle",
                "direction": "up",
                "buttons": axis_name_buttons,
            },
        ],
        "sliders": [
            {
                "currentvalue": {"prefix": "Axis 1"},
                "x": 0.0,
                "y": -0.15,
                "xanchor": "left",
                "yanchor": "bottom",
                "steps": slider_steps,
                "visible": False,
            }
        ],
    }
