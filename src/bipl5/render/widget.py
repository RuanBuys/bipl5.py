"""``Bipl5Widget``: HTML delivery of the interactive biplot.

Replaces R's htmlwidgets machinery: the figure is rendered with plotly.py
and the vendored ``bipl5_plotly.js`` is injected as a post-render script
together with the JSON payload, calling ``window.bipl5Attach()`` exactly as
``htmlwidgets::onRender`` does in R.
"""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path

import numpy as np

__all__ = ["Bipl5Widget", "payload_json", "PLOTLY_JS_VERSION"]

_JS_DIR = Path(__file__).parent / "js"
_JS_PATH = _JS_DIR / "bipl5_plotly.js"

# bipl5's JavaScript targets the plotly.js 2.x API (the major version bundled
# by R's plotly package). plotly.py 6+ ships plotly.js 3.x, which dropped
# 2.x coercions the vendored JS relies on, so rendering pins 2.x explicitly.
PLOTLY_JS_VERSION = "2.35.2"
_PLOTLY_CDN_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"
_PLOTLY_JS_PATH = _JS_DIR / f"plotly-{PLOTLY_JS_VERSION}.min.js"


class _PayloadEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def payload_json(payload) -> str:
    """Serialize the side-channel payload for embedding in a script tag.

    ``NaN`` is emitted as the JavaScript literal (the string is evaluated
    as JS, not parsed as strict JSON), matching what htmlwidgets produces
    for R ``NA`` values consumed by the bipl5 JS.
    """
    return json.dumps(payload, cls=_PayloadEncoder, allow_nan=True)


def bipl5_js_source() -> str:
    """The vendored bipl5_plotly.js, byte-for-byte from the R package."""
    return _JS_PATH.read_text(encoding="utf-8")


class Bipl5Widget:
    """An interactive bipl5 figure ready to render as HTML.

    - In Jupyter/Quarto the widget displays itself (``_repr_html_``).
    - ``save(path)`` writes a self-contained HTML file.
    - ``show()`` opens the widget in a browser.
    - ``figure`` and ``payload`` expose the underlying plotly figure dict
      and the JS payload for inspection.
    """

    def __init__(self, figure: dict, payload: dict | None):
        self.figure = figure
        self.payload = payload

    def _post_script(self) -> str:
        if self.payload is None:
            return ""
        return (
            bipl5_js_source()
            + "\nwindow.bipl5Attach(document.getElementById('{plot_id}'), "
            + "null, "
            + payload_json(self.payload)
            + ");"
        )

    @staticmethod
    def _plotly_js_tag(include_plotlyjs) -> str:
        if include_plotlyjs is False or include_plotlyjs is None:
            return ""
        if include_plotlyjs is True:
            return f"<script>{_PLOTLY_JS_PATH.read_text(encoding='utf-8')}</script>"
        if include_plotlyjs == "cdn":
            return f'<script src="{_PLOTLY_CDN_URL}"></script>'
        if isinstance(include_plotlyjs, str) and include_plotlyjs.endswith(".js"):
            return f'<script src="{include_plotlyjs}"></script>'
        raise ValueError(
            "include_plotlyjs must be True (embed plotly.js "
            f"{PLOTLY_JS_VERSION}), False, 'cdn', or a .js URL."
        )

    def to_html(
        self,
        full_html: bool = True,
        include_plotlyjs="cdn",
        div_id: str | None = None,
    ) -> str:
        """Render the widget to an HTML string.

        ``include_plotlyjs``: ``"cdn"`` (default) references the pinned
        plotly.js 2.x from the plotly CDN; ``True`` embeds the vendored copy
        for fully offline files; a ``.js`` URL string references that URL.
        """
        import plotly.io as pio

        post_script = self._post_script() or None
        inner = pio.to_html(
            self.figure,
            validate=False,
            full_html=False,
            include_plotlyjs=False,
            include_mathjax="cdn",
            post_script=post_script,
            div_id=div_id,
        )
        plotly_tag = self._plotly_js_tag(include_plotlyjs)
        if not full_html:
            return plotly_tag + inner
        return (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8" />\n'
            f"{plotly_tag}\n</head>\n<body>\n{inner}\n</body>\n</html>"
        )

    def _repr_html_(self) -> str:
        return self.to_html(full_html=False)

    def save(self, path, include_plotlyjs=True) -> Path:
        """Write a standalone HTML file (plotly.js embedded by default, so
        the file works offline)."""
        path = Path(path)
        path.write_text(
            self.to_html(full_html=True, include_plotlyjs=include_plotlyjs),
            encoding="utf-8",
        )
        return path

    def show(self) -> None:  # pragma: no cover - opens a browser
        """Open the widget in the default web browser."""
        tmp = Path(tempfile.mkstemp(suffix=".html")[1])
        self.save(tmp)
        webbrowser.open(tmp.as_uri())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_traces = len(self.figure.get("data", []))
        has_js = self.payload is not None
        return (
            f"<Bipl5Widget: {n_traces} traces, "
            f"{'interactive bipl5 JS attached' if has_js else 'plain figure'}; "
            "use .save('biplot.html'), .show(), or display in a notebook>"
        )
