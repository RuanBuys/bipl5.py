"""Headless-browser smoke test: verifies the vendored bipl5 JavaScript
attaches to the rendered figure and its interactive controls work.

Skipped automatically when Playwright or a Chromium build is unavailable.
"""

import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

_CHROMIUM = os.environ.get("BIPL5_TEST_CHROMIUM", "/opt/pw-browsers/chromium")


@pytest.fixture(scope="module")
def biplot_html(tmp_path_factory):
    import numpy as np
    import pandas as pd

    import bipl5

    rng = np.random.default_rng(42)
    latent = rng.normal(size=(60, 2))
    mixing = np.array([[2.0, 0.3, -1.0, 0.5], [0.2, 1.5, 0.8, -0.7]])
    frame = pd.DataFrame(
        latent @ mixing + rng.normal(scale=0.4, size=(60, 4)),
        columns=["alpha", "beta", "gamma", "delta"],
    )
    grp = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    bp = bipl5.init_biplot(frame).scale_mds("pca", classes=grp)
    path = tmp_path_factory.mktemp("html") / "biplot.html"
    bp.append_mds_display((1, 3)).plot().save(path)
    return path


@pytest.mark.skipif(
    not Path(_CHROMIUM).exists(), reason="no Chromium available"
)
def test_js_attaches_and_controls_work(biplot_html):
    errors = []
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=_CHROMIUM)
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(biplot_html.as_uri())
        page.wait_for_timeout(2500)

        result = page.evaluate(
            """async () => {
            const el = document.querySelector('.plotly-graph-div');
            const fire = (menu, idx) => {
                const btn = Object.assign({}, menu.buttons[idx], {_index: idx});
                el.emit('plotly_buttonclicked', {menu: menu, button: btn, active: idx});
            };
            const wait = ms => new Promise(r => setTimeout(r, ms));
            const vis = tag => el.data.filter(tr =>
                (Array.isArray(tr.meta) ? tr.meta[0] : tr.meta) === tag
                && tr.visible === true).length;

            const out = {attached: !!el.bipl5};
            fire(el.layout.updatemenus[0], 1);           // Translated Axes
            await wait(1200);
            out.tdaVisible = vis('ExpAx');
            out.densVisible = vis('density');

            const pcMenu = el.layout.updatemenus.find(m => m.name === 'PC_toggle');
            fire(pcMenu, 1);                             // PC 1 & 3
            await wait(2500);
            out.pcKey = el.bipl5.currentPCKey;

            fire(el.layout.updatemenus[0], 0);           // Measures of Fit
            await wait(2000);
            out.fitOpen = el.bipl5.fitOpen;
            out.fitTraces = el.data.filter(tr =>
                Array.isArray(tr.meta) && tr.meta[0] === 'FitPanel').length;
            return out;
        }"""
        )
        browser.close()

    assert not errors, f"page errors: {errors}"
    assert result["attached"] is True
    assert result["tdaVisible"] == 4
    assert result["densVisible"] == 15
    assert result["pcKey"] == "PC 1 & 3"
    assert result["fitOpen"] is True
    assert result["fitTraces"] == 5
