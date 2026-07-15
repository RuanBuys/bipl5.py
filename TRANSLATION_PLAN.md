# bipl5 → bipl5.py: Systematic Translation Plan

This document is the master plan for translating the **bipl5** R package
(RuanBuys/bipl5) into a native Python package in this repository. It is based
on a full audit of the R sources at version 1.1.0.

**Scope decisions (agreed up front):**

- The **`init_biplot()` pipeline is the API being translated**:
  `init_biplot() |> scale_mds() |> format_samples() |> score_axes() |> plot()`,
  plus the object-manipulation verbs (`append_mdsDisplay()`,
  `remove_mdsDisplay()`, `extract()`, `overlay_fit()`) and the print methods.
- The **`wrap_bipl5()` generic and its four S3 methods are dropped**. They
  exist only to adapt externally-built biplotEZ objects, and biplotEZ does not
  exist in Python. Everything `wrap_bipl5.*` produces is reachable through
  `scale_mds()` anyway.
- **biplotEZ is replaced by an internal ordination engine** ("biplotEZ-lite")
  that implements only what bipl5 actually consumes (audited below).
- The **custom JavaScript layer is reused nearly verbatim** — the reactivity
  lives in `inst/htmlwidgets/bipl5_plotly.js` (~2,400 lines) and talks to
  plotly.js, not to R. Python only has to deliver the same JSON payload.

---

## 1. What the R package actually does (architecture recap)

The R package has three layers:

1. **Math layer (delegated to biplotEZ today).** Ordination (PCA/CVA/PCO/
   regression), calibrated-axis coordinates, and fit measures. This is the
   only layer that does not exist in Python and must be written from scratch.
2. **Payload layer (pure R, translates 1:1).** Builds plotly *trace/layout
   dictionaries* — not figures — into `mdsDisplay` objects: sample traces
   with markdown hover tables, calibrated linear axes with tick annotations,
   translated density axes (TDA), vector annotations, fit-measure panels,
   slider metadata. Files: `build_secondary_biplots.R`,
   `wrap_bipl5_helper.R`, `mdsDisplay_constructor.R`, `PCAbiplot_Helper.R`,
   `Tickmarks.R`, `Translate.R`, `Translate_Helpers.R`, `format_samples.R`,
   `score_axes.R`, `ax_pred.R`, `FitPanel_funcs.R`.
3. **Render layer.** `plot()` seeds a plotly figure with the first
   mdsDisplay, then attaches `bipl5_plotly.js` with a JSON payload
   (`{p, cols, mdsDisplays, fm_mdsDisplay, fitDisplay, ax_slider,
   initialPCKey}`) via `htmlwidgets::onRender`. The JS entry point is
   `window.bipl5Attach(el, x, data)`.

Because layer 2 already builds plain data structures (lists of lists), the
bulk of the package translates mechanically to Python dicts/lists. The real
work is layer 1 (new math code) and layer 3 (a Python HTML/Jupyter delivery
mechanism replacing htmlwidgets).

---

## 2. biplotEZ audit — the exact surface to reimplement

Every biplotEZ call site in the retained pipeline, and every field bipl5
reads off the returned object:

### 2.1 Functions called

| biplotEZ call | Used by | Python replacement |
|---|---|---|
| `biplot(data, classes, group.aes, center, scaled, Title)` | `scale_mds()` | `ordination.base.Ordination` — stores `raw_X`, centered/scaled `X`, `means`, `sd`, `n`, `p`, group factor, title |
| `PCA(bp, dim.biplot, e.vects, group.aes, show.class.means, correlation.biplot)` | `scale_mds("pca")`, `append_mdsDisplay()` | `ordination.pca.pca()` — SVD of `X` |
| `CVA(bp, classes, dim.biplot, e.vects, weightedCVA, show.class.means, low.dim)` | `scale_mds("cva")`, `append_mdsDisplay()` | `ordination.cva.cva()` — two-sided eigenproblem of B w.r.t. W |
| `PCO(bp, Dmat, dist.func, dist.func.cat, dim.biplot, e.vects, group.aes, show.class.means, axes, ...)` | `scale_mds("pco")` | `ordination.pco.pco()` — classical MDS + regression or spline axes |
| `regress(bp, Z, group.aes, show.group.means, axes)` | `scale_mds("regress")` | `ordination.regress.regress()` — regression biplot on user coords |
| `samples(x)` | all compile paths | default sample aesthetics (colour/pch per group) |
| `axes(x)` | all compile paths | default axis aesthetics (`col`, `tick.col`, `tick.label.col`) |
| `fit.measures(x)` | pca/cva compile paths | quality, axis predictivity, adequacy, **sample predictivity** (and within-class variant for CVA) |
| `means(x)` | cva path, class means | class-mean coordinates `Zmeans` + aesthetics |
| `axes_coordinates(x)` | pco/regress paths, `clean_linear_axes_coordinates()` | calibrated tick coordinates per axis (list of `k×3` matrices: x, y, tick label; `k×4` for splines with a labelled-tick flag) |

### 2.2 Fields consumed from the "ez object"

The internal result object must expose (Python names in parentheses):

`X`, `raw.X` (`raw_X`), `Z`, `Zmeans`, `Lmat` (full p×p right singular
vectors / canonical weights), `Vr` (p×2 selected columns), `eigenvalues`,
`e.vects` (`e_vects`), `means`, `sd`, `center`, `scaled`, `n`, `p`,
`group.aes` (`group`), `samples$col/pch`, `axes$col/tick.col/tick.label.col`,
`class.means` (`show_class_means`), `means.aes`, `sample.predictivity`,
`within.class.sample.predictivity`, `alpha.bags`/`alpha.bag.aes`,
`conc.ellipses`/`conc.ellipse.aes`, `PCOaxes`, `dim.biplot`, `ax.one.unit`.

A single `@dataclass OrdinationResult` with optional fields covers all four
methods; `biplot_type` discriminates.

### 2.3 The math, concretely

- **Center/scale:** `X = (raw_X - means) / sd` per flags; keep both matrices
  because the payload builders un-center X to show raw values in hover text
  (`scale_mds_restore_raw_x()`).
- **PCA:** SVD of `X`; `Lmat = V`; `Vr = V[:, e_vects]`; `Z = X @ Vr`
  (correlation-biplot variant scales `Z` and `Vr` by singular values as
  biplotEZ does); `eigenvalues` = squared singular values (biplotEZ
  convention — verify against fixtures; only ratios matter for the fit
  strings, but absolute values appear in the scree plot). `ax.one.unit` =
  per-variable axis direction `v_j / (v_jᵀ v_j)` scaled for calibration.
- **CVA:** within-class `W` and between-class `B` matrices; solve the
  two-sided eigenproblem (Cholesky of `W`, then SVD); `weightedCVA` options
  (weighted / unweighted class contributions); `Zmeans` for class means;
  `low.dim` strategy when `#classes − 1 < 2` (biplotEZ's `"sample.opt"`).
  CVA displays disable fit measures and vector display, matching R.
- **PCO:** distance matrix from `Dmat`/`dist_func`/`dist_func_cat` (+ extra
  kwargs forwarded to the distance function, as `scale_mds` does); classical
  MDS via double-centering `−½ J D² J`; eigendecomposition; `Z = V Λ^{1/2}`.
  Axes either **regression** (linear, calibrated like PCA) or **splines**
  (non-linear trajectories; biplotEZ interpolates then smooths — use
  `scipy.interpolate`; exact numeric parity is not achievable here, target
  *visual* parity and structural payload parity).
- **regress:** user-supplied `Z` (n×2); axis directions from
  `B = (ZᵀZ)⁻¹ZᵀX`; calibrated ticks along each axis; the Alves-style
  read-off (`obtain_xhat` regression branch) reconstructs `Xhat` by
  interpolating tick values — already implemented in R in bipl5 itself and
  translates directly.
- **Fit measures:** cumulative axis predictivities (`ax_pred.R`), cumulative
  adequacies and marginal predictivities (`FitPanel_funcs.R`), display
  quality strings (`fit_quality`, `regression_fit_quality[_tex]` in
  `biplotEZ_helper.R`) are already bipl5-side R code → straight NumPy
  translations. Only **sample predictivity** (per-observation, from
  `fit.measures()`) is new math: implement from Gardner-Lubbe, le Roux &
  Gower (2008) and validate against fixtures.
- **Alpha bags / concentration ellipses:** in the retained pipeline these are
  read from the ordination object when present. biplotEZ computes them
  (`alpha.bags`, `ellipses`). Port later as optional methods on the Python
  biplot object (Phase 6): α-bag = convex-hull peeling; concentration
  ellipse = χ² covariance ellipse per class.

### 2.4 R base/stats/cluster utilities that need Python equivalents

These are small but **parity-critical** — they shape every tick mark and
density curve:

| R function | Where used | Python plan |
|---|---|---|
| `pretty(x, n)` | tick selection (`Tickmarks.R`, `keep_pretty_axis_ticks`) | port R's `pretty` algorithm (`labeling`-style; small pure function, unit-tested against R outputs) |
| `stats::density(x, from, to, n=128)` | TDA kernel densities | Gaussian KDE with **R's `bw.nrd0` bandwidth** evaluated on a fixed grid — do *not* use `scipy.stats.gaussian_kde` defaults (Scott's rule ≠ nrd0) |
| `stats::approx(rule = 2)` | `obtain_xhat`, tick interpolation | `numpy.interp` (clamps at ends = rule 2) |
| `cluster::ellipsoidhull` + `predict.ellipsoid` | TDA bounding ellipse, `insert_polygon_EZ` ellipses | implement **minimum-volume enclosing ellipsoid** (Khachiyan's algorithm, ~40 lines) + point generator |
| `grDevices::adjustcolor`, `rgb` | polygon fills, `colorpal` | tiny colour utils (hex + alpha) |
| `stats::median(diff(...))`, `scale()` | tick extension, centering | NumPy one-liners |
| `crayon` tree printing | `print.*` methods | plain-ANSI/`rich`-optional `__repr__` |

---

## 3. Public API mapping

Design rule: the R pipe becomes **method chaining**; every exported R symbol
gets a snake_case Python equivalent; dimension-pair naming (`mdsDisplay_12`,
`"PC 1 & 3"`) and **1-based PC indices stay 1-based** in the public API so
the vocabulary matches the R package and the JS payload.

```python
import bipl5
import pandas as pd

iris = pd.read_csv("iris.csv")

fig = (
    bipl5.init_biplot(iris, center=True, scale=False)
    .scale_mds("pca", classes=iris["Species"])
    .format_samples(stratify="col", by="Species")
    .score_axes()
    .plot()          # returns a Bipl5Widget: _repr_html_ in Jupyter,
)                    # .save("biplot.html"), .show() in a browser
```

| R export | Python | Notes |
|---|---|---|
| `init_biplot(data, center, scale)` | `bipl5.init_biplot(data, center=True, scale=False) -> BiplotSpec` | accepts `pandas.DataFrame` (non-numeric columns retained for `format_samples`) or 2-D `numpy.ndarray` |
| `scale_mds(x, type, ...)` | `BiplotSpec.scale_mds(type="pca", **kwargs) -> Biplot` | keeps the alias resolution (`dimensions`/`eigenvectors`/`classes`/`weighted_cva`/`Dmat`/`dist_func`/`Z`…) but only snake_case aliases; dot-name aliases (`dim.biplot`) die with R syntax |
| `format_samples(x, stratify, by, col, pch)` | `Biplot.format_samples(stratify="col", by=None, col=None, pch=None)` | `by` = column name or array-like; dual-stratification state machine ported as-is |
| `score_axes(x, digits)` | `Biplot.score_axes(digits=2)` | Alves direct-reading errors added to hover tables; warns and no-ops on spline axes |
| `append_mdsDisplay(object, eigenvectors)` | `Biplot.append_mds_display((4, 5))` | re-runs the ordination for the new pair; replays stored `format_samples` state |
| `remove_mdsDisplay(object, mdsDisplay)` | `Biplot.remove_mds_display("mdsDisplay_13")` | string instead of NSE bare name |
| `extract(object, expr, from, what)` | `Biplot.extract("mdsDisplay_12.Data.sample_coordinates")` / `Biplot.extract("fit_measures.CumPred")` | R's `$`-path NSE becomes a dotted-string path; mdsDisplay subset returns a plottable `Biplot`; fit-graph paths return a `BiplotFit` |
| `overlay_fit(x, overlay)` | `Biplot.overlay_fit(overlay=True)` | stores default `fit_display` mode |
| `plot(x, fit_display)` | `Biplot.plot(fit_display="inherit") -> Bipl5Widget` | see §5 rendering |
| `plot(bipl5_fit)` | `BiplotFit.plot() -> plotly.graph_objects.Figure` | R returns ggplot2; Python returns a static plotly Figure (matplotlib not needed as a dependency) |
| `print.*` methods | `__repr__` on `Biplot`, `MdsDisplay`, `BiplotData`, `FitMeasures` | same tree output, locale-safe symbols already adopted upstream |
| `colorpal(number)` | `bipl5.colorpal(number=16)` | identical 16 hex colours |
| `Symbol_List()` | `bipl5.symbol_list()` | plotly symbol vocabulary |

Class hierarchy replacing S3:

```
BiplotSpec        (bipl5_spec)      – data + center/scale flags
Biplot            (bipl5_biplot)    – dict of MdsDisplay + FitMeasures + meta;
                                      .biplot_type in {"pca","cva","pco","reg"}
MdsDisplay        (bipl5_mdsDisplay)– trace_data/layout/config bundle + Data
BiplotData        (bipl5_data)      – sample/axes/translated-axes coordinates
FitMeasures       (bipl5_fitmeasures)
BiplotFit         (bipl5_fit)
OrdinationResult  (internal; the "ez object")
```

Dropped: `wrap_bipl5()` + methods, `.onAttach` startup message, knitr/quarto
S3 shims (Jupyter/Quarto get HTML via `_repr_html_` instead).

---

## 4. Module map (R file → Python module)

```
src/bipl5/
├── __init__.py            # public API: init_biplot, colorpal, symbol_list, classes
├── spec.py                # init_biplot.R (spec + scale_mds dispatch/aliases)
├── biplot.py              # wrap_bipl5.R §constructors, plot(), extract(),
│                          #   append/remove_mds_display, overlay_fit, subset
├── format_samples.py      # format_samples.R (state machine + trace rebuild)
├── score_axes.py          # score_axes.R
├── ordination/
│   ├── base.py            # biplotEZ::biplot equivalent (center/scale, groups)
│   ├── pca.py             # biplotEZ::PCA
│   ├── cva.py             # biplotEZ::CVA
│   ├── pco.py             # biplotEZ::PCO (+ spline axes)
│   ├── regress.py         # biplotEZ::regress
│   ├── aesthetics.py      # samples()/axes()/means() defaults
│   ├── calibration.py     # axes_coordinates + Tickmarks.R + tick post-
│   │                      #   processing from wrap_bipl5_helper.R/PCAbiplot_Helper.R
│   └── fit_measures.py    # fit.measures() + ax_pred.R + FitPanel_funcs.R
│                          #   + fit_quality/regression_fit_* from biplotEZ_helper.R
├── display/
│   ├── mds_display.py     # mdsDisplay_constructor.R
│   ├── builders.py        # build_secondary_biplots.R (scaffolding layout,
│   │                      #   Z traces, class means, polygons, axes, TDA,
│   │                      #   fit panels, slider control)
│   ├── build_one.py       # wrap_bipl5_helper.R (build_one_mdsDisplay,
│   │                      #   build_spline_mdsDisplay, obtain_xhat)
│   └── hover.py           # PCAbiplot_Helper.R (hovertext tables, padding)
├── geometry.py            # Translate.R + Translate_Helpers.R (rotations,
│                          #   MoveLines/translate, MoveDensities, quadrants,
│                          #   shorten_axes, obtain_zhat) + MVEE
├── rcompat.py             # pretty(), nrd0 density(), approx(), colour utils
├── symbols.py             # Plotting_Symbols.R (pch↔plotly maps, colorpal,
│                          #   symbol_list)
└── render/
    ├── figure.py          # plot_scaffolding (biplotEZ_helper.R) as a
    │                      #   plotly.graph_objects.Figure factory
    ├── widget.py          # Bipl5Widget: HTML assembly, _repr_html_, save()
    └── js/
        ├── bipl5_plotly.js    # copied byte-for-byte from R inst/htmlwidgets
        └── bipl5_spline.js    # extracted from insert_spline_js() string
```

---

## 5. Rendering strategy (replacing htmlwidgets)

This is the only architectural piece with no direct Python analogue, and it
should be decided first (Phase 0 spike):

1. Build the figure with **plotly.py** (`render/figure.py` reproduces
   `plot_scaffolding()`: axis domains, updatemenus with `method="skip"`
   buttons, hidden slider, `barmode="stack"`, MathJax config).
2. Add the first mdsDisplay's traces/annotations directly to the figure
   (mirror of `plot.bipl5_biplot` steps 2–3).
3. Serialize the side-channel payload to JSON with a custom encoder
   (NumPy → lists, `None` handling, and R's scalar-vs-length-1-list quirks
   preserved where the JS relies on them — e.g. traces built with
   `x = list(Z[i, 1])` must stay arrays).
4. Emit HTML: `fig.to_html(post_script=...)` gives a hook that runs after
   plot creation with the div id interpolated — inject
   `<script>{bipl5_plotly.js}</script>` plus
   `bipl5Attach(document.getElementById("{plot_id}"), null, {payload});`.
   `Bipl5Widget` wraps this: `_repr_html_` (Jupyter/Quarto), `.save(path)`,
   `.show()`, and `.figure` to escape-hatch the underlying plotly Figure.
5. The JS file is treated as **vendored, shared source**: any upstream fix in
   the R repo is copied across. Payload key names (`mdsDisplays`,
   `fm_mdsDisplay`, `fitDisplay`, `ax_slider`, `initialPCKey`, `p`, `cols`,
   `class_mean_hover`) are frozen contract.

Risk to burn down in the spike: plotly R and plotly.py serialize figures
slightly differently (R's `layoutAttrs` merging, attribute defaults). The
JS reads `el.data`/`el.layout` at runtime, so what matters is the rendered
DOM state, not how it was authored — verify with one end-to-end iris PCA
HTML rendered from hand-built payload fixtures before writing any math.

---

## 6. Phased roadmap

Each phase lands as one or more PRs with tests; later phases depend on
earlier ones. Suggested order optimizes for an early end-to-end demo.

**Phase 0 — Infrastructure + rendering spike**
- Modernize `pyproject.toml` (deps: `numpy`, `scipy`, `pandas`, `plotly`;
  dev: `pytest`, `pytest-cov`, `ruff`; Python ≥3.10), package data for the
  JS, CI workflow, drop cookiecutter placeholders.
- Vendor `bipl5_plotly.js`; build `render/widget.py`; prove the JS attaches
  and the PC-toggle/TDA buttons work against a **hand-exported payload**
  (generate one JSON fixture from the R package for iris PCA).

**Phase 1 — R-compat utilities + geometry (pure functions, heavily tested)**
- `rcompat.py`: `pretty`, `nrd0`-bandwidth KDE on `[from, to]` grids,
  `approx`, colour helpers. Golden tests against values exported from R.
- `geometry.py`: `RotationConstructor` (block rotation matrices), `translate`,
  `MoveLines`, `MoveDensities`, `compute_density_inflation`,
  `get_quads_axes`, `shorten_axes`, `obtain_zhat`, `equation`, MVEE.

**Phase 2 — Ordination engine, PCA first**
- `ordination/base.py` + `pca.py` + `aesthetics.py` + `calibration.py` +
  `fit_measures.py`.
- Numeric parity tests vs R fixtures (iris, with/without scaling,
  correlation biplot, non-default `e_vects`), with **sign-canonicalization**
  (SVD signs are arbitrary in both LAPACK and R; fix a convention —
  largest-magnitude loading positive — and align fixtures before comparing).

**Phase 3 — Payload layer for PCA**
- `display/` + `hover.py` + `spec.py` (PCA branch only) + minimal `biplot.py`
  (`Biplot`, `MdsDisplay`, `BiplotData`, `FitMeasures`, `plot()`).
- Structural parity tests: trace counts, `meta` tags, `legendgroup`s,
  annotation counts, slider config, hover-table strings vs R JSON fixtures.
- **Milestone: `init_biplot(iris).scale_mds("pca").plot()` fully interactive.**

**Phase 4 — API verbs**
- `format_samples` (single + dual stratification + TDA density rebuild),
  `score_axes`, `append/remove_mds_display`, `extract`, `overlay_fit`,
  `BiplotFit.plot()`, `__repr__` tree printers, `colorpal`, `symbol_list`.

**Phase 5 — Remaining ordinations**
- CVA (incl. `Zmeans`, class means, `weighted_cva`, `low_dim`), regression
  biplots (incl. `R²` MathJax quality string), PCO with regression axes,
  then PCO spline axes + `bipl5_spline.js` path.

**Phase 6 — Polish & release**
- Optional alpha-bags/concentration ellipses; docs site (the existing
  `docs/` Sphinx skeleton + a worked notebook per biplot type); README
  rewrite; API reference; version 0.2.0 to PyPI; port the R vignettes
  (`pca-biplots`, `regression-biplots`) as notebooks.

---

## 7. Testing & parity strategy

1. **Golden fixtures from R.** Add `tools/export_fixtures.R` to the R repo
   (or a `fixtures/` dir here) that serializes, for a fixed set of datasets
   and calls: the ordination fields (§2.2), `axes_coordinates` output, fit
   measures, and complete mdsDisplay payloads as JSON. Python tests compare
   against these with `rtol=1e-10` for math, exact for strings/structure.
2. **Port the testthat suite.** The R repo's 11 test files map ~1:1 onto
   pytest modules (`test-init-and-scale-mds.R` → `test_scale_mds.py`, etc.);
   they encode a lot of behavioural detail (alias validation errors, tick
   trimming edge cases, dual-stratification legend rules) worth keeping.
3. **Property tests** for the pure geometry (rotation round-trips, ticks on
   the axis line, densities anchored at the shifted axis).
4. **One rendered-HTML smoke test per biplot type** asserting the JS,
   payload, and `bipl5Attach` call are present and the HTML loads in
   headless Chromium (Playwright, optional CI job).

Known parity traps to encode in tests early:

- SVD/eigen **sign indeterminacy** (fix convention, §6 Phase 2).
- R `factor()` sorts levels alphabetically; use ordered
  `pandas.Categorical` with sorted categories to match legend order.
- R `pretty()` and `density(bw = "nrd0")` defaults (do not approximate).
- 1-based `e_vects` everywhere in public API and names (`mdsDisplay_12`);
  convert to 0-based only at NumPy call sites.
- JSON payload shape: R length-1 vectors serialize as scalars via
  `htmlwidgets` unless wrapped in `list()` — mirror each case as the JS
  expects (audit `customdata`, single-point traces, `meta` arrays).
- Hover tables are **fixed-width text** aligned with `Courier New`; padding
  logic must match to the character.

---

## 8. Dependency changes for `pyproject.toml`

```toml
[tool.poetry.dependencies]
python = ">=3.10"
numpy = ">=1.24"
scipy = ">=1.10"        # KDE grid eval, interpolate (splines), cdist for PCO
pandas = ">=2.0"
plotly = ">=5.15"

[tool.poetry.group.dev.dependencies]
pytest = "*"
pytest-cov = "*"
ruff = "*"
```

(`cluster`, `crayon`, `ggplot2`, `htmltools`, `htmlwidgets`, `knitr` from the
R DESCRIPTION all disappear: MVEE is implemented in-house, printing is plain
text, static fit plots use plotly, and rendering is `render/widget.py`.)

---

## 9. Deliberate behaviour differences (documented, not accidental)

- No `wrap_bipl5()`; no biplotEZ interop of any kind.
- Default sample aesthetics come from `colorpal()` + a fixed plotly symbol
  cycle rather than biplotEZ's palette (validated once against fixtures so
  defaults still look the same for ≤16 groups).
- `extract()` takes dotted strings, not unevaluated expressions.
- `BiplotFit.plot()` returns plotly, not ggplot2.
- Errors become `TypeError`/`ValueError` with the same messages where
  practical (the testthat ports assert on message substrings).
- Spline-axis PCO targets structural/visual parity, not bit-identical
  curves (different smoothing-spline implementations).
