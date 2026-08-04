# Coming from the R package

bipl5.py ports the R package's `init_biplot()` pipeline. The R pipe
becomes method chaining, names become snake_case, and everything else is
designed to feel familiar.

```r
# R
init_biplot(iris) |>
  scale_mds("pca", classes = iris$Species) |>
  format_samples(stratify = "col", by = Species) |>
  score_axes() |>
  plot()
```

```python
# Python
(bipl5.init_biplot(iris)
    .scale_mds("pca", classes=iris["Species"])
    .format_samples(stratify="col", by="Species")
    .score_axes()
    .plot())
```

## API mapping

| R | Python | Notes |
|---|---|---|
| `init_biplot(data, center, scale)` | `bipl5.init_biplot(data, center=True, scale=False)` | takes a DataFrame or 2-D array |
| `scale_mds(x, type, ...)` | `.scale_mds("pca", **kwargs)` | snake_case argument aliases only (`dim.biplot` → `dimensions`/`dim_biplot`) |
| `format_samples(x, stratify, by, col, pch)` | `.format_samples(stratify=, by=, col=, pch=)` | `by` is a column-name string, an array, or a named Series (its name labels the legend, standing in for R's bare-symbol NSE) |
| `score_axes(x, digits)` | `.score_axes(digits=2)` | identical semantics |
| `append_mdsDisplay(x, c(1, 3))` | `.append_mds_display((1, 3))` | PC indices stay **1-based** |
| `remove_mdsDisplay(x, mdsDisplay_13)` | `.remove_mds_display("mdsDisplay_13")` | string instead of a bare name |
| `extract(x, mdsDisplay_12$Data$sample_coordinates)` | `.extract("mdsDisplay_12.Data.sample_coordinates")` | dotted-string paths instead of `$` expressions |
| `extract(x, fit_measures$CumPred)` | `.extract("fit_measures.CumPred")` | returns a plottable `BiplotFit` |
| `overlay_fit(x, TRUE)` | `.overlay_fit(True)` | same stored-default semantics |
| `plot(x)` | `.plot()` | returns a `Bipl5Widget`: displays itself in notebooks, `.save("f.html")`, `.show()` |
| `plot(bipl5_fit)` | `BiplotFit.plot()` | returns an interactive figure (plotly) rather than ggplot2 |
| `print(x)` | `repr(x)` / just evaluate it | same tree diagrams, unicode/ASCII + colours auto-detected |
| `colorpal(n)`, `Symbol_List()` | `bipl5.colorpal(n)`, `bipl5.symbol_list()` | identical palette |
| `wrap_bipl5()` | — | not ported: it adapts externally built biplotEZ objects, which don't exist in Python |

## Things that stay the same

- **1-based dimension vocabulary**: `e_vects=(1, 3)`, `mdsDisplay_13`,
  "PC 1 & 3" — exactly the R names.
- The **JavaScript layer is byte-identical**: the interactive behaviour
  is the same code the R package ships.
- Display quality strings, hover-table layout, tree printing, default
  colours and symbols all match the R output.

## Things to know

- The ordination mathematics is an internal port of the parts of
  **biplotEZ** that bipl5 uses (`bipl5.ordination`), following the
  GitHub 2.3 sources — including the C++ spline optimizer.
- Grouping vectors follow R factor semantics: pandas Categoricals keep
  their category order; other inputs get levels in order of appearance
  (`format_samples`) or sorted (`scale_mds` classes), matching R.
- Widgets pin **plotly.js 2.x** (what R's plotly bundles); plotly.py's
  bundled 3.x is not compatible with the shared JavaScript.
