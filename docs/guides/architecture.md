# Architecture

bipl5.py is a three-layer port of the R package, sharing its JavaScript
verbatim.

## 1. Ordination engine (`bipl5.ordination`)

A self-contained NumPy/pandas port of the parts of
[biplotEZ](https://github.com/MuViSU/biplotEZ) that bipl5 relies on
(GitHub 2.3 sources): `biplot()`, `pca()`, `cva()`, `pco()`,
`regress()`, aesthetics defaults, `fit_measures()`,
`axes_coordinates()` and the spline-axis optimizer. No plotting code —
it is designed to grow into a full biplotEZ replication for static
biplots.

```python
from bipl5.ordination import biplot, axes_coordinates

ez = biplot(data, scaled=True).pca(e_vects=(1, 2)).fit_measures()
ez.Z                    # sample coordinates
ez.axis_predictivity    # measures of fit
axes_coordinates(ez)    # calibrated tick marks
```

## 2. Payload layer (`bipl5.display`, `bipl5.geometry`)

Builds the plotly trace/layout dictionaries ("mdsDisplays") the
interactive layer consumes: sample traces with fixed-width hover tables,
calibrated axes with tick annotations, translated density axes (axis
translation geometry, minimum-volume enclosing ellipsoid, R-compatible
kernel densities), fit-measure panels and slider metadata. Trace `meta`
tags are a frozen contract with the JavaScript.

## 3. Rendering (`bipl5.render`)

`plot()` seeds a plotly figure with the first display and attaches the
vendored `bipl5_plotly.js` — **byte-identical to the R package's** —
with a JSON payload carrying the remaining displays and fit measures.
Spline biplots attach the lighter `bipl5_spline.js` instead. Rendering
pins plotly.js 2.x, the major version the JavaScript targets.

## Parity with R

- Numerical conventions were taken directly from the biplotEZ sources
  (eigenvalue scaling, axis calibration, CVA weighting, fit measures in
  the appropriate metrics), and R-critical primitives are faithful ports:
  `pretty()` tick selection, `bw.nrd0` kernel densities, R factor
  semantics, R number formatting in hover tables.
- Deliberate differences are small and documented: dotted-string
  `extract()` paths, plotly instead of ggplot2 for standalone fit
  graphs, and Python extensions `seed`/`verbose` on the spline control.
- Spline axes match R structurally and behaviourally, not bit-for-bit —
  the optimizer is multi-start random search by construction.
