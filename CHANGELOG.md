# Changelog

<!--next-version-placeholder-->

## v0.2.0 (unreleased)

Complete Python port of the R package's `init_biplot()` pipeline.

- Interactive API: `init_biplot()` → `scale_mds()` (PCA incl. correlation
  biplots, CVA incl. the `sample.opt` low-dimension strategy, PCO with
  regression or spline axes, regression biplots) → `plot()`, rendering
  the same reactive HTML widgets as the R package via its vendored
  JavaScript (plotly.js pinned to 2.x).
- Verbs: `format_samples()` (single and dual stratification with
  translated-density rebuilds), `score_axes()` (Alves direct-reading
  errors), `append_mds_display()` / `remove_mds_display()`, `extract()`
  with dotted paths, `overlay_fit()`, tree-style printing.
- Internal ordination engine (`bipl5.ordination`) porting the parts of
  biplotEZ 2.3 (GitHub sources) that bipl5 relies on, including the
  C++-backed spline-axis optimizer reimplemented in NumPy.
- `Bipl5Widget` for notebook display, standalone offline HTML saves and
  browser preview.
- Documentation: executed tutorials with live widgets, guides (including
  an R-to-Python migration table) and a full API reference.
- Static, publication-ready biplots: `Biplot.plot_static()` (and
  `plot_static()` on ordination objects) renders any 2-D biplot with
  matplotlib, porting the R package's `wrap_bipl5_gg()` /
  `geom_calibrated_axis()`. Calibrated axes are laid out at draw time so
  tick marks always reach the panel edge; spline axes are drawn as curves;
  `format_samples()` styling (incl. dual stratification), class means,
  captions and themes (`"bw"`, `"minimal"`, `"classic"`, `"biplotEZ"`)
  carry over. `bipl5.static.add_calibrated_axes()` adds calibrated axes to
  any matplotlib plot. matplotlib is now a dependency.
- `plot_static(halo_colour=..., halo_width=...)` (and `AxisStyle.halo_colour`
  / `halo_width`) outline the calibrated axes — lines, tick marks, marker
  values and variable names — so they stay legible over a heatmap or other
  busy layer drawn beneath the biplot.

## v0.1.0 (24/03/2026)

- Initial package skeleton.
