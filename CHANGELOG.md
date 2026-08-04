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

## v0.1.0 (24/03/2026)

- Initial package skeleton.
