---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Static biplots for publication

`plot()` gives the interactive widget; `plot_static()` draws the same
biplot as a **static matplotlib figure**, ready for a paper, a thesis or
a slide. It is the Python counterpart of the R package's
`wrap_bipl5_gg()` and follows biplotEZ's way of drawing calibrated
biplots: each variable is an axis line through the display, with
calibrated tick marks you read by orthogonal projection.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
bp = (
    bipl5.init_biplot(iris, scale=True)
    .scale_mds("pca", classes=iris["Species"])
    .format_samples(stratify="col", by="Species")
)
sb = bp.plot_static(title="Iris PCA biplot")
```

The result is a {class}`~bipl5.static.StaticBiplot` holding an ordinary
matplotlib figure. Save it as vector graphics for print, or as a
high-resolution raster:

```python
sb.save("iris-biplot.pdf")            # also .svg / .eps
sb.save("iris-biplot.png", dpi=600)
```

## Tick marks run to the edge of the plot

`axes_coordinates()` calibrates each axis over the range of its fitted
values, so its tick marks stop somewhere inside the plot. `plot_static()`
recovers the exact value-to-position map those ticks imply and
**regenerates the markers along the whole visible part of every axis**.
It keeps biplotEZ's marker values, spacing and phase, but the ticks
always reach the panel border.

This happens every time the figure is drawn, so it also holds when you
change the window afterwards:

```{code-cell} ipython3
sb = bp.plot_static(caption="")
sb.ax.set_xlim(0, 3.6)
sb.ax.set_ylim(-1.5, 1.5);
```

Pass `tick_extend=False` to draw only the markers `axes_coordinates()`
returned. Marker values that the border would cut, or that collide with a
variable name, are left out (`tidy_text=False` keeps them all).

## Every biplot type

CVA biplots get their class means, regression biplots report $R^2$ as the
caption, and PCO biplots with spline axes are drawn as curves with
locally oriented tick marks.

```{code-cell} ipython3
cva = bipl5.init_biplot(iris).scale_mds("cva", classes=iris["Species"])
sb = cva.plot_static(square=True)
```

```{code-cell} ipython3
pco = bipl5.init_biplot(iris, scale=True).scale_mds(
    "pco", axes="splines", classes=iris["Species"],
    spline_control={"gamma": 25, "bigsigmaactivate": 2, "seed": 1},
)
sb = pco.plot_static()
```

Other dimension pairs are one argument away, and `ax=` draws into an
existing matplotlib axes, which is handy for multi-panel figures:

```{code-cell} ipython3
import matplotlib.pyplot as plt

bp3 = bp.append_mds_display((1, 3)).append_mds_display((2, 3))
fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
for ax, pair in zip(axs, [(1, 2), (1, 3), (2, 3)]):
    bp3.plot_static(pair, ax=ax, legend=False, caption="")
```

## Styling

`format_samples()` styling carries over, including dual stratification:

```{code-cell} ipython3
iris["Size"] = np.where(iris["Sepal.Length"] > 5.8, "large", "small")
dual = (
    bipl5.init_biplot(iris, scale=True)
    .scale_mds("pca", classes=iris["Species"])
    .format_samples(stratify="col", by="Species")
    .format_samples(stratify="symbol", by="Size", pch=[16, 2])
)
sb = dual.plot_static()
```

The options mirror `wrap_bipl5_gg()`: axis colours (`axis_colour`,
`tick_colour`, `text_colour`, `label_colour`, any of which can be
`"biplotEZ"` to use the colours set on the ordination), line width and
type, text sizes, `label_placement`, point size and opacity, sample
labels, class means, `legend_title`, `title` / `subtitle` / `caption`, and
the window (`xlim`, `ylim`, `expand`, `square`, `exp_factor`). The
`theme` picks the scaffolding: `"bw"` (default), `"minimal"`, `"classic"`,
or `"biplotEZ"`, a plain framed panel in biplotEZ's style:

```{code-cell} ipython3
sb = bp.plot_static(
    theme="biplotEZ", axis_colour="biplotEZ", exp_factor=1.2, square=True,
    label_placement="peripheral",
)
```

## Adding your own layers

Everything is plain matplotlib, so layers compose by z-order. The biplot
sits at `bipl5.static.ZORDER`: calibrated axes at 1, then polygons,
samples and class means above them. Anything you draw between 0.5 (the
theme's grid) and 1 lies underneath the whole biplot.

Each linear axis also exposes its calibration. `cal.predict(x, y)` returns
the value read off that axis at any point in the plane, which makes a
heatmap of predicted values easy:

```{code-cell} ipython3
sb = bp.plot_static(caption="")
(x0, x1), (y0, y1) = sb.window
gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
petal_length = sb.calibrations[2].predict(gx, gy)
im = sb.ax.imshow(
    petal_length, extent=(x0, x1, y0, y1), origin="lower",
    cmap="Greys", alpha=0.35, aspect="auto", zorder=0.8,
)
cb = sb.figure.colorbar(im, ax=sb.ax, shrink=0.6, location="bottom", label="Petal.Length")
```

## Calibrated axes on hand-built plots

{func}`bipl5.static.add_calibrated_axes` is the counterpart of R's
`geom_calibrated_axis()`. It adds the same axes to any matplotlib plot:

```{code-cell} ipython3
from bipl5.ordination import axes_coordinates, biplot
from bipl5.static import add_calibrated_axes

ez = biplot(iris, scaled=True).pca()
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
ax.scatter(ez.Z[:, 0], ez.Z[:, 1], s=8, color="grey")
ax.set_aspect("equal")
artist = add_calibrated_axes(ax, axes_coordinates(ez), ez.col_names)
```

The ordination objects have `ez.plot_static()` too, the analogue of
calling `wrap_bipl5_gg()` on a biplotEZ object.
