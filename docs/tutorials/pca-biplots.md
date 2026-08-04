---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# PCA biplots

The flagship display: a principal component analysis biplot with
**calibrated axes** — each original variable is drawn as an axis through
the display, with tick marks placed so that orthogonally projecting a
sample point onto the axis reads off (approximately) that sample's value.

```{code-cell} ipython3
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
bp = bipl5.init_biplot(iris, center=True, scale=True).scale_mds(
    "pca", classes=iris["Species"]
)
bp
```

`init_biplot()` stores the data (non-numeric columns are kept aside for
formatting steps); `scale_mds("pca")` runs the ordination and compiles the
display. Printing the object shows its tree of components. Now plot it:

```{code-cell} ipython3
bp.plot()
```

Everything in the widget is interactive:

- **Hover** a sample point for the fixed-width table of actual vs
  predicted values, plus the observation's *sample predictivity*.
- **Click on a calibrated axis** to drop prediction lines from that
  location to every axis.
- **Measures of Fit** opens the right-hand panel: cumulative axis
  predictivities, cumulative adequacies, the scree plot, the variance
  decomposition, and a summary table.
- **Translated Axes** declutters the view: axes are translated out of the
  data cloud, with per-class kernel densities superimposed. In this mode,
  **Edit: Axes** lets you slide individual axes.
- **Vector Display** switches to the classical arrow view of the loadings.

## More dimension pairs

Additional PC pairs become entries in the dropdown:

```{code-cell} ipython3
bp3 = bp.append_mds_display((1, 3)).append_mds_display((2, 3))
bp3.plot()
```

`remove_mds_display("mdsDisplay_13")` drops a pair again, and
`extract("mdsDisplay_12")` returns a one-pair biplot. Underlying numbers
are a dotted path away:

```{code-cell} ipython3
bp.extract("mdsDisplay_12.Data.sample_coordinates")[:5]
```

## Fit measures as standalone graphs

```{code-cell} ipython3
bp.extract("fit_measures.CumPred").plot()
```

## Correlation biplots

`correlation_biplot=True` rescales the display so cosines of angles
between axes approximate correlations:

```{code-cell} ipython3
bipl5.init_biplot(iris, scale=True).scale_mds(
    "pca", classes=iris["Species"], correlation_biplot=True
).plot()
```

## References

- Gabriel, K.R. (1971). The biplot graphic display of matrices with
  application to principal component analysis. *Biometrika*, 58(3).
- Gower, J.C., Lubbe, S. and le Roux, N.J. (2011). *Understanding
  Biplots*. Wiley.
