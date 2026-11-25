---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Regression biplots

A regression biplot starts from **display coordinates you supply** — any
two-dimensional configuration, from any method — and fits calibrated
axes by regressing the data on those coordinates. The display quality is
reported as an $R^2$ decomposition over the display dimensions.

```{code-cell} ipython3
import pandas as pd
import bipl5
from bipl5.ordination import biplot, pco

iris = pd.read_csv("../data/iris.csv")

# any external 2-D configuration works; here: a classical MDS embedding
Z = pco(biplot(iris)).Z

bp = bipl5.init_biplot(iris).scale_mds(
    "regress", Z=Z, classes=iris["Species"]
)
bp.plot()
```

The quality label above the plot is the sequential decomposition

$$R^2_{disp} = R_1^2 + R_{2 \mid 1}^2,$$

computed by orthogonalizing the display dimensions in order, so the two
contributions add up while preserving the displayed Dim 1 / Dim 2
labelling (rendered with MathJax in the widget).

```{code-cell} ipython3
bp.meta["fit_quality"]
```

Predicted values in the hover tables are read directly off the
calibrated axes by projection and interpolation — the same mechanism
`score_axes()` uses.
