---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# PCO biplots

Principal coordinate analysis (classical multidimensional scaling)
embeds samples from a **distance matrix** rather than from the raw data,
then adds calibrated axes for the original variables. Any Euclidean-
embeddable distance works — supply a matrix (`Dmat=`), a distance
function (`dist_func=`), or let bipl5 default to Euclidean distances.

```{code-cell} ipython3
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
bp = bipl5.init_biplot(iris).scale_mds("pco", classes=iris["Species"])
bp.plot()
```

With the default Euclidean distance this reproduces the PCA
configuration; the value of PCO is swapping in other distances, e.g.:

```python
def sqrt_manhattan(X, **kwargs):
    import numpy as np
    diff = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    return np.sqrt(diff)

bp = bipl5.init_biplot(iris).scale_mds("pco", dist_func=sqrt_manhattan)
```

Axes are fitted to the embedding by regression (`axes="regression"`,
the default): each variable's axis direction comes from regressing the
data on the display coordinates.

## Spline (non-linear) axes

When a variable's relationship to the display is not linear, straight
calibrated axes read poorly. `axes="splines"` fits a **B-spline
trajectory** per variable instead, following the biplotEZ 2.3 optimizer:
a nearest-curve-point loss minimized by a multi-start Nelder-Mead search.

```{code-cell} ipython3
import numpy as np

rng = np.random.default_rng(7)
t = rng.uniform(-2, 2, size=60)
curved = pd.DataFrame({
    "linear": t + rng.normal(scale=0.2, size=60),
    "quadratic": t**2 + rng.normal(scale=0.3, size=60),
    "sine": np.sin(t) + rng.normal(scale=0.2, size=60),
})

bp = bipl5.init_biplot(curved).scale_mds(
    "pco",
    axes="splines",
    # the R defaults run 250 optimizer restarts per axis; for the docs we
    # trade some polish for speed — drop spline_control for full quality
    spline_control={"gamma": 10, "bigsigmaactivate": 2, "seed": 1,
                    "verbose": False},
)
bp.plot()
```

The `quadratic` axis visibly curves. Click anywhere on a spline axis to
pin a tick annotation at that reading; clicking a legend entry hides the
whole axis with its tick marks.

`spline_control` accepts every knob of biplotEZ's
`biplot.spline.axis.control()` (`tau`, `nmu`, `u`, `v`, `lambda_`,
`gamma`, sigma parameters, `itmax`, …) plus two Python extensions:
`seed` for reproducibility and `verbose` to silence progress output.

Spline displays deliberately omit translated axes, fit measures and
`score_axes()` — direct readings are only defined for straight calibrated
axes.
