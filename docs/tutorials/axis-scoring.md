---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Axis scoring

How trustworthy is a value read directly off a calibrated biplot axis?
`score_axes()` answers per observation and per variable with the
direct-reading diagnostic of Alves (2012): for observation $i$ and
variable $j$,

$$\delta_{ij} = \frac{\lvert x_{ij} - \widehat{x}_{ij} \rvert}{s_j},$$

where $x_{ij}$ is the actual value, $\widehat{x}_{ij}$ the value read
off the calibrated axis (the orthogonal projection of the sample point
onto the displayed axis), and $s_j$ the scaling constant — $1$ for
unscaled data, the column standard deviation when the biplot was built
with `scale=True`. It is reported as a percentage in an extra **Error**
column of every hover table.

```{code-cell} ipython3
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
bp = (
    bipl5.init_biplot(iris, scale=True)
    .scale_mds("pca", classes=iris["Species"])
    .append_mds_display((1, 3))
    .score_axes(digits=2)
)
bp.plot()
```

Hover any point: the table now carries Actual, Pred **and Error**. Two
details worth noticing:

- The error is computed **per dimension pair** — switch to *PC 1 & 3*
  in the dropdown and the readings (and errors) change, because the
  read-off value depends on the displayed plane.
- Scoring is deliberately opt-in: it is a diagnostic step inserted into
  the pipeline, mirroring the R package's
  `init_biplot(data) |> scale_mds() |> score_axes() |> plot()`.

Spline axes do not admit a single calibrated reading, so `score_axes()`
warns and returns spline biplots unchanged.

## References

- Alves, M.R. (2012). Evaluation of the predictive power of biplot axes
  to automate the construction and layout of biplots based on the
  accuracy of direct readings from common outputs of multivariate
  analyses: 1. application to principal component analysis. *Journal of
  Chemometrics*, 26(5), 180–190.
