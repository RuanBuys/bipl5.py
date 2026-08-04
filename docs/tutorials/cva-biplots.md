---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# CVA biplots

Canonical variate analysis biplots display the data in the space that
maximally separates known classes: the between-class scatter is
decomposed relative to the within-class scatter, and samples are plotted
on the leading canonical variates with calibrated axes for the original
variables.

```{code-cell} ipython3
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
bp = bipl5.init_biplot(iris).scale_mds("cva", classes=iris["Species"])
bp.plot()
```

Class means are shown by default (`show_class_means=True` is the CVA
default), and hover tables report the **within-class sample
predictivity** — each observation's representation quality relative to
its class mean, in the appropriate within-class metric.

## Weighting options

`weighted_cva` controls how classes contribute to the between-class
matrix, exactly as in biplotEZ:

```{code-cell} ipython3
bipl5.init_biplot(iris).scale_mds(
    "cva", classes=iris["Species"], weighted_cva="unweightedI"
).plot()
```

Options are `"weighted"` (class sizes, the default), `"unweightedI"`
(identity) and `"unweightedCent"` (centring matrix).

## Two-class problems

With `g` classes the canonical space has `g - 1` dimensions, so two
classes give a one-dimensional space. The `sample.opt` strategy adds a
second display dimension by maximizing sample predictivity (a warning
reminds you this happened):

```{code-cell} ipython3
import warnings

two = iris.assign(Half=["A"] * 75 + ["B"] * 75)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bp2 = bipl5.init_biplot(two).scale_mds("cva", classes=two["Half"])
bp2.plot()
```

## References

- Gardner-Lubbe, S., le Roux, N.J. and Gower, J.C. (2008). Measures of
  fit in principal component and canonical variate analyses. *Journal of
  Applied Statistics*, 35(9).
