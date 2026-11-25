---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Formatting samples

`format_samples()` restyles the observation layer after the ordination
is fitted — grouping, colours and plotting symbols — without refitting
anything.

```{code-cell} ipython3
import pandas as pd
import bipl5

iris = pd.read_csv("../data/iris.csv")
# a second, independent grouping for later
iris["Size"] = pd.qcut(iris["Petal.Length"], 2, labels=["short", "long"])

bp = bipl5.init_biplot(iris, scale=True).scale_mds("pca")
```

## Single stratification

Colour the samples by a stored column (the legend section takes the
column's name):

```{code-cell} ipython3
by_species = bp.format_samples(stratify="col", by="Species")
by_species.plot()
```

`by=` also accepts an array-like of length *n*, `col=` a list of custom
colours (one per class), and `stratify="symbol"` with `pch=` changes
plotting symbols instead — numeric base-R codes or plotly symbol names:

```{code-cell} ipython3
by_species.format_samples(
    stratify="symbol", by="Species", pch=[15, 17, 18]
).plot()
```

Because both calls used the **same** grouping, the legend stays unified:
one section, each class carrying both its colour and its symbol.

## Dual stratification

Using a **different** grouping in the second call splits the legend into
two independent sections — colour by species, symbol by size:

```{code-cell} ipython3
dual = by_species.format_samples(
    stratify="symbol", by="Size", pch=[16, 0]
)
dual.plot()
```

Clicking a *Species* entry hides that species across both sizes;
clicking a *Size* entry hides that size across all species. Internally
the samples are split into hidden colour × symbol combination traces,
which is also what the hover shows.

## Translated-axis densities

The kernel densities on the translated axes always follow the **colour**
grouping: a colour call rebuilds them, a symbol-only call leaves them
unchanged. Toggle **Translated Axes** in either widget above to compare.

Formatting state is stored on the object, so displays added later with
`append_mds_display()` inherit it automatically.
