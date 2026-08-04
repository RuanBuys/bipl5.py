---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# bipl5

**Reactive calibrated-axes biplots in Python** — a full port of the
[bipl5 R package](https://github.com/RuanBuys/bipl5), rendering PCA, CVA,
PCO and regression biplots as interactive HTML with calibrated axes,
translated density axes, and measures of fit.

```{code-cell} ipython3
:tags: [hide-input]
import pandas as pd
import bipl5

iris = pd.read_csv("data/iris.csv")
(
    bipl5.init_biplot(iris, scale=True)
    .scale_mds("pca", classes=iris["Species"])
    .format_samples(stratify="col", by="Species")
    .score_axes()
    .plot()
)
```

The biplot above is live: hover a point for its Actual / Pred / Error
table, click **Translated Axes** or **Measures of Fit**, switch PC pairs,
or click on a calibrated axis to read off predictions.

## Installation

```bash
pip install bipl5
```

## The pipeline in one look

```python
import bipl5

bp = (
    bipl5.init_biplot(data, center=True, scale=False)  # 1. store the data
    .scale_mds("pca", classes=data["Species"])         # 2. run the ordination
    .format_samples(stratify="col", by="Species")      # 3. style the samples
    .score_axes()                                      # 4. axis-reading errors
)
bp.plot()                    # display in a notebook…
bp.plot().save("biplot.html")  # …or save standalone offline HTML
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/pca-biplots
tutorials/cva-biplots
tutorials/pco-biplots
tutorials/regression-biplots
tutorials/formatting-samples
tutorials/axis-scoring
```

```{toctree}
:maxdepth: 1
:caption: Guides

guides/coming-from-r
guides/saving-and-sharing
guides/architecture
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
api/ordination
changelog
contributing
conduct
```
