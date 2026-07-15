# bipl5

Create reactive biplots

## Installation

```bash
$ pip install bipl5
```

## Usage

The interactive pipeline mirrors the R package, with method chaining in
place of R's pipe:

```python
import pandas as pd
import bipl5

data = pd.read_csv("mydata.csv")

bp = (
    bipl5.init_biplot(data, center=True, scale=False)
    .scale_mds("pca", classes=data["Species"])   # or "cva" / "pco" / "regress"
    .format_samples(stratify="col", by="Species")  # colour the samples
    .score_axes()                                # optional: Alves reading errors
)

widget = bp.plot()      # displays itself in Jupyter/Quarto
widget.save("biplot.html")   # standalone offline HTML file
widget.show()                # open in a browser
```

The rendered biplot carries the same reactivity as the R package —
calibrated axes with click-to-predict, translated density axes, vector
display, PC-pair switching, and measures-of-fit panels — driven by the same
JavaScript, vendored unchanged from the R package.

More of the API:

```python
# dual stratification: colour by one variable, symbol by another —
# two independent legend sections, each toggling across the other
bp = bp.format_samples(stratify="symbol", by="Band", pch=[15, 17])

bp = bp.append_mds_display((1, 3))          # add a PC pair to the dropdown
bp = bp.remove_mds_display("mdsDisplay_13") # and remove it again
bp = bp.overlay_fit(True)                   # default fit-measure display mode

coords = bp.extract("mdsDisplay_12.Data.sample_coordinates")
fit = bp.extract("fit_measures.CumPred")    # plottable fit graph
fit.plot()
```

The internal ordination engine — a Python port of the parts of the R
package [biplotEZ](https://cran.r-project.org/package=biplotEZ) that bipl5
relies on — is available as `bipl5.ordination`:

```python
from bipl5.ordination import biplot, axes_coordinates

ez = (
    biplot(data, group_aes=data["Species"], scaled=True)
    .pca(e_vects=(1, 2), show_class_means=True)
    .fit_measures()
)
ez.Z                     # sample coordinates in the biplot plane
ez.axis_predictivity     # per-variable measures of fit
axes_coordinates(ez)     # calibrated tick marks for every biplot axis
```

Implemented: PCA (incl. correlation biplots), CVA (incl. the `sample.opt`
low-dimension strategy), PCO with regression axes, and regression biplots
on user-supplied coordinates. Not yet ported: spline axes (see
`TRANSLATION_PLAN.md` for the roadmap).

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`bipl5` was created by Ruan Buys. It is licensed under the terms of the MIT license.

## Credits

`bipl5` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
