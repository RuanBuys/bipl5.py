# bipl5

Create reactive biplots

## Installation

```bash
$ pip install bipl5
```

## Usage

The interactive `init_biplot()` pipeline is still under construction (see
`TRANSLATION_PLAN.md`). The internal ordination engine — a Python port of the
parts of the R package [biplotEZ](https://cran.r-project.org/package=biplotEZ)
that bipl5 relies on — is available today:

```python
import pandas as pd
from bipl5.ordination import biplot, axes_coordinates

data = pd.read_csv("mydata.csv")

bp = (
    biplot(data, group_aes=data["Species"], scaled=True)
    .pca(e_vects=(1, 2), show_class_means=True)   # or .cva() / .pco() / .regress()
    .fit_measures()
)

bp.Z                     # sample coordinates in the biplot plane
bp.axis_predictivity     # per-variable measures of fit
axes_coordinates(bp)     # calibrated tick marks for every biplot axis
```

Implemented methods: PCA (incl. correlation biplots), CVA (incl. the
`sample.opt` low-dimension strategy), PCO with regression axes, and
regression biplots on user-supplied coordinates. Spline axes are not yet
ported.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`bipl5` was created by Ruan Buys. It is licensed under the terms of the MIT license.

## Credits

`bipl5` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
