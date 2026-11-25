"""Static, publication-ready biplots rendered with matplotlib.

The static counterpart of the interactive plotly widget — a port of the R
package's ``wrap_bipl5_gg()`` / ``geom_calibrated_axis()``:

>>> bp = bipl5.init_biplot(iris).scale_mds("pca", classes=iris["Species"])
>>> fig = bp.plot_static(title="Iris")       # a StaticBiplot
>>> fig.save("iris-biplot.pdf")              # vector output

Importing this module imports matplotlib; ``import bipl5`` alone does not.
"""

from .calibrated_axis import AxisStyle, CalibratedAxes, CalibratedAxis, add_calibrated_axes
from .helpers import AxisCalibration, axis_calibration
from .markers import marker_style
from .plot import THEMES, ZORDER, StaticBiplot, plot_static

__all__ = [
    "plot_static",
    "StaticBiplot",
    "CalibratedAxes",
    "CalibratedAxis",
    "AxisStyle",
    "AxisCalibration",
    "axis_calibration",
    "add_calibrated_axes",
    "marker_style",
    "THEMES",
    "ZORDER",
]
