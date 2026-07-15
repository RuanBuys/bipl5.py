# read version from installed package
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bipl5")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

from bipl5 import ordination
from bipl5.biplot import Biplot, BiplotFit
from bipl5.format_samples import format_samples
from bipl5.score_axes import score_axes
from bipl5.spec import BiplotSpec, init_biplot
from bipl5.symbols import colorpal, symbol_list

__all__ = [
    "init_biplot",
    "BiplotSpec",
    "Biplot",
    "BiplotFit",
    "format_samples",
    "score_axes",
    "colorpal",
    "symbol_list",
    "ordination",
    "__version__",
]
