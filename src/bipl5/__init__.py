# read version from installed package
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bipl5")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

from bipl5 import ordination

__all__ = ["ordination", "__version__"]
