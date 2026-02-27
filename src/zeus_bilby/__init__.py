"""Interface and plugin for using zeus(-mcmc) in bilby."""

from importlib.metadata import PackageNotFoundError, version

from .sampler import Zeus

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = ["Zeus"]
