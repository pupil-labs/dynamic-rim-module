"""Top-level entry-point for the <dynamic_content_on_rim> package"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil-labs-dynamic-rim")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]

from . import dynamic_rim

if __name__ == "__main__":
    dynamic_rim.main()
