"""Top-level entry-point for the <dynamic_content_on_rim> package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.dynamic_content_on_rim")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
