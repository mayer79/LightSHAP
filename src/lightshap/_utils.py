import numpy as np

# Handle polars import at module level
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


def get_dataclass(data):
    """Determine the type of the input.

    Returns a string indicating whether the input is pandas,
    numpy, or polars, or raises a TypeError otherwise.

    Both Series-like and DataFrame-like objects are accepted.

    Parameters
    ----------
    data : DataFrame-like or Series-like
        The input data to determine the type of.

    Returns
    -------
    str
        A string indicating the type of the data: "pd" for pandas,
        "np" for numpy array, or "pl" for polars DataFrame.
    """
    if isinstance(data, np.ndarray):
        return "np"
    if hasattr(data, "iloc"):
        return "pd"
    if hasattr(data, "with_columns"):
        return "pl"
    else:
        msg = "Unknown data class. Expected 'numpy', 'pandas', or 'polars'"
        raise KeyError(msg)


def get_polars():
    """Get polars module or raise error if not available."""
    if not HAS_POLARS:
        raise ImportError(
            "polars is required but is not installed. "
            "Install it with: pip install polars"
        )
    return pl
