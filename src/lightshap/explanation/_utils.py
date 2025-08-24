from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import gaussian_kde


def safe_to_float(x):
    """
    Convert a pandas Series to float, handling categorical items.
    The function preserves missing values.

    Parameters
    ----------
    x : pd.Series
        Input data to convert.

    Returns
    -------
    pd.Series
        Converted Series with float or int values.
    """
    is_numeric = pd.api.types.is_numeric_dtype(x)
    is_categorical = isinstance(x.dtype, pd.CategoricalDtype)

    if not is_numeric and not is_categorical:
        msg = f"Unsupported dtype {x.dtype} for Series."
        raise TypeError(msg)
    elif is_categorical:
        x = x.cat.codes.replace(-1, np.nan)

    return x.astype(float)


def min_max_scale(x):
    """
    Scale the input data to the range [0, 1] using min-max scaling.

    Parameters
    ----------
    X : pd.Series
        Input data to scale.

    Returns
    -------
    pd.Series
        Scaled data in the range [0, 1].
    """

    if x.isna().all():
        return x

    xmin, xmax = x.min(), x.max()

    # Constant column (retaining missing values)
    if xmax == xmin:
        return x * 0.0 + 0.5

    return (x - xmin) / (xmax - xmin)


def halton(i, base=2):
    """
    Generate the i-th element of the Halton sequence.

    Source: https://en.wikipedia.org/wiki/Halton_sequence

    Parameters
    ----------
    i : int
        Index (1-based)
    base : int, optional
        Base for the sequence, default is 2

    Returns
    -------
    float
        The i-th value in the sequence
    """
    result = 0
    f = 1
    while i > 0:
        f /= base
        result += f * (i % base)
        i = i // base
    return result


def halton_sequence(n, base=2):
    """
    Generate the first n elements of the Halton sequence.

    Parameters
    ----------
    n : int
        Number of elements to generate
    base : int, optional
        Base for the sequence, default is 2

    Returns
    -------
    numpy.ndarray
        Array of the first n elements in the Halton sequence
    """
    return np.array([halton(i + 1, base) for i in range(n)])


def beeswarm_jitter(values, halton_vals=None):
    """
    Compute jitter values for beeswarm plot based on density.

    Parameters
    ----------
    values : array-like
        Values to create jitter for
    halton_vals : array-like
        Precomputed Halton sequence for jittering

    Returns
    -------
    numpy.ndarray
        Jitter values for each point
    """
    if len(values) == 1:
        return np.zeros(1, dtype=float)

    # Density at each point
    try:
        kde = gaussian_kde(values)
        density = kde(values)
        density_normalized = density / density.max()
    except ValueError:
        # Uniform if KDE fails
        density_normalized = np.ones_like(values, dtype=float)

    # Quasi-random values based on ranks
    if halton_vals is None:
        halton_vals = halton_sequence(len(values))
    ranks = np.argsort(np.argsort(values))
    shifts = halton_vals[ranks] - 0.5

    # Scale shifts by density
    return 2 * shifts * density_normalized


def plot_layout(p):
    """
    Determine plot layout based on the number of plots

    Parameters
    ----------
    p : int
        Number of plots

    Returns
    -------
    tuple
        Number of rows and columns for the plot layout
    """
    if p <= 3:
        return 1, p
    elif p <= 6:
        return (p + 1) // 2, 2
    elif p <= 12:
        return (p + 2) // 3, 3
    else:
        return (p + 3) // 4, 4


def _check_features(features, all_features, name="features"):
    """
    Check and validate feature names.

    Parameters
    ----------
    features : iterable
        Feature names to check.
    all_features : iterable
        All available feature names.
    name : str, optional
        Name of the feature set (for error messages).

    Returns
    -------
    iterable
        Validated feature names.
    """
    if features is None:
        return all_features
    elif isinstance(features, Iterable) and not isinstance(features, str):
        if not set(features).issubset(all_features):
            msg = f"Some {features} are not present in the data."
            raise ValueError(msg)
    else:
        msg = f"{name} must be an iterable of names, or None."
        raise TypeError(msg)

    return features


def _safe_cor(x, y):
    """
    Compute Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    x : array-like
        First input array.
    y : array-like
        Second input array.

    Returns
    -------
    float
        The Pearson correlation coefficient, or 0 if not computable.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return 0.0
    x, y = x[ok], y[ok]
    x_sd, y_sd = x.std(ddof=1), y.std(ddof=1)

    if x_sd <= 1e-7 or y_sd <= 1e-7:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def get_text_bbox(ax):
    """Get the bounding box of the text labels in the plot in the order
    x left, x right, y bottom, y top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the text labels.

    Returns
    -------
    tuple
        The bounding box coordinates of the text labels (x left, x right, y bottom, y top).
    """
    renderer = ax.get_figure().canvas.get_renderer()
    left, right, bottom, top = [], [], [], []
    for text in ax.texts:
        text_bbox = text.get_window_extent(renderer=renderer)
        text_bbox_data = text_bbox.transformed(ax.transData.inverted())

        # Might be simplified
        left.append(text_bbox_data.x0)
        right.append(text_bbox_data.x1)
        bottom.append(text_bbox_data.y0)
        top.append(text_bbox_data.y1)
    return min(left), max(right), min(bottom), max(top)


def color_axis_info(z, cmap, max_color_labels, max_color_label_length, **kwargs):
    """
    Prepare color axis information for a given color feature.

    Helper function of plot.scatter().

    Parameters
    ----------
    z : pd.Series
        The color feature values.
    cmap : str or matplotlib colormap
        The colormap to use.
    max_color_labels : int
        The maximum number of color labels to display.
    max_color_label_length : int
        The maximum length of color labels.

    Returns
    -------
    dict
        A dictionary containing color axis information.
    """
    out = {}
    if isinstance(z.dtype, pd.CategoricalDtype):
        out["categorical"] = True
        out["mapping"] = dict(enumerate(z.cat.categories))
        z = z.cat.codes.replace(-1, np.nan)
        n = out["n_colors"] = len(out["mapping"])
        base_colors = plt.get_cmap(cmap, n)(np.linspace(0, 1, n))
        out["cmap"] = ListedColormap(base_colors)
        out["norm"] = BoundaryNorm(np.arange(-0.5, n + 0.5), n)

        # Reduce number of labels on color bar
        if n > max_color_labels:
            step = int(np.ceil(n / max_color_labels))
            for i, key in enumerate(out["mapping"]):
                if 0 < i < n - 1 and i % step > 0:
                    out["mapping"][key] = ""

        # Truncate long labels
        for key, value in out["mapping"].items():
            if len(value) > max_color_label_length:
                out["mapping"][key] = value[:max_color_label_length]
    else:
        out["cmap"] = plt.get_cmap(cmap)
        out["categorical"] = False

    out["values"] = z
    out["cmap"].set_bad("gray", alpha=kwargs.get("alpha", 1.0))

    return out
