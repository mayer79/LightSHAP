import functools
from itertools import combinations, product

import numpy as np

from lightshap._utils import get_dataclass, get_polars


def replicate_data(X, m):
    """Replicate data X m times.

    If X has rows 0, 1, 2, then the output will have rows 0, 1, 2, 0, 1, 2, ... .

    Parameters
    ----------
    X : DataFrame, array
        The input data to be replicated.
    m : int
        The number of times to replicate the data.

    Returns
    -------
    DataFrame, array
        The replicated data.
    """

    if m <= 0:
        msg = "Replication factor m must be positive"
        raise ValueError(msg)

    xclass = get_dataclass(X)

    if xclass == "np":
        return np.tile(X, (m, 1))
    J = np.tile(np.arange(X.shape[0]), m)
    if xclass == "pd":
        return X.iloc[J]  # no reset index required

    return X[J]  # polars


def repeat_masks(Z, m, pl_schema=None):
    """
    Repeat the masks m times.

    Parameters
    ----------
    Z : ndarray
        The input masks to be repeated.
    m : int
        The number of times to repeat the masks.
    pl_schema : list, optional
        The column names to use if the output is a polars DataFrame.

    Returns
    -------
    ndarray or pl.DataFrame
        The repeated masks.
    """
    out = np.repeat(Z, m, axis=0)
    if pl_schema is not None:
        pl = get_polars()
        out = pl.DataFrame(out, schema=pl_schema)
    return out


def welford_iteration(new_value, avg, sum_squared, j):
    """
    Welford's method for updating the mean and variance incrementally.

    Parameters
    ----------

    new_value: float
        The new value to incorporate into the average and sum of squares.
    avg: float
        The current average.
    sum_squared: float
        The current sum of squared differences from the average.
    j: int
        The number of observations so far (1 for the first obs).
    This is used to compute the updated average and variance.

    Returns
    -------
    tuple
        A tuple containing the updated average and the updated sum of squares.

    """
    delta = new_value - avg
    avg += delta / j
    sum_squared += delta * (new_value - avg)
    return avg, sum_squared


def check_convergence(beta_n, sum_squared, n_iter, tol):
    """Standard error <= tolerance times range of values?

    Checks if the standard error for each output dimension is smaller or equal
    to the tolerance times the range of SHAP values.

    Required only for sampling version of permutation SHAP.

    Parameters
    ----------
    beta_n : array-like
        (p x K) matrix of SHAP values.
    sum_squared : array-like
        (p x K) matrix of sum of squares.
    n_iter : array-like
        Number of iterations.
    tol : float
        The tolerance level.

    Returns
    -------
    bool
        True if the convergence criterion is met for all output dimensions,
        False otherwise.
    """

    shap_range = np.ptp(beta_n, axis=0)
    converged = sum_squared.max(axis=0) <= (tol * n_iter * shap_range) ** 2
    return all(converged)


def generate_all_masks(p):
    """
    Generate a matrix of all possible boolean combinations for p features.

    This creates a 2^p x p boolean matrix where each row is a unique
    combination of True/False values, representing all possible subsets of features.

    Required only for exact permutation SHAP.

    Parameters
    ----------
    p : int
        Number of features

    Returns
    -------
    numpy.ndarray
        A 2^p x p boolean matrix with all possible combinations
    """

    return np.array(list(product([False, True], repeat=p)), dtype=bool)


def generate_partly_exact_masks(p, degree):
    """
    List all length p vectors z with sum(z) in {degree, p - degree} and
    organize them in a boolean matrix with p columns and either choose(p, degree) or
    2 * choose(p, degree) rows.

    Parameters
    ----------
    p : int
        Number of features.
    degree : int
        Degree of the hybrid approach.

    Returns
    -------
    np.ndarray
        A boolean matrix with partly exact masks.
    """
    if degree < 1:
        msg = "degree must be at least 1"
        raise ValueError(msg)
    if 2 * degree > p:
        msg = "p must be >= 2 * degree"
        raise ValueError(msg)
    if degree == 1:
        Z = np.eye(p, dtype=bool)
    else:
        comb = np.array(list(combinations(range(p), r=degree)))
        Z = np.zeros((len(comb), p), dtype=bool)
        row_indices = np.repeat(np.arange(len(comb)), degree)
        col_indices = comb.flatten()
        Z[row_indices, col_indices] = True

    if 2 * degree != p:
        Z = np.vstack((~Z, Z))

    return Z


def random_permutation_from_start(p, start, rng):
    """
    Returns a random permutation of integers from 0 to p-1 starting with value `start`.

    Required only for sampling version of permutation SHAP.

    Parameters
    ----------
    p : int
        Length of the permutation.
    start : int
        The first element of the permutation.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list
        A list representing a random permutation of integers from 0 to p-1,
        starting with the specified `start` value.
    """
    remaining = [i for i in range(p) if i != start]
    rng.shuffle(remaining)
    return [start, *remaining]


def generate_permutation_masks(J, degree):
    """
    Creates a (2 * (p - 1 - 2 * degree) x p) on-off-matrix with
    antithetic permutation scheme.

    Required only for sampling version of permutation SHAP.

    Parameters
    ----------
    J : list
        A permutation vector of length p.
    degree : int
        Row sums of the returned matrix will be within [1 + degree, p - degree - 1].

    Returns
    -------
       A (2 * (p - 1 - 2 * degree) x p) boolean on-off-matrix.
    """
    m = len(J) - 1
    if m <= 2 * degree:
        msg = "J must have at least 2 * degree + 2 elements"
        raise ValueError(msg)
    Z = np.ones((m, m + 1), dtype=bool)
    for i in range(m):
        Z[i : (m + 1), J[i]] = False
    if degree > 0:
        Z = Z[degree:-degree]
    return np.vstack((Z, ~Z))


def check_or_derive_background_data(bg_X, bg_w, bg_n, X, random_state):
    """
    Checks or derives background data against X.

    Parameters
    ----------
    bg_X : DataFrame, array
        Background data.
    bg_w : array-like
        Background weights.
    bg_n : int
        Maximum number of observations in the background data (if bg_X = None).
    X : DataFrame, array
        Input data.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        A tuple containing the background data and weights.
    """
    n, p = X.shape
    xclass = get_dataclass(X)

    # Deal with background weights
    if bg_w is not None:
        if get_dataclass(bg_w) in ("pd", "pl"):
            bg_w = bg_w.to_numpy()  # will use exclusively in np.average(...)
        if bg_w.ndim != 1:
            msg = "bg_w must be a 1D array-like object."
            raise ValueError(msg)
        if bg_X is None and bg_w.shape[0] != n:
            msg = "bg_w must have the same length as X if bg_X is None."
            raise ValueError(msg)
        elif bg_X is not None and bg_w.shape[0] != bg_X.shape[0]:
            msg = "bg_w must have the same length as bg_X."
            raise ValueError(msg)
        if not any(bg_w > 0):
            msg = "bg_w must have at least one positive weight."
            raise ValueError(msg)

    if bg_X is None:
        if n <= 20:
            msg = "Background data must be provided or X must have at least 20 rows."
            raise ValueError(msg)
        bg_X = X.copy() if xclass in ("np", "pd") else X
        if n > bg_n:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(n, size=bg_n, replace=False)
            bg_X = bg_X[indices] if xclass != "pd" else bg_X.iloc[indices]
            if bg_w is not None:
                bg_w = bg_w[indices]
    else:
        if get_dataclass(bg_X) != xclass:
            msg = f"Background data must be of type {xclass}."
            raise TypeError(msg)
        if xclass == "np" and p != bg_X.shape[1]:
            msg = f"Background data must have {p} columns, but has {bg_X.shape[1]}."
            raise ValueError(msg)
        if xclass in ("pd", "pl"):
            if set(bg_X.columns).issuperset(set(X.columns)):
                bg_X = bg_X[X.columns]
            else:
                msg = "Background data must have at least the same columns as X."
                raise ValueError(msg)
    return bg_X, bg_w


def safe_predict(func):
    """Turns predictions into (n x K) numpy array."""
    if not callable(func):
        msg = "predict must be a callable."
        raise TypeError(msg)

    @functools.wraps(func)
    def wrapped(X):
        return np.asarray(func(X)).reshape(X.shape[0], -1)

    return wrapped


def collapse_potential(X, bg_X, bg_w):
    """
    Collapse potential per row against background data bg_X.

    The idea is as follows: if a value in a row of X is equal to q * 100% rows in bg_X,
    then (in exact mode) about q * 100% / 2 of the predictions can be skipped
    (potential). This is accumulated over all columns in X in a multiplicative way.

    Note that missing values are not considered in the current code.

    Parameters
    ----------
    X : DataFrame, array
        The rows to be explained.
    bg_X : DataFrame, array
        The background data.
    bg_w : array
        Weights for the background data.

    Returns
    -------
    float
        The potential collapse value, which is 1 minus the product of
        (1 - proportion of equal values per column) divided by 2.

    """
    if not isinstance(X, np.ndarray):  # then X must be polars or pandas
        X = X.to_numpy()
        bg_X = bg_X.to_numpy()

    potential = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        potential[i] = np.average(X[i] == bg_X, axis=0, weights=bg_w) / 2

    return 1 - np.prod(1 - potential, axis=1)


def collapse_with_index(x, xclass):
    """
    Get unique rows of x and indices to reconstruct the original x.

    Parameters
    ----------
    x : array-like or DataFrame
        Input data to find unique rows in
    xclass : str
        Type of x: 'numpy', 'pandas', or 'polars'

    Returns
    -------
    tuple
        (unique_x, indices_to_reconstruct)
        - unique_x: x with only unique rows
        - indices_to_reconstruct: indices to map from unique_x back to x, or None.
    """
    ix_reconstruct = None

    if xclass == "np":
        try:
            _, ix, ix_reconstruct = np.unique(
                x, return_index=True, return_inverse=True, axis=0
            )
            ix, ix_reconstruct = ix.squeeze(), ix_reconstruct.squeeze()
            unique_x = x[ix]
        except TypeError:
            # If unique fails (e.g., with mixed dtypes), return original data
            unique_x = x
    elif xclass == "pd":
        unique_x = x.drop_duplicates().reset_index(drop=True)

        if len(unique_x) < len(x):
            ix_reconstruct = x.merge(
                unique_x.reset_index(names="_unique_idx_"),
                on=list(x.columns),
                how="left",
            )["_unique_idx_"].to_numpy()
        else:
            unique_x = x
    elif xclass == "pl":
        pl = get_polars()
        unique_x = x.unique(maintain_order=True)

        if len(unique_x) < len(x):
            ix_reconstruct = (
                x.join(
                    unique_x.with_row_index("_unique_idx_"),
                    on=x.columns,
                    how="left",
                    maintain_order="left",
                    nulls_equal=True,
                )
                .select(pl.col("_unique_idx_"))
                .to_numpy()
                .flatten()
            )
        else:
            unique_x = x

    return unique_x, ix_reconstruct


def masked_predict(predict, masks_rep, x, bg_rep, weights, xclass, collapse, bg_n):
    """
    Masked predict function.

    For each on-off vector (rows in mask), the (weighted) average prediction
    is returned from a dataset using x as the "on" values and the background data as
    the "off" values.

    Parameters
    ----------
    predict : callable
        Prediction function that takes data as input and returns a length K vector
        for each row.
    masks_rep : ndarray or pl.DataFrame
        An ((m * n_bg) x p) ndarray or pl.DataFrame with on-off values (boolean mask).
    x : ndarray, pd.Series, or pl.DataFrame
        Row to be explained. Note that for polars, we expect a DataFrame with one row,
        while for pandas, we expect a Series with colnames as index.
    bg_rep : DataFrame, array
        Background data stacked m times, i.e., having shape ((m * n_bg) x p)
    weights : array, optional
        A vector with case weights (of the same length as the unstacked background data).
    xclass : str
        The type of the background data, either "pd" for pandas DataFrame,
        "np" for numpy array, or "pl" for polars DataFrame.
    collapse : bool
        Whether to deduplicate the prediction data.
    bg_n : int
        How many rows does the (non-replicated) background data have.

    Returns
    -------
    array
        A (m x K) ndarray with masked predictions.
    """
    # Apply the masks
    if xclass == "np":
        # If x would have been replicated: bg_masked[mask_rep] = x[mask_rep]
        bg_masked = bg_rep.copy()
        for i in range(masks_rep.shape[1]):
            bg_masked[masks_rep[:, i], i] = x[i]
    elif xclass == "pd":
        bg_masked = bg_rep.copy()
        for i, v in enumerate(bg_masked.columns.to_list()):
            bg_masked.loc[masks_rep[:, i], v] = x[v]
    else:  # polars DataFrame
        pl = get_polars()
        bg_masked = bg_rep.with_columns(
            pl.when(masks_rep[v]).then(pl.lit(x[v])).otherwise(pl.col(v)).alias(v)
            for v in bg_rep.columns
        )
    if collapse:
        bg_masked, ix_reconstruct = collapse_with_index(bg_masked, xclass=xclass)
    else:
        ix_reconstruct = None

    preds = predict(bg_masked)

    if ix_reconstruct is not None:
        preds = preds[ix_reconstruct]

    m_masks = masks_rep.shape[0] // bg_n
    preds = preds.reshape(m_masks, preds.shape[0] // m_masks, -1)  # Avoids splitting
    return np.average(preds, axis=1, weights=weights)  # (m x K)
