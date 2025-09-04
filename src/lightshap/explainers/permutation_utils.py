import numpy as np
from scipy.special import binom

from lightshap.utils import get_dataclass

from ._utils import (
    check_convergence,
    generate_all_masks,
    generate_partly_exact_masks,
    generate_permutation_masks,
    masked_predict,
    random_permutation_from_start,
    repeat_masks,
    replicate_data,
    welford_iteration,
)


def one_permshap(
    i,
    predict,
    how,
    bg_w,
    v0,
    max_iter,
    tol,
    random_state,
    X,
    v1,
    precalc,
    collapse,
    bg_n,
):
    """
    Explain a single row of input data.

    Parameters:

    Returns:
    - Explanation for the row.
    """
    p = X.shape[1]
    K = v0.shape[1]
    xclass = get_dataclass(X)
    x = X[i] if xclass != "pd" else X.iloc[i]
    v1 = v1[[i]]  # (1 x K)

    # Container for results
    beta_n = np.zeros((p, K), dtype=float)
    sum_squared = np.zeros_like(beta_n)
    converged = False
    j = 0

    if how == "exact":
        vz = np.zeros((2**p, K), dtype=float)
        vz[0] = v0
        vz[-1] = v1

        vz[1:-1] = masked_predict(
            predict=predict,
            masks_rep=precalc["masks_exact_rep"],
            x=x,
            bg_rep=precalc["bg_exact_rep"],
            weights=bg_w,
            xclass=xclass,
            collapse=collapse[i],
            bg_n=bg_n,
        )

        for k in range(p):
            on, off = precalc["positions"][k]
            # Remember that shapley_weights have been computed without the first row
            beta_n[k] = np.average(
                vz[on] - vz[off], axis=0, weights=precalc["shapley_weights"][on - 1]
            )
        return beta_n, sum_squared, True, 1

    # Sampling mode
    rng = np.random.default_rng(random_state)

    vz_balanced = masked_predict(
        predict=predict,
        masks_rep=precalc["masks_balanced_rep"],
        x=x,
        bg_rep=precalc["bg_balanced_rep"],
        weights=bg_w,
        xclass=xclass,
        collapse=False,
        bg_n=bg_n,
    )

    pl_schema = None if xclass != "pl" else X.columns

    # vz has constant first, middle, and last row
    vz = np.zeros((2 * p + 1, K), dtype=float)
    vz[[0, -1]] = v1
    vz[p] = v0

    # Important positions to be filled in vz
    from_balanced = [1, 1 + p, p - 1, 2 * p - 1]
    from_iter = np.r_[2 : (p - 1), (p + 2) : (2 * p - 1)]

    while not converged and j < max_iter:
        # Cycle through p
        k = j % p
        chain = random_permutation_from_start(p, start=k, rng=rng)
        masks = generate_permutation_masks(chain, degree=1)
        j += 1

        vzj = masked_predict(
            predict=predict,
            masks_rep=repeat_masks(masks, m=bg_n, pl_schema=pl_schema),
            x=x,
            bg_rep=precalc["bg_sampling_rep"],
            weights=bg_w,
            xclass=xclass,
            collapse=False,
            bg_n=bg_n,
        )

        # Fill vz first by pre-calculated masks, then by current iteration
        vz[from_balanced] = vz_balanced[[k, k + p, chain[p - 1] + p, chain[p - 1]]]
        vz[from_iter] = vzj

        # Evaluate Shapley's formula 2p times
        J = np.argsort(chain)
        forward = vz[J] - vz[J + 1]
        backward = vz[p + J + 1] - vz[p + J]
        new_value = (forward + backward) / 2

        beta_n, sum_squared = welford_iteration(
            new_value=new_value, avg=beta_n, sum_squared=sum_squared, j=j
        )

        if j > 1:  # otherwise, sum_squared is still 0
            converged = check_convergence(
                beta_n=beta_n, sum_squared=sum_squared, n_iter=j, tol=tol
            )

    return beta_n, np.sqrt(sum_squared) / j, converged, j


def precalculate_permshap(p, bg_X, how):
    """
    Precalculate objects needed for sampling version of permutation SHAP.

    Parameters:
    ----------
    p : int
        Number of features.
    bg_X : DataFrame, array
        Background data.
    how : str
        Either "exact" or "sampling".

    Returns:
    -------
    dict
        Precalculated objects for permutation SHAP.
    """
    pl_schema = None if get_dataclass(bg_X) != "pl" else bg_X.columns
    bg_n = bg_X.shape[0]

    if how == "exact":
        M = generate_all_masks(p)
        other_players = M[1:].sum(axis=1) - 1  # first row cannot be "on"

        precalc = {
            "masks_exact_rep": repeat_masks(M[1:-1], m=bg_n, pl_schema=pl_schema),
            "bg_exact_rep": replicate_data(bg_X, m=2**p - 2),  # masks_rep.shape[0]
            "shapley_weights": calculate_shapley_weights(p, other_players),
            "positions": positions_for_exact(M),
        }
    elif p >= 4:  #  how == "sampling"
        M = generate_partly_exact_masks(p, degree=1)
        precalc = {
            "masks_balanced_rep": repeat_masks(M, m=bg_n, pl_schema=pl_schema),
            "bg_balanced_rep": replicate_data(bg_X, 2 * p),
            "bg_sampling_rep": replicate_data(bg_X, 2 * (p - 3)),
        }
    else:
        msg = "sampling method not implemented for p < 4."
        raise ValueError(msg)

    precalc["bg_n"] = bg_n

    return precalc


def calculate_shapley_weights(p, ell):
    """Calculate Shapley weights for a given number of features and off-features.

    The function is vectorized over ell.

    Parameters:
    ----------
    p : int
        Total number of features.
    ell : array-like
        Number of features that are off (not included in the subset).

    Returns:
    -------
    float
        The Shapley weight for the given number of features and off-features.
    """
    return 1.0 / binom(p, ell) / (p - ell)


def positions_for_exact(mask):
    """
    Precomputes positions for exact permutation SHAP.

    For each feature j, this function calculates the indices of the rows in the full
    mask with column j = True ("on"), and the indices of *corresponding* off rows.

    Parameters:
    ----------
    mask : (2**p, p) boolean matrix
        Matrix representing on-off info

    Returns:
    -------
    list of length p
        Each element represents a tuple with
        - Row indices in `mask` of "on" positions for feature j
        - Row indices in `mask` with corresponding "off" positions for feature j
    """
    p = mask.shape[1]
    codes = np.arange(mask.shape[0])  # Row index = binary code of the row

    positions = []
    for j in range(p):
        on = codes[mask[:, j]]
        off = on - 2 ** (p - 1 - j)  # trick to turn "bit" off
        positions.append((on, off))

    return positions
