import numpy as np
from scipy.special import binom

from lightshap._utils import get_dataclass

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


def one_kernelshap(
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
    xclass = get_dataclass(X)
    x = X[i] if xclass != "pd" else X.iloc[i]
    v1 = v1[[i]]  # (1 x K)
    degree = 0 + (how == "h1") + 2 * (how == "h2")
    constraint = v1 - v0  # (1 x K)

    if how != "sampling":
        vz = masked_predict(
            predict=predict,
            masks_rep=precalc["masks_exact_rep"],
            x=x,
            bg_rep=precalc["bg_exact_rep"],
            weights=bg_w,
            xclass=xclass,
            collapse=collapse[i],
            bg_n=bg_n,
        )

        A_exact = precalc["A"]
        b_exact = precalc["Z"].astype(float).T @ (precalc["w"] * (vz - v0))  #  (p x K)

        # Some of the hybrid cases are exact as well
        if how == "exact" or p // 2 == degree:
            beta_n = kernel_solver(A_exact, b_exact, constraint=constraint)  #  (p x K)
            return beta_n, np.zeros_like(beta_n), True, 1

        # Sampling part, using A_exact and b_exact to fill up the weights
        # Credits: https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley.py

    # Sampling part
    rng = np.random.default_rng(random_state)
    pl_schema = None if xclass != "pl" else X.columns
    j = 0

    # Container for results
    beta_n = np.zeros((p, v0.shape[1]), dtype=float)
    sum_squared = np.zeros_like(beta_n)
    converged = False

    A_sum = np.zeros((p, p), dtype=float)
    b_sum = np.zeros_like(beta_n)
    est_m = np.zeros_like(beta_n)

    while not converged and j < max_iter:
        input_sampling = prepare_input_sampling(
            p=p, degree=degree, start=j % p, rng=rng
        )
        j += 1
        Z = input_sampling["Z"]

        # Expensive
        vz = masked_predict(
            predict=predict,
            masks_rep=repeat_masks(Z, m=bg_n, pl_schema=pl_schema),
            x=x,
            bg_rep=precalc["bg_sampling_rep"],
            weights=bg_w,
            xclass=xclass,
            collapse=False,
            bg_n=bg_n,
        )
        A_new = input_sampling["A"]
        b_new = Z.astype(float).T @ (input_sampling["w"] * (vz - v0))

        # Fill the exact part
        if how != "sampling":
            A_new += A_exact
            b_new += b_exact

        # Solve regression on new values to determine standard error and convergence
        est_new = kernel_solver(A_new, b_new, constraint=constraint)
        _, sum_squared = welford_iteration(
            new_value=est_new, avg=est_m, sum_squared=sum_squared, j=j
        )

        # Solve regression on accumulated values
        A_sum += A_new
        b_sum += b_new

        if j > 1:
            beta_n = kernel_solver(A_sum / j, b_sum / j, constraint=v1 - v0)
            converged = check_convergence(
                beta_n=beta_n, sum_squared=sum_squared, n_iter=j, tol=tol
            )
        elif max_iter == 1:  # if n_iter == 1 and max_iter == 1
            beta_n = est_new
            converged = False
            sum_squared = np.full_like(sum_squared, fill_value=np.nan, dtype=float)

    return beta_n, np.sqrt(sum_squared) / j, converged, j


def calculate_kernel_weights(p):
    """
    Kernel weights normalized to a non-empty subset S
    {1, ..., p - 1}.

    The weights represent the Kernel weights given that the coalition vector has
    already been generated.

    Parameters:
    ----------
    p : int
        Total number of features.
    Returns:
    -------
    np.ndarray
        Normalized weights for Kernel SHAP.
    """

    S = np.arange(1, p)
    probs = 1.0 / (binom(p, S) * S * (p - S))

    return probs / probs.sum()


def calculate_kernel_weights_per_coalition_size(p, degree=0):
    """
    Kernel SHAP weights normalized to a non-empty subset S
    {degree + 1, ..., p - degree - 1}.

    The weights represent the Kernel weights of a given coalition size.

    Parameters:
    ----------
    p : int
        Total number of features.
    degree : int
        Degree of the hybrid approach.

    Returns:
    -------
    np.ndarray
        Normalized weights for Kernel SHAP.
    """
    if p < 2 * degree + 2:
        msg = "The number of features p must be at least 2 * degree + 2."
        raise ValueError(msg)

    S = np.arange(1 + degree, p - degree)
    probs = 1.0 / (S * (p - S))

    return probs / probs.sum()


def calculate_exact_prop(p, degree):
    """Total weight to spend.

    How much Kernel SHAP weights do coalitions of size
    {1, ..., deg, ..., p-deg-1 ..., p-1} have?

    Parameters:
    ----------
    p : int
        Total number of features.
    degree : int
        Degree of the hybrid approach, default 0.

    Returns:
    -------
    float
        Value between 0 and 1.
    """
    if degree <= 0:
        return 0.0

    kw = calculate_kernel_weights_per_coalition_size(p)
    w_total = 2.0 * kw[np.arange(degree)].sum()
    if p == 2 * degree:
        w_total -= kw[degree - 1]
    return w_total


def prepare_input_exact(p):
    """
    Calculate the input for exact permutation SHAP.

    This function generates the masks, weights, and A matrix needed for exact
    permutation SHAP.

    Parameters:
    ----------
    p : int
        Number of features.
    Returns:
    -------
    tuple
        A tuple containing:
        - Z: A (2p x p) double matrix with all possible masks.
        - w: A (2p,) array of weights corresponding to the masks.
        - A: A (p x p) matrix used in the SHAP calculations.
    """
    Z = generate_all_masks(p)[1:-1]
    kw = calculate_kernel_weights(p)
    return prepare_Z_w_A(Z, kw=kw, w_total=1.0)


def prepare_input_hybrid(p, degree):
    """
    Calculate the (partial) input for partly exact Kernel SHAP.

    Create Z, w, A for vectors z with sum(z) in {degree, p-degree}
    for k in {1, ..., degree}.
    The total weights do not sum to one, except in the special (exact)
    case degree=p-degree.
    (The remaining weight will be added in the process with calculate_input_sampling().
    Note that for a given k, the weights are constant.

    Parameters:
    ----------
    p : int
        Number of features.
    degree : int
        Degree of the hybrid approach.

    Returns:
    -------
    tuple
        A tuple containing:
        - Z: A boolean matrix with partly exact masks.
        - w: An array of weights corresponding to the masks.
        - A: A (p x p) matrix used in the Kernel SHAP calculations.
    """
    if degree < 1:
        msg = "degree must be at least 1"
        raise ValueError(msg)
    if 2 * degree > p:
        msg = "p must be >= 2 * degree"
        raise ValueError(msg)

    Z_list = []

    for k in range(degree):
        Z = generate_partly_exact_masks(p, degree=k + 1)
        Z_list.append(Z)
    Z = np.vstack(Z_list)
    kw = calculate_kernel_weights(p)
    w_total = calculate_exact_prop(p, degree=degree)  # total weight to spend

    return prepare_Z_w_A(Z, kw=kw, w_total=w_total)


def prepare_input_sampling(p, degree, start, rng):
    """
    Calculate input for sampling Kernel SHAP.

    Let m = 2 * (p - 1 - 2 * degree) be the number of masks to sample.

    Provides random input for paired SHAP sampling:
     - Z: Matrix with m on-off vectors z with sum(z) following
       Kernel weights.
     - w: (m, 1) array of weights corresponding to the masks.
     - A: Matrix A = Z'wZ

    If degree > 0, vectors z with sum(z) restricted to [degree+1, p-degree-1] are drawn.
    This case is used in combination with calculate_input_partly_exact(). Then,
    sum(w) < 1.

    Parameters:
    ----------
    p : int
        Number of features.
    degree : int
        Degree of the hybrid approach.
    start : int
        Starting index for the random permutation.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns:
    -------
    tuple
        A tuple containing:
        - Z: A (m x p) boolean matrix with sampled masks.
        - w: A (m, 1) array of weights corresponding to the masks.
        - A: A (p x p) matrix used in the Kernel SHAP calculations.
    """
    if p < 2 * degree + 2:
        msg = "The number of features p must be at least 2 * degree + 2."
        raise ValueError(msg)

    J = random_permutation_from_start(p, start, rng=rng)
    Z = generate_permutation_masks(J, degree=degree)

    # How much of the total weight do we need to cover?
    w_total = 1.0 if degree == 0 else 1.0 - calculate_exact_prop(p, degree)
    kw = calculate_kernel_weights_per_coalition_size(p)

    return prepare_Z_w_A(Z, kw=kw, w_total=w_total)


def prepare_Z_w_A(Z, kw, w_total=1.0):
    """
    Prepare Z, w, and A for Kernel SHAP.

    Parameters:
    ----------
    Z : np.ndarray
        A boolean matrix with masks.
    kw : np.ndarray
        Kernel weights for each row sum of Z.
    w_total : float, default=1.0
        Total weight to be distributed among the masks.

    Returns:
    -------
    dict
        A dictionary containing:
        - Z: The input mask matrix.
        - w: The weights for the masks, normalized to w_total.
        - A: The (p x p) matrix used in Kernel SHAP calculations.
    """
    w = kw[np.count_nonzero(Z, axis=1) - 1].reshape(-1, 1)
    w *= w_total / w.sum()
    Zf = Z.astype(float)
    A = Zf.T @ (w * Zf)

    return {"Z": Z, "w": w, "A": A}


# Precalculation of things that can be reused over rows
def precalculate_kernelshap(p, bg_X, how):
    """
    Precalculate objects that can be reused over rows for Kernel SHAP.

    Parameters:
    ----------
    p : int
        Number of features.
    bg_X : DataFrame, array
        Background data.
    how : str
        Either "exact", "h2", "h1", or "sampling".

    Returns:
    -------
        dict
            Precalculated objects for Kernel SHAP.
    """
    pl_schema = None if get_dataclass(bg_X) != "pl" else bg_X.columns
    bg_n = bg_X.shape[0]
    degree = 0 + (how == "h1") + 2 * (how == "h2")

    if how == "exact":
        precalc = prepare_input_exact(p)
    elif how in ("h1", "h2"):
        precalc = prepare_input_hybrid(p, degree=degree)
    else:
        precalc = {}

    # Add replicated version of bg_X, and for the exact part, also of X
    if how != "sampling":
        Z = precalc["Z"]
        precalc["masks_exact_rep"] = repeat_masks(Z, m=bg_n, pl_schema=pl_schema)
        precalc["bg_exact_rep"] = replicate_data(bg_X, Z.shape[0])
    if how != "exact":
        precalc["bg_sampling_rep"] = replicate_data(bg_X, 2 * (p - 1 - 2 * degree))

    return precalc


def kernel_solver(A, b, constraint):
    """
    Solve the kernel SHAP constrained optimization.

    We are following Ian Covert's approach in
    https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley.py

    Alternatively, to avoid any singular matrix issues, we could use the following:

    Ainv = np.linalg.pinv(A)
    s = (Ainv @ b).sum(axis=0) - constraint
    s /= Ainv.sum()
    return Ainv @ (b - s[np.newaxis, :])

    The current implementation could be improved by glueing b and 1 together
    to decompose A only once (idea by Christian Lorentzen).

    Parameters:
    ----------
    A : np.ndarray
        (p x p) matrix.
    b : np.ndarray
        (p x K) matrix.
    constraint : np.ndarray
        (1 x K) array equal to v1 - v0.

    Returns:
    -------
    np.ndarray
        (p x K) matrix with the solution to the optimization problem.

    Example:
    >>> A = np.array([[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]])
    >>> b = np.arange(6).reshape(-1, 2)
    >>> constraint = np.arange(2).reshape(1, -1)
    >>> kernel_solver(A, b, constraint)
    """
    try:
        Ainv1 = np.linalg.solve(A, np.ones((A.shape[1], 1)))
        Ainvb = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as err:
        msg = "Matrix A is singular, try hybrid approach or set higher m."
        raise ValueError(msg) from err
    num = np.sum(Ainvb, axis=0, keepdims=True) - constraint
    return Ainvb - Ainv1 @ num / Ainv1.sum()
