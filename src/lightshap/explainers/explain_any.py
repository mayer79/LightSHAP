import warnings

import joblib
import numpy as np

from lightshap.explanation.explanation import Explanation

from ._utils import check_or_derive_background_data, collapse_potential, safe_predict
from .kernel_utils import one_kernelshap, precalculate_kernelshap
from .parallel import ParallelPbar
from .permutation_utils import one_permshap, precalculate_permshap


def explain_any(
    predict,
    X,
    bg_X=None,
    bg_w=None,
    bg_n=200,
    method=None,
    how=None,
    max_iter=None,
    tol=0.01,
    random_state=None,
    n_jobs=1,
    verbose=True,
):
    """
    SHAP values for any model

    Calculate SHAP values for any model using either Kernel SHAP or Permutation SHAP.
    By default, it uses Permutation SHAP for p <= 8 features and a hybrid between
    exact and sampling Kernel SHAP for p > 8 features.

    Parameters
    ----------
    predict : callable
        A callable to get predictions, i.e. `predict(X)`.

    X : pd.DataFrame, pl.DataFrame, np.ndarray
        Input data for which explanations are to be generated. Should contain only
        feature columns.

    bg_X : pd.DataFrame, pl.DataFrame, np.ndarray, or None, default=None
        Background data used to integrate out "switched off" features,
        typically a representative sample of the training data with 100 to 500 rows.
        Should contain the same columns as `X`.
        If None, 200 rows of `X` are randomly selected as background data.

    bg_w : pd.Series, pl.Series, np.ndarray, or None, default=None
        Weights for the background data. If None, equal weights are used.
        If `bg_X` is None, `bg_w` must have the same length as `X`.

    bg_n : int, default=200
        If `bg_X` is None, that many rows are randomly selected from `X`
        to use as background data. Values between 50 and 500 are recommended.

    method: str, or None, default=None
        Either "kernel", "permutation", or None.
        If None, it is set to "permutation" when p <= 8, and to "kernel" otherwise.

    how: str, or None, default=None
        If "exact", exact SHAP values are computed. If "sampling", iterative sampling
        is used to approximate SHAP values. For Kernel SHAP, hybrid approaches between
        "sampling" and "exact" options are available: "h1" uses exact calculations
        for coalitions of size 1 and p-1, whereas "h2" uses exact calculations
        for coalitions of size 1, 2, and p-2, p-1.
        If None, it is set to "exact" when p <= 8. Otherwise, if method=="permutation",
        it is set to "sampling". For Kernel SHAP, if 8 < p <= 16, it is set to "h2",
        and to "h1" when p > 16.

    max_iter : int or None, default=None
        If None, it is set to 10 * p, where p is the number of features in `X`.
        Maximum number of iterations for the algorithm. Each iteration represents
        a forward and backward pass through a random permutation of the values from
        1+degree to p-1-degree, where degree is 0 if how=="sampling", 1 if how=="h1",
        and 2 if how=="h2". p subsequent iterations are starting with different values.
        Thus, `max_iter` should be a multiple of p. Not used when mode=="exact".

    tol : float, default=0.01
        Tolerance for convergence. The algorithm stops when the standard error
        is smaller or equal to `tol * range(shap_values)` for each output dimension.
        Not used when mode=="exact".

    random_state : int or None, default=None
        Integer random seed to initialize numpy's random generator. Required for
        non-exact algorithms, and to subsample the background data if `bg_X` is None.

    n_jobs : int, default=1
        Number of parallel jobs to run. If 1, no parallelization is used.
        If -1, all available cores are used. Uses joblib.

    verbose : bool, default=True
        If True, prints information.

    Returns
    -------
    Explanation object
    """
    bg_X, bg_w = check_or_derive_background_data(
        bg_X=bg_X, bg_w=bg_w, bg_n=bg_n, X=X, random_state=random_state
    )
    bg_n = bg_X.shape[0]
    n, p = X.shape

    if p < 2:
        msg = "At least two features are required."
        raise ValueError(msg)

    if method is None:
        method = "permutation" if p <= 8 else "kernel"
    elif method not in ("permutation", "kernel"):
        msg = "method must be 'permutation', 'kernel', or None."
        raise ValueError(msg)

    if how is None:
        if p <= 8:
            how = "exact"
        elif method == "permutation":
            how = "sampling"
        else:  # "kernel"
            how = "h2" if p <= 16 else "h1"
    elif method == "permutation" and how not in ("exact", "sampling"):
        msg = "how must be 'exact', 'sampling', or None for permutation SHAP."
        raise ValueError(msg)
    elif method == "kernel" and how not in ("exact", "sampling", "h1", "h2"):
        msg = "how must be 'exact', 'sampling', 'h1', 'h2', or None for kernel SHAP."
        raise ValueError(msg)
    if method == "permutation" and how == "sampling" and p < 4:
        msg = (
            "Sampling Permutation SHAP is not supported for p < 4."
            "Use how='exact' instead."
        )
        raise ValueError(msg)
    if method == "kernel" and how == "h1" and p < 4:
        msg = (
            "Degree 1 hybrid Kernel SHAP is not supported for p < 4."
            "Use how='exact' instead."
        )
        raise ValueError(msg)
    elif method == "kernel" and how == "h2" and p < 6:
        msg = (
            "Degree 2 hybrid Kernel SHAP is not supported for p < 6."
            "Use how='exact' instead."
        )
        raise ValueError(msg)

    if max_iter is None:
        max_iter = 10 * p
    elif not isinstance(max_iter, int) or max_iter < 1:
        msg = "max_iter must be a positive integer or None."
        raise ValueError(msg)

    # Ensures predictions are (n, K) numpy arrays
    predict = safe_predict(predict)

    # Get base value (v0) and predictions (v1)
    v1 = predict(X)  # (n x K)
    v0 = np.average(predict(bg_X), weights=bg_w, axis=0, keepdims=True)  # (1 x K)

    # Precalculation of things that can be reused over rows
    if method == "permutation":
        precalc = precalculate_permshap(p, bg_X, how=how)
    else:  # method == "kernel"
        precalc = precalculate_kernelshap(p, bg_X, how=how)

    # Should we try to deduplicate prediction data? Only if we can save 25% of rows.
    if False:  # how in ("exact", "h2"):
        collapse = collapse_potential(X, bg_X=bg_X, bg_w=bg_w) >= 0.25
    else:
        collapse = np.zeros(n, dtype=bool)

    if verbose:
        how_text = how
        if how in ("h1", "h2"):
            prop_ex = 100 * precalc["w"].sum()
            how_text = f"hybrid degree {1 if how == 'h1' else 2}, {prop_ex:.0f}% exact"
        print(f"{method.title()} SHAP ({how_text})")

    res = ParallelPbar(disable=not verbose)(n_jobs=n_jobs)(
        joblib.delayed(one_permshap if method == "permutation" else one_kernelshap)(
            i,
            predict=predict,
            how=how,
            bg_w=bg_w,
            v0=v0,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            X=X,
            v1=v1,
            precalc=precalc,
            collapse=collapse,
            bg_n=bg_n,
        )
        for i in range(n)
    )

    shap_values, se, converged, n_iter = map(np.stack, zip(*res))

    if converged is not None and not converged.all():
        non_converged = converged.shape[0] - np.count_nonzero(converged)
        warnings.warn(
            f"{non_converged} rows did not converge. "
            f"Consider using a larger tol or higher max_iter.",
            UserWarning,
            stacklevel=2,
        )

    return Explanation(
        shap_values,
        X=X,
        baseline=v0,
        standard_errors=se,
        converged=converged,
        n_iter=n_iter,
    )
