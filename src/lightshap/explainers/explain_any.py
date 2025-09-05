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
        A callable to get predictions, e.g., `model.predict`, `model.predict_proba`,
        or lambda x: scipy.special.logit(model.predict_proba(x)[:, -1]).

    X : pd.DataFrame, pl.DataFrame, np.ndarray
        Input data for which explanations are to be generated. Should contain only
        the p feature columns. Must be compatible with `predict`.

    bg_X : pd.DataFrame, pl.DataFrame, np.ndarray, or None, default=None
        Background data used to integrate out "switched off" features,
        typically a representative sample of the training data with 100 to 500 rows.
        Should contain the same columns as `X`, and be compatible with `predict`.
        If None, up to `bg_n` rows of `X` are randomly selected.

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
        for coalitions of size 1, 2, p-2, and p-1.
        If None, it is set to "exact" when p <= 8. Otherwise, if method=="permutation",
        it is set to "sampling". For Kernel SHAP, if 8 < p <= 16, it is set to "h2",
        and to "h1" when p > 16.

    max_iter : int or None, default=None
        Maximum number of iterations for non-exact algorithms. Each iteration represents
        a forward and backward pass through a random permutation.
        For permutation SHAP, one iteration allows to evaluate Shapley's formula
        2*p times (twice per feature).
        p subsequent iterations are starting with different values for faster
        convergence. If None, it is set to 10 * p.

    tol : float, default=0.01
        Tolerance for convergence. The algorithm stops when the estimated standard
        errors are all smaller or equal to `tol * range(shap_values)`
        for each output dimension. Not used when how=="exact".

    random_state : int or None, default=None
        Integer random seed to initialize numpy's random generator. Required for
        non-exact algorithms, and to subsample the background data if `bg_X` is None.

    n_jobs : int, default=1
        Number of parallel jobs to run via joblib. If 1, no parallelization is used.
        If -1, all available cores are used.

    verbose : bool, default=True
        If True, prints information and the tqdm progress bar.

    Returns
    -------
    Explanation object

    Examples
    --------
    **Example 1: Working with Numpy input**

    >>> import numpy as np
    >>> from lightshap import explain_any
    >>>
    >>> # Create synthetic data
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((1000, 4))
    >>>
    >>> # In practice, you would use model.predict, model.predict_proba,
    >>> # or a function thereof, e.g.,
    >>> # lambda X: scipy.special.logit(model.predict_proba(X))
    >>> def predict_function(X):
    ...     linear = X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3]
    ...     interactions = X[:, 0] * X[:, 1] - X[:, 1] * X[:, 2]
    ...     return (linear + interactions).reshape(-1, 1)
    >>>
    >>> # Explain with numpy array (no feature names initially)
    >>> explanation = explain_any(
    ...     predict=predict_function,
    ...     X=X[:100],  # Explain first 100 rows
    ... )
    >>>
    >>> # Set meaningful feature names
    >>> feature_names = ["temperature", "pressure", "humidity", "wind_speed"]
    >>> explanation = explanation.set_feature_names(feature_names)
    >>>
    >>> # Generate plots
    >>> explanation.plot.bar()
    >>> explanation.plot.scatter(["temperature", "humidity"])
    >>> explanation.plot.waterfall(row_id=0)

    **Example 2: Polars input with categorical features**

    >>> import numpy as np
    >>> import polars as pl
    >>> from lightshap import explain_any
    >>>
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>>
    >>> df = pl.DataFrame({
    ...     "age": rng.uniform(18, 80, n).round(),
    ...     "income": rng.exponential(50000, n).round(-3),
    ...     "education": rng.choice(["high_school", "college", "graduate", "phd"], n),
    ...     "region": rng.choice(["north", "south", "east", "west"], n),
    ... }).with_columns([
    ...     pl.col("education").cast(pl.Categorical),
    ...     pl.col("region").cast(pl.Categorical),
    ... ])
    >>>
    >>> # Again, in practice you would use a fitted model's predict instead
    >>> def predict_function(X):
    ...     pred = X["age"] / 50 + X["income"] / 100_000 * (
    ...         1 + 0.5 * X["education"].is_in(["graduate", "phd"])
    ...     )
    ...     return pred
    >>>
    >>> explanation = explain_any(
    ...     predict=predict_function,
    ...     X=df[:200],  # Explain first 200 rows
    ...     bg_X=df[200:400],  # Pass background dataset or use (subset) of X
    ... )
    >>>
    >>> explanation.plot.beeswarm()
    >>> explanation.plot.scatter()
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

    shap_values, se, converged, n_iter = map(np.stack, zip(*res, strict=False))

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
