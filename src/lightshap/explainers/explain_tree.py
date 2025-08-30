import sys

from lightshap.explanation.explanation import Explanation


def explain_tree(model, X):
    """
    Calculate TreeSHAP for XGBoost, LightGBM, and CatBoost models.

    The following model types are supported:

    - xgboost.Booster
    - xgboost.XGBModel
    - xgboost.XGBRegressor
    - xgboost.XGBClassifier
    - xgboost.XGBRFClassifier
    - xgboost.XGBRFRegressor
    - lightgbm.Booster
    - lightgbm.LGBMModel
    - lightgbm.LGBMRanker
    - lightgbm.LGBMRegressor
    - lightgbm.LGBMClassifier
    - catboost.CatBoost
    - catboost.CatBoostClassifier
    - catboost.CatBoostRanker
    - catboost.CatBoostRegressor

    Parameters
    ----------
    model : XGBoost, LightGBM, or CatBoost model
        A fitted model.
    X : array-like
        The input data for which SHAP values are to be computed.

    Returns
    -------
    Explanation
        An Explanation object.

    Examples
    --------

    >>> # Example 1: XGBoost regression
    >>> import numpy as np
    >>> import pandas as pd
    >>> from lightshap import explain_tree
    >>>
    >>> import xgboost as xgb
    >>>
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame(
    ...     {
    ...         "X1": rng.normal(0, 1, 100),
    ...         "X2": rng.uniform(-2, 2, 100),
    ...         "X3": rng.choice([0, 1, 2], 100),
    ...     }
    ... )
    >>> y = X["X1"] + X["X2"] ** 2 + X["X3"] + rng.normal(0, 0.1, 100)
    >>> model = xgb.train({"learning_rate": 0.1}, xgb.DMatrix(X, label=y))
    >>>
    >>> explanation = explain_tree(model, X)
    >>> explanation.plot.beeswarm()
    >>> explanation.plot.scatter()

    >>> # Example 2: LightGBM Multi-Class Classification
    >>> import numpy as np
    >>> import pandas as pd
    >>> from lightgbm import LGBMClassifier
    >>> from lightshap import explain_tree
    >>>
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame(
    ...     {
    ...         "X1": rng.normal(0, 1, 100),
    ...         "X2": rng.uniform(-2, 2, 100),
    ...         "X3": rng.choice([0, 1, 2], 100),
    ...     }
    ... )
    >>> y = X["X1"] + X["X2"] ** 2 + X["X3"] + rng.normal(0, 0.1, 100)
    >>> y = pd.cut(y, bins=3, labels=[0, 1, 2])
    >>> model = LGBMClassifier(max_depth=3, verbose=-1)
    >>> model.fit(X, y)
    >>>
    >>> # SHAP analysis
    >>> explanation = explain_tree(model, X)
    >>> explanation.set_output_names(["Class 0", "Class 1", "Class 2"])
    >>> explanation.plot.bar()
    >>> explanation.plot.scatter(which_output=0)  # Class 0
    """
    if is_xgboost(model):
        shap_values, X, feature_names = xgb_shap(model, X=X)
    elif is_lightgbm(model):
        shap_values, X, feature_names = lgb_shap(model, X=X)
    elif is_catboost(model):
        shap_values, X, feature_names = catboost_shap(model, X=X)
    else:
        msg = (
            "Model must be a LightGBM, XGBoost, or CatBoost model."
            "Note that not all model subtypes are supported."
        )
        raise TypeError(msg)

    # Extract baseline
    if shap_values.ndim >= 3:  # (n x K x p) multi-output model
        baseline = shap_values[0, :, -1]
        shap_values = shap_values[:, :, :-1].swapaxes(1, 2)  # (n x p x K)
    else:
        baseline = shap_values[0, -1]
        shap_values = shap_values[:, :-1]

    # Note that shap_values have shape (n, p) or (n, p, K) at this point, even
    # for single row X.
    return Explanation(shap_values, X=X, baseline=baseline, feature_names=feature_names)


def is_lightgbm(x):
    """Returns True if x is a LightGBM model with SHAP support.

    The following model types are supported:

    - lightgbm.Booster
    - lightgbm.LGBMModel
    - lightgbm.LGBMRanker
    - lightgbm.LGBMRegressor
    - lightgbm.LGBMClassifier

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is a LightGBM model with SHAP support, and False otherwise.
    """
    try:
        lgb = sys.modules["lightgbm"]
    except KeyError:
        return False
    return isinstance(
        x,
        lgb.Booster
        | lgb.LGBMModel
        | lgb.LGBMRanker
        | lgb.LGBMRegressor
        | lgb.LGBMClassifier,
    )


def is_xgboost(x):
    """Returns True if x is an XGBoost model with SHAP support.

    The following model types are supported:

        - xgboost.Booster
        - xgboost.XGBModel
        - xgboost.XGBRegressor
        - xgboost.XGBClassifier
        - xgboost.XGBRFClassifier
        - xgboost.XGBRFRegressor

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is an XGBoost model with SHAP support, and False otherwise.
    """
    try:
        xgb = sys.modules["xgboost"]
    except KeyError:
        return False
    return isinstance(
        x,
        xgb.Booster
        | xgb.XGBRanker
        | xgb.XGBModel
        | xgb.XGBRegressor
        | xgb.XGBClassifier
        | xgb.XGBRFClassifier
        | xgb.XGBRFRegressor,
    )


def is_catboost(x):
    """Returns True if x is a CatBoost model with SHAP support.

    The following model types are supported:

    - catboost.CatBoost
    - catboost.CatBoostClassifier
    - catboost.CatBoostRanker
    - catboost.CatBoostRegressor

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is a CatBoost model with SHAP support, and False otherwise.
    """
    try:
        catboost = sys.modules["catboost"]
    except KeyError:
        return False
    return isinstance(
        x,
        catboost.CatBoost
        | catboost.CatBoostClassifier
        | catboost.CatBoostRanker
        | catboost.CatBoostRegressor,
    )


def lgb_shap(model, X):
    """Calculate SHAP values for LightGBM models.

    Parameters
    ----------
    model : lightgbm.Booster or similar
        The LightGBM model to explain.

    X : array-like
        The input data for which to compute SHAP values. Passed to model.predict().
        Cannot be a lightgbm.Dataset.

    Returns
    -------
    shap_values : np.ndarray
        The computed SHAP values.

    X : Same as input X.

    feature_names : list
        A list of feature names.
    """

    import lightgbm as lgb  # noqa: PLC0415

    if isinstance(X, lgb.Dataset):
        msg = "X cannot be a lgb.Dataset."
        raise TypeError(msg)

    n, p = X.shape

    shap_values = model.predict(X, pred_contrib=True)

    # Multi-output: Turn (n x (K * (p + 1))) -> (n x K x (p + 1))
    if shap_values.shape[1] != p + 1:
        shap_values = shap_values.reshape(n, -1, p + 1)

    # Extract feature names
    if isinstance(model, lgb.Booster):
        feature_names = model.feature_name()
    else:
        feature_names = model.feature_name_
    return shap_values, X, feature_names


def xgb_shap(model, X):
    """Calculate SHAP values for XGBoost models.

    Parameters
    ----------
    model : xgboost.Booster or similar
        The XGBoost model to explain.

    X : xgb.DMatrix or array-like
        The input data for which to compute SHAP values.

    Returns
    -------
    shap_values : np.ndarray
        The computed SHAP values.

    X : array-like
        If X is a xgb.DMatrix, the result of X.get_data().toarray().
        Otherwise, the input X.

    feature_names : list, or None
        A list of feature names, or None.

    """

    import xgboost as xgb  # noqa: PLC0415

    # Sklearn API predict() does not have pred_contribs argument
    if not isinstance(model, xgb.Booster):
        model = model.get_booster()

    if not isinstance(X, xgb.DMatrix):
        X_pred = xgb.DMatrix(X)
    else:
        X_pred = X
        X = X.get_data().toarray()

    shap_values = model.predict(X_pred, pred_contribs=True)

    return shap_values, X, model.feature_names


def catboost_shap(model, X):
    """Calculate SHAP values for CatBoost models.

    Parameters
    ----------
    model : catboost.CatBoost or similar
        The CatBoost model to explain.

    X : catboost.Pool or array-like
        The input data for which to compute SHAP values.

    Returns
    -------
    shap_values : np.ndarray
        The computed SHAP values.

    X : array-like
        If X is a catboost.Pool, the result of X.get_features(). Otherwise, the input X.

    feature_names : list
        A list of feature names.
    """

    import catboost  # noqa: PLC0415

    if not isinstance(X, catboost.Pool):
        X_pred = catboost.Pool(X, cat_features=model.get_cat_feature_indices())
    else:
        X_pred = X
        X = X.get_features()

    shap_values = model.get_feature_importance(data=X_pred, fstr_type="ShapValues")

    return shap_values, X, model.feature_names_
