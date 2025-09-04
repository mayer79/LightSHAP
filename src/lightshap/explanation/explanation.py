import math

import numpy as np
import pandas as pd

from lightshap.utils import get_dataclass

from ._utils import _check_features, _safe_cor, safe_to_float
from .explanationplotter import ExplanationPlotter


class Explanation:
    """SHAP Explanation object that encapsulates model explanations.

    The Explanation class provides a comprehensive framework for storing, analyzing,
    and visualizing SHAP (SHapley Additive exPlanations) values, which help interpret
    machine learning model predictions. This class supports both single-output and
    multi-output models, handles feature importance analysis, and offers various
    visualization methods.

    The class stores SHAP values along with the associated data points, baseline
    values, and optionally includes standard errors, convergence indicators, and
    iteration counts for approximation methods. It provides methods to select subsets
    of the data, calculate feature importance, and create various visualizations
    including waterfall plots, dependence plots, summary plots, and importance plots.

    Parameters
    ----------
    shap_values : numpy.ndarray
        numpy.ndarray of shape (n_obs, n_features) for single-output models, and
        of shape (n_obs, n_features, n_outputs) for multi-output models.

    X : pandas.DataFrame, polars.DataFrame, numpy.ndarray
        Feature values corresponding to `shap_values`. The columns must be in the
        same order.

    baseline : float or numpy.ndarray, default=0.0
        The baseline value(s) representing the expected model output when all
        features are missing. For single-output models, either a scalar or a
        numpy.ndarray of shape (1, ).
        For multi-output models, an array of shape (n_outputs,).

    feature_names : list or None, default=None
        Feature names. If None and X is a pandas DataFrame, column names
        are used. If None and X is not a DataFrame, default names are generated.

    output_names : list or None, default=None
        Names of the outputs for multi-output models. If None, default names are
        generated.

    standard_errors : numpy.ndarray or None, default=None
        Standard errors of the SHAP values. Must have the same shape as shap_values,
        or None. Only relevant for approximate methods.

    converged : numpy.ndarray or None, default=None
        Boolean array indicating the convergence status per observation. Only
        relevant for approximate methods.

    n_iter : numpy.ndarray or None, default=None
        Number of iterations per observation. Only relevant for approximate methods.

    Attributes
    ----------
    shap_values : numpy.ndarray
        numpy.ndarray of shape (n_obs, n_features) for single-output models, and
        of shape (n_obs, n_features, n_outputs) for multi-output models.

    X : pandas.DataFrame
        The feature values corresponding to `shap_values`. Note that the index
        is reset to the values 0 to n_obs - 1.

    baseline : numpy.ndarray
        Baseline value(s). Has shape (1, ) for single-output models, and
        shape (n_outputs, ) for multi-output models.

    standard_errors : numpy.ndarray or None
        Standard errors of the SHAP values of the same shape as `shap_values`
        (if available).

    converged : numpy.ndarray or None
        Convergence indicators of shape (n_obs, ) (if available).

    n_iter : numpy.ndarray or None
        Iteration counts of shape (n_obs, ) (if available).

    shape : tuple
        Shape of `shap_values`.

    ndim : int
        Number of dimensions of the SHAP values (2 or 3).

    feature_names : list
        Feature names.

    output_names : list or None
        Output names for multi-output models. None for single-output models.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from lightshap import Explanation
    >>>
    >>> # Example data
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> shap_values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>>
    >>> explanation = Explanation(shap_values, X, baseline=0.5)
    >>>
    >>> # Waterfall plot of first observation
    >>> explanation.plot.waterfall(row_id=0)
    """

    def __init__(
        self,
        shap_values,
        X,
        baseline=0.0,
        feature_names=None,
        output_names=None,
        standard_errors=None,
        converged=None,
        n_iter=None,
    ):
        if not isinstance(shap_values, np.ndarray) or shap_values.shape[0] < 1:
            msg = "SHAP values must be a numpy array with at least one row."
            raise TypeError(msg)

        n = shap_values.shape[0]

        if standard_errors is not None and standard_errors.shape != shap_values.shape:
            msg = (
                f"Shape {standard_errors.shape} of standard_errors does not match "
                f"shape {shap_values.shape} of SHAP values."
            )
            raise ValueError(msg)
        if converged is not None and converged.shape[0] != n:
            msg = (
                f"Length {converged.shape[0]} of converged does not match "
                f"number {n} of rows of SHAP values."
            )
            raise ValueError(msg)
        if n_iter is not None and n_iter.shape[0] != n:
            msg = (
                f"Length {n_iter.shape[0]} of n_iter does not match "
                f"number {n} of rows of SHAP values."
            )
            raise ValueError(msg)

        # Drop third dimension of shap_values and standard_errors if unnecessary
        if shap_values.ndim == 3 and shap_values.shape[2] == 1:
            shap_values = shap_values.reshape(n, -1)
            if standard_errors is not None:
                standard_errors = standard_errors.reshape(n, -1)
        elif not 2 <= shap_values.ndim <= 3:
            msg = "SHAP values must be 2D or 3D."
            raise ValueError(msg)

        # Baseline should have shape (K, )
        if not isinstance(baseline, np.ndarray):
            baseline = np.asarray(baseline)
        baseline = baseline.flatten()  #   turn into 1D array
        K = 1 if shap_values.ndim == 2 else shap_values.shape[2]
        if baseline.shape[0] != K:
            msg = (
                f"Length {len(baseline)} of baseline does not match "
                f"number {K} of output dimensions."
            )
            raise ValueError(msg)

        self.shap_values = shap_values
        self.baseline = baseline
        self.standard_errors = standard_errors
        self.converged = converged
        self.n_iter = n_iter

        # Some attributes for convenience
        self.shape = shap_values.shape
        self.ndim = shap_values.ndim

        self.set_output_names(output_names)

        # Setting X also sets feature names
        self.set_X(X)
        if feature_names is not None:
            self.set_feature_names(feature_names)

    @property
    def plot(self):
        """
        Accessor for plotting methods.

        Examples
        --------
        >>> explanation.plot.bar()
        >>> explanation.plot.waterfall(row_id=0)
        >>> explanation.plot.beeswarm()
        >>> explanation.plot.scatter(features=["feature1", "feature2"])
        """
        return ExplanationPlotter(self)

    def __repr__(self):
        # Get shapes and sample sizes for display
        n = self.shape[0]
        n_display = min(2, n)

        out = "SHAP Explanation\n\n"

        # SHAP values section
        out += f"SHAP values {self.shape}, first {n_display}:\n"
        out += f"{self.shap_values[:n_display]!r}\n\n"

        # Data section
        out += f"X, first {n_display} rows:\n"
        out += str(self.X.head(2))

        return out

    def __len__(self):
        return self.shape[0]

    def filter(self, indices):
        """
        Filter the SHAP values by array-like.

        Parameters
        ----------
        indices : array-like
            Integer or boolean array-like to filter the SHAP values and data.

        Returns
        -------
        Explanation
            A new Explanation object with filtered SHAP values and data.
        """
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices)

        if not (np.issubdtype(indices.dtype, np.integer) or indices.dtype == np.bool_):
            msg = "indices must be an integer or boolean array-like."
            raise TypeError(msg)

        values = self.shap_values[indices]
        se = self.standard_errors[indices] if self.standard_errors is not None else None
        X = self.X[indices] if indices.dtype == np.bool_ else self.X.iloc[indices]

        return Explanation(
            shap_values=values,
            X=X,
            baseline=self.baseline,
            output_names=self.output_names,
            standard_errors=se,
            converged=self.converged[indices] if self.converged is not None else None,
            n_iter=self.n_iter[indices] if self.n_iter is not None else None,
        )

    def select_output(self, index):
        """
        Select specific output dimension from the SHAP values. Useful if
        predictions are multi-output.

        Parameters
        ----------
        index : Int or str
            Index or name of the output dimension to select.

        Returns
        -------
        Explanation
            A new Explanation object with only the selected output.
        """
        if self.ndim != 3:
            return self

        if self.output_names is not None and isinstance(index, str):
            index = self.output_names.index(index)
        elif not isinstance(index, int):
            msg = "index must be an integer or string."
            raise TypeError(msg)

        if self.standard_errors is not None:
            se = self.standard_errors[:, :, index]
        else:
            se = None

        return Explanation(
            shap_values=self.shap_values[:, :, index],
            X=self.X,
            baseline=self.baseline[[index]],  # need to keep np.array
            output_names=None,
            standard_errors=se,
            converged=self.converged,
            n_iter=self.n_iter,
        )

    def set_feature_names(self, feature_names):
        """
        Set feature names of 'X'.

        Parameters
        ----------
        feature_names : list or array-like
            Feature names to set.
        """
        p = self.X.shape[1]
        if len(feature_names) != p:
            msg = (
                f"Length {len(feature_names)} of feature_names does not match "
                f"number {p} of columns in X."
            )
            raise ValueError(msg)
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)

        self.X.columns = self.feature_names = feature_names

        return self

    def set_output_names(self, output_names=None):
        """
        If predictions are multi-output, set names of the additional dimension.

        Parameters
        ----------
        output_names : list or array-like, optional
            Output names to set.
        """
        if self.ndim == 3:
            K = self.shap_values.shape[2]
            if output_names is None:
                output_names = list(range(K))
            elif len(output_names) != K:
                msg = (
                    f"Length {len(output_names)} of output_names does not match "
                    f"number {K} of outputs in SHAP values."
                )
                raise ValueError(msg)
        else:
            output_names = None

        if output_names is not None and not isinstance(output_names, list):
            output_names = list(output_names)

        self.output_names = output_names

        return self

    def set_X(self, X):
        """Set X and self.feature_names.

        `X` is converted to pandas. String and object columns are converted to
        categoricals, while numeric columns are left unchanged. Other column types
        will raise a TypeError.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame or polars.DataFrame
            New data to set. Columns must match the order of SHAP values.
        """
        if X.shape != self.shap_values.shape[:2]:
            msg = (
                f"Shape {X.shape} of X does not match shape "
                f"{self.shap_values.shape[:2]} of SHAP values."
            )
            raise ValueError(msg)

        xclass = get_dataclass(X)
        if xclass == "np":
            if hasattr(self, "feature_names") and self.feature_names is not None:
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                X = pd.DataFrame(X)
        elif xclass == "pl":
            try:
                X = X.to_pandas()
            except Exception as e:
                msg = (
                    "Failed to convert polars DataFrame to pandas. "
                    "Make sure polars is properly installed: pip install polars"
                )
                raise ImportError(msg) from e
        else:  # pd
            X = X.reset_index(drop=True)

        # Columns will stay numeric/boolean or become categorical
        for v in X.columns:
            is_numeric = pd.api.types.is_numeric_dtype(X[v])
            is_categorical = isinstance(X[v].dtype, pd.CategoricalDtype)
            if not is_numeric and not is_categorical:
                is_string = pd.api.types.is_string_dtype(X[v])
                is_object = pd.api.types.is_object_dtype(X[v])

                if is_string or is_object:
                    X[v] = X[v].astype("category")
                else:
                    msg = f"Column {v} has unsupported dtype {X[v].dtype}."
                    raise TypeError(msg)

        self.X = X
        self.feature_names = self.X.columns.to_list()

        return self

    def importance(self, which_output=None):
        """
        Calculate mean absolute SHAP values for each feature (and output dimension).

        Parameters
        ----------
        which_output : int or string, optional
            Index or name of the output dimension to calculate importance for.
            If None, all outputs are considered. Only relevant for multi-output models.

        Returns
        -------
        pd.Series or pd.DataFrame
            Series containing mean absolute SHAP values sorted by importance.
            In case of multi-output models, it returns a DataFrame, and the sort
            order is determined by the average importance across all outputs.
        """
        if self.ndim == 3 and which_output is not None:
            self = self.select_output(which_output)  # noqa: PLW0642

        imp = np.abs(self.shap_values).mean(axis=0)

        if self.ndim == 2:
            imp = pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        else:  # ndim == 3 -> we sort by average importance across outputs
            imp = pd.DataFrame(imp, index=self.feature_names, columns=self.output_names)
            imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
        return imp

    def interaction_heuristic(self, features=None, color_features=None):
        """Interaction heuristic.

        For each feature/color_feature combination, the weighted average absolute
        Pearson correlation coefficient between the SHAP values of the feature
        and the values of the color_feature is calculated. The larger the value,
        the higher the potential interaction.

        Notes:

        - Non-numeric color features are converted to numeric, which does not always
          make sense.
        - Missing values in the color feature are currently discarded.
        - The number of non-missing color values in the bins are used as weight to
          compute the weighted average.

        Parameters
        ----------
        features : list, optional
            List of feature names. If None, all features are used.
        color_features : list, optional
            List of color feature names. If None, all features are used.

        Returns
        -------
        pd.DataFrame
            DataFrame with interaction heuristics. `feature_names` serve as index,
            and `color_features` as columns.
        """
        features = _check_features(features, self.feature_names)
        color_features = _check_features(
            color_features, self.feature_names, name="color features"
        )

        idx = [self.feature_names.index(f) for f in features]

        df = self.X[features]
        df_color = self.X[color_features].apply(safe_to_float)  # to numeric
        df_shap = pd.DataFrame(self.shap_values[:, idx], columns=df.columns)

        nbins = math.ceil(min(np.sqrt(df.shape[0]), df.shape[0] / 20))

        out = pd.DataFrame(0.0, index=df.columns, columns=df_color.columns)

        for xname in df.columns:
            xgroups = df[xname]

            if pd.api.types.is_numeric_dtype(xgroups) and xgroups.nunique() > nbins:
                xgroups = pd.qcut(xgroups, nbins + 1, duplicates="drop", labels=False)

            pick = [column for column in df_color.columns if column != xname]
            grouped = df_color[pick].groupby(xgroups, dropna=False, observed=True)
            corr = grouped.corrwith(df_shap[xname], method=_safe_cor)
            out.loc[xname, pick] = np.average(
                corr.abs(), weights=grouped.count(), axis=0
            )

        return out
