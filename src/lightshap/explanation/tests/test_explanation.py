import numpy as np
import pandas as pd
import polars as pl
import pytest

from lightshap.explanation.explanation import Explanation


def create_test_data(n=10, n_outputs=1):
    """Helper function to create test data."""
    rng = np.random.default_rng(0)

    # Create feature data
    X = pd.DataFrame(
        {
            "numeric_feature": rng.random(n),
            "categorical_feature": rng.choice(["A", "B", "C"], n),
            "feature_3": rng.integers(0, 10, n),
        }
    )

    if n_outputs == 1:
        shap_values = rng.random((n, 3)) - 0.5
        baseline = 0.5
    else:
        shap_values = rng.random((n, 3, n_outputs)) - 0.5
        baseline = rng.random(n_outputs)

    return X, shap_values, baseline


class TestExplanationInit:
    """Test suite for Explanation initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with minimal parameters."""
        X, shap_values, baseline = create_test_data()
        explanation = Explanation(shap_values, X, baseline)

        assert explanation.shape == shap_values.shape
        assert explanation.ndim == 2
        assert len(explanation.feature_names) == X.shape[1]
        assert explanation.baseline.shape == (1,)

    def test_multi_output_initialization(self):
        """Test initialization with multi-output data."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        assert explanation.shape == shap_values.shape
        assert explanation.ndim == 3
        assert explanation.baseline.shape == (3,)
        assert explanation.output_names == [0, 1, 2]

    def test_custom_feature_names(self):
        """Test initialization with custom feature names."""
        X, shap_values, baseline = create_test_data()
        feature_names = ["feat1", "feat2", "feat3"]
        explanation = Explanation(shap_values, X, baseline, feature_names=feature_names)

        assert explanation.feature_names == feature_names
        assert list(explanation.X.columns) == feature_names

    def test_custom_output_names(self):
        """Test initialization with custom output names."""
        X, shap_values, baseline = create_test_data(n_outputs=2)
        output_names = ["class_A", "class_B"]
        explanation = Explanation(shap_values, X, baseline, output_names=output_names)

        assert explanation.output_names == output_names

    def test_with_standard_errors(self):
        """Test initialization with standard errors."""
        X, shap_values, baseline = create_test_data()
        standard_errors = np.abs(shap_values) * 0.1
        explanation = Explanation(
            shap_values, X, baseline, standard_errors=standard_errors
        )

        assert explanation.standard_errors is not None
        assert explanation.standard_errors.shape == shap_values.shape

    def test_with_convergence_info(self):
        """Test initialization with convergence information."""
        X, shap_values, baseline = create_test_data()
        converged = np.array(
            [True, False, True, True, False, True, True, True, False, True]
        )
        n_iter = np.array([10, 100, 15, 8, 100, 12, 9, 11, 100, 13])

        explanation = Explanation(
            shap_values, X, baseline, converged=converged, n_iter=n_iter
        )

        assert explanation.converged is not None
        assert explanation.n_iter is not None
        assert len(explanation.converged) == len(shap_values)
        assert len(explanation.n_iter) == len(shap_values)


class TestExplanationValidation:
    """Test suite for Explanation validation and error handling."""

    def test_invalid_shap_values_type(self):
        """Test that non-numpy array SHAP values raise TypeError."""
        X, _, baseline = create_test_data()
        with pytest.raises(TypeError, match="SHAP values must be a numpy array"):
            Explanation([[1, 2, 3]], X, baseline)

    def test_empty_shap_values(self):
        """Test that empty SHAP values raise TypeError."""
        X, _, baseline = create_test_data()
        with pytest.raises(TypeError, match="SHAP values must be a numpy array"):
            Explanation(np.array([]), X, baseline)

    def test_wrong_shap_values_dimensions(self):
        """Test that invalid SHAP values dimensions raise ValueError."""
        X, _, baseline = create_test_data()
        with pytest.raises(ValueError, match="SHAP values must be 2D or 3D"):
            Explanation(np.array([1, 2, 3]), X, baseline)

    def test_mismatched_standard_errors_shape(self):
        """Test that mismatched standard errors shape raises ValueError."""
        X, shap_values, baseline = create_test_data()
        wrong_se = np.random.random((5, 2))  # Wrong shape

        with pytest.raises(
            ValueError, match="Shape .* of standard_errors does not match"
        ):
            Explanation(shap_values, X, baseline, standard_errors=wrong_se)

    def test_mismatched_converged_length(self):
        """Test that mismatched converged length raises ValueError."""
        X, shap_values, baseline = create_test_data()
        wrong_converged = np.array([True, False])  # Wrong length

        with pytest.raises(ValueError, match="Length .* of converged does not match"):
            Explanation(shap_values, X, baseline, converged=wrong_converged)

    def test_mismatched_n_iter_length(self):
        """Test that mismatched n_iter length raises ValueError."""
        X, shap_values, baseline = create_test_data()
        wrong_n_iter = np.array([10, 20])  # Wrong length

        with pytest.raises(ValueError, match="Length .* of n_iter does not match"):
            Explanation(shap_values, X, baseline, n_iter=wrong_n_iter)

    def test_mismatched_baseline_length(self):
        """Test that mismatched baseline length raises ValueError."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        wrong_baseline = np.array([0.5, 0.6])  # Wrong length for 3 outputs

        with pytest.raises(ValueError, match="Length .* of baseline does not match"):
            Explanation(shap_values, X, wrong_baseline)

    def test_mismatched_X_shape(self):
        """Test that mismatched X shape raises ValueError."""
        X, shap_values, baseline = create_test_data()
        wrong_X = X.iloc[:5, :]  # Wrong number of rows

        with pytest.raises(ValueError, match="Shape .* of X does not match"):
            Explanation(shap_values, wrong_X, baseline)

    def test_mismatched_feature_names_length(self):
        """Test that mismatched feature names length raises ValueError."""
        X, shap_values, baseline = create_test_data()
        wrong_feature_names = ["feat1", "feat2"]  # Missing one name

        with pytest.raises(
            ValueError, match="Length .* of feature_names does not match"
        ):
            Explanation(shap_values, X, baseline, feature_names=wrong_feature_names)

    def test_mismatched_output_names_length(self):
        """Test that mismatched output names length raises ValueError."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        wrong_output_names = ["out1", "out2"]  # Missing one name

        with pytest.raises(
            ValueError, match="Length .* of output_names does not match"
        ):
            Explanation(shap_values, X, baseline, output_names=wrong_output_names)


class TestExplanationMethods:
    """Test suite for Explanation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X, self.shap_values, self.baseline = create_test_data(n=20)
        self.explanation = Explanation(self.shap_values, self.X, self.baseline)

    def test_len(self):
        """Test __len__ method."""
        assert len(self.explanation) == 20

    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(self.explanation)
        assert "SHAP Explanation" in repr_str
        assert "SHAP values" in repr_str
        assert "X, first" in repr_str

    def test_filter_with_indices(self):
        """Test filter method with integer indices."""
        indices = [0, 2, 4, 6]
        filtered = self.explanation.filter(indices)

        assert len(filtered) == 4
        assert filtered.shape[0] == 4
        assert np.array_equal(filtered.shap_values, self.shap_values[indices])

    def test_filter_with_boolean_mask(self):
        """Test filter method with boolean mask."""
        rng = np.random.default_rng(0)
        mask = rng.choice([True, False], size=20)
        filtered = self.explanation.filter(mask)

        assert len(filtered) == mask.sum()
        assert filtered.shape[0] == mask.sum()

    def test_filter_invalid_indices(self):
        """Test filter method with invalid indices."""
        with pytest.raises(
            TypeError, match="indices must be an integer or boolean array-like"
        ):
            self.explanation.filter([1.5, 2.5, 3.5])

    def test_select_output_single_output(self):
        """Test select_output with single-output explanation."""
        selected = self.explanation.select_output(0)
        assert selected is self.explanation  # Should return same object

    def test_select_output_multi_output_by_index(self):
        """Test select_output with multi-output explanation using index."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        selected = explanation.select_output(1)
        assert selected.ndim == 2
        assert selected.shape == (len(X), len(X.columns))
        assert selected.baseline.shape == (1,)

    def test_select_output_multi_output_by_name(self):
        """Test select_output with multi-output explanation using name."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        output_names = ["class_A", "class_B", "class_C"]
        explanation = Explanation(shap_values, X, baseline, output_names=output_names)

        selected = explanation.select_output("class_B")
        assert selected.ndim == 2
        assert selected.shape == (len(X), len(X.columns))

    def test_select_output_invalid_type(self):
        """Test select_output with invalid index type."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        with pytest.raises(TypeError, match="index must be an integer or string"):
            explanation.select_output(1.5)

    def test_importance_single_output(self):
        """Test importance calculation for single-output model."""
        importance = self.explanation.importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) == self.X.shape[1]
        assert importance.index.tolist() == self.explanation.feature_names
        # Descending?
        assert all(
            importance.iloc[i] >= importance.iloc[i + 1]
            for i in range(len(importance) - 1)
        )

    def test_importance_multi_output(self):
        """Test importance calculation for multi-output model."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        importance = explanation.importance()

        assert isinstance(importance, pd.DataFrame)
        assert importance.shape == (X.shape[1], 3)

    def test_importance_specific_output(self):
        """Test importance calculation for specific output."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        importance = explanation.importance(which_output=1)

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]

    def test_interaction_heuristic_default(self):
        """Test interaction heuristic with default parameters."""
        heuristic = self.explanation.interaction_heuristic()

        assert isinstance(heuristic, pd.DataFrame)
        assert heuristic.shape == (
            len(self.explanation.feature_names),
            len(self.explanation.feature_names),
        )
        assert heuristic.index.tolist() == self.explanation.feature_names
        assert heuristic.columns.tolist() == self.explanation.feature_names

    def test_interaction_heuristic_subset_features(self):
        """Test interaction heuristic with subset of features."""
        features = ["numeric_feature", "feature_3"]
        color_features = ["categorical_feature"]

        heuristic = self.explanation.interaction_heuristic(features, color_features)

        assert heuristic.shape == (len(features), len(color_features))
        assert heuristic.index.tolist() == features
        assert heuristic.columns.tolist() == color_features

    def test_set_feature_names(self):
        """Test setting new feature names."""
        new_names = ["new_feat1", "new_feat2", "new_feat3"]
        self.explanation.set_feature_names(new_names)

        assert self.explanation.feature_names == new_names
        assert list(self.explanation.X.columns) == new_names

    def test_set_feature_names_wrong_length(self):
        """Test setting feature names with wrong length."""
        wrong_names = ["feat1", "feat2"]  # Too few names

        with pytest.raises(
            ValueError, match="Length .* of feature_names does not match"
        ):
            self.explanation.set_feature_names(wrong_names)

    def test_set_output_names_multi_output(self):
        """Test setting output names for multi-output model."""
        X, shap_values, baseline = create_test_data(n_outputs=3)
        explanation = Explanation(shap_values, X, baseline)

        new_names = ["output_A", "output_B", "output_C"]
        explanation.set_output_names(new_names)

        assert explanation.output_names == new_names

    def test_set_output_names_single_output(self):
        """Test setting output names for single-output model."""
        self.explanation.set_output_names(["single_output"])
        assert (
            self.explanation.output_names is None
        )  # Should remain None for single output

    def test_set_X_numpy_array(self):
        """Test setting X with numpy array."""
        col_names = ["numeric_feature", "categorical_feature", "feature_3"]
        rng = np.random.default_rng(0)
        new_X = rng.random((20, 3))
        self.explanation.set_X(new_X)

        assert isinstance(self.explanation.X, pd.DataFrame)
        assert self.explanation.X.shape == (20, 3)
        assert self.explanation.X.columns.tolist() == col_names  # names are retained

    def test_set_X_polars(self):
        """Test setting X with polars DataFrame."""
        rng = np.random.default_rng(0)
        new_X = pl.DataFrame(
            {
                "numeric_feature": rng.random(20),
                "categorical_feature": rng.choice(["A", "B", "C"], 20),
                "feature_3": rng.integers(0, 10, 20),
            }
        )
        self.explanation.set_X(new_X)

        assert isinstance(self.explanation.X, pd.DataFrame)
        assert self.explanation.X.shape == (20, 3)
        assert isinstance(
            self.explanation.X.categorical_feature.dtype, pd.CategoricalDtype
        )

    def test_set_X_wrong_shape(self):
        """Test setting X with wrong shape."""
        wrong_X = np.random.random((15, 3))  # Wrong shape

        with pytest.raises(ValueError, match="Shape .* of X does not match"):
            self.explanation.set_X(wrong_X)

    def test_plot_accessor(self):
        """Test that plot accessor returns ExplanationPlotter."""
        from lightshap.explanation.explanationplotter import ExplanationPlotter

        assert isinstance(self.explanation.plot, ExplanationPlotter)


class TestExplanationEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_single_observation(self):
        """Test with single observation."""
        X, shap_values, baseline = create_test_data(n=1)
        explanation = Explanation(shap_values, X, baseline)

        assert len(explanation) == 1
        assert explanation.shape[0] == 1

    def test_squeeze_unnecessary_dimension(self):
        """Test that unnecessary third dimension is squeezed."""
        X, shap_values, baseline = create_test_data()
        # Add unnecessary third dimension
        shap_values_3d = shap_values.reshape(10, 3, 1)

        explanation = Explanation(shap_values_3d, X, baseline)

        assert explanation.ndim == 2  # Should be squeezed to 2D
        assert explanation.shape == (10, 3)

    def test_baseline_conversion(self):
        """Test that baseline is properly converted to numpy array."""
        X, shap_values, _ = create_test_data()

        # Test with scalar
        explanation1 = Explanation(shap_values, X, 0.5)
        assert isinstance(explanation1.baseline, np.ndarray)
        assert explanation1.baseline.shape == (1,)

        # Test with list
        explanation2 = Explanation(shap_values, X, [0.5])
        assert isinstance(explanation2.baseline, np.ndarray)
        assert explanation2.baseline.shape == (1,)

    def test_categorical_conversion(self):
        """Test that string/object columns are converted to categorical."""
        X, shap_values, baseline = create_test_data()

        rng = np.random.default_rng(0)
        X = X.assign(feature_3=lambda x: rng.choice(["x", "y", "z"], size=x.shape[0]))

        explanation = Explanation(shap_values, X, baseline)

        assert isinstance(explanation.X.iloc[:, 2].dtype, pd.CategoricalDtype)

    def test_unsupported_column_type(self):
        """Test that unsupported column types raise TypeError."""
        X, shap_values, baseline = create_test_data()

        # Add unsupported column type
        X = X.assign(
            feature_3=lambda x: pd.date_range("2020-01-01", periods=x.shape[0])
        )

        with pytest.raises(TypeError, match="Column .* has unsupported dtype"):
            Explanation(shap_values, X, baseline)
