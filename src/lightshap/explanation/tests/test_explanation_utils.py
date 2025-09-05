import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lightshap.explanation._utils import (
    _check_features,
    _safe_cor,
    beeswarm_jitter,
    color_axis_info,
    get_text_bbox,
    halton,
    halton_sequence,
    min_max_scale,
    plot_layout,
    safe_to_float,
)


class TestSafeToFloat:
    """Test the safe_to_float utility function."""

    def test_numeric_series(self):
        """Test with numeric pandas Series."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = safe_to_float(s)
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(result, expected)

    def test_integer_series(self):
        """Test with integer pandas Series."""
        s = pd.Series([1, 2, 3])
        result = safe_to_float(s)
        expected = pd.Series([1.0, 2.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_categorical_series(self):
        """Test with categorical pandas Series."""
        s = pd.Series(["a", "b", "c", "a"], dtype="category")
        result = safe_to_float(s)
        expected = pd.Series([0.0, 1.0, 2.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_categorical_with_nan(self):
        """Test categorical series with missing values."""
        s = pd.Series(["a", "b", None, "a"], dtype="category")
        result = safe_to_float(s)
        expected = pd.Series([0.0, 1.0, np.nan, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_float_with_nan(self):
        """Test float series with missing values."""
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        result = safe_to_float(s)
        pd.testing.assert_series_equal(result, s)

    def test_unsupported_dtype_error(self):
        """Test error with unsupported data type."""
        s = pd.Series(["a", "b", "c"])  # String without category
        with pytest.raises(TypeError, match="Unsupported dtype"):
            safe_to_float(s)


class TestMinMaxScale:
    """Test the min_max_scale utility function."""

    def test_basic_scaling(self):
        """Test basic min-max scaling."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = min_max_scale(s)
        expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_constant_values(self):
        """Test scaling with constant values."""
        s = pd.Series([5, 5, 5, 5])
        result = min_max_scale(s)
        expected = pd.Series([0.5, 0.5, 0.5, 0.5])
        pd.testing.assert_series_equal(result, expected)

    def test_with_missing_values(self):
        """Test scaling with missing values."""
        s = pd.Series([1, np.nan, 3, 4, 5])
        result = min_max_scale(s)
        expected = pd.Series([0.0, np.nan, 0.5, 0.75, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_all_missing_values(self):
        """Test scaling when all values are missing."""
        s = pd.Series([np.nan, np.nan, np.nan])
        result = min_max_scale(s)
        pd.testing.assert_series_equal(result, s)


class TestHaltonSequence:
    """Test Halton sequence generation."""

    def test_halton_base_2(self):
        """Test Halton sequence with base 2."""
        result = halton(1, base=2)
        assert result == 0.5

        result = halton(2, base=2)
        assert result == 0.25

        result = halton(3, base=2)
        assert result == 0.75

    def test_halton_sequence_generation(self):
        """Test generation of multiple Halton sequence values."""
        result = halton_sequence(4, base=2)
        expected = np.array([0.5, 0.25, 0.75, 0.125])
        np.testing.assert_array_almost_equal(result, expected)

    def test_halton_sequence_length(self):
        """Test that halton_sequence returns correct length."""
        n = 10
        result = halton_sequence(n)
        assert len(result) == n


class TestBeeswarmJitter:
    """Test beeswarm jitter calculation."""

    def test_single_value(self):
        """Test jitter with single value."""
        values = np.array([1.0])
        result = beeswarm_jitter(values)
        expected = np.array([0.0])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_values(self):
        """Test jitter with multiple values."""
        values = np.array([1, 2, 3, 4, 5])
        result = beeswarm_jitter(values)
        assert len(result) == len(values)
        assert result.dtype == float

    def test_identical_values(self):
        """Test jitter with identical values."""
        values = np.array([2, 2, 2, 2])
        result = beeswarm_jitter(values)
        assert len(result) == len(values)
        # Should not raise an error even with identical values

    def test_jitter_range(self):
        """Test that jitter values are reasonable."""
        values = np.random.randn(50)
        result = beeswarm_jitter(values)
        # Jitter should generally be within reasonable bounds
        assert np.abs(result).max() <= 1.0


class TestPlotLayout:
    """Test plot layout determination."""

    def test_small_numbers(self):
        """Test layout for small numbers of plots."""
        assert plot_layout(1) == (1, 1)
        assert plot_layout(2) == (1, 2)
        assert plot_layout(3) == (1, 3)

    def test_medium_numbers(self):
        """Test layout for medium numbers of plots."""
        assert plot_layout(4) == (2, 2)
        assert plot_layout(5) == (3, 2)
        assert plot_layout(6) == (3, 2)

    def test_larger_numbers(self):
        """Test layout for larger numbers of plots."""
        assert plot_layout(7) == (3, 3)
        assert plot_layout(9) == (3, 3)
        assert plot_layout(12) == (4, 3)

    def test_very_large_numbers(self):
        """Test layout for very large numbers of plots."""
        assert plot_layout(13) == (4, 4)
        assert plot_layout(16) == (4, 4)
        assert plot_layout(20) == (5, 4)


class TestCheckFeatures:
    """Test feature checking and validation."""

    def test_none_features(self):
        """Test with None features (should return all)."""
        all_features = ["a", "b", "c"]
        result = _check_features(None, all_features)
        assert result == all_features

    def test_valid_features(self):
        """Test with valid feature subset."""
        features = ["a", "c"]
        all_features = ["a", "b", "c"]
        result = _check_features(features, all_features)
        assert result == features

    def test_invalid_features_error(self):
        """Test error with invalid features."""
        features = ["a", "d"]  # "d" not in all_features
        all_features = ["a", "b", "c"]
        with pytest.raises(ValueError, match="Some .* are not present"):
            _check_features(features, all_features)

    def test_non_iterable_error(self):
        """Test error with non-iterable features."""
        features = "a"  # String is iterable but should be treated as single item
        all_features = ["a", "b", "c"]
        with pytest.raises(TypeError, match="must be an iterable"):
            _check_features(features, all_features)


class TestSafeCor:
    """Test safe correlation calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = _safe_cor(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_no_correlation(self):
        """Test no correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 1, 1, 1, 1])  # Constant
        result = _safe_cor(x, y)
        assert result == 0.0

    def test_with_nan_values(self):
        """Test correlation with NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = _safe_cor(x, y)
        # Should compute correlation on valid pairs only
        assert abs(result - 1.0) < 1e-10

    def test_insufficient_data(self):
        """Test correlation with insufficient valid data."""
        x = np.array([1, np.nan])
        y = np.array([2, np.nan])
        result = _safe_cor(x, y)
        assert result == 0.0

    def test_zero_variance(self):
        """Test correlation when one variable has zero variance."""
        x = np.array([1, 1, 1, 1])
        y = np.array([1, 2, 3, 4])
        result = _safe_cor(x, y)
        assert result == 0.0


class TestGetTextBbox:
    """Test text bounding box calculation."""

    def test_text_bbox_basic(self):
        """Test basic text bounding box calculation."""
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Test text")
        ax.text(0.2, 0.8, "Another text")

        bbox = get_text_bbox(ax)
        assert len(bbox) == 4  # (left, right, bottom, top)
        assert all(isinstance(coord, int | float) for coord in bbox)

        plt.close(fig)


class TestColorAxisInfo:
    """Test color axis information preparation."""

    def test_categorical_color_feature(self):
        """Test with categorical color feature."""
        z = pd.Series(["A", "B", "C", "A"], dtype="category")
        result = color_axis_info(
            z, "viridis", max_color_labels=10, max_color_label_length=20
        )

        assert result["categorical"] is True
        assert "mapping" in result
        assert "cmap" in result
        assert "norm" in result
        assert "values" in result
        assert result["n_colors"] == 3

    def test_numeric_color_feature(self):
        """Test with numeric color feature."""
        z = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = color_axis_info(
            z, "viridis", max_color_labels=10, max_color_label_length=20
        )

        assert result["categorical"] is False
        assert "cmap" in result
        assert "values" in result
        pd.testing.assert_series_equal(result["values"], z)

    def test_categorical_label_truncation(self):
        """Test label truncation for long categorical labels."""
        categories = ["very_long_category_name", "short"]
        z = pd.Series(categories, dtype="category")
        result = color_axis_info(
            z, "viridis", max_color_labels=10, max_color_label_length=5
        )

        # Check that long labels are truncated
        for value in result["mapping"].values():
            assert len(value) <= 5

    def test_too_many_categorical_labels(self):
        """Test behavior with too many categorical labels."""
        categories = [f"cat_{i}" for i in range(20)]
        z = pd.Series(categories, dtype="category")
        result = color_axis_info(
            z, "viridis", max_color_labels=5, max_color_label_length=20
        )

        # Should reduce number of labels shown
        non_empty_labels = sum(1 for v in result["mapping"].values() if v != "")
        assert non_empty_labels <= 7  # Some labels should be empty

    def test_categorical_with_missing(self):
        """Test categorical feature with missing values."""
        z = pd.Series(["A", "B", None, "A"], dtype="category")
        result = color_axis_info(
            z, "viridis", max_color_labels=10, max_color_label_length=20
        )

        assert result["categorical"] is True
        # Missing values should be handled as NaN in the values
        assert pd.isna(result["values"]).any()

    def test_colormap_bad_value_setting(self):
        """Test that colormap handles bad values (NaN) correctly."""
        z = pd.Series([1.0, 2.0, np.nan, 4.0])
        result = color_axis_info(
            z, "viridis", max_color_labels=10, max_color_label_length=20, alpha=0.5
        )

        # The colormap should have bad color set
        assert hasattr(result["cmap"], "_rgba_bad")
