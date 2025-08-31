import numpy as np
import pandas as pd
import pytest
from scipy.special import binom

from lightshap.explainers._utils import generate_all_masks
from lightshap.explainers.permutation_utils import (
    calculate_shapley_weights,
    one_permshap,
    positions_for_exact,
    precalculate_permshap,
)


class TestCalculateShapleyWeights:
    """Test Shapley weight calculation."""

    def test_shapley_weights_single_value(self):
        """Test Shapley weights for single values."""
        # For p=3, ell=1: weight = 1 / (binom(3,1) * (3-1)) = 1 / (3 * 2) = 1/6
        result = calculate_shapley_weights(p=3, ell=1)
        expected = 1.0 / (binom(3, 1) * (3 - 1))
        assert np.isclose(result, expected)

    def test_shapley_weights_array(self):
        """Test Shapley weights for array input."""
        p = 4
        ell = np.array([1, 2, 3])
        result = calculate_shapley_weights(p, ell)
        expected = np.array(
            [
                1.0 / (binom(4, 1) * (4 - 1)),
                1.0 / (binom(4, 2) * (4 - 2)),
                1.0 / (binom(4, 3) * (4 - 3)),
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)


class TestPositionsForExact:
    """Test position calculation for exact permutation SHAP."""

    def test_positions_simple_case(self):
        """Test positions for a simple 3-feature case."""
        mask = generate_all_masks(3)
        positions = positions_for_exact(mask)

        assert len(positions) == 3

        # Check first feature positions
        on_indices, off_indices = positions[0]
        assert len(on_indices) == len(off_indices)

        # Verify on/off relationship
        for on_idx, off_idx in zip(on_indices, off_indices, strict=False):
            assert mask[on_idx, 0]
            assert not mask[off_idx, 0]
            # Other features should be the same
            np.testing.assert_array_equal(mask[on_idx, 1:], mask[off_idx, 1:])

    def test_positions_binary_relationship(self):
        """Test that positions maintain correct binary relationships."""
        p = 4
        mask = generate_all_masks(p)
        positions = positions_for_exact(mask)

        for j in range(p):
            on_indices, off_indices = positions[j]
            power_of_two = 2 ** (p - 1 - j)

            for on_idx, off_idx in zip(on_indices, off_indices, strict=False):
                # The difference should be exactly 2^(p-1-j)
                assert on_idx - off_idx == power_of_two


class TestPrecalculatePermshap:
    """Test precalculation for permutation SHAP."""

    def test_exact_precalculation(self):
        """Test precalculation for exact method."""
        n = 10
        p = 3
        rng = np.random.default_rng(seed=0)
        X = pd.DataFrame(rng.standard_normal(size=(n, p)), columns=["A", "B", "C"])
        precalc = precalculate_permshap(p=p, bg_X=X, how="exact")

        # Check required keys
        required_keys = [
            "masks_exact_rep",
            "bg_exact_rep",
            "shapley_weights",
            "positions",
            "bg_n",
        ]
        for key in required_keys:
            assert key in precalc

        # Check dimensions
        assert precalc["masks_exact_rep"].shape == ((2**p - 2) * n, p)
        assert precalc["bg_exact_rep"].shape == ((2**p - 2) * n, p)
        assert len(precalc["shapley_weights"]) == 2**p - 1
        assert len(precalc["positions"]) == p
        assert precalc["bg_n"] == n

    def test_sampling_precalculation(self):
        """Test precalculation for sampling method."""
        n = 10
        p = 4
        rng = np.random.default_rng(seed=0)
        X = pd.DataFrame(rng.standard_normal(size=(n, p)), columns=["A", "B", "C", "D"])
        precalc = precalculate_permshap(p=p, bg_X=X, how="sampling")

        # Check required keys
        required_keys = [
            "masks_balanced_rep",
            "bg_balanced_rep",
            "bg_sampling_rep",
            "bg_n",
        ]
        for key in required_keys:
            assert key in precalc

        # Check dimensions
        assert precalc["masks_balanced_rep"].shape == (8 * n, p)  # 2*p * bg_n
        assert precalc["bg_balanced_rep"].shape == (8 * n, p)
        assert precalc["bg_sampling_rep"].shape == (2 * n, p)  # 2*(p-3) * bg_n

    def test_sampling_error_small_p(self):
        """Test that sampling raises error for p < 4."""
        rng = np.random.default_rng(seed=0)
        X = pd.DataFrame(rng.standard_normal(size=(10, 3)), columns=["A", "B", "C"])

        with pytest.raises(
            ValueError, match="sampling method not implemented for p < 4"
        ):
            precalculate_permshap(p=3, bg_X=X, how="sampling")

    def test_precalculation_with_numpy(self):
        """Test precalculation works with numpy arrays."""
        rng = np.random.default_rng(seed=0)
        X = rng.standard_normal(size=(10, 3))
        precalc = precalculate_permshap(p=3, bg_X=X, how="exact")

        assert "masks_exact_rep" in precalc
        assert isinstance(precalc["masks_exact_rep"], np.ndarray)


class TestOnePermshap:
    """Test single row explanation."""

    def test_exact_permshap_single_row(self):
        """Test exact permutation SHAP for a single row."""
        # Set up test data
        rng = np.random.default_rng(seed=0)
        X = pd.DataFrame(rng.standard_normal(size=(20, 3)), columns=["A", "B", "C"])
        bg_X = X.iloc[:10]
        bg_w = None

        # Simple linear model
        weights = np.array([1.0, 2.0, -1.0])

        def predict_fn(X):
            return (X.values @ weights).reshape(-1, 1)

        v0 = predict_fn(bg_X).mean(keepdims=True)
        v1 = predict_fn(X)

        precalc = precalculate_permshap(p=3, bg_X=bg_X, how="exact")

        shap_values, se, converged, n_iter = one_permshap(
            i=0,
            predict=predict_fn,
            how="exact",
            bg_w=bg_w,
            v0=v0,
            max_iter=1,
            tol=0.01,
            random_state=0,
            X=X,
            v1=v1,
            precalc=precalc,
            collapse=np.array([False]),
            bg_n=10,
        )

        # Check output shapes
        assert shap_values.shape == (3, 1)
        assert se.shape == (3, 1)
        assert converged
        assert n_iter == 1

        # Check efficiency property: sum of SHAP values = prediction - baseline
        prediction_diff = v1[0] - v0[0]
        shap_sum = shap_values.sum(axis=0)
        np.testing.assert_array_almost_equal(shap_sum, prediction_diff)

    def test_sampling_permshap_single_row(self):
        """Test sampling permutation SHAP for a single row."""
        X_large = pd.DataFrame(np.random.randn(20, 4), columns=["A", "B", "C", "D"])
        bg_X_large = X_large.iloc[:10]

        # Linear model
        weights_large = np.array([1.0, 2.0, -1.0, 0.5])

        def predict_fn_large(X):
            return (X.values @ weights_large).reshape(-1, 1)

        v0_large = predict_fn_large(bg_X_large).mean(keepdims=True)
        v1_large = predict_fn_large(X_large)

        precalc = precalculate_permshap(p=4, bg_X=bg_X_large, how="sampling")

        shap_values, se, converged, n_iter = one_permshap(
            i=0,
            predict=predict_fn_large,
            how="sampling",
            bg_w=None,
            v0=v0_large,
            max_iter=10,
            tol=0.01,
            random_state=0,
            X=X_large,
            v1=v1_large,
            precalc=precalc,
            collapse=np.array([False]),
            bg_n=10,
        )

        # Check output shapes
        assert shap_values.shape == (4, 1)
        assert se.shape == (4, 1)
        assert isinstance(converged, bool)
        assert n_iter == 2  # the first two iterations return identical values

        # Check approximate efficiency (sampling might not be perfect)
        prediction_diff = v1_large[0] - v0_large[0]
        shap_sum = shap_values.sum(axis=0)
        np.testing.assert_array_almost_equal(shap_sum, prediction_diff, decimal=1)

    def test_output_shapes_multioutput(self):
        """Test output shapes for multi-output model."""
        # Set up test data for this specific test
        rng = np.random.default_rng(seed=0)
        X = pd.DataFrame(rng.standard_normal(size=(20, 3)), columns=["A", "B", "C"])
        bg_X = X.iloc[:10]
        weights = np.array([1.0, 2.0, -1.0])

        def predict_multioutput(X):
            X = X.values
            # Two outputs: linear combination and squared sum
            out1 = (X @ weights).reshape(-1, 1)
            out2 = (X**2).sum(axis=1, keepdims=True)
            return np.hstack([out1, out2])

        v0_multi = predict_multioutput(bg_X).mean(axis=0, keepdims=True)
        v1_multi = predict_multioutput(X)

        precalc = precalculate_permshap(p=3, bg_X=bg_X, how="exact")

        shap_values, se, _, _ = one_permshap(
            i=0,
            predict=predict_multioutput,
            how="exact",
            bg_w=None,
            v0=v0_multi,
            max_iter=10,
            tol=0.01,
            random_state=0,
            X=X,
            v1=v1_multi,
            precalc=precalc,
            collapse=np.array([False]),
            bg_n=10,
        )

        # Check output shapes for 2 outputs
        assert shap_values.shape == (3, 2)
        assert se.shape == (3, 2)
