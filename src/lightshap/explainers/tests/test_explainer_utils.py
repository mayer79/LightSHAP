import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest

from lightshap.explainers._utils import (
    check_convergence,
    check_or_derive_background_data,
    collapse_potential,
    collapse_with_index,
    generate_all_masks,
    generate_partly_exact_masks,
    generate_permutation_masks,
    masked_predict,
    random_permutation_from_start,
    repeat_masks,
    replicate_data,
    safe_predict,
    welford_iteration,
)


class TestReplicateData:
    """Test data replication function."""

    def test_replicate_numpy(self):
        """Test replicating numpy array."""
        X = np.array([[1, 2], [3, 4]])
        result = replicate_data(X, m=2)
        expected = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_replicate_pandas(self):
        """Test replicating pandas DataFrame."""
        X = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = replicate_data(X, m=3)
        expected = pd.DataFrame({"A": [1, 2, 1, 2, 1, 2], "B": [3, 4, 3, 4, 3, 4]})
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_replicate_polars(self):
        """Test replicating polars DataFrame."""
        X = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = replicate_data(X, m=3)
        expected = pl.DataFrame({"A": [1, 2, 1, 2, 1, 2], "B": [3, 4, 3, 4, 3, 4]})
        pl_testing.assert_frame_equal(result, expected)

    def test_replicate_error_nonpositive_m(self):
        """Test error for non-positive replication factor."""
        X = np.array([[1, 2]])
        with pytest.raises(ValueError, match="Replication factor m must be positive"):
            replicate_data(X, m=0)


class TestRepeatMasks:
    """Test mask repetition function."""

    def test_repeat_masks_numpy(self):
        """Test repeating masks as numpy array."""
        Z = np.array([[True, False], [False, True]], dtype=bool)
        result = repeat_masks(Z, m=2)
        expected = np.array(
            [[True, False], [True, False], [False, True], [False, True]], dtype=bool
        )
        np.testing.assert_array_equal(result, expected)

    def test_repeat_masks_polars(self):
        """Test repeating masks with polars schema."""
        Z = np.array([[True, False], [False, True]])
        result = repeat_masks(Z, m=2, pl_schema=["x", "z"])
        expected = pl.DataFrame(
            {
                "x": [True, True, False, False],
                "z": [False, False, True, True],
            }
        )
        pl_testing.assert_frame_equal(result, expected)


class TestWelfordIteration:
    """Test Welford's method for incremental statistics."""

    def test_welford_single_iteration(self):
        """Test single Welford iteration."""
        new_value = 5.0
        avg = 3.0
        sum_squared = 2.0
        j = 2

        new_avg, new_sum_squared = welford_iteration(new_value, avg, sum_squared, j)

        # Manual calculation: delta = 5 - 3 = 2, new_avg = 3 + 2/2 = 4
        # new_sum_squared = 2 + 2 * (5 - 4) = 4
        assert np.isclose(new_avg, 4.0)
        assert np.isclose(new_sum_squared, 4.0)

    def test_welford_equals_sample_stats(self):
        """Test that Welford equals sample statistics."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        avg = 0.0
        sum_squared = 0.0

        for j, value in enumerate(data, 1):
            avg, sum_squared = welford_iteration(value, avg, sum_squared, j)

        # Should equal sample mean
        assert np.isclose(avg, data.mean())

        # sum_squared / (n-1) should equal sample variance
        sample_var = sum_squared / (len(data) - 1)
        expected_var = data.var(ddof=1)
        assert np.isclose(sample_var, expected_var)


class TestCheckConvergence:
    """Test convergence checking function."""

    def test_convergence_achieved(self):
        """Test when convergence is achieved."""
        beta_n = np.array([[1.0, 2.0], [3.0, 4.0]])
        sum_squared = np.array([[0.001, 0.001], [0.001, 0.001]])
        n_iter = 10
        tol = 0.1

        result = check_convergence(beta_n, sum_squared, n_iter, tol)
        assert result

    def test_convergence_not_achieved(self):
        """Test when convergence is not achieved."""
        beta_n = np.array([[1.0, 2.0], [3.0, 4.0]])
        sum_squared = np.array([[10.0, 10.0], [10.0, 10.0]])  # Large errors
        n_iter = 10
        tol = 0.01

        result = check_convergence(beta_n, sum_squared, n_iter, tol)
        assert not result


class TestGenerateAllMasks:
    """Test generation of all possible masks."""

    def test_generate_all_masks_small(self):
        """Test generating all masks for small p."""
        result = generate_all_masks(p=2)
        expected = np.array(
            [[False, False], [False, True], [True, False], [True, True]], dtype=bool
        )
        np.testing.assert_array_equal(result, expected)

    def test_generate_all_masks_size(self):
        """Test that all masks has correct size."""
        p = 7
        result = generate_all_masks(7)
        assert result.shape == (2**p, p)
        assert result.dtype == bool


class TestGeneratePartlyExactMasks:
    """Test generation of partly exact masks."""

    def test_partly_exact_masks_degree_1(self):
        """Test partly exact masks for degree 1."""
        result = generate_partly_exact_masks(p=3, degree=1)

        # Should have identity matrix and its complement
        expected_identity = np.eye(3, dtype=bool)
        expected_complement = ~expected_identity
        expected = np.vstack([expected_complement, expected_identity])

        np.testing.assert_array_equal(result, expected)

    def test_partly_exact_masks_degree_2(self):
        """Test partly exact masks for degree 2."""
        p = 7
        result = generate_partly_exact_masks(p=p, degree=2)

        assert result.shape == ((p - 1) * p, p)
        assert np.isin(result.sum(axis=1), np.array([2, p - 2])).all()

    def test_partly_exact_masks_special_degree_2(self):
        """Test partly exact masks special case for degree 2."""
        p = 4
        result = generate_partly_exact_masks(p=p, degree=2)

        m = int((p - 1) * p / 2)
        assert result.shape == (m, p)
        assert result.sum(axis=1).tolist() == [2] * m

    def test_partly_exact_masks_errors(self):
        """Test error cases for partly exact masks."""
        with pytest.raises(ValueError, match="degree must be at least 1"):
            generate_partly_exact_masks(p=5, degree=0)

        with pytest.raises(ValueError, match="p must be >= 2 \\* degree"):
            generate_partly_exact_masks(p=3, degree=2)


class TestRandomPermutationFromStart:
    """Test random permutation generation."""

    def test_permutation_starts_correctly(self):
        """Test that permutation starts with specified value."""
        rng = np.random.default_rng(0)
        result = random_permutation_from_start(p=5, start=2, rng=rng)

        assert result[0] == 2
        assert len(result) == 5
        assert set(result) == set(range(5))

    def test_permutation_deterministic(self):
        """Test that permutation is deterministic with same seed."""
        rng1 = np.random.default_rng(2)
        rng2 = np.random.default_rng(2)

        result1 = random_permutation_from_start(p=4, start=1, rng=rng1)
        result2 = random_permutation_from_start(p=4, start=1, rng=rng2)

        assert result1 == result2


class TestGeneratePermutationMasks:
    """Test permutation mask generation."""

    def test_permutation_masks_degree_0(self):
        """Test permutation mask generation degree 0."""
        J = [0, 1, 2, 3]
        result = generate_permutation_masks(J, degree=0)

        # Should have 2 * (p - 1 - 2*degree) = 2 * 3 = 6 rows
        assert result.shape == (6, 4)
        assert result.dtype == bool
        assert np.isin(result.sum(axis=1), np.array([1, 2, 3])).all()

    def test_permutation_masks_degree_1(self):
        """Test permutation mask generation degree 1."""
        J = [0, 1, 2, 3, 4]
        result = generate_permutation_masks(J, degree=1)

        # Should have 2 * (p - 1 - 2*degree) = 2 * 2 = 4 rows
        assert result.shape == (4, 5)
        assert result.dtype == bool
        assert np.isin(result.sum(axis=1), np.array([2, 3])).all()

    def test_permutation_masks_degree_2(self):
        """Test permutation mask generation degree 2."""
        J = [0, 1, 2, 3, 4, 5]
        result = generate_permutation_masks(J, degree=2)

        # Should have 2 * (p - 1 - 2*degree) = 2 * 1 = 2 rows
        assert result.shape == (2, 6)
        assert result.dtype == bool
        assert result.sum(axis=1).tolist() == [3] * 2

    def test_permutation_masks_error(self):
        """Test error for insufficient elements."""
        J = [0, 1]  # Too short for degree=1
        with pytest.raises(
            ValueError, match="J must have at least 2 \\* degree \\+ 2 elements"
        ):
            generate_permutation_masks(J, degree=1)


class TestCheckOrDeriveBackgroundData:
    """Test background data checking and derivation."""

    def test_derive_background_from_X(self):
        """Test deriving background data from X."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["A", "B", "C"])

        bg_X, bg_w = check_or_derive_background_data(
            bg_X=None, bg_w=None, bg_n=10, X=X, random_state=0
        )

        assert bg_X.shape == (10, 3)
        assert bg_w is None
        assert set(bg_X.columns) == set(X.columns)

    def test_use_provided_background(self):
        """Test using provided background data."""
        X = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        bg_X_input = pd.DataFrame({"A": [7, 8], "B": [9, 10]})

        bg_X, bg_w = check_or_derive_background_data(
            bg_X=bg_X_input, bg_w=None, bg_n=5, X=X, random_state=42
        )

        pd.testing.assert_frame_equal(bg_X, bg_X_input)
        assert bg_w is None

    def test_background_weights_validation(self):
        """Test background weights validation."""
        X = pd.DataFrame({"A": np.arange(30)})
        bg_w = np.array([0.2] * 30)

        # Should work with valid weights
        bg_X, bg_w_out = check_or_derive_background_data(
            bg_X=None, bg_w=bg_w, bg_n=10, X=X, random_state=0
        )

        assert len(bg_w_out) == len(bg_X) == 10

    def test_background_data_errors(self):
        """Test various error conditions."""
        X = pd.DataFrame({"A": [1, 2]})

        # X too small without background
        with pytest.raises(ValueError, match="Background data must be provided"):
            check_or_derive_background_data(
                bg_X=None, bg_w=None, bg_n=5, X=X, random_state=42
            )

        # Wrong weight dimensions
        with pytest.raises(ValueError, match="bg_w must be a 1D array-like object"):
            check_or_derive_background_data(
                bg_X=None, bg_w=np.array([[1, 2]]), bg_n=5, X=X, random_state=42
            )


class TestSafePredict:
    """Test safe prediction wrapper."""

    def test_safe_predict_basic(self):
        """Test basic safe predict functionality."""

        def dummy_predict(X):
            return np.sum(X, axis=1)

        safe_fn = safe_predict(dummy_predict)
        X = np.array([[1, 2], [3, 4]])
        result = safe_fn(X)

        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result.flatten(), [3, 7])

    def test_safe_predict_multioutput(self):
        """Test safe predict with multi-output."""

        def dummy_predict(X):
            return np.column_stack([X.sum(axis=1), X.prod(axis=1)])

        safe_fn = safe_predict(dummy_predict)
        X = np.array([[1, 2], [3, 4]])
        result = safe_fn(X)

        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[3, 2], [7, 12]])

    def test_safe_predict_error_non_callable(self):
        """Test error for non-callable input."""
        with pytest.raises(TypeError, match="predict must be a callable"):
            safe_predict("not_a_function")


class TestCollapsePotential:
    """Test collapse potential calculation."""

    def test_collapse_potential_basic(self):
        """Test basic collapse potential calculation."""
        X = pd.DataFrame({"A": [1, 1], "B": [2, 1]})
        bg_X = pd.DataFrame({"A": [1, 1, 2], "B": [2, 3, 4]})
        bg_w = np.array([1, 1, 1])

        result = collapse_potential(X, bg_X, bg_w)

        # First row: A=1 matches 2/3 of bg, B=2 matches 1/3 of bg
        # potential = 1 - (1 - 2/3/2) * (1 - 1/3/2) = 1 - (2/3) * (5/6)
        expected = np.array([1 - (2 / 3) * (5 / 6), 1 - (2 / 3) * (1)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_collapse_potential_numpy(self):
        """Test collapse potential with numpy arrays."""
        X = np.array([[1, 2], [1, 1]])
        bg_X = np.array([[1, 2], [1, 3], [2, 4]])
        bg_w = np.array([1, 1, 1])

        result = collapse_potential(X, bg_X, bg_w)

        expected = np.array([1 - (2 / 3) * (5 / 6), 1 - (2 / 3) * (1)])
        np.testing.assert_array_almost_equal(result, expected)


class TestCollapseWithIndex:
    """Test collapse with index reconstruction."""

    def test_collapse_numpy_unique(self):
        """Test collapse with numpy arrays having unique rows."""
        x = np.array([[1, 2], [3, 4], [1, 2]])
        unique_x, ix_reconstruct = collapse_with_index(x, xclass="np")

        assert unique_x.shape[0] < x.shape[0]  # Should be collapsed
        assert ix_reconstruct is not None

        # Reconstruction should work
        reconstructed = unique_x[ix_reconstruct]
        np.testing.assert_array_equal(reconstructed, x)

    def test_collapse_pandas_no_duplicates(self):
        """Test collapse with pandas DataFrame without duplicates."""
        x = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        unique_x, ix_reconstruct = collapse_with_index(x, xclass="pd")

        pd.testing.assert_frame_equal(unique_x, x)
        assert ix_reconstruct is None

    def test_collapse_pandas_with_duplicates(self):
        """Test collapse with pandas DataFrame with duplicates."""
        x = pd.DataFrame({"A": [1, 2, 1], "B": ["A", "B", "A"]})
        unique_x, ix_reconstruct = collapse_with_index(x, xclass="pd")

        assert len(unique_x) == 2  # Should be collapsed
        assert ix_reconstruct is not None

        # Reconstruction should work
        reconstructed = unique_x.iloc[ix_reconstruct].reset_index(drop=True)
        pd.testing.assert_frame_equal(reconstructed, x)

    def test_collapse_polars_no_duplicates(self):
        """Test collapse with polars DataFrame without duplicates."""
        x = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        unique_x, ix_reconstruct = collapse_with_index(x, xclass="pl")

        pl_testing.assert_frame_equal(unique_x, x)
        assert ix_reconstruct is None

    def test_collapse_polars_with_duplicates(self):
        """Test collapse with polars DataFrame with duplicates."""
        x = pl.DataFrame({"A": [1, 2, 1], "B": ["A", "B", "A"]})
        unique_x, ix_reconstruct = collapse_with_index(x, xclass="pl")

        assert len(unique_x) == 2  # Should be collapsed
        assert ix_reconstruct is not None

        # Reconstruction should work
        reconstructed = unique_x[ix_reconstruct]
        pl_testing.assert_frame_equal(reconstructed, x)


class TestMaskedPredict:
    """Test masked prediction function."""

    def test_masked_predict_numpy(self):
        """Test masked predict with numpy arrays."""

        def predict_fn(X):
            return X.sum(axis=1)

        masks_rep = np.array(
            [[True, False], [True, False], [False, True], [False, True]],
            dtype=bool,
        )
        x = np.array([10, 20])
        bg_rep = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        weights = None

        result = masked_predict(
            predict=safe_predict(predict_fn),
            masks_rep=masks_rep,
            x=x,
            bg_rep=bg_rep,
            weights=weights,
            xclass="np",
            collapse=False,
            bg_n=2,
        )

        assert np.array_equal(result, np.array([13.0, 22.0]).reshape(-1, 1))

    def test_masked_predict_pandas(self):
        """Test masked predict with pandas DataFrames."""

        def predict_fn(X):
            return X.sum(axis=1)

        masks_rep = np.array(
            [[True, False], [True, False], [False, True], [False, True]],
            dtype=bool,
        )
        x = pd.Series([10, 20], index=["A", "B"])
        bg_rep = pd.DataFrame({"A": [1, 3, 1, 3], "B": [2, 4, 2, 4]})
        weights = None

        result = masked_predict(
            predict=safe_predict(predict_fn),
            masks_rep=masks_rep,
            x=x,
            bg_rep=bg_rep,
            weights=weights,
            xclass="pd",
            collapse=False,
            bg_n=2,
        )

        assert np.array_equal(result, np.array([13.0, 22.0]).reshape(-1, 1))

    def test_masked_predict_polars(self):
        """Test masked predict with polars DataFrames."""

        def predict_fn(X):
            return X["A"] + X["B"]

        masks_rep = np.array(
            [[True, False], [True, False], [False, True], [False, True]],
            dtype=bool,
        )
        masks_rep = pl.DataFrame(masks_rep, schema=["A", "B"])
        x = pl.DataFrame({"A": [10], "B": [20]})
        bg_rep = pl.DataFrame({"A": [1, 3, 1, 3], "B": [2, 4, 2, 4]})
        weights = None

        result = masked_predict(
            predict=safe_predict(predict_fn),
            masks_rep=masks_rep,
            x=x,
            bg_rep=bg_rep,
            weights=weights,
            xclass="pl",
            collapse=False,
            bg_n=2,
        )

        assert np.array_equal(result, np.array([13.0, 22.0]).reshape(-1, 1))
