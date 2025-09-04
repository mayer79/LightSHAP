import numpy as np
import pandas as pd
import pytest

from lightshap.explainers.kernel_utils import (
    calculate_exact_prop,
    calculate_kernel_weights,
    calculate_kernel_weights_per_coalition_size,
    kernel_solver,
    one_kernelshap,
    precalculate_kernelshap,
    prepare_input_exact,
    prepare_input_hybrid,
    prepare_input_sampling,
)


class TestCalculateKernelWeights:
    """Test Kernel SHAP weight calculation."""

    def test_kernel_weights_basic(self):
        """Test kernel weights for basic case."""
        p = 4
        weights = calculate_kernel_weights(p)

        # Test against R: kernelshap:::kernel_weights(4)
        expected = [0.4, 0.2, 0.4]
        np.testing.assert_array_almost_equal(weights, expected)

    def test_kernel_weights_per_coalition_size_degree_0(self):
        """Test kernel weights per coalition size for degree 0."""
        p = 5
        degree = 0
        weights = calculate_kernel_weights_per_coalition_size(p, degree)

        # Test against R: kernelshap:::kernel_weights_per_coalition_size(5)
        expected = [0.3, 0.2, 0.2, 0.3]
        np.testing.assert_array_almost_equal(weights, expected)

    def test_kernel_weights_per_coalition_size_degree_1(self):
        """Test kernel weights per coalition size for degree 1."""
        p = 6
        degree = 1
        weights = calculate_kernel_weights_per_coalition_size(p, degree)

        # Test against R: kernelshap::kernel_weights_per_coalition_size(6, 2:4)
        expected = [0.3461538, 0.3076923, 0.3461538]
        np.testing.assert_array_almost_equal(weights, expected, decimal=5)

    def test_kernel_weights_per_coalition_size_error(self):
        """Test error when p is too small for degree."""
        with pytest.raises(
            ValueError, match="The number of features p must be at least"
        ):
            calculate_kernel_weights_per_coalition_size(p=3, degree=1)


class TestCalculateExactProp:
    """Test exact proportion calculation."""

    def test_exact_prop_zero_degree(self):
        """Test exact proportion for degree 0."""
        result = calculate_exact_prop(p=5, degree=0)
        assert result == 0.0

    def test_exact_prop_positive_degree(self):
        """Test exact proportion for positive degree."""
        p = 6
        degree = 2
        result = calculate_exact_prop(p, degree)

        # test against R kernelshap:::prop_exact(6, 2)
        assert np.isclose(result, 0.8540146)

    def test_exact_prop_half_features(self):
        """Test exact proportion when degree = p/2."""
        p = 6
        degree = 3
        result = calculate_exact_prop(p, degree)

        # test against R kernelshap:::prop_exact(6, 3)
        assert np.isclose(result, 1.0)


class TestPrepareInputs:
    """Test input preparation functions."""

    def test_prepare_input_exact(self):
        """Test exact input preparation."""
        p = 3
        result = prepare_input_exact(p)

        # Check required keys
        assert "Z" in result
        assert "w" in result
        assert "A" in result

        # Check dimensions
        assert result["Z"].shape == (2**p - 2, p)  # All masks except empty and full
        assert result["w"].shape == (2**p - 2, 1)
        assert result["A"].shape == (p, p)

        # Weights
        assert np.isclose(result["w"].sum(), 1.0)
        assert (result["w"] > 0).all()

        # A should be symmetric
        np.testing.assert_array_almost_equal(result["A"], result["A"].T)

    def test_prepare_input_hybrid(self):
        """Test hybrid input preparation."""
        p = 6
        degree = 2
        result = prepare_input_hybrid(p, degree)

        # Check required keys
        assert "Z" in result
        assert "w" in result
        assert "A" in result

        # Check dimensions
        expected_rows = 2 * (6 + 15)  # 2 * (choose(6, 1) + choose(6, 2))
        assert result["Z"].shape == (expected_rows, p)
        assert result["w"].shape == (expected_rows, 1)
        assert result["A"].shape == (p, p)

        # Weights
        assert (result["w"] > 0).all()

        # A should be symmetric
        np.testing.assert_array_almost_equal(result["A"], result["A"].T)

    def test_prepare_input_hybrid_errors(self):
        """Test hybrid input preparation error cases."""
        with pytest.raises(ValueError, match="degree must be at least 1"):
            prepare_input_hybrid(p=5, degree=0)

        with pytest.raises(ValueError, match="p must be >= 2 \\* degree"):
            prepare_input_hybrid(p=3, degree=2)

    def test_prepare_input_sampling(self):
        """Test sampling input preparation."""
        p = 5
        degree = 1
        start = 0
        rng = np.random.default_rng(0)

        result = prepare_input_sampling(p, degree, start, rng)

        # Check required keys
        assert "Z" in result
        assert "w" in result
        assert "A" in result

        # Check dimensions
        expected_rows = 2 * (p - 1 - 2 * degree)
        assert result["Z"].shape == (expected_rows, p)
        assert result["w"].shape == (expected_rows, 1)
        assert result["A"].shape == (p, p)

        # A should be symmetric
        np.testing.assert_array_almost_equal(result["A"], result["A"].T)

    def test_prepare_input_sampling_approximately_exact(self):
        """Test that sampling A approximates exact A when repeating many times."""

        rng = np.random.default_rng(0)
        nsim = 1000

        for p in [4, 5, 6]:
            A_samp = np.zeros((p, p))
            for j in range(nsim):
                A_samp += prepare_input_sampling(p, degree=0, start=j % p, rng=rng)["A"]
            A_samp /= nsim
            A_exact = prepare_input_exact(p)["A"]
            assert np.abs(A_exact - A_samp).max() < 0.01

    def test_prepare_input_hybrid_approximately_exact(self):
        """Test that hybrid A approximates exact A for different degrees when repeating
        many times.
        """

        rng = np.random.default_rng(0)
        nsim = 1000

        for p, degree in ((4, 1), (5, 1), (6, 1), (6, 2), (7, 2)):
            A_sampling = np.zeros((p, p))
            for j in range(nsim):
                A_sampling += prepare_input_sampling(
                    p, degree=degree, start=j % p, rng=rng
                )["A"]
            A_hybrid_exact = prepare_input_hybrid(p, degree=degree)["A"]
            A_hybrid_sampling = A_sampling / nsim
            A_hybrid = A_hybrid_sampling + A_hybrid_exact
            A_exact = prepare_input_exact(p)["A"]
            assert np.abs(A_exact - A_hybrid).max() < 0.01

    def test_prepare_input_hybrid_sampling_give_weight_one(self):
        """Test hybrid and sampling weights sum to 1."""
        for p, degree in ((4, 1), (5, 1), (6, 1), (6, 2), (7, 2)):
            start = 0
            rng = np.random.default_rng(0)

            sampling = prepare_input_sampling(p, degree, start, rng)["w"]
            hybrid = prepare_input_hybrid(p, degree)["w"]

            assert np.isclose(sampling.sum() + hybrid.sum(), 1.0)

    def test_prepare_input_sampling_error(self):
        """Test sampling input preparation error case."""
        rng = np.random.default_rng(42)
        with pytest.raises(
            ValueError, match="The number of features p must be at least"
        ):
            prepare_input_sampling(p=3, degree=1, start=0, rng=rng)


class TestKernelSolver:
    """Test kernel solver function."""

    def test_kernel_solver_basic(self):
        """Test basic kernel solver functionality."""
        # Simple well-conditioned system
        A = np.array([[1.0, 0.1], [0.1, 1.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        constraint = np.array([[4.0, 6.0]])  # Sum constraint

        result = kernel_solver(A, b, constraint)

        # Check against R
        # A = rbind(c(1.0, 0.1), c(0.1, 1.0))
        # b = rbind(c(1.0, 2.0), c(3.0, 4.0))
        # constraint = c(4.0, 6.0)
        # kernelshap:::solver(A, b, constraint)

        expected = [[0.8888889, 1.888889], [3.1111111, 4.1111111]]
        np.testing.assert_array_almost_equal(result, expected)

    def test_kernel_solver_singular_matrix(self):
        """Test kernel solver with singular matrix."""
        # Singular matrix
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([[1.0], [1.0]])
        constraint = np.array([[2.0]])

        with pytest.raises(ValueError, match="Matrix A is singular"):
            kernel_solver(A, b, constraint)


class TestPrecalculateKernelShap:
    """Test precalculation for Kernel SHAP."""

    def test_precalculate_exact(self):
        """Test precalculation for exact method."""
        p = 3
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, p)), columns=["A", "B", "C"])

        result = precalculate_kernelshap(p, X, how="exact")

        # Check required keys
        required_keys = ["Z", "w", "A", "masks_exact_rep", "bg_exact_rep"]
        for key in required_keys:
            assert key in result

        # Check dimensions
        assert result["Z"].shape == (2**p - 2, p)
        assert result["masks_exact_rep"].shape == ((2**p - 2) * 10, p)
        assert result["bg_exact_rep"].shape == ((2**p - 2) * 10, p)

    def test_precalculate_hybrid(self):
        """Test precalculation for hybrid methods."""
        p = 6
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, p)), columns=list("ABCDEF"))

        for how in ["h1", "h2"]:
            result = precalculate_kernelshap(p, X, how=how)

            # Check required keys
            required_keys = [
                "Z",
                "w",
                "A",
                "masks_exact_rep",
                "bg_exact_rep",
                "bg_sampling_rep",
            ]
            for key in required_keys:
                assert key in result

    def test_precalculate_sampling(self):
        """Test precalculation for sampling method."""
        p = 5
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, p)), columns=list("ABCDE"))

        result = precalculate_kernelshap(p, X, how="sampling")

        # Check required keys - sampling doesn't have exact parts
        assert "bg_sampling_rep" in result
        assert "masks_exact_rep" not in result
        assert "bg_exact_rep" not in result


class TestOneKernelShap:
    """Test single row explanation with Kernel SHAP."""

    def test_exact_kernelshap_single_row(self):
        """Test exact Kernel SHAP for a single row."""
        # Set up test data
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 3)), columns=["A", "B", "C"])
        bg_X = X.iloc[:10]
        bg_w = None

        # Simple linear model
        weights = np.array([1.0, 2.0, -1.0])

        def predict_fn(X):
            return (X.values @ weights).reshape(-1, 1)

        v0 = predict_fn(bg_X).mean(keepdims=True)
        v1 = predict_fn(X)

        precalc = precalculate_kernelshap(p=3, bg_X=bg_X, how="exact")

        shap_values, se, converged, n_iter = one_kernelshap(
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

        # Check efficiency property
        prediction_diff = v1[0] - v0[0]
        shap_sum = shap_values.sum(axis=0)
        np.testing.assert_array_almost_equal(shap_sum, prediction_diff)

    def test_sampling_kernelshap_single_row(self):
        """Test sampling Kernel SHAP for a single row."""
        X_large = pd.DataFrame(np.random.randn(20, 5), columns=list("ABCDE"))
        bg_X_large = X_large.iloc[:10]

        # Extend the linear model
        weights_large = np.array([1.0, 2.0, -1.0, 0.5, 0.3])

        def predict_fn_large(X):
            return (X.values @ weights_large).reshape(-1, 1)

        v0_large = predict_fn_large(bg_X_large).mean(keepdims=True)
        v1_large = predict_fn_large(X_large)

        precalc = precalculate_kernelshap(p=5, bg_X=bg_X_large, how="sampling")

        shap_values, se, converged, n_iter = one_kernelshap(
            i=0,
            predict=predict_fn_large,
            how="sampling",
            bg_w=None,
            v0=v0_large,
            max_iter=20,
            tol=0.01,
            random_state=0,
            X=X_large,
            v1=v1_large,
            precalc=precalc,
            collapse=np.array([False]),
            bg_n=10,
        )

        # Check output shapes
        assert shap_values.shape == (5, 1)
        assert se.shape == (5, 1)
        assert isinstance(converged, bool)
        assert n_iter == 2

        # Check approximate efficiency (sampling might not be perfect)
        prediction_diff = v1_large[0] - v0_large[0]
        shap_sum = shap_values.sum(axis=0)
        np.testing.assert_array_almost_equal(shap_sum, prediction_diff)

    def test_hybrid_kernelshap_single_row(self):
        """Test hybrid Kernel SHAP for a single row."""
        X_medium = pd.DataFrame(np.random.randn(20, 7), columns=list("ABCDEFG"))
        bg_X_medium = X_medium.iloc[:10]

        # Linear model for 7 features
        weights_medium = np.array([1.0, 2.0, -1.0, 0.5, 0.3, 0.1, 0.2])

        def predict_fn_medium(X):
            return (X.values @ weights_medium).reshape(-1, 1)

        v0_medium = predict_fn_medium(bg_X_medium).mean(keepdims=True)
        v1_medium = predict_fn_medium(X_medium)

        for how in ["h1", "h2"]:
            precalc = precalculate_kernelshap(p=7, bg_X=bg_X_medium, how=how)

            shap_values, se, _, _ = one_kernelshap(
                i=0,
                predict=predict_fn_medium,
                how=how,
                bg_w=None,
                v0=v0_medium,
                max_iter=10,
                tol=0.01,
                random_state=0,
                X=X_medium,
                v1=v1_medium,
                precalc=precalc,
                collapse=np.array([False]),
                bg_n=10,
            )

            # Check output shapes
            assert shap_values.shape == (7, 1)
            assert se.shape == (7, 1)

            # Check efficiency property (hybrid should be exact for linear models)
            prediction_diff = v1_medium[0] - v0_medium[0]
            shap_sum = shap_values.sum(axis=0)
            np.testing.assert_array_almost_equal(shap_sum, prediction_diff)
            shap_sum = shap_values.sum(axis=0)
            np.testing.assert_array_almost_equal(shap_sum, prediction_diff)
