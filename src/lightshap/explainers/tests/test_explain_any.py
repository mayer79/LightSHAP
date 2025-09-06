import numpy as np
import pandas as pd
import pytest

from lightshap import explain_any

ATOL = 1e-6


def data_regression():
    rng = np.random.default_rng(1)

    n = 100
    X = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, size=n),
            "x2": rng.uniform(0, 1, size=n),
            "x3": rng.choice(["A", "B", "C"], size=n),
            "x4": pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n)),
            "x5": rng.uniform(0, 1, size=n),
            "x6": pd.Categorical(rng.choice(["e", "f", "g", "h"], size=n)),
        }
    )
    return X


@pytest.mark.parametrize("use_sample_weights", [False, True])
def test_exact_permutation_vs_kernel_shap_identical(use_sample_weights):
    """Test that exact methods return identical results."""

    X = data_regression()

    # Predict with interactions of order 4
    def predict(X):
        return (
            X["x1"] * X["x2"] * (X["x3"].isin(["A", "C"]) + 1) * (X["x4"].cat.codes + 1)
            + X["x5"]
            + X["x6"].cat.codes
        )

    if use_sample_weights:
        rng = np.random.default_rng(1)
        sample_weights = rng.uniform(0.0, 1.0, size=X.shape[0])
    else:
        sample_weights = None

    X_small = X.head(10)

    # Get explanations using exact permutation SHAP
    explanation_perm = explain_any(
        predict=predict,
        X=X_small,
        bg_X=X,
        bg_w=sample_weights,
        method="permutation",
        how="exact",
        verbose=False,
    )

    # Get explanations using exact kernel SHAP
    explanation_kernel = explain_any(
        predict=predict,
        X=X_small,
        bg_X=X,
        bg_w=sample_weights,
        method="kernel",
        how="exact",
        verbose=False,
    )

    np.testing.assert_allclose(
        explanation_perm.shap_values, explanation_kernel.shap_values, atol=ATOL
    )
    np.testing.assert_allclose(
        explanation_perm.baseline, explanation_kernel.baseline, atol=ATOL
    )


@pytest.mark.parametrize("use_sample_weights", [False, True])
@pytest.mark.parametrize(
    ("method", "how"),
    [
        ("kernel", "exact"),
        ("kernel", "sampling"),
        ("kernel", "h1"),
        ("kernel", "h2"),
        ("permutation", "sampling"),
    ],
)
def test_permutation_vs_kernel_shap_with_interactions(use_sample_weights, method, how):
    """Test that algorithms agree for models with interactions of order up to two."""

    X = data_regression()

    # Predict with interactions of order 2
    def predict(X):
        return (
            X["x1"] * X["x2"]
            + (X["x3"].isin(["A", "C"]) + 1) * (X["x4"].cat.codes + 1)
            + X["x5"]
            + X["x6"].cat.codes
        )

    if use_sample_weights:
        rng = np.random.default_rng(1)
        sample_weights = rng.uniform(0.0, 1.0, size=X.shape[0])
    else:
        sample_weights = None

    X_small = X.head(5)

    # Get explanations using permutation SHAP
    reference = explain_any(
        predict=predict,
        X=X_small,
        bg_X=X,
        bg_w=sample_weights,
        method="permutation",
        how="exact",
        verbose=False,
    )

    explanation = explain_any(
        predict=predict,
        X=X_small,
        bg_X=X,
        bg_w=sample_weights,
        method=method,
        how=how,
        verbose=False,
    )

    np.testing.assert_allclose(
        reference.shap_values, explanation.shap_values, atol=ATOL
    )
    np.testing.assert_allclose(explanation.baseline, reference.baseline, atol=ATOL)


@pytest.mark.parametrize("method", ["kernel", "permutation"])
def test_against_shap_library_reference(method):
    """Test against known SHAP values from the Python shap library."""
    # Expected values from shap library (Exact explainers)
    # Note that sampling methods do not perform very well here as the features
    # are extremely highly correlated.

    expected = np.array(
        [
            [-1.19621609, -1.24184808, -0.9567848, 3.87942037, -0.33825, 0.54562519],
            [-1.64922699, -1.20770105, -1.18388581, 4.54321217, -0.33795, -0.41082395],
        ]
    )

    n = 100
    X = pd.DataFrame(
        {
            "x1": np.arange(1, n + 1) / 100,
            "x2": np.log(np.arange(1, n + 1)),
            "x3": np.sqrt(np.arange(1, n + 1)),
            "x4": np.sin(np.arange(1, n + 1)),
            "x5": (np.arange(1, n + 1) / 100) ** 2,
            "x6": np.cos(np.arange(1, n + 1)),
        }
    )

    def predict(X):
        return X["x1"] * X["x2"] * X["x3"] * X["x4"] + X["x5"] + X["x6"]

    X_test = X.head(2)

    explanation = explain_any(
        predict=predict,
        X=X_test,
        bg_X=X,
        method=method,
        how="exact",
        verbose=False,
    )

    # Reference via shap.explainers.ExactExplainer(predict, X)(X_test) (shap 0.47.2)

    np.testing.assert_allclose(explanation.shap_values, expected, atol=ATOL)


class TestWeights:
    """Test class for weight-related functionality."""

    @pytest.mark.parametrize(
        ("method", "how"),
        [
            ("kernel", "exact"),
            ("kernel", "sampling"),
            ("kernel", "h1"),
            ("kernel", "h2"),
            ("permutation", "exact"),
            ("permutation", "sampling"),
        ],
    )
    def test_constant_weights_equal_no_weights(self, method, how):
        """Test that constant weights equal no weights."""

        X = data_regression()

        def predict(X):
            return (
                X["x1"]
                * X["x2"]
                * (X["x3"].isin(["A", "C"]) + 1)
                * (X["x4"].cat.codes + 1)
                + X["x5"]
                + X["x6"].cat.codes
            )

        bg_w = np.full(X.shape[0], 2.0)
        X_small = X.head(10)

        # Get explanations without weights
        explanation_no_weights = explain_any(
            predict=predict,
            X=X_small,
            bg_X=X,
            bg_w=None,
            method=method,
            how=how,
            verbose=False,
            random_state=1,
        )

        # Get explanations with constant weights
        explanation_constant_weights = explain_any(
            predict=predict,
            X=X_small,
            bg_X=X,
            bg_w=bg_w,
            method=method,
            how=how,
            verbose=False,
            random_state=1,
        )

        np.testing.assert_allclose(
            explanation_no_weights.shap_values,
            explanation_constant_weights.shap_values,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            explanation_no_weights.baseline,
            explanation_constant_weights.baseline,
            atol=ATOL,
        )

    @pytest.mark.parametrize(
        ("method", "how"),
        [
            ("kernel", "exact"),
            ("kernel", "sampling"),
            ("kernel", "h1"),
            ("kernel", "h2"),
            ("permutation", "exact"),
            ("permutation", "sampling"),
        ],
    )
    def test_non_constant_weights_differ_from_no_weights(self, method, how):
        """Test that non-constant weights give different results than no weights."""

        X = data_regression()

        def predict(X):
            return (
                X["x1"] * X["x2"]
                + (X["x3"].isin(["A", "C"]) + 1) * (X["x4"].cat.codes + 1)
                + X["x5"]
                + X["x6"].cat.codes
            )

        # Create non-constant weights
        rng = np.random.default_rng(1)
        bg_w = rng.uniform(0.1, 2.0, size=X.shape[0])
        X_small = X.head(20)

        # Get explanations without weights
        explanation_no_weights = explain_any(
            predict=predict,
            X=X_small,
            bg_X=X,
            bg_w=None,
            method=method,
            how=how,
            verbose=False,
            random_state=1,
        )

        # Get explanations with non-constant weights
        explanation_weighted = explain_any(
            predict=predict,
            X=X_small,
            bg_X=X,
            bg_w=bg_w,
            method=method,
            how=how,
            verbose=False,
            random_state=1,
        )

        # Results should be different (not allclose)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                explanation_no_weights.shap_values,
                explanation_weighted.shap_values,
                atol=ATOL,
            )


@pytest.mark.parametrize(
    ("method", "how"),
    [
        ("permutation", "sampling"),
        ("kernel", "sampling"),
        ("kernel", "h1"),
        ("kernel", "h2"),
    ],
)
def test_sampling_methods_approximate_exact(method, how):
    """Test that sampling methods approximate exact results within tolerance."""
    # Note that we are using a model with interactions of order > 2 to see
    # differences between the methods.
    n = 100
    rng = np.random.default_rng(1)

    X = pd.DataFrame(rng.uniform(0, 2, (n, 6)), columns=[f"x{i}" for i in range(6)])

    def predict(X):
        return X["x0"] * X["x1"] * X["x2"] + X["x3"] * X["x4"] * X["x5"]

    X_test = X.head(5)

    # Exact reference
    exact = explain_any(
        predict=predict,
        X=X_test,
        bg_X=X,
        method="permutation",
        how="exact",
        verbose=False,
    )

    # Approximations of increasing quality
    approx = []
    for tol in [0.01, 0.005, 0.0025]:
        approx.append(
            explain_any(
                predict=predict,
                X=X_test,
                bg_X=X,
                method=method,
                how=how,
                tol=tol,
                max_iter=500,  # to avoid convergence warnings
                verbose=False,
                random_state=1,
            )
        )
    mae = [np.abs(apr.shap_values - exact.shap_values).mean() for apr in approx]

    # Approximations get better with smaller tolerance (but not necessarily strictly)
    assert mae[0] >= mae[1] >= mae[2]

    # These tolerances differ by factor of 4, so that equality is very unlikely
    assert mae[0] > mae[2]

    # Threshold somewhat arbitrary, but small given that average prediction is 2
    assert mae[2] <= 0.005


class TestErrorConditions:
    """Test class for error conditions and bad inputs."""

    def test_too_few_features(self):
        """Test that p < 2 raises ValueError."""
        X = pd.DataFrame({"x1": [1, 2, 3]})

        def predict(X):
            return X["x1"]

        with pytest.raises(ValueError, match="At least two features are required"):
            explain_any(predict, X)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        X = data_regression()

        def predict(X):
            return X["x1"] + X["x2"]

        with pytest.raises(
            ValueError, match="method must be 'permutation', 'kernel', or None"
        ):
            explain_any(predict, X, method="invalid")

    @pytest.mark.parametrize("how", ["invalid", "h1", "h2"])
    def test_invalid_how_for_permutation(self, how):
        """Test that invalid how for permutation SHAP raises ValueError."""
        X = data_regression()

        def predict(X):
            return X["x1"] + X["x2"]

        with pytest.raises(
            ValueError,
            match="how must be 'exact', 'sampling', or None for permutation SHAP",
        ):
            explain_any(predict, X, method="permutation", how=how)

    @pytest.mark.parametrize("how", ["invalid", "h3"])
    def test_invalid_how_for_kernel(self, how):
        """Test that invalid how for kernel SHAP raises ValueError."""
        X = data_regression()

        def predict(X):
            return X["x1"] + X["x2"]

        with pytest.raises(
            ValueError,
            match="how must be 'exact', 'sampling', 'h1', 'h2', or None for kernel SHAP",
        ):
            explain_any(predict, X, method="kernel", how=how)

    def test_sampling_permutation_too_few_features(self):
        """Test that sampling permutation SHAP with p < 4 raises ValueError."""
        X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": [7, 8, 9]})

        def predict(X):
            return X["x1"] + X["x2"] + X["x3"]

        with pytest.raises(
            ValueError, match="Sampling Permutation SHAP is not supported for p < 4"
        ):
            explain_any(predict, X, bg_X=X, method="permutation", how="sampling")

    def test_h1_kernel_too_few_features(self):
        """Test that h1 kernel SHAP with p < 4 raises ValueError."""
        X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": [7, 8, 9]})

        def predict(X):
            return X["x1"] + X["x2"] + X["x3"]

        with pytest.raises(
            ValueError, match="Degree 1 hybrid Kernel SHAP is not supported for p < 4"
        ):
            explain_any(predict, X, bg_X=X, method="kernel", how="h1")

    def test_h2_kernel_too_few_features(self):
        """Test that h2 kernel SHAP with p < 6 raises ValueError."""
        X = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [4, 5, 6, 7, 8],
                "x3": [7, 8, 9, 10, 11],
                "x4": [10, 11, 12, 13, 14],
                "x5": [13, 14, 15, 16, 17],
            }
        )

        def predict(X):
            return X["x1"] + X["x2"] + X["x3"] + X["x4"] + X["x5"]

        with pytest.raises(
            ValueError, match="Degree 2 hybrid Kernel SHAP is not supported for p < 6"
        ):
            explain_any(predict, X, bg_X=X, method="kernel", how="h2")

    @pytest.mark.parametrize("max_iter", [0, -1, -10, 0.5, "invalid"])
    def test_invalid_max_iter(self, max_iter):
        """Test that invalid max_iter raises ValueError."""
        X = data_regression()

        def predict(X):
            return X["x1"] + X["x2"]

        with pytest.raises(
            ValueError, match="max_iter must be a positive integer or None"
        ):
            explain_any(predict, X, max_iter=max_iter)
