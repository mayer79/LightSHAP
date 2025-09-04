import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from lightshap import explain_any


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
def test_parallel_vs_serial_methods(method, how):
    """Test that methods give consistent results in parallel mode."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=1)
    model = RandomForestClassifier(n_estimators=10, random_state=1)
    model.fit(X, y)

    X_small = X[0:5]

    # Serial execution
    result_serial = explain_any(
        model.predict_proba,
        X_small,
        bg_X=X,
        method=method,
        how=how,
        n_jobs=1,
        verbose=False,
        random_state=1,
    )

    # Parallel execution
    result_parallel = explain_any(
        model.predict_proba,
        X_small,
        bg_X=X,
        method=method,
        how=how,
        n_jobs=2,
        verbose=False,
        random_state=1,
    )

    np.testing.assert_allclose(
        result_serial.shap_values,
        result_parallel.shap_values,
        err_msg=f"Results too different for method={method}, how={how}",
    )
