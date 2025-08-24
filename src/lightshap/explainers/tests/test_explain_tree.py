import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor

from lightshap import explain_tree


def classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=1,
    )
    return X, y


def regression_data():
    return make_regression(n_samples=100, n_features=4, random_state=1)


class TestXGBoost:
    """Test XGBoost models with explain_tree."""

    def test_xgboost_booster_regression_dmatrix(self):
        """Test XGBoost Booster with DMatrix for regression."""
        X, y = regression_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        model = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=10)

        # Get SHAP values directly from XGBoost
        expected_shap = model.predict(dtrain, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, dtrain)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == feature_names

    def test_xgboost_booster_regression_numpy(self):
        """Test XGBoost Booster with numpy array for regression."""
        X, y = regression_data()
        dtrain = xgb.DMatrix(X, label=y)

        model = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=10)

        # Get SHAP values directly from XGBoost
        expected_shap = model.predict(dtrain, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, X)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]

    def test_xgboost_regressor_pandas(self):
        """Test XGBRegressor with pandas DataFrame."""
        X, y = regression_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = xgb.XGBRegressor(n_estimators=10, random_state=0)
        model.fit(X_df, y)

        # Get SHAP values directly from XGBoost
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_df)
        expected_shap = booster.predict(dtest, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, X_df)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        pd.testing.assert_frame_equal(expl.X, X_df)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == feature_names

    def test_xgboost_classifier_multiclass(self):
        """Test XGBClassifier with 3 classes."""
        X, y = classification_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = xgb.XGBClassifier(n_estimators=10, random_state=0)
        model.fit(X_df, y)

        # Get SHAP values directly from XGBoost
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_df)
        expected_shap = booster.predict(dtest, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, X_df)

        # For multiclass, expected shape is (n, K, p+1) -> (n, p, K)
        expected_values = expected_shap[:, :, :-1].swapaxes(1, 2)
        expected_baseline = expected_shap[0, :, -1]

        np.testing.assert_allclose(expl.shap_values, expected_values)
        pd.testing.assert_frame_equal(expl.X, X_df)
        np.testing.assert_allclose(expl.baseline, expected_baseline)
        assert expl.feature_names == feature_names

    def test_xgboost_rf_regressor(self):
        """Test XGBRFRegressor."""
        X, y = regression_data()

        model = xgb.XGBRFRegressor(n_estimators=10, random_state=0)
        model.fit(X, y)

        # Get SHAP values directly from XGBoost
        booster = model.get_booster()
        dtest = xgb.DMatrix(X)
        expected_shap = booster.predict(dtest, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, X)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]

    def test_xgboost_rf_classifier(self):
        """Test XGBRFClassifier with 3 classes."""
        X, y = classification_data()

        model = xgb.XGBRFClassifier(n_estimators=10, random_state=0)
        model.fit(X, y)

        # Get SHAP values directly from XGBoost
        booster = model.get_booster()
        dtest = xgb.DMatrix(X)
        expected_shap = booster.predict(dtest, pred_contribs=True)

        # Test explain_tree
        expl = explain_tree(model, X)

        # For multiclass, expected shape is (n, K, p+1) -> (n, p, K)
        expected_values = expected_shap[:, :, :-1].swapaxes(1, 2)
        expected_baseline = expected_shap[0, :, -1]

        np.testing.assert_allclose(expl.shap_values, expected_values)
        np.testing.assert_allclose(expl.X, X)
        np.testing.assert_allclose(expl.baseline, expected_baseline)


class TestLightGBM:
    """Test LightGBM models with explain_tree."""

    def test_lightgbm_booster_numpy(self):
        """Test LightGBM Booster with numpy array."""
        X, y = regression_data()
        train_data = lgb.Dataset(X, label=y)

        model = lgb.train(
            {"objective": "regression", "verbose": -1}, train_data, num_boost_round=10
        )

        # Get SHAP values directly from LightGBM
        expected_shap = model.predict(X, pred_contrib=True)

        # Test explain_tree
        expl = explain_tree(model, X)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == model.feature_name()

    def test_lightgbm_regressor_pandas(self):
        """Test LGBMRegressor with pandas DataFrame."""
        X, y = regression_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1, random_state=0)
        model.fit(X_df, y)

        # Get SHAP values directly from LightGBM
        expected_shap = model.predict(X_df, pred_contrib=True)

        # Test explain_tree
        expl = explain_tree(model, X_df)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        pd.testing.assert_frame_equal(expl.X, X_df)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == feature_names

    def test_lightgbm_regressor_numpy(self):
        """Test LGBMRegressor with numpy array."""
        X, y = regression_data()

        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1, random_state=0)
        model.fit(X, y)

        # Get SHAP values directly from LightGBM
        expected_shap = model.predict(X, pred_contrib=True)

        # Test explain_tree
        expl = explain_tree(model, X)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]

    def test_lightgbm_classifier_multiclass(self):
        """Test LGBMClassifier with 3 classes."""
        X, y = classification_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=0)
        model.fit(X_df, y)

        # Get SHAP values directly from LightGBM
        expected_shap = model.predict(X_df, pred_contrib=True)

        # Test explain_tree
        expl = explain_tree(model, X_df)

        # For multiclass, reshape from (n, K*(p+1)) to (n, K, p+1) then (n, p, K)
        n, p = X_df.shape
        expected_shap_reshaped = expected_shap.reshape(n, -1, p + 1)
        expected_values = expected_shap_reshaped[:, :, :-1].swapaxes(1, 2)
        expected_baseline = expected_shap_reshaped[0, :, -1]

        np.testing.assert_allclose(expl.shap_values, expected_values)
        pd.testing.assert_frame_equal(expl.X, X_df)
        np.testing.assert_allclose(expl.baseline, expected_baseline)
        assert expl.feature_names == feature_names


class TestCatBoost:
    """Test CatBoost models with explain_tree."""

    def test_catboost_regressor_pandas(self):
        """Test CatBoostRegressor with pandas DataFrame."""
        X, y = regression_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = catboost.CatBoostRegressor(iterations=10, verbose=False, random_state=0)
        model.fit(X_df, y)

        # Get SHAP values directly from CatBoost
        pool = catboost.Pool(X_df)
        expected_shap = model.get_feature_importance(data=pool, fstr_type="ShapValues")

        # Test explain_tree
        expl = explain_tree(model, X_df)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        pd.testing.assert_frame_equal(expl.X, X_df)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == model.feature_names_

    def test_catboost_regressor_numpy(self):
        """Test CatBoostRegressor with numpy array."""
        X, y = regression_data()

        model = catboost.CatBoostRegressor(iterations=10, verbose=False, random_state=0)
        model.fit(X, y)

        # Get SHAP values directly from CatBoost
        pool = catboost.Pool(X, cat_features=model.get_cat_feature_indices())
        expected_shap = model.get_feature_importance(data=pool, fstr_type="ShapValues")

        # Test explain_tree
        expl = explain_tree(model, X)

        np.testing.assert_allclose(expl.shap_values, expected_shap[:, :-1])
        np.testing.assert_allclose(expl.X, X)
        assert expl.baseline == expected_shap[0, -1]
        assert expl.feature_names == model.feature_names_

    def test_catboost_classifier_multiclass_pandas(self):
        """Test CatBoostClassifier with 3 classes and pandas DataFrame."""
        X, y = classification_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = catboost.CatBoostClassifier(
            iterations=10, verbose=False, random_state=0
        )
        model.fit(X_df, y)

        # Get SHAP values directly from CatBoost
        pool = catboost.Pool(X_df)
        expected_shap = model.get_feature_importance(data=pool, fstr_type="ShapValues")

        # Test explain_tree
        expl = explain_tree(model, X_df)

        # For multiclass, expected shape is (n, K, p+1) -> (n, p, K)
        expected_values = expected_shap[:, :, :-1].swapaxes(1, 2)
        expected_baseline = expected_shap[0, :, -1]

        np.testing.assert_allclose(expl.shap_values, expected_values)
        pd.testing.assert_frame_equal(expl.X, X_df)
        np.testing.assert_allclose(expl.baseline, expected_baseline)
        assert expl.feature_names == feature_names

    def test_catboost_classifier_multiclass_numpy(self):
        """Test CatBoostClassifier with 3 classes and numpy array."""
        X, y = classification_data()
        model = catboost.CatBoostClassifier(
            iterations=10, verbose=False, random_state=0
        )
        model.fit(X, y)

        # Get SHAP values directly from CatBoost
        pool = catboost.Pool(X, cat_features=model.get_cat_feature_indices())
        expected_shap = model.get_feature_importance(data=pool, fstr_type="ShapValues")

        # Test explain_tree
        expl = explain_tree(model, X)

        # For multiclass, expected shape is (n, K, p+1) -> (n, p, K)
        expected_values = expected_shap[:, :, :-1].swapaxes(1, 2)
        expected_baseline = expected_shap[0, :, -1]

        np.testing.assert_allclose(expl.shap_values, expected_values)
        np.testing.assert_allclose(expl.X, X)
        np.testing.assert_allclose(expl.baseline, expected_baseline)
        assert expl.feature_names == model.feature_names_


class TestErrorHandling:
    """Test error handling for unsupported models."""

    def test_unsupported_model_raises_error(self):
        """Test that unsupported models raise TypeError."""

        X, y = make_regression(n_samples=100, n_features=4, random_state=1)
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X, y)

        with pytest.raises(
            TypeError, match="Model must be a LightGBM, XGBoost, or CatBoost model"
        ):
            explain_tree(model, X)

    def test_lgb_dataset_raises_error(self):
        """Test that LightGBM Dataset as X raises TypeError."""
        X, y = make_regression(n_samples=100, n_features=4, random_state=1)
        train_data = lgb.Dataset(X, label=y)

        model = lgb.train(
            {"objective": "regression", "verbose": -1}, train_data, num_boost_round=10
        )

        with pytest.raises(TypeError, match="X cannot be a lgb.Dataset"):
            explain_tree(model, train_data)
