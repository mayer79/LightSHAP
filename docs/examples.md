# Examples

## Example 1: CatBoost model for diamond prices

```python
import catboost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Load and prepare diamond data
diamonds = fetch_openml(data_id=42225, as_frame=True)

X = diamonds.data.assign(
    log_carat=lambda x: np.log(x.carat),  # better visualization
    clarity=lambda x: pd.Categorical(
        x.clarity, categories=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    ),
    cut=lambda x: pd.Categorical(
        x.cut, categories=["Fair", "Good", "Very Good", "Premium", "Ideal"]
    ),
)[["log_carat", "cut", "color", "clarity"]]
y = np.log(diamonds.target)

# Fit model
model = catboost.CatBoostRegressor(
    iterations=100, depth=4, cat_features=["cut", "color", "clarity"], verbose=0
)
model.fit(X, y=y)
```

### TreeSHAP analysis

```python
from lightshap import explain_tree

X_explain = X.sample(1000, random_state=0)
explanation = explain_tree(model, X_explain)

explanation.plot.bar()
explanation.plot.beeswarm()
explanation.plot.scatter(sharey=False)
explanation.plot.waterfall(row_id=0)
```

#### SHAP importance

![SHAP importance](images/tree_bar.png)

#### SHAP summary

![SHAP summary](images/tree_beeswarm.png)

#### SHAP dependence

![SHAP dependence](images/tree_scatter.png)

#### Individual explanation

![SHAP waterfall](images/tree_waterfall.png)


## Example 2: Linear regression with interactions

> **Note:** This example requires `glum`. Install with `pip install glum`

```python
from glum import GeneralizedLinearRegressor

# Fit with interactions
glm = GeneralizedLinearRegressor(
    family="gaussian",
    formula="log_carat * (clarity + cut + color)",
    drop_first=True,
)
glm.fit(X, y=y)
```

### Model-agnostic SHAP analysis

```python
from lightshap import explain_any

X_explain = X.sample(1000, random_state=0)
explanation = explain_any(glm.predict, X_explain)

explanation.plot.bar()
explanation.plot.beeswarm() 
explanation.plot.scatter(sharey=False)
explanation.plot.waterfall(row_id=0)
```
