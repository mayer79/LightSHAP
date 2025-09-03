![Logo](./docs/images/logo.svg?raw=true)

# LightSHAP

| | |
| --- | --- |
| Package | [![PyPI - Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/) [![GitHub release](https://img.shields.io/github/v/release/mayer79/LightSHAP)](https://github.com/mayer79/LightSHAP/releases) [![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/mayer79/LightSHAP) |
| CI/CD | [![CI - Test](https://github.com/mayer79/LightSHAP/actions/workflows/test.yml/badge.svg)](https://github.com/mayer79/LightSHAP/actions/workflows/test.yml) [![Release](https://github.com/mayer79/LightSHAP/actions/workflows/release.yml/badge.svg)](https://github.com/mayer79/LightSHAP/actions/workflows/release.yml) |
| Quality | [![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![GitHub issues](https://img.shields.io/github/issues/mayer79/LightSHAP)](https://github.com/mayer79/LightSHAP/issues) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![GitHub contributors](https://img.shields.io/github/contributors/mayer79/LightSHAP)](https://github.com/mayer79/LightSHAP/graphs/contributors) |

**Lightweight Python implementation of SHAP (SHapley Additive exPlanations).**

## Key Features

- **Tree Models**: TreeSHAP wrappers for XGBoost, LightGBM, and CatBoost via `explain_tree()`
- **Model-Agnostic**: Permutation SHAP and Kernel SHAP via `explain_any()`
- **Visualization**: Flexible plots

**Highlights of the agnostic explainer:**

1. Exact and sampling versions of permutation SHAP and Kernel SHAP
2. Sampling versions iterate until convergence, and provide standard errors
3. Parallel processing via joblib
4. Supports multi-output models
5. Supports case weights
6. Accepts numpy, pandas, and polars input, and categorical features

## Quick Start

```python
from lightshap import explain_any, explain_tree

# For any model
explanation = explain_any(model.predict, X)

# For tree models (XGBoost, LightGBM, CatBoost)  
explanation = explain_tree(model, X)

# Create plots
explanation.plot.bar()       # Feature importance
explanation.plot.beeswarm()  # Summary plot
explanation.plot.scatter()   # Dependence plots
explanation.plot.waterfall() # Individual explanation
```

## Installation

**From Test PyPI:**

```bash
pip install -i https://test.pypi.org/simple/ lightshap
```

**Requirements:** Python 3.11+ | Linux, Mac, Windows

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

![SHAP importance](./docs/images/tree_bar.png?raw=true)

#### SHAP summary

![SHAP summary](./docs/images/tree_beeswarm.png?raw=true)

#### SHAP dependence

![SHAP dependence](./docs/images/tree_scatter.png?raw=true)

#### Individual explanation

![SHAP waterfall](./docs/images/tree_waterfall.png?raw=true)

<br/>

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

## API Reference

### Main Functions

- `explain_any(predict, X, ...)` - Model-agnostic SHAP for any model
- `explain_tree(model, X, ...)` - TreeSHAP for XGBoost, LightGBM, and CatBoost models

### Explanation Object

- `.plot.bar()` - Feature importance plot
- `.plot.beeswarm()` - SHAP summary plot  
- `.plot.scatter()` - SHAP dependence plots
- `.plot.waterfall()` - Individual prediction explanation
- `.importance()` - Feature importance values
- `.filter()` - Filter observations by condition
- `.select_output()`- Select output (for multi-output models)
- `.set_feature_names()` - Change or set feature names, e.g., when `X` is numpy
- `.set_output_names()` - Change or set output names (for multi-output models)
- `.set_X()` - Change input data
- `...`

## Contributing

Contributions are highly appreciated! When contributing, you agree that your contributions will be subject to the [MIT License](https://github.com/mayer79/lightshap/blob/main/LICENSE).

Please feel free to open an issue for bug reports, feature requests, or general discussions.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgements

LightSHAP builds on top of wonderful packages like numpy, pandas, and matplotlib.

It is heavily influenced by these projects:

[shap](https://github.com/slundberg/shap) |
[shapley-regression](https://github.com/iancovert/shapley-regression) |
[kernelshap](https://github.com/ModelOriented/kernelshap) |
[shapviz](https://github.com/ModelOriented/shapviz)

# <a name="references">References</a>

<details>
<summary>
    <em>"A Unified Approach to Interpreting Model Predictions" (S. M. Lundberg and S.-I. Lee 2017)</em>
</summary>
<br/>
    <pre>
@incollection{lundberglee2017,
 title = {A Unified Approach to Interpreting Model Predictions},
 author = {Lundberg, Scott M and Lee, Su-In},
 booktitle = {Advances in Neural Information Processing Systems 30},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {4765--4774},
 year = {2017},
 publisher = {Curran Associates, Inc.},
 url = {https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
}
</pre>
<a href="https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf">Paper link</a>
</details>

<details>
<summary>
    <em>"Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression" (I. Covert and S.-I. Lee 2020)</em>
</summary>
<br/>
    <pre>
@inproceedings{covertlee2020,
  title={Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression},
  author={Ian Covert and Su-In Lee},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2020},
  url={https://proceedings.mlr.press/v130/covert21a/covert21a.pdf}
}
</pre>
<a href="https://proceedings.mlr.press/v130/covert21a/covert21a.pdf">Paper link</a>
</details>

