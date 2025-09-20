# LightSHAP

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

## Documentation

- [API Reference](api.md) - Detailed API documentation
- [Examples](examples.md) - Usage examples and tutorials
