# Changelog

## [Unreleased]

### Added
- More and better unit tests

### Fixed
- Tests requiring xgboost, lightgbm, and catboost are now skipped if the packages are not available

### Technical Details
- Renamed some files for consistency
- Updated GitHub release workflow to use "release" environment

## [0.1.4] - 2025-09-02

### Changed
- Migrated from setuptools to hatchling build backend
- Implemented dynamic versioning with hatch-vcs (version now reads from git tags)
- Modernized build system configuration

## [0.1.3] - 2025-09-02

### Fixed
- Fixed TestPyPI trusted publisher configuration error
- Resolved "invalid-publisher" error in release workflow

## [0.1.2] - 2025-09-02

### Fixed
- Fixed release workflow (removed unnecessary release environment)

## [0.1.1] - 2025-09-02

### Fixed
- Fixed release workflow to only publish to TestPyPI (removed PyPI publishing)
- Resolved workflow syntax errors and duplicate steps

## [0.1.0] - 2025-09-02

### Added
- Initial beta release of LightSHAP
- Model-agnostic SHAP via `explain_any()` function
  - Support for Permutation SHAP and Kernel SHAP
  - Exact and sampling methods with convergence detection
  - Hybrid approaches for large feature sets
- TreeSHAP via `explain_tree()` function
  - Support for XGBoost, LightGBM, and CatBoost
- Comprehensive visualization suite
  - Bar plots for feature importance
  - Beeswarm plots for summary visualization
  - Scatter plots to describe effects
  - Waterfall plots for individual explanations
- Multi-output model support
- Background data weighting
- Parallel processing via joblib
- Support for pandas, numpy, and polars DataFrames
- Categorical feature handling
- Standard error estimation for sampling methods

### Technical Details
- Python 3.11+ support
- Modern build system with Hatch
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Code quality enforcement with Ruff

