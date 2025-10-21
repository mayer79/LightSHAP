"""LightSHAP: Lightweight SHAP implementation."""

from ._version import __version__
from .explainers import explain_any, explain_tree
from .explanation import Explanation

__all__ = ["__version__", "explain_any", "explain_tree", "Explanation"]
