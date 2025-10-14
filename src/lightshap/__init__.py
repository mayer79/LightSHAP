"""LightSHAP: Lightweight SHAP implementation."""

from ._version import __version__
from .explainers import explain_any, explain_tree

__all__ = ["__version__", "explain_any", "explain_tree"]
