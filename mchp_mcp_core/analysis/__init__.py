"""
Analysis module for documentation quality checking.

Provides tools for analyzing documentation consistency and quality:
- Terminology consistency checking
- Pattern detection
- Style analysis
"""

from mchp_mcp_core.analysis.terminology import (
    Term,
    TermVariation,
    TerminologyAnalyzer
)

__all__ = [
    "Term",
    "TermVariation",
    "TerminologyAnalyzer",
]
