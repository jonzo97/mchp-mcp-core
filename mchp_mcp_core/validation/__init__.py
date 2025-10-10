"""
Validation module for documentation completeness and correctness.

Provides tools for validating documentation quality:
- Semantic completeness validation
- Evidence checking
- Claim validation
"""

from mchp_mcp_core.validation.quality import (
    Claim,
    Evidence,
    ValidationResult,
    CompletenessValidator
)

__all__ = [
    "Claim",
    "Evidence",
    "ValidationResult",
    "CompletenessValidator",
]
