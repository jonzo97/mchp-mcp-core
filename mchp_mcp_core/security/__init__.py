"""
Security utilities module.

Provides security features:
- PII redaction (emails, phones, SSNs)
- Path validation (prevent path traversal)
- File type restrictions
- Workspace sandboxing
"""

from mchp_mcp_core.security.pii import PIIRedactor, redact_pii
from mchp_mcp_core.security.validation import (
    validate_path,
    sanitize_filename,
    validate_file_type,
    validate_file_size
)

__all__ = [
    "PIIRedactor",
    "redact_pii",
    "validate_path",
    "sanitize_filename",
    "validate_file_type",
    "validate_file_size",
]
