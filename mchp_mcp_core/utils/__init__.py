"""
Common utilities module.

Provides shared utilities:
- Configuration (Pydantic Settings-based)
- Logging (structured logging with secret masking)
- Hashing (file checksums and deduplication)
- Helper functions
"""

from mchp_mcp_core.utils.config import (
    ExtractionConfig,
    StorageConfig,
    EmbeddingConfig,
    LLMConfig,
    SecurityConfig,
    MCPCoreSettings,
    get_settings,
    load_config_from_dict
)
from mchp_mcp_core.utils.logger import get_logger
from mchp_mcp_core.utils.hashing import (
    compute_checksum,
    compute_checksums,
    find_duplicates
)

__all__ = [
    # Config
    "ExtractionConfig",
    "StorageConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "SecurityConfig",
    "MCPCoreSettings",
    "get_settings",
    "load_config_from_dict",
    # Logging
    "get_logger",
    # Hashing
    "compute_checksum",
    "compute_checksums",
    "find_duplicates",
]
