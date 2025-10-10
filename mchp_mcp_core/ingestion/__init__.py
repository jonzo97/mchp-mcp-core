"""
Ingestion module for document processing orchestration.

Provides high-level orchestration for multi-format document ingestion:
- Directory scanning and job creation
- Parallel processing with progress tracking
- Manifest tracking (optional)
- Error handling and reporting
- JSONL export and HTML report generation
"""

from mchp_mcp_core.ingestion.orchestrator import (
    IngestionOrchestrator,
    IngestionJob,
    IngestionResult,
    sha256
)

__all__ = [
    "IngestionOrchestrator",
    "IngestionJob",
    "IngestionResult",
    "sha256",
]
