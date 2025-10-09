"""
Storage layer for vector databases and caching.

Provides adapters for Qdrant, ChromaDB, and SQLite with hybrid search support.
"""

from mchp_mcp_core.storage.schemas import (
    DocumentChunk,
    SearchQuery,
    SearchResult,
    SearchResponse,
    IngestionReport,
    EvaluationMetrics,
    RetrievalContext
)
from mchp_mcp_core.storage.qdrant import QdrantVectorStore

__all__ = [
    # Schemas
    "DocumentChunk",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "IngestionReport",
    "EvaluationMetrics",
    "RetrievalContext",
    # Vector stores
    "QdrantVectorStore",
]
