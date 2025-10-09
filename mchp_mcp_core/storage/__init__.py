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
from mchp_mcp_core.storage.chromadb import ChromaDBVectorStore
from mchp_mcp_core.storage.sqlite import SQLiteCache
from mchp_mcp_core.storage.manifest import (
    ManifestStatus,
    ManifestEntry,
    DocumentManifest,
    ManifestRepository
)

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
    "ChromaDBVectorStore",
    # Caching
    "SQLiteCache",
    # Manifest
    "ManifestStatus",
    "ManifestEntry",
    "DocumentManifest",
    "ManifestRepository",
]
