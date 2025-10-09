"""
mchp-mcp-core: Shared core library for Microchip MCP servers and RAG tools.

This library provides reusable components for building MCP servers and RAG systems,
following the principle that MCP servers should be thin wrappers around business logic.

Modules:
    extractors: Document extraction (PDF, PPTX, tables, chunking)
    storage: Vector stores (Qdrant, ChromaDB) and caching (SQLite)
    embeddings: Embedding models (sentence-transformers)
    llm: LLM API clients and integration patterns
    security: PII redaction, path validation, sandboxing
    utils: Configuration, logging, common utilities
    models: Shared Pydantic models
"""

__version__ = "0.1.0"
__author__ = "Jonathan Orgill"
__email__ = "jonathan.orgill@microchip.com"

# Import key classes for convenience
from mchp_mcp_core import extractors, storage, embeddings, llm, security, utils, models

__all__ = [
    "extractors",
    "storage",
    "embeddings",
    "llm",
    "security",
    "utils",
    "models",
    "__version__",
]
