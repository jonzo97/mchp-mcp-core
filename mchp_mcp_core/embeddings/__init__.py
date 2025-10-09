"""
Embedding models module.

Provides clean wrappers for embedding models:
- sentence-transformers (bge-small-en and similar models)
- Auto device detection (CPU/CUDA/MPS)
- Batch processing with progress bars
- Optional disk caching
"""

from mchp_mcp_core.embeddings.sentence_transformers import EmbeddingModel

__all__ = ["EmbeddingModel"]
