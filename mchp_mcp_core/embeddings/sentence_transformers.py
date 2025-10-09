"""
Embedding model wrapper using sentence-transformers.

Provides a clean interface for embedding text using models like bge-small-en-v1.5
with support for CPU/GPU, caching, and batch processing.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.

    Features:
    - Auto-detection of device (CPU/GPU/MPS)
    - Batch processing with progress bars
    - Optional caching to disk
    - Normalized embeddings for cosine similarity

    Example:
        >>> embedder = EmbeddingModel()
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = embedder.embed(texts)
        >>> embeddings.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: Optional[int] = None,
        hf_cache_dir: Optional[Path] = None
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5)
            device: Device to use ('cpu', 'cuda', 'mps', default: auto-detect)
            cache_dir: Directory to cache embeddings (default: None)
            batch_size: Batch size for encoding (default: 32)
            hf_cache_dir: HuggingFace cache directory (default: None)
        """
        # Model configuration
        self.model_name = model_name or "BAAI/bge-small-en-v1.5"
        self.batch_size = batch_size or 32
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Device configuration
        if device:
            self.device = device
        else:
            self.device = self._auto_detect_device()

        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=str(hf_cache_dir) if hf_cache_dir else None
        )

        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

        # Initialize cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_index = self._load_cache_index()
        else:
            self._cache_index = {}

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_cache_index(self) -> dict:
        """Load the cache index from disk."""
        index_path = self.cache_dir / "embedding_cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache index: {e}")
                return {}
        return {}

    def _save_cache_index(self):
        """Save the cache index to disk."""
        if not self.cache_dir:
            return

        index_path = self.cache_dir / "embedding_cache_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self._cache_index, f)
        except Exception as e:
            logger.warning(f"Could not save cache index: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if it exists."""
        if not self.cache_dir:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key in self._cache_index:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    return np.load(cache_file)
                except Exception as e:
                    logger.warning(f"Could not load cached embedding: {e}")

        return None

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding to disk."""
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        try:
            np.save(cache_file, embedding)
            self._cache_index[cache_key] = True
        except Exception as e:
            logger.warning(f"Could not cache embedding: {e}")

    def embed(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            show_progress: Show progress bar
            normalize: Normalize embeddings for cosine similarity

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        # Check cache for single text
        if len(texts) == 1 and self.cache_dir:
            cached = self._get_cached_embedding(texts[0])
            if cached is not None:
                return cached.reshape(1, -1)

        # Encode texts
        if show_progress and len(texts) > 10:
            logger.info(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        # Cache if single text
        if len(texts) == 1 and self.cache_dir:
            self._cache_embedding(texts[0], embeddings)
            self._save_cache_index()

        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Convenience method to embed a single query.

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            1D numpy array of shape (dimension,)
        """
        embedding = self.embed([query], show_progress=False, normalize=normalize)
        return embedding[0]


__all__ = ["EmbeddingModel"]
