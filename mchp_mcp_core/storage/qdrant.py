"""
Vector store implementation using Qdrant with native hybrid search support.

Leverages Qdrant's built-in BM25 + vector fusion for optimal retrieval.
"""
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from mchp_mcp_core.embeddings.sentence_transformers import EmbeddingModel
from mchp_mcp_core.storage.schemas import DocumentChunk, SearchQuery, SearchResult
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Vector store using Qdrant with native hybrid search.

    Features:
    - Hybrid search (BM25 + vector) in single API call
    - On-disk storage for large corpora
    - SHA-256 deduplication
    - Automatic collection creation
    - Batch upsert with progress tracking

    Example:
        >>> from mchp_mcp_core.utils.config import StorageConfig
        >>> config = StorageConfig()
        >>> store = QdrantVectorStore(config=config)
        >>> chunks = [DocumentChunk(...), ...]
        >>> store.add_documents(chunks)
        >>> results = store.search(SearchQuery(query="quantum computing", top_k=5))
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        qdrant_on_disk: Optional[bool] = None,
        hybrid_search: Optional[bool] = None
    ):
        """
        Initialize Qdrant vector store.

        Args:
            config: StorageConfig object or dict with storage settings
            collection_name: Name of the collection (default: from config or 'documents')
            embedding_model: Embedding model instance (default: create new)
            qdrant_host: Qdrant server host (default: from config or 'localhost')
            qdrant_port: Qdrant server port (default: from config or 6333)
            qdrant_on_disk: Store vectors on disk (default: from config or True)
            hybrid_search: Enable hybrid search (default: from config or True)
        """
        # Handle config
        if hasattr(config, 'qdrant_host'):
            # Pydantic config
            self.qdrant_host = qdrant_host or config.qdrant_host
            self.qdrant_port = qdrant_port or config.qdrant_port
            self.collection_name = collection_name or config.collection_name
            self.qdrant_on_disk = qdrant_on_disk if qdrant_on_disk is not None else config.qdrant_on_disk
            self.hybrid_search = hybrid_search if hybrid_search is not None else config.hybrid_search
        elif config:
            # Dict config
            storage_config = config.get('storage', {})
            self.qdrant_host = qdrant_host or storage_config.get('qdrant_host', 'localhost')
            self.qdrant_port = qdrant_port or storage_config.get('qdrant_port', 6333)
            self.collection_name = collection_name or storage_config.get('collection_name', 'documents')
            self.qdrant_on_disk = qdrant_on_disk if qdrant_on_disk is not None else storage_config.get('qdrant_on_disk', True)
            self.hybrid_search = hybrid_search if hybrid_search is not None else storage_config.get('hybrid_search', True)
        else:
            # Defaults
            self.qdrant_host = qdrant_host or 'localhost'
            self.qdrant_port = qdrant_port or 6333
            self.collection_name = collection_name or 'documents'
            self.qdrant_on_disk = qdrant_on_disk if qdrant_on_disk is not None else True
            self.hybrid_search = hybrid_search if hybrid_search is not None else True

        # Initialize embedding model
        self.embedder = embedding_model or EmbeddingModel()

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        # SHA-256 hash set for deduplication
        self._hash_set = set()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")

            # Load existing hashes for deduplication
            self._load_existing_hashes()

        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedder.dimension,
                        distance=Distance.COSINE,
                        on_disk=self.qdrant_on_disk
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info("Collection created successfully")

    def _load_existing_hashes(self):
        """Load SHA-256 hashes from existing points for deduplication."""
        try:
            # Scroll through all points to get hashes
            offset = None
            total_loaded = 0

            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for point in points:
                    if "sha256" in point.payload:
                        self._hash_set.add(point.payload["sha256"])
                        total_loaded += 1

                if next_offset is None:
                    break
                offset = next_offset

            if total_loaded > 0:
                logger.info(f"Loaded {total_loaded} existing hashes for deduplication")

        except Exception as e:
            logger.warning(f"Could not load existing hashes: {e}")

    def _compute_sparse_vector(self, text: str) -> models.SparseVector:
        """
        Compute sparse vector with TF-IDF weighting for text.

        Uses improved tokenization and TF-IDF normalization for better
        retrieval quality in hybrid search.

        Args:
            text: Input text

        Returns:
            models.SparseVector: Sparse vector representation
        """
        import re
        import math

        # Improved tokenization: lowercase, remove punctuation, split
        text = text.lower()
        # Remove punctuation but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 1]  # Filter single chars

        if not tokens:
            # Return empty sparse vector if no tokens
            return models.SparseVector(indices=[], values=[])

        # Calculate term frequency
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Calculate TF scores with normalization
        doc_length = len(tokens)
        tf_scores = {}

        for term, freq in term_freq.items():
            # TF with document length normalization
            # Using logarithmic scaling: log(1 + freq)
            tf = math.log(1 + freq) / math.log(1 + doc_length)
            tf_scores[term] = tf

        # Create sparse vector (term_hash -> tf_score)
        # Use dict to handle hash collisions by summing values
        hash_to_score = {}

        for term, score in sorted(tf_scores.items()):
            # Use stable hash for term index (MD5 truncated)
            term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16) % 1000000
            # Sum scores if hash collision occurs
            hash_to_score[term_hash] = hash_to_score.get(term_hash, 0.0) + score

        # Sort by hash for consistent ordering
        sorted_items = sorted(hash_to_score.items())
        indices = [idx for idx, _ in sorted_items]
        values = [float(val) for _, val in sorted_items]

        return models.SparseVector(indices=indices, values=values)

    def _build_filter_from_query(self, query: SearchQuery) -> Optional[models.Filter]:
        """
        Build Qdrant filter from SearchQuery enhanced filter parameters.

        Supports:
        - document_type: exact match
        - product_family: exact match
        - category_tags: must contain all tags (AND logic)
        - date_from/date_to: date range filter
        - filters: generic key-value filters

        Args:
            query: SearchQuery with filter parameters

        Returns:
            models.Filter or None if no filters specified
        """
        conditions = []

        # Generic filters (legacy support)
        if query.filters:
            for key, value in query.filters.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

        # Document type filter
        if query.document_type:
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=query.document_type)
                )
            )

        # Product family filter
        if query.product_family:
            conditions.append(
                models.FieldCondition(
                    key="product_family",
                    match=models.MatchValue(value=query.product_family)
                )
            )

        # Category tags filter (must have ALL tags)
        if query.category_tags:
            for tag in query.category_tags:
                conditions.append(
                    models.FieldCondition(
                        key="category_tags",
                        match=models.MatchValue(value=tag)
                    )
                )

        # Date range filter
        if query.date_from or query.date_to:
            # Note: Qdrant expects ISO 8601 strings for date comparison
            date_conditions = []

            if query.date_from:
                date_conditions.append(
                    models.FieldCondition(
                        key="document_date",
                        range=models.Range(gte=query.date_from)
                    )
                )

            if query.date_to:
                date_conditions.append(
                    models.FieldCondition(
                        key="document_date",
                        range=models.Range(lte=query.date_to)
                    )
                )

            conditions.extend(date_conditions)

        # Return filter if any conditions, else None
        if conditions:
            return models.Filter(must=conditions)
        return None

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        skip_duplicates: bool = True,
        show_progress: bool = True
    ) -> Tuple[int, int]:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for upsert operations
            skip_duplicates: Skip chunks with duplicate SHA-256 hashes
            show_progress: Show progress bar

        Returns:
            Tuple[int, int]: (chunks_added, duplicates_skipped)
        """
        chunks_added = 0
        duplicates_skipped = 0

        # Filter out duplicates if requested
        if skip_duplicates:
            unique_chunks = []
            for chunk in chunks:
                if chunk.sha256 not in self._hash_set:
                    unique_chunks.append(chunk)
                    self._hash_set.add(chunk.sha256)
                else:
                    duplicates_skipped += 1
            chunks = unique_chunks

        if not chunks:
            logger.info("No new chunks to add")
            return 0, duplicates_skipped

        # Process in batches
        iterator = range(0, len(chunks), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding documents")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            # Extract texts for embedding
            texts = [chunk.text for chunk in batch_chunks]

            # Generate dense embeddings
            dense_embeddings = self.embedder.embed(texts, show_progress=False)

            # Create points for Qdrant
            points = []
            for i, chunk in enumerate(batch_chunks):
                # Generate unique ID
                point_id = str(uuid.uuid4())

                # Dense vector - handle both numpy and tensor outputs
                dense_vector = dense_embeddings[i]
                if hasattr(dense_vector, 'cpu'):
                    # It's a tensor
                    dense_vector = dense_vector.cpu().tolist()
                elif hasattr(dense_vector, 'tolist'):
                    # It's a numpy array
                    dense_vector = dense_vector.tolist()
                else:
                    # Already a list
                    dense_vector = list(dense_vector)

                # Sparse vector (BM25)
                sparse_vector = self._compute_sparse_vector(chunk.text)

                # Payload (metadata)
                payload = chunk.to_dict()

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector
                    },
                    payload=payload
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            chunks_added += len(points)

        logger.info(f"Added {chunks_added} chunks, skipped {duplicates_skipped} duplicates")
        return chunks_added, duplicates_skipped

    def search(
        self,
        query: SearchQuery,
        embedding_model: Optional[EmbeddingModel] = None
    ) -> List[SearchResult]:
        """
        Search the vector store with hybrid search.

        Uses Qdrant's native BM25 + vector fusion with RRF (Reciprocal Rank Fusion).

        Args:
            query: SearchQuery object with search parameters
            embedding_model: Optional embedding model (default: use store's)

        Returns:
            List[SearchResult]: Ranked search results
        """
        embedder = embedding_model or self.embedder

        if query.hybrid and self.hybrid_search:
            # Hybrid search: BM25 + vector with RRF fusion
            return self._hybrid_search(query, embedder)
        else:
            # Vector-only search
            return self._vector_search(query, embedder)

    def _vector_search(
        self,
        query: SearchQuery,
        embedder: EmbeddingModel
    ) -> List[SearchResult]:
        """Perform vector-only search."""
        # Embed query
        query_vector = embedder.embed_query(query.query)

        # Handle numpy array
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
        else:
            query_vector = list(query_vector)

        # Build filter from query parameters
        query_filter = self._build_filter_from_query(query)

        # Search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_vector),
            query_filter=query_filter,
            limit=query.top_k,
            score_threshold=query.score_threshold
        )

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            result = SearchResult(
                id=str(hit.id),
                title=hit.payload.get("title", "Unknown"),
                source_path=hit.payload.get("source_path", "Unknown"),
                slide_or_page=hit.payload.get("slide_or_page", 0),
                snippet=hit.payload.get("text", "")[:300] + "...",
                score=hit.score,
                doc_id=hit.payload.get("doc_id", "Unknown"),
                chunk_id=hit.payload.get("chunk_id", 0),
                updated_at=hit.payload.get("updated_at")
            )
            results.append(result)

        return results

    def _hybrid_search(
        self,
        query: SearchQuery,
        embedder: EmbeddingModel
    ) -> List[SearchResult]:
        """
        Perform hybrid search using Qdrant's native RRF fusion.

        This is the key advantage of Qdrant: single API call for BM25 + vector.
        """
        # Embed query for dense vector
        query_vector_dense = embedder.embed_query(query.query)

        # Handle numpy array
        if hasattr(query_vector_dense, 'tolist'):
            query_vector_dense = query_vector_dense.tolist()
        else:
            query_vector_dense = list(query_vector_dense)

        # Compute sparse vector for BM25
        query_vector_sparse = self._compute_sparse_vector(query.query)

        # Build filter from query parameters
        query_filter = self._build_filter_from_query(query)

        # Hybrid search with prefetch + RRF fusion
        # This retrieves top_k*2 from each method, then fuses to get top_k
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector_dense,
                    using="dense",
                    limit=query.top_k * 2,
                    filter=query_filter
                ),
                models.Prefetch(
                    query=query_vector_sparse,
                    using="sparse",
                    limit=query.top_k * 2,
                    filter=query_filter
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=query.top_k,
            score_threshold=query.score_threshold
        )

        # Convert to SearchResult objects
        results = []
        for hit in search_results.points:
            result = SearchResult(
                id=str(hit.id),
                title=hit.payload.get("title", "Unknown"),
                source_path=hit.payload.get("source_path", "Unknown"),
                slide_or_page=hit.payload.get("slide_or_page", 0),
                snippet=hit.payload.get("text", "")[:300] + "...",
                score=hit.score,
                doc_id=hit.payload.get("doc_id", "Unknown"),
                chunk_id=hit.payload.get("chunk_id", 0),
                updated_at=hit.payload.get("updated_at")
            )
            results.append(result)

        return results

    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.name
        }

    def delete_collection(self):
        """Delete the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")


__all__ = ["QdrantVectorStore"]
