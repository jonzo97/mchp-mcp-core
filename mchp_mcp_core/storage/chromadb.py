"""
ChromaDB vector store implementation with graceful fallback.

Development-friendly alternative to Qdrant with automatic degradation if unavailable.
"""
from typing import Any, Dict, List, Optional, Tuple

from mchp_mcp_core.embeddings.sentence_transformers import EmbeddingModel
from mchp_mcp_core.storage.schemas import DocumentChunk, SearchQuery, SearchResult
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaDBVectorStore:
    """
    Vector store using ChromaDB with graceful fallback.

    Features:
    - Development-friendly setup (no server required)
    - Document-per-collection isolation
    - Automatic degradation if ChromaDB unavailable
    - Compatible with existing storage interface

    Example:
        >>> from mchp_mcp_core.utils.config import StorageConfig
        >>> config = StorageConfig(chromadb_path="./chroma_data")
        >>> store = ChromaDBVectorStore(config=config)
        >>> if store.is_available():
        ...     store.add_documents(chunks)
        ...     results = store.search(SearchQuery(query="test", top_k=5))
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            config: StorageConfig object or dict with storage settings
            collection_name: Name of the collection (default: from config or 'documents')
            embedding_model: Embedding model instance (default: create new)
            db_path: Path to ChromaDB persistence directory
        """
        # Handle config
        if hasattr(config, 'chromadb_path'):
            # Pydantic config
            self.db_path = db_path or config.chromadb_path or "./chroma_data"
            self.collection_name = collection_name or config.collection_name
        elif config:
            # Dict config
            storage_config = config.get('storage', {})
            self.db_path = db_path or storage_config.get('chromadb_path', './chroma_data')
            self.collection_name = collection_name or storage_config.get('collection_name', 'documents')
        else:
            # Defaults
            self.db_path = db_path or './chroma_data'
            self.collection_name = collection_name or 'documents'

        # Initialize embedding model
        self.embedder = embedding_model or EmbeddingModel()

        # Try to initialize ChromaDB
        self.client = None
        self.collection = None
        self.available = False

        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client with error handling."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,  # No telemetry
                    allow_reset=False,            # Security: prevent accidental wipes
                )
            )

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            self.available = True
            logger.info(f"ChromaDB initialized at: {self.db_path}")
            logger.info(f"Collection: {self.collection_name} ({self.collection.count()} documents)")

        except ImportError:
            logger.warning("ChromaDB not installed. Install with: pip install chromadb")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB: {e}")
            self.available = False

    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        return self.available

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Tuple[int, int]:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for adding documents
            show_progress: Show progress bar

        Returns:
            Tuple[int, int]: (chunks_added, duplicates_skipped)
        """
        if not self.available:
            logger.error("ChromaDB not available")
            return 0, 0

        if not chunks:
            logger.info("No chunks to add")
            return 0, 0

        try:
            # Prepare data
            ids = [f"{chunk.doc_id}_{chunk.slide_or_page}_{chunk.chunk_id}" for chunk in chunks]
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.to_dict() for chunk in chunks]

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedder.embed(texts, show_progress=show_progress)

            # Convert to list format
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = [list(e) for e in embeddings]

            # Add to ChromaDB in batches
            chunks_added = 0
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))

                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings_list[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )

                chunks_added += (end_idx - i)

            logger.info(f"Added {chunks_added} chunks to ChromaDB")
            return chunks_added, 0

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0, 0

    def search(
        self,
        query: SearchQuery,
        embedding_model: Optional[EmbeddingModel] = None
    ) -> List[SearchResult]:
        """
        Search the vector store.

        Args:
            query: SearchQuery object with search parameters
            embedding_model: Optional embedding model (default: use store's)

        Returns:
            List[SearchResult]: Ranked search results
        """
        if not self.available:
            logger.warning("ChromaDB not available")
            return []

        try:
            embedder = embedding_model or self.embedder

            # Generate query embedding
            query_embedding = embedder.embed_query(query.query)

            # Convert to list
            if hasattr(query_embedding, 'tolist'):
                query_embedding_list = query_embedding.tolist()
            else:
                query_embedding_list = list(query_embedding)

            # Build metadata filter if provided
            where_filter = None
            if query.filters:
                where_filter = query.filters
            elif query.document_type or query.product_family:
                where_filter = {}
                if query.document_type:
                    where_filter["document_type"] = query.document_type
                if query.product_family:
                    where_filter["product_family"] = query.product_family

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=query.top_k,
                where=where_filter
            )

            # Format results
            search_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                    result = SearchResult(
                        id=results['ids'][0][i],
                        title=metadata.get('title', 'Unknown'),
                        source_path=metadata.get('source_path', 'Unknown'),
                        slide_or_page=metadata.get('slide_or_page', 0),
                        snippet=results['documents'][0][i][:300] + "...",
                        score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        doc_id=metadata.get('doc_id', 'Unknown'),
                        chunk_id=metadata.get('chunk_id', 0),
                        updated_at=metadata.get('updated_at')
                    )
                    search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        if not self.available:
            return {"available": False}

        try:
            return {
                "available": True,
                "name": self.collection_name,
                "points_count": self.collection.count(),
                "path": self.db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"available": False, "error": str(e)}

    def delete_collection(self):
        """Delete the collection (use with caution)."""
        if not self.available:
            logger.warning("ChromaDB not available")
            return

        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def list_collections(self) -> List[str]:
        """List all collections."""
        if not self.available:
            return []

        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []


__all__ = ["ChromaDBVectorStore"]
