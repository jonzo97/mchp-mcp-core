"""
SQLite caching and state management layer.

Provides persistent storage for document chunks, embeddings, and metadata
with async support. Useful as a fallback when vector stores are unavailable.
"""
import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from mchp_mcp_core.storage.schemas import DocumentChunk
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCache:
    """
    Async SQLite cache for document storage and state management.

    Features:
    - Document chunk persistence
    - Embedding storage (fallback)
    - Metadata tracking
    - Async operations
    - State management

    Example:
        >>> async with SQLiteCache("./cache.db") as cache:
        ...     await cache.insert_chunks(chunks)
        ...     stored = await cache.get_chunks(document_id)
    """

    def __init__(self, db_path: str = "./cache.db"):
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self):
        """Establish database connection and create tables."""
        self.conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info(f"SQLite cache connected: {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.debug("SQLite cache connection closed")

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        # Document chunks table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                title TEXT,
                source_path TEXT,
                updated_at TEXT,
                slide_or_page INTEGER,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                sha256 TEXT,
                document_type TEXT,
                product_family TEXT,
                document_date TEXT,
                version TEXT,
                created_at TEXT
            )
        """)

        # Index for faster lookups
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
            ON chunks (doc_id)
        """)

        # Embeddings table (for fallback when vector store unavailable)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                embedding_vector TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
            )
        """)

        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id
            ON embeddings (chunk_id)
        """)

        # Document metadata table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                source_path TEXT,
                total_pages INTEGER,
                total_chunks INTEGER,
                created_at TEXT,
                last_updated TEXT,
                has_embeddings BOOLEAN DEFAULT 0,
                embedding_model TEXT,
                metadata TEXT
            )
        """)

        await self.conn.commit()

    # === Chunk Operations ===

    async def insert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Insert document chunks into cache.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0

        inserted = 0
        now = datetime.now().isoformat()

        for chunk in chunks:
            # Generate unique chunk ID
            chunk_id = f"{chunk.doc_id}_{chunk.slide_or_page}_{chunk.chunk_id}"

            try:
                await self.conn.execute("""
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, title, source_path, updated_at,
                        slide_or_page, chunk_index, text, sha256,
                        document_type, product_family, document_date, version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id, chunk.doc_id, chunk.title, chunk.source_path, chunk.updated_at,
                    chunk.slide_or_page, chunk.chunk_id, chunk.text, chunk.sha256,
                    chunk.document_type, chunk.product_family, chunk.document_date,
                    chunk.version, now
                ))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting chunk {chunk_id}: {e}")

        await self.conn.commit()
        logger.info(f"Inserted {inserted} chunks into cache")
        return inserted

    async def get_chunks(
        self,
        doc_id: str,
        chunk_type: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a document.

        Args:
            doc_id: Document identifier
            chunk_type: Optional filter by chunk type

        Returns:
            List of DocumentChunk objects
        """
        query = """
            SELECT doc_id, title, source_path, updated_at, slide_or_page,
                   chunk_index, text, sha256, document_type, product_family,
                   document_date, version
            FROM chunks
            WHERE doc_id = ?
            ORDER BY slide_or_page, chunk_index
        """

        chunks = []
        async with self.conn.execute(query, (doc_id,)) as cursor:
            async for row in cursor:
                chunk = DocumentChunk(
                    doc_id=row[0],
                    title=row[1] or "Unknown",
                    source_path=row[2] or "Unknown",
                    updated_at=row[3] or datetime.now().isoformat(),
                    slide_or_page=row[4] or 0,
                    chunk_id=row[5] or 0,
                    text=row[6],
                    sha256=row[7] or "",
                    document_type=row[8],
                    product_family=row[9],
                    document_date=row[10],
                    version=row[11]
                )
                chunks.append(chunk)

        return chunks

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        query = """
            SELECT doc_id, title, source_path, updated_at, slide_or_page,
                   chunk_index, text, sha256, document_type, product_family,
                   document_date, version
            FROM chunks
            WHERE chunk_id = ?
        """

        async with self.conn.execute(query, (chunk_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return DocumentChunk(
                    doc_id=row[0],
                    title=row[1] or "Unknown",
                    source_path=row[2] or "Unknown",
                    updated_at=row[3] or datetime.now().isoformat(),
                    slide_or_page=row[4] or 0,
                    chunk_id=row[5] or 0,
                    text=row[6],
                    sha256=row[7] or "",
                    document_type=row[8],
                    product_family=row[9],
                    document_date=row[10],
                    version=row[11]
                )

        return None

    # === Embedding Operations ===

    async def insert_embeddings(
        self,
        chunk_embeddings: List[tuple[str, List[float]]],
        model_name: str
    ) -> int:
        """
        Store embeddings for chunks.

        Args:
            chunk_embeddings: List of (chunk_id, embedding_vector) tuples
            model_name: Name of the embedding model

        Returns:
            Number of embeddings inserted
        """
        inserted = 0
        now = datetime.now().isoformat()

        for chunk_id, embedding_vector in chunk_embeddings:
            try:
                await self.conn.execute("""
                    INSERT INTO embeddings (chunk_id, embedding_vector, embedding_model, created_at)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, json.dumps(embedding_vector), model_name, now))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting embedding for {chunk_id}: {e}")

        await self.conn.commit()
        return inserted

    async def get_embedding(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding for a chunk.

        Returns:
            Dict with 'vector' and 'model' or None if not found
        """
        async with self.conn.execute("""
            SELECT embedding_vector, embedding_model
            FROM embeddings
            WHERE chunk_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (chunk_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'vector': json.loads(row[0]),
                    'model': row[1]
                }

        return None

    async def has_embeddings(self, doc_id: str) -> bool:
        """Check if a document has embeddings."""
        async with self.conn.execute("""
            SELECT COUNT(DISTINCT e.chunk_id)
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            WHERE c.doc_id = ?
        """, (doc_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    # === Document Metadata ===

    async def insert_document_metadata(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        total_pages: int = 0,
        metadata: Optional[Dict] = None
    ):
        """Insert or update document metadata."""
        now = datetime.now().isoformat()

        await self.conn.execute("""
            INSERT OR REPLACE INTO documents (
                doc_id, title, source_path, total_pages,
                created_at, last_updated, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, title, source_path, total_pages,
            now, now, json.dumps(metadata or {})
        ))
        await self.conn.commit()

    async def update_document_chunk_count(self, doc_id: str, total_chunks: int):
        """Update total chunk count for a document."""
        await self.conn.execute("""
            UPDATE documents
            SET total_chunks = ?, last_updated = ?
            WHERE doc_id = ?
        """, (total_chunks, datetime.now().isoformat(), doc_id))
        await self.conn.commit()

    async def get_document_stats(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a document."""
        async with self.conn.execute("""
            SELECT d.title, d.source_path, d.total_pages, d.total_chunks,
                   d.has_embeddings, d.embedding_model, d.created_at,
                   COUNT(DISTINCT c.chunk_id) as actual_chunks
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            WHERE d.doc_id = ?
            GROUP BY d.doc_id
        """, (doc_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'doc_id': doc_id,
                    'title': row[0],
                    'source_path': row[1],
                    'total_pages': row[2],
                    'total_chunks': row[3],
                    'has_embeddings': bool(row[4]),
                    'embedding_model': row[5],
                    'created_at': row[6],
                    'actual_chunks': row[7]
                }

        return None

    # === Utility Methods ===

    async def delete_document(self, doc_id: str):
        """Delete all data for a document."""
        await self.conn.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)", (doc_id,))
        await self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        await self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await self.conn.commit()
        logger.info(f"Deleted document: {doc_id}")

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in cache."""
        documents = []
        async with self.conn.execute("""
            SELECT doc_id, title, source_path, total_chunks, created_at
            FROM documents
            ORDER BY created_at DESC
        """) as cursor:
            async for row in cursor:
                documents.append({
                    'doc_id': row[0],
                    'title': row[1],
                    'source_path': row[2],
                    'total_chunks': row[3],
                    'created_at': row[4]
                })

        return documents


__all__ = ["SQLiteCache"]
