"""
Document manifest system for version tracking and ingestion state management.

Tracks document versions, checksums, and processing status using SQLModel.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class ManifestStatus(str, Enum):
    """
    Document processing lifecycle states.

    - STAGED: Document added but not processed
    - QUEUED: Queued for processing
    - EXTRACTING: Text/structure extraction in progress
    - INDEXING: Adding to vector store
    - READY: Fully processed and searchable
    - FAILED: Processing failed
    """

    STAGED = "staged"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ManifestEntry(SQLModel, table=True):
    """
    SQLModel representation of a manifest entry.

    Tracks document versions, checksums, and processing state.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True, description="Document identifier (usually filename without extension)")
    version: str = Field(index=True, description="Document version string")
    source_url: Optional[str] = Field(default=None, description="Source URL or file path")
    checksum: str = Field(unique=True, index=True, description="SHA-256 checksum for deduplication")
    size_bytes: int = Field(description="File size in bytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    status: ManifestStatus = Field(default=ManifestStatus.STAGED, description="Processing status")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    notes: Optional[str] = Field(default=None, description="Processing notes or error messages")


@dataclass
class DocumentManifest:
    """
    Serializable manifest representation for transport and business logic.

    Lightweight alternative to ManifestEntry for passing data between layers.
    """

    doc_id: str
    version: str
    checksum: str
    size_bytes: int
    status: ManifestStatus
    source_url: Optional[str] = None
    page_count: Optional[int] = None
    notes: Optional[str] = None


class ManifestRepository:
    """
    Database-backed manifest persistence.

    Features:
    - Checksum-based deduplication
    - Version tracking
    - Status updates
    - Query by status

    Example:
        >>> from pathlib import Path
        >>> repo = ManifestRepository(Path("./manifest.db"))
        >>> manifest = DocumentManifest(
        ...     doc_id="datasheet_001",
        ...     version="v1.0",
        ...     checksum="abc123...",
        ...     size_bytes=1024000,
        ...     status=ManifestStatus.STAGED
        ... )
        >>> entry = repo.upsert(manifest)
        >>> repo.update_status(entry.checksum, ManifestStatus.READY)
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize manifest repository.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)
        logger.info(f"Manifest repository initialized: {db_path}")

    def upsert(self, entry: DocumentManifest) -> ManifestEntry:
        """
        Insert or update a manifest entry.

        If an entry with the same checksum exists, it will be updated.
        Otherwise, a new entry is created.

        Args:
            entry: DocumentManifest to upsert

        Returns:
            ManifestEntry: The created or updated entry
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).where(ManifestEntry.checksum == entry.checksum)
            existing = session.exec(stmt).first()
            now = dt.datetime.now(dt.timezone.utc)

            if existing:
                # Update existing entry
                existing.doc_id = entry.doc_id
                existing.version = entry.version
                existing.status = entry.status
                existing.source_url = entry.source_url
                existing.size_bytes = entry.size_bytes
                existing.page_count = entry.page_count
                existing.notes = entry.notes
                existing.updated_at = now
                session.add(existing)
                session.commit()
                session.refresh(existing)
                logger.debug(f"Updated manifest entry: {entry.doc_id} (checksum: {entry.checksum[:8]}...)")
                return existing

            # Create new entry
            manifest_entry = ManifestEntry(
                doc_id=entry.doc_id,
                version=entry.version,
                checksum=entry.checksum,
                size_bytes=entry.size_bytes,
                status=entry.status,
                source_url=entry.source_url,
                page_count=entry.page_count,
                notes=entry.notes,
                created_at=now,
                updated_at=now,
            )
            session.add(manifest_entry)
            session.commit()
            session.refresh(manifest_entry)
            logger.info(f"Created manifest entry: {entry.doc_id} v{entry.version}")
            return manifest_entry

    def get_by_checksum(self, checksum: str) -> Optional[ManifestEntry]:
        """
        Retrieve a manifest entry by checksum.

        Args:
            checksum: Document checksum

        Returns:
            ManifestEntry or None if not found
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).where(ManifestEntry.checksum == checksum)
            return session.exec(stmt).first()

    def get_by_doc_id(self, doc_id: str) -> Iterable[ManifestEntry]:
        """
        Get all versions of a document.

        Args:
            doc_id: Document identifier

        Returns:
            Iterable of ManifestEntry objects
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).where(ManifestEntry.doc_id == doc_id).order_by(ManifestEntry.created_at.desc())
            return session.exec(stmt).all()

    def list_by_status(self, status: ManifestStatus) -> Iterable[ManifestEntry]:
        """
        List all documents with a specific status.

        Args:
            status: ManifestStatus to filter by

        Returns:
            Iterable of ManifestEntry objects
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).where(ManifestEntry.status == status)
            return session.exec(stmt).all()

    def update_status(
        self,
        checksum: str,
        status: ManifestStatus,
        notes: Optional[str] = None,
        page_count: Optional[int] = None
    ) -> None:
        """
        Update the status of a manifest entry.

        Args:
            checksum: Document checksum
            status: New status
            notes: Optional notes (e.g., error messages)
            page_count: Optional page count update

        Raises:
            ValueError: If entry with checksum not found
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).where(ManifestEntry.checksum == checksum)
            entry = session.exec(stmt).first()

            if not entry:
                raise ValueError(f"Manifest entry with checksum {checksum} not found")

            entry.status = status
            entry.updated_at = dt.datetime.now(dt.timezone.utc)

            if notes:
                entry.notes = notes

            if page_count is not None:
                entry.page_count = page_count

            session.add(entry)
            session.commit()
            logger.info(f"Updated manifest status: {entry.doc_id} -> {status.value}")

    def list_all(self, limit: int = 100) -> Iterable[ManifestEntry]:
        """
        List all manifest entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            Iterable of ManifestEntry objects
        """
        with Session(self.engine) as session:
            stmt = select(ManifestEntry).order_by(ManifestEntry.updated_at.desc()).limit(limit)
            return session.exec(stmt).all()

    def get_stats(self) -> dict:
        """
        Get statistics about manifested documents.

        Returns:
            Dict with counts by status
        """
        with Session(self.engine) as session:
            stats = {}
            for status in ManifestStatus:
                stmt = select(ManifestEntry).where(ManifestEntry.status == status)
                count = len(session.exec(stmt).all())
                stats[status.value] = count

            # Total count
            stmt = select(ManifestEntry)
            stats['total'] = len(session.exec(stmt).all())

            return stats


__all__ = ["ManifestStatus", "ManifestEntry", "DocumentManifest", "ManifestRepository"]
