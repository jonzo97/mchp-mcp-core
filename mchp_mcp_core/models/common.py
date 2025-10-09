"""
Common Pydantic models shared across MCP servers.

These models provide standard schemas for documents, chunks, search queries,
and other common data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class ExtractedChunk:
    """
    Represents a chunk of extracted content from a document.

    Attributes:
        chunk_id: Unique identifier for this chunk
        content: The actual text/content of the chunk
        page_start: Starting page number (1-indexed)
        page_end: Ending page number (1-indexed)
        chunk_type: Type of chunk ('text', 'table', 'figure', etc.)
        section_hierarchy: Section/heading hierarchy this chunk belongs to
        metadata: Additional metadata (extraction method, quality, etc.)
    """
    chunk_id: str
    content: str
    page_start: int
    page_end: int
    chunk_type: str  # 'text', 'table', 'figure'
    section_hierarchy: str
    metadata: Dict[str, Any]


class DocumentChunk(BaseModel):
    """
    Pydantic model for a document chunk (for storage/API).

    This is similar to ExtractedChunk but uses Pydantic for validation.
    """
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content text")
    document_id: str = Field(..., description="Parent document ID")
    page_start: int = Field(ge=1, description="Starting page number")
    page_end: int = Field(ge=1, description="Ending page number")
    chunk_type: str = Field(default="text", description="Chunk type")
    section: Optional[str] = Field(None, description="Section hierarchy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_0_a1b2c3d4e5f6",
                "content": "This is example content...",
                "document_id": "doc_123",
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "text",
                "section": "1.0 Introduction",
                "metadata": {"extraction_method": "pymupdf"}
            }
        }


class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    hybrid: bool = Field(default=True, description="Use hybrid search (BM25 + vector)")


class SearchResult(BaseModel):
    """Search result item."""
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_id: Optional[str] = Field(None, description="Parent document ID")
    page: Optional[int] = Field(None, description="Page number")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_0_a1b2c3d4e5f6",
                "content": "Relevant content snippet...",
                "score": 0.87,
                "metadata": {"page": 1, "section": "Introduction"},
                "document_id": "doc_123",
                "page": 1
            }
        }


class DocumentMetadata(BaseModel):
    """Document metadata."""
    document_id: str = Field(..., description="Document identifier")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author name")
    total_pages: int = Field(ge=1, description="Total number of pages")
    filename: str = Field(..., description="Original filename")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
