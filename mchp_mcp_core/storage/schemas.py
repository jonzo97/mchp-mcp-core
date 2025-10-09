"""
Pydantic schemas for document chunks and search results.

Provides type-safe data models for the storage and retrieval pipeline.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document.

    Used throughout the ingestion and retrieval pipeline.
    """

    doc_id: str = Field(..., description="Unique document identifier (usually file path)")
    title: str = Field(..., description="Document title (usually filename without extension)")
    source_path: str = Field(..., description="Path or URL to source document")
    updated_at: str = Field(..., description="Last modified timestamp (ISO 8601)")
    slide_or_page: int = Field(..., description="Slide number (PPTX) or page number (PDF)")
    chunk_id: int = Field(..., description="Chunk index within the slide/page")
    text: str = Field(..., description="Extracted text content")
    sha256: str = Field(..., description="SHA-256 hash for deduplication")

    # Enhanced metadata for filtering and categorization
    document_type: Optional[str] = Field(None, description="Document type: Datasheet, User Guide, Programming Guide, etc.")
    category_tags: List[str] = Field(default_factory=list, description="Category tags (subfolder-based + semantic)")
    product_family: Optional[str] = Field(None, description="Product family: PolarFire, IGLOO2, SmartFusion2, etc.")
    document_date: Optional[str] = Field(None, description="Document date extracted from content or filename (ISO 8601)")
    version: Optional[str] = Field(None, description="Document version (e.g., VB, V11, DS00003831)")
    subfolder_path: str = Field(default="", description="Relative path from docs/ root for categorization")

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text is not empty."""
        if not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()

    @field_validator("slide_or_page", "chunk_id")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure slide/page and chunk_id are positive."""
        if v < 0:
            raise ValueError("slide_or_page and chunk_id must be >= 0")
        return v

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(**data)


class SearchQuery(BaseModel):
    """Request schema for search endpoint."""

    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    hybrid: bool = Field(default=True, description="Use hybrid search (BM25 + vector)")
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Metadata filters (e.g., {'doc_id': 'path/to/doc.pdf'})"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    # Enhanced filter parameters
    document_type: Optional[str] = Field(None, description="Filter by document type")
    product_family: Optional[str] = Field(None, description="Filter by product family")
    category_tags: Optional[List[str]] = Field(None, description="Filter by category tags (AND logic)")
    date_from: Optional[str] = Field(None, description="Filter documents from this date (ISO 8601 or YYYY-MM)")
    date_to: Optional[str] = Field(None, description="Filter documents until this date (ISO 8601 or YYYY-MM)")


class SearchResult(BaseModel):
    """
    Single search result with citation metadata.

    Designed to provide all information needed for citation display.
    """

    id: str = Field(..., description="Vector store point ID")
    title: str = Field(..., description="Document title")
    source_path: str = Field(..., description="Source file path or URL")
    slide_or_page: int = Field(..., description="Slide or page number")
    snippet: str = Field(..., description="Text snippet (truncated if long)")
    score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    doc_id: str = Field(..., description="Document identifier")
    chunk_id: int = Field(..., description="Chunk index within slide/page")
    updated_at: Optional[str] = Field(None, description="Document last modified time")

    @property
    def location(self) -> str:
        """Human-readable location string (e.g., 'Slide 5' or 'Page 12')."""
        # Simple heuristic: if source ends with .pptx, it's a slide
        if self.source_path.lower().endswith((".pptx", ".ppt")):
            return f"Slide {self.slide_or_page}"
        else:
            return f"Page {self.slide_or_page}"

    @property
    def citation(self) -> str:
        """Formatted citation string."""
        return f"{self.title}, {self.location}"


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""

    results: List[SearchResult] = Field(..., description="List of search results")
    query: str = Field(..., description="Original query")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    total_results: int = Field(..., description="Total number of results found")
    hybrid_search: bool = Field(..., description="Whether hybrid search was used")


class IngestionReport(BaseModel):
    """Report generated after document ingestion."""

    total_files: int = Field(..., description="Total files processed")
    total_chunks: int = Field(..., description="Total chunks created")
    duplicates_skipped: int = Field(..., description="Duplicate chunks skipped")
    errors: int = Field(..., description="Number of errors encountered")
    duration_seconds: float = Field(..., description="Total ingestion time")
    files_processed: List[str] = Field(..., description="List of processed file paths")
    error_files: List[Dict[str, str]] = Field(..., description="Files that caused errors")

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return ((self.total_files - self.errors) / self.total_files) * 100

    @property
    def chunks_per_file(self) -> float:
        """Average chunks per successfully processed file."""
        successful_files = self.total_files - self.errors
        if successful_files == 0:
            return 0.0
        return self.total_chunks / successful_files


class EvaluationMetrics(BaseModel):
    """Metrics from P@K evaluation."""

    precision_at_1: float = Field(..., description="Precision at rank 1")
    precision_at_3: float = Field(..., description="Precision at rank 3")
    precision_at_5: float = Field(..., description="Precision at rank 5")
    mean_reciprocal_rank: float = Field(..., description="MRR across all queries")
    total_queries: int = Field(..., description="Total queries evaluated")
    queries_with_results: int = Field(..., description="Queries that returned results")

    @property
    def pass_threshold(self) -> bool:
        """Check if P@3 meets minimum threshold (0.7)."""
        return self.precision_at_3 >= 0.7


class RetrievalContext(BaseModel):
    """
    Context assembled for LLM generation.

    Contains retrieved chunks formatted for LLM consumption.
    """

    query: str = Field(..., description="Original user query")
    chunks: List[SearchResult] = Field(..., description="Retrieved and ranked chunks")
    system_prompt: Optional[str] = Field(None, description="System prompt for LLM")
    max_tokens: int = Field(default=2000, description="Maximum tokens for context")

    def format_for_llm(self) -> str:
        """
        Format retrieved context as a string for LLM consumption.

        Returns:
            str: Formatted context with citations
        """
        context_parts = ["Retrieved context:\n"]

        for i, chunk in enumerate(self.chunks, start=1):
            context_parts.append(
                f"\n[{i}] {chunk.citation}\n"
                f"{chunk.snippet}\n"
            )

        return "\n".join(context_parts)

    def get_citation_map(self) -> Dict[int, SearchResult]:
        """
        Get a mapping of citation numbers to search results.

        Returns:
            Dict[int, SearchResult]: Citation number -> SearchResult
        """
        return {i: chunk for i, chunk in enumerate(self.chunks, start=1)}


__all__ = [
    "DocumentChunk",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "IngestionReport",
    "EvaluationMetrics",
    "RetrievalContext"
]
