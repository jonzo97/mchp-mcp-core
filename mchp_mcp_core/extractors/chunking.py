"""
Document chunking strategies.

Provides intelligent text chunking with two strategies:
1. Fixed-size chunking with overlap
2. Semantic chunking (section-aware, preserves coherence)
"""

import re
from typing import Dict, List

from mchp_mcp_core.models.common import ExtractedChunk


def perform_intelligent_chunking(
    chunks: List[ExtractedChunk],
    chunk_size: int = 1500,
    overlap: int = 200,
    chunking_strategy: str = "fixed",
    min_chunk_size: int = 500,
    max_chunk_size: int = 2500
) -> List[ExtractedChunk]:
    """
    Perform intelligent chunking on extracted content.

    Tables and figures are kept as single chunks.
    Text chunks are split if they exceed chunk_size.

    Args:
        chunks: List of extracted chunks
        chunk_size: Target chunk size (for fixed strategy)
        overlap: Overlap between chunks
        chunking_strategy: 'fixed' or 'semantic'
        min_chunk_size: Minimum chunk size (for semantic strategy)
        max_chunk_size: Maximum chunk size (for semantic strategy)

    Returns:
        List of final chunks with intelligent splitting applied
    """
    final_chunks = []

    for chunk in chunks:
        # Keep tables and figures as single chunks
        if chunk.chunk_type in ["table", "figure"]:
            final_chunks.append(chunk)
            continue

        # Split text chunks if they're too large
        if len(chunk.content) > chunk_size:
            if chunking_strategy == 'semantic':
                sub_chunks = split_text_chunk_semantic(
                    chunk,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    overlap=overlap
                )
            else:
                sub_chunks = split_text_chunk_fixed(
                    chunk,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


def split_text_chunk_fixed(
    chunk: ExtractedChunk,
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[ExtractedChunk]:
    """
    Split a large text chunk using fixed-size strategy with overlap.

    Args:
        chunk: Chunk to split
        chunk_size: Target size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of sub-chunks
    """
    sub_chunks = []
    text = chunk.content

    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    current_text = ""
    chunk_index = 0

    for para in paragraphs:
        # If adding this paragraph would exceed chunk size
        if len(current_text) + len(para) > chunk_size and current_text:
            # Save current chunk
            sub_chunk = ExtractedChunk(
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_index}",
                content=current_text.strip(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_type=chunk.chunk_type,
                section_hierarchy=chunk.section_hierarchy,
                metadata={**chunk.metadata, "sub_chunk": chunk_index, "chunking_strategy": "fixed"}
            )
            sub_chunks.append(sub_chunk)
            chunk_index += 1

            # Start new chunk with overlap
            overlap_text = current_text[-overlap:] if len(current_text) > overlap else current_text
            current_text = overlap_text + "\n\n" + para
        else:
            current_text += "\n\n" + para if current_text else para

    # Add final chunk
    if current_text.strip():
        sub_chunk = ExtractedChunk(
            chunk_id=f"{chunk.chunk_id}_sub_{chunk_index}",
            content=current_text.strip(),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_type=chunk.chunk_type,
            section_hierarchy=chunk.section_hierarchy,
            metadata={**chunk.metadata, "sub_chunk": chunk_index, "chunking_strategy": "fixed"}
        )
        sub_chunks.append(sub_chunk)

    return sub_chunks if sub_chunks else [chunk]


def split_text_chunk_semantic(
    chunk: ExtractedChunk,
    min_chunk_size: int = 500,
    max_chunk_size: int = 2500,
    overlap: int = 200
) -> List[ExtractedChunk]:
    """
    Split a large text chunk using semantic-aware strategy.

    Preserves section boundaries, paragraph structure, and topic coherence.

    Args:
        chunk: Chunk to split
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        overlap: Overlap size in characters (used for semantic overlap)

    Returns:
        List of sub-chunks
    """
    sub_chunks = []
    text = chunk.content

    # Identify semantic boundaries (section headers, major paragraph breaks)
    semantic_segments = identify_semantic_segments(text)

    current_text = ""
    chunk_index = 0

    for segment in semantic_segments:
        segment_text = segment['text']
        is_section_boundary = segment['is_boundary']

        # Force new chunk at section boundaries (if current chunk has content)
        if is_section_boundary and current_text and len(current_text) >= min_chunk_size:
            # Save current chunk
            sub_chunk = ExtractedChunk(
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_index}",
                content=current_text.strip(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_type=chunk.chunk_type,
                section_hierarchy=chunk.section_hierarchy,
                metadata={**chunk.metadata, "sub_chunk": chunk_index, "chunking_strategy": "semantic"}
            )
            sub_chunks.append(sub_chunk)
            chunk_index += 1

            # Start new chunk with semantic context (smaller overlap)
            overlap_text = get_semantic_overlap(current_text, overlap)
            current_text = overlap_text + "\n\n" + segment_text if overlap_text else segment_text

        # If adding this segment would exceed max size
        elif len(current_text) + len(segment_text) > max_chunk_size and current_text:
            # Save current chunk
            sub_chunk = ExtractedChunk(
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_index}",
                content=current_text.strip(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_type=chunk.chunk_type,
                section_hierarchy=chunk.section_hierarchy,
                metadata={**chunk.metadata, "sub_chunk": chunk_index, "chunking_strategy": "semantic"}
            )
            sub_chunks.append(sub_chunk)
            chunk_index += 1

            # Start new chunk with overlap
            overlap_text = get_semantic_overlap(current_text, overlap)
            current_text = overlap_text + "\n\n" + segment_text if overlap_text else segment_text
        else:
            current_text += "\n\n" + segment_text if current_text else segment_text

    # Add final chunk (if it meets minimum size or is the only chunk)
    if current_text.strip() and (len(current_text) >= min_chunk_size or chunk_index == 0):
        sub_chunk = ExtractedChunk(
            chunk_id=f"{chunk.chunk_id}_sub_{chunk_index}",
            content=current_text.strip(),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_type=chunk.chunk_type,
            section_hierarchy=chunk.section_hierarchy,
            metadata={**chunk.metadata, "sub_chunk": chunk_index, "chunking_strategy": "semantic"}
        )
        sub_chunks.append(sub_chunk)
    elif current_text.strip() and sub_chunks:
        # If final segment is too small, append to last chunk
        sub_chunks[-1].content += "\n\n" + current_text.strip()

    return sub_chunks if sub_chunks else [chunk]


def identify_semantic_segments(text: str) -> List[Dict]:
    """
    Identify semantic segments in text (sections, paragraphs, topic boundaries).

    Returns:
        List of dicts with 'text' and 'is_boundary' (True if section header)
    """
    segments = []

    # Split by double newline (paragraph boundaries)
    paragraphs = text.split('\n\n')

    # Patterns for section headers
    section_patterns = [
        re.compile(r'^(\d+(?:\.\d+)*)\s+[A-Z]'),           # "3.1 SECTION"
        re.compile(r'^(\d+(?:\.\d+)*)\s+[a-z]'),           # "3.1 introduction"
        re.compile(r'(?:^|\n)\s*(\d+(?:\.\d+)*)\s+\w+'),   # Number + any word
        re.compile(r'^SECTION\s+(\d+(?:\.\d+)*)'),         # "SECTION 3.1"
        re.compile(r'^Section\s+(\d+(?:\.\d+)*)'),         # "Section 3.1"
        re.compile(r'^[A-Z][A-Z\s]{10,}$'),                # ALL CAPS headings
        re.compile(r'^[A-Z][a-z\s]+:$'),                   # Title case with colon
    ]

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue

        # Check if this paragraph is a section header
        is_boundary = any(pattern.match(para_stripped) for pattern in section_patterns)

        # Also detect headers by length (very short paragraphs at line start)
        if not is_boundary and len(para_stripped) < 60 and '\n' not in para_stripped:
            # Might be a header if it starts with capital and doesn't end with period
            if para_stripped[0].isupper() and not para_stripped.endswith('.'):
                is_boundary = True

        segments.append({
            'text': para_stripped,
            'is_boundary': is_boundary
        })

    return segments


def get_semantic_overlap(text: str, overlap_size: int = 200) -> str:
    """
    Get semantically meaningful overlap (last complete sentence or two).

    More intelligent than fixed character overlap.

    Args:
        text: Text to get overlap from
        overlap_size: Suggested overlap size (used as max limit)

    Returns:
        Overlap text (last 1-2 sentences)
    """
    if len(text) < 100:
        return ""

    # Try to get last 1-2 sentences
    sentences = re.split(r'[.!?]\s+', text)

    if len(sentences) >= 2:
        # Return last 2 sentences (or less if they're too long)
        overlap_candidates = sentences[-2:]
        overlap_text = '. '.join(overlap_candidates)

        # Limit overlap to reasonable size
        max_overlap = min(overlap_size * 2, 400)
        if len(overlap_text) > max_overlap:
            return overlap_text[-max_overlap:]
        return overlap_text

    # Fallback: last N characters
    return text[-overlap_size:] if len(text) > overlap_size else text


__all__ = [
    "perform_intelligent_chunking",
    "split_text_chunk_fixed",
    "split_text_chunk_semantic",
    "identify_semantic_segments",
    "get_semantic_overlap"
]
