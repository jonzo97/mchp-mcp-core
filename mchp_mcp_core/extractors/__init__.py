"""
Document extraction module.

Provides extractors for various document formats:
- PDF (PyMuPDF-based with structure preservation)
- Tables (pdfplumber multi-strategy)
- PPTX (python-pptx slide parsing)
- Chunking (fixed-size and semantic strategies)
"""

from mchp_mcp_core.extractors.chunking import (
    perform_intelligent_chunking,
    split_text_chunk_fixed,
    split_text_chunk_semantic
)
from mchp_mcp_core.extractors.tables import (
    extract_tables_from_pdf,
    extract_tables_multi_strategy,
    is_table_empty,
    is_table_sparse,
    table_to_markdown,
    find_table_caption
)
from mchp_mcp_core.extractors.pdf import PDFExtractor
from mchp_mcp_core.extractors.pptx import PPTXExtractor

__all__ = [
    # Chunking
    "perform_intelligent_chunking",
    "split_text_chunk_fixed",
    "split_text_chunk_semantic",
    # Tables
    "extract_tables_from_pdf",
    "extract_tables_multi_strategy",
    "is_table_empty",
    "is_table_sparse",
    "table_to_markdown",
    "find_table_caption",
    # Extractors
    "PDFExtractor",
    "PPTXExtractor",
]
