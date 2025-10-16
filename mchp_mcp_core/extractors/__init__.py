"""
Document extraction module.

Provides extractors for various document formats:
- PDF (PyMuPDF-based with structure preservation)
- Tables (pdfplumber multi-strategy)
- Advanced table extraction (multi-tool with confidence scoring) NEW!
- PPTX (python-pptx slide parsing)
- DOCX (python-docx document parsing)
- Chunking (fixed-size and semantic strategies)
- Metadata (document type, product family, version, date extraction)
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
from mchp_mcp_core.extractors.table_extractors import (
    TableExtractor,
    PdfPlumberExtractor,
    CamelotExtractor,
    PyMuPDFExtractor,
    ExtractedTable,
    ExtractionResult,
    ExtractionStrategy,
    TableComplexity,
)
from mchp_mcp_core.extractors.table_consensus import (
    TableConsensusEngine,
    HybridConsensusEngine,
    TableMatch,
    ConsensusResult,
)
from mchp_mcp_core.extractors.table_llm import (
    VisionLLMTableExtractor,
    VisionLLMConfig,
    VisionLLMProvider,
)
from mchp_mcp_core.extractors.table_validation import (
    TableValidator,
    ValidationIssue,
    ValidationResult,
    detect_header_row,
    split_multi_row_header,
)
from mchp_mcp_core.extractors.table_detection import (
    TableDetector,
    DetectionIssue,
    DetectionResult,
)
from mchp_mcp_core.extractors.table_evaluation import (
    TableEvaluator,
    EvaluationMetrics,
    MetricType,
    compute_straight_through_processing_rate,
)
from mchp_mcp_core.extractors.pdf import PDFExtractor
from mchp_mcp_core.extractors.pptx import PPTXExtractor
from mchp_mcp_core.extractors.docx import DOCXExtractor
from mchp_mcp_core.extractors.metadata import MetadataExtractor, extract_metadata

__all__ = [
    # Chunking
    "perform_intelligent_chunking",
    "split_text_chunk_fixed",
    "split_text_chunk_semantic",
    # Tables (legacy/simple API)
    "extract_tables_from_pdf",
    "extract_tables_multi_strategy",
    "is_table_empty",
    "is_table_sparse",
    "table_to_markdown",
    "find_table_caption",
    # Advanced table extraction (NEW)
    "TableExtractor",
    "PdfPlumberExtractor",
    "CamelotExtractor",
    "PyMuPDFExtractor",
    "ExtractedTable",
    "ExtractionResult",
    "ExtractionStrategy",
    "TableComplexity",
    "TableConsensusEngine",
    "HybridConsensusEngine",
    "TableMatch",
    "ConsensusResult",
    # Vision LLM extraction (NEW)
    "VisionLLMTableExtractor",
    "VisionLLMConfig",
    "VisionLLMProvider",
    # Table validation (NEW)
    "TableValidator",
    "ValidationIssue",
    "ValidationResult",
    "detect_header_row",
    "split_multi_row_header",
    # Table detection (Phase 1)
    "TableDetector",
    "DetectionIssue",
    "DetectionResult",
    # Table evaluation (Phase 2)
    "TableEvaluator",
    "EvaluationMetrics",
    "MetricType",
    "compute_straight_through_processing_rate",
    # Extractors
    "PDFExtractor",
    "PPTXExtractor",
    "DOCXExtractor",
    # Metadata
    "MetadataExtractor",
    "extract_metadata",
]
