"""
Advanced table extraction with multi-tool support and confidence scoring.

Provides a modular architecture for extracting tables from PDFs using multiple
extraction tools (pdfplumber, Camelot, PyMuPDF) with quality validation and
confidence scoring based on cross-tool agreement.

This module enables production-grade table extraction with quantitative
accuracy metrics, addressing common concerns about PDF table extraction reliability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class ExtractionStrategy(Enum):
    """Extraction strategy/mode used by extractor."""
    STANDARD = "standard"          # Default extraction
    LATTICE = "lattice"            # Grid-based (visible borders)
    STREAM = "stream"              # Text-based (borderless)
    LINES = "lines"                # Line detection
    TEXT = "text"                  # Pure text positioning
    HYBRID = "hybrid"              # Multiple strategies combined


class TableComplexity(Enum):
    """Complexity classification for tables."""
    SIMPLE = "simple"              # Regular grid, no merged cells
    MEDIUM = "medium"              # Some merged cells or formatting
    COMPLEX = "complex"            # Multi-level headers, irregular structure
    VERY_COMPLEX = "very_complex"  # Nested tables, rotated, or multi-page


@dataclass
class TableCell:
    """Represents a single table cell."""
    row: int
    col: int
    value: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    is_header: bool = False

    def __hash__(self):
        return hash((self.row, self.col, self.value))


@dataclass
class ExtractedTable:
    """
    Represents an extracted table with metadata and quality metrics.

    This is the primary output format for all extractors.
    """
    # Table data
    data: List[List[str]]  # 2D array of cell values

    # Location
    page_num: int
    table_index: int  # Index on the page (0-based)
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box on page

    # Metadata
    caption: Optional[str] = None
    rows: int = 0
    columns: int = 0

    # Extraction details
    extractor_name: str = "unknown"
    strategy: ExtractionStrategy = ExtractionStrategy.STANDARD
    extraction_time_ms: float = 0.0

    # Quality metrics (populated by validation)
    confidence: float = 0.0  # 0.0 to 1.0
    complexity: TableComplexity = TableComplexity.SIMPLE
    has_borders: bool = False
    sparsity: float = 0.0  # Ratio of empty cells

    # Validation issues
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived fields."""
        if self.data:
            self.rows = len(self.data)
            self.columns = len(self.data[0]) if self.data else 0
            self.sparsity = self._calculate_sparsity()

    def _calculate_sparsity(self) -> float:
        """Calculate ratio of empty cells."""
        if not self.data:
            return 1.0

        total_cells = sum(len(row) for row in self.data)
        if total_cells == 0:
            return 1.0

        empty_cells = sum(
            1 for row in self.data for cell in row
            if not cell or not str(cell).strip()
        )

        return empty_cells / total_cells

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.data:
            return ""

        lines = []

        # Header row
        header = [cell or "" for cell in self.data[0]]
        lines.append("| " + " | ".join(header) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        for row in self.data[1:]:
            cells = [cell or "" for cell in row]
            # Pad if necessary
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells[:len(header)]) + " |")

        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Check if table has meaningful content."""
        if not self.data or len(self.data) < 2:
            return True

        non_empty_cells = sum(
            1 for row in self.data for cell in row
            if cell and str(cell).strip() and str(cell).strip() not in ['', '---']
        )

        return non_empty_cells < 3

    def get_cell(self, row: int, col: int) -> Optional[str]:
        """Safely get cell value."""
        if 0 <= row < len(self.data) and 0 <= col < len(self.data[row]):
            return self.data[row][col]
        return None


@dataclass
class ExtractionResult:
    """Complete extraction result from a single extractor."""
    tables: List[ExtractedTable]
    page_num: int
    extractor_name: str
    total_time_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableExtractor(ABC):
    """
    Abstract base class for table extractors.

    All concrete extractors (PdfPlumber, Camelot, PyMuPDF, etc.) must
    implement this interface to ensure consistency and enable consensus-based
    extraction.

    Example:
        >>> extractor = PdfPlumberExtractor()
        >>> if extractor.is_available():
        ...     result = extractor.extract_tables("document.pdf", page_num=5)
        ...     for table in result.tables:
        ...         print(f"Table {table.table_index}: {table.rows}x{table.columns}")
    """

    def __init__(self, name: str):
        """Initialize extractor with name."""
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if extractor dependencies are available.

        Returns:
            True if extractor can be used, False otherwise
        """
        pass

    @abstractmethod
    def extract_tables(
        self,
        pdf_path: str | Path,
        page_num: int,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract tables from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            **kwargs: Extractor-specific options

        Returns:
            ExtractionResult with extracted tables and metadata
        """
        pass

    def extract_all_tables(
        self,
        pdf_path: str | Path,
        **kwargs
    ) -> Dict[int, ExtractionResult]:
        """
        Extract tables from all pages in PDF.

        Args:
            pdf_path: Path to PDF file
            **kwargs: Extractor-specific options

        Returns:
            Dictionary mapping page numbers to extraction results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Get page count
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)

        results = {}
        for page_num in range(page_count):
            try:
                result = self.extract_tables(pdf_path, page_num, **kwargs)
                if result.tables:  # Only include pages with tables
                    results[page_num] = result
            except Exception as e:
                self.logger.error(f"Error extracting from page {page_num}: {e}")
                results[page_num] = ExtractionResult(
                    tables=[],
                    page_num=page_num,
                    extractor_name=self.name,
                    total_time_ms=0.0,
                    success=False,
                    error=str(e)
                )

        return results

    def validate_table(self, table: ExtractedTable) -> ExtractedTable:
        """
        Perform basic validation and populate issues/warnings.

        Args:
            table: Extracted table to validate

        Returns:
            Table with updated issues and warnings
        """
        # Check for empty table
        if table.is_empty():
            table.issues.append("Table appears to be empty or has insufficient content")

        # Check sparsity
        if table.sparsity > 0.7:
            table.warnings.append(f"High sparsity ({table.sparsity:.1%}): many empty cells")

        # Check column consistency
        if table.data:
            col_counts = [len(row) for row in table.data]
            if len(set(col_counts)) > 1:
                table.issues.append(
                    f"Inconsistent column counts: {set(col_counts)} "
                    "(possible merged cells or extraction error)"
                )

        # Check for suspiciously small tables
        if table.rows < 2:
            table.issues.append("Table has less than 2 rows (may not be a real table)")

        if table.columns < 2:
            table.issues.append("Table has less than 2 columns (may not be a real table)")

        return table


class PdfPlumberExtractor(TableExtractor):
    """
    Table extractor using pdfplumber library.

    Supports multiple extraction strategies:
    - Standard: Default pdfplumber extraction
    - Lines: Line-based extraction (for tables with visible borders)
    - Text: Text-based extraction (for borderless tables)
    - Hybrid: Try all strategies and return best result

    This is the current production extractor, now wrapped in the new interface.
    """

    def __init__(self, default_strategy: ExtractionStrategy = ExtractionStrategy.HYBRID):
        """
        Initialize pdfplumber extractor.

        Args:
            default_strategy: Default extraction strategy to use
        """
        super().__init__("pdfplumber")
        self.default_strategy = default_strategy

    def is_available(self) -> bool:
        """Check if pdfplumber is available."""
        try:
            import pdfplumber
            return True
        except ImportError:
            return False

    def extract_tables(
        self,
        pdf_path: str | Path,
        page_num: int,
        strategy: Optional[ExtractionStrategy] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract tables from a page using pdfplumber.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            strategy: Extraction strategy (or use default)
            **kwargs: Additional pdfplumber settings

        Returns:
            ExtractionResult with extracted tables
        """
        import time

        start_time = time.time()
        strategy = strategy or self.default_strategy

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return ExtractionResult(
                        tables=[],
                        page_num=page_num,
                        extractor_name=self.name,
                        total_time_ms=0.0,
                        success=False,
                        error=f"Page {page_num} out of range (PDF has {len(pdf.pages)} pages)"
                    )

                page = pdf.pages[page_num]

                # Extract based on strategy
                if strategy == ExtractionStrategy.HYBRID:
                    raw_tables = self._extract_hybrid(page, **kwargs)
                elif strategy == ExtractionStrategy.LINES:
                    raw_tables = self._extract_lines(page, **kwargs)
                elif strategy == ExtractionStrategy.TEXT:
                    raw_tables = self._extract_text(page, **kwargs)
                else:  # STANDARD
                    raw_tables = page.extract_tables(**kwargs)

                # Convert to ExtractedTable objects
                tables = []
                for idx, raw_table in enumerate(raw_tables or []):
                    if raw_table and len(raw_table) >= 2:
                        table = ExtractedTable(
                            data=raw_table,
                            page_num=page_num,
                            table_index=idx,
                            extractor_name=self.name,
                            strategy=strategy,
                            extraction_time_ms=(time.time() - start_time) * 1000
                        )

                        # Add caption if found
                        table.caption = self._find_caption(page, idx)

                        # Validate
                        table = self.validate_table(table)

                        if not table.is_empty():
                            tables.append(table)

                elapsed_ms = (time.time() - start_time) * 1000

                return ExtractionResult(
                    tables=tables,
                    page_num=page_num,
                    extractor_name=self.name,
                    total_time_ms=elapsed_ms,
                    success=True,
                    metadata={"strategy": strategy.value}
                )

        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_num}: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ExtractionResult(
                tables=[],
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )

    def _extract_hybrid(self, page, **kwargs) -> List[List[List[str]]]:
        """Try multiple strategies and return best result."""
        # Strategy 1: Standard
        tables = page.extract_tables(**kwargs)
        if tables and any(self._is_non_empty(t) for t in tables):
            return tables

        # Strategy 2: Lines
        tables = self._extract_lines(page, **kwargs)
        if tables and any(self._is_non_empty(t) for t in tables):
            return tables

        # Strategy 3: Text
        return self._extract_text(page, **kwargs)

    def _extract_lines(self, page, **kwargs) -> List[List[List[str]]]:
        """Extract using line-based strategy."""
        settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            **kwargs
        }
        return page.extract_tables(table_settings=settings)

    def _extract_text(self, page, **kwargs) -> List[List[List[str]]]:
        """Extract using text-based strategy."""
        settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            **kwargs
        }
        return page.extract_tables(table_settings=settings)

    def _is_non_empty(self, table: List[List[str]]) -> bool:
        """Check if raw table has content."""
        if not table or len(table) < 2:
            return False

        non_empty_cells = sum(
            1 for row in table for cell in row
            if cell and str(cell).strip()
        )
        return non_empty_cells >= 3

    def _find_caption(self, page, table_index: int) -> Optional[str]:
        """Find table caption using regex patterns."""
        import re

        text = page.extract_text()
        if not text:
            return None

        patterns = [
            rf'Table\s+{table_index + 1}[:\.]?\s+(.+?)(?:\n|$)',
            rf'TABLE\s+{table_index + 1}[:\.]?\s+(.+?)(?:\n|$)',
            rf'Table\s+(\d+-\d+)[:\.]?\s+(.+?)(?:\n|$)',
            rf'TABLE\s+(\d+-\d+)[:\.]?\s+(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                caption = match.group(match.lastindex).strip()
                if len(caption) > 3:
                    return caption[:200]

        return None


class CamelotExtractor(TableExtractor):
    """
    Table extractor using Camelot library.

    Camelot specializes in extracting tables from PDFs and offers two modes:
    - Lattice: For tables with visible grid lines (best accuracy)
    - Stream: For tables without lines (uses text positioning)

    Camelot often outperforms pdfplumber on tables with clear borders.

    Note: Requires camelot-py[cv] and opencv-python packages.
    """

    def __init__(self, default_mode: str = "lattice"):
        """
        Initialize Camelot extractor.

        Args:
            default_mode: Default extraction mode ("lattice" or "stream")
        """
        super().__init__("camelot")
        self.default_mode = default_mode

    def is_available(self) -> bool:
        """Check if Camelot is available."""
        try:
            import camelot
            return True
        except ImportError:
            return False

    def extract_tables(
        self,
        pdf_path: str | Path,
        page_num: int,
        mode: Optional[str] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract tables using Camelot.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            mode: Extraction mode ("lattice" or "stream", or use default)
            **kwargs: Additional Camelot options

        Returns:
            ExtractionResult with extracted tables
        """
        if not self.is_available():
            return ExtractionResult(
                tables=[],
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=0.0,
                success=False,
                error="Camelot not installed. Install with: pip install camelot-py[cv] opencv-python"
            )

        import time
        import camelot

        start_time = time.time()
        mode = mode or self.default_mode

        try:
            # Camelot uses 1-indexed pages
            pages_str = str(page_num + 1)

            # Extract tables
            raw_tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages_str,
                flavor=mode,
                **kwargs
            )

            # Convert to ExtractedTable objects
            tables = []
            for idx, camelot_table in enumerate(raw_tables):
                # Get data as list of lists
                data = camelot_table.df.values.tolist()

                # Add header row
                header = camelot_table.df.columns.tolist()
                full_data = [header] + data

                # Get accuracy score from Camelot
                accuracy = camelot_table.accuracy if hasattr(camelot_table, 'accuracy') else 0.0

                table = ExtractedTable(
                    data=full_data,
                    page_num=page_num,
                    table_index=idx,
                    extractor_name=self.name,
                    strategy=ExtractionStrategy.LATTICE if mode == "lattice" else ExtractionStrategy.STREAM,
                    extraction_time_ms=(time.time() - start_time) * 1000,
                    confidence=accuracy / 100.0  # Camelot returns 0-100, we use 0-1
                )

                # Get bounding box if available
                if hasattr(camelot_table, '_bbox'):
                    table.bbox = camelot_table._bbox

                # Determine if has borders (lattice mode suggests borders)
                table.has_borders = (mode == "lattice")

                # Validate
                table = self.validate_table(table)

                if not table.is_empty():
                    tables.append(table)

            elapsed_ms = (time.time() - start_time) * 1000

            return ExtractionResult(
                tables=tables,
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=elapsed_ms,
                success=True,
                metadata={"mode": mode, "raw_count": len(raw_tables)}
            )

        except Exception as e:
            self.logger.error(f"Error extracting tables with Camelot from page {page_num}: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ExtractionResult(
                tables=[],
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )


class PyMuPDFExtractor(TableExtractor):
    """
    Table extractor using PyMuPDF (fitz) library.

    PyMuPDF is fast and already a dependency of mchp-mcp-core. It extracts
    tables by analyzing text blocks and their positioning. Works well for
    simple, well-formatted tables.

    Best for: Simple tables, speed-critical applications
    """

    def __init__(self):
        """Initialize PyMuPDF extractor."""
        super().__init__("pymupdf")

    def is_available(self) -> bool:
        """Check if PyMuPDF is available."""
        try:
            import fitz
            return True
        except ImportError:
            return False

    def extract_tables(
        self,
        pdf_path: str | Path,
        page_num: int,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract tables using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            **kwargs: Additional options

        Returns:
            ExtractionResult with extracted tables
        """
        if not self.is_available():
            return ExtractionResult(
                tables=[],
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=0.0,
                success=False,
                error="PyMuPDF not installed (should be available as dependency)"
            )

        import time
        import fitz

        start_time = time.time()

        try:
            doc = fitz.open(pdf_path)

            if page_num >= len(doc):
                doc.close()
                return ExtractionResult(
                    tables=[],
                    page_num=page_num,
                    extractor_name=self.name,
                    total_time_ms=0.0,
                    success=False,
                    error=f"Page {page_num} out of range (PDF has {len(doc)} pages)"
                )

            page = doc[page_num]

            # Find tables using PyMuPDF's table detection
            table_finder = page.find_tables()

            tables = []
            if table_finder and hasattr(table_finder, 'tables'):
                for idx, table_obj in enumerate(table_finder.tables):
                    # Extract table data
                    table_data = table_obj.extract()

                    if table_data and len(table_data) >= 2:
                        table = ExtractedTable(
                            data=table_data,
                            page_num=page_num,
                            table_index=idx,
                            extractor_name=self.name,
                            strategy=ExtractionStrategy.STANDARD,
                            extraction_time_ms=(time.time() - start_time) * 1000
                        )

                        # Get bounding box
                        if hasattr(table_obj, 'bbox'):
                            table.bbox = table_obj.bbox

                        # Validate
                        table = self.validate_table(table)

                        if not table.is_empty():
                            tables.append(table)

            doc.close()

            elapsed_ms = (time.time() - start_time) * 1000

            return ExtractionResult(
                tables=tables,
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=elapsed_ms,
                success=True,
                metadata={"raw_count": len(table_finder.tables) if table_finder and hasattr(table_finder, 'tables') else 0}
            )

        except Exception as e:
            self.logger.error(f"Error extracting tables with PyMuPDF from page {page_num}: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ExtractionResult(
                tables=[],
                page_num=page_num,
                extractor_name=self.name,
                total_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )


__all__ = [
    "ExtractionStrategy",
    "TableComplexity",
    "TableCell",
    "ExtractedTable",
    "ExtractionResult",
    "TableExtractor",
    "PdfPlumberExtractor",
    "CamelotExtractor",
    "PyMuPDFExtractor",
]
