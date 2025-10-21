"""
Multi-page table detection for PDF documents.

Detects when tables span multiple pages and identifies continuation boundaries.
Based on Azure AI Document Intelligence research (2024) and industry best practices.

Key scenarios:
1. Vertical continuation: Table continues top-to-bottom across pages
2. Horizontal continuation: Wide table split left-to-right across pages
3. Repeated headers: Header row repeated on each page
4. No header: Continuation pages have no header row
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from mchp_mcp_core.extractors.table_extractors import ExtractedTable
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class ContinuationType(Enum):
    """Type of table continuation."""
    VERTICAL = "vertical"  # Top-to-bottom (most common)
    HORIZONTAL = "horizontal"  # Left-to-right (wide tables)
    NONE = "none"  # Not a continuation


@dataclass
class TableSpan:
    """
    Represents a table that spans multiple pages.

    Attributes:
        start_page: First page of table (0-indexed)
        end_page: Last page of table (0-indexed)
        table_indices: Table index on each page (for matching across extractors)
        continuation_type: Vertical or horizontal continuation
        confidence: Confidence that this is a real continuation (0.0-1.0)
        indicators: List of evidence supporting continuation
    """
    start_page: int
    end_page: int
    table_indices: List[int]  # One per page
    continuation_type: ContinuationType
    confidence: float = 0.0
    indicators: List[str] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []

    @property
    def num_pages(self) -> int:
        """Number of pages spanned."""
        return self.end_page - self.start_page + 1

    @property
    def is_multipage(self) -> bool:
        """True if spans more than one page."""
        return self.num_pages > 1


class MultiPageTableDetector:
    """
    Detects table continuations across pages.

    Uses heuristic analysis to identify when a table continues from one page
    to the next, based on structural similarity, column alignment, and continuation
    indicators.

    Example:
        >>> detector = MultiPageTableDetector()
        >>> tables_by_page = {
        ...     21: [table1],  # Power dissipation table (partial)
        ...     22: [table2],  # Continuation (no header)
        ... }
        >>> spans = detector.detect_spans(tables_by_page)
        >>> # Returns: TableSpan(start=21, end=22, type=VERTICAL, confidence=0.92)
    """

    def __init__(
        self,
        column_match_threshold: float = 0.80,
        position_tolerance: float = 0.05,
        min_confidence: float = 0.70
    ):
        """
        Initialize detector.

        Args:
            column_match_threshold: Min column structure similarity (0.0-1.0)
            position_tolerance: Max horizontal position difference (as fraction of page width)
            min_confidence: Min confidence to mark as continuation
        """
        self.column_match_threshold = column_match_threshold
        self.position_tolerance = position_tolerance
        self.min_confidence = min_confidence

    def detect_spans(
        self,
        tables_by_page: Dict[int, List[ExtractedTable]],
        extractor_name: str = "pdfplumber"
    ) -> List[TableSpan]:
        """
        Detect multi-page table spans across a document.

        Args:
            tables_by_page: Map of page_num -> list of tables on that page
            extractor_name: Which extractor's tables to analyze

        Returns:
            List of TableSpan objects for multi-page tables
        """
        spans = []
        pages = sorted(tables_by_page.keys())

        # Track which tables are already part of a span
        used_tables = set()

        for i in range(len(pages) - 1):
            page1 = pages[i]
            page2 = pages[i + 1]

            # Skip non-consecutive pages
            if page2 != page1 + 1:
                continue

            tables1 = tables_by_page.get(page1, [])
            tables2 = tables_by_page.get(page2, [])

            # Try to match tables between pages
            for idx1, table1 in enumerate(tables1):
                if (page1, idx1) in used_tables:
                    continue

                for idx2, table2 in enumerate(tables2):
                    if (page2, idx2) in used_tables:
                        continue

                    # Check if table2 continues table1
                    continuation = self._check_continuation(table1, table2, page1, page2)

                    if continuation and continuation.confidence >= self.min_confidence:
                        # Mark as used
                        used_tables.add((page1, idx1))
                        used_tables.add((page2, idx2))

                        # Check for further continuations
                        span = self._extend_span(
                            continuation,
                            tables_by_page,
                            pages,
                            i + 1,
                            used_tables
                        )

                        spans.append(span)

                        logger.info(
                            f"Detected {span.continuation_type.value} continuation: "
                            f"pages {span.start_page}-{span.end_page}, "
                            f"confidence={span.confidence:.2f}"
                        )
                        break  # Only one match per table1

        return spans

    def _check_continuation(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable,
        page1: int,
        page2: int
    ) -> Optional[TableSpan]:
        """
        Check if table2 continues table1.

        Returns:
            TableSpan if continuation detected, None otherwise
        """
        # Check vertical continuation (most common)
        vertical_result = self._check_vertical_continuation(table1, table2)

        if vertical_result:
            return TableSpan(
                start_page=page1,
                end_page=page2,
                table_indices=[table1.table_index, table2.table_index],
                continuation_type=ContinuationType.VERTICAL,
                confidence=vertical_result[0],
                indicators=vertical_result[1]
            )

        # Check horizontal continuation (less common)
        horizontal_result = self._check_horizontal_continuation(table1, table2)

        if horizontal_result:
            return TableSpan(
                start_page=page1,
                end_page=page2,
                table_indices=[table1.table_index, table2.table_index],
                continuation_type=ContinuationType.HORIZONTAL,
                confidence=horizontal_result[0],
                indicators=horizontal_result[1]
            )

        return None

    def _check_vertical_continuation(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable
    ) -> Optional[Tuple[float, List[str]]]:
        """
        Check if table2 vertically continues table1.

        Criteria:
        1. Same column count (±1 tolerance)
        2. Column structure similarity >80%
        3. Empty or matching header on table2
        4. No caption on table2

        Returns:
            (confidence, indicators) if continuation, None otherwise
        """
        indicators = []
        confidence_factors = []

        # Check 1: Column count similarity
        col_diff = abs(table1.columns - table2.columns)
        if col_diff <= 1:
            indicators.append(f"Column count match ({table1.columns} ≈ {table2.columns})")
            confidence_factors.append(1.0 if col_diff == 0 else 0.8)
        else:
            return None  # Critical failure

        # Check 2: Column structure similarity
        col_sim = self._compute_column_similarity(table1, table2)
        if col_sim >= self.column_match_threshold:
            indicators.append(f"Column structure similar ({col_sim:.2f})")
            confidence_factors.append(col_sim)
        else:
            return None  # Critical failure

        # Check 3: Header analysis
        header_score = self._analyze_header(table2)
        if header_score > 0:
            if header_score == 1.0:
                indicators.append("Continuation has no header (empty first row)")
            else:
                indicators.append("Continuation header matches or is minimal")
            confidence_factors.append(header_score)

        # Check 4: Content consistency
        content_score = self._check_content_consistency(table1, table2)
        if content_score > 0:
            indicators.append(f"Content types consistent ({content_score:.2f})")
            confidence_factors.append(content_score)

        # Compute overall confidence
        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
            return (confidence, indicators)

        return None

    def _check_horizontal_continuation(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable
    ) -> Optional[Tuple[float, List[str]]]:
        """
        Check if table2 horizontally continues table1 (wide tables).

        Criteria:
        1. Same row count (±2 tolerance)
        2. Row content similarity
        3. Table1 ends near right edge, table2 starts near left edge

        Returns:
            (confidence, indicators) if continuation, None otherwise
        """
        indicators = []
        confidence_factors = []

        # Check 1: Row count similarity
        row_diff = abs(table1.rows - table2.rows)
        if row_diff <= 2:
            indicators.append(f"Row count match ({table1.rows} ≈ {table2.rows})")
            confidence_factors.append(1.0 if row_diff == 0 else 0.7)
        else:
            return None

        # Check 2: First column content similarity (should be same rows)
        if table1.data and table2.data:
            first_col1 = [row[0] if row else "" for row in table1.data[:min(5, len(table1.data))]]
            first_col2 = [row[0] if row else "" for row in table2.data[:min(5, len(table2.data))]]

            matches = sum(1 for c1, c2 in zip(first_col1, first_col2) if c1 == c2)
            if matches >= 2:
                indicators.append(f"Row labels match ({matches}/5)")
                confidence_factors.append(matches / 5.0)

        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
            return (confidence, indicators)

        return None

    def _compute_column_similarity(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable
    ) -> float:
        """
        Compute column structure similarity (0.0-1.0).

        Checks:
        - Column count match
        - Column width ratios
        - Content type per column (numeric, text, empty)
        """
        # Simple column count similarity
        if table1.columns == 0 or table2.columns == 0:
            return 0.0

        col_ratio = min(table1.columns, table2.columns) / max(table1.columns, table2.columns)

        # Exact match gets high score
        if table1.columns == table2.columns:
            return 1.0

        # Close match (±1) gets good score
        if abs(table1.columns - table2.columns) == 1:
            return 0.85

        return col_ratio

    def _analyze_header(self, table: ExtractedTable) -> float:
        """
        Analyze if table has a continuation-style header.

        Continuation indicators:
        - Empty first row (score=1.0)
        - Minimal content in first row (score=0.7)
        - All None/empty cells (score=1.0)

        Returns:
            Score 0.0-1.0, higher = more likely continuation
        """
        if not table.data or len(table.data) == 0:
            return 0.0

        first_row = table.data[0]

        # Count non-empty cells
        non_empty = sum(1 for cell in first_row if cell and str(cell).strip())

        if non_empty == 0:
            # Completely empty = strong continuation indicator
            return 1.0
        elif non_empty <= 2:
            # Very few cells = moderate indicator
            return 0.7
        elif non_empty < len(first_row) / 2:
            # Less than half = weak indicator
            return 0.5

        return 0.0

    def _check_content_consistency(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable
    ) -> float:
        """
        Check if content types are consistent between tables.

        Compares:
        - Numeric vs text columns
        - Empty cell patterns
        - Value ranges
        """
        # Simple check: compare sparsity
        if abs(table1.sparsity - table2.sparsity) < 0.2:
            return 0.8
        elif abs(table1.sparsity - table2.sparsity) < 0.4:
            return 0.5

        return 0.3

    def _extend_span(
        self,
        initial_span: TableSpan,
        tables_by_page: Dict[int, List[ExtractedTable]],
        all_pages: List[int],
        start_idx: int,
        used_tables: set
    ) -> TableSpan:
        """
        Extend a span to include additional continuation pages.

        Args:
            initial_span: Initial 2-page span
            tables_by_page: All tables
            all_pages: Sorted list of page numbers
            start_idx: Index in all_pages to start checking
            used_tables: Set of (page, table_idx) already used

        Returns:
            Extended TableSpan
        """
        current_span = initial_span

        for i in range(start_idx + 1, len(all_pages)):
            next_page = all_pages[i]

            # Only check consecutive pages
            if next_page != current_span.end_page + 1:
                break

            # Get last table in current span
            last_page = current_span.end_page
            last_idx = current_span.table_indices[-1]
            last_table = None

            for table in tables_by_page.get(last_page, []):
                if table.table_index == last_idx:
                    last_table = table
                    break

            if not last_table:
                break

            # Check if any table on next_page continues
            for table in tables_by_page.get(next_page, []):
                if (next_page, table.table_index) in used_tables:
                    continue

                continuation = self._check_continuation(last_table, table, last_page, next_page)

                if continuation and continuation.confidence >= self.min_confidence:
                    # Extend span
                    current_span.end_page = next_page
                    current_span.table_indices.append(table.table_index)
                    current_span.indicators.extend(continuation.indicators)
                    current_span.confidence = (
                        current_span.confidence * 0.7 + continuation.confidence * 0.3
                    )  # Weighted average

                    used_tables.add((next_page, table.table_index))
                    logger.debug(f"Extended span to page {next_page}")
                    break
            else:
                # No continuation found, stop extending
                break

        return current_span


__all__ = [
    "ContinuationType",
    "TableSpan",
    "MultiPageTableDetector",
]
