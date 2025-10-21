"""
Table merging strategies for multi-page tables.

Merges tables detected across multiple pages into single unified tables.
Handles vertical (top-to-bottom) and horizontal (left-to-right) continuations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from mchp_mcp_core.extractors.table_extractors import ExtractedTable, TableComplexity
from mchp_mcp_core.extractors.table_multipage import ContinuationType, TableSpan
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MergeResult:
    """Result of table merge operation."""
    merged_table: ExtractedTable
    source_pages: List[int]
    merge_type: ContinuationType
    rows_merged: int
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class TableMerger:
    """
    Merges multi-page tables into single unified tables.

    Supports:
    - Vertical merging (top-to-bottom continuation)
    - Horizontal merging (left-to-right continuation)
    - Header deduplication
    - Column alignment

    Example:
        >>> merger = TableMerger()
        >>> tables = [table_page21, table_page22]  # 32 rows + 57 rows
        >>> result = merger.merge_vertical(tables, (21, 22))
        >>> print(result.merged_table.rows)  # 89 rows
    """

    def __init__(
        self,
        preserve_headers: bool = True,
        align_columns: bool = True
    ):
        """
        Initialize merger.

        Args:
            preserve_headers: Keep first table's header, remove from continuations
            align_columns: Attempt to align columns when column counts differ
        """
        self.preserve_headers = preserve_headers
        self.align_columns = align_columns

    def merge_span(
        self,
        tables: List[ExtractedTable],
        span: TableSpan
    ) -> MergeResult:
        """
        Merge tables according to span information.

        Args:
            tables: List of tables to merge (one per page)
            span: TableSpan with continuation information

        Returns:
            MergeResult with merged table
        """
        if not tables:
            raise ValueError("No tables provided for merging")

        if len(tables) == 1:
            logger.warning("Only one table provided, nothing to merge")
            return MergeResult(
                merged_table=tables[0],
                source_pages=[span.start_page],
                merge_type=span.continuation_type,
                rows_merged=0
            )

        if span.continuation_type == ContinuationType.VERTICAL:
            return self.merge_vertical(tables, (span.start_page, span.end_page))
        elif span.continuation_type == ContinuationType.HORIZONTAL:
            return self.merge_horizontal(tables, (span.start_page, span.end_page))
        else:
            raise ValueError(f"Unknown continuation type: {span.continuation_type}")

    def merge_vertical(
        self,
        tables: List[ExtractedTable],
        page_range: tuple[int, int]
    ) -> MergeResult:
        """
        Merge tables vertically (top-to-bottom).

        Process:
        1. Use first table's header
        2. Remove headers from continuation tables
        3. Concatenate all data rows
        4. Align columns if needed

        Args:
            tables: Tables to merge (in page order)
            page_range: (start_page, end_page)

        Returns:
            MergeResult with merged table
        """
        if not tables:
            raise ValueError("No tables to merge")

        issues = []
        all_rows = []

        # Start with first table's header
        first_table = tables[0]
        if first_table.data:
            all_rows.append(first_table.data[0])  # Header

            # Add data rows from first table (skip header)
            all_rows.extend(first_table.data[1:])

        rows_from_first = len(all_rows)

        # Process continuation tables
        for i, table in enumerate(tables[1:], start=1):
            if not table.data:
                issues.append(f"Table on page {page_range[0] + i} has no data")
                continue

            # Check if continuation has a header
            has_header = self._has_header(table)

            if has_header:
                # Skip header row, only take data
                continuation_rows = table.data[1:]
                issues.append(f"Removed duplicate header from page {page_range[0] + i}")
            else:
                # No header, take all rows
                continuation_rows = table.data

            # Handle column count mismatches
            if continuation_rows and len(continuation_rows[0]) != len(all_rows[0]):
                if self.align_columns:
                    continuation_rows = self._align_columns(
                        continuation_rows,
                        target_cols=len(all_rows[0])
                    )
                    issues.append(
                        f"Aligned columns on page {page_range[0] + i} "
                        f"({len(table.data[0])} → {len(all_rows[0])})"
                    )
                else:
                    issues.append(
                        f"Column count mismatch on page {page_range[0] + i}: "
                        f"{len(table.data[0])} vs {len(all_rows[0])}"
                    )

            all_rows.extend(continuation_rows)

        # Create merged table
        merged_table = ExtractedTable(
            data=all_rows,
            page_num=page_range[0],  # Start page
            table_index=first_table.table_index,
            confidence=first_table.confidence,  # Will be recomputed
            complexity=TableComplexity.COMPLEX,  # Multi-page = complex
        )

        rows_merged = len(all_rows) - rows_from_first

        logger.info(
            f"Vertically merged {len(tables)} tables: "
            f"{rows_from_first} + {rows_merged} = {len(all_rows)} rows"
        )

        return MergeResult(
            merged_table=merged_table,
            source_pages=list(range(page_range[0], page_range[1] + 1)),
            merge_type=ContinuationType.VERTICAL,
            rows_merged=rows_merged,
            issues=issues
        )

    def merge_horizontal(
        self,
        tables: List[ExtractedTable],
        page_range: tuple[int, int]
    ) -> MergeResult:
        """
        Merge tables horizontally (left-to-right).

        Process:
        1. Merge each row across tables
        2. Align rows if row counts differ
        3. Concatenate columns

        Args:
            tables: Tables to merge (in page order)
            page_range: (start_page, end_page)

        Returns:
            MergeResult with merged table
        """
        if not tables:
            raise ValueError("No tables to merge")

        issues = []

        # Find maximum row count
        max_rows = max(len(t.data) if t.data else 0 for t in tables)

        if max_rows == 0:
            issues.append("All tables are empty")
            return MergeResult(
                merged_table=ExtractedTable(data=[], page_num=page_range[0], table_index=0),
                source_pages=list(range(page_range[0], page_range[1] + 1)),
                merge_type=ContinuationType.HORIZONTAL,
                rows_merged=0,
                issues=issues
            )

        # Merge row by row
        merged_rows = []
        for row_idx in range(max_rows):
            merged_row = []

            for table in tables:
                if table.data and row_idx < len(table.data):
                    merged_row.extend(table.data[row_idx])
                else:
                    # Pad with empty cells if table is shorter
                    if tables[0].data and row_idx < len(tables[0].data):
                        # Use same column count as first table
                        merged_row.extend([""] * (len(tables[0].data[0]) if tables[0].data[0] else 0))

            merged_rows.append(merged_row)

        # Create merged table
        merged_table = ExtractedTable(
            data=merged_rows,
            page_num=page_range[0],
            table_index=tables[0].table_index,
            confidence=tables[0].confidence,
            complexity=TableComplexity.COMPLEX,
        )

        cols_merged = sum(t.columns for t in tables) - tables[0].columns

        logger.info(
            f"Horizontally merged {len(tables)} tables: "
            f"{max_rows} rows, {merged_table.columns} columns"
        )

        return MergeResult(
            merged_table=merged_table,
            source_pages=list(range(page_range[0], page_range[1] + 1)),
            merge_type=ContinuationType.HORIZONTAL,
            rows_merged=cols_merged,
            issues=issues
        )

    def _has_header(self, table: ExtractedTable) -> bool:
        """
        Check if table has a header row.

        Indicators:
        - Non-empty first row
        - First row differs from second row
        - First row has mostly text (not numbers)
        """
        if not table.data or len(table.data) < 2:
            return False

        first_row = table.data[0]
        second_row = table.data[1] if len(table.data) > 1 else None

        # Check if first row is empty
        non_empty_cells = sum(1 for cell in first_row if cell and str(cell).strip())

        if non_empty_cells == 0:
            return False  # Empty = no header

        # Check if first row differs from second row
        if second_row:
            differences = sum(
                1 for c1, c2 in zip(first_row, second_row)
                if str(c1).strip() != str(c2).strip()
            )

            if differences < len(first_row) / 2:
                return False  # Too similar = probably continuation, not header

        return True

    def _align_columns(
        self,
        rows: List[List],
        target_cols: int
    ) -> List[List]:
        """
        Align rows to have target column count.

        Args:
            rows: Rows to align
            target_cols: Desired column count

        Returns:
            Aligned rows
        """
        aligned_rows = []

        for row in rows:
            current_cols = len(row)

            if current_cols == target_cols:
                aligned_rows.append(row)
            elif current_cols < target_cols:
                # Pad with empty cells
                padded_row = row + [""] * (target_cols - current_cols)
                aligned_rows.append(padded_row)
            else:
                # Truncate (with warning logged elsewhere)
                aligned_rows.append(row[:target_cols])

        return aligned_rows


__all__ = [
    "MergeResult",
    "TableMerger",
]
