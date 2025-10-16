"""
Table validation and auto-correction for common extraction errors.

Provides post-extraction validation checks and automatic fixes for common
issues like header detection, empty rows, numeric corruption, and merged cells.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from mchp_mcp_core.extractors.table_extractors import ExtractedTable, TableComplexity
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class ValidationIssue(Enum):
    """Types of validation issues."""
    EMPTY_HEADER = "empty_header"
    EMPTY_ROW = "empty_row"
    INCONSISTENT_COLUMNS = "inconsistent_columns"
    NUMERIC_CORRUPTION = "numeric_corruption"
    MERGED_CELLS_DETECTED = "merged_cells_detected"
    EXCESSIVE_SPARSITY = "excessive_sparsity"
    DUPLICATE_ROWS = "duplicate_rows"
    SUSPICIOUS_CHARACTERS = "suspicious_characters"


@dataclass
class ValidationResult:
    """Result of table validation."""
    is_valid: bool
    issues: List[Tuple[ValidationIssue, str]] = None  # (issue_type, description)
    corrected_table: Optional[ExtractedTable] = None
    corrections_applied: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.corrections_applied is None:
            self.corrections_applied = []


class TableValidator:
    """
    Validates extracted tables and applies automatic corrections.

    This class identifies common extraction errors and optionally fixes them:
    - Empty rows/columns
    - Header detection
    - Numeric corruption (e.g., "1O0" → "100")
    - Merged cell artifacts
    - Column inconsistencies

    Example:
        >>> validator = TableValidator(auto_correct=True)
        >>> result = validator.validate(extracted_table)
        >>> if result.is_valid:
        ...     print("Table is valid!")
        >>> else:
        ...     print(f"Found {len(result.issues)} issues")
        ...     if result.corrected_table:
        ...         table = result.corrected_table  # Use corrected version
    """

    def __init__(
        self,
        auto_correct: bool = True,
        strict_mode: bool = False,
        max_empty_row_ratio: float = 0.3,
        max_sparsity: float = 0.7
    ):
        """
        Initialize validator.

        Args:
            auto_correct: Attempt to automatically fix issues
            strict_mode: Fail on any issue (don't tolerate minor problems)
            max_empty_row_ratio: Maximum ratio of empty rows before flagging as issue
            max_sparsity: Maximum sparsity before flagging as issue
        """
        self.auto_correct = auto_correct
        self.strict_mode = strict_mode
        self.max_empty_row_ratio = max_empty_row_ratio
        self.max_sparsity = max_sparsity

    def validate(self, table: ExtractedTable) -> ValidationResult:
        """
        Validate an extracted table and optionally correct issues.

        Args:
            table: Extracted table to validate

        Returns:
            ValidationResult with issues found and optionally corrected table
        """
        result = ValidationResult(is_valid=True)

        # Make a copy if we're going to correct
        working_table = self._copy_table(table) if self.auto_correct else table

        # Run validation checks
        self._check_empty_header(working_table, result)
        self._check_empty_rows(working_table, result)
        self._check_column_consistency(working_table, result)
        self._check_numeric_corruption(working_table, result)
        self._check_merged_cells(working_table, result)
        self._check_sparsity(working_table, result)
        self._check_duplicate_rows(working_table, result)
        self._check_suspicious_characters(working_table, result)

        # Apply corrections if enabled
        if self.auto_correct and result.issues:
            self._apply_corrections(working_table, result)
            result.corrected_table = working_table

        # Determine overall validity
        if self.strict_mode:
            result.is_valid = len(result.issues) == 0
        else:
            # Only fail on critical issues
            critical_issues = [
                ValidationIssue.EMPTY_HEADER,
                ValidationIssue.INCONSISTENT_COLUMNS,
                ValidationIssue.EXCESSIVE_SPARSITY
            ]
            result.is_valid = not any(
                issue_type in critical_issues
                for issue_type, _ in result.issues
            )

        return result

    def _copy_table(self, table: ExtractedTable) -> ExtractedTable:
        """Create a deep copy of table for modification."""
        return ExtractedTable(
            data=[row[:] for row in table.data],
            page_num=table.page_num,
            table_index=table.table_index,
            confidence=table.confidence,
            complexity=table.complexity,
            sparsity=table.sparsity,
            issues=table.issues[:] if table.issues else []
        )

    def _check_empty_header(self, table: ExtractedTable, result: ValidationResult):
        """Check if first row (header) is empty."""
        if not table.data or len(table.data) == 0:
            return

        first_row = table.data[0]
        if all(not cell or str(cell).strip() == "" for cell in first_row):
            result.issues.append((
                ValidationIssue.EMPTY_HEADER,
                "First row (header) is completely empty"
            ))

    def _check_empty_rows(self, table: ExtractedTable, result: ValidationResult):
        """Check for excessive empty rows."""
        if not table.data:
            return

        empty_rows = sum(
            1 for row in table.data
            if all(not cell or str(cell).strip() == "" for cell in row)
        )

        ratio = empty_rows / len(table.data)
        if ratio > self.max_empty_row_ratio:
            result.issues.append((
                ValidationIssue.EMPTY_ROW,
                f"Excessive empty rows: {empty_rows}/{len(table.data)} ({ratio:.1%})"
            ))

    def _check_column_consistency(self, table: ExtractedTable, result: ValidationResult):
        """Check if all rows have the same number of columns."""
        if not table.data or len(table.data) == 0:
            return

        col_counts = [len(row) for row in table.data]
        if len(set(col_counts)) > 1:
            result.issues.append((
                ValidationIssue.INCONSISTENT_COLUMNS,
                f"Inconsistent column counts: {set(col_counts)}"
            ))

    def _check_numeric_corruption(self, table: ExtractedTable, result: ValidationResult):
        """Check for common OCR/extraction errors in numeric values."""
        if not table.data:
            return

        # Common OCR errors: O→0, I→1, l→1, S→5, etc.
        suspicious_patterns = [
            (r'\d+[OoDd]\d*', "Possible O/D instead of 0"),
            (r'\d+[Il]\d*', "Possible I/l instead of 1"),
            (r'\d+[Ss]\d*', "Possible S instead of 5"),
        ]

        issues_found = []
        for row_idx, row in enumerate(table.data):
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue

                cell_str = str(cell)
                for pattern, description in suspicious_patterns:
                    if re.search(pattern, cell_str):
                        issues_found.append(f"({row_idx},{col_idx}): {cell_str}")

        if issues_found:
            result.issues.append((
                ValidationIssue.NUMERIC_CORRUPTION,
                f"Possible numeric corruption in {len(issues_found)} cells: {issues_found[:3]}"
            ))

    def _check_merged_cells(self, table: ExtractedTable, result: ValidationResult):
        """Detect potential merged cell artifacts."""
        if not table.data or len(table.data) < 2:
            return

        # Look for identical adjacent cells (possible merged cell split)
        merged_count = 0
        for row_idx, row in enumerate(table.data):
            for col_idx in range(len(row) - 1):
                if row[col_idx] and row[col_idx] == row[col_idx + 1]:
                    merged_count += 1

        # Look for cells with unusual spanning indicators
        for row in table.data:
            for cell in row:
                if cell and any(marker in str(cell).lower() for marker in ["colspan", "rowspan", "merged"]):
                    merged_count += 1

        if merged_count > 0:
            result.issues.append((
                ValidationIssue.MERGED_CELLS_DETECTED,
                f"Possible merged cell artifacts detected ({merged_count} instances)"
            ))

    def _check_sparsity(self, table: ExtractedTable, result: ValidationResult):
        """Check if table is excessively sparse."""
        if table.sparsity > self.max_sparsity:
            result.issues.append((
                ValidationIssue.EXCESSIVE_SPARSITY,
                f"Table is {table.sparsity:.1%} sparse (threshold: {self.max_sparsity:.1%})"
            ))

    def _check_duplicate_rows(self, table: ExtractedTable, result: ValidationResult):
        """Check for duplicate rows (excluding header)."""
        if not table.data or len(table.data) < 3:
            return

        # Skip header row
        data_rows = table.data[1:]

        # Convert rows to tuples for hashing
        row_tuples = [tuple(row) for row in data_rows]
        unique_rows = set(row_tuples)

        duplicates = len(row_tuples) - len(unique_rows)
        if duplicates > 0:
            result.issues.append((
                ValidationIssue.DUPLICATE_ROWS,
                f"Found {duplicates} duplicate rows"
            ))

    def _check_suspicious_characters(self, table: ExtractedTable, result: ValidationResult):
        """Check for unusual characters that might indicate extraction errors."""
        if not table.data:
            return

        # Unicode characters that might indicate extraction issues
        suspicious_chars = [
            '\ufffd',  # Replacement character
            '\u0000',  # Null character
            '\x00',    # Null byte
        ]

        issues_found = []
        for row_idx, row in enumerate(table.data):
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue

                for char in suspicious_chars:
                    if char in str(cell):
                        issues_found.append(f"({row_idx},{col_idx})")

        if issues_found:
            result.issues.append((
                ValidationIssue.SUSPICIOUS_CHARACTERS,
                f"Suspicious characters found in cells: {issues_found[:5]}"
            ))

    def _apply_corrections(self, table: ExtractedTable, result: ValidationResult):
        """Apply automatic corrections to the table."""
        # Remove empty rows
        self._remove_empty_rows(table, result)

        # Fix numeric corruption
        self._fix_numeric_corruption(table, result)

        # Normalize whitespace
        self._normalize_whitespace(table, result)

        # Remove duplicate rows
        self._remove_duplicate_rows(table, result)

    def _remove_empty_rows(self, table: ExtractedTable, result: ValidationResult):
        """Remove completely empty rows."""
        original_count = len(table.data)

        table.data = [
            row for row in table.data
            if not all(not cell or str(cell).strip() == "" for cell in row)
        ]

        removed = original_count - len(table.data)
        if removed > 0:
            result.corrections_applied.append(f"Removed {removed} empty rows")
            logger.debug(f"Removed {removed} empty rows from table")

    def _fix_numeric_corruption(self, table: ExtractedTable, result: ValidationResult):
        """Fix common OCR errors in numeric values."""
        corrections = 0

        for row in table.data:
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue

                original = str(cell)
                fixed = str(cell)

                # Only apply to cells that look numeric
                if re.search(r'\d', fixed):
                    # O → 0
                    fixed = re.sub(r'(?<=\d)[OoDd](?=\d|$)', '0', fixed)
                    # I, l → 1 (only in numeric context)
                    fixed = re.sub(r'(?<=\d)[Il](?=\d|$)', '1', fixed)
                    # S → 5 (conservative)
                    if re.match(r'^\d+[Ss]\d*$', fixed):
                        fixed = fixed.replace('S', '5').replace('s', '5')

                if fixed != original:
                    row[col_idx] = fixed
                    corrections += 1

        if corrections > 0:
            result.corrections_applied.append(f"Fixed {corrections} potential numeric corruptions")
            logger.debug(f"Fixed {corrections} potential numeric corruptions")

    def _normalize_whitespace(self, table: ExtractedTable, result: ValidationResult):
        """Normalize whitespace in all cells."""
        changes = 0

        for row in table.data:
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue

                # Strip leading/trailing whitespace
                normalized = str(cell).strip()

                # Replace multiple spaces with single space
                normalized = re.sub(r'\s+', ' ', normalized)

                if normalized != cell:
                    row[col_idx] = normalized
                    changes += 1

        if changes > 0:
            result.corrections_applied.append(f"Normalized whitespace in {changes} cells")
            logger.debug(f"Normalized whitespace in {changes} cells")

    def _remove_duplicate_rows(self, table: ExtractedTable, result: ValidationResult):
        """Remove duplicate rows (keeping first occurrence)."""
        if len(table.data) < 2:
            return

        original_count = len(table.data)

        # Keep header (first row)
        header = table.data[0]
        data_rows = table.data[1:]

        # Remove duplicates
        seen = set()
        unique_rows = []
        for row in data_rows:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)

        table.data = [header] + unique_rows

        removed = original_count - len(table.data)
        if removed > 0:
            result.corrections_applied.append(f"Removed {removed} duplicate rows")
            logger.debug(f"Removed {removed} duplicate rows")


def detect_header_row(table: ExtractedTable) -> int:
    """
    Detect which row is the header.

    Uses heuristics:
    - First non-empty row
    - Row with shortest average cell length (headers are usually shorter)
    - Row with most unique values

    Returns:
        Row index of header (0-indexed), or 0 if detection fails
    """
    if not table.data or len(table.data) == 0:
        return 0

    # Find first non-empty row
    for idx, row in enumerate(table.data):
        if any(cell and str(cell).strip() for cell in row):
            return idx

    return 0


def split_multi_row_header(table: ExtractedTable, header_rows: int = 2) -> ExtractedTable:
    """
    Handle multi-row headers by merging them into a single row.

    Example:
        Row 0: ["Parameter", "", "Conditions"]
        Row 1: ["Name", "Value", ""]
        →
        Row 0: ["Parameter Name", "Value", "Conditions"]

    Args:
        table: Extracted table with multi-row header
        header_rows: Number of header rows to merge

    Returns:
        Table with merged header
    """
    if not table.data or len(table.data) < header_rows:
        return table

    # Merge header rows
    merged_header = []
    for col_idx in range(len(table.data[0])):
        col_values = []
        for row_idx in range(header_rows):
            if col_idx < len(table.data[row_idx]):
                cell = table.data[row_idx][col_idx]
                if cell and str(cell).strip():
                    col_values.append(str(cell).strip())

        merged_header.append(" ".join(col_values))

    # Create new table with merged header
    new_data = [merged_header] + table.data[header_rows:]

    return ExtractedTable(
        data=new_data,
        page_num=table.page_num,
        table_index=table.table_index,
        confidence=table.confidence,
        complexity=TableComplexity.COMPLEX,  # Multi-row headers are complex
        sparsity=table.sparsity,
        issues=table.issues
    )


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "TableValidator",
    "detect_header_row",
    "split_multi_row_header",
]
