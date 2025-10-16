"""
Table detection and validation to filter false positives.

Provides pre-extraction validation to distinguish real tables from layout artifacts
like headers, footers, margins, and text alignment patterns that extractors might
misinterpret as tables.

Based on industry research showing that 40-60% of low-confidence extractions are
false positives (layout artifacts, not semantic tables).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from mchp_mcp_core.extractors.table_extractors import ExtractedTable
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class DetectionIssue(Enum):
    """Types of table detection issues."""
    TOO_SMALL = "too_small"  # < min rows/columns
    TOO_SPARSE = "too_sparse"  # > max empty cell ratio
    NONSENSE_HEADER = "nonsense_header"  # Numeric indices, empty cells
    PAGE_ARTIFACT = "page_artifact"  # Header/footer/margin
    BROKEN_TEXT = "broken_text"  # Words split mid-character
    NO_STRUCTURE = "no_structure"  # No apparent grid pattern
    LOW_CONTENT_RATIO = "low_content_ratio"  # Too few non-empty cells


@dataclass
class DetectionResult:
    """Result of table detection validation."""
    is_valid_table: bool
    confidence: float  # 0.0-1.0, higher = more likely to be real table
    issues: List[Tuple[DetectionIssue, str]] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []


class TableDetector:
    """
    Pre-extraction validator to filter false positives.

    Distinguishes real tables from layout artifacts using heuristics:
    - Size validation (minimum rows/columns)
    - Sparsity checks (not too many empty cells)
    - Header validation (not numeric indices or all empty)
    - Text quality (no broken words, proper alignment)
    - Structural patterns (grid-like arrangement)

    Example:
        >>> detector = TableDetector(min_rows=3, min_columns=2, max_sparsity=0.6)
        >>> result = detector.validate(extracted_table)
        >>> if result.is_valid_table:
        ...     print(f"Valid table with {result.confidence:.2f} confidence")
        >>> else:
        ...     print(f"False positive: {result.issues}")
    """

    def __init__(
        self,
        min_rows: int = 2,
        min_columns: int = 2,
        max_sparsity: float = 0.70,
        min_content_ratio: float = 0.15,
        strict_mode: bool = False
    ):
        """
        Initialize table detector.

        Args:
            min_rows: Minimum rows for valid table
            min_columns: Minimum columns for valid table
            max_sparsity: Maximum ratio of empty cells
            min_content_ratio: Minimum ratio of cells with actual content
            strict_mode: Fail on any issue (default: tolerate minor issues)
        """
        self.min_rows = min_rows
        self.min_columns = min_columns
        self.max_sparsity = max_sparsity
        self.min_content_ratio = min_content_ratio
        self.strict_mode = strict_mode

    def validate(self, table: ExtractedTable) -> DetectionResult:
        """
        Validate if extracted table is a real table or false positive.

        Args:
            table: Extracted table to validate

        Returns:
            DetectionResult with validation outcome and confidence score
        """
        result = DetectionResult(is_valid_table=True, confidence=1.0)

        # Run validation checks
        self._check_size(table, result)
        self._check_sparsity(table, result)
        self._check_content_ratio(table, result)
        self._check_header_quality(table, result)
        self._check_text_quality(table, result)
        self._check_structure(table, result)

        # Compute overall confidence
        self._compute_confidence(result)

        # Determine validity
        if self.strict_mode:
            result.is_valid_table = len(result.issues) == 0
        else:
            # Only fail on critical issues
            critical_issues = [
                DetectionIssue.TOO_SMALL,
                DetectionIssue.TOO_SPARSE,
                DetectionIssue.NO_STRUCTURE,
                DetectionIssue.LOW_CONTENT_RATIO
            ]
            result.is_valid_table = not any(
                issue_type in critical_issues
                for issue_type, _ in result.issues
            )

        return result

    def _check_size(self, table: ExtractedTable, result: DetectionResult):
        """Check if table meets minimum size requirements."""
        if table.rows < self.min_rows:
            result.issues.append((
                DetectionIssue.TOO_SMALL,
                f"Table has {table.rows} rows (minimum: {self.min_rows})"
            ))

        if table.columns < self.min_columns:
            result.issues.append((
                DetectionIssue.TOO_SMALL,
                f"Table has {table.columns} columns (minimum: {self.min_columns})"
            ))

    def _check_sparsity(self, table: ExtractedTable, result: DetectionResult):
        """Check if table has too many empty cells."""
        if table.sparsity > self.max_sparsity:
            result.issues.append((
                DetectionIssue.TOO_SPARSE,
                f"Table is {table.sparsity:.1%} empty (max: {self.max_sparsity:.1%})"
            ))

    def _check_content_ratio(self, table: ExtractedTable, result: DetectionResult):
        """Check if table has sufficient actual content."""
        if not table.data:
            result.issues.append((
                DetectionIssue.LOW_CONTENT_RATIO,
                "Table has no data"
            ))
            return

        # Count cells with substantial content (>2 chars)
        content_cells = 0
        total_cells = 0

        for row in table.data:
            for cell in row:
                total_cells += 1
                if cell and len(str(cell).strip()) > 2:
                    content_cells += 1

        content_ratio = content_cells / total_cells if total_cells > 0 else 0.0

        if content_ratio < self.min_content_ratio:
            result.issues.append((
                DetectionIssue.LOW_CONTENT_RATIO,
                f"Only {content_ratio:.1%} of cells have content (min: {self.min_content_ratio:.1%})"
            ))

    def _check_header_quality(self, table: ExtractedTable, result: DetectionResult):
        """Check if header row looks legitimate."""
        if not table.data or len(table.data) == 0:
            return

        header = table.data[0]

        # Check for all-empty header
        non_empty = [cell for cell in header if cell and str(cell).strip()]
        if len(non_empty) == 0:
            result.warnings.append("Header row is completely empty")

        # Check for numeric sequence headers (likely auto-generated)
        # e.g., [0, 1, 2, 3] or ['0', '1', '2', '3']
        if self._is_numeric_sequence(header):
            result.issues.append((
                DetectionIssue.NONSENSE_HEADER,
                f"Header appears to be auto-generated numeric sequence: {header[:5]}"
            ))

    def _check_text_quality(self, table: ExtractedTable, result: DetectionResult):
        """Check for broken text (words split mid-character)."""
        if not table.data:
            return

        broken_words = []

        for row_idx, row in enumerate(table.data[:5]):  # Check first 5 rows
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue

                cell_str = str(cell).strip()

                # Check for common broken word patterns
                if self._has_broken_text(cell_str):
                    broken_words.append(f"({row_idx},{col_idx}): '{cell_str}'")

        if len(broken_words) >= 3:  # Multiple broken words = likely artifact
            result.issues.append((
                DetectionIssue.BROKEN_TEXT,
                f"Found {len(broken_words)} broken words: {broken_words[:3]}"
            ))

    def _check_structure(self, table: ExtractedTable, result: DetectionResult):
        """Check if table has consistent structure."""
        if not table.data or len(table.data) < 2:
            return

        # Check column consistency
        col_counts = [len(row) for row in table.data]

        if len(set(col_counts)) > 3:  # Too many different column counts
            result.issues.append((
                DetectionIssue.NO_STRUCTURE,
                f"Inconsistent column counts: {set(col_counts)}"
            ))

    def _compute_confidence(self, result: DetectionResult):
        """Compute overall confidence score based on issues."""
        # Start with perfect confidence
        confidence = 1.0

        # Penalize based on issue severity
        issue_penalties = {
            DetectionIssue.TOO_SMALL: 0.5,
            DetectionIssue.TOO_SPARSE: 0.3,
            DetectionIssue.NONSENSE_HEADER: 0.3,
            DetectionIssue.BROKEN_TEXT: 0.2,
            DetectionIssue.NO_STRUCTURE: 0.4,
            DetectionIssue.LOW_CONTENT_RATIO: 0.4,
            DetectionIssue.PAGE_ARTIFACT: 0.5
        }

        for issue_type, _ in result.issues:
            penalty = issue_penalties.get(issue_type, 0.1)
            confidence -= penalty

        # Slight penalty for warnings
        confidence -= len(result.warnings) * 0.05

        # Clamp to [0.0, 1.0]
        result.confidence = max(0.0, min(1.0, confidence))

    def _is_numeric_sequence(self, header: List) -> bool:
        """Check if header is a numeric sequence like [0, 1, 2, 3]."""
        if len(header) < 3:
            return False

        # Try to convert to integers
        try:
            nums = []
            for cell in header:
                if cell is None or str(cell).strip() == '':
                    return False
                nums.append(int(str(cell).strip()))

            # Check if it's a sequence (difference of 1 or -1)
            if len(nums) < 3:
                return False

            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            return all(d == 1 for d in diffs) or all(d == -1 for d in diffs)

        except (ValueError, TypeError):
            return False

    def _has_broken_text(self, text: str) -> bool:
        """Check if text looks like broken/split words."""
        if len(text) < 3:
            return False

        # Pattern 1: Lowercase letter followed immediately by uppercase
        # e.g., "ines for Getti" (should be "lines for Getting")
        if re.search(r'[a-z][A-Z]', text) and ' ' not in text:
            return True

        # Pattern 2: Very short fragments without spaces
        # e.g., "Guidel" (should be "Guidelines")
        if len(text) < 8 and not ' ' in text and text.endswith(('el', 'es', 'ng', 'ed', 'er', 'or', 'at', 'ion')):
            return True

        # Pattern 3: Common incomplete words
        broken_patterns = [
            r'^[a-z]+\s*$',  # Single lowercase word fragment
            r'^\d+[a-z]{1,2}$',  # Number followed by 1-2 letters
        ]

        for pattern in broken_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False


__all__ = [
    "DetectionIssue",
    "DetectionResult",
    "TableDetector",
]
