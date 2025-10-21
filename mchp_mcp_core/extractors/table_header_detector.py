"""
Table header detection using multiple heuristics.

Implements 8 sophisticated heuristics inspired by table-header-detective
and academic research to detect header rows in extracted tables.

Author: Claude (Anthropic)
Date: 2025-10-17
Phase: 3D - Enhanced Header Detection
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HeaderDetectionResult:
    """Result of header detection."""
    header_row_indices: List[int]  # Which rows are headers
    confidence: float  # 0.0-1.0 overall confidence
    row_scores: Dict[int, float]  # Per-row confidence scores
    method: str = "heuristic"  # Detection method used


class TableHeaderDetector:
    """
    Detects header rows using multiple heuristics.

    Implements 8 heuristics:
    1. Capitalization patterns (ALL CAPS, Title Case)
    2. Type differences (strings vs numbers)
    3. Text length (headers tend to be shorter)
    4. Empty cell patterns (merged cells, spanning)
    5. Special characters (colons, parentheses)
    6. Value uniqueness (unique column identifiers)
    7. Header terminology ("Name", "ID", "Description")
    8. Numeric sequences (data rows have sequential IDs)

    Usage:
        >>> detector = TableHeaderDetector()
        >>> from mchp_mcp_core.extractors import ExtractedTable
        >>> result = detector.detect_header_rows(table)
        >>> print(f"Headers: {result.header_row_indices}, confidence: {result.confidence:.2f}")
    """

    def __init__(self, max_header_rows: int = 5):
        """
        Initialize detector.

        Args:
            max_header_rows: Maximum number of header rows to detect (default: 5)
        """
        self.max_header_rows = max_header_rows

        # Common header terms (case-insensitive)
        self.header_terms = {
            'name', 'id', 'type', 'description', 'value', 'number', 'pin',
            'function', 'address', 'register', 'bit', 'field', 'mode',
            'status', 'control', 'signal', 'port', 'channel', 'index'
        }

    def detect_header_rows(self, table_data: List[List[str]]) -> HeaderDetectionResult:
        """
        Detect header rows in table data.

        Args:
            table_data: 2D list of strings representing table

        Returns:
            HeaderDetectionResult with detected header rows and confidence
        """
        if not table_data or len(table_data) < 2:
            return HeaderDetectionResult(
                header_row_indices=[0] if table_data else [],
                confidence=0.5,
                row_scores={},
                method="heuristic_default"
            )

        # Limit search to first N rows
        max_rows = min(self.max_header_rows, len(table_data))

        # Compute scores for each heuristic
        row_scores = {}
        for row_idx in range(max_rows):
            score = self._compute_row_score(table_data, row_idx)
            row_scores[row_idx] = score

        # Determine header cutoff
        header_indices = self._determine_header_cutoff(row_scores)

        # Compute overall confidence
        if header_indices:
            avg_header_score = sum(row_scores[i] for i in header_indices) / len(header_indices)
            confidence = avg_header_score
        else:
            confidence = 0.5  # Neutral

        return HeaderDetectionResult(
            header_row_indices=header_indices,
            confidence=confidence,
            row_scores=row_scores,
            method="heuristic_multi"
        )

    def _compute_row_score(self, table_data: List[List[str]], row_idx: int) -> float:
        """
        Compute header likelihood score for a single row.

        Returns score from 0.0 (definitely data) to 1.0 (definitely header).
        """
        row = table_data[row_idx]

        # Get comparison rows (data rows below)
        data_rows = table_data[row_idx + 1:min(row_idx + 6, len(table_data))]

        scores = []

        # Heuristic 1: Capitalization patterns
        scores.append(self._check_capitalization_patterns(row))

        # Heuristic 2: Type differences (compared to data rows)
        if data_rows:
            scores.append(self._check_type_differences(row, data_rows))

        # Heuristic 3: Text length (headers shorter)
        if data_rows:
            scores.append(self._check_text_length(row, data_rows))

        # Heuristic 4: Empty cell patterns
        scores.append(self._check_empty_cell_patterns(row))

        # Heuristic 5: Special characters
        scores.append(self._check_special_characters(row))

        # Heuristic 6: Value uniqueness
        if data_rows:
            scores.append(self._check_value_uniqueness(row, data_rows))

        # Heuristic 7: Header terminology
        scores.append(self._check_terminology(row))

        # Heuristic 8: Numeric sequences (data rows)
        if data_rows:
            data_score = self._check_numeric_sequences(row, data_rows)
            # Invert: high numeric sequence score = low header score
            scores.append(1.0 - data_score)

        # Weighted average
        return sum(scores) / len(scores) if scores else 0.5

    def _check_capitalization_patterns(self, row: List[str]) -> float:
        """
        Check for header capitalization patterns (ALL CAPS, Title Case).

        Returns: 0.0-1.0 score
        """
        all_caps_count = 0
        title_case_count = 0
        text_cells = 0

        for cell in row:
            cell_str = str(cell).strip()
            if not cell_str or len(cell_str) < 2:
                continue

            # Skip numeric cells
            if cell_str.replace('.', '').replace('-', '').isdigit():
                continue

            text_cells += 1

            # Check ALL CAPS
            if cell_str.isupper():
                all_caps_count += 1

            # Check Title Case (first letter capital)
            if cell_str[0].isupper():
                title_case_count += 1

        if text_cells == 0:
            return 0.5

        # Headers often have ALL CAPS or Title Case
        all_caps_ratio = all_caps_count / text_cells
        title_case_ratio = title_case_count / text_cells

        return max(all_caps_ratio, title_case_ratio * 0.7)  # ALL CAPS stronger signal

    def _check_type_differences(self, row: List[str], data_rows: List[List[str]]) -> float:
        """
        Headers tend to be text, data rows tend to have more numbers.

        Returns: 0.0-1.0 score
        """
        # Count numeric cells in header candidate
        header_numeric = sum(1 for cell in row if str(cell).strip().replace('.', '').replace('-', '').isdigit())
        header_text = len(row) - header_numeric

        # Count numeric cells in data rows
        data_numeric = 0
        data_total = 0
        for data_row in data_rows:
            for cell in data_row:
                if str(cell).strip():
                    data_total += 1
                    if str(cell).strip().replace('.', '').replace('-', '').isdigit():
                        data_numeric += 1

        if data_total == 0:
            return 0.5

        header_numeric_ratio = header_numeric / len(row) if row else 0
        data_numeric_ratio = data_numeric / data_total

        # Headers have lower numeric ratio than data
        if header_numeric_ratio < data_numeric_ratio:
            return 0.8
        elif header_numeric_ratio == 0:
            return 0.9  # No numbers = likely header
        else:
            return 0.3

    def _check_text_length(self, row: List[str], data_rows: List[List[str]]) -> float:
        """
        Headers tend to be shorter than data cells.

        Returns: 0.0-1.0 score
        """
        header_lengths = [len(str(cell).strip()) for cell in row if str(cell).strip()]
        if not header_lengths:
            return 0.5

        avg_header_len = sum(header_lengths) / len(header_lengths)

        # Get average data cell length
        data_lengths = []
        for data_row in data_rows:
            data_lengths.extend([len(str(cell).strip()) for cell in data_row if str(cell).strip()])

        if not data_lengths:
            return 0.5

        avg_data_len = sum(data_lengths) / len(data_lengths)

        # Headers typically shorter
        if avg_header_len < avg_data_len:
            ratio = avg_header_len / avg_data_len if avg_data_len > 0 else 0
            return 0.6 + (0.3 * (1 - ratio))  # 0.6-0.9 range
        else:
            return 0.4

    def _check_empty_cell_patterns(self, row: List[str]) -> float:
        """
        Headers may have merged cells (empty cells for spanning).

        Returns: 0.0-1.0 score
        """
        empty_count = sum(1 for cell in row if not str(cell).strip() or str(cell).strip() in ['-', '—', ''])
        empty_ratio = empty_count / len(row) if row else 0

        # Some empty cells suggest merged/spanning header
        if 0.2 < empty_ratio < 0.7:
            return 0.7
        elif empty_ratio == 0:
            return 0.5  # Neutral
        else:
            return 0.3  # Too many or too few empty cells

    def _check_special_characters(self, row: List[str]) -> float:
        """
        Headers often contain colons, parentheses, brackets.

        Returns: 0.0-1.0 score
        """
        special_chars = re.compile(r'[:()\[\]/]')
        cells_with_special = sum(1 for cell in row if special_chars.search(str(cell)))

        if cells_with_special > 0:
            return 0.6  # Mild positive signal
        else:
            return 0.5  # Neutral

    def _check_value_uniqueness(self, row: List[str], data_rows: List[List[str]]) -> float:
        """
        Header cells should be unique column identifiers.

        Returns: 0.0-1.0 score
        """
        # Check if header row has unique values
        non_empty = [str(cell).strip() for cell in row if str(cell).strip()]
        unique_ratio = len(set(non_empty)) / len(non_empty) if non_empty else 0

        # Headers should have high uniqueness
        return unique_ratio * 0.8  # Scale to 0.0-0.8

    def _check_terminology(self, row: List[str]) -> float:
        """
        Check for common header-related words.

        Returns: 0.0-1.0 score
        """
        matches = 0
        for cell in row:
            cell_lower = str(cell).strip().lower()
            # Check for any header term
            if any(term in cell_lower for term in self.header_terms):
                matches += 1

        if matches > 0:
            ratio = matches / len(row)
            return 0.5 + (0.5 * ratio)  # 0.5-1.0 range
        else:
            return 0.5  # Neutral

    def _check_numeric_sequences(self, row: List[str], data_rows: List[List[str]]) -> float:
        """
        Data rows often have sequential IDs (1, 2, 3...).
        Returns HIGH score if looks like data row (sequential numbers).

        Returns: 0.0-1.0 score (higher = more like data row)
        """
        # Check if first column has sequential numbers
        try:
            first_cells = [row[0]] + [data_row[0] for data_row in data_rows if data_row]
            numbers = []
            for cell in first_cells[:5]:  # Check first 5
                cell_str = str(cell).strip()
                if cell_str.isdigit():
                    numbers.append(int(cell_str))

            if len(numbers) >= 3:
                # Check if sequential (allowing gaps)
                diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
                if all(d >= 0 and d <= 2 for d in diffs):  # Sequential or near-sequential
                    return 0.9  # HIGH data row score
        except (IndexError, ValueError):
            pass

        return 0.3  # Low data row score = could be header

    def _determine_header_cutoff(self, row_scores: Dict[int, float]) -> List[int]:
        """
        Determine which rows are headers based on scores.

        Uses threshold-based approach: consecutive rows with score > 0.55
        """
        if not row_scores:
            return [0]  # Default to first row

        # Find consecutive high-scoring rows from the start
        headers = []
        for row_idx in sorted(row_scores.keys()):
            score = row_scores[row_idx]
            if score > 0.55:  # Threshold for header
                headers.append(row_idx)
            else:
                # Stop at first low-scoring row
                break

        # Always include at least row 0 if it's close to threshold
        if not headers or (0 not in headers and row_scores.get(0, 0) > 0.45):
            headers = [0] + headers

        # Ensure consecutive
        if headers:
            headers = list(range(min(headers), max(headers) + 1))

        return headers if headers else [0]
