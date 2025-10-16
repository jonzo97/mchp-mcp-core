"""
Table extraction evaluation metrics.

Provides industry-standard metrics for assessing table extraction quality:
- TEDS (Tree Edit Distance-based Similarity)
- S-TEDS (Structure-only TEDS)
- Traditional metrics (Precision, Recall, F1)
- Cell-level accuracy
- Row/column accuracy

Based on research from:
- "Image-based Table Recognition: Data, Model, and Evaluation" (Zhong et al., 2020)
- "GriTS: Grid Table Similarity" (Smock et al., 2022)
- Microsoft table-transformer benchmarks
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from mchp_mcp_core.extractors.table_extractors import ExtractedTable
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    TEDS = "teds"  # Tree Edit Distance-based Similarity
    TEDS_STRUCT = "teds_struct"  # Structure-only TEDS
    PRECISION = "precision"  # Cell-level precision
    RECALL = "recall"  # Cell-level recall
    F1 = "f1"  # F1 score
    ACCURACY = "accuracy"  # Exact match accuracy
    ROW_ACCURACY = "row_accuracy"  # Row-level accuracy
    COLUMN_ACCURACY = "column_accuracy"  # Column-level accuracy


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Primary metrics
    teds: Optional[float] = None  # 0.0-1.0
    teds_struct: Optional[float] = None  # 0.0-1.0

    # Cell-level metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None

    # Structural metrics
    row_accuracy: Optional[float] = None
    column_accuracy: Optional[float] = None

    # Detailed counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    correct_cells: int = 0
    total_cells_ground_truth: int = 0
    total_cells_predicted: int = 0

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = ["Evaluation Metrics:"]

        if self.teds is not None:
            lines.append(f"  TEDS: {self.teds:.4f}")
        if self.teds_struct is not None:
            lines.append(f"  TEDS (struct): {self.teds_struct:.4f}")

        if self.precision is not None:
            lines.append(f"  Precision: {self.precision:.4f}")
        if self.recall is not None:
            lines.append(f"  Recall: {self.recall:.4f}")
        if self.f1 is not None:
            lines.append(f"  F1: {self.f1:.4f}")
        if self.accuracy is not None:
            lines.append(f"  Accuracy: {self.accuracy:.4f}")

        if self.row_accuracy is not None:
            lines.append(f"  Row Accuracy: {self.row_accuracy:.4f}")
        if self.column_accuracy is not None:
            lines.append(f"  Column Accuracy: {self.column_accuracy:.4f}")

        lines.append(f"  TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives}")

        return "\n".join(lines)


class TableEvaluator:
    """
    Comprehensive table extraction evaluator.

    Computes industry-standard metrics for comparing extracted tables
    against ground truth. Supports multiple metric types and granularities.

    Example:
        >>> evaluator = TableEvaluator()
        >>> metrics = evaluator.evaluate(
        ...     predicted_table=extracted,
        ...     ground_truth_table=reference,
        ...     metrics=[MetricType.TEDS, MetricType.F1]
        ... )
        >>> print(f"TEDS: {metrics.teds:.2f}, F1: {metrics.f1:.2f}")
    """

    def __init__(self, use_teds: bool = True):
        """
        Initialize evaluator.

        Args:
            use_teds: Enable TEDS metric (requires table-recognition-metric)
        """
        self.use_teds = use_teds

        # Try to import TEDS
        self.teds_available = False
        if use_teds:
            try:
                from table_recognition_metric import TEDS as TEDSMetric
                self.teds_metric = TEDSMetric()
                self.teds_metric_struct = TEDSMetric(structure_only=True)
                self.teds_available = True
                logger.debug("TEDS metric available")
            except ImportError:
                logger.warning(
                    "table-recognition-metric not installed. "
                    "Install with: pip install table-recognition-metric"
                )

    def evaluate(
        self,
        predicted_table: ExtractedTable,
        ground_truth_table: ExtractedTable,
        metrics: Optional[List[MetricType]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate predicted table against ground truth.

        Args:
            predicted_table: Extracted table to evaluate
            ground_truth_table: Ground truth reference table
            metrics: List of metrics to compute (None = all)

        Returns:
            EvaluationMetrics with computed scores
        """
        result = EvaluationMetrics()

        # Determine which metrics to compute
        if metrics is None:
            metrics = list(MetricType)

        # Compute TEDS metrics
        if MetricType.TEDS in metrics and self.teds_available:
            result.teds = self._compute_teds(predicted_table, ground_truth_table)

        if MetricType.TEDS_STRUCT in metrics and self.teds_available:
            result.teds_struct = self._compute_teds_struct(predicted_table, ground_truth_table)

        # Compute cell-level metrics
        if any(m in metrics for m in [MetricType.PRECISION, MetricType.RECALL, MetricType.F1, MetricType.ACCURACY]):
            self._compute_cell_metrics(predicted_table, ground_truth_table, result)

        # Compute structural metrics
        if MetricType.ROW_ACCURACY in metrics:
            result.row_accuracy = self._compute_row_accuracy(predicted_table, ground_truth_table)

        if MetricType.COLUMN_ACCURACY in metrics:
            result.column_accuracy = self._compute_column_accuracy(predicted_table, ground_truth_table)

        return result

    def _compute_teds(
        self,
        predicted: ExtractedTable,
        ground_truth: ExtractedTable
    ) -> float:
        """
        Compute TEDS (Tree Edit Distance-based Similarity).

        TEDS compares tables as HTML trees, measuring both structure
        and content similarity.

        Args:
            predicted: Predicted table
            ground_truth: Ground truth table

        Returns:
            TEDS score (0.0-1.0, higher is better)
        """
        if not self.teds_available:
            logger.warning("TEDS metric not available")
            return 0.0

        try:
            # Convert tables to HTML
            pred_html = self._table_to_html(predicted)
            gt_html = self._table_to_html(ground_truth)

            # Compute TEDS
            score = self.teds_metric(pred_html, gt_html)

            return float(score)

        except Exception as e:
            logger.error(f"Error computing TEDS: {e}")
            return 0.0

    def _compute_teds_struct(
        self,
        predicted: ExtractedTable,
        ground_truth: ExtractedTable
    ) -> float:
        """
        Compute S-TEDS (structure-only TEDS).

        S-TEDS ignores cell content and only compares structure.

        Args:
            predicted: Predicted table
            ground_truth: Ground truth table

        Returns:
            S-TEDS score (0.0-1.0, higher is better)
        """
        if not self.teds_available:
            logger.warning("TEDS metric not available")
            return 0.0

        try:
            # Convert tables to HTML
            pred_html = self._table_to_html(predicted)
            gt_html = self._table_to_html(ground_truth)

            # Compute structure-only TEDS
            score = self.teds_metric_struct(pred_html, gt_html)

            return float(score)

        except Exception as e:
            logger.error(f"Error computing S-TEDS: {e}")
            return 0.0

    def _compute_cell_metrics(
        self,
        predicted: ExtractedTable,
        ground_truth: ExtractedTable,
        result: EvaluationMetrics
    ):
        """
        Compute cell-level precision, recall, F1, accuracy.

        Args:
            predicted: Predicted table
            ground_truth: Ground truth table
            result: EvaluationMetrics to update
        """
        # Count true positives, false positives, false negatives
        tp = 0
        fp = 0
        fn = 0
        correct_cells = 0
        total_cells = 0

        # Create cell position maps
        gt_cells = self._create_cell_map(ground_truth)
        pred_cells = self._create_cell_map(predicted)

        # Count matches
        for pos, gt_value in gt_cells.items():
            total_cells += 1

            if pos in pred_cells:
                pred_value = pred_cells[pos]

                # Normalize for comparison
                gt_norm = self._normalize_cell(gt_value)
                pred_norm = self._normalize_cell(pred_value)

                if gt_norm == pred_norm:
                    tp += 1
                    correct_cells += 1
                else:
                    fp += 1  # Wrong content
            else:
                fn += 1  # Missing cell

        # Count false positives (cells in prediction but not in ground truth)
        for pos in pred_cells:
            if pos not in gt_cells:
                fp += 1

        # Store counts
        result.true_positives = tp
        result.false_positives = fp
        result.false_negatives = fn
        result.correct_cells = correct_cells
        result.total_cells_ground_truth = len(gt_cells)
        result.total_cells_predicted = len(pred_cells)

        # Compute metrics
        result.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        result.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result.f1 = (
            2 * result.precision * result.recall / (result.precision + result.recall)
            if (result.precision + result.recall) > 0 else 0.0
        )
        result.accuracy = correct_cells / total_cells if total_cells > 0 else 0.0

    def _compute_row_accuracy(
        self,
        predicted: ExtractedTable,
        ground_truth: ExtractedTable
    ) -> float:
        """Compute row-level accuracy."""
        if not ground_truth.data:
            return 0.0

        correct_rows = 0

        for row_idx in range(min(len(predicted.data), len(ground_truth.data))):
            pred_row = predicted.data[row_idx]
            gt_row = ground_truth.data[row_idx]

            # Check if entire row matches
            if len(pred_row) == len(gt_row):
                if all(
                    self._normalize_cell(p) == self._normalize_cell(g)
                    for p, g in zip(pred_row, gt_row)
                ):
                    correct_rows += 1

        return correct_rows / len(ground_truth.data)

    def _compute_column_accuracy(
        self,
        predicted: ExtractedTable,
        ground_truth: ExtractedTable
    ) -> float:
        """Compute column-level accuracy."""
        if not ground_truth.data or not ground_truth.data[0]:
            return 0.0

        num_columns = len(ground_truth.data[0])
        correct_columns = 0

        for col_idx in range(num_columns):
            # Extract column from both tables
            gt_col = [row[col_idx] if col_idx < len(row) else None for row in ground_truth.data]
            pred_col = [
                row[col_idx] if col_idx < len(row) else None
                for row in predicted.data[:len(ground_truth.data)]
            ]

            # Check if entire column matches
            if len(pred_col) == len(gt_col):
                if all(
                    self._normalize_cell(p) == self._normalize_cell(g)
                    for p, g in zip(pred_col, gt_col)
                ):
                    correct_columns += 1

        return correct_columns / num_columns

    def _create_cell_map(self, table: ExtractedTable) -> Dict[Tuple[int, int], str]:
        """Create map of (row, col) -> cell_value."""
        cell_map = {}

        for row_idx, row in enumerate(table.data):
            for col_idx, cell in enumerate(row):
                cell_map[(row_idx, col_idx)] = cell

        return cell_map

    def _normalize_cell(self, cell: any) -> str:
        """Normalize cell value for comparison."""
        if cell is None:
            return ""

        # Convert to string and normalize whitespace
        normalized = str(cell).strip()
        normalized = " ".join(normalized.split())  # Collapse multiple spaces

        return normalized.lower()

    def _table_to_html(self, table: ExtractedTable) -> str:
        """
        Convert table to HTML format for TEDS.

        Args:
            table: Table to convert

        Returns:
            HTML string representation
        """
        html_parts = ["<html><body><table>"]

        for row_idx, row in enumerate(table.data):
            html_parts.append("<tr>")

            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                # Escape HTML special characters
                cell_str = (
                    cell_str.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )

                # Use <th> for header row, <td> for data
                tag = "th" if row_idx == 0 else "td"
                html_parts.append(f"<{tag}>{cell_str}</{tag}>")

            html_parts.append("</tr>")

        html_parts.append("</table></body></html>")

        return "".join(html_parts)


def compute_straight_through_processing_rate(
    results: List[Tuple[ExtractedTable, ExtractedTable, EvaluationMetrics]],
    threshold: float = 0.85
) -> float:
    """
    Compute Straight Through Processing (STP) rate.

    STP measures the percentage of tables that require no human intervention,
    based on extraction quality exceeding a threshold.

    Args:
        results: List of (predicted, ground_truth, metrics) tuples
        threshold: Quality threshold for auto-approval (default: 0.85)

    Returns:
        STP rate (0.0-1.0)
    """
    if not results:
        return 0.0

    auto_approved = sum(
        1 for _, _, metrics in results
        if (metrics.f1 is not None and metrics.f1 >= threshold)
        or (metrics.teds is not None and metrics.teds >= threshold)
    )

    return auto_approved / len(results)


__all__ = [
    "MetricType",
    "EvaluationMetrics",
    "TableEvaluator",
    "compute_straight_through_processing_rate",
]
