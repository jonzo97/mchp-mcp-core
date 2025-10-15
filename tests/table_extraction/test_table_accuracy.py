"""
Table extraction accuracy test suite.

Tests table extraction quality against ground truth annotations.
Provides quantitative metrics for evaluating extraction accuracy.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from mchp_mcp_core.extractors import TableConsensusEngine, ExtractedTable
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CellComparison:
    """Result of comparing a single cell."""
    row: int
    col: int
    expected: str
    extracted: str
    match: bool
    is_critical: bool = False


@dataclass
class TableAccuracyMetrics:
    """Accuracy metrics for a single table."""
    # Identifiers
    ground_truth_file: str
    pdf_filename: str
    page_num: int
    table_index: int
    description: str

    # Extraction success
    extraction_success: bool = False
    extraction_error: Optional[str] = None

    # Consensus metrics
    extractors_found: List[str] = field(default_factory=list)
    consensus_confidence: float = 0.0
    agreement_score: float = 0.0
    structure_score: float = 0.0
    cell_similarity: float = 0.0

    # Structure accuracy
    expected_rows: int = 0
    expected_cols: int = 0
    extracted_rows: int = 0
    extracted_cols: int = 0
    structure_match: bool = False

    # Cell-level accuracy
    total_cells: int = 0
    matching_cells: int = 0
    critical_cells_total: int = 0
    critical_cells_matching: int = 0
    empty_cells_expected: int = 0
    empty_cells_extracted: int = 0
    empty_cells_correct: int = 0

    # Computed metrics
    cell_accuracy: float = 0.0
    critical_cell_accuracy: float = 0.0
    empty_cell_precision: float = 0.0
    empty_cell_recall: float = 0.0
    overall_score: float = 0.0

    # Detailed results
    mismatched_cells: List[CellComparison] = field(default_factory=list)

    def compute_metrics(self):
        """Compute derived metrics from raw counts."""
        # Cell accuracy
        if self.total_cells > 0:
            self.cell_accuracy = self.matching_cells / self.total_cells

        # Critical cell accuracy
        if self.critical_cells_total > 0:
            self.critical_cell_accuracy = self.critical_cells_matching / self.critical_cells_total
        else:
            self.critical_cell_accuracy = 1.0  # No critical cells = perfect

        # Empty cell metrics
        if self.empty_cells_extracted > 0:
            self.empty_cell_precision = self.empty_cells_correct / self.empty_cells_extracted
        else:
            self.empty_cell_precision = 1.0

        if self.empty_cells_expected > 0:
            self.empty_cell_recall = self.empty_cells_correct / self.empty_cells_expected
        else:
            self.empty_cell_recall = 1.0

        # Overall score (weighted average)
        # Prioritize: structure > cell accuracy > critical cells
        structure_ok = 1.0 if self.structure_match else 0.5
        self.overall_score = (
            0.2 * structure_ok +
            0.4 * self.cell_accuracy +
            0.3 * self.critical_cell_accuracy +
            0.1 * (self.empty_cell_precision + self.empty_cell_recall) / 2
        )


@dataclass
class AccuracyReport:
    """Aggregate accuracy report across all test cases."""
    test_cases: List[TableAccuracyMetrics]

    # Aggregate metrics
    total_tests: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0

    avg_cell_accuracy: float = 0.0
    avg_critical_accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_overall_score: float = 0.0

    # Confidence vs accuracy correlation
    high_conf_high_acc: int = 0  # conf > 0.85 and acc > 0.90
    high_conf_low_acc: int = 0   # conf > 0.85 and acc < 0.80
    low_conf_high_acc: int = 0   # conf < 0.70 and acc > 0.90
    low_conf_low_acc: int = 0    # conf < 0.70 and acc < 0.80

    def compute_aggregates(self):
        """Compute aggregate metrics from test cases."""
        if not self.test_cases:
            return

        self.total_tests = len(self.test_cases)
        self.successful_extractions = sum(1 for tc in self.test_cases if tc.extraction_success)
        self.failed_extractions = self.total_tests - self.successful_extractions

        # Average metrics (only for successful extractions)
        successful = [tc for tc in self.test_cases if tc.extraction_success]
        if successful:
            self.avg_cell_accuracy = sum(tc.cell_accuracy for tc in successful) / len(successful)
            self.avg_critical_accuracy = sum(tc.critical_cell_accuracy for tc in successful) / len(successful)
            self.avg_confidence = sum(tc.consensus_confidence for tc in successful) / len(successful)
            self.avg_overall_score = sum(tc.overall_score for tc in successful) / len(successful)

        # Confidence vs accuracy correlation
        for tc in successful:
            if tc.consensus_confidence > 0.85:
                if tc.cell_accuracy > 0.90:
                    self.high_conf_high_acc += 1
                elif tc.cell_accuracy < 0.80:
                    self.high_conf_low_acc += 1
            elif tc.consensus_confidence < 0.70:
                if tc.cell_accuracy > 0.90:
                    self.low_conf_high_acc += 1
                elif tc.cell_accuracy < 0.80:
                    self.low_conf_low_acc += 1


class TableAccuracyEvaluator:
    """Evaluates table extraction accuracy against ground truth."""

    def __init__(
        self,
        ground_truth_dir: Path,
        sample_pdfs_dir: Path,
        extractors: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.

        Args:
            ground_truth_dir: Directory containing ground truth JSON files
            sample_pdfs_dir: Directory containing sample PDF files
            extractors: List of extractor names to use (default: all available)
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.sample_pdfs_dir = Path(sample_pdfs_dir)
        self.consensus_engine = TableConsensusEngine(extractors=extractors)

    def load_ground_truth(self, json_file: Path) -> Dict:
        """Load ground truth from JSON file."""
        with open(json_file) as f:
            return json.load(f)

    def evaluate_single_table(self, ground_truth: Dict) -> TableAccuracyMetrics:
        """
        Evaluate extraction accuracy for a single table.

        Args:
            ground_truth: Ground truth dictionary

        Returns:
            TableAccuracyMetrics with detailed results
        """
        metrics = TableAccuracyMetrics(
            ground_truth_file=str(ground_truth.get("pdf_filename", "unknown")),
            pdf_filename=ground_truth["pdf_filename"],
            page_num=ground_truth["page_num"],
            table_index=ground_truth["table_index"],
            description=ground_truth.get("description", ""),
            expected_rows=ground_truth["expected_table"]["rows"],
            expected_cols=ground_truth["expected_table"]["columns"]
        )

        # Get PDF path
        pdf_path = self.sample_pdfs_dir / ground_truth["pdf_filename"]
        if not pdf_path.exists():
            metrics.extraction_error = f"PDF file not found: {pdf_path}"
            return metrics

        # Extract with consensus
        try:
            result = self.consensus_engine.extract_with_consensus(
                pdf_path=str(pdf_path),
                page_num=ground_truth["page_num"]
            )
        except Exception as e:
            metrics.extraction_error = f"Extraction failed: {e}"
            return metrics

        if not result.success or not result.matches:
            metrics.extraction_error = "No tables extracted"
            return metrics

        # Find the matching table
        table_idx = ground_truth["table_index"]
        if table_idx >= len(result.matches):
            metrics.extraction_error = f"Table index {table_idx} not found (only {len(result.matches)} tables extracted)"
            return metrics

        match = result.matches[table_idx]
        extracted_table = match.best_version

        if not extracted_table:
            metrics.extraction_error = "No best version selected"
            return metrics

        metrics.extraction_success = True
        metrics.extractors_found = list(match.versions.keys())
        metrics.consensus_confidence = match.confidence
        metrics.agreement_score = match.agreement_score
        metrics.structure_score = match.structure_score
        metrics.cell_similarity = match.cell_similarity

        # Compare structure
        metrics.extracted_rows = extracted_table.rows
        metrics.extracted_cols = extracted_table.columns
        metrics.structure_match = (
            metrics.extracted_rows == metrics.expected_rows and
            metrics.extracted_cols == metrics.expected_cols
        )

        # Compare cells
        expected_data = ground_truth["expected_table"]["data"]
        extracted_data = extracted_table.data

        validation_rules = ground_truth.get("validation_rules", {})
        exact_match = validation_rules.get("exact_match_required", False)
        allow_whitespace = validation_rules.get("allow_whitespace_differences", True)
        critical_cells = set(tuple(cell) for cell in validation_rules.get("critical_cells", []))

        metrics.total_cells = min(len(expected_data), len(extracted_data)) * min(
            len(expected_data[0]) if expected_data else 0,
            len(extracted_data[0]) if extracted_data else 0
        )
        metrics.critical_cells_total = len(critical_cells)

        # Compare each cell
        for row in range(min(len(expected_data), len(extracted_data))):
            for col in range(min(len(expected_data[row]), len(extracted_data[row]))):
                expected_cell = expected_data[row][col]
                extracted_cell = extracted_data[row][col]

                is_critical = (row, col) in critical_cells
                match = self._cells_match(expected_cell, extracted_cell, exact_match, allow_whitespace)

                if match:
                    metrics.matching_cells += 1
                    if is_critical:
                        metrics.critical_cells_matching += 1
                else:
                    comparison = CellComparison(
                        row=row,
                        col=col,
                        expected=expected_cell,
                        extracted=extracted_cell,
                        match=False,
                        is_critical=is_critical
                    )
                    metrics.mismatched_cells.append(comparison)

                # Track empty cells
                if not expected_cell or expected_cell.strip() == "":
                    metrics.empty_cells_expected += 1
                    if not extracted_cell or extracted_cell.strip() == "":
                        metrics.empty_cells_correct += 1

                if not extracted_cell or extracted_cell.strip() == "":
                    metrics.empty_cells_extracted += 1

        metrics.compute_metrics()
        return metrics

    def _cells_match(
        self,
        expected: str,
        extracted: str,
        exact_match: bool,
        allow_whitespace: bool
    ) -> bool:
        """Check if two cells match according to rules."""
        if allow_whitespace:
            expected = expected.strip()
            extracted = extracted.strip()

        if exact_match:
            return expected == extracted
        else:
            # Case-insensitive comparison
            return expected.lower() == extracted.lower()

    def evaluate_all(self) -> AccuracyReport:
        """
        Evaluate all ground truth files in the directory.

        Returns:
            AccuracyReport with aggregate metrics
        """
        test_cases = []

        # Find all ground truth JSON files
        gt_files = list(self.ground_truth_dir.glob("*.json"))
        gt_files = [f for f in gt_files if f.name != "template.json"]

        if not gt_files:
            logger.warning(f"No ground truth files found in {self.ground_truth_dir}")
            return AccuracyReport(test_cases=[])

        logger.info(f"Found {len(gt_files)} ground truth files")

        # Evaluate each file
        for gt_file in gt_files:
            logger.info(f"Evaluating: {gt_file.name}")

            try:
                ground_truth = self.load_ground_truth(gt_file)
                metrics = self.evaluate_single_table(ground_truth)
                metrics.ground_truth_file = gt_file.name
                test_cases.append(metrics)

                # Log results
                if metrics.extraction_success:
                    logger.info(f"  ✓ Success | Confidence: {metrics.consensus_confidence:.2f} | Cell Acc: {metrics.cell_accuracy:.2%}")
                else:
                    logger.error(f"  ✗ Failed: {metrics.extraction_error}")

            except Exception as e:
                logger.error(f"Error evaluating {gt_file.name}: {e}")
                metrics = TableAccuracyMetrics(
                    ground_truth_file=gt_file.name,
                    pdf_filename="",
                    page_num=0,
                    table_index=0,
                    description="",
                    extraction_error=str(e)
                )
                test_cases.append(metrics)

        # Generate report
        report = AccuracyReport(test_cases=test_cases)
        report.compute_aggregates()

        return report

    def generate_html_report(self, report: AccuracyReport, output_path: Path):
        """Generate HTML report with detailed results."""
        html = self._build_html_report(report)

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")

    def _build_html_report(self, report: AccuracyReport) -> str:
        """Build HTML report content."""
        # Summary section
        summary = f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-value">{report.successful_extractions}/{report.total_tests}</div>
                    <div class="metric-label">Successful Extractions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.avg_cell_accuracy:.1%}</div>
                    <div class="metric-label">Avg Cell Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.avg_confidence:.2f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.avg_overall_score:.2f}</div>
                    <div class="metric-label">Overall Score</div>
                </div>
            </div>
        </div>
        """

        # Confidence vs accuracy section
        confidence_section = f"""
        <div class="correlation">
            <h2>Confidence vs Accuracy Correlation</h2>
            <table>
                <tr>
                    <th></th>
                    <th>High Accuracy (>0.90)</th>
                    <th>Low Accuracy (<0.80)</th>
                </tr>
                <tr>
                    <th>High Confidence (>0.85)</th>
                    <td class="good">{report.high_conf_high_acc}</td>
                    <td class="bad">{report.high_conf_low_acc}</td>
                </tr>
                <tr>
                    <th>Low Confidence (<0.70)</th>
                    <td class="warning">{report.low_conf_high_acc}</td>
                    <td class="neutral">{report.low_conf_low_acc}</td>
                </tr>
            </table>
        </div>
        """

        # Test cases section
        cases_html = []
        for tc in report.test_cases:
            status_class = "success" if tc.extraction_success else "failure"
            status_icon = "✓" if tc.extraction_success else "✗"

            if tc.extraction_success:
                details = f"""
                <div class="details">
                    <p><strong>Extractors:</strong> {', '.join(tc.extractors_found)}</p>
                    <p><strong>Confidence:</strong> {tc.consensus_confidence:.2f} (Agreement: {tc.agreement_score:.2f}, Structure: {tc.structure_score:.2f}, Similarity: {tc.cell_similarity:.2f})</p>
                    <p><strong>Structure:</strong> {tc.extracted_rows}x{tc.extracted_cols} (expected: {tc.expected_rows}x{tc.expected_cols}) {'✓' if tc.structure_match else '✗'}</p>
                    <p><strong>Cell Accuracy:</strong> {tc.cell_accuracy:.1%} ({tc.matching_cells}/{tc.total_cells} cells)</p>
                    <p><strong>Critical Cell Accuracy:</strong> {tc.critical_cell_accuracy:.1%} ({tc.critical_cells_matching}/{tc.critical_cells_total} cells)</p>
                    <p><strong>Overall Score:</strong> {tc.overall_score:.2f}</p>
                </div>
                """

                if tc.mismatched_cells:
                    details += "<h4>Mismatched Cells:</h4><ul>"
                    for mismatch in tc.mismatched_cells[:10]:  # Show first 10
                        critical_marker = " [CRITICAL]" if mismatch.is_critical else ""
                        details += f"<li>({mismatch.row}, {mismatch.col}){critical_marker}: Expected '{mismatch.expected}' → Got '{mismatch.extracted}'</li>"
                    if len(tc.mismatched_cells) > 10:
                        details += f"<li>... and {len(tc.mismatched_cells) - 10} more</li>"
                    details += "</ul>"
            else:
                details = f"<p class='error'>{tc.extraction_error}</p>"

            cases_html.append(f"""
            <div class="test-case {status_class}">
                <h3>{status_icon} {tc.ground_truth_file}</h3>
                <p class="description">{tc.description}</p>
                {details}
            </div>
            """)

        cases_section = f"""
        <div class="test-cases">
            <h2>Test Cases</h2>
            {''.join(cases_html)}
        </div>
        """

        # Full HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Table Extraction Accuracy Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1e1e1e; color: #e0e0e0; }}
                h1 {{ color: #14b8a6; }}
                h2 {{ color: #14b8a6; border-bottom: 2px solid #14b8a6; padding-bottom: 5px; }}
                .summary, .correlation, .test-cases {{ margin: 30px 0; }}
                .metric-cards {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                .metric-card {{ background: #2a2a2a; padding: 20px; border-radius: 8px; min-width: 150px; border: 1px solid #444; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #14b8a6; }}
                .metric-label {{ font-size: 14px; color: #888; margin-top: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: #2a2a2a; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #444; }}
                th {{ background: #333; color: #14b8a6; }}
                .good {{ background: #0d5e52; color: white; }}
                .bad {{ background: #7f1d1d; color: white; }}
                .warning {{ background: #78350f; color: white; }}
                .neutral {{ background: #374151; color: white; }}
                .test-case {{ background: #2a2a2a; padding: 20px; margin: 15px 0; border-radius: 8px; border: 1px solid #444; }}
                .test-case.success {{ border-left: 4px solid #10b981; }}
                .test-case.failure {{ border-left: 4px solid #ef4444; }}
                .description {{ color: #888; font-style: italic; }}
                .details {{ margin: 15px 0; font-size: 14px; }}
                .details p {{ margin: 5px 0; }}
                .error {{ color: #ef4444; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Table Extraction Accuracy Report</h1>
            {summary}
            {confidence_section}
            {cases_section}
        </body>
        </html>
        """


# Pytest fixtures and tests

@pytest.fixture
def evaluator():
    """Create evaluator fixture."""
    gt_dir = Path(__file__).parent / "ground_truth"
    pdf_dir = Path(__file__).parent / "sample_pdfs"
    return TableAccuracyEvaluator(gt_dir, pdf_dir)


@pytest.mark.skipif(
    not (Path(__file__).parent / "sample_pdfs").exists() or
    len(list((Path(__file__).parent / "sample_pdfs").glob("*.pdf"))) == 0,
    reason="No sample PDFs available"
)
def test_evaluate_all_tables(evaluator):
    """Test evaluation of all ground truth tables."""
    report = evaluator.evaluate_all()

    assert report.total_tests > 0, "No test cases found"
    assert report.successful_extractions > 0, "No successful extractions"

    # Check that some tables have good accuracy
    high_accuracy_count = sum(
        1 for tc in report.test_cases
        if tc.extraction_success and tc.cell_accuracy >= 0.85
    )

    assert high_accuracy_count > 0, "No tables achieved high accuracy (≥0.85)"

    # Generate HTML report
    report_path = Path(__file__).parent / "accuracy_report.html"
    evaluator.generate_html_report(report, report_path)
    assert report_path.exists()


def test_ground_truth_format():
    """Test that ground truth files have correct format."""
    gt_dir = Path(__file__).parent / "ground_truth"
    gt_files = [f for f in gt_dir.glob("*.json") if f.name != "template.json"]

    required_fields = ["pdf_filename", "page_num", "table_index", "expected_table"]

    for gt_file in gt_files:
        with open(gt_file) as f:
            data = json.load(f)

        for field in required_fields:
            assert field in data, f"{gt_file.name} missing required field: {field}"

        assert "data" in data["expected_table"], f"{gt_file.name} missing expected_table.data"
        assert "rows" in data["expected_table"], f"{gt_file.name} missing expected_table.rows"
        assert "columns" in data["expected_table"], f"{gt_file.name} missing expected_table.columns"


if __name__ == "__main__":
    # Run evaluation from command line
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        gt_dir = Path(__file__).parent / "ground_truth"
        pdf_dir = Path(__file__).parent / "sample_pdfs"

        evaluator = TableAccuracyEvaluator(gt_dir, pdf_dir)
        report = evaluator.evaluate_all()

        print("\n" + "=" * 60)
        print("TABLE EXTRACTION ACCURACY REPORT")
        print("=" * 60)
        print(f"\nTotal tests: {report.total_tests}")
        print(f"Successful extractions: {report.successful_extractions}/{report.total_tests}")
        print(f"\nAverage cell accuracy: {report.avg_cell_accuracy:.1%}")
        print(f"Average critical cell accuracy: {report.avg_critical_accuracy:.1%}")
        print(f"Average confidence: {report.avg_confidence:.2f}")
        print(f"Average overall score: {report.avg_overall_score:.2f}")

        print(f"\nConfidence vs Accuracy Correlation:")
        print(f"  High conf (>0.85) + High acc (>0.90): {report.high_conf_high_acc}")
        print(f"  High conf (>0.85) + Low acc (<0.80):  {report.high_conf_low_acc} ⚠")
        print(f"  Low conf (<0.70) + High acc (>0.90):  {report.low_conf_high_acc}")
        print(f"  Low conf (<0.70) + Low acc (<0.80):   {report.low_conf_low_acc}")

        # Generate HTML report
        report_path = Path(__file__).parent / "accuracy_report.html"
        evaluator.generate_html_report(report, report_path)
        print(f"\nHTML report: {report_path}")
        print("=" * 60)
    else:
        print("Usage: python test_table_accuracy.py evaluate")
        print("Or run with pytest: pytest test_table_accuracy.py -v")
