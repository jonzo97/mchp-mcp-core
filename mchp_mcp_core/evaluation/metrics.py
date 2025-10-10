"""
Evaluation metrics module for retrieval quality assessment.

Provides standard information retrieval metrics:
- Precision@K (P@1, P@3, P@5, etc.)
- Mean Reciprocal Rank (MRR)
- Batch evaluation utilities
- HTML/JSON report generation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mchp_mcp_core.storage import SearchResult
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class QueryEvaluation:
    """Evaluation result for a single query."""
    query: str
    expected_doc: str
    category: str
    p_at_1: float
    p_at_3: float
    p_at_5: float
    mrr: float
    latency_ms: float
    top_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report with metrics and query details."""
    timestamp: str
    total_queries: int
    p_at_1: float
    p_at_3: float
    p_at_5: float
    mrr: float
    avg_latency_ms: float
    queries: List[QueryEvaluation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def precision_at_k(
    results: List[SearchResult],
    expected_doc: str,
    k: int = 3,
    match_fn: Optional[Callable[[SearchResult, str], bool]] = None
) -> float:
    """
    Calculate Precision@K metric.

    Checks if expected document appears in top K results.

    Args:
        results: List of search results (ordered by relevance)
        expected_doc: Expected document identifier (filename or doc_id)
        k: Number of top results to consider
        match_fn: Optional custom matching function (default: filename substring match)

    Returns:
        1.0 if expected doc found in top K, else 0.0

    Example:
        >>> p_at_3 = precision_at_k(results, "PolarFire_Handbook.pdf", k=3)
        >>> print(f"P@3: {p_at_3}")
        P@3: 1.0
    """
    if not results:
        return 0.0

    # Default matching function: case-insensitive substring match on filename
    if match_fn is None:
        def default_match(result: SearchResult, expected: str) -> bool:
            result_name = Path(result.source_path).name
            return expected.lower() in result_name.lower()
        match_fn = default_match

    # Check top K results
    for result in results[:k]:
        if match_fn(result, expected_doc):
            return 1.0

    return 0.0


def mean_reciprocal_rank(
    results: List[SearchResult],
    expected_doc: str,
    match_fn: Optional[Callable[[SearchResult, str], bool]] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) metric.

    Returns the reciprocal of the rank of the first relevant result.
    MRR = 1/rank (e.g., rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.333, not found → 0.0)

    Args:
        results: List of search results (ordered by relevance)
        expected_doc: Expected document identifier
        match_fn: Optional custom matching function

    Returns:
        Reciprocal rank (1/rank) or 0.0 if not found

    Example:
        >>> mrr = mean_reciprocal_rank(results, "UserGuide.pdf")
        >>> print(f"MRR: {mrr:.3f}")
        MRR: 0.500  # Found at rank 2
    """
    if not results:
        return 0.0

    # Default matching function
    if match_fn is None:
        def default_match(result: SearchResult, expected: str) -> bool:
            result_name = Path(result.source_path).name
            return expected.lower() in result_name.lower()
        match_fn = default_match

    # Find first matching result
    for i, result in enumerate(results):
        if match_fn(result, expected_doc):
            return 1.0 / (i + 1)

    return 0.0


def recall_at_k(
    results: List[SearchResult],
    expected_docs: List[str],
    k: int = 10,
    match_fn: Optional[Callable[[SearchResult, str], bool]] = None
) -> float:
    """
    Calculate Recall@K metric.

    Measures what fraction of relevant documents are retrieved in top K.

    Args:
        results: List of search results
        expected_docs: List of relevant document identifiers
        k: Number of top results to consider
        match_fn: Optional custom matching function

    Returns:
        Recall score (0.0 to 1.0)

    Example:
        >>> relevant = ["Doc1.pdf", "Doc2.pdf", "Doc3.pdf"]
        >>> recall = recall_at_k(results, relevant, k=5)
        >>> print(f"Recall@5: {recall:.2f}")
        Recall@5: 0.67  # Found 2 out of 3
    """
    if not expected_docs:
        return 0.0

    if not results:
        return 0.0

    # Default matching function
    if match_fn is None:
        def default_match(result: SearchResult, expected: str) -> bool:
            result_name = Path(result.source_path).name
            return expected.lower() in result_name.lower()
        match_fn = default_match

    # Count matches in top K
    found = 0
    for expected in expected_docs:
        for result in results[:k]:
            if match_fn(result, expected):
                found += 1
                break  # Count each expected doc only once

    return found / len(expected_docs)


class EvaluationMetrics:
    """
    Evaluation metrics calculator for batch evaluation.

    Features:
    - Batch query evaluation
    - Multiple metric calculation (P@K, MRR, Recall)
    - Query categorization and filtering
    - HTML/JSON report generation

    Example:
        >>> evaluator = EvaluationMetrics(search_fn=vector_store.search)
        >>> test_queries = [
        ...     {"query": "SPI interface", "expected_doc": "Datasheet.pdf", "category": "peripherals"},
        ...     {"query": "power modes", "expected_doc": "UserGuide.pdf", "category": "power"}
        ... ]
        >>> report = evaluator.evaluate_queries(test_queries)
        >>> print(f"P@3: {report.p_at_3:.1%}")
        P@3: 85.0%
    """

    def __init__(
        self,
        search_fn: Callable[[str], List[SearchResult]],
        k_values: List[int] = None
    ):
        """
        Initialize evaluator.

        Args:
            search_fn: Function that takes a query string and returns SearchResults
            k_values: List of K values to calculate P@K for (default: [1, 3, 5])
        """
        self.search_fn = search_fn
        self.k_values = k_values or [1, 3, 5]

    def evaluate_query(
        self,
        query: str,
        expected_doc: str,
        category: str = "general",
        top_k: int = 10
    ) -> QueryEvaluation:
        """
        Evaluate a single query.

        Args:
            query: Query string
            expected_doc: Expected document identifier
            category: Query category (for grouping)
            top_k: Number of results to retrieve

        Returns:
            QueryEvaluation with metrics
        """
        import time

        # Execute search
        start = time.time()
        results = self.search_fn(query)
        latency_ms = (time.time() - start) * 1000

        # Calculate metrics
        p1 = precision_at_k(results, expected_doc, k=1)
        p3 = precision_at_k(results, expected_doc, k=3)
        p5 = precision_at_k(results, expected_doc, k=5)
        mrr = mean_reciprocal_rank(results, expected_doc)

        # Extract top results for reporting
        top_results = [
            {
                "rank": i + 1,
                "title": r.title,
                "score": r.score,
                "source_path": Path(r.source_path).name,
                "location": r.location
            }
            for i, r in enumerate(results[:5])
        ]

        return QueryEvaluation(
            query=query,
            expected_doc=expected_doc,
            category=category,
            p_at_1=p1,
            p_at_3=p3,
            p_at_5=p5,
            mrr=mrr,
            latency_ms=latency_ms,
            top_results=top_results
        )

    def evaluate_queries(
        self,
        test_queries: List[Dict[str, str]],
        show_progress: bool = True
    ) -> EvaluationReport:
        """
        Evaluate multiple queries in batch.

        Args:
            test_queries: List of dicts with 'query', 'expected_doc', 'category' keys
            show_progress: Whether to show progress

        Returns:
            Complete evaluation report
        """
        if show_progress:
            logger.info(f"🔍 Evaluating {len(test_queries)} queries...")

        query_results = []
        p1_scores = []
        p3_scores = []
        p5_scores = []
        mrr_scores = []
        latencies = []

        for i, query_data in enumerate(test_queries, 1):
            query = query_data["query"]
            expected_doc = query_data["expected_doc"]
            category = query_data.get("category", "general")

            if show_progress:
                logger.info(f"[{i}/{len(test_queries)}] {query[:60]}...")

            # Evaluate
            eval_result = self.evaluate_query(query, expected_doc, category)
            query_results.append(eval_result)

            # Collect scores
            p1_scores.append(eval_result.p_at_1)
            p3_scores.append(eval_result.p_at_3)
            p5_scores.append(eval_result.p_at_5)
            mrr_scores.append(eval_result.mrr)
            latencies.append(eval_result.latency_ms)

            # Progress indicator
            if show_progress:
                status = "✅" if eval_result.p_at_3 == 1.0 else "⚠️" if eval_result.p_at_5 == 1.0 else "❌"
                logger.info(f"  {status} P@3={eval_result.p_at_3:.1f} | Latency={eval_result.latency_ms:.0f}ms")

        # Calculate aggregate metrics
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_queries=len(test_queries),
            p_at_1=sum(p1_scores) / len(p1_scores) if p1_scores else 0.0,
            p_at_3=sum(p3_scores) / len(p3_scores) if p3_scores else 0.0,
            p_at_5=sum(p5_scores) / len(p5_scores) if p5_scores else 0.0,
            mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            queries=query_results
        )

        if show_progress:
            self._print_summary(report)

        return report

    def _print_summary(self, report: EvaluationReport):
        """Print evaluation summary to console."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Precision@1:  {report.p_at_1:.3f} ({report.p_at_1 * 100:.1f}%)")
        logger.info(f"Precision@3:  {report.p_at_3:.3f} ({report.p_at_3 * 100:.1f}%)")
        logger.info(f"Precision@5:  {report.p_at_5:.3f} ({report.p_at_5 * 100:.1f}%)")
        logger.info(f"MRR:          {report.mrr:.3f}")
        logger.info(f"Avg Latency:  {report.avg_latency_ms:.0f}ms")
        logger.info("=" * 60)

    def export_to_dict(self, report: EvaluationReport) -> Dict[str, Any]:
        """Export report to dictionary for JSON serialization."""
        return {
            "timestamp": report.timestamp,
            "total_queries": report.total_queries,
            "metrics": {
                "p_at_1": report.p_at_1,
                "p_at_3": report.p_at_3,
                "p_at_5": report.p_at_5,
                "mrr": report.mrr,
                "avg_latency_ms": report.avg_latency_ms
            },
            "queries": [
                {
                    "query": q.query,
                    "expected_doc": q.expected_doc,
                    "category": q.category,
                    "p_at_1": q.p_at_1,
                    "p_at_3": q.p_at_3,
                    "p_at_5": q.p_at_5,
                    "mrr": q.mrr,
                    "latency_ms": q.latency_ms,
                    "top_results": q.top_results
                }
                for q in report.queries
            ],
            "metadata": report.metadata
        }

    def generate_html_report(self, report: EvaluationReport, title: str = "Retrieval Evaluation") -> str:
        """
        Generate HTML report with evaluation results.

        Args:
            report: Evaluation report
            title: Report title

        Returns:
            HTML string
        """
        metrics = {
            "p_at_1": report.p_at_1,
            "p_at_3": report.p_at_3,
            "p_at_5": report.p_at_5,
            "mrr": report.mrr,
            "avg_latency_ms": report.avg_latency_ms
        }

        # Categorize queries
        passed_p3 = [q for q in report.queries if q.p_at_3 == 1.0]
        failed_p3 = [q for q in report.queries if q.p_at_3 < 1.0]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #e5e7eb;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #14b8a6; margin-bottom: 10px; }}
        h2 {{ color: #5eead4; margin-top: 40px; margin-bottom: 20px; }}
        .timestamp {{ color: #9ca3af; margin-bottom: 40px; }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #404040;
        }}
        .metric-label {{ color: #9ca3af; font-size: 0.9rem; margin-bottom: 8px; }}
        .metric-value {{ font-size: 2rem; font-weight: 600; color: #14b8a6; }}
        .metric-value.warning {{ color: #f59e0b; }}
        .metric-value.error {{ color: #ef4444; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #404040;
        }}
        th {{
            background: #333333;
            color: #5eead4;
            font-weight: 600;
        }}
        tr:hover {{ background: #333333; }}

        .pass {{ color: #10b981; }}
        .fail {{ color: #ef4444; }}
        .query-text {{ max-width: 400px; }}
        .top-results {{ font-size: 0.85rem; color: #9ca3af; }}
        .result-item {{ margin: 4px 0; }}
        .rank {{ color: #14b8a6; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p class="timestamp">Generated: {report.timestamp}</p>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Precision@1</div>
                <div class="metric-value {'warning' if metrics['p_at_1'] < 0.6 else ''}">{metrics['p_at_1']:.3f}</div>
                <div class="metric-label">{metrics['p_at_1']*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision@3</div>
                <div class="metric-value {'error' if metrics['p_at_3'] < 0.7 else ''}">{metrics['p_at_3']:.3f}</div>
                <div class="metric-label">{metrics['p_at_3']*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision@5</div>
                <div class="metric-value">{metrics['p_at_5']:.3f}</div>
                <div class="metric-label">{metrics['p_at_5']*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Reciprocal Rank</div>
                <div class="metric-value">{metrics['mrr']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value">{metrics['avg_latency_ms']:.0f}<span style="font-size:1rem">ms</span></div>
            </div>
        </div>

        <h2>✅ Passed Queries ({len(passed_p3)}/{report.total_queries})</h2>
        <table>
            <tr>
                <th>Query</th>
                <th>Category</th>
                <th>P@1</th>
                <th>P@3</th>
                <th>P@5</th>
                <th>MRR</th>
                <th>Latency</th>
            </tr>
"""

        for q in passed_p3:
            html += f"""
            <tr>
                <td class="query-text">{q.query}</td>
                <td>{q.category}</td>
                <td class="pass">{q.p_at_1:.1f}</td>
                <td class="pass">{q.p_at_3:.1f}</td>
                <td class="pass">{q.p_at_5:.1f}</td>
                <td>{q.mrr:.2f}</td>
                <td>{q.latency_ms:.0f}ms</td>
            </tr>
"""

        html += f"""
        </table>

        <h2>❌ Failed Queries ({len(failed_p3)}/{report.total_queries})</h2>
"""

        if failed_p3:
            html += """
        <table>
            <tr>
                <th>Query</th>
                <th>Expected Doc</th>
                <th>Top 5 Results</th>
                <th>P@3</th>
            </tr>
"""
            for q in failed_p3:
                top_results_html = "<div class='top-results'>"
                for r in q.top_results:
                    top_results_html += f"<div class='result-item'><span class='rank'>#{r['rank']}</span> {r['source_path']} ({r['score']:.2f})</div>"
                top_results_html += "</div>"

                html += f"""
            <tr>
                <td class="query-text">{q.query}</td>
                <td>{q.expected_doc}</td>
                <td>{top_results_html}</td>
                <td class="fail">{q.p_at_3:.1f}</td>
            </tr>
"""
            html += "</table>"
        else:
            html += "<p style='color: #10b981; margin: 20px 0;'>🎉 All queries passed P@3!</p>"

        html += """
    </div>
</body>
</html>
"""

        return html
