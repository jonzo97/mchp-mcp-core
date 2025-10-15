"""
Table extraction consensus engine with confidence scoring.

Extracts tables using multiple tools and computes confidence scores based on
cross-tool agreement. This addresses the core reliability concern with PDF
table extraction.

The consensus approach provides quantitative confidence metrics that enable
automated decision-making about when manual verification is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mchp_mcp_core.extractors.table_extractors import (
    ExtractionResult,
    ExtractedTable,
    TableExtractor,
    PdfPlumberExtractor,
    CamelotExtractor,
    PyMuPDFExtractor,
)
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TableMatch:
    """
    Represents a matched table across multiple extractors.

    When multiple extractors find the same table (based on position and content),
    this class holds all versions and computed metrics.
    """
    table_index: int
    page_num: int

    # Versions from different extractors
    versions: Dict[str, ExtractedTable] = field(default_factory=dict)

    # Consensus metrics
    agreement_score: float = 0.0  # 0.0-1.0, based on how many extractors agree
    cell_similarity: float = 0.0   # 0.0-1.0, average cell-level similarity
    structure_score: float = 0.0   # 0.0-1.0, consistency of row/col counts

    # Final consensus
    best_version: Optional[ExtractedTable] = None
    confidence: float = 0.0  # 0.0-1.0, overall confidence in this table

    # Metadata
    extractor_count: int = 0
    disagreements: List[str] = field(default_factory=list)


@dataclass
class ConsensusResult:
    """Complete consensus result for a page."""
    page_num: int
    matches: List[TableMatch]
    total_extractors_run: int
    success: bool = True
    error: Optional[str] = None


class TableConsensusEngine:
    """
    Consensus-based table extraction engine.

    Runs multiple extractors on the same PDF and computes confidence scores
    based on cross-tool agreement. Provides quantitative metrics for
    determining when manual verification is needed.

    Confidence Scoring:
        - 0.95-1.0: High confidence, 3+ extractors agree
        - 0.80-0.95: Good confidence, 2+ extractors agree closely
        - 0.60-0.80: Medium confidence, some agreement but differences exist
        - 0.40-0.60: Low confidence, significant disagreement
        - 0.0-0.40: Very low confidence, extractors disagree or only 1 found table

    Example:
        >>> engine = TableConsensusEngine(
        ...     extractors=["pdfplumber", "camelot"],
        ...     min_confidence=0.7
        ... )
        >>> result = engine.extract_with_consensus("doc.pdf", page_num=5)
        >>> for match in result.matches:
        ...     if match.confidence >= 0.8:
        ...         print(f"Table {match.table_index}: High confidence ✓")
        ...     else:
        ...         print(f"Table {match.table_index}: Review needed ⚠")
    """

    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        prefer_extractor: Optional[str] = None
    ):
        """
        Initialize consensus engine.

        Args:
            extractors: List of extractor names to use ("pdfplumber", "camelot", "pymupdf")
                       If None, uses all available extractors
            min_confidence: Minimum confidence threshold (0.0-1.0)
            prefer_extractor: If multiple versions have same confidence, prefer this one
        """
        self.extractors_map = {
            "pdfplumber": PdfPlumberExtractor(),
            "camelot": CamelotExtractor(),
            "pymupdf": PyMuPDFExtractor(),
        }

        # Determine which extractors to use
        if extractors:
            self.active_extractors = {
                name: ext for name, ext in self.extractors_map.items()
                if name in extractors and ext.is_available()
            }
        else:
            # Use all available
            self.active_extractors = {
                name: ext for name, ext in self.extractors_map.items()
                if ext.is_available()
            }

        self.min_confidence = min_confidence
        self.prefer_extractor = prefer_extractor

        logger.info(f"Initialized consensus engine with extractors: {list(self.active_extractors.keys())}")

    def extract_with_consensus(
        self,
        pdf_path: str,
        page_num: int,
        **extractor_kwargs
    ) -> ConsensusResult:
        """
        Extract tables with consensus scoring.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            **extractor_kwargs: Additional options passed to all extractors

        Returns:
            ConsensusResult with matched tables and confidence scores
        """
        # Run all extractors
        extraction_results: Dict[str, ExtractionResult] = {}

        for name, extractor in self.active_extractors.items():
            try:
                result = extractor.extract_tables(pdf_path, page_num, **extractor_kwargs)
                extraction_results[name] = result
                logger.debug(f"{name}: Found {len(result.tables)} tables")
            except Exception as e:
                logger.error(f"Error running {name} extractor: {e}")

        if not extraction_results:
            return ConsensusResult(
                page_num=page_num,
                matches=[],
                total_extractors_run=0,
                success=False,
                error="No extractors ran successfully"
            )

        # Match tables across extractors
        matches = self._match_tables(extraction_results, page_num)

        # Compute confidence for each match
        for match in matches:
            self._compute_confidence(match)

        # Filter by minimum confidence
        if self.min_confidence > 0.0:
            matches = [m for m in matches if m.confidence >= self.min_confidence]

        return ConsensusResult(
            page_num=page_num,
            matches=matches,
            total_extractors_run=len(extraction_results),
            success=True
        )

    def _match_tables(
        self,
        extraction_results: Dict[str, ExtractionResult],
        page_num: int
    ) -> List[TableMatch]:
        """
        Match tables across extractors.

        Tables are matched based on:
        1. Position (bounding box overlap)
        2. Size (similar row/column counts)
        3. Content (cell-level similarity)

        Returns:
            List of TableMatch objects
        """
        # Collect all tables from all extractors
        all_tables: Dict[str, List[ExtractedTable]] = {}
        for name, result in extraction_results.items():
            if result.success:
                all_tables[name] = result.tables

        if not all_tables:
            return []

        # Simple matching strategy: match by table_index for now
        # TODO: Implement bbox-based matching for better accuracy
        matches: Dict[int, TableMatch] = {}

        for extractor_name, tables in all_tables.items():
            for table in tables:
                if table.table_index not in matches:
                    matches[table.table_index] = TableMatch(
                        table_index=table.table_index,
                        page_num=page_num
                    )

                match = matches[table.table_index]
                match.versions[extractor_name] = table
                match.extractor_count = len(match.versions)

        return list(matches.values())

    def _compute_confidence(self, match: TableMatch):
        """
        Compute confidence score for a matched table.

        Confidence is based on:
        1. Agreement score (how many extractors found this table)
        2. Cell similarity (how similar are the extracted values)
        3. Structure score (consistency of dimensions)

        Updates match object in-place with computed scores.
        """
        # 1. Agreement score (0.0-1.0)
        extractor_ratio = match.extractor_count / max(len(self.active_extractors), 1)
        match.agreement_score = extractor_ratio

        # 2. Structure score
        if match.extractor_count >= 2:
            match.structure_score = self._compute_structure_score(match)
        else:
            match.structure_score = 0.5  # Neutral if only one extractor

        # 3. Cell similarity
        if match.extractor_count >= 2:
            match.cell_similarity = self._compute_cell_similarity(match)
        else:
            match.cell_similarity = 0.5  # Neutral if only one extractor

        # Combined confidence (weighted average)
        match.confidence = (
            0.4 * match.agreement_score +
            0.3 * match.structure_score +
            0.3 * match.cell_similarity
        )

        # Select best version
        match.best_version = self._select_best_version(match)

        # If we have a confident match, use that confidence
        if match.best_version and match.best_version.confidence > 0.0:
            # Combine extractor's own confidence with consensus confidence
            match.confidence = (match.confidence + match.best_version.confidence) / 2.0

    def _compute_structure_score(self, match: TableMatch) -> float:
        """
        Compute structure consistency score.

        Checks if row/column counts are consistent across extractors.
        """
        if not match.versions:
            return 0.0

        row_counts = [table.rows for table in match.versions.values()]
        col_counts = [table.columns for table in match.versions.values()]

        # Perfect consistency = 1.0
        if len(set(row_counts)) == 1 and len(set(col_counts)) == 1:
            return 1.0

        # Compute variance
        avg_rows = sum(row_counts) / len(row_counts)
        avg_cols = sum(col_counts) / len(col_counts)

        row_variance = sum((r - avg_rows) ** 2 for r in row_counts) / len(row_counts)
        col_variance = sum((c - avg_cols) ** 2 for c in col_counts) / len(col_counts)

        # Convert variance to score (lower variance = higher score)
        # Allow up to 20% variance for "good" score
        max_acceptable_variance = (avg_rows * 0.2) ** 2
        row_score = max(0.0, 1.0 - (row_variance / max(max_acceptable_variance, 1.0)))

        max_acceptable_variance = (avg_cols * 0.2) ** 2
        col_score = max(0.0, 1.0 - (col_variance / max(max_acceptable_variance, 1.0)))

        return (row_score + col_score) / 2.0

    def _compute_cell_similarity(self, match: TableMatch) -> float:
        """
        Compute cell-level similarity across versions.

        Compares cell contents using fuzzy string matching.
        """
        if len(match.versions) < 2:
            return 0.5

        # Compare all pairs of versions
        versions_list = list(match.versions.values())
        similarities = []

        for i in range(len(versions_list)):
            for j in range(i + 1, len(versions_list)):
                sim = self._compare_tables(versions_list[i], versions_list[j])
                similarities.append(sim)

        # Average similarity across all pairs
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compare_tables(self, table1: ExtractedTable, table2: ExtractedTable) -> float:
        """
        Compare two tables cell-by-cell.

        Returns similarity score (0.0-1.0).
        """
        # If dimensions don't match, penalize heavily
        if table1.rows != table2.rows or table1.columns != table2.columns:
            # Partial credit for similar dimensions
            size_similarity = 1.0 - abs(table1.rows - table2.rows) / max(table1.rows, table2.rows, 1)
            size_similarity *= 1.0 - abs(table1.columns - table2.columns) / max(table1.columns, table2.columns, 1)
            return size_similarity * 0.3  # Max 30% if dimensions differ

        # Compare cell by cell
        matching_cells = 0
        total_cells = 0

        for row_idx in range(min(table1.rows, table2.rows)):
            for col_idx in range(min(table1.columns, table2.columns)):
                cell1 = table1.get_cell(row_idx, col_idx)
                cell2 = table2.get_cell(row_idx, col_idx)

                total_cells += 1

                # Normalize cells
                cell1_norm = (cell1 or "").strip().lower()
                cell2_norm = (cell2 or "").strip().lower()

                # Exact match
                if cell1_norm == cell2_norm:
                    matching_cells += 1
                # Fuzzy match (simple: check if one contains the other)
                elif cell1_norm and cell2_norm and (
                    cell1_norm in cell2_norm or cell2_norm in cell1_norm
                ):
                    matching_cells += 0.7  # Partial credit

        if total_cells == 0:
            return 0.0

        return matching_cells / total_cells

    def _select_best_version(self, match: TableMatch) -> Optional[ExtractedTable]:
        """
        Select the best version from available versions.

        Selection criteria:
        1. Prefer the version with most content (least sparse)
        2. Prefer the version from preferred extractor if specified
        3. Prefer the version with highest individual confidence
        """
        if not match.versions:
            return None

        # If preference specified and available, use it
        if self.prefer_extractor and self.prefer_extractor in match.versions:
            return match.versions[self.prefer_extractor]

        # Otherwise, select based on quality
        best = None
        best_score = -1.0

        for extractor_name, table in match.versions.items():
            # Score = completeness + confidence
            completeness = 1.0 - table.sparsity
            score = (completeness + table.confidence) / 2.0

            if score > best_score:
                best_score = score
                best = table

        return best


__all__ = [
    "TableMatch",
    "ConsensusResult",
    "TableConsensusEngine",
]
