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
from mchp_mcp_core.extractors.table_detection import TableDetector
from mchp_mcp_core.extractors.table_header_detector import TableHeaderDetector
from mchp_mcp_core.extractors.table_multipage import (
    MultiPageTableDetector,
    TableSpan,
    ContinuationType,
)
from mchp_mcp_core.extractors.table_merging import TableMerger, MergeResult
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
    cell_similarity: float = 0.0   # 0.0-1.0, average cell-level similarity (all cells)
    structure_score: float = 0.0   # 0.0-1.0, consistency of row/col counts

    # Separate accuracy tracking (Phase 3A enhancement)
    header_accuracy: float = 0.0   # 0.0-1.0, header row agreement
    data_row_accuracy: float = 0.0 # 0.0-1.0, data row agreement (critical!)
    estimated_header_rows: int = 0 # Number of header rows detected

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


def detect_table_region_consensus(
    extractors: Dict[str, TableExtractor],
    pdf_path: str,
    page_num: int,
    padding: float = 10.0,
    **extractor_kwargs
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect table region using consensus of all extractors (union strategy).

    This solves the problem where extractors report different bounding boxes
    for the same table, leading to misalignment in consensus metrics. By
    computing the UNION of all bboxes, we ensure all extractors look at the
    same region.

    Strategy:
        1. Run all extractors to detect table bboxes
        2. Compute union: (min_x0, min_y0, max_x1, max_y1)
        3. Add padding for safety
        4. Return unified region for all extractors to use

    Why union over alternatives:
        - If one extractor's bbox is too small (like Camelot), others expand it
        - If one extractor misses part of table, others catch it
        - Not dependent on any single extractor being perfect
        - Self-correcting for outliers

    Args:
        extractors: Dict of active extractors
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        padding: Padding to add around union bbox (default: 10 points)
        **extractor_kwargs: Additional options passed to extractors

    Returns:
        Union bbox (x0, y0, x1, y1) or None if no bboxes detected

    Example:
        >>> extractors = {"pdfplumber": PdfPlumberExtractor(), ...}
        >>> bbox = detect_table_region_consensus(extractors, "doc.pdf", 55)
        >>> # bbox: (44.3, 69.5, 569.7, 746.5)  # Union of all extractors
    """
    bboxes = []

    for name, extractor in extractors.items():
        try:
            # Run extractor to get bboxes
            result = extractor.extract_tables(pdf_path, page_num, **extractor_kwargs)

            if result.success and result.tables:
                for table in result.tables:
                    if table.bbox:
                        bboxes.append(table.bbox)
                        logger.debug(f"{name} bbox: {table.bbox}")
        except Exception as e:
            logger.warning(f"Error getting bbox from {name}: {e}")

    if not bboxes:
        logger.debug("No bboxes detected, will use full page")
        return None  # Fallback to full page

    # Compute union: (min_x0, min_y0, max_x1, max_y1)
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)

    # Add padding for safety
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 += padding
    y1 += padding

    union_bbox = (x0, y0, x1, y1)
    logger.info(f"Consensus bbox (union of {len(bboxes)} extractors): {union_bbox}")

    return union_bbox


def extract_table_title(
    pdf_path: str,
    page_num: int,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Optional[str]:
    """
    Extract table title from PDF page.

    Searches for "Table X-Y. Description" pattern in text above the table bbox.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        bbox: Table bounding box (x0, y0, x1, y1). If provided, searches region above bbox.

    Returns:
        Table title string or None if not found
    """
    import re
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            logger.warning(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
            return None

        page = doc[page_num]

        # Define search region
        if bbox:
            # Search region above the table (expand upward by 100 points)
            search_bbox = (bbox[0], max(0, bbox[1] - 100), bbox[2], bbox[1])
        else:
            # Search entire page if no bbox provided
            search_bbox = None

        # Extract text from search region
        if search_bbox:
            text = page.get_text("text", clip=search_bbox)
        else:
            text = page.get_text("text")

        doc.close()

        # Search for table title pattern: "Table X-Y. Description"
        # Also match variations: "Table X. Description", "TABLE X-Y. Description"
        pattern = r'Table\s+(\d+(?:-\d+)?)\.\s+([^\n]+)'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        # Find the match closest to the table (lowest Y coordinate in search region)
        # For now, just return the first match found
        for match in matches:
            title = match.group(0).strip()
            logger.debug(f"Found table title: {title}")
            return title

        logger.debug("No table title found")
        return None

    except Exception as e:
        logger.error(f"Error extracting table title: {e}")
        return None


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
        prefer_extractor: Optional[str] = None,
        enable_detection_filter: bool = True,
        use_consensus_bbox: bool = True
    ):
        """
        Initialize consensus engine.

        Args:
            extractors: List of extractor names to use ("pdfplumber", "camelot", "pymupdf")
                       If None, uses all available extractors
            min_confidence: Minimum confidence threshold (0.0-1.0)
            prefer_extractor: If multiple versions have same confidence, prefer this one
            enable_detection_filter: Enable pre-extraction validation to filter false positives
            use_consensus_bbox: Use consensus bbox detection (union strategy) to fix misalignment
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
        self.use_consensus_bbox = use_consensus_bbox

        # Table detection filter (Phase 1 enhancement)
        self.enable_detection_filter = enable_detection_filter
        self.detector = TableDetector(
            min_rows=2,
            min_columns=2,
            max_sparsity=0.70,
            min_content_ratio=0.15,
            strict_mode=False
        ) if enable_detection_filter else None

        # Multi-page table detection (Phase 3A enhancement)
        self.multipage_detector = MultiPageTableDetector(
            column_match_threshold=0.80,
            position_tolerance=0.05,
            min_confidence=0.70
        )
        self.table_merger = TableMerger(
            preserve_headers=True,
            align_columns=True
        )

        logger.info(f"Initialized consensus engine with extractors: {list(self.active_extractors.keys())}")
        if enable_detection_filter:
            logger.info("Detection filter enabled (filters false positives)")
        if use_consensus_bbox:
            logger.info("Consensus bbox detection enabled (fixes extractor misalignment)")

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
        # Step 1: Detect consensus bbox if enabled
        consensus_bbox = None
        if self.use_consensus_bbox:
            try:
                consensus_bbox = detect_table_region_consensus(
                    self.active_extractors,
                    pdf_path,
                    page_num,
                    **extractor_kwargs
                )
                if consensus_bbox:
                    logger.info(f"Using consensus bbox: {consensus_bbox}")
            except Exception as e:
                logger.warning(f"Error detecting consensus bbox: {e}, falling back to full page")

        # Step 2: Run all extractors with unified region (if detected)
        extraction_results: Dict[str, ExtractionResult] = {}

        for name, extractor in self.active_extractors.items():
            try:
                # Pass consensus_bbox as 'region' parameter if detected
                kwargs = extractor_kwargs.copy()
                if consensus_bbox:
                    kwargs['region'] = consensus_bbox

                result = extractor.extract_tables(pdf_path, page_num, **kwargs)
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

        # Apply detection filter to remove false positives
        if self.detector:
            filtered_matches = []
            for match in matches:
                # Validate each version
                validation_results = {}
                for extractor_name, table in match.versions.items():
                    validation_results[extractor_name] = self.detector.validate(table)

                # Keep match if at least one version passes validation
                valid_versions = {
                    name: table
                    for name, table in match.versions.items()
                    if validation_results[name].is_valid_table
                }

                if valid_versions:
                    # Update match with only valid versions
                    match.versions = valid_versions
                    match.extractor_count = len(valid_versions)
                    filtered_matches.append(match)

                    # Log filtered versions
                    if len(valid_versions) < len(validation_results):
                        invalid_count = len(validation_results) - len(valid_versions)
                        logger.debug(
                            f"Detection filter removed {invalid_count} false positive(s) "
                            f"for table {match.table_index} on page {page_num}"
                        )
                else:
                    logger.debug(
                        f"Detection filter rejected table {match.table_index} "
                        f"on page {page_num}: all versions failed validation"
                    )

            matches = filtered_matches

        # Extract table titles for each match
        for match in matches:
            # Use consensus bbox if available, otherwise use first available bbox
            bbox_for_title = consensus_bbox if consensus_bbox else None
            if not bbox_for_title and match.versions:
                # Try to get bbox from first version that has one
                for table in match.versions.values():
                    if table.bbox:
                        bbox_for_title = table.bbox
                        break

            # Extract title
            title = extract_table_title(pdf_path, page_num, bbox_for_title)

            # Assign title to all versions
            if title:
                for table in match.versions.values():
                    table.caption = title
                logger.debug(f"Assigned title to table {match.table_index}: {title}")

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

    def extract_with_multipage_detection(
        self,
        pdf_path: str,
        page_range: Tuple[int, int],
        extractor_name: str = "pdfplumber",
        **extractor_kwargs
    ) -> Dict[int, ConsensusResult]:
        """
        Extract tables with multi-page continuation detection.

        This method enhances standard consensus extraction by detecting when
        tables span multiple pages and merging them into unified tables.

        Process:
        1. Run consensus extraction on each page individually
        2. Detect multi-page table spans using structural analysis
        3. Merge detected continuations into single tables
        4. Re-compute consensus on merged tables
        5. Return improved confidence scores

        Args:
            pdf_path: Path to PDF file
            page_range: (start_page, end_page) inclusive, 0-indexed
            extractor_name: Which extractor to use for continuation detection
                           (default: "pdfplumber", most reliable for structure)
            **extractor_kwargs: Additional options passed to extractors

        Returns:
            Dict mapping page_num -> ConsensusResult
            Multi-page tables appear on their start_page with merged data

        Example:
            >>> engine = TableConsensusEngine()
            >>> # Page 21-22 have a 2-page table
            >>> results = engine.extract_with_multipage_detection(
            ...     pdf_path="datasheet.pdf",
            ...     page_range=(21, 22)
            ... )
            >>> # Page 21 result contains merged 89-row table
            >>> # Page 22 result is empty (merged into page 21)
            >>> print(f"Page 21: {results[21].matches[0].best_version.rows} rows")
            89
        """
        start_page, end_page = page_range

        # Step 1: Extract tables from each page independently
        logger.info(f"Extracting tables from pages {start_page}-{end_page}")
        page_results: Dict[int, ConsensusResult] = {}
        tables_by_page: Dict[int, List[ExtractedTable]] = {}

        for page_num in range(start_page, end_page + 1):
            result = self.extract_with_consensus(pdf_path, page_num, **extractor_kwargs)
            page_results[page_num] = result

            # Collect tables from specified extractor for multi-page detection
            if result.success and result.matches:
                tables_by_page[page_num] = [
                    match.versions.get(extractor_name) or match.best_version
                    for match in result.matches
                    if match.versions.get(extractor_name) or match.best_version
                ]

        if not tables_by_page:
            logger.warning("No tables found on any page in range")
            return page_results

        # Step 2: Detect multi-page table spans
        logger.info(f"Detecting multi-page continuations using {extractor_name} extractor")
        spans = self.multipage_detector.detect_spans(tables_by_page, extractor_name)

        if not spans:
            logger.info("No multi-page continuations detected")
            return page_results

        logger.info(f"Detected {len(spans)} multi-page table span(s)")

        # Step 3: Merge detected spans
        for span in spans:
            if not span.is_multipage:
                continue

            logger.info(
                f"Merging {span.continuation_type.value} table: "
                f"pages {span.start_page}-{span.end_page} "
                f"(confidence={span.confidence:.2f})"
            )

            # Collect tables to merge
            tables_to_merge = []
            for page_num in range(span.start_page, span.end_page + 1):
                if page_num in tables_by_page:
                    # Find table matching the span's index
                    for table in tables_by_page[page_num]:
                        if table.table_index in span.table_indices:
                            tables_to_merge.append(table)
                            break

            if len(tables_to_merge) < 2:
                logger.warning(f"Insufficient tables to merge for span {span.start_page}-{span.end_page}")
                continue

            # Merge tables
            try:
                merge_result = self.table_merger.merge_span(tables_to_merge, span)

                logger.info(
                    f"Merged {len(tables_to_merge)} tables: "
                    f"{merge_result.merged_table.rows} rows, "
                    f"{merge_result.merged_table.columns} columns"
                )

                if merge_result.issues:
                    for issue in merge_result.issues:
                        logger.debug(f"  Merge issue: {issue}")

                # Step 4: Update results with merged table
                # Replace the match on start_page with merged version
                start_result = page_results[span.start_page]

                # Find the match to replace
                for match in start_result.matches:
                    if match.table_index == span.table_indices[0]:
                        # Update all versions with merged table
                        for extractor_name_key in match.versions.keys():
                            match.versions[extractor_name_key] = merge_result.merged_table

                        # Re-compute confidence (should be higher now)
                        old_confidence = match.confidence
                        self._compute_confidence(match)

                        logger.info(
                            f"Updated table {match.table_index} on page {span.start_page}: "
                            f"confidence {old_confidence:.2f} → {match.confidence:.2f}"
                        )
                        break

                # Remove tables from continuation pages (they've been merged)
                for page_num in range(span.start_page + 1, span.end_page + 1):
                    if page_num in page_results:
                        cont_result = page_results[page_num]
                        cont_result.matches = [
                            m for m in cont_result.matches
                            if m.table_index not in span.table_indices[1:]
                        ]

            except Exception as e:
                logger.error(f"Failed to merge span {span.start_page}-{span.end_page}: {e}")

        return page_results

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

        # 4. Separate header/data accuracy (Phase 3A enhancement)
        if match.extractor_count >= 2:
            header_acc, data_acc, header_rows = self._compute_header_data_accuracy(match)
            match.header_accuracy = header_acc
            match.data_row_accuracy = data_acc
            match.estimated_header_rows = header_rows
        else:
            match.header_accuracy = 0.5
            match.data_row_accuracy = 0.5
            match.estimated_header_rows = 1  # Assume 1 header row

        # Combined confidence (weighted average)
        # Weight data row accuracy higher (70%) than header accuracy (30%)
        # since data accuracy is what matters for production
        combined_similarity = 0.3 * match.header_accuracy + 0.7 * match.data_row_accuracy

        match.confidence = (
            0.4 * match.agreement_score +
            0.3 * match.structure_score +
            0.3 * combined_similarity
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

    def _align_data_rows_by_content(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable,
        header_rows1: int,
        header_rows2: int
    ) -> List[Tuple[int, int]]:
        """
        Align data rows between two tables based on content matching.

        Uses first column as anchor to match rows by content (e.g., PA0 to PA0)
        instead of by index. This handles cases where extractors have different
        numbers of header rows.

        Args:
            table1: First table
            table2: Second table
            header_rows1: Number of header rows in table1
            header_rows2: Number of header rows in table2

        Returns:
            List of (row_idx1, row_idx2) tuples for matching data rows
        """
        if not table1.data or not table2.data:
            return []

        # Skip headers, extract data rows
        data1 = table1.data[header_rows1:] if len(table1.data) > header_rows1 else []
        data2 = table2.data[header_rows2:] if len(table2.data) > header_rows2 else []

        if not data1 or not data2:
            return []

        # Create anchor index using first column values
        # Format: {anchor_value: data_row_index}
        anchor1 = {}
        for i, row in enumerate(data1):
            if row and len(row) > 0:
                anchor_val = str(row[0]).strip()
                if anchor_val:  # Skip empty anchors
                    anchor1[anchor_val] = i

        anchor2 = {}
        for i, row in enumerate(data2):
            if row and len(row) > 0:
                anchor_val = str(row[0]).strip()
                if anchor_val:
                    anchor2[anchor_val] = i

        # Find common anchors
        common_anchors = set(anchor1.keys()) & set(anchor2.keys())

        if not common_anchors:
            logger.debug("No common anchors found for content-based alignment")
            return []

        # Build alignment mapping
        # Convert data row indices back to absolute row indices
        alignment = [
            (anchor1[key] + header_rows1, anchor2[key] + header_rows2)
            for key in sorted(common_anchors)
        ]

        logger.debug(
            f"Aligned {len(alignment)} data rows by content "
            f"(out of {len(data1)} and {len(data2)} data rows)"
        )

        return alignment

    def _compute_header_data_accuracy(self, match: TableMatch) -> Tuple[float, float, int]:
        """
        Compute separate accuracy for header rows vs data rows.

        Uses per-extractor header detection and content-based row alignment
        to accurately compare tables even when extractors report different
        header row counts.

        Headers can have formatting differences (multi-row, merged cells)
        while data rows should be identical. This separates the two concerns.

        Returns:
            (header_accuracy, data_row_accuracy, estimated_header_rows)
        """
        if len(match.versions) < 2:
            return (0.5, 0.5, 1)

        # Step 1: Per-extractor header detection using TableHeaderDetector
        header_detector = TableHeaderDetector(max_header_rows=10)
        header_info = {}

        for name, table in match.versions.items():
            if table.data:
                result = header_detector.detect_header_rows(table.data)
                detected_headers = len(result.header_row_indices)

                # Safety check: prevent detecting ALL rows as headers
                # Always leave at least 1 row as data, or max 80% as headers
                max_allowed_headers = max(1, int(len(table.data) * 0.8))
                if detected_headers >= len(table.data):
                    detected_headers = min(detected_headers, max_allowed_headers)
                    logger.warning(
                        f"{name}: Header detector marked all {len(table.data)} rows as headers, "
                        f"capping at {detected_headers} to preserve data rows"
                    )

                header_info[name] = {
                    'header_row_count': detected_headers,
                    'confidence': result.confidence,
                    'indices': result.header_row_indices[:detected_headers]  # Trim if capped
                }
                logger.debug(
                    f"{name}: Detected {detected_headers} header rows "
                    f"(confidence: {result.confidence:.2f})"
                )
            else:
                header_info[name] = {'header_row_count': 1, 'confidence': 0.5, 'indices': [0]}

        # Step 2: Determine consensus header structure
        # Prefer pdfplumber/pymupdf, ignore Camelot for header structure
        reference_header_rows = None
        for name in ['pdfplumber', 'pymupdf']:
            if name in header_info:
                reference_header_rows = header_info[name]['header_row_count']
                logger.debug(f"Using {name} header structure: {reference_header_rows} rows")
                break

        if reference_header_rows is None:
            # Fallback: use median header row count
            counts = [info['header_row_count'] for info in header_info.values()]
            reference_header_rows = sorted(counts)[len(counts) // 2]
            logger.debug(f"Using median header structure: {reference_header_rows} rows")

        # Step 3: Compute header accuracy (structural similarity)
        # Compare header row counts and structure
        header_similarities = []
        versions_list = list(match.versions.items())

        for i in range(len(versions_list)):
            for j in range(i + 1, len(versions_list)):
                name_i, table_i = versions_list[i]
                name_j, table_j = versions_list[j]

                headers_i = header_info[name_i]['header_row_count']
                headers_j = header_info[name_j]['header_row_count']

                # Structural similarity: how close are header counts?
                max_headers = max(headers_i, headers_j)
                if max_headers > 0:
                    structural_sim = 1.0 - abs(headers_i - headers_j) / max_headers
                    header_similarities.append(structural_sim)

        header_accuracy = sum(header_similarities) / len(header_similarities) if header_similarities else 0.5

        # Step 4: Compute data row accuracy using content-based alignment
        data_similarities = []

        for i in range(len(versions_list)):
            for j in range(i + 1, len(versions_list)):
                name_i, table_i = versions_list[i]
                name_j, table_j = versions_list[j]

                headers_i = header_info[name_i]['header_row_count']
                headers_j = header_info[name_j]['header_row_count']

                # Get content-based alignment
                alignment = self._align_data_rows_by_content(
                    table_i, table_j, headers_i, headers_j
                )

                if alignment:
                    # Compare aligned rows
                    matching_rows = 0
                    for row_i, row_j in alignment:
                        if row_i < len(table_i.data) and row_j < len(table_j.data):
                            # Compare cells in the aligned rows
                            data_i = table_i.data[row_i]
                            data_j = table_j.data[row_j]

                            # Cell-level comparison
                            matching_cells = 0
                            total_cells = max(len(data_i), len(data_j))

                            for k in range(min(len(data_i), len(data_j))):
                                cell_i = str(data_i[k]).strip()
                                cell_j = str(data_j[k]).strip()
                                if cell_i == cell_j:
                                    matching_cells += 1

                            row_similarity = matching_cells / total_cells if total_cells > 0 else 0.0
                            if row_similarity >= 0.8:  # Consider row matched if 80%+ cells match
                                matching_rows += 1

                    # Data accuracy: ratio of matching rows
                    data_accuracy = matching_rows / len(alignment) if alignment else 0.0
                    data_similarities.append(data_accuracy)

                    logger.debug(
                        f"{name_i} vs {name_j}: {matching_rows}/{len(alignment)} "
                        f"aligned rows match (accuracy: {data_accuracy:.2f})"
                    )
                else:
                    # Fallback: use index-based comparison if no alignment found
                    logger.warning(
                        f"No content-based alignment found for {name_i} vs {name_j}, "
                        "falling back to index-based comparison"
                    )
                    min_data_rows = min(
                        len(table_i.data) - headers_i,
                        len(table_j.data) - headers_j
                    )
                    if min_data_rows > 0:
                        fallback_sim = self._compare_table_rows(
                            table_i, table_j,
                            start_row=headers_i,
                            end_row=headers_i + min_data_rows
                        )
                        data_similarities.append(fallback_sim)

        data_row_accuracy = sum(data_similarities) / len(data_similarities) if data_similarities else 0.5

        return (header_accuracy, data_row_accuracy, reference_header_rows)

    def _compare_table_rows(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable,
        start_row: int,
        end_row: int
    ) -> float:
        """
        Compare specific rows between two tables.

        Args:
            table1: First table
            table2: Second table
            start_row: Starting row index (inclusive)
            end_row: Ending row index (exclusive)

        Returns:
            Similarity score (0.0-1.0)
        """
        matching_cells = 0
        total_cells = 0

        for row_idx in range(start_row, min(end_row, len(table1.data if table1.data else []), len(table2.data if table2.data else []))):
            if row_idx >= len(table1.data) or row_idx >= len(table2.data):
                continue

            row1 = table1.data[row_idx]
            row2 = table2.data[row_idx]

            for col_idx in range(min(len(row1), len(row2))):
                cell1 = row1[col_idx] if col_idx < len(row1) else ""
                cell2 = row2[col_idx] if col_idx < len(row2) else ""

                total_cells += 1

                # Normalize
                cell1_norm = str(cell1).strip().lower()
                cell2_norm = str(cell2).strip().lower()

                if cell1_norm == cell2_norm:
                    matching_cells += 1
                elif cell1_norm and cell2_norm and (cell1_norm in cell2_norm or cell2_norm in cell1_norm):
                    matching_cells += 0.7

        if total_cells == 0:
            return 1.0  # No cells to compare, assume match

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


class HybridConsensusEngine(TableConsensusEngine):
    """
    Consensus engine with optional LLM fallback for low-confidence extractions.

    Extends TableConsensusEngine to use vision LLMs as an additional extractor
    or tiebreaker when traditional methods produce low-confidence results.

    Workflow:
    1. Run traditional consensus (pdfplumber, camelot, pymupdf)
    2. Compute confidence scores
    3. If confidence < threshold AND LLM enabled:
       → Use LLM as additional extractor
       → Recompute consensus with LLM result
    4. Return best version with improved confidence

    Privacy Features:
    - LLM is opt-in (disabled by default)
    - Respects local-only and allow_cloud flags
    - PII redaction before LLM calls
    - Audit logging when LLM is used

    Example:
        >>> from mchp_mcp_core.extractors import HybridConsensusEngine
        >>> from mchp_mcp_core.extractors.table_llm import VisionLLMConfig, VisionLLMProvider
        >>>
        >>> llm_config = VisionLLMConfig(
        ...     enabled=True,
        ...     provider=VisionLLMProvider.OPENAI,
        ...     model="gpt-4o",
        ...     api_key="sk-...",
        ...     allow_cloud=True
        ... )
        >>> engine = HybridConsensusEngine(
        ...     llm_config=llm_config,
        ...     llm_fallback_threshold=0.70
        ... )
        >>> result = engine.extract_with_consensus("doc.pdf", page_num=5)
        >>> # LLM used automatically if confidence < 0.70
    """

    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        llm_config: Optional[Any] = None,  # VisionLLMConfig
        llm_fallback_threshold: float = 0.70,
        llm_use_for_no_results: bool = True,
        min_confidence: float = 0.0,
        prefer_extractor: Optional[str] = None
    ):
        """
        Initialize hybrid consensus engine.

        Args:
            extractors: List of traditional extractor names
            llm_config: VisionLLMConfig for LLM fallback (None = disabled)
            llm_fallback_threshold: Use LLM if consensus confidence < this value
            llm_use_for_no_results: Use LLM if no traditional extractors find tables
            min_confidence: Minimum confidence threshold for results
            prefer_extractor: Preferred extractor name if tie
        """
        super().__init__(extractors, min_confidence, prefer_extractor)

        # LLM configuration
        self.llm_config = llm_config
        self.llm_fallback_threshold = llm_fallback_threshold
        self.llm_use_for_no_results = llm_use_for_no_results

        # Initialize LLM extractor if configured
        self.llm_extractor = None
        if llm_config and llm_config.enabled:
            try:
                from mchp_mcp_core.extractors.table_llm import VisionLLMTableExtractor
                self.llm_extractor = VisionLLMTableExtractor(llm_config)

                if self.llm_extractor.is_available():
                    logger.info(f"LLM fallback enabled: provider={llm_config.provider.value}, threshold={llm_fallback_threshold:.2f}")
                else:
                    logger.warning("LLM fallback configured but not available")
                    self.llm_extractor = None
            except Exception as e:
                logger.error(f"Failed to initialize LLM extractor: {e}")
                self.llm_extractor = None

    def extract_with_consensus(
        self,
        pdf_path: str,
        page_num: int,
        **extractor_kwargs
    ) -> ConsensusResult:
        """
        Extract tables with consensus and optional LLM fallback.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            **extractor_kwargs: Additional options for extractors

        Returns:
            ConsensusResult with improved confidence from LLM if used
        """
        # Run traditional consensus first
        result = super().extract_with_consensus(pdf_path, page_num, **extractor_kwargs)

        # Check if LLM fallback should be used
        if not self.llm_extractor:
            return result  # LLM not available

        # Case 1: No tables found by any extractor
        if not result.matches and self.llm_use_for_no_results:
            logger.info("No tables found by traditional extractors, trying LLM fallback")
            return self._extract_with_llm_only(pdf_path, page_num, result)

        # Case 2: Low confidence tables
        llm_used = False
        for match in result.matches:
            if match.confidence < self.llm_fallback_threshold:
                logger.info(f"Table {match.table_index}: Low confidence ({match.confidence:.2f}), using LLM fallback")
                self._add_llm_to_match(match, pdf_path, page_num)
                llm_used = True

        if llm_used:
            logger.info("LLM fallback applied to low-confidence tables")

        return result

    def _extract_with_llm_only(
        self,
        pdf_path: str,
        page_num: int,
        original_result: ConsensusResult
    ) -> ConsensusResult:
        """
        Extract using LLM when traditional extractors found nothing.

        Args:
            pdf_path: Path to PDF
            page_num: Page number
            original_result: Original consensus result (no matches)

        Returns:
            Updated ConsensusResult with LLM results
        """
        try:
            llm_result = self.llm_extractor.extract_tables(pdf_path, page_num)

            if llm_result.success and llm_result.tables:
                logger.info(f"LLM found {len(llm_result.tables)} tables")

                # Create matches from LLM results
                for table in llm_result.tables:
                    match = TableMatch(
                        table_index=table.table_index,
                        page_num=page_num
                    )
                    match.versions["vision_llm"] = table
                    match.extractor_count = 1
                    match.best_version = table

                    # Set confidence based on LLM's own confidence
                    match.confidence = table.confidence
                    match.agreement_score = 0.5  # Only 1 extractor
                    match.structure_score = 0.8   # Assume LLM got structure right
                    match.cell_similarity = 0.8   # Not applicable for single extractor

                    original_result.matches.append(match)

                original_result.success = True

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")

        return original_result

    def _add_llm_to_match(
        self,
        match: TableMatch,
        pdf_path: str,
        page_num: int
    ):
        """
        Add LLM result to an existing match and recompute confidence.

        Args:
            match: TableMatch to enhance with LLM
            pdf_path: Path to PDF
            page_num: Page number
        """
        try:
            # Extract with LLM
            llm_result = self.llm_extractor.extract_tables(pdf_path, page_num)

            if llm_result.success and llm_result.tables:
                # Find table matching the index
                llm_table = None
                for table in llm_result.tables:
                    if table.table_index == match.table_index:
                        llm_table = table
                        break

                if not llm_table and len(llm_result.tables) > 0:
                    # If index doesn't match, use first table
                    llm_table = llm_result.tables[0]

                if llm_table:
                    # Add LLM result to versions
                    match.versions["vision_llm"] = llm_table
                    match.extractor_count = len(match.versions)

                    # Recompute confidence with LLM included
                    self._compute_confidence(match)

                    logger.debug(f"Added LLM result to table {match.table_index}, new confidence: {match.confidence:.2f}")

        except Exception as e:
            logger.error(f"Failed to add LLM result to match: {e}")


__all__ = [
    "TableMatch",
    "ConsensusResult",
    "TableConsensusEngine",
    "HybridConsensusEngine",
]
