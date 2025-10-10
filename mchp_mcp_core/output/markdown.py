"""
Markdown generation module for document reassembly.

Provides utilities for converting search results and document chunks back into
structured markdown documents with table of contents, headers, and formatting.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Set

from mchp_mcp_core.storage import SearchResult, DocumentChunk
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class MarkdownGenerator:
    """
    Generates markdown documents from search results or document chunks.

    Features:
    - Document header with metadata
    - Automatic table of contents generation
    - Section hierarchy detection
    - Page/slide reference comments
    - Cross-reference report integration
    - Configurable formatting templates

    Example:
        >>> generator = MarkdownGenerator(
        ...     include_toc=True,
        ...     include_page_refs=True
        ... )
        >>> markdown = generator.generate_from_results(
        ...     results=search_results,
        ...     title="PolarFire FPGA Documentation",
        ...     metadata={"product": "PolarFire", "version": "v2.0"}
        ... )
        >>> with open("output.md", "w") as f:
        ...     f.write(markdown)
    """

    def __init__(
        self,
        include_toc: bool = True,
        include_page_refs: bool = True,
        toc_min_sections: int = 3,
        section_patterns: Optional[List[str]] = None
    ):
        """
        Initialize markdown generator.

        Args:
            include_toc: Whether to generate table of contents
            include_page_refs: Whether to include page reference comments
            toc_min_sections: Minimum number of sections to generate TOC
            section_patterns: Regex patterns to exclude from TOC (e.g., "Figure \\d+")
        """
        self.include_toc = include_toc
        self.include_page_refs = include_page_refs
        self.toc_min_sections = toc_min_sections
        self.section_patterns = section_patterns or [
            r'^Figure\s+\d+$',  # "Figure 1", "Figure 2", etc.
            r'^Table\s+\d+$',   # "Table 1", "Table 2", etc.
            r'^Unknown Section$',
            r'^\[Figure\s+\d+\]$',
            r'^\[Table\s+\d+\]$'
        ]

    def generate_from_results(
        self,
        results: List[SearchResult],
        title: str = "Document",
        metadata: Optional[Dict] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Generate markdown document from search results.

        Args:
            results: List of search results
            title: Document title
            metadata: Optional metadata dictionary
            query: Original search query (if applicable)

        Returns:
            Complete markdown document as string
        """
        metadata = metadata or {}
        parts = []

        # Generate header
        parts.append(self._generate_header(title, metadata, query))

        # Generate table of contents if enabled
        if self.include_toc and len(results) >= self.toc_min_sections:
            toc = self._generate_toc_from_results(results)
            if toc:
                parts.append(toc)
                parts.append("\n---\n")

        # Add results grouped by document
        parts.append(self._format_results(results))

        return "".join(parts)

    def generate_from_chunks(
        self,
        chunks: List[DocumentChunk],
        title: str = "Document",
        metadata: Optional[Dict] = None,
        cross_ref_report: Optional[Dict] = None
    ) -> str:
        """
        Generate markdown document from document chunks.

        Args:
            chunks: List of document chunks (sorted by page/position)
            title: Document title
            metadata: Optional metadata dictionary
            cross_ref_report: Optional cross-reference validation report

        Returns:
            Complete markdown document as string
        """
        metadata = metadata or {}
        parts = []

        # Generate header
        parts.append(self._generate_header(title, metadata))

        # Generate table of contents
        if self.include_toc and len(chunks) >= self.toc_min_sections:
            toc = self._generate_toc_from_chunks(chunks)
            if toc:
                parts.append(toc)
                parts.append("\n---\n")

        # Add content
        parts.append(self._format_chunks(chunks))

        # Add cross-reference report if provided
        if cross_ref_report:
            parts.append("\n---\n")
            parts.append(self._generate_crossref_report(cross_ref_report))

        return "".join(parts)

    def _generate_header(
        self,
        title: str,
        metadata: Dict,
        query: Optional[str] = None
    ) -> str:
        """Generate document header with metadata."""
        header_lines = [f"# {title}\n\n"]

        # Add query if provided
        if query:
            header_lines.append(f"**Query:** {query}  \n")

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                header_lines.append(f"**{key.replace('_', ' ').title()}:** {value}  \n")

        # Add generation timestamp
        header_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")

        header_lines.append("\n---\n\n")
        return "".join(header_lines)

    def _generate_toc_from_results(self, results: List[SearchResult]) -> str:
        """Generate table of contents from search results."""
        toc_lines = ["## Table of Contents\n\n"]
        seen_docs = set()

        for result in results:
            doc_title = result.title
            if doc_title not in seen_docs:
                seen_docs.add(doc_title)
                anchor = self._make_anchor(doc_title)
                toc_lines.append(f"- [{doc_title}](#{anchor})\n")

        if len(seen_docs) < self.toc_min_sections:
            return ""

        return "".join(toc_lines) + "\n"

    def _generate_toc_from_chunks(self, chunks: List[DocumentChunk]) -> str:
        """Generate table of contents from document chunks."""
        toc_lines = ["## Table of Contents\n\n"]
        sections_seen: Set[str] = set()

        for chunk in chunks:
            # Extract section from metadata or text
            section = self._extract_section(chunk)

            if section and section not in sections_seen:
                # Check if section should be excluded
                if not self._should_exclude_section(section):
                    sections_seen.add(section)
                    anchor = self._make_anchor(section)
                    toc_lines.append(f"- [{section}](#{anchor})\n")

        if len(sections_seen) < self.toc_min_sections:
            return ""

        return "".join(toc_lines) + "\n"

    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results grouped by document."""
        output_parts = []
        current_doc = None

        for result in results:
            # Add document header if changed
            if result.title != current_doc:
                current_doc = result.title
                anchor = self._make_anchor(current_doc)
                output_parts.append(f"\n## <a id=\"{anchor}\"></a>{current_doc}\n\n")
                output_parts.append(f"**Source:** {result.source_path}  \n")
                output_parts.append(f"**Last Updated:** {result.updated_at or 'N/A'}  \n\n")

            # Add result content
            if self.include_page_refs:
                output_parts.append(f"<!-- {result.location} | Score: {result.score:.3f} -->\n")

            output_parts.append(f"### {result.location}\n\n")
            output_parts.append(f"{result.snippet}\n\n")

        return "".join(output_parts)

    def _format_chunks(self, chunks: List[DocumentChunk]) -> str:
        """Format document chunks with section headers."""
        output_parts = []
        current_section = None

        for chunk in chunks:
            # Add section header if changed
            section = self._extract_section(chunk)
            if section and section != current_section:
                current_section = section
                anchor = self._make_anchor(section)
                output_parts.append(f"\n## <a id=\"{anchor}\"></a>{section}\n\n")

            # Add page reference
            if self.include_page_refs:
                page_label = "Slide" if chunk.source_path.endswith(('.pptx', '.ppt')) else "Page"
                output_parts.append(f"<!-- {page_label} {chunk.slide_or_page} -->\n")

            # Add content
            output_parts.append(f"{chunk.text}\n\n")

        return "".join(output_parts)

    def _generate_crossref_report(self, report: Dict) -> str:
        """Generate cross-reference validation report."""
        lines = ["# Cross-Reference Validation Report\n\n"]

        total = report.get('total_references', 0)
        valid = report.get('valid_references', 0)
        invalid = report.get('invalid_references', 0)

        lines.append(f"**Total References:** {total}  \n")
        lines.append(f"**Valid:** {valid}  \n")
        lines.append(f"**Invalid:** {invalid}  \n\n")

        # Add broken references if any
        broken = report.get('broken_references', [])
        if broken:
            lines.append("## Broken References\n\n")
            lines.append("| Reference | Type | Target | Page | Chunk ID |\n")
            lines.append("|-----------|------|--------|------|----------|\n")

            for ref in broken:
                lines.append(
                    f"| {ref['text']} | {ref['type']} | {ref['target']} | "
                    f"{ref['page']} | {ref['chunk_id']} |\n"
                )

        # Add statistics by type
        by_type = report.get('by_type', {})
        if by_type:
            lines.append("\n## References by Type\n\n")
            lines.append("| Type | Total | Valid | Invalid |\n")
            lines.append("|------|-------|-------|---------|\ n")

            for ref_type, stats in by_type.items():
                lines.append(
                    f"| {ref_type} | {stats['total']} | {stats['valid']} | "
                    f"{stats['invalid']} |\n"
                )

        return "".join(lines)

    def generate_summary_report(
        self,
        stats: Dict,
        timing: Optional[Dict] = None
    ) -> str:
        """
        Generate summary report with statistics.

        Args:
            stats: Statistics dictionary with document/processing info
            timing: Optional timing data dictionary

        Returns:
            Markdown formatted summary report
        """
        lines = ["# Summary Report\n\n"]
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

        # Document stats
        if 'document' in stats:
            doc = stats['document']
            lines.append("## Document Information\n\n")
            lines.append(f"**Filename:** {doc.get('filename', 'N/A')}  \n")
            lines.append(f"**Total Pages:** {doc.get('total_pages', 'N/A')}  \n")
            lines.append(f"**Total Chunks:** {doc.get('total_chunks', 'N/A')}  \n\n")

        # Processing stats
        if 'processing' in stats:
            proc = stats['processing']
            lines.append("## Processing Statistics\n\n")
            lines.append(f"**Total Processed:** {proc.get('total', 0)}  \n")
            lines.append(f"**Completed:** {proc.get('completed', 0)}  \n")
            lines.append(f"**Failed:** {proc.get('failed', 0)}  \n")

            if timing:
                lines.append(f"\n**Processing Time:** {timing.get('total_time', 0):.1f} seconds\n")
                lines.append(f"**Speed:** {timing.get('items_per_second', 0):.2f} items/second  \n\n")

        return "".join(lines)

    def _extract_section(self, chunk: DocumentChunk) -> Optional[str]:
        """Extract section name from chunk metadata or text."""
        # Try metadata first
        if hasattr(chunk, 'section_hierarchy'):
            return getattr(chunk, 'section_hierarchy')

        # Try extracting from text (look for heading patterns)
        text = chunk.text.strip()
        lines = text.split('\n', 1)
        if lines and len(lines[0]) < 100:  # Reasonable heading length
            first_line = lines[0].strip()
            # Check if it looks like a heading (short, no special chars, etc.)
            if first_line and not any(c in first_line for c in ['(', ')', ':', ';', '.']):
                return first_line

        return None

    def _should_exclude_section(self, section: str) -> bool:
        """Check if section should be excluded from TOC."""
        for pattern in self.section_patterns:
            if re.match(pattern, section):
                return True

        # Exclude very short sections
        if len(section) < 3:
            return True

        return False

    def _make_anchor(self, text: str) -> str:
        """Create URL-safe anchor from text."""
        # Convert to lowercase, replace spaces and special chars
        anchor = text.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')


def format_as_citation(result: SearchResult) -> str:
    """
    Format search result as a citation string.

    Args:
        result: Search result to format

    Returns:
        Formatted citation (e.g., "PolarFire Handbook, Page 42")

    Example:
        >>> citation = format_as_citation(result)
        >>> print(citation)
        PolarFire FPGA Datasheet, Page 15
    """
    return result.citation


def format_as_markdown_list(results: List[SearchResult], include_scores: bool = False) -> str:
    """
    Format search results as a markdown list.

    Args:
        results: List of search results
        include_scores: Whether to include relevance scores

    Returns:
        Markdown formatted list

    Example:
        >>> markdown = format_as_markdown_list(results, include_scores=True)
        >>> print(markdown)
        - **PolarFire Datasheet** (Page 15, Score: 0.892): The PolarFire FPGA family...
        - **User Guide** (Page 3, Score: 0.845): Programming options include...
    """
    lines = []
    for result in results:
        score_str = f", Score: {result.score:.3f}" if include_scores else ""
        lines.append(
            f"- **{result.title}** ({result.location}{score_str}): {result.snippet[:150]}..."
        )
    return "\n".join(lines)


def format_as_json_export(results: List[SearchResult]) -> Dict:
    """
    Format search results for JSON export.

    Args:
        results: List of search results

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "generated_at": datetime.now().isoformat(),
        "total_results": len(results),
        "results": [
            {
                "id": result.id,
                "title": result.title,
                "source_path": result.source_path,
                "location": result.location,
                "score": result.score,
                "snippet": result.snippet,
                "doc_id": result.doc_id,
                "slide_or_page": result.slide_or_page,
                "chunk_id": result.chunk_id,
                "updated_at": result.updated_at
            }
            for result in results
        ]
    }
