"""
Output generation module for markdown formatting and filtering.

Provides tools for:
- Markdown document generation from search results or chunks
- Table of contents generation
- Result filtering and prioritization by severity
- Citation formatting and export utilities
"""

from mchp_mcp_core.output.markdown import (
    MarkdownGenerator,
    format_as_citation,
    format_as_markdown_list,
    format_as_json_export
)
from mchp_mcp_core.output.filter import (
    OutputFilter,
    Severity,
    FilteredChange,
    prioritize_for_review,
    filter_results_by_score,
    deduplicate_results
)

__all__ = [
    # Markdown generation
    "MarkdownGenerator",
    "format_as_citation",
    "format_as_markdown_list",
    "format_as_json_export",
    # Filtering and prioritization
    "OutputFilter",
    "Severity",
    "FilteredChange",
    "prioritize_for_review",
    "filter_results_by_score",
    "deduplicate_results",
]
