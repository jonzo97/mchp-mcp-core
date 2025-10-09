"""
Smart metadata extraction for document categorization and filtering.

Extracts structured metadata from document paths and content including:
- Document types (Datasheet, User Guide, Application Note, etc.)
- Product families/categories (configurable patterns)
- Document dates (from filenames and content)
- Version strings (various formats)
- Category tags from folder structure

This module provides a flexible framework that can be extended with
domain-specific patterns while maintaining a consistent interface.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """
    Extracts structured metadata from documents for filtering and categorization.

    The extractor uses regex patterns to identify document types, product families,
    versions, and dates from both filenames and document content. Patterns can be
    extended or overridden for domain-specific use cases.

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(
        ...     path=Path("docs/PolarFire/PolarFire-FPGA-Datasheet-DS00003831.pdf"),
        ...     first_page_text="PolarFire FPGA Datasheet\\nRevision 5\\nJanuary 2024"
        ... )
        >>> print(metadata["product_family"])  # "PolarFire"
        >>> print(metadata["document_type"])   # "Datasheet"
        >>> print(metadata["document_date"])   # "2024-01"

        # Custom patterns for domain-specific products
        >>> custom_extractor = MetadataExtractor(
        ...     product_patterns=[
        ...         (r"widget[_\s]?pro", "Widget Pro"),
        ...         (r"gadget[_\s]?x", "Gadget X"),
        ...     ]
        ... )
    """

    # Default document type patterns (order matters - more specific first)
    DEFAULT_DOC_TYPE_PATTERNS = [
        # Meeting and status documents
        (r"monthly[_\s]call", "Monthly Call"),
        (r"monthly[_\s]report", "Monthly Report"),
        (r"monthly[_\s]meeting", "Monthly Meeting"),
        (r"call[_\s]notes", "Call Notes"),
        (r"meeting[_\s]notes", "Meeting Notes"),
        (r"status[_\s]report", "Status Report"),
        (r"project[_\s]update", "Project Update"),
        (r"quarterly[_\s]review", "Quarterly Review"),
        # Technical documents
        (r"datasheet", "Datasheet"),
        (r"user[_\s]guide", "User Guide"),
        (r"programming[_\s]guide", "Programming Guide"),
        (r"application[_\s]note", "Application Note"),
        (r"design[_\s]guide", "Design Guide"),
        (r"reference[_\s]manual", "Reference Manual"),
        (r"quick[_\s]start", "Quick Start Guide"),
        (r"migration[_\s]guide", "Migration Guide"),
        (r"board[_\s]design", "Board Design Guide"),
        (r"programming[_\s]options", "Programming Options"),
        (r"security[_\s]guide", "Security Guide"),
        (r"clocking[_\s].*guide", "Clocking Guide"),
        (r"transceiver[_\s].*guide", "Transceiver Guide"),
        (r"auto[_\s]update", "Auto Update Guide"),
        (r"error[_\s]report", "Error Report"),
        (r"release[_\s]notes", "Release Notes"),
        (r"errata", "Errata"),
        (r"white[_\s]paper", "White Paper"),
        (r"technical[_\s]brief", "Technical Brief"),
        # Presentations
        (r"presentation", "Presentation"),
        (r"slides", "Presentation"),
    ]

    # Default product family patterns (commonly used as examples - extend for your domain)
    DEFAULT_PRODUCT_PATTERNS = [
        (r"polarfire[_\s]?soc", "PolarFire SoC"),
        (r"polarfire", "PolarFire"),
        (r"igloo2", "IGLOO2"),
        (r"smartfusion2", "SmartFusion2"),
        (r"rtg4", "RTG4"),
        (r"rtpolarfire", "RT PolarFire"),
    ]

    # Version patterns (capture groups)
    VERSION_PATTERNS = [
        r"[_\-\s]V([A-Z0-9]+)(?:[_\-\.\s]|$)",  # VB, V11, VA
        r"[_\-\s]v(\d+\.\d+)",                  # v1.0, v2.3
        r"DS(\d+)",                              # DS00003831
        r"UG(\d+)",                              # UG0726
        r"AC(\d+)",                              # AC466
        r"AN(\d+)",                              # AN123
        r"Rev(?:ision)?[_\s]?(\d+)",            # Revision 5, Rev 3
    ]

    # Date patterns (from content - title slides/first pages)
    DATE_PATTERNS = [
        # "January 2024", "Jan 2024"
        r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+(\d{4})",
        # "2024-01", "2024/01"
        r"(\d{4})[-/](\d{2})",
        # "01/2024", "01-2024"
        r"(\d{2})[-/](\d{4})",
        # "2024 Jan", "2024 January"
        r"(\d{4})[,\s]+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    ]

    MONTH_MAP = {
        "january": "01", "jan": "01",
        "february": "02", "feb": "02",
        "march": "03", "mar": "03",
        "april": "04", "apr": "04",
        "may": "05",
        "june": "06", "jun": "06",
        "july": "07", "jul": "07",
        "august": "08", "aug": "08",
        "september": "09", "sep": "09",
        "october": "10", "oct": "10",
        "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }

    def __init__(
        self,
        doc_type_patterns: Optional[list[tuple[str, str]]] = None,
        product_patterns: Optional[list[tuple[str, str]]] = None,
        version_patterns: Optional[list[str]] = None,
        date_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize metadata extractor with optional custom patterns.

        Args:
            doc_type_patterns: List of (regex, label) tuples for document types.
                              If None, uses DEFAULT_DOC_TYPE_PATTERNS.
            product_patterns: List of (regex, label) tuples for product families.
                             If None, uses DEFAULT_PRODUCT_PATTERNS.
            version_patterns: List of regex patterns for version extraction.
                             If None, uses VERSION_PATTERNS.
            date_patterns: List of regex patterns for date extraction.
                          If None, uses DATE_PATTERNS.

        Example:
            >>> # Use defaults
            >>> extractor = MetadataExtractor()
            >>>
            >>> # Override product patterns for your domain
            >>> custom = MetadataExtractor(
            ...     product_patterns=[
            ...         (r"widget[_\s]pro", "Widget Pro"),
            ...         (r"gadget", "Gadget Series"),
            ...     ]
            ... )
        """
        self.doc_type_patterns = doc_type_patterns or self.DEFAULT_DOC_TYPE_PATTERNS
        self.product_patterns = product_patterns or self.DEFAULT_PRODUCT_PATTERNS
        self.version_patterns = version_patterns or self.VERSION_PATTERNS
        self.date_patterns = date_patterns or self.DATE_PATTERNS

    def extract(
        self,
        path: Path | str,
        first_page_text: Optional[str] = None,
        docs_root: Optional[Path | str] = None,
    ) -> dict[str, Any]:
        """
        Extract metadata from document path and optional first page content.

        Args:
            path: Path to document file
            first_page_text: Text from first slide/page (for date extraction)
            docs_root: Root docs directory (for subfolder categorization)

        Returns:
            Dict with keys:
                - document_type: Detected type (e.g., "Datasheet", "User Guide")
                - product_family: Detected product (e.g., "PolarFire", "Widget Pro")
                - version: Version string if found
                - document_date: ISO 8601 date (YYYY-MM) if found
                - category_tags: List of tags from subfolder, type, product
                - subfolder_path: Relative path from docs_root

        Example:
            >>> extractor = MetadataExtractor()
            >>> meta = extractor.extract(
            ...     Path("docs/Products/Widget-Datasheet-V2.pdf"),
            ...     first_page_text="Widget Datasheet\\nJanuary 2024",
            ...     docs_root=Path("docs")
            ... )
            >>> print(meta)
            {
                "document_type": "Datasheet",
                "product_family": None,
                "version": "2",
                "document_date": "2024-01",
                "category_tags": ["Products", "datasheet"],
                "subfolder_path": "Products"
            }
        """
        path = Path(path) if isinstance(path, str) else path
        docs_root = Path(docs_root) if isinstance(docs_root, str) and docs_root else docs_root

        filename = path.name
        filename_lower = filename.lower()

        metadata: dict[str, Any] = {
            "document_type": self._extract_document_type(filename_lower),
            "product_family": self._extract_product_family(filename_lower),
            "version": self._extract_version(filename),
            "document_date": None,
            "category_tags": [],
            "subfolder_path": "",
        }

        # Extract date from first page content if available
        if first_page_text:
            metadata["document_date"] = self._extract_date_from_content(first_page_text)

        # If no date from content, try filename
        if not metadata["document_date"]:
            metadata["document_date"] = self._extract_date_from_filename(filename)

        # Extract subfolder tags
        if docs_root:
            try:
                relative_path = path.parent.relative_to(docs_root)
                metadata["subfolder_path"] = str(relative_path)

                # Add subfolder names as tags (skip '.')
                if relative_path != Path("."):
                    parts = [p for p in relative_path.parts if p != "."]
                    metadata["category_tags"].extend(parts)
            except ValueError:
                # Path is not relative to docs_root
                logger.debug(f"Path {path} not relative to {docs_root}")

        # Add document type and product as tags
        if metadata["document_type"]:
            metadata["category_tags"].append(
                metadata["document_type"].lower().replace(" ", "_")
            )
        if metadata["product_family"]:
            metadata["category_tags"].append(
                metadata["product_family"].lower().replace(" ", "_")
            )

        # Remove duplicates while preserving order
        seen = set()
        metadata["category_tags"] = [
            tag for tag in metadata["category_tags"] if tag not in seen and not seen.add(tag)
        ]

        logger.debug(f"Extracted metadata for {filename}: {metadata}")
        return metadata

    def _extract_document_type(self, filename_lower: str) -> Optional[str]:
        """Extract document type from filename using pattern matching."""
        for pattern, doc_type in self.doc_type_patterns:
            if re.search(pattern, filename_lower):
                return doc_type
        return None

    def _extract_product_family(self, filename_lower: str) -> Optional[str]:
        """Extract product family from filename using pattern matching."""
        for pattern, product in self.product_patterns:
            if re.search(pattern, filename_lower):
                return product
        return None

    def _extract_version(self, filename: str) -> Optional[str]:
        """Extract version string from filename."""
        for pattern in self.version_patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        return None

    def _extract_date_from_content(self, text: str) -> Optional[str]:
        """
        Extract date from document content (typically title slide/first page).

        Returns ISO 8601 format (YYYY-MM).
        """
        # Limit search to first 500 characters (title area)
        search_text = text[:500]

        for pattern in self.date_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                groups = match.groups()

                # Pattern 1: "January 2024" or "Jan 2024"
                if len(groups) == 2 and groups[0].isalpha():
                    month_str = groups[0].lower()
                    year = groups[1]
                    month = self.MONTH_MAP.get(month_str)
                    if month:
                        return f"{year}-{month}"

                # Pattern 2: "2024-01" (already in correct format)
                elif len(groups) == 2 and groups[0].isdigit() and len(groups[0]) == 4:
                    year, month = groups
                    return f"{year}-{month}"

                # Pattern 3: "01/2024" or "01-2024"
                elif len(groups) == 2 and groups[0].isdigit() and len(groups[0]) == 2:
                    month, year = groups
                    return f"{year}-{month}"

                # Pattern 4: "2024 Jan" or "2024 January"
                elif len(groups) == 2 and groups[1].isalpha():
                    year = groups[0]
                    month_str = groups[1].lower()
                    month = self.MONTH_MAP.get(month_str)
                    if month:
                        return f"{year}-{month}"

        return None

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract date from filename if present."""
        # Look for year patterns in filename (2000-2099)
        year_match = re.search(r"(20\d{2})", filename)
        if year_match:
            year = year_match.group(1)

            # Try to find month nearby
            for month_name, month_num in self.MONTH_MAP.items():
                if month_name in filename.lower():
                    return f"{year}-{month_num}"

            # Return year-only as YYYY-01 (conservative - assume January)
            return f"{year}-01"

        return None

    def extract_from_batch(
        self,
        paths: list[Path | str],
        first_pages: Optional[dict[Path, str]] = None,
        docs_root: Optional[Path | str] = None,
    ) -> dict[Path, dict[str, Any]]:
        """
        Extract metadata for multiple documents.

        Args:
            paths: List of document paths
            first_pages: Optional dict mapping paths to first page text
            docs_root: Root docs directory

        Returns:
            Dict mapping paths to metadata dicts

        Example:
            >>> extractor = MetadataExtractor()
            >>> paths = list(Path("./docs").rglob("*.pdf"))
            >>> results = extractor.extract_from_batch(paths, docs_root=Path("./docs"))
            >>> for path, meta in results.items():
            ...     print(f"{path.name}: {meta['document_type']}")
        """
        first_pages = first_pages or {}
        results = {}

        for path_item in paths:
            path = Path(path_item) if isinstance(path_item, str) else path_item
            first_page_text = first_pages.get(path)
            results[path] = self.extract(path, first_page_text, docs_root)

        return results


# Convenience function for single-file extraction
def extract_metadata(
    path: Path | str,
    first_page_text: Optional[str] = None,
    docs_root: Optional[Path | str] = None,
    **extractor_kwargs,
) -> dict[str, Any]:
    """
    Extract metadata from a single document.

    This is a convenience function that creates a MetadataExtractor instance
    and calls extract() in one step.

    Args:
        path: Path to document
        first_page_text: Text from first slide/page
        docs_root: Root docs directory
        **extractor_kwargs: Additional kwargs passed to MetadataExtractor()
                           (e.g., custom product_patterns)

    Returns:
        Metadata dict

    Example:
        >>> # Simple usage
        >>> meta = extract_metadata("docs/Widget-Datasheet.pdf")
        >>>
        >>> # With custom patterns
        >>> meta = extract_metadata(
        ...     "docs/Widget-Pro-Guide.pdf",
        ...     product_patterns=[
        ...         (r"widget[_\s]pro", "Widget Pro"),
        ...     ]
        ... )
    """
    extractor = MetadataExtractor(**extractor_kwargs)
    return extractor.extract(path, first_page_text, docs_root)
