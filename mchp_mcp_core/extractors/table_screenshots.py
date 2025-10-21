"""
PDF table screenshot generation for manual verification.

Generates PNG images of table regions from PDF pages for visual inspection
and ground truth creation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from mchp_mcp_core.extractors.table_extractors import ExtractedTable
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class TableScreenshotGenerator:
    """
    Generates screenshots of PDF table regions.

    Uses PyMuPDF for fast, high-quality rendering of table areas.
    Supports:
    - Single-page table screenshots
    - Multi-page table screenshots (one image per page)
    - Bbox extraction with padding
    - Configurable resolution (DPI)

    Example:
        >>> generator = TableScreenshotGenerator()
        >>> screenshot_path = generator.capture_table(
        ...     pdf_path="datasheet.pdf",
        ...     page_num=21,
        ...     table=extracted_table,
        ...     output_dir="screenshots"
        ... )
        >>> print(f"Saved to: {screenshot_path}")
    """

    def __init__(
        self,
        dpi: int = 300,
        padding: int = 10,
        max_width: int = 2000,
        max_height: int = 3000
    ):
        """
        Initialize screenshot generator.

        Args:
            dpi: Resolution for rendering (default: 300)
            padding: Pixels to add around table bbox (default: 10)
            max_width: Maximum image width in pixels
            max_height: Maximum image height in pixels
        """
        self.dpi = dpi
        self.padding = padding
        self.max_width = max_width
        self.max_height = max_height

    def capture_table(
        self,
        pdf_path: str | Path,
        page_num: int,
        table: Optional[ExtractedTable] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        output_path: Optional[str | Path] = None
    ) -> str:
        """
        Capture screenshot of a single table.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            table: ExtractedTable (if bbox not provided, uses full page)
            bbox: Manual bbox (x0, y0, x1, y1) in PDF coordinates
            output_path: Where to save PNG (auto-generated if None)

        Returns:
            Path to saved screenshot
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Generate output path if not provided
        if output_path is None:
            pdf_name = pdf_path.stem
            output_path = Path(f"{pdf_name}_page{page_num}_table.png")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open PDF
        doc = fitz.open(pdf_path)

        if page_num < 0 or page_num >= len(doc):
            raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")

        page = doc[page_num]

        # Determine bbox
        if bbox:
            clip_bbox = fitz.Rect(bbox)
        elif table and hasattr(table, 'bbox') and table.bbox:
            clip_bbox = fitz.Rect(table.bbox)
        else:
            # Use full page if no bbox available
            clip_bbox = page.rect
            logger.warning(f"No bbox available, using full page for page {page_num}")

        # Add padding
        clip_bbox = self._add_padding(clip_bbox, page.rect)

        # Render to image
        pix = page.get_pixmap(dpi=self.dpi, clip=clip_bbox)

        # Check size limits
        if pix.width > self.max_width or pix.height > self.max_height:
            logger.warning(
                f"Image size {pix.width}x{pix.height} exceeds limits, "
                f"scaling down to {self.max_width}x{self.max_height}"
            )
            # Scale down (PyMuPDF will maintain aspect ratio)
            scale = min(self.max_width / pix.width, self.max_height / pix.height)
            new_dpi = int(self.dpi * scale)
            pix = page.get_pixmap(dpi=new_dpi, clip=clip_bbox)

        # Save
        pix.save(str(output_path))
        doc.close()

        logger.info(f"Saved table screenshot: {output_path} ({pix.width}x{pix.height})")

        return str(output_path)

    def capture_multipage_table(
        self,
        pdf_path: str | Path,
        page_range: Tuple[int, int],
        tables: List[ExtractedTable],
        output_dir: str | Path
    ) -> List[str]:
        """
        Capture screenshots for a multi-page table.

        Args:
            pdf_path: Path to PDF file
            page_range: (start_page, end_page) inclusive, 0-indexed
            tables: List of ExtractedTable objects (one per page)
            output_dir: Directory to save screenshots

        Returns:
            List of paths to saved screenshots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem

        screenshots = []

        for page_offset, table in enumerate(tables):
            page_num = page_range[0] + page_offset

            if page_num > page_range[1]:
                break

            output_path = output_dir / f"{pdf_name}_page{page_num}_table{table.table_index}.png"

            screenshot = self.capture_table(
                pdf_path=pdf_path,
                page_num=page_num,
                table=table,
                output_path=output_path
            )

            screenshots.append(screenshot)

        logger.info(
            f"Captured {len(screenshots)} screenshots for multi-page table "
            f"(pages {page_range[0]}-{page_range[1]})"
        )

        return screenshots

    def capture_table_with_versions(
        self,
        pdf_path: str | Path,
        page_num: int,
        extractor_tables: dict[str, ExtractedTable],
        output_dir: str | Path
    ) -> dict[str, str]:
        """
        Capture screenshots for all extractor versions of a table.

        Useful for manual verification - shows what each extractor extracted.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            extractor_tables: Dict of extractor_name -> ExtractedTable
            output_dir: Directory to save screenshots

        Returns:
            Dict of extractor_name -> screenshot_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem

        screenshots = {}

        for extractor_name, table in extractor_tables.items():
            output_path = output_dir / f"{pdf_name}_page{page_num}_{extractor_name}.png"

            screenshot = self.capture_table(
                pdf_path=pdf_path,
                page_num=page_num,
                table=table,
                output_path=output_path
            )

            screenshots[extractor_name] = screenshot

        logger.info(f"Captured {len(screenshots)} extractor versions for page {page_num}")

        return screenshots

    def _add_padding(
        self,
        bbox: fitz.Rect,
        page_rect: fitz.Rect
    ) -> fitz.Rect:
        """
        Add padding around bbox, constrained to page boundaries.

        Args:
            bbox: Original bbox
            page_rect: Page boundaries

        Returns:
            Padded bbox
        """
        padded = fitz.Rect(
            max(bbox.x0 - self.padding, page_rect.x0),
            max(bbox.y0 - self.padding, page_rect.y0),
            min(bbox.x1 + self.padding, page_rect.x1),
            min(bbox.y1 + self.padding, page_rect.y1)
        )

        return padded


def estimate_table_bbox(table: ExtractedTable, page_rect: fitz.Rect) -> Tuple[float, float, float, float]:
    """
    Estimate table bounding box when not provided.

    Uses heuristics based on page position and table size.

    Args:
        table: Extracted table
        page_rect: Page dimensions

    Returns:
        (x0, y0, x1, y1) bbox in PDF coordinates
    """
    # If table has bbox, use it
    if hasattr(table, 'bbox') and table.bbox:
        return table.bbox

    # Otherwise, estimate based on page position
    # Assume table takes up most of page width, positioned in middle

    margin = 50  # Points (about 0.7 inches)
    x0 = page_rect.x0 + margin
    x1 = page_rect.x1 - margin

    # Estimate height based on number of rows
    # Typical row height: 15-20 points
    row_height = 18
    header_height = 30

    estimated_height = header_height + (table.rows * row_height)
    estimated_height = min(estimated_height, page_rect.height - 2 * margin)

    # Center vertically
    y0 = page_rect.y0 + (page_rect.height - estimated_height) / 2
    y1 = y0 + estimated_height

    logger.debug(f"Estimated bbox for {table.rows}x{table.columns} table: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")

    return (x0, y0, x1, y1)


__all__ = [
    "TableScreenshotGenerator",
    "estimate_table_bbox",
]
