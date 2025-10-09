"""
PDF extraction with intelligent chunking.

Extracts text, tables, and figures from PDFs with structure preservation.
Uses PyMuPDF for text/structure and pdfplumber for tables.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from mchp_mcp_core.models.common import ExtractedChunk
from mchp_mcp_core.extractors.chunking import perform_intelligent_chunking
from mchp_mcp_core.extractors.tables import extract_tables_from_pdf
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """
    Handles PDF extraction and intelligent chunking.

    Features:
    - Text extraction with structure preservation
    - Table extraction (multi-strategy)
    - Figure/image extraction with captions
    - Intelligent chunking (fixed-size or semantic)
    - Section hierarchy detection

    Example:
        >>> from mchp_mcp_core.utils.config import ExtractionConfig
        >>> config = ExtractionConfig()
        >>> extractor = PDFExtractor(config)
        >>> chunks = extractor.extract_document("datasheet.pdf", "doc_123")
    """

    def __init__(self, config: Any):
        """
        Initialize PDF extractor.

        Args:
            config: ExtractionConfig object or dict with extraction settings
        """
        # Handle both dict and Pydantic config
        if hasattr(config, 'chunk_size'):
            self.chunk_size = config.chunk_size
            self.overlap = config.overlap
            self.preserve_sections = config.preserve_sections
            self.extract_images = config.extract_images
            self.chunking_strategy = config.chunking_strategy
            self.min_chunk_size = config.min_chunk_size
            self.max_chunk_size = config.max_chunk_size
        else:
            # Fallback for dict config
            self.chunk_size = config.get('chunk_size', 1500)
            self.overlap = config.get('overlap', 200)
            self.preserve_sections = config.get('preserve_sections', True)
            self.extract_images = config.get('extract_images', True)
            self.chunking_strategy = config.get('chunking_strategy', 'fixed')
            self.min_chunk_size = config.get('min_chunk_size', 500)
            self.max_chunk_size = config.get('max_chunk_size', 2500)

        self.toc_sections = set()  # Valid section numbers from TOC

    def extract_document(self, pdf_path: str, document_id: str) -> List[ExtractedChunk]:
        """
        Extract and chunk a PDF document.

        Args:
            pdf_path: Path to the PDF file
            document_id: Unique identifier for the document

        Returns:
            List of extracted chunks
        """
        logger.info(f"Extracting PDF: {pdf_path}")
        chunks = []

        # Use PyMuPDF for text and structure
        doc = fitz.open(pdf_path)

        # First pass: extract structure and identify section breaks
        document_structure = self._extract_structure(doc)

        # Second pass: extract content with context
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks
            text_chunks = self._extract_text_from_page(page, page_num, document_structure)
            chunks.extend(text_chunks)

            # Extract images/figures
            if self.extract_images:
                figure_chunks = self._extract_figures_from_page(page, page_num, document_id)
                chunks.extend(figure_chunks)

        doc.close()

        # Third pass: extract tables using pdfplumber
        logger.info("Extracting tables...")
        table_chunks = extract_tables_from_pdf(
            pdf_path,
            document_id,
            generate_chunk_id_func=self._generate_chunk_id
        )
        chunks.extend(table_chunks)

        # Fourth pass: perform intelligent chunking on text sections
        logger.info(f"Performing {self.chunking_strategy} chunking...")
        final_chunks = perform_intelligent_chunking(
            chunks,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            chunking_strategy=self.chunking_strategy,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size
        )

        logger.info(f"Extracted {len(final_chunks)} chunks from {pdf_path}")
        return final_chunks

    def _extract_toc_sections(self, doc: fitz.Document) -> set:
        """
        Extract section numbers from PDF table of contents.

        Returns:
            Set of valid section numbers from TOC
        """
        section_numbers = set()

        try:
            toc = doc.get_toc()
            for level, title, page in toc:
                # Extract section number from title (e.g., "13.0 I/O Ports" -> "13.0")
                match = re.match(r'^(\d+(?:\.\d+)*)', title.strip())
                if match:
                    section_numbers.add(match.group(1))
        except Exception as e:
            logger.warning(f"Could not extract TOC sections: {e}")

        return section_numbers

    def _extract_structure(self, doc: fitz.Document) -> Dict[int, str]:
        """
        Extract document structure (sections, headings).

        Returns:
            Dictionary mapping page numbers to section hierarchy
        """
        structure = {}
        current_section = ""

        # Extract valid sections from TOC first (most reliable)
        toc_sections = self._extract_toc_sections(doc)
        self.toc_sections = toc_sections

        # Section patterns
        section_patterns = [
            re.compile(r'^(\d+(?:\.\d+)*)\s+[A-Z]'),           # "1.2.3 TITLE"
            re.compile(r'^(\d+(?:\.\d+)*)\s+[a-z]'),           # "1.2.3 introduction"
            re.compile(r'(?:^|\n)\s*(\d+(?:\.\d+)*)\s+\w+'),   # Any word after number
            re.compile(r'^SECTION\s+(\d+(?:\.\d+)*)'),         # "SECTION 3.1"
            re.compile(r'^Section\s+(\d+(?:\.\d+)*)'),         # "Section 3.1"
        ]

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        text = "".join([span["text"] for span in line.get("spans", [])])

                        # Detect section headers by font size and pattern
                        if line.get("spans"):
                            font_size = line["spans"][0].get("size", 0)

                            # Headers typically have larger font
                            if font_size > 12 and any(p.match(text.strip()) for p in section_patterns):
                                current_section = text.strip()

            structure[page_num] = current_section

        return structure

    def _extract_text_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        structure: Dict[int, str]
    ) -> List[ExtractedChunk]:
        """Extract text content from a page."""
        chunks = []
        text = page.get_text()

        if not text.strip():
            return chunks

        # Create chunk metadata
        metadata = {
            "page": page_num + 1,
            "extraction_method": "pymupdf",
            "has_images": len(page.get_images()) > 0
        }

        section_hierarchy = structure.get(page_num, "Unknown Section")

        chunk = ExtractedChunk(
            chunk_id=self._generate_chunk_id(text, page_num),
            content=text,
            page_start=page_num + 1,
            page_end=page_num + 1,
            chunk_type="text",
            section_hierarchy=section_hierarchy,
            metadata=metadata
        )

        chunks.append(chunk)
        return chunks

    def _extract_figures_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        document_id: str
    ) -> List[ExtractedChunk]:
        """Extract figures/images from a page."""
        chunks = []
        images = page.get_images()

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = page.parent.extract_image(xref)

            # Look for figure caption nearby
            caption = self._find_figure_caption(page, img_index)

            metadata = {
                "page": page_num + 1,
                "image_index": img_index,
                "image_format": base_image["ext"],
                "image_size": len(base_image["image"]),
                "caption": caption,
                "xref": xref
            }

            content = f"[Figure {img_index + 1}]\n"
            if caption:
                content += f"Caption: {caption}\n"

            chunk = ExtractedChunk(
                chunk_id=self._generate_chunk_id(content, page_num, f"fig_{img_index}"),
                content=content,
                page_start=page_num + 1,
                page_end=page_num + 1,
                chunk_type="figure",
                section_hierarchy=f"Figure {img_index + 1}",
                metadata=metadata
            )

            chunks.append(chunk)

        return chunks

    def _find_figure_caption(self, page: fitz.Page, img_index: int) -> Optional[str]:
        """Attempt to find a caption for a figure."""
        text = page.get_text()

        # Look for common caption patterns
        patterns = [
            rf"Figure\s+{img_index + 1}[:\.]?\s+(.+?)(?:\n|$)",
            rf"Fig\.\s+{img_index + 1}[:\.]?\s+(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _generate_chunk_id(self, content: str, page_num: int, suffix: str = "") -> str:
        """Generate a unique chunk ID."""
        hash_input = f"{content[:100]}{page_num}{suffix}".encode()
        hash_val = hashlib.md5(hash_input).hexdigest()[:12]
        return f"chunk_{page_num}_{hash_val}{('_' + suffix) if suffix else ''}"

    def get_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata."""
        doc = fitz.open(pdf_path)

        metadata = {
            "total_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "filename": Path(pdf_path).name
        }

        doc.close()
        return metadata


__all__ = ["PDFExtractor"]
