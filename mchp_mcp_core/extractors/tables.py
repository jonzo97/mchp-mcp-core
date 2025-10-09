"""
Table extraction utilities using pdfplumber.

Provides multi-strategy table extraction with quality validation:
- Line-based extraction (for tables with visible borders)
- Text-based extraction (for borderless tables)
- Quality assessment (sparse detection, empty cell filtering)
- Markdown conversion
"""

import re
from typing import List, Optional

import pdfplumber

from mchp_mcp_core.models.common import ExtractedChunk


def extract_tables_from_pdf(
    pdf_path: str,
    document_id: str,
    generate_chunk_id_func=None
) -> List[ExtractedChunk]:
    """
    Extract tables from PDF using pdfplumber with multi-strategy approach.

    Args:
        pdf_path: Path to PDF file
        document_id: Document identifier
        generate_chunk_id_func: Optional function to generate chunk IDs

    Returns:
        List of table chunks
    """
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Multi-strategy table extraction
            tables = extract_tables_multi_strategy(page)

            for table_index, table in enumerate(tables):
                # Skip empty tables (after all strategies)
                if not table or len(table) < 2:
                    continue

                # Check if table has actual content
                if is_table_empty(table):
                    continue

                # Convert table to markdown format
                table_md = table_to_markdown(table)

                # Find table caption
                caption = find_table_caption(page, table_index)

                content = f"[Table {table_index + 1}]\n"
                if caption:
                    content += f"Caption: {caption}\n\n"
                content += table_md

                metadata = {
                    "page": page_num + 1,
                    "table_index": table_index,
                    "rows": len(table),
                    "columns": len(table[0]) if table else 0,
                    "caption": caption,
                    "extraction_quality": "good" if not is_table_sparse(table) else "sparse"
                }

                # Generate chunk ID
                if generate_chunk_id_func:
                    chunk_id = generate_chunk_id_func(content, page_num, f"tbl_{table_index}")
                else:
                    chunk_id = f"{document_id}_p{page_num}_tbl_{table_index}"

                chunk = ExtractedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    page_start=page_num + 1,
                    page_end=page_num + 1,
                    chunk_type="table",
                    section_hierarchy=f"Table {table_index + 1}",
                    metadata=metadata
                )

                chunks.append(chunk)

    return chunks


def extract_tables_multi_strategy(page) -> List:
    """
    Try multiple extraction strategies to get the best table data.

    Strategy 1: Standard extraction
    Strategy 2: Line-based (for tables with visible borders)
    Strategy 3: Text-based (for borderless tables)

    Args:
        page: pdfplumber page object

    Returns:
        List of extracted tables
    """
    # Strategy 1: Standard extraction
    tables = page.extract_tables()

    # If we got good tables, return them
    if tables and any(not is_table_empty(t) for t in tables):
        return tables

    # Strategy 2: Line-based extraction (for tables with visible borders)
    tables = page.extract_tables(table_settings={
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
    })

    if tables and any(not is_table_empty(t) for t in tables):
        return tables

    # Strategy 3: Text-based extraction (for borderless tables)
    tables = page.extract_tables(table_settings={
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    })

    return tables or []


def is_table_empty(table: List[List]) -> bool:
    """
    Check if a table has actual content.

    Args:
        table: Table data (list of lists)

    Returns:
        True if table is empty or has insufficient content
    """
    if not table or len(table) < 2:
        return True

    # Count non-empty cells
    non_empty_cells = sum(
        1 for row in table for cell in row
        if cell and str(cell).strip() and str(cell).strip() not in ['', '---']
    )

    # Need at least 3 non-empty cells for a valid table
    return non_empty_cells < 3


def is_table_sparse(table: List[List]) -> bool:
    """
    Check if table has many empty cells.

    Args:
        table: Table data (list of lists)

    Returns:
        True if more than 50% of cells are empty
    """
    if not table:
        return True

    total_cells = sum(len(row) for row in table)
    if total_cells == 0:
        return True

    empty_cells = sum(
        1 for row in table for cell in row
        if not cell or not str(cell).strip()
    )

    return (empty_cells / total_cells) > 0.5  # More than 50% empty


def table_to_markdown(table: List[List[str]]) -> str:
    """
    Convert table data to markdown format.

    Args:
        table: Table data (list of lists)

    Returns:
        Markdown-formatted table string
    """
    if not table:
        return ""

    md_lines = []

    # Header row
    header = [cell or "" for cell in table[0]]
    md_lines.append("| " + " | ".join(header) + " |")

    # Separator
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    # Data rows
    for row in table[1:]:
        cells = [cell or "" for cell in row]
        # Pad if necessary
        while len(cells) < len(header):
            cells.append("")
        md_lines.append("| " + " | ".join(cells[:len(header)]) + " |")

    return "\n".join(md_lines)


def find_table_caption(page, table_index: int) -> Optional[str]:
    """
    Improved caption detection with better patterns.

    Args:
        page: pdfplumber page object
        table_index: Index of the table on the page

    Returns:
        Caption text if found, None otherwise
    """
    text = page.extract_text()

    if not text:
        return None

    # Try multiple caption patterns
    patterns = [
        rf'Table\s+{table_index + 1}[:\.]?\s+(.+?)(?:\n|$)',
        rf'TABLE\s+{table_index + 1}[:\.]?\s+(.+?)(?:\n|$)',
        rf'Table\s+(\d+-\d+)[:\.]?\s+(.+?)(?:\n|$)',  # Hyphenated table numbers
        rf'TABLE\s+(\d+-\d+)[:\.]?\s+(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Get the caption text (last group)
            caption = match.group(match.lastindex).strip()
            if len(caption) > 3:  # Avoid single character matches
                return caption[:200]  # Limit caption length

    return None


__all__ = [
    "extract_tables_from_pdf",
    "extract_tables_multi_strategy",
    "is_table_empty",
    "is_table_sparse",
    "table_to_markdown",
    "find_table_caption"
]
