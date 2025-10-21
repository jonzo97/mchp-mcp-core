"""
Generate screenshots of extracted tables for visual inspection.
Also exports table data to CSV for manual review.
"""
import csv
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine, TableScreenshotGenerator

def generate_screenshots(pdf_path: Path, page_num: int, screenshot_dir: Path, csv_dir: Path):
    """Generate screenshots and CSV exports for all extractor versions of tables on a page."""

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Extract tables
    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        enable_detection_filter=False  # Show all extractions
    )

    result = engine.extract_with_consensus(str(pdf_path), page_num=page_num)

    if not result.matches:
        print(f"No tables found on page {page_num}")
        return

    # Generate screenshots
    screenshot_gen = TableScreenshotGenerator(dpi=300, padding=10)

    for match in result.matches:
        print(f"\nTable {match.table_index} on page {page_num}:")
        print(f"  Extractors: {list(match.versions.keys())}")

        # Screenshot each extractor's version
        screenshots = screenshot_gen.capture_table_with_versions(
            pdf_path=pdf_path,
            page_num=page_num,
            extractor_tables=match.versions,
            output_dir=screenshot_dir
        )

        # Export each extractor's version to CSV
        for extractor, screenshot_path in screenshots.items():
            table = match.versions[extractor]

            # Export to CSV
            csv_path = csv_dir / f"page{page_num}_table{match.table_index}_{extractor}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table.data)

            print(f"  {extractor}: {table.rows}x{table.columns}")
            print(f"    Screenshot: {screenshot_path}")
            print(f"    CSV: {csv_path}")

if __name__ == "__main__":
    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")
    base_dir = Path("manual_review")
    screenshot_dir = base_dir / "screenshots"
    csv_dir = base_dir / "extracted_tables"

    print("Generating screenshots and CSV exports for pages 21-22...")
    print("(Note: Using 0-indexed page numbers)")
    print()

    # Page 21 (0-indexed)
    print("="*80)
    print("Page 21 (0-indexed)")
    print("="*80)
    generate_screenshots(
        pdf_path,
        page_num=21,
        screenshot_dir=screenshot_dir / "page21",
        csv_dir=csv_dir
    )

    # Page 22 (0-indexed)
    print("\n" + "="*80)
    print("Page 22 (0-indexed)")
    print("="*80)
    generate_screenshots(
        pdf_path,
        page_num=22,
        screenshot_dir=screenshot_dir / "page22",
        csv_dir=csv_dir
    )

    print("\n" + "="*80)
    print("Output:")
    print("="*80)
    print(f"Screenshots: {screenshot_dir.absolute()}")
    print(f"CSV files: {csv_dir.absolute()}")
    print(f"\nSee {base_dir / 'README.md'} for manual review workflow")
