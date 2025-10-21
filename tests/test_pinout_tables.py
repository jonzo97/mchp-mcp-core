"""
Test table extraction on actual multi-page pinout tables (pages 31-33).
"""
import csv
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine, TableScreenshotGenerator

def test_pinout_tables():
    """Test extraction on real pinout tables (pages 31-33)."""

    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")
    base_dir = Path("manual_review")
    screenshot_dir = base_dir / "screenshots"
    csv_dir = base_dir / "extracted_tables"

    # Real pages 31-33 = 0-indexed pages 30-32
    pages = [30, 31, 32]

    print("="*80)
    print("Testing Pinout Table Extraction (Real Pages 31-33)")
    print("="*80)
    print()

    # Initialize engine
    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        enable_detection_filter=False  # Show all extractions
    )
    screenshot_gen = TableScreenshotGenerator(dpi=300, padding=10)

    # Extract from each page
    for page_num in pages:
        real_page = page_num + 1
        print(f"\n{'='*80}")
        print(f"Page {page_num} (0-indexed) = Real Page {real_page}")
        print(f"{'='*80}")

        result = engine.extract_with_consensus(str(pdf_path), page_num=page_num)

        if not result.matches:
            print(f"⚠️  No tables found on page {page_num}")
            continue

        page_screenshot_dir = screenshot_dir / f"page{page_num}"
        page_screenshot_dir.mkdir(parents=True, exist_ok=True)

        for match in result.matches:
            print(f"\nTable {match.table_index}:")
            print(f"  Consensus confidence: {match.confidence:.2f}")
            print(f"  Agreement: {match.agreement_score:.2f}")
            print(f"  Structure: {match.structure_score:.2f}")
            print(f"  Cell similarity (all): {match.cell_similarity:.2f}")
            print(f"  📋 Header accuracy: {match.header_accuracy:.2f} ({match.estimated_header_rows} rows)")
            print(f"  ✅ Data row accuracy: {match.data_row_accuracy:.2f} (CRITICAL)")
            print(f"  Extractors: {list(match.versions.keys())}")

            # Show each extractor's version
            for extractor_name, table in match.versions.items():
                print(f"    - {extractor_name}: {table.rows}x{table.columns}, confidence={table.confidence:.2f}")

            # Generate screenshots
            screenshots = screenshot_gen.capture_table_with_versions(
                pdf_path=pdf_path,
                page_num=page_num,
                extractor_tables=match.versions,
                output_dir=page_screenshot_dir
            )

            # Export to CSV
            for extractor_name, table in match.versions.items():
                csv_path = csv_dir / f"page{page_num}_table{match.table_index}_{extractor_name}.csv"
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(table.data)

                print(f"    CSV: {csv_path.name}")

    # Now test multi-page detection
    print("\n" + "="*80)
    print("Multi-Page Continuation Detection Test")
    print("="*80)

    results = engine.extract_with_multipage_detection(
        pdf_path=str(pdf_path),
        page_range=(30, 32),
        extractor_name="pdfplumber"
    )

    print("\nResults after multi-page detection:")
    for page_num in pages:
        real_page = page_num + 1
        if results[page_num].matches:
            print(f"\nPage {page_num} (Real {real_page}): {len(results[page_num].matches)} table(s)")
            for match in results[page_num].matches:
                print(f"  Table {match.table_index}: {match.best_version.rows}x{match.best_version.columns}, conf={match.confidence:.2f}")
        else:
            print(f"\nPage {page_num} (Real {real_page}): Empty (merged into previous page)")

    print("\n" + "="*80)
    print(f"Screenshots: {screenshot_dir.absolute()}")
    print(f"CSV files: {csv_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    test_pinout_tables()
