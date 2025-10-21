"""
Generate screenshots and CSVs with consensus bbox enabled.

This shows the CURRENT state after the bbox fix.
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors import TableConsensusEngine
from mchp_mcp_core.extractors.table_screenshots import TableScreenshotGenerator

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"
OUTPUT_DIR = Path(__file__).parent.parent / "manual_review" / "screenshots"

# Test pages from EXTRACTOR_HARMONY_ANALYSIS.md
TEST_PAGES = [25, 55, 70]

def generate_screenshots_for_page(page_num: int):
    """Generate screenshots and CSVs for a single page."""
    print(f"\n{'='*80}")
    print(f"Generating screenshots for Page {page_num}")
    print(f"{'='*80}\n")

    # Create page directory
    page_dir = OUTPUT_DIR / f"page{page_num}_with_consensus_bbox"
    page_dir.mkdir(parents=True, exist_ok=True)

    # Extract with consensus bbox
    engine = TableConsensusEngine(use_consensus_bbox=True)
    result = engine.extract_with_consensus(str(PDF_PATH), page_num)

    if not result.success or not result.matches:
        print(f"❌ No tables found on page {page_num}")
        return

    match = result.matches[0]

    # Generate README with metrics
    readme_content = f"""# Page {page_num} - Consensus BBox Results

**Generated**: 2025-10-20 (with consensus bbox fix)

## Consensus Metrics

- **Confidence**: {match.confidence:.2f}
- **Agreement Score**: {match.agreement_score:.2f}
- **Structure Score**: {match.structure_score:.2f}
- **Cell Similarity**: {match.cell_similarity:.2f}

### Critical Metrics:
- **Header Accuracy**: {match.header_accuracy:.2f}
- **Data Row Accuracy**: {match.data_row_accuracy:.2f} {'✅' if match.data_row_accuracy >= 0.90 else '⚠️' if match.data_row_accuracy >= 0.70 else '❌'}
- **Estimated Header Rows**: {match.estimated_header_rows}

## Extractors

"""

    # Process each extractor version
    screenshot_gen = TableScreenshotGenerator()

    for name, table in match.versions.items():
        print(f"  {name}: {table.rows} rows, {table.columns} columns")

        # Add to README
        readme_content += f"\n### {name}\n"
        readme_content += f"- **Rows**: {table.rows}\n"
        readme_content += f"- **Columns**: {table.columns}\n"
        readme_content += f"- **BBox**: {table.bbox}\n"
        readme_content += f"- **Sparsity**: {table.sparsity:.2f}\n"

        # Find PA0
        pa0_row = None
        for i, row in enumerate(table.data):
            if row and len(row) > 0 and str(row[0]).strip() == 'PA0':
                pa0_row = i
                readme_content += f"- **PA0 found**: Row {i}\n"
                break

        if pa0_row is None:
            readme_content += f"- **PA0 found**: ❌ NOT FOUND\n"

        # Generate screenshot
        screenshot_path = page_dir / f"page{page_num}_{name}.png"
        try:
            screenshot_gen.capture_table(
                pdf_path=str(PDF_PATH),
                page_num=page_num,
                table=table,
                output_path=str(screenshot_path)
            )
            print(f"    ✅ Screenshot: {screenshot_path.name}")
        except Exception as e:
            print(f"    ❌ Screenshot failed: {e}")
            import traceback
            traceback.print_exc()

        # Export CSV
        csv_path = page_dir / f"page{page_num}_table0_{name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table.data)
        print(f"    ✅ CSV: {csv_path.name}")

    # Add comparison section to README
    readme_content += f"\n## Expected Behavior (Post-Fix)\n\n"
    readme_content += f"All extractors should:\n"
    readme_content += f"- ✅ Use same table region (consensus bbox)\n"
    readme_content += f"- ✅ Detect PA0 (was missing in Camelot before fix)\n"
    readme_content += f"- ⚠️ May have different row counts (header detection issue - Task #7)\n"
    readme_content += f"- ⚠️ May have PA0 at different row indices (alignment issue - Task #8)\n"

    # Write README
    readme_path = page_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"  ✅ README: {readme_path.name}")

    print(f"\n✅ Page {page_num} complete: {page_dir}")

def main():
    """Generate screenshots for all test pages."""
    print("="*80)
    print("Generating Consensus BBox Screenshots")
    print("="*80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pages: {TEST_PAGES}")

    for page_num in TEST_PAGES:
        try:
            generate_screenshots_for_page(page_num)
        except Exception as e:
            print(f"\n❌ ERROR on page {page_num}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("✅ All screenshots generated!")
    print("="*80)
    print(f"\nView results in: {OUTPUT_DIR}")
    print(f"\nDirectories created:")
    for page_num in TEST_PAGES:
        page_dir = OUTPUT_DIR / f"page{page_num}_with_consensus_bbox"
        if page_dir.exists():
            print(f"  - {page_dir.name}/")

if __name__ == "__main__":
    main()
