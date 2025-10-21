"""Test header consensus on PolarFire datasheet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors import TableConsensusEngine

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PolarFire-FPGA-Datasheet-DS00003831.pdf"

# Test different table structures
TEST_PAGES = [0, 10, 15]

def test_polarfire():
    """Test on PolarFire datasheet with different naming conventions."""
    print("=" * 80)
    print("Testing PolarFire Datasheet")
    print("=" * 80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Pages: {TEST_PAGES}\n")

    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        use_consensus_bbox=True
    )

    for page_num in TEST_PAGES:
        print(f"\n{'='*80}")
        print(f"Page {page_num}")
        print(f"{'='*80}\n")

        result = engine.extract_with_consensus(str(PDF_PATH), page_num)

        if not result.success or not result.matches:
            print(f"❌ No tables found on page {page_num}")
            continue

        for idx, match in enumerate(result.matches):
            print(f"\nTable {idx}:")
            print(f"  Confidence: {match.confidence:.2f}")
            print(f"  Data Row Accuracy: {match.data_row_accuracy:.2f}")
            print(f"  Header Accuracy: {match.header_accuracy:.2f}")
            print(f"  Estimated Header Rows: {match.estimated_header_rows}")

            # Display table title
            if match.best_version and match.best_version.caption:
                print(f"  📌 Title: {match.best_version.caption}")
            else:
                print(f"  📌 Title: Not found")

            # Per-extractor details
            print(f"  Extractors:")
            for name, table in match.versions.items():
                caption_info = f"'{table.caption}'" if table.caption else "None"
                print(f"    {name}: {table.rows}x{table.columns}, caption={caption_info}")

            # Verdict
            if match.data_row_accuracy >= 0.90:
                print(f"  ✅ EXCELLENT data quality")
            elif match.data_row_accuracy >= 0.70:
                print(f"  ✅ GOOD data quality")
            else:
                print(f"  ⚠️  NEEDS REVIEW")

    print("\n" + "=" * 80)
    print("✅ PolarFire test complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_polarfire()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
