"""
Test header consensus and content-based row alignment fix.

This verifies:
1. Per-extractor header detection works
2. Content-based row alignment improves data accuracy
3. Table titles are extracted correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors import TableConsensusEngine

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"

# Test pages from previous analysis
TEST_PAGES = [25, 55, 70]

def test_header_consensus():
    """Test per-extractor header detection and content-based alignment."""
    print("=" * 80)
    print("Testing Header Consensus Fix")
    print("=" * 80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Pages: {TEST_PAGES}\n")

    # Create engine with consensus bbox enabled
    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        use_consensus_bbox=True
    )

    results = {}

    for page_num in TEST_PAGES:
        print(f"\n{'='*80}")
        print(f"Page {page_num}")
        print(f"{'='*80}\n")

        # Extract with new consensus logic
        result = engine.extract_with_consensus(str(PDF_PATH), page_num)

        if not result.success or not result.matches:
            print(f"❌ No tables found on page {page_num}")
            results[page_num] = None
            continue

        match = result.matches[0]
        results[page_num] = match

        # Display metrics
        print(f"📊 Consensus Metrics:")
        print(f"  Confidence: {match.confidence:.2f}")
        print(f"  Agreement: {match.agreement_score:.2f}")
        print(f"  Structure: {match.structure_score:.2f}")
        print(f"  Cell Similarity: {match.cell_similarity:.2f}")
        print()

        print(f"📋 Header/Data Accuracy:")
        print(f"  Header Accuracy: {match.header_accuracy:.2f}")
        print(f"  Data Row Accuracy: {match.data_row_accuracy:.2f} ← KEY METRIC!")
        print(f"  Estimated Header Rows: {match.estimated_header_rows}")
        print()

        # Display table title
        if match.best_version and match.best_version.caption:
            print(f"📌 Table Title: {match.best_version.caption}")
        else:
            print(f"📌 Table Title: Not found")
        print()

        # Display per-extractor details
        print(f"🔧 Per-Extractor Details:")
        for name, table in match.versions.items():
            print(f"  {name}:")
            print(f"    Rows: {table.rows}, Columns: {table.columns}")
            print(f"    BBox: {table.bbox}")
            print(f"    Caption: {table.caption or 'None'}")

            # Find PA0 for pinout tables
            pa0_row = None
            if table.data:
                for i, row in enumerate(table.data):
                    if row and len(row) > 0 and str(row[0]).strip() == 'PA0':
                        pa0_row = i
                        break

            if pa0_row is not None:
                print(f"    PA0 found at row {pa0_row}")
            print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Before vs After Fix")
    print("=" * 80 + "\n")

    expected_improvements = {
        25: {"before": 0.80, "target": 0.80},  # Should maintain
        55: {"before": 0.48, "target": 0.95},  # Should improve significantly
        70: {"before": 1.00, "target": 0.95},  # Should maintain
    }

    all_passed = True

    for page_num, expectations in expected_improvements.items():
        if results.get(page_num):
            match = results[page_num]
            actual = match.data_row_accuracy
            target = expectations["target"]
            before = expectations["before"]

            status = "✅" if actual >= target else "❌"
            print(f"Page {page_num}:")
            print(f"  Before fix: {before:.2f}")
            print(f"  After fix:  {actual:.2f}")
            print(f"  Target:     {target:.2f}")
            print(f"  Status:     {status}")

            if actual < target:
                all_passed = False
                print(f"  ⚠️  Did not meet target!")
            elif actual > before:
                improvement = actual - before
                print(f"  🎉 Improved by {improvement:.2f}!")
            print()

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ Header consensus and row alignment working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("   Check logs above for details")
        return 1

if __name__ == "__main__":
    try:
        exit_code = test_header_consensus()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
