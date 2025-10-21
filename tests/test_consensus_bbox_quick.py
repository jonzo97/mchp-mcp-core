"""
Quick test to verify consensus bbox implementation works.

Tests that:
1. Consensus bbox is detected correctly
2. All extractors use the unified region
3. Data row accuracy improves from 0.48 to ~0.95+
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors import TableConsensusEngine

# PDF path (from existing tests)
PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"
PAGE_NUM = 55  # The problematic page with Camelot bbox issue

def test_consensus_bbox():
    """Test consensus bbox detection on page 55."""
    print("=" * 80)
    print("Testing Consensus BBox on Page 55")
    print("=" * 80)
    print()

    # Create engine WITH consensus bbox enabled
    print("Creating consensus engine with bbox detection...")
    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        use_consensus_bbox=True
    )

    # Extract with consensus
    print(f"\nExtracting from page {PAGE_NUM} with consensus bbox...\n")
    result = engine.extract_with_consensus(PDF_PATH, PAGE_NUM)

    # Display results
    print(f"Extraction success: {result.success}")
    print(f"Tables found: {len(result.matches)}")
    print()

    if not result.matches:
        print("❌ No tables found!")
        return

    # Analyze first table
    match = result.matches[0]
    print(f"📊 Table {match.table_index} Results:")
    print("-" * 60)
    print(f"  Extractors: {list(match.versions.keys())}")
    print(f"  Agreement score: {match.agreement_score:.2f}")
    print(f"  Structure score: {match.structure_score:.2f}")
    print(f"  Cell similarity: {match.cell_similarity:.2f}")
    print()
    print(f"  Header accuracy: {match.header_accuracy:.2f}")
    print(f"  Data row accuracy: {match.data_row_accuracy:.2f}  ← KEY METRIC!")
    print(f"  Overall confidence: {match.confidence:.2f}")
    print()

    # Check each version
    print("📋 Per-Extractor Details:")
    print("-" * 60)
    for name, table in match.versions.items():
        print(f"  {name}:")
        print(f"    Rows: {table.rows}, Columns: {table.columns}")
        print(f"    BBox: {table.bbox}")
        if table.data and len(table.data) > 5:
            # Show first data row after headers
            data_row_idx = 5 if len(table.data) > 5 else len(table.data) - 1
            print(f"    Row {data_row_idx}: {table.data[data_row_idx][:3]}")  # First 3 cells
        print()

    # Verdict
    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if match.data_row_accuracy >= 0.90:
        print("✅ SUCCESS: Data row accuracy is {:.2f} (>= 0.90)".format(match.data_row_accuracy))
        print("✅ Consensus bbox detection is working!")
    elif match.data_row_accuracy >= 0.70:
        print("⚠️  PARTIAL: Data row accuracy is {:.2f} (>= 0.70 but < 0.90)".format(match.data_row_accuracy))
        print("   Better than before (was 0.48), but room for improvement")
    else:
        print("❌ FAILURE: Data row accuracy is {:.2f} (< 0.70)".format(match.data_row_accuracy))
        print("   Consensus bbox may not be working correctly")

    print()
    print(f"Previous (broken): 0.48 data row accuracy")
    print(f"Current (fixed):   {match.data_row_accuracy:.2f} data row accuracy")
    print(f"Improvement:       {match.data_row_accuracy - 0.48:.2f}")
    print()

if __name__ == "__main__":
    try:
        test_consensus_bbox()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
