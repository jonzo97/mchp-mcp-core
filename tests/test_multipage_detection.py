"""
Test multi-page table detection on PIC32CZ datasheet.

Tests the problematic page 21-22 power dissipation table that was causing
low confidence scores due to the page break.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine

def test_multipage_detection():
    """Test multi-page detection on PIC32CZ pages 21-22."""

    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

    if not pdf_path.exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        return

    print("=" * 80)
    print("Testing Multi-Page Table Detection")
    print("=" * 80)
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: 21-22 (Power Dissipation Table)")
    print()

    # Initialize consensus engine
    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        enable_detection_filter=True
    )

    # Test 1: Extract WITHOUT multi-page detection (baseline)
    print("🔍 Test 1: Standard Extraction (per-page, no merging)")
    print("-" * 80)

    result_p21 = engine.extract_with_consensus(str(pdf_path), page_num=21)
    result_p22 = engine.extract_with_consensus(str(pdf_path), page_num=22)

    print(f"\nPage 21: {len(result_p21.matches)} table(s)")
    for match in result_p21.matches:
        print(f"  Table {match.table_index}:")
        print(f"    Dimensions: {match.best_version.rows}x{match.best_version.columns}")
        print(f"    Confidence: {match.confidence:.2f}")
        print(f"    Extractors: {list(match.versions.keys())}")
        for name, table in match.versions.items():
            print(f"      - {name}: {table.rows}x{table.columns}")

    print(f"\nPage 22: {len(result_p22.matches)} table(s)")
    for match in result_p22.matches:
        print(f"  Table {match.table_index}:")
        print(f"    Dimensions: {match.best_version.rows}x{match.best_version.columns}")
        print(f"    Confidence: {match.confidence:.2f}")
        print(f"    Extractors: {list(match.versions.keys())}")

    # Test 2: Extract WITH multi-page detection
    print("\n" + "=" * 80)
    print("🔍 Test 2: Multi-Page Detection (WITH merging)")
    print("-" * 80)

    results = engine.extract_with_multipage_detection(
        pdf_path=str(pdf_path),
        page_range=(21, 22),
        extractor_name="pdfplumber"
    )

    print(f"\nPage 21 (after merging): {len(results[21].matches)} table(s)")
    for match in results[21].matches:
        print(f"  Table {match.table_index}:")
        print(f"    Dimensions: {match.best_version.rows}x{match.best_version.columns}")
        print(f"    Confidence: {match.confidence:.2f}")
        print(f"    Extractors: {list(match.versions.keys())}")
        for name, table in match.versions.items():
            print(f"      - {name}: {table.rows}x{table.columns}")

    print(f"\nPage 22 (after merging): {len(results[22].matches)} table(s)")
    for match in results[22].matches:
        print(f"  Table {match.table_index}:")
        print(f"    Dimensions: {match.best_version.rows}x{match.best_version.columns}")
        print(f"    Confidence: {match.confidence:.2f}")

    # Compare results
    print("\n" + "=" * 80)
    print("📊 Comparison")
    print("-" * 80)

    if result_p21.matches and results[21].matches:
        baseline_conf = result_p21.matches[0].confidence
        merged_conf = results[21].matches[0].confidence
        baseline_rows = result_p21.matches[0].best_version.rows
        merged_rows = results[21].matches[0].best_version.rows

        print(f"Confidence: {baseline_conf:.2f} → {merged_conf:.2f} ({merged_conf - baseline_conf:+.2f})")
        print(f"Rows: {baseline_rows} → {merged_rows} ({merged_rows - baseline_rows:+d})")

        if merged_conf > baseline_conf:
            print("✅ Multi-page detection IMPROVED confidence!")
        else:
            print("⚠️  Confidence did not improve")

        if merged_rows > baseline_rows:
            print("✅ Successfully merged continuation rows!")
        else:
            print("⚠️  No additional rows merged")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_multipage_detection()
