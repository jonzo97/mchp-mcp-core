"""
Test the TableHeaderDetector on page 55 extractors.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine
from mchp_mcp_core.extractors.table_header_detector import TableHeaderDetector

pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")
engine = TableConsensusEngine()
detector = TableHeaderDetector()

print("="*80)
print("Testing TableHeaderDetector on Page 55")
print("="*80)

result = engine.extract_with_consensus(str(pdf_path), page_num=55)

if result.matches:
    match = result.matches[0]

    print(f"\nTesting header detection on each extractor:\n")

    for name, table in match.versions.items():
        print(f"{'='*60}")
        print(f"{name}: {table.rows} rows")
        print(f"{'='*60}")

        # Detect headers
        header_result = detector.detect_header_rows(table.data)

        print(f"  Detected header rows: {header_result.header_row_indices}")
        print(f"  Confidence: {header_result.confidence:.2f}")
        print(f"  Method: {header_result.method}")

        # Show first few rows with scores
        print(f"\n  Row-by-row analysis (first 10):")
        for i in range(min(10, len(table.data) if table.data else 0)):
            score = header_result.row_scores.get(i, 0.0)
            is_header = i in header_result.header_row_indices
            marker = "📋 HEADER" if is_header else "📊 DATA  "
            row_preview = str(table.data[i][:3]) if table.data and len(table.data[i]) > 0 else "[]"
            print(f"    Row {i:2d}: {marker} (score={score:.2f}) {row_preview}...")

        # Show where data starts
        if header_result.header_row_indices:
            data_start = max(header_result.header_row_indices) + 1
            print(f"\n  ✅ Data starts at row {data_start}")
            if table.data and data_start < len(table.data):
                print(f"     First data row: {table.data[data_start][:3]}...")
        print()

    # Compare header detection across extractors
    print("="*80)
    print("Summary")
    print("="*80)

    header_counts = {}
    for name, table in match.versions.items():
        header_result = detector.detect_header_rows(table.data)
        header_counts[name] = len(header_result.header_row_indices)

    print(f"\nHeader row counts:")
    for name, count in header_counts.items():
        print(f"  {name}: {count} header rows")

    print(f"\n📊 Current metric (OLD method): {match.estimated_header_rows} header rows")
    print(f"   This assumes all extractors have the same header count ❌")

    print(f"\n✅ NEW method detects per-extractor:")
    for name, count in header_counts.items():
        print(f"   {name}: {count} headers → data starts at row {count}")
