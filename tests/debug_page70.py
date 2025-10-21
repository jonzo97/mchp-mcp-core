"""Debug Page 70 alignment issue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors import TableConsensusEngine
from mchp_mcp_core.extractors.table_header_detector import TableHeaderDetector

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"

# Extract tables
engine = TableConsensusEngine(
    extractors=["pdfplumber", "camelot", "pymupdf"],
    use_consensus_bbox=True
)

result = engine.extract_with_consensus(str(PDF_PATH), 70)

if not result.matches:
    print("No matches found!")
    sys.exit(1)

match = result.matches[0]

print("="*80)
print("Page 70 Diagnostic")
print("="*80 + "\n")

# Manually run header detection on each extractor
detector = TableHeaderDetector(max_header_rows=10)

for name, table in match.versions.items():
    print(f"\n{name.upper()}:")
    print(f"  Total rows: {len(table.data)}")

    # Run header detection
    header_result = detector.detect_header_rows(table.data)
    print(f"  Detected headers: {len(header_result.header_row_indices)} rows")
    print(f"  Header indices: {header_result.header_row_indices}")
    print(f"  Header confidence: {header_result.confidence:.2f}")

    # Show all rows with first column
    print(f"  All rows (first column):")
    for i, row in enumerate(table.data[:15]):  # Show first 15 rows
        if row is None:
            first_cell = "None"
        elif len(row) == 0:
            first_cell = "Empty row"
        else:
            first_cell = row[0] if row[0] is not None else "None"
        is_header = "H" if i in header_result.header_row_indices else "D"
        print(f"    [{is_header}] Row {i}: \"{str(first_cell)[:50]}\"")

    # Show data rows (after headers)
    header_count = len(header_result.header_row_indices)
    data_rows = table.data[header_count:]
    print(f"\n  Data rows (after {header_count} headers):")
    for i, row in enumerate(data_rows[:10]):
        first_cell = row[0] if row and len(row) > 0 else ""
        abs_row = header_count + i
        print(f"    Row {abs_row} (data[{i}]): \"{first_cell[:50]}\"")

print("\n" + "="*80)
print("Alignment Test")
print("="*80 + "\n")

# Test alignment manually
versions_list = list(match.versions.items())
for i in range(len(versions_list)):
    for j in range(i + 1, len(versions_list)):
        name_i, table_i = versions_list[i]
        name_j, table_j = versions_list[j]

        # Detect headers for each
        result_i = detector.detect_header_rows(table_i.data)
        result_j = detector.detect_header_rows(table_j.data)

        headers_i = len(result_i.header_row_indices)
        headers_j = len(result_j.header_row_indices)

        # Extract data rows
        data_i = table_i.data[headers_i:] if len(table_i.data) > headers_i else []
        data_j = table_j.data[headers_j:] if len(table_j.data) > headers_j else []

        print(f"\n{name_i} vs {name_j}:")
        print(f"  {name_i}: {headers_i} headers, {len(data_i)} data rows")
        print(f"  {name_j}: {headers_j} headers, {len(data_j)} data rows")

        # Build anchors
        anchors_i = {}
        for idx, row in enumerate(data_i):
            if row and len(row) > 0:
                anchor = str(row[0]).strip()
                if anchor:
                    anchors_i[anchor] = idx

        anchors_j = {}
        for idx, row in enumerate(data_j):
            if row and len(row) > 0:
                anchor = str(row[0]).strip()
                if anchor:
                    anchors_j[anchor] = idx

        print(f"  {name_i} anchors: {list(anchors_i.keys())[:5]}")
        print(f"  {name_j} anchors: {list(anchors_j.keys())[:5]}")

        common = set(anchors_i.keys()) & set(anchors_j.keys())
        print(f"  Common anchors: {common}")
        print(f"  Alignment possible: {'Yes' if common else 'No'}")
