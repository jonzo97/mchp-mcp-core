"""
Debug the full consensus bbox flow to see what's happening.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors.table_consensus import detect_table_region_consensus
from mchp_mcp_core.extractors.table_extractors import (
    PdfPlumberExtractor,
    CamelotExtractor,
    PyMuPDFExtractor
)

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"
PAGE_NUM = 55

print("=" * 80)
print("Debugging Consensus BBox Flow")
print("=" * 80)
print()

# Step 1: Detect consensus bbox
extractors = {
    "pdfplumber": PdfPlumberExtractor(),
    "camelot": CamelotExtractor(),
    "pymupdf": PyMuPDFExtractor()
}

print("STEP 1: Detecting consensus bbox...")
print("-" * 60)

consensus_bbox = detect_table_region_consensus(extractors, str(PDF_PATH), PAGE_NUM)
print(f"\n✅ Consensus BBox: {consensus_bbox}")
print()

# Step 2: Extract with each extractor using the consensus bbox
print("STEP 2: Extracting with consensus bbox...")
print("-" * 60)

for name, extractor in extractors.items():
    print(f"\n{name.upper()}:")
    result = extractor.extract_tables(str(PDF_PATH), PAGE_NUM, region=consensus_bbox)

    if result.tables:
        table = result.tables[0]
        print(f"  Rows: {table.rows}, Columns: {table.columns}")
        print(f"  BBox: {table.bbox}")

        if table.data:
            # Find first data row (skip potential headers)
            print(f"  First 3 rows:")
            for i in range(min(3, len(table.data))):
                print(f"    Row {i}: {table.data[i][:3]}")

            # Find PA0
            pa0_found = False
            for i, row in enumerate(table.data):
                if row and len(row) > 0 and str(row[0]).strip() == 'PA0':
                    print(f"  ✅ PA0 found at row {i}: {row[:3]}")
                    pa0_found = True
                    break

            if not pa0_found:
                print(f"  ❌ PA0 NOT FOUND in table!")
    else:
        print(f"  No tables found")

print("\n" + "=" * 80)
print("Expected: All extractors should find PA0 and have similar row counts")
print("=" * 80)
