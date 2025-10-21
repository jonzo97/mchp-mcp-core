"""
Debug script to verify region parameter is being passed correctly to extractors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mchp_mcp_core.extractors.table_extractors import (
    PdfPlumberExtractor,
    CamelotExtractor,
    PyMuPDFExtractor
)

PDF_PATH = Path(__file__).parent.parent / "test_datasheets" / "PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf"
PAGE_NUM = 55

# Test region (from PyMuPDF's detection)
TEST_REGION = (51.13, 69.53, 560.92, 555.34)

print("=" * 80)
print("Testing Region Parameter Passing")
print("=" * 80)
print(f"\nPDF: {PDF_PATH}")
print(f"Page: {PAGE_NUM}")
print(f"Test Region: {TEST_REGION}")
print()

# Test each extractor
for name, extractor_class in [
    ("pdfplumber", PdfPlumberExtractor),
    ("camelot", CamelotExtractor),
    ("pymupdf", PyMuPDFExtractor)
]:
    print(f"\n{name.upper()}:")
    print("-" * 60)

    extractor = extractor_class()

    if not extractor.is_available():
        print(f"  ❌ Not available")
        continue

    # Extract WITHOUT region
    print("  WITHOUT region:")
    result1 = extractor.extract_tables(str(PDF_PATH), PAGE_NUM)
    if result1.tables:
        table1 = result1.tables[0]
        print(f"    Rows: {table1.rows}, Columns: {table1.columns}")
        print(f"    BBox: {table1.bbox}")
        if table1.data and len(table1.data) > 5:
            print(f"    Row 5: {table1.data[5][:3]}")
    else:
        print(f"    No tables found")

    # Extract WITH region
    print("  WITH region:")
    result2 = extractor.extract_tables(str(PDF_PATH), PAGE_NUM, region=TEST_REGION)
    if result2.tables:
        table2 = result2.tables[0]
        print(f"    Rows: {table2.rows}, Columns: {table2.columns}")
        print(f"    BBox: {table2.bbox}")
        if table2.data and len(table2.data) > 5:
            print(f"    Row 5: {table2.data[5][:3]}")

        # Check if rows changed
        if result1.tables:
            row_diff = table2.rows - table1.rows
            if row_diff == 0:
                print(f"    ⚠️  Row count UNCHANGED (region may not be working!)")
            else:
                print(f"    ✅ Row count changed by {row_diff}")
    else:
        print(f"    No tables found")

print("\n" + "=" * 80)
print("Expected: Row counts and bboxes should CHANGE when region is applied")
print("=" * 80)
