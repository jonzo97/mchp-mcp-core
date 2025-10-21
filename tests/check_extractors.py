"""Check which extractors found tables on both pages."""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine

pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

engine = TableConsensusEngine(
    extractors=["pdfplumber", "camelot", "pymupdf"],
    enable_detection_filter=False  # Disable filter to see all raw extractions
)

print("Page 21 extractors:")
r21 = engine.extract_with_consensus(str(pdf_path), page_num=21)
for match in r21.matches:
    print(f"  Table {match.table_index}:")
    for name, table in match.versions.items():
        print(f"    - {name}: {table.rows}x{table.columns}")

print("\nPage 22 extractors:")
r22 = engine.extract_with_consensus(str(pdf_path), page_num=22)
for match in r22.matches:
    print(f"  Table {match.table_index}:")
    for name, table in match.versions.items():
        print(f"    - {name}: {table.rows}x{table.columns}")

print("\nExtractors on both pages:")
p21_extractors = set()
for match in r21.matches:
    p21_extractors.update(match.versions.keys())

p22_extractors = set()
for match in r22.matches:
    p22_extractors.update(match.versions.keys())

both = p21_extractors & p22_extractors
print(f"  {both if both else 'None - this explains why detection failed!'}")
