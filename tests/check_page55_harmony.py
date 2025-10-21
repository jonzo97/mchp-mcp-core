"""
Deep dive into page 55 to understand header vs data accuracy.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine
import csv

pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")
engine = TableConsensusEngine()

print("="*80)
print("Page 55 Deep Dive - Header vs Data Accuracy")
print("="*80)

result = engine.extract_with_consensus(str(pdf_path), page_num=55)

if result.matches:
    match = result.matches[0]

    print(f"\n📊 Overall Metrics:")
    print(f"  Consensus confidence: {match.confidence:.2f}")
    print(f"  Agreement score: {match.agreement_score:.2f}")
    print(f"  Structure score: {match.structure_score:.2f}")
    print(f"  Cell similarity: {match.cell_similarity:.2f}")
    print(f"\n📋 Separate Accuracy Metrics:")
    print(f"  Header accuracy: {match.header_accuracy:.2f} ({match.estimated_header_rows} header rows)")
    print(f"  Data row accuracy: {match.data_row_accuracy:.2f} (CRITICAL)")

    print(f"\n📐 Extractor Results:")
    for name, table in match.versions.items():
        print(f"  {name}: {table.rows} rows x {table.columns} cols")
        if table.data and len(table.data) > 0:
            print(f"    First row: {table.data[0][:3]}...")
            print(f"    Last row:  {table.data[-1][:3]}...")

    # Export CSVs
    csv_dir = Path("manual_review/extracted_tables")
    csv_dir.mkdir(parents=True, exist_ok=True)

    for name, table in match.versions.items():
        csv_path = csv_dir / f"page55_table0_{name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if table.data:
                writer.writerows(table.data)
        print(f"\n  ✅ Exported: {csv_path.name}")

    # Now let's manually check what the overlap is
    print("\n" + "="*80)
    print("Manual Row Comparison")
    print("="*80)

    if 'pdfplumber' in match.versions and 'camelot' in match.versions:
        pdf_data = match.versions['pdfplumber'].data
        cam_data = match.versions['camelot'].data

        print(f"\npdfplumber rows: {len(pdf_data)}")
        print(f"Camelot rows: {len(cam_data)}")

        print(f"\npdfplumber first 3 rows:")
        for i, row in enumerate(pdf_data[:3]):
            print(f"  Row {i}: {row[:3]}")

        print(f"\nCamelot first 3 rows:")
        for i, row in enumerate(cam_data[:3]):
            print(f"  Row {i}: {row[:3]}")

        print(f"\npdfplumber last 3 rows:")
        for i, row in enumerate(pdf_data[-3:]):
            print(f"  Row {len(pdf_data)-3+i}: {row[:3]}")

        print(f"\nCamelot last 3 rows:")
        for i, row in enumerate(cam_data[-3:]):
            print(f"  Row {len(cam_data)-3+i}: {row[:3]}")

print("\n" + "="*80)
