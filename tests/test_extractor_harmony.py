"""
Test extractor harmony on various table types.

Tests multiple tables to verify that extractors are working in harmony
and producing consistent results.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine
from mchp_mcp_core.extractors.table_screenshots import TableScreenshotGenerator

def test_table(engine, pdf_path: Path, page_num: int, description: str):
    """Test extraction on a specific page."""
    print(f"\n{'='*80}")
    print(f"Page {page_num} (Real {page_num+1}) - {description}")
    print('='*80)

    result = engine.extract_with_consensus(str(pdf_path), page_num=page_num)

    if not result.matches:
        print("❌ No tables found")
        return

    for idx, match in enumerate(result.matches):
        print(f"\nTable {idx}:")
        print(f"  📊 Consensus confidence: {match.confidence:.2f}")
        print(f"  🤝 Agreement: {match.agreement_score:.2f} ({match.extractor_count}/3 extractors)")
        print(f"  📐 Structure: {match.structure_score:.2f}")
        print(f"  🔤 Cell similarity (all): {match.cell_similarity:.2f}")
        print(f"  📋 Header accuracy: {match.header_accuracy:.2f} ({match.estimated_header_rows} rows)")
        print(f"  ✅ Data row accuracy: {match.data_row_accuracy:.2f} (CRITICAL)")

        print(f"\n  Extractor Results:")
        for extractor_name, table in match.versions.items():
            print(f"    - {extractor_name}: {table.rows}x{table.columns}, confidence={table.confidence:.2f}")

        # Check for harmony issues
        row_counts = [t.rows for t in match.versions.values()]
        col_counts = [t.columns for t in match.versions.values()]

        if len(set(row_counts)) > 1:
            print(f"\n  ⚠️  Row count disagreement: {dict(zip(match.versions.keys(), row_counts))}")
        else:
            print(f"\n  ✅ Row count harmony: All extractors agree on {row_counts[0]} rows")

        if len(set(col_counts)) > 1:
            print(f"  ⚠️  Column count disagreement: {dict(zip(match.versions.keys(), col_counts))}")
        else:
            print(f"  ✅ Column count harmony: All extractors agree on {col_counts[0]} columns")

        # Data quality assessment
        if match.data_row_accuracy >= 0.90:
            print(f"\n  🎉 EXCELLENT data quality (≥90%)")
        elif match.data_row_accuracy >= 0.70:
            print(f"\n  ✅ GOOD data quality (70-90%)")
        elif match.data_row_accuracy >= 0.50:
            print(f"\n  ⚠️  ACCEPTABLE data quality (50-70%) - may need manual review")
        else:
            print(f"\n  ❌ POOR data quality (<50%) - requires manual verification")

        # Generate screenshots and CSV exports
        print(f"\n  📸 Generating screenshots and CSV exports...")
        generator = TableScreenshotGenerator()
        generator.capture_table_with_versions(
            str(pdf_path),
            page_num,
            match.versions,
            output_dir="manual_review/screenshots"
        )

        # Export CSVs for detailed comparison
        import csv
        from pathlib import Path
        csv_dir = Path("manual_review/extracted_tables")
        csv_dir.mkdir(parents=True, exist_ok=True)

        for extractor_name, table in match.versions.items():
            csv_path = csv_dir / f"page{page_num}_table{idx}_{extractor_name}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if table.data:
                    writer.writerows(table.data)
            print(f"    CSV: {csv_path.name}")

def main():
    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

    print("="*80)
    print("Extractor Harmony Testing")
    print("="*80)
    print("\nTesting various table types to verify extractor consistency...")

    engine = TableConsensusEngine()

    # Test cases from find_tables.py results
    test_cases = [
        (55, "Large table - High agreement (41x9)"),
        (65, "Medium table - High agreement (12x9)"),
        (70, "Medium table - High agreement (14x9)"),
        (5, "Large table - Low confidence (47x2, disagreement case)"),
        (25, "Small table (8x5)"),
    ]

    for page_num, description in test_cases:
        test_table(engine, pdf_path, page_num, description)

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\n✅ All tests complete")
    print(f"📸 Screenshots saved to: manual_review/screenshots/")
    print(f"📊 CSV exports saved to: manual_review/extracted_tables/")

if __name__ == "__main__":
    main()
