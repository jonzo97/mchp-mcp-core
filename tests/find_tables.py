"""
Scan through a PDF to find all tables and their locations.

Helps identify good test cases for extractor validation.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine

def main():
    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

    # Get total pages
    import fitz
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    print("="*80)
    print(f"Scanning {pdf_path.name} ({total_pages} pages)")
    print("="*80)

    engine = TableConsensusEngine()

    # Sample every 5th page to find tables quickly
    sample_pages = list(range(0, min(total_pages, 100), 5))

    tables_found = {}

    for page_num in sample_pages:
        print(f"\rScanning page {page_num}...", end="", flush=True)

        result = engine.extract_with_consensus(str(pdf_path), page_num=page_num)

        if result.matches:
            for match in result.matches:
                if match.confidence > 0.5 and match.best_version.rows >= 3:
                    tables_found[page_num] = {
                        'matches': len(result.matches),
                        'rows': match.best_version.rows,
                        'cols': match.best_version.columns,
                        'confidence': match.confidence,
                        'extractors': list(match.versions.keys())
                    }

    print("\n" + "="*80)
    print(f"Found {len(tables_found)} pages with tables")
    print("="*80)

    # Group by table size/type
    small_tables = {}   # < 10 rows
    medium_tables = {}  # 10-30 rows
    large_tables = {}   # > 30 rows

    for page, info in tables_found.items():
        if info['rows'] < 10:
            small_tables[page] = info
        elif info['rows'] < 30:
            medium_tables[page] = info
        else:
            large_tables[page] = info

    if small_tables:
        print(f"\n📊 Small Tables ({len(small_tables)}):")
        for page, info in sorted(small_tables.items()):
            print(f"  Page {page:3d} (Real {page+1:3d}): {info['rows']:2d}x{info['cols']:2d}, conf={info['confidence']:.2f}, extractors={info['extractors']}")

    if medium_tables:
        print(f"\n📊 Medium Tables ({len(medium_tables)}):")
        for page, info in sorted(medium_tables.items()):
            print(f"  Page {page:3d} (Real {page+1:3d}): {info['rows']:2d}x{info['cols']:2d}, conf={info['confidence']:.2f}, extractors={info['extractors']}")

    if large_tables:
        print(f"\n📊 Large Tables ({len(large_tables)}):")
        for page, info in sorted(large_tables.items()):
            print(f"  Page {page:3d} (Real {page+1:3d}): {info['rows']:2d}x{info['cols']:2d}, conf={info['confidence']:.2f}, extractors={info['extractors']}")

    # Recommend test cases
    print("\n" + "="*80)
    print("Recommended Test Cases")
    print("="*80)

    recommendations = []

    # Find high-agreement tables
    for page, info in sorted(tables_found.items()):
        if len(info['extractors']) == 3 and info['confidence'] > 0.8:
            recommendations.append((page, "High agreement (all extractors agree)", info))

    # Find low-confidence tables
    for page, info in sorted(tables_found.items()):
        if info['confidence'] < 0.7 and info['rows'] >= 5:
            recommendations.append((page, "Low confidence (extractors disagree)", info))

    # Find different sizes
    if small_tables:
        page = list(small_tables.keys())[0]
        recommendations.append((page, "Small table example", small_tables[page]))

    if medium_tables:
        page = list(medium_tables.keys())[0]
        recommendations.append((page, "Medium table example", medium_tables[page]))

    for page, reason, info in recommendations[:6]:  # Top 6 recommendations
        print(f"\n✅ Page {page} (Real {page+1}):")
        print(f"   Reason: {reason}")
        print(f"   Size: {info['rows']}x{info['cols']}, Confidence: {info['confidence']:.2f}")
        print(f"   Extractors: {info['extractors']}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
