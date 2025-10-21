"""
Tune Camelot parameters to improve extraction accuracy.

Tests different configurations to find best settings for pinout tables.
"""
from pathlib import Path
from mchp_mcp_core.extractors import CamelotExtractor

def test_camelot_config(pdf_path: Path, page_num: int, config_name: str, mode: str = "lattice", **camelot_params):
    """Test a specific Camelot configuration."""

    # Create extractor with specified mode
    extractor = CamelotExtractor(default_mode=mode, **camelot_params)

    if not extractor.is_available():
        print("❌ Camelot not available")
        return None

    result = extractor.extract_tables(str(pdf_path), page_num=page_num)

    if not result.success or not result.tables:
        print(f"  {config_name}: ❌ No tables extracted")
        return None

    table = result.tables[0]
    print(f"  {config_name}:")
    print(f"    Rows: {table.rows}, Columns: {table.columns}")
    print(f"    Confidence: {table.confidence:.2f}")
    print(f"    Time: {result.total_time_ms:.0f}ms")

    return table

def main():
    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")
    page_num = 31  # Real page 32 (0-indexed)

    print("="*80)
    print("Camelot Configuration Tuning - Page 31 (Real 32)")
    print("="*80)
    print(f"Baseline: pdfplumber/pymupdf get 59-77 rows")
    print(f"Current Camelot: 55-58 rows (missing ~19 rows)")
    print()

    # Test LATTICE mode configurations
    print("🔷 LATTICE MODE (line-based detection)")
    print("-" * 80)

    lattice_configs = [
        {
            "name": "L1. Default lattice",
            "mode": "lattice",
            "params": {}
        },
        {
            "name": "L2. Lower line_scale (detect finer lines)",
            "mode": "lattice",
            "params": {"line_scale": 20}  # Default is 40
        },
        {
            "name": "L3. Very low line_scale (very fine lines)",
            "mode": "lattice",
            "params": {"line_scale": 15}
        },
        {
            "name": "L4. Higher line_scale (thicker lines)",
            "mode": "lattice",
            "params": {"line_scale": 60}
        },
        {
            "name": "L5. Increased line_tol (more lenient line detection)",
            "mode": "lattice",
            "params": {"line_tol": 5}  # Default is 2
        },
        {
            "name": "L6. Increased joint_tol (more lenient joints)",
            "mode": "lattice",
            "params": {"joint_tol": 5}  # Default is 2
        },
        {
            "name": "L7. Copy text vertically",
            "mode": "lattice",
            "params": {"copy_text": ['v']}
        },
        {
            "name": "L8. Combined: low line_scale + increased tolerances",
            "mode": "lattice",
            "params": {"line_scale": 20, "line_tol": 5, "joint_tol": 5}
        },
        {
            "name": "L9. Combined: low line_scale + copy_text",
            "mode": "lattice",
            "params": {"line_scale": 20, "copy_text": ['v']}
        },
    ]

    # Test STREAM mode configurations
    print("\n🔶 STREAM MODE (text-based detection)")
    print("-" * 80)

    stream_configs = [
        {
            "name": "S1. Default stream",
            "mode": "stream",
            "params": {}
        },
        {
            "name": "S2. Increased row_tol (merge rows more)",
            "mode": "stream",
            "params": {"row_tol": 5}  # Default is 2
        },
        {
            "name": "S3. Very high row_tol (aggressive merge)",
            "mode": "stream",
            "params": {"row_tol": 10}
        },
        {
            "name": "S4. Increased edge_tol",
            "mode": "stream",
            "params": {"edge_tol": 100}  # Default is 50
        },
        {
            "name": "S5. Combined: row_tol + edge_tol",
            "mode": "stream",
            "params": {"row_tol": 5, "edge_tol": 100}
        },
        {
            "name": "S6. Combined: aggressive all",
            "mode": "stream",
            "params": {"row_tol": 7, "column_tol": 5, "edge_tol": 100}
        },
    ]

    results = {}

    # Test all lattice configurations
    for config in lattice_configs:
        result = test_camelot_config(
            pdf_path,
            page_num,
            config["name"],
            mode=config["mode"],
            **config["params"]
        )
        if result:
            results[config["name"]] = result

    # Test all stream configurations
    for config in stream_configs:
        result = test_camelot_config(
            pdf_path,
            page_num,
            config["name"],
            mode=config["mode"],
            **config["params"]
        )
        if result:
            results[config["name"]] = result

    # Find best configuration
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if results:
        # Sort by row count
        sorted_results = sorted(results.items(), key=lambda x: x[1].rows, reverse=True)

        print("\n🏆 Top 5 Configurations by Row Count:")
        for i, (name, table) in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {name}")
            print(f"     {table.rows} rows x {table.columns} columns")
            print(f"     Confidence: {table.confidence:.2f}")

        best_name, best_table = sorted_results[0]

        print(f"\n✅ Best: {best_name}")
        print(f"   {best_table.rows} rows (target: 59-77)")

        if best_table.rows >= 59:
            print(f"   🎉 SUCCESS! Matches pdfplumber/pymupdf")
        elif best_table.rows > 55:
            improvement = best_table.rows - 55
            print(f"   📈 Improvement: +{improvement} rows over baseline (55)")
        else:
            print(f"   ⚠️  No improvement over baseline")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
