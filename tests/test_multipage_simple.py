"""
Simple test for multi-page table detection.
"""
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine

def main():
    pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    engine = TableConsensusEngine(
        extractors=["pdfplumber", "camelot", "pymupdf"],
        enable_detection_filter=True
    )

    print("\n" + "="*80)
    print("BASELINE: Per-page extraction (no merging)")
    print("="*80)

    # Baseline: page 21
    r21 = engine.extract_with_consensus(str(pdf_path), page_num=21)
    if r21.matches:
        m = r21.matches[0]
        print(f"Page 21: {m.best_version.rows}x{m.best_version.columns} | Conf: {m.confidence:.2f}")

    # Baseline: page 22
    r22 = engine.extract_with_consensus(str(pdf_path), page_num=22)
    if r22.matches:
        m = r22.matches[0]
        print(f"Page 22: {m.best_version.rows}x{m.best_version.columns} | Conf: {m.confidence:.2f}")

    print("\n" + "="*80)
    print("MULTI-PAGE: With continuation detection")
    print("="*80)

    results = engine.extract_with_multipage_detection(
        pdf_path=str(pdf_path),
        page_range=(21, 22),
        extractor_name="pdfplumber"
    )

    # After merging
    if results[21].matches:
        m = results[21].matches[0]
        print(f"Page 21 (merged): {m.best_version.rows}x{m.best_version.columns} | Conf: {m.confidence:.2f}")

    if results[22].matches:
        m = results[22].matches[0]
        print(f"Page 22 (merged): {m.best_version.rows}x{m.best_version.columns} | Conf: {m.confidence:.2f}")
    else:
        print(f"Page 22 (merged): Empty (continuation merged into page 21)")

    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)

    if r21.matches and results[21].matches:
        baseline_rows = r21.matches[0].best_version.rows
        merged_rows = results[21].matches[0].best_version.rows
        baseline_conf = r21.matches[0].confidence
        merged_conf = results[21].matches[0].confidence

        print(f"Rows: {baseline_rows} → {merged_rows} (Δ {merged_rows - baseline_rows:+d})")
        print(f"Confidence: {baseline_conf:.2f} → {merged_conf:.2f} (Δ {merged_conf - baseline_conf:+.2f})")

        if merged_rows > baseline_rows and merged_conf > baseline_conf:
            print("✅ SUCCESS: Multi-page detection improved results!")
        elif merged_rows > baseline_rows:
            print("⚠️  PARTIAL: Rows merged but confidence needs improvement")
        else:
            print("❌ No improvement detected")

    print()

if __name__ == "__main__":
    main()
