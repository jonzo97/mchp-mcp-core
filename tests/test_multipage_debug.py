"""Debug multi-page detection."""
import logging
from pathlib import Path
from mchp_mcp_core.extractors import TableConsensusEngine

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

pdf_path = Path("test_datasheets/PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749.pdf")

engine = TableConsensusEngine(
    extractors=["pdfplumber", "camelot", "pymupdf"],
    enable_detection_filter=False  # Disable to see all extractions
)

print("\n" + "="*80)
print("Testing with pdfplumber (page 21: 32x22, page 22: 57x9)")
print("="*80)

results = engine.extract_with_multipage_detection(
    pdf_path=str(pdf_path),
    page_range=(21, 22),
    extractor_name="pdfplumber"
)

print(f"\nPage 21 result: {len(results[21].matches)} tables")
for m in results[21].matches:
    print(f"  {m.best_version.rows}x{m.best_version.columns}, conf={m.confidence:.2f}")

print(f"Page 22 result: {len(results[22].matches)} tables")
for m in results[22].matches:
    print(f"  {m.best_version.rows}x{m.best_version.columns}, conf={m.confidence:.2f}")
