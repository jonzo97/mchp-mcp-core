# Manual Review Screenshots

This directory contains screenshots and CSV exports for manual verification of table extraction quality.

## Current Status

✅ **Cleaned on 2025-10-20** - All outdated screenshots archived

## Consensus BBox Implementation (2025-10-20)

**Problem Solved**: Extractors were using different bounding boxes, causing table misalignment
- Camelot's bbox started at Y=228 (WRONG - missed table top)
- PyMuPDF's bbox started at Y=69 (CORRECT)
- pdfplumber used full page (no bbox)

**Solution**: Union strategy - compute bbox from all extractors, use largest region
- Result: All extractors now see full table
- Camelot now correctly detects PA0 (was missing before!)

**Implementation**:
- `detect_table_region_consensus()` in `table_consensus.py`
- Added `region` parameter to all extractors (pdfplumber, Camelot, PyMuPDF)
- Integrated into `TableConsensusEngine.extract_with_consensus()`

## Next Steps

**Remaining Issue**: Data row accuracy still 0.48 (target: 0.95+)
- **Root Cause**: Row alignment problem
  - pdfplumber/PyMuPDF: PA0 at row 2
  - Camelot: PA0 at row 7 (includes 5 extra header rows)
- **Solution**: Tasks #7 & #8
  - Per-extractor header detection (variable-length headers)
  - Content-based row alignment (match PA0 to PA0, not row[2] to row[7])

## Generating New Screenshots

To generate screenshots with consensus bbox:

```python
from mchp_mcp_core.extractors import TableConsensusEngine

engine = TableConsensusEngine(use_consensus_bbox=True)
result = engine.extract_with_consensus(pdf_path, page_num)
# Generate screenshots from result.matches
```

## Archived Screenshots

Old screenshots (pre-bbox fix) archived to:
`manual_review/archive/screenshots_pre_bbox_fix_2025-10-20/`
