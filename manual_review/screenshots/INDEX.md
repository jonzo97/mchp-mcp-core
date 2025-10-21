# Screenshot Index - What to Look At

**Generated**: 2025-10-20 (with consensus bbox fix)

## ✅ Consensus BBox Implementation Working!

All extractors now use a unified table region (union of all detected bboxes).

## 📂 What to Review

### **START HERE: Page 55** ← Most Important
- **Directory**: `page55_with_consensus_bbox/`
- **Why**: This is the page where Camelot previously failed (bbox Y=228 vs actual Y=69)
- **What to check**:
  - Open the 3 CSV files side-by-side
  - **KEY**: All 3 extractors should now have PA0 (Camelot was missing it before!)
  - Compare row alignment: PA0 appears at different row indices
    - pdfplumber: PA0 at row 2
    - camelot: PA0 at row 7 (5 extra header rows)
    - pymupdf: PA0 at row 2

### Page 25 - Simple Table
- **Directory**: `page25_with_consensus_bbox/`
- **Expected**: High agreement (simple table structure)

### Page 70 - Medium Table
- **Directory**: `page70_with_consensus_bbox/`
- **Expected**: Boundary disagreement but good data quality

## 📊 Quick Comparison

| Page | pdfplumber rows | camelot rows | pymupdf rows | Issue |
|------|----------------|--------------|--------------|-------|
| 25   | 8              | 20           | 8            | Camelot includes extra rows |
| 55   | 34             | 41           | 34           | **Camelot has 7 extra header rows** |
| 70   | 5              | 14           | 5            | Camelot includes extra context |

## 🔍 What Each Directory Contains

Each `pageNN_with_consensus_bbox/` has:
- `README.md` - Detailed metrics (confidence, accuracy, bbox info)
- `pageNN_table0_pdfplumber.csv` - Extracted data
- `pageNN_table0_camelot.csv` - Extracted data
- `pageNN_table0_pymupdf.csv` - Extracted data

## ✅ What's Fixed

- ✅ Camelot now detects PA0 (was missing before bbox fix!)
- ✅ All extractors use same table region
- ✅ No more cropped screenshots

## ⚠️ What's Still Broken

- ❌ Data row accuracy: 0.48 (target: 0.95+)
- **Root cause**: Row alignment issue
  - Extractors have PA0 at different row indices
  - Consensus engine compares row[2] to row[7] → MISMATCH

## 🔧 Next Steps (Tasks #7 & #8)

**Task #7**: Per-extractor header detection
- Detect that Camelot has 7 header rows, not 2
- Variable-length header support

**Task #8**: Content-based row alignment
- Match PA0 to PA0, not row[2] to row[7]
- Align by content, not by index

Once these are done, data accuracy should hit 0.95+!

---

**To open in VSCode**: The project is already open, navigate to:
`manual_review/screenshots/`
