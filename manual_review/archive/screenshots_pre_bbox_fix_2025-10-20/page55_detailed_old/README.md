# Page 55 Table Extraction - Manual Review

This directory contains screenshots and CSV exports for manual inspection of the Page 55 table extraction issue.

## Files

### Screenshots (Visual Inspection)
- `PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749_page55_pdfplumber.png` - pdfplumber extraction
- `PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749_page55_camelot.png` - Camelot (stream mode) extraction
- `PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749_page55_pymupdf.png` - PyMuPDF extraction

### CSV Files (Data Comparison)
- `page55_table0_pdfplumber.csv` - 34 rows extracted by pdfplumber
- `page55_table0_camelot.csv` - 41 rows extracted by Camelot
- `page55_table0_pymupdf.csv` - 34 rows extracted by PyMuPDF

## The Problem

**Current Metrics Report**:
```
Header accuracy: 0.35 (5 header rows)
Data row accuracy: 0.48 (CRITICAL)
```

**Expected Reality**:
- All 32 data rows (PA0-PA31) should be IDENTICAL
- Data row accuracy should be ~1.00, not 0.48

## Header Detection Test Results

When testing the new `TableHeaderDetector`:

**pdfplumber** (34 rows total):
- Detected headers: rows 0-4 (5 rows)
- **ISSUE**: Rows 2-4 are PA0, PA1, PA2 (actual data!) marked as headers ❌
- **Should be**: Only rows 0-1 are headers, data starts at row 2

**Camelot** (41 rows total):
- Detected headers: rows 0-4 (5 rows)
- Rows 0-4: Column numbers, empty, table title, empty, actual headers
- **Should be**: Headers end around row 6, data starts at row 7 (PA0)

**PyMuPDF** (34 rows total):
- Same as pdfplumber

## What to Look For

1. **Where does the actual table start?** (Find "PIN NAME" header)
2. **Where does PA0 appear?** (First data row)
3. **How many header rows are there really?**
4. **Do PA0-PA31 data rows look identical across extractors?**

## Expected Findings

Looking at the CSVs:
- pdfplumber row 2 should be PA0
- Camelot row 7 should be PA0
- Both should have identical data for PA0-PA31 rows
- Header row counts differ: pdfplumber (2) vs Camelot (7)

This is why we need:
1. Per-extractor header detection (different counts)
2. Content-based row alignment (PA0 → PA0, not row[5] → row[5])
