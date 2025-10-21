# Manual Review Workspace

This directory contains materials for manual verification of table extraction quality.

## Structure

```
manual_review/
├── screenshots/          # PDF screenshots showing extracted table regions
│   ├── page21/          # Screenshots organized by page
│   └── page22/
├── extracted_tables/    # Extracted table data in CSV format
│   ├── page21_table0_pdfplumber.csv
│   ├── page21_table0_camelot.csv
│   └── ...
├── ground_truth/        # Manually verified correct extractions
│   ├── page21_table0.csv    # Ground truth for benchmarking
│   └── annotations.json     # Quality annotations
└── README.md            # This file
```

## Workflow

### 1. Generate Screenshots and Extractions
```bash
python tests/generate_table_screenshots.py
```

This creates:
- **Screenshots**: Visual representation of what each extractor found
- **CSV files**: Extracted table data for inspection

### 2. Manual Review Process

For each table:
1. Open screenshots in `screenshots/pageNN/`
2. Compare all 3 extractor versions (pdfplumber, camelot, pymupdf)
3. Open corresponding CSV files in `extracted_tables/`
4. Identify the best extraction or note issues

### 3. Create Ground Truth

When you find a correct extraction:
1. Copy the best CSV to `ground_truth/pageNN_tableN.csv`
2. Or manually correct and save
3. Add quality notes to `ground_truth/annotations.json`

### 4. Benchmarking

Ground truth files are used to:
- Compute accuracy metrics (TEDS, F1, precision, recall)
- Calculate Straight Through Processing (STP) rate
- Identify extractor strengths/weaknesses

## File Naming Convention

**Screenshots**: `{pdf_name}_page{N}_{extractor}.png`
- Example: `PIC32CZ-CA80-CA90-Family-Data-Sheet-DS60001749_page21_pdfplumber.png`

**Extracted Tables**: `page{N}_table{M}_{extractor}.csv`
- Example: `page21_table0_pdfplumber.csv`

**Ground Truth**: `page{N}_table{M}.csv`
- Example: `page21_table0.csv`

## Notes

- This folder is for working data and can be regenerated
- Ground truth files should be committed to git for reproducibility
- Screenshots are large (~1-2MB each) - consider .gitignore if needed
