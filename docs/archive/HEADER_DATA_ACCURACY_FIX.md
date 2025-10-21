# Header/Data Accuracy Fix - Technical Specification

## Problem Statement

Current implementation in `table_consensus.py:_compute_header_data_accuracy()` **incorrectly assumes all extractors have the same number of header rows**.

### Page 55 Example (Reality)
```
pdfplumber: 34 rows = 2 header + 32 data (PA0-PA31)
Camelot:    41 rows = 7 "header" + 32 data (PA0-PA31) + 2 extra
PyMuPDF:    34 rows = 2 header + 32 data (PA0-PA31)

TRUTH: All 32 data rows (PA0-PA31) are IDENTICAL
```

### Current Algorithm (WRONG)
```python
# Estimates single header row count (returns 5)
estimated_header_rows = self._estimate_header_rows(versions_list[0])

# Compares rows 0-5 as "headers" (wrong: Camelot has 7, pdfplumber has 2)
header_accuracy = compare(rows[0:5], rows[0:5])  # 0.35

# Compares rows 5+ as "data" (wrong: misaligned)
data_accuracy = compare(rows[5:], rows[5:])  # 0.48

# SHOULD BE: data_accuracy = 1.00 (PA0-PA31 identical)
```

## Required Fix

### Step 1: Per-Extractor Header Detection
```python
def _detect_headers_per_extractor(self, table: ExtractedTable) -> int:
    """
    Detect header rows for THIS extractor (not assuming same across all).

    Heuristics:
    1. Look for rows with column names (PIN NAME, etc.)
    2. Detect merged/spanning cells
    3. Find first row with consistent data pattern (PA0, PA1, etc.)
    4. Stop at first data row

    Returns:
        Number of header rows for this specific table
    """
```

### Step 2: Content-Based Row Alignment
```python
def _align_data_rows(
    self,
    table1: ExtractedTable,
    table2: ExtractedTable,
    header_rows1: int,
    header_rows2: int
) -> List[Tuple[int, int]]:
    """
    Align data rows by CONTENT, not by index.

    Example:
        table1 row 2: ['PA0', '1', '1', ...]
        table2 row 7: ['PA0', '1', '1', ...]
        → These match! (row 2 aligns with row 7)

    Returns:
        List of (table1_row_idx, table2_row_idx) pairs that align
    """
```

### Step 3: Separate Metrics
```python
def _compute_header_data_accuracy(self, match: TableMatch) -> Tuple[float, float, dict]:
    """
    NEW RETURN VALUE:

    Returns:
        (
            data_row_accuracy: float,  # 0-1, content match on aligned data rows
            header_accuracy: float,    # 0-1, structural similarity of headers
            metadata: dict             # Detailed breakdown
        )

    metadata = {
        'header_rows_per_extractor': {'pdfplumber': 2, 'camelot': 7, ...},
        'aligned_data_rows': 32,
        'matching_data_rows': 32,
        'data_row_pairs': [(2, 7), (3, 8), ...],  # Alignment mapping
        'boundary_differences': 'Camelot includes table title + column indices'
    }
    """
```

## Implementation Plan

### File to Modify
`mchp_mcp_core/extractors/table_consensus.py`

### Functions to Change

1. **`_estimate_header_rows()` → `_detect_header_rows_robust()`**
   - Add multiple heuristics
   - Handle edge cases (no headers, multi-row headers, table titles)
   - Return confidence score

2. **NEW: `_align_data_rows_by_content()`**
   - Use first column as anchor (PIN NAME, etc.)
   - Find matching patterns
   - Handle partial overlaps

3. **`_compare_table_rows()` → `_compare_aligned_rows()`**
   - Accept alignment mapping
   - Compare only aligned pairs
   - Handle unaligned rows separately

4. **`_compute_header_data_accuracy()` - Major refactor**
   - Call `_detect_header_rows_robust()` for EACH extractor
   - Call `_align_data_rows_by_content()` for pairwise comparisons
   - Compute separate metrics
   - Return detailed metadata

### Expected Output After Fix

```python
# Page 55 results:
match.data_row_accuracy = 1.00  # Perfect! (32/32 rows match)
match.header_accuracy = 0.35    # Different (7 vs 2 header rows)
match.metadata = {
    'header_rows': {'pdfplumber': 2, 'camelot': 7, 'pymupdf': 2},
    'aligned_data_rows': 32,
    'matching_data_rows': 32,
    'alignment': [(2, 7, 2), (3, 8, 3), ...],  # (pdf_row, cam_row, pym_row)
    'camelot_extras': ['Table title', 'Column indices', ...]
}
```

## Testing Plan

1. **Unit test**: `test_header_detection()`
   - Page 55: Should detect 2 (pdfplumber) vs 7 (Camelot)
   - Page 25: Should detect 1 (all extractors)
   - Page 70: Should detect variable

2. **Unit test**: `test_row_alignment()`
   - Page 55: Should align PA0-PA31 correctly
   - Verify (row 2, row 7) pair for PA0

3. **Integration test**: Re-run `test_extractor_harmony.py`
   - Page 25: Should still show 100% harmony
   - Page 55: Should show 100% data accuracy (not 48%)
   - Page 70: Should show 100% data accuracy

4. **Regression test**: Pages 30-32 multi-page
   - Should maintain 0.88 confidence
   - Should not break existing functionality

## Success Criteria

✅ Page 55 reports: `data_row_accuracy = 1.00` (currently 0.48)
✅ Page 70 reports: `data_row_accuracy = 1.00` (currently 1.00 - should maintain)
✅ Page 25 reports: `data_row_accuracy = 0.80` (currently 0.80 - should maintain)
✅ Alignment metadata shows correct row pairings
✅ Multi-page detection still works (pages 30-32)

## Files to Update After Fix

1. `docs/EXTRACTOR_HARMONY_ANALYSIS.md` - Corrected metrics
2. `docs/ARCHITECTURE.md` - Updated consensus scoring description
3. `README.md` - Updated feature description if needed
