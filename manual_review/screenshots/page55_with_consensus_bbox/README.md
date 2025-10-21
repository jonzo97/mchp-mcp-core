# Page 55 - Consensus BBox Results

**Generated**: 2025-10-20 (with consensus bbox fix)

## Consensus Metrics

- **Confidence**: 0.90
- **Agreement Score**: 1.00
- **Structure Score**: 0.90
- **Cell Similarity**: 0.50

### Critical Metrics:
- **Header Accuracy**: 0.35
- **Data Row Accuracy**: 0.48 ❌
- **Estimated Header Rows**: 5

## Extractors


### pdfplumber
- **Rows**: 34
- **Columns**: 9
- **BBox**: None
- **Sparsity**: 0.12
- **PA0 found**: Row 2

### camelot
- **Rows**: 41
- **Columns**: 9
- **BBox**: (34.27362, 59.53182705965909, 579.67966, 756.4677222992701)
- **Sparsity**: 0.25
- **PA0 found**: Row 7

### pymupdf
- **Rows**: 34
- **Columns**: 9
- **BBox**: (51.13555714621473, 69.5318270596591, 560.9208238389757, 555.3356548108553)
- **Sparsity**: 0.12
- **PA0 found**: Row 2

## Expected Behavior (Post-Fix)

All extractors should:
- ✅ Use same table region (consensus bbox)
- ✅ Detect PA0 (was missing in Camelot before fix)
- ⚠️ May have different row counts (header detection issue - Task #7)
- ⚠️ May have PA0 at different row indices (alignment issue - Task #8)
