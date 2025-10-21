# Page 25 - Consensus BBox Results

**Generated**: 2025-10-20 (with consensus bbox fix)

## Consensus Metrics

- **Confidence**: 0.82
- **Agreement Score**: 1.00
- **Structure Score**: 0.50
- **Cell Similarity**: 0.35

### Critical Metrics:
- **Header Accuracy**: 0.25
- **Data Row Accuracy**: 0.42 ❌
- **Estimated Header Rows**: 5

## Extractors


### pdfplumber
- **Rows**: 8
- **Columns**: 5
- **BBox**: None
- **Sparsity**: 0.03
- **PA0 found**: ❌ NOT FOUND

### camelot
- **Rows**: 20
- **Columns**: 5
- **BBox**: (62.62007999999997, 195.09009999999995, 556.7718505859375, 588.7659708658854)
- **Sparsity**: 0.53
- **PA0 found**: ❌ NOT FOUND

### pymupdf
- **Rows**: 8
- **Columns**: 5
- **BBox**: (79.40130805969238, 364.93133544921875, 546.7718505859375, 578.7659708658854)
- **Sparsity**: 0.03
- **PA0 found**: ❌ NOT FOUND

## Expected Behavior (Post-Fix)

All extractors should:
- ✅ Use same table region (consensus bbox)
- ✅ Detect PA0 (was missing in Camelot before fix)
- ⚠️ May have different row counts (header detection issue - Task #7)
- ⚠️ May have PA0 at different row indices (alignment issue - Task #8)
