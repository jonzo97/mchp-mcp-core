# Page 70 - Consensus BBox Results

**Generated**: 2025-10-20 (with consensus bbox fix)

## Consensus Metrics

- **Confidence**: 0.88
- **Agreement Score**: 1.00
- **Structure Score**: 0.50
- **Cell Similarity**: 0.40

### Critical Metrics:
- **Header Accuracy**: 0.35
- **Data Row Accuracy**: 1.00 ✅
- **Estimated Header Rows**: 5

## Extractors


### pdfplumber
- **Rows**: 5
- **Columns**: 9
- **BBox**: None
- **Sparsity**: 0.36
- **PA0 found**: ❌ NOT FOUND

### camelot
- **Rows**: 14
- **Columns**: 9
- **BBox**: (34.273619999999994, 46.60959498087565, 578.9130299999999, 756.0811345238095)
- **Sparsity**: 0.63
- **PA0 found**: ❌ NOT FOUND

### pymupdf
- **Rows**: 5
- **Columns**: 9
- **BBox**: (51.02361721462674, 56.60959498087565, 560.9763793945312, 156.1329872824929)
- **Sparsity**: 0.36
- **PA0 found**: ❌ NOT FOUND

## Expected Behavior (Post-Fix)

All extractors should:
- ✅ Use same table region (consensus bbox)
- ✅ Detect PA0 (was missing in Camelot before fix)
- ⚠️ May have different row counts (header detection issue - Task #7)
- ⚠️ May have PA0 at different row indices (alignment issue - Task #8)
