# Archived Screenshots - Pre-BBox Fix (2025-10-20)

These screenshots were generated BEFORE the consensus bbox implementation.

**Issue**: Extractors used different bounding boxes, causing misalignment.
- Camelot's bbox: Y=228 (WRONG - missed top of table)
- PyMuPDF's bbox: Y=69 (CORRECT)
- pdfplumber: No bbox (used full page)

**Result**: Camelot screenshots are visibly cropped at the top, missing PA0.

**Fixed**: 2025-10-20 with consensus bbox (union strategy)

