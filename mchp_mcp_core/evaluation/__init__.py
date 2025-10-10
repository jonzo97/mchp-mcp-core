"""
Evaluation module for retrieval quality assessment.

Provides metrics and utilities for evaluating search/retrieval systems:
- Precision@K (P@1, P@3, P@5, etc.)
- Mean Reciprocal Rank (MRR)
- Recall@K
- Batch evaluation with reporting
"""

from mchp_mcp_core.evaluation.metrics import (
    precision_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    QueryEvaluation,
    EvaluationReport,
    EvaluationMetrics
)

__all__ = [
    # Metrics functions
    "precision_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    # Data models
    "QueryEvaluation",
    "EvaluationReport",
    # Evaluation class
    "EvaluationMetrics",
]
