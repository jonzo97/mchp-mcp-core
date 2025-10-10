"""
Output filtering and severity classification module.

Provides utilities for filtering, ranking, and prioritizing search results
or review findings by severity and relevance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from mchp_mcp_core.storage import SearchResult
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    """Severity levels for review findings or issues."""
    CRITICAL = 0    # Technical errors, broken references, unsupported claims
    HIGH = 1        # Missing sections, important suggestions, consistency issues
    MEDIUM = 2      # Terminology inconsistencies, style violations
    LOW = 3         # Minor formatting, suggestions
    IGNORE = 4      # Whitespace, trivial formatting (suppressed by default)


@dataclass
class FilteredChange:
    """A change or finding with severity classification."""
    original: str
    corrected: str
    reason: str
    change_type: str
    severity: Severity
    position: int
    confidence: float
    metadata: Dict[str, Any] = None


class OutputFilter:
    """
    Filter and prioritize output by severity and relevance.

    Features:
    - Severity classification (CRITICAL, HIGH, MEDIUM, LOW, IGNORE)
    - Verbosity control (quiet, normal, verbose)
    - Change categorization and filtering
    - Summary formatting with emojis
    - Statistics generation

    Example:
        >>> filter_obj = OutputFilter(verbosity='normal')
        >>> changes = [
        ...     {'type': 'crossref', 'reason': 'section not found', 'valid': False},
        ...     {'type': 'style', 'reason': 'terminology standardization'},
        ... ]
        >>> filtered = filter_obj.filter_changes(changes)
        >>> print(filter_obj.format_summary(filtered))
        🔴 Critical Issues (1):
           1. section not found
        🟡 Medium Priority (1):
           - 1 terminology inconsistencies
    """

    def __init__(self, verbosity: str = 'normal'):
        """
        Initialize output filter.

        Args:
            verbosity: Verbosity level ('quiet', 'normal', or 'verbose')
                - quiet: Only critical and high priority
                - normal: Critical, high, and medium priority
                - verbose: Everything including low priority
        """
        self.verbosity = verbosity

        # Severity thresholds by verbosity
        self.min_severity = {
            'quiet': Severity.HIGH,
            'normal': Severity.MEDIUM,
            'verbose': Severity.IGNORE
        }.get(verbosity, Severity.MEDIUM)

    def classify_change(self, change: Dict) -> Severity:
        """
        Classify a change or finding by severity.

        Classification rules:
        - CRITICAL: Broken references, unsupported claims, technical errors
        - HIGH: LLM suggestions, missing sections, low confidence changes
        - MEDIUM: Terminology inconsistencies, style violations, spelling
        - LOW: Minor formatting issues
        - IGNORE: Whitespace, trivial spacing

        Args:
            change: Change dictionary with 'type', 'reason', 'confidence', etc.

        Returns:
            Severity level
        """
        change_type = change.get('type', '').lower()
        reason = change.get('reason', '').lower()

        # CRITICAL: Technical accuracy errors
        if change_type == 'crossref' and not change.get('valid', True):
            return Severity.CRITICAL

        if 'unsupported claim' in reason or 'missing evidence' in reason:
            return Severity.CRITICAL

        if 'broken reference' in reason or 'section not found' in reason:
            return Severity.CRITICAL

        if 'technical error' in reason or 'incorrect spec' in reason:
            return Severity.CRITICAL

        # HIGH: Important but not critical
        if change_type == 'llm_suggestion':
            return Severity.HIGH

        if 'missing section' in reason or 'incomplete' in reason:
            return Severity.HIGH

        if change.get('confidence', 1.0) < 0.8 and change_type != 'grammar':
            return Severity.HIGH

        if 'ambiguous' in reason or 'unclear' in reason:
            return Severity.HIGH

        # MEDIUM: Consistency and style
        if 'terminology' in reason or 'inconsistent' in reason:
            return Severity.MEDIUM

        if change_type == 'style':
            return Severity.MEDIUM

        if 'spelling' in change_type:
            return Severity.MEDIUM

        if change_type == 'grammar' and 'agreement' in reason:
            return Severity.MEDIUM

        # LOW: Minor issues
        if 'formatting' in reason and 'space' not in reason:
            return Severity.LOW

        if change_type == 'suggestion':
            return Severity.LOW

        # IGNORE: Trivial whitespace
        if change_type == 'grammar' and ('double space' in reason or 'whitespace' in reason):
            return Severity.IGNORE

        if 'extra space' in reason or 'spacing' in reason:
            return Severity.IGNORE

        # Default to LOW for unknown types
        return Severity.LOW

    def filter_changes(self, changes: List[Dict]) -> Dict[str, Any]:
        """
        Filter and categorize changes by severity.

        Args:
            changes: List of change dictionaries

        Returns:
            Dictionary with categorized changes:
            - 'critical': List of critical issues
            - 'high': List of high priority issues
            - 'medium': List of medium priority issues
            - 'low': List of low priority issues
            - 'suppressed': Count of suppressed items
        """
        categorized = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'suppressed': 0
        }

        for change in changes:
            severity = self.classify_change(change)

            # Apply verbosity filter
            if severity.value > self.min_severity.value:
                categorized['suppressed'] += 1
                continue

            # Add to appropriate category with severity label
            change_with_severity = {**change, 'severity': severity.name.lower()}

            if severity == Severity.CRITICAL:
                categorized['critical'].append(change_with_severity)
            elif severity == Severity.HIGH:
                categorized['high'].append(change_with_severity)
            elif severity == Severity.MEDIUM:
                categorized['medium'].append(change_with_severity)
            elif severity == Severity.LOW:
                categorized['low'].append(change_with_severity)

        return categorized

    def format_summary(self, filtered_changes: Dict[str, List[Dict]]) -> str:
        """
        Format filtered changes as a readable summary with emoji indicators.

        Args:
            filtered_changes: Output from filter_changes()

        Returns:
            Formatted summary string with color-coded severity indicators
        """
        lines = []

        # Critical issues
        if filtered_changes['critical']:
            lines.append(f"🔴 Critical Issues ({len(filtered_changes['critical'])}):")
            for i, change in enumerate(filtered_changes['critical'][:10], 1):  # Top 10
                lines.append(f"   {i}. {change.get('reason', 'Unknown issue')}")
                if change.get('original'):
                    lines.append(f"      Original: \"{change['original']}\"")
                if change.get('corrected'):
                    lines.append(f"      Corrected: \"{change['corrected']}\"")
            if len(filtered_changes['critical']) > 10:
                lines.append(f"   ... and {len(filtered_changes['critical']) - 10} more")
            lines.append("")

        # High priority
        if filtered_changes['high']:
            lines.append(f"🟠 High Priority ({len(filtered_changes['high'])}):")
            for i, change in enumerate(filtered_changes['high'][:5], 1):  # Top 5
                lines.append(f"   {i}. {change.get('reason', 'Unknown issue')}")
            if len(filtered_changes['high']) > 5:
                lines.append(f"   ... and {len(filtered_changes['high']) - 5} more")
            lines.append("")

        # Medium priority (summary only in normal mode)
        if filtered_changes['medium'] and self.verbosity != 'quiet':
            lines.append(f"🟡 Medium Priority ({len(filtered_changes['medium'])}):")
            if self.verbosity == 'verbose':
                for i, change in enumerate(filtered_changes['medium'][:10], 1):
                    lines.append(f"   {i}. {change.get('reason', 'Unknown issue')}")
            else:
                term_count = len([c for c in filtered_changes['medium'] if 'terminology' in c.get('reason', '')])
                style_count = len([c for c in filtered_changes['medium'] if 'style' in c.get('type', '')])
                if term_count > 0:
                    lines.append(f"   - {term_count} terminology inconsistencies")
                if style_count > 0:
                    lines.append(f"   - {style_count} style suggestions")
            lines.append("")

        # Low priority (count only in verbose mode)
        if filtered_changes['low'] and self.verbosity == 'verbose':
            lines.append(f"🟢 Low Priority ({len(filtered_changes['low'])}): Minor formatting suggestions")
            lines.append("")

        # Suppressed count
        if filtered_changes['suppressed'] > 0:
            lines.append(
                f"⚪ Suppressed: {filtered_changes['suppressed']} low-value items "
                f"(use verbosity='verbose' to see)"
            )
            lines.append("")

        # Summary stats
        total_actionable = (
            len(filtered_changes['critical']) +
            len(filtered_changes['high']) +
            len(filtered_changes['medium'])
        )

        if total_actionable > 0:
            lines.append(f"📊 Actionable Items: {total_actionable}")
        else:
            lines.append("✅ No significant issues found!")

        return "\n".join(lines)

    def get_statistics(self, filtered_changes: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Get statistics about filtered changes.

        Args:
            filtered_changes: Output from filter_changes()

        Returns:
            Dictionary with statistics:
            - total_changes: Total number of changes
            - actionable_changes: Critical + high priority
            - by_severity: Count by each severity level
            - signal_to_noise_ratio: Ratio of actionable to total
        """
        total_changes = (
            len(filtered_changes['critical']) +
            len(filtered_changes['high']) +
            len(filtered_changes['medium']) +
            len(filtered_changes['low']) +
            filtered_changes['suppressed']
        )

        actionable_changes = (
            len(filtered_changes['critical']) +
            len(filtered_changes['high'])
        )

        return {
            'total_changes': total_changes,
            'actionable_changes': actionable_changes,
            'by_severity': {
                'critical': len(filtered_changes['critical']),
                'high': len(filtered_changes['high']),
                'medium': len(filtered_changes['medium']),
                'low': len(filtered_changes['low']),
                'suppressed': filtered_changes['suppressed']
            },
            'signal_to_noise_ratio': actionable_changes / max(1, total_changes)
        }


def prioritize_for_review(changes: List[Dict], max_items: int = 50) -> List[Dict]:
    """
    Prioritize changes for human review queue.

    Orders changes by severity (critical → high → medium → low) and
    returns up to max_items prioritized items.

    Args:
        changes: List of all changes
        max_items: Maximum items to return

    Returns:
        Prioritized list of changes (highest severity first)

    Example:
        >>> changes = get_all_changes()
        >>> top_priority = prioritize_for_review(changes, max_items=20)
        >>> for change in top_priority:
        ...     print(f"{change['severity']}: {change['reason']}")
    """
    filter_obj = OutputFilter()
    categorized = filter_obj.filter_changes(changes)

    prioritized = []

    # Add critical (all of them)
    prioritized.extend(categorized['critical'])

    # Add high (up to limit)
    remaining = max_items - len(prioritized)
    if remaining > 0:
        prioritized.extend(categorized['high'][:remaining])

    # Add medium (up to limit)
    remaining = max_items - len(prioritized)
    if remaining > 0:
        prioritized.extend(categorized['medium'][:remaining])

    # Add low (up to limit)
    remaining = max_items - len(prioritized)
    if remaining > 0:
        prioritized.extend(categorized['low'][:remaining])

    return prioritized


def filter_results_by_score(
    results: List[SearchResult],
    min_score: float = 0.7,
    max_results: int = 10
) -> List[SearchResult]:
    """
    Filter search results by minimum score threshold.

    Args:
        results: List of search results
        min_score: Minimum relevance score (0.0 to 1.0)
        max_results: Maximum number of results to return

    Returns:
        Filtered and truncated list of search results

    Example:
        >>> filtered = filter_results_by_score(
        ...     results,
        ...     min_score=0.75,
        ...     max_results=5
        ... )
    """
    filtered = [r for r in results if r.score >= min_score]
    return filtered[:max_results]


def deduplicate_results(
    results: List[SearchResult],
    similarity_threshold: float = 0.95
) -> List[SearchResult]:
    """
    Deduplicate search results based on content similarity.

    Uses simple overlap heuristic: if snippet similarity > threshold,
    keep only the higher-scored result.

    Args:
        results: List of search results
        similarity_threshold: Jaccard similarity threshold (0.0 to 1.0)

    Returns:
        Deduplicated list of search results

    Example:
        >>> unique_results = deduplicate_results(results, similarity_threshold=0.9)
    """
    if len(results) <= 1:
        return results

    # Simple deduplication by snippet similarity
    deduplicated = []
    seen_snippets = []

    for result in results:
        snippet_words = set(result.snippet.lower().split())

        # Check similarity with all seen snippets
        is_duplicate = False
        for seen_words in seen_snippets:
            # Calculate Jaccard similarity
            intersection = len(snippet_words & seen_words)
            union = len(snippet_words | seen_words)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(result)
            seen_snippets.append(snippet_words)

    return deduplicated
