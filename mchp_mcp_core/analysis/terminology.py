"""
Terminology Consistency Analyzer.

Extracts technical terms and detects variations/inconsistencies across
documentation to maintain professional consistency.

Examples of issues detected:
- Brand inconsistency: "Wi-Fi" vs "WiFi" vs "wifi"
- Formatting inconsistency: "I2C" vs "I²C"
- Terminology inconsistency: "SPI module" vs "SPI peripheral" vs "SPI interface"

The analyzer is configurable with custom term patterns for domain-specific
terminology beyond the provided examples.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Term:
    """
    Represents a technical term instance found in documentation.

    Attributes:
        term_id: Unique identifier for this term instance
        canonical_form: Normalized form for grouping (lowercase, no special chars)
        actual_form: How the term actually appears in the document
        category: Term category (e.g., 'connectivity', 'peripheral', 'feature')
        locations: List of (chunk_id, page_number) tuples where term appears
        count: Number of occurrences
    """
    term_id: str
    canonical_form: str
    actual_form: str
    category: str
    locations: list[tuple[str, int]]
    count: int = 1


@dataclass
class TermVariation:
    """
    Represents multiple variations of the same term found in documentation.

    Attributes:
        canonical: Canonical (normalized) form of the term
        variations: Dict mapping each variation to its occurrence count
        category: Term category
        total_instances: Total occurrences across all variations
        recommended_form: Suggested standard form to use
        severity: Issue severity ('critical', 'high', 'medium', 'low')
        reason: Explanation of why this is flagged
    """
    canonical: str
    variations: dict[str, int]
    category: str
    total_instances: int
    recommended_form: str
    severity: str
    reason: str


class TerminologyAnalyzer:
    """
    Analyzes terminology consistency across technical documentation.

    The analyzer uses regex patterns to identify technical terms and their
    variations, then flags inconsistencies based on configurable severity rules.

    Example:
        >>> # Use default patterns
        >>> analyzer = TerminologyAnalyzer()
        >>> report = await analyzer.analyze_terminology(chunks)
        >>> print(f"Found {report['inconsistent_terms']} issues")

        >>> # Custom patterns for your domain
        >>> custom_patterns = {
        ...     'networking': [
        ...         (r'TCP/IP', 'TCP/IP'),
        ...         (r'tcpip', 'TCP/IP'),
        ...     ]
        ... }
        >>> analyzer = TerminologyAnalyzer(term_patterns=custom_patterns)
    """

    # Default term patterns (examples - customize for your domain)
    DEFAULT_TERM_PATTERNS = {
        'connectivity': [
            # Wi-Fi variations
            (r'Wi-?Fi[®™]?', 'Wi-Fi®'),
            (r'WiFi[®™]?', 'Wi-Fi®'),
            (r'wifi', 'Wi-Fi®'),
            (r'Wi-?fi', 'Wi-Fi®'),
            # Bluetooth
            (r'Bluetooth[®™]?', 'Bluetooth®'),
            (r'BT\b', 'Bluetooth®'),
            # USB
            (r'USB\s*[\d.]+', 'USB'),
            (r'Universal\s+Serial\s+Bus', 'USB'),
            # Ethernet
            (r'Ethernet', 'Ethernet'),
            (r'ETH\b', 'Ethernet'),
        ],
        'peripherals': [
            # SPI variations
            (r'SPI\s+(?:module|peripheral|interface)', 'SPI peripheral'),
            (r'Serial\s+Peripheral\s+Interface', 'SPI peripheral'),
            # I2C variations
            (r'I[2²]C', 'I²C'),
            (r'Inter-Integrated\s+Circuit', 'I²C'),
            # UART
            (r'UART', 'UART'),
            (r'Universal\s+Asynchronous\s+Receiver/Transmitter', 'UART'),
            # CAN
            (r'CAN\s*FD?', 'CAN'),
            (r'Controller\s+Area\s+Network', 'CAN'),
            # ADC/DAC
            (r'ADC', 'ADC'),
            (r'Analog.to.Digital\s+Converter', 'ADC'),
            (r'DAC', 'DAC'),
            (r'Digital.to.Analog\s+Converter', 'DAC'),
        ],
        'features': [
            # DMA
            (r'DMA', 'DMA'),
            (r'Direct\s+Memory\s+Access', 'DMA'),
            # PWM
            (r'PWM', 'PWM'),
            (r'Pulse.Width\s+Modulation', 'PWM'),
            # Timer variations
            (r'[Tt]imer', 'timer'),
            (r'TIMER', 'Timer'),
        ],
        'memory': [
            # Flash
            (r'Flash\s+memory', 'Flash memory'),
            (r'flash', 'Flash'),
            # SRAM
            (r'SRAM', 'SRAM'),
            (r'Static\s+RAM', 'SRAM'),
            # EEPROM
            (r'EEPROM', 'EEPROM'),
        ]
    }

    def __init__(
        self,
        term_patterns: Optional[dict[str, list[tuple[str, str]]]] = None,
        enabled: bool = True
    ):
        """
        Initialize terminology analyzer.

        Args:
            term_patterns: Custom term patterns dict mapping category to
                          list of (regex, recommended_form) tuples.
                          If None, uses DEFAULT_TERM_PATTERNS.
            enabled: Whether analysis is enabled

        Example:
            >>> custom = {
            ...     'protocols': [
            ...         (r'TCP/IP', 'TCP/IP'),
            ...         (r'tcpip', 'TCP/IP'),
            ...     ]
            ... }
            >>> analyzer = TerminologyAnalyzer(term_patterns=custom)
        """
        self.enabled = enabled
        self.term_patterns = term_patterns or self.DEFAULT_TERM_PATTERNS

        # Build comprehensive pattern list with compiled regexes
        self.all_patterns: list[tuple[re.Pattern, str, str]] = []
        for category, patterns in self.term_patterns.items():
            for pattern, recommended in patterns:
                self.all_patterns.append((
                    re.compile(pattern, re.IGNORECASE),
                    recommended,
                    category
                ))

    async def analyze_terminology(self, chunks: list[Any]) -> dict[str, Any]:
        """
        Analyze terminology consistency across all chunks.

        Args:
            chunks: Document chunks with 'content', 'chunk_id', and 'page_start' attributes

        Returns:
            Terminology analysis report dict with keys:
                - enabled: Whether analysis was run
                - total_unique_terms: Number of unique terms found
                - total_instances: Total term occurrences
                - inconsistent_terms: Number of terms with variations
                - consistency_rate: Percentage of consistent terms
                - by_severity: Count dict by severity level
                - critical_issues: List of critical issues
                - high_priority_issues: List of high priority issues
                - medium_priority_issues: List of medium priority issues
                - low_priority_issues: List of low priority issues
                - summary_text: Human-readable summary

        Example:
            >>> report = await analyzer.analyze_terminology(chunks)
            >>> print(report['summary_text'])
            >>> for issue in report['critical_issues']:
            ...     print(f"{issue['term']}: {issue['recommended']}")
        """
        if not self.enabled:
            return {'enabled': False}

        logger.info("Starting terminology consistency analysis...")

        # Step 1: Extract all term instances
        term_instances = self._extract_term_instances(chunks)
        logger.info(f"Extracted {len(term_instances)} term instances")

        # Step 2: Group by canonical form (detect variations)
        term_groups = self._group_variations(term_instances)
        logger.info(f"Found {len(term_groups)} unique terms")

        # Step 3: Identify inconsistencies
        inconsistencies = self._identify_inconsistencies(term_groups)
        logger.info(f"Detected {len(inconsistencies)} terminology inconsistencies")

        # Step 4: Generate report
        report = self._generate_report(inconsistencies, term_groups)

        return report

    def _extract_term_instances(self, chunks: list[Any]) -> list[Term]:
        """Extract all technical term instances from chunks."""
        term_instances: list[Term] = []
        term_id_counter = 0

        for chunk in chunks:
            content = chunk.content

            # Try each pattern
            for pattern_re, recommended, category in self.all_patterns:
                matches = pattern_re.finditer(content)

                for match in matches:
                    actual_form = match.group(0)
                    canonical = self._canonicalize(actual_form)

                    term = Term(
                        term_id=f"term_{term_id_counter}",
                        canonical_form=canonical,
                        actual_form=actual_form,
                        category=category,
                        locations=[(chunk.chunk_id, chunk.page_start)],
                        count=1
                    )
                    term_instances.append(term)
                    term_id_counter += 1

        return term_instances

    def _canonicalize(self, term: str) -> str:
        """
        Convert term to canonical form for grouping.

        Examples:
            "Wi-Fi®" -> "wifi"
            "WiFi" -> "wifi"
            "I²C" -> "i2c"
            "I2C" -> "i2c"
        """
        # Remove special characters
        canonical = re.sub(r'[®™©\-_\s]', '', term)
        # Normalize superscripts
        canonical = canonical.replace('²', '2').replace('³', '3')
        # Lowercase
        canonical = canonical.lower()

        return canonical

    def _group_variations(
        self,
        term_instances: list[Term]
    ) -> dict[str, list[Term]]:
        """Group term instances by canonical form."""
        groups: dict[str, list[Term]] = defaultdict(list)

        for term in term_instances:
            groups[term.canonical_form].append(term)

        return dict(groups)

    def _identify_inconsistencies(
        self,
        term_groups: dict[str, list[Term]]
    ) -> list[TermVariation]:
        """
        Identify terms with multiple variations.

        Args:
            term_groups: Terms grouped by canonical form

        Returns:
            List of terms with inconsistent usage, sorted by severity and frequency
        """
        inconsistencies: list[TermVariation] = []

        for canonical, instances in term_groups.items():
            # Count variations
            variation_counts = Counter(term.actual_form for term in instances)

            # Only report if there are multiple variations
            if len(variation_counts) > 1:
                # Determine recommended form
                category = instances[0].category
                recommended = self._determine_recommended_form(
                    canonical, variation_counts, category
                )

                # Calculate severity
                severity, reason = self._calculate_severity(
                    canonical, variation_counts, category
                )

                inconsistency = TermVariation(
                    canonical=canonical,
                    variations=dict(variation_counts),
                    category=category,
                    total_instances=sum(variation_counts.values()),
                    recommended_form=recommended,
                    severity=severity,
                    reason=reason
                )
                inconsistencies.append(inconsistency)

        # Sort by severity and total instances
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        inconsistencies.sort(
            key=lambda x: (severity_order[x.severity], -x.total_instances)
        )

        return inconsistencies

    def _determine_recommended_form(
        self,
        canonical: str,
        variations: Counter,
        category: str
    ) -> str:
        """
        Determine the recommended form for a term.

        Priority:
        1. Branded forms (®, ™) take precedence
        2. Proper formatting (superscripts for I²C, etc.)
        3. Most common form
        """
        # Check for branded forms (®, ™)
        for form in variations:
            if '®' in form or '™' in form:
                return form

        # Check for proper formatting (superscripts for I²C, etc.)
        if canonical == 'i2c':
            for form in variations:
                if '²' in form:
                    return form

        # Default to most common form
        most_common = variations.most_common(1)[0][0]
        return most_common

    def _calculate_severity(
        self,
        canonical: str,
        variations: Counter,
        category: str
    ) -> tuple[str, str]:
        """
        Calculate severity of terminology inconsistency.

        Returns:
            Tuple of (severity, reason)
        """
        total = sum(variations.values())
        variation_count = len(variations)

        # Critical: Brand compliance issues
        if category == 'connectivity':
            has_branded = any('®' in v or '™' in v for v in variations)
            has_unbranded = any('®' not in v and '™' not in v for v in variations)

            if has_branded and has_unbranded:
                return 'critical', 'Brand compliance issue (mix of branded/unbranded)'

        # High: Many instances, many variations
        if total > 20 and variation_count > 3:
            return 'high', f'High frequency ({total} instances) with {variation_count} variations'

        # Medium: Moderate inconsistency
        if total > 10:
            return 'medium', f'Moderate frequency ({total} instances) with inconsistent usage'

        # Low: Few instances
        return 'low', f'Low frequency ({total} instances)'

    def _generate_report(
        self,
        inconsistencies: list[TermVariation],
        all_terms: dict[str, list[Term]]
    ) -> dict[str, Any]:
        """Generate terminology analysis report."""

        # Group by severity
        by_severity: dict[str, list[dict[str, Any]]] = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }

        for inc in inconsistencies:
            by_severity[inc.severity].append({
                'term': inc.canonical,
                'recommended': inc.recommended_form,
                'variations': inc.variations,
                'total_instances': inc.total_instances,
                'category': inc.category,
                'reason': inc.reason
            })

        # Calculate statistics
        total_unique_terms = len(all_terms)
        total_instances = sum(len(instances) for instances in all_terms.values())
        inconsistent_terms = len(inconsistencies)

        report = {
            'enabled': True,
            'total_unique_terms': total_unique_terms,
            'total_instances': total_instances,
            'inconsistent_terms': inconsistent_terms,
            'consistency_rate': round(
                100 * (1 - inconsistent_terms / total_unique_terms)
                if total_unique_terms > 0 else 0,
                1
            ),
            'by_severity': {
                'critical': len(by_severity['critical']),
                'high': len(by_severity['high']),
                'medium': len(by_severity['medium']),
                'low': len(by_severity['low'])
            },
            'critical_issues': by_severity['critical'],
            'high_priority_issues': by_severity['high'],
            'medium_priority_issues': by_severity['medium'],
            'low_priority_issues': by_severity['low'],
            'summary_text': self._generate_summary_text(
                inconsistencies, total_instances
            )
        }

        return report

    def _generate_summary_text(
        self,
        inconsistencies: list[TermVariation],
        total_instances: int
    ) -> str:
        """Generate human-readable summary."""
        if not inconsistencies:
            return "✅ No terminology inconsistencies detected. All technical terms are used consistently."

        critical = sum(1 for i in inconsistencies if i.severity == 'critical')
        high = sum(1 for i in inconsistencies if i.severity == 'high')

        summary = f"Found {len(inconsistencies)} terminology inconsistencies across {total_instances} term instances.\n"

        if critical > 0:
            summary += f"⚠️ {critical} critical issues (brand compliance)\n"

        if high > 0:
            summary += f"🔸 {high} high-priority issues (high frequency with many variations)\n"

        # Top 3 issues
        summary += "\nTop issues:\n"
        for i, inc in enumerate(inconsistencies[:3], 1):
            summary += f"{i}. \"{inc.canonical}\" - {len(inc.variations)} variations, recommend \"{inc.recommended_form}\"\n"

        return summary
