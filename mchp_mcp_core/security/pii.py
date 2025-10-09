"""
PII (Personally Identifiable Information) redaction utilities.

Provides regex-based detection and redaction of sensitive personal information
including emails, phone numbers, SSNs, credit cards, IP addresses, and addresses.
"""

import re
from typing import Dict, List


class PIIRedactor:
    """
    Redact Personally Identifiable Information (PII) from text.

    Uses regex patterns to detect and redact common PII types.

    Example:
        >>> redactor = PIIRedactor()
        >>> text = "Contact me at john@example.com or 555-123-4567"
        >>> redactor.redact(text)
        'Contact me at [REDACTED] or [REDACTED]'
    """

    # Email addresses
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )

    # Phone numbers (various formats)
    PHONE_PATTERNS = [
        re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # 123-456-7890, 123.456.7890, 1234567890
        re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),  # (123) 456-7890
        re.compile(r'\b\+?1?\s*\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),  # +1 (123) 456-7890
    ]

    # Social Security Numbers
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    # Credit card numbers (simple 16-digit detection)
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

    # IP addresses
    IP_PATTERN = re.compile(
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    )

    # US addresses (simplified - looks for street patterns)
    # This is a basic pattern and may have false positives
    ADDRESS_PATTERN = re.compile(
        r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ssn: bool = True,
        redact_credit_cards: bool = True,
        redact_ip_addresses: bool = False,
        redact_addresses: bool = False,
        replacement_text: str = "[REDACTED]"
    ):
        """
        Initialize PII redactor.

        Args:
            redact_emails: Redact email addresses
            redact_phones: Redact phone numbers
            redact_ssn: Redact social security numbers
            redact_credit_cards: Redact credit card numbers
            redact_ip_addresses: Redact IP addresses (be careful, may over-redact)
            redact_addresses: Redact street addresses (basic pattern, may have false positives)
            replacement_text: Text to use for redacted content
        """
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssn = redact_ssn
        self.redact_credit_cards = redact_credit_cards
        self.redact_ip_addresses = redact_ip_addresses
        self.redact_addresses = redact_addresses
        self.replacement_text = replacement_text

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text potentially containing PII

        Returns:
            Text with PII redacted
        """
        if not text:
            return text

        redacted = text

        # Redact emails
        if self.redact_emails:
            redacted = self.EMAIL_PATTERN.sub(self.replacement_text, redacted)

        # Redact phone numbers
        if self.redact_phones:
            for pattern in self.PHONE_PATTERNS:
                redacted = pattern.sub(self.replacement_text, redacted)

        # Redact SSN
        if self.redact_ssn:
            redacted = self.SSN_PATTERN.sub(self.replacement_text, redacted)

        # Redact credit cards
        if self.redact_credit_cards:
            redacted = self.CREDIT_CARD_PATTERN.sub(self.replacement_text, redacted)

        # Redact IP addresses (optional, can be overly aggressive)
        if self.redact_ip_addresses:
            redacted = self.IP_PATTERN.sub(self.replacement_text, redacted)

        # Redact addresses (optional, basic pattern)
        if self.redact_addresses:
            redacted = self.ADDRESS_PATTERN.sub(self.replacement_text, redacted)

        return redacted

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text without redacting.

        Useful for logging/auditing what PII was found.

        Args:
            text: Input text

        Returns:
            Dictionary of PII type -> list of found instances
        """
        found = {
            "emails": [],
            "phones": [],
            "ssn": [],
            "credit_cards": [],
            "ip_addresses": [],
            "addresses": []
        }

        if not text:
            return found

        # Find emails
        if self.redact_emails:
            found["emails"] = self.EMAIL_PATTERN.findall(text)

        # Find phone numbers
        if self.redact_phones:
            for pattern in self.PHONE_PATTERNS:
                found["phones"].extend(pattern.findall(text))

        # Find SSN
        if self.redact_ssn:
            found["ssn"] = self.SSN_PATTERN.findall(text)

        # Find credit cards
        if self.redact_credit_cards:
            found["credit_cards"] = self.CREDIT_CARD_PATTERN.findall(text)

        # Find IP addresses
        if self.redact_ip_addresses:
            found["ip_addresses"] = self.IP_PATTERN.findall(text)

        # Find addresses
        if self.redact_addresses:
            found["addresses"] = self.ADDRESS_PATTERN.findall(text)

        return found


def redact_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """
    Convenience function to redact common PII types.

    Uses default settings: redacts emails, phones, SSN, credit cards.

    Args:
        text: Input text
        replacement: Replacement text for redacted content

    Returns:
        Text with PII redacted

    Example:
        >>> redact_pii("Contact me at john@example.com or 555-123-4567")
        'Contact me at [REDACTED] or [REDACTED]'
    """
    redactor = PIIRedactor(replacement_text=replacement)
    return redactor.redact(text)


__all__ = ["PIIRedactor", "redact_pii"]
