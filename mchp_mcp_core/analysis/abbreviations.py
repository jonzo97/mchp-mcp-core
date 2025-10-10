"""
Technical abbreviation expansion for improved search recall.

Expands common technical abbreviations to their full forms, helping non-expert
users find documents and improving search recall when queries use full terms.
"""
from __future__ import annotations

from typing import Optional

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


# Default technical abbreviations (extensible)
DEFAULT_ABBREVIATIONS = {
    # Communication protocols
    'SPI': 'Serial Peripheral Interface',
    'I2C': 'Inter-Integrated Circuit',
    'UART': 'Universal Asynchronous Receiver/Transmitter',
    'USB': 'Universal Serial Bus',
    'CAN': 'Controller Area Network',
    'CANFD': 'CAN with Flexible Data-Rate',
    'LIN': 'Local Interconnect Network',
    'PCIe': 'Peripheral Component Interconnect Express',
    'SATA': 'Serial AT Attachment',

    # Wireless
    'WiFi': 'Wireless Fidelity',
    'BLE': 'Bluetooth Low Energy',
    'NFC': 'Near Field Communication',
    'RFID': 'Radio-Frequency Identification',
    'LoRa': 'Long Range',
    'LoRaWAN': 'Long Range Wide Area Network',

    # Memory
    'SRAM': 'Static Random-Access Memory',
    'DRAM': 'Dynamic Random-Access Memory',
    'SDRAM': 'Synchronous Dynamic Random-Access Memory',
    'DDR': 'Double Data Rate',
    'EEPROM': 'Electrically Erasable Programmable Read-Only Memory',
    'EPROM': 'Erasable Programmable Read-Only Memory',
    'ROM': 'Read-Only Memory',
    'RAM': 'Random-Access Memory',
    'FLASH': 'Flash Memory',

    # Processing
    'CPU': 'Central Processing Unit',
    'GPU': 'Graphics Processing Unit',
    'MCU': 'Microcontroller Unit',
    'MPU': 'Microprocessor Unit',
    'DSP': 'Digital Signal Processor',
    'FPGA': 'Field-Programmable Gate Array',
    'ASIC': 'Application-Specific Integrated Circuit',
    'SoC': 'System on Chip',

    # Peripherals
    'ADC': 'Analog-to-Digital Converter',
    'DAC': 'Digital-to-Analog Converter',
    'PWM': 'Pulse-Width Modulation',
    'DMA': 'Direct Memory Access',
    'GPIO': 'General-Purpose Input/Output',
    'RTC': 'Real-Time Clock',
    'WDT': 'Watchdog Timer',
    'CRC': 'Cyclic Redundancy Check',

    # Standards
    'IEEE': 'Institute of Electrical and Electronics Engineers',
    'ISO': 'International Organization for Standardization',
    'IEC': 'International Electrotechnical Commission',
    'ANSI': 'American National Standards Institute',

    # Security
    'AES': 'Advanced Encryption Standard',
    'DES': 'Data Encryption Standard',
    'RSA': 'Rivest-Shamir-Adleman',
    'SHA': 'Secure Hash Algorithm',
    'TLS': 'Transport Layer Security',
    'SSL': 'Secure Sockets Layer',

    # Power
    'PMU': 'Power Management Unit',
    'PMIC': 'Power Management Integrated Circuit',
    'LDO': 'Low-Dropout Regulator',
    'DCDC': 'DC-to-DC Converter',
    'SMPS': 'Switched-Mode Power Supply',
}


class AbbreviationExpander:
    """
    Expand technical abbreviations to improve search recall.

    Example:
        >>> expander = AbbreviationExpander()
        >>> text = "The SPI interface supports I2C fallback"
        >>> expanded = expander.expand(text)
        >>> print(expanded)
        "The Serial Peripheral Interface (SPI) supports Inter-Integrated Circuit (I2C) fallback"
    """

    def __init__(
        self,
        abbreviations: Optional[dict[str, str]] = None,
        include_defaults: bool = True
    ):
        """
        Initialize abbreviation expander.

        Args:
            abbreviations: Custom abbreviation dict. If None, uses DEFAULT_ABBREVIATIONS.
            include_defaults: If True and custom abbreviations provided, merge with defaults.

        Example:
            >>> # Use defaults
            >>> expander = AbbreviationExpander()
            >>>
            >>> # Custom only
            >>> expander = AbbreviationExpander(
            ...     abbreviations={'API': 'Application Programming Interface'},
            ...     include_defaults=False
            ... )
            >>>
            >>> # Merge with defaults
            >>> expander = AbbreviationExpander(
            ...     abbreviations={'API': 'Application Programming Interface'},
            ...     include_defaults=True
            ... )
        """
        if abbreviations and include_defaults:
            self.abbreviations = {**DEFAULT_ABBREVIATIONS, **abbreviations}
        elif abbreviations:
            self.abbreviations = abbreviations
        else:
            self.abbreviations = DEFAULT_ABBREVIATIONS

    def expand(self, text: str, format_style: str = 'parenthetical') -> str:
        """
        Expand abbreviations in text.

        Args:
            text: Input text with abbreviations
            format_style: How to format expansions:
                - 'parenthetical': "SPI" → "Serial Peripheral Interface (SPI)"
                - 'replace': "SPI" → "Serial Peripheral Interface"
                - 'append': "SPI" → "SPI (Serial Peripheral Interface)"

        Returns:
            Text with expanded abbreviations

        Example:
            >>> expander = AbbreviationExpander()
            >>> expander.expand("SPI bus", format_style='parenthetical')
            "Serial Peripheral Interface (SPI) bus"
            >>> expander.expand("SPI bus", format_style='append')
            "SPI (Serial Peripheral Interface) bus"
        """
        result = text

        for abbr, full in self.abbreviations.items():
            # Only match whole words (not part of larger words)
            import re
            pattern = rf'\b{re.escape(abbr)}\b'

            if format_style == 'parenthetical':
                replacement = f"{full} ({abbr})"
            elif format_style == 'replace':
                replacement = full
            elif format_style == 'append':
                replacement = f"{abbr} ({full})"
            else:
                replacement = f"{full} ({abbr})"

            result = re.sub(pattern, replacement, result)

        return result

    def expand_query(self, query: str) -> str:
        """
        Expand abbreviations in search query (uses 'append' style).

        This is optimized for search: keeps original term and adds expansion
        to improve recall without losing precision.

        Example:
            >>> expander = AbbreviationExpander()
            >>> expander.expand_query("SPI timing")
            "SPI (Serial Peripheral Interface) timing"
        """
        return self.expand(query, format_style='append')

    def get_expansion(self, abbr: str) -> Optional[str]:
        """
        Get expansion for a single abbreviation.

        Args:
            abbr: Abbreviation to look up

        Returns:
            Full form if found, None otherwise

        Example:
            >>> expander = AbbreviationExpander()
            >>> expander.get_expansion('SPI')
            'Serial Peripheral Interface'
        """
        return self.abbreviations.get(abbr.upper())


# Convenience function
def expand_abbreviations(
    text: str,
    abbreviations: Optional[dict[str, str]] = None,
    format_style: str = 'parenthetical'
) -> str:
    """
    Expand abbreviations in text (convenience function).

    Args:
        text: Input text
        abbreviations: Custom abbreviations (uses defaults if None)
        format_style: Formatting style ('parenthetical', 'replace', 'append')

    Returns:
        Text with expanded abbreviations

    Example:
        >>> expand_abbreviations("SPI and I2C protocols")
        "Serial Peripheral Interface (SPI) and Inter-Integrated Circuit (I2C) protocols"
    """
    expander = AbbreviationExpander(abbreviations)
    return expander.expand(text, format_style)
