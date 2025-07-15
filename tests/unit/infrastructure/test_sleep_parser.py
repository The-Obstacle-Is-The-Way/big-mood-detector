"""
Tests for Apple HealthKit Sleep Data Parser

Test-driven development for clinical-grade sleep data extraction.
Following Clean Architecture principles.
"""

import pytest
from big_mood_detector.infrastructure.parsers.sleep_parser import SleepParser


class TestSleepParser:
    """Test suite for SleepParser - Apple HealthKit sleep data extraction."""
    
    def test_sleep_parser_exists(self):
        """Test that SleepParser class can be instantiated."""
        # ARRANGE & ACT
        parser = SleepParser()
        
        # ASSERT
        assert parser is not None
        assert isinstance(parser, SleepParser) 