"""
Tests for JSON-based Health Data Parsers
Following TDD and Clean Architecture principles.
"""

import json
from datetime import date

import pytest

class TestSleepJSONParser:
    """Test cases for Sleep JSON parser."""

    @pytest.fixture
    def parser(self):
        return SleepJSONParser()

    @pytest.fixture
    def sleep_json_data(self):
        """Sample sleep data in Apple Health export JSON format."""
        return {
            "data": {
                "metrics": [
                    {
                        "name": "sleep_analysis",
                        "data": [
                            {
                                "rem": 0.93333333333333335,
                                "totalSleep": 6.7833333333333341,
                                "source": "John's Apple Watch",
                                "deep": 0.60833333333333328,
                                "sleepStart": "2025-01-14 23:29:06 -0500",
                                "date": "2025-01-15 00:00:00 -0500",
                                "sleepEnd": "2025-01-15 06:42:36 -0500",
                                "awake": 0.44166666666666671,
                                "asleep": 0,
                                "core": 5.2416666666666671,
                            },
                            {
                                "date": "2025-01-16 00:00:00 -0500",
                                "deep": 0.67499999999999993,
                                "sleepStart": "2025-01-15 23:10:51 -0500",
                                "sleepEnd": "2025-01-16 04:35:21 -0500",
                                "core": 3.2250000000000005,
                                "awake": 0.23333333333333331,
                                "totalSleep": 5.1750000000000007,
                                "rem": 1.2749999999999999,
                                "asleep": 0,
                                "source": "John's Apple Watch",
                            },
                        ],
                    }
                ]
            }
        }

    def test_parse_sleep_data(self, parser, sleep_json_data):
        """Test parsing sleep data from JSON."""
        from big_mood_detector.domain.entities.sleep_record import SleepState

        # ACT
        records = parser.parse(sleep_json_data)

        # ASSERT
        assert len(records) == 2

        # Check first record
        first = records[0]
        assert first.source_name == "John's Apple Watch"
        assert first.state == SleepState.ASLEEP
        assert first.start_date.date() == date(2025, 1, 14)
        assert first.end_date.date() == date(2025, 1, 15)

        # Sleep phases are not stored in the entity itself
        # They would be calculated by the aggregator
        assert first.duration_hours == pytest.approx(7.22, rel=0.1)  # ~7h 13m

    def test_parse_from_file(self, parser, tmp_path):
        """Test parsing from file path."""
        # ARRANGE
        test_file = tmp_path / "sleep.json"
        test_data = {
            "data": {
                "metrics": [
                    {
                        "name": "sleep_analysis",
                        "data": [
                            {
                                "totalSleep": 8.0,
                                "source": "Apple Watch",
                                "sleepStart": "2025-01-01 23:00:00 -0500",
                                "date": "2025-01-02 00:00:00 -0500",
                                "sleepEnd": "2025-01-02 07:00:00 -0500",
                                "rem": 2.0,
                                "deep": 1.5,
                                "core": 4.5,
                                "awake": 0.5,
                            }
                        ],
                    }
                ]
            }
        }
        test_file.write_text(json.dumps(test_data))

        # ACT
        records = parser.parse_file(str(test_file))

        # ASSERT
        assert len(records) == 1
        assert records[0].duration_hours == 8.0

class TestHeartRateJSONParser:
    """Test cases for Heart Rate JSON parser."""

    @pytest.fixture
    def parser(self):
        return HeartRateJSONParser()

    @pytest.fixture
    def heart_rate_json_data(self):
        """Sample heart rate data in Apple Health export JSON format."""
        return {
            "data": {
                "metrics": [
                    {
                        "name": "heart_rate",
                        "data": [
                            {
                                "Max": 120,
                                "source": "John's Apple Watch",
                                "date": "2025-01-15 00:00:00 -0500",
                                "Min": 66,
                                "Avg": 91.33171558446449,
                            },
                            {
                                "source": "John's Apple Watch",
                                "Avg": 87.274991399167192,
                                "Min": 53,
                                "Max": 131,
                                "date": "2025-01-16 00:00:00 -0500",
                            },
                        ],
                    }
                ]
            }
        }

    def test_parse_heart_rate_data(self, parser, heart_rate_json_data):
        """Test parsing heart rate data from JSON."""
        from big_mood_detector.domain.entities.heart_rate_record import HeartMetricType

        # ACT
        records = parser.parse(heart_rate_json_data)

        # ASSERT
        assert len(records) == 2

        # Check first day
        first = records[0]
        assert first.source_name == "John's Apple Watch"
        assert first.metric_type == HeartMetricType.HEART_RATE
        assert first.timestamp.date() == date(2025, 1, 15)
        assert first.value == pytest.approx(91.33, rel=0.01)  # Average

        # Min/max would be stored separately or in aggregated data
        # For now we just check the average value

    def test_parse_resting_heart_rate(self, parser):
        """Test parsing resting heart rate data."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            MotionContext,
        )

        # ARRANGE
        resting_data = {
            "data": {
                "metrics": [
                    {
                        "name": "resting_heart_rate",
                        "data": [
                            {
                                "value": 58,
                                "source": "John's Apple Watch",
                                "date": "2025-01-15 00:00:00 -0500",
                            }
                        ],
                    }
                ]
            }
        }

        # ACT
        records = parser.parse_resting_heart_rate(resting_data)

        # ASSERT
        assert len(records) == 1
        assert records[0].value == 58
        assert records[0].metric_type == HeartMetricType.HEART_RATE
        assert records[0].motion_context == MotionContext.SEDENTARY

class TestActivityJSONParser:
    """Test cases for Activity JSON parser."""

    @pytest.fixture
    def parser(self):
        return ActivityJSONParser()

    @pytest.fixture
    def step_count_json_data(self):
        """Sample step count data in Apple Health export JSON format."""
        return {
            "data": {
                "metrics": [
                    {
                        "name": "step_count",
                        "data": [
                            {
                                "date": "2025-01-15 00:00:00 -0500",
                                "source": "John's Apple Watch|John's iPhone",
                                "qty": 7582,
                            },
                            {
                                "date": "2025-01-16 00:00:00 -0500",
                                "qty": 17402,
                                "source": "John's Apple Watch|John's iPhone",
                            },
                        ],
                    }
                ]
            }
        }

    def test_parse_step_count_data(self, parser, step_count_json_data):
        """Test parsing step count data from JSON."""
        from big_mood_detector.domain.entities.activity_record import ActivityType

        # ACT
        records = parser.parse(step_count_json_data)

        # ASSERT
        assert len(records) == 2

        # Check first record
        first = records[0]
        assert first.activity_type == ActivityType.STEP_COUNT
        assert first.value == 7582
        assert first.start_date.date() == date(2025, 1, 15)
        assert "Apple Watch" in first.source_name

    def test_parse_distance_data(self, parser):
        """Test parsing walking/running distance data."""
        from big_mood_detector.domain.entities.activity_record import ActivityType

        # ARRANGE
        distance_data = {
            "data": {
                "metrics": [
                    {
                        "name": "walking_running_distance",
                        "data": [
                            {
                                "date": "2025-01-15 00:00:00 -0500",
                                "source": "John's iPhone",
                                "qty": 5.2,  # kilometers
                            }
                        ],
                    }
                ]
            }
        }

        # ACT
        records = parser.parse_distance(distance_data)

        # ASSERT
        assert len(records) == 1
        assert records[0].activity_type == ActivityType.DISTANCE_WALKING
        assert records[0].value == 5.2
        assert records[0].unit == "km"

    def test_parse_multiple_sources(self, parser):
        """Test handling multiple data sources."""
        # ARRANGE
        multi_source_data = {
            "data": {
                "metrics": [
                    {
                        "data": [
                            {
                                "date": "2025-01-15 00:00:00 -0500",
                                "source": "Apple Watch|iPhone|iPad",
                                "qty": 10000,
                            }
                        ],
                    }
                ]
            }
        }

        # ACT
        records = parser.parse(multi_source_data)

        # ASSERT
        assert len(records) == 1
        assert "Apple Watch" in records[0].source_name
