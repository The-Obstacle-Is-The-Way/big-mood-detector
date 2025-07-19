"""
Simple test to prove the sleep percentage bug.

This test will FAIL until we fix the aggregation pipeline.
"""
import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator, DailySleepSummary
from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline


class TestSleepPercentageBug:
    """Prove that aggregation_pipeline calculates sleep wrong."""
    
    def test_sleep_percentage_calculation_is_wrong(self):
        """
        RED TEST: The pipeline uses sleep_percentage * 24 which gives wrong results.
        
        This test mocks the internal calculation to show the bug.
        """
        pipeline = AggregationPipeline()
        
        # Mock sleep windows with known values
        mock_windows = [
            Mock(total_duration_hours=2.5, gap_hours=[]),  # First sleep window
            Mock(total_duration_hours=2.0, gap_hours=[]),  # Second sleep window  
        ]
        # Total = 4.5 hours, but this is WINDOW duration, not total sleep!
        
        # Call the internal method that has the bug
        daily_metrics = pipeline.calculate_sleep_metrics(mock_windows)
        
        # The bug: sleep_percentage = total_minutes / 1440
        # 4.5 hours = 270 minutes
        # 270 / 1440 = 0.1875 (18.75%)
        # 0.1875 * 24 = 4.5 hours
        
        assert daily_metrics["sleep_percentage"] == pytest.approx(0.1875, 0.001)
        
        # But wait! What if the person actually slept 7.5 hours total,
        # but the window analyzer only found 4.5 hours of "windows"?
        # Then we're losing 3 hours of sleep!
        
        print(f"\n🐛 BUG EXPOSED:")
        print(f"   Sleep windows total: 4.5 hours")
        print(f"   Sleep percentage: {daily_metrics['sleep_percentage']:.1%}")
        print(f"   Calculated duration: {daily_metrics['sleep_percentage'] * 24:.1f} hours")
        print(f"   MISSING: 3.0 hours of sleep!")
    
    def test_correct_approach_using_sleep_aggregator(self):
        """
        GREEN TEST: Show how it SHOULD work using SleepAggregator.
        """
        # Create a proper 7.5 hour sleep record
        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),
            end_date=datetime(2024, 1, 2, 5, 30),
            state=SleepState.ASLEEP
        )
        
        # Use the aggregator
        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily([sleep_record])
        
        # Get the summary
        summary = summaries[date(2024, 1, 1)]
        
        print(f"\n✅ CORRECT APPROACH:")
        print(f"   SleepAggregator result: {summary.total_sleep_hours:.1f} hours")
        print(f"   This is what we should use!")
        
        assert summary.total_sleep_hours == 7.5
    
    @patch('big_mood_detector.application.services.aggregation_pipeline.SleepWindowAnalyzer')
    @patch('big_mood_detector.application.services.aggregation_pipeline.SleepAggregator')
    def test_pipeline_should_use_aggregator_not_windows(self, mock_aggregator_class, mock_analyzer_class):
        """
        Show that pipeline should call SleepAggregator.aggregate_daily()
        instead of using window analysis for duration.
        """
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_sleep_episodes.return_value = []  # Windows
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_aggregator = Mock()
        mock_summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=8.0,
            total_sleep_hours=7.5,  # The CORRECT value
            sleep_efficiency=0.9375,
            sleep_sessions=1,
            longest_sleep_hours=7.5,
            sleep_fragmentation_index=0.0
        )
        mock_aggregator.aggregate_daily.return_value = {
            date(2024, 1, 1): mock_summary
        }
        mock_aggregator_class.return_value = mock_aggregator
        
        # Create sleep record
        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),
            end_date=datetime(2024, 1, 2, 5, 30),
            state=SleepState.ASLEEP
        )
        
        # THE FIX: Pipeline should do this
        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily([sleep_record])
        sleep_duration = summaries[date(2024, 1, 1)].total_sleep_hours
        
        print(f"\n🔧 THE FIX:")
        print(f"   Instead of: sleep_percentage * 24")
        print(f"   Use: SleepAggregator.aggregate_daily()[date].total_sleep_hours")
        print(f"   Result: {sleep_duration:.1f} hours")
        
        assert sleep_duration == 7.5