"""
Unit tests for Sleep Window Analyzer

Tests the merging of sleep episodes within 3.75-hour windows,
critical for accurate bipolar disorder detection.
"""

from datetime import datetime, timedelta
import pytest

from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.sleep_window_analyzer import (
    SleepWindowAnalyzer,
    SleepWindow,
    WindowAnalysisResult
)


class TestSleepWindowAnalyzer:
    """Test sleep window analysis following clinical guidelines."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default 3.75-hour threshold."""
        return SleepWindowAnalyzer()
    
    def test_single_sleep_episode_creates_one_window(self, analyzer):
        """Single sleep episode should create one window."""
        # Arrange
        sleep_record = SleepRecord(
            start_time=datetime(2024, 1, 1, 23, 0),
            end_time=datetime(2024, 1, 2, 7, 0),
            source="test"
        )
        
        # Act
        windows = analyzer.analyze_sleep_episodes([sleep_record])
        
        # Assert
        assert len(windows) == 1
        assert windows[0].start_time == sleep_record.start_time
        assert windows[0].end_time == sleep_record.end_time
        assert windows[0].total_duration_hours == 8.0
        assert windows[0].episode_count == 1
    
    def test_close_episodes_merge_into_single_window(self, analyzer):
        """Episodes within 3.75 hours should merge."""
        # Arrange - two naps 2 hours apart
        episode1 = SleepRecord(
            start_time=datetime(2024, 1, 1, 14, 0),
            end_time=datetime(2024, 1, 1, 15, 30),  # 1.5h nap
            source="test"
        )
        episode2 = SleepRecord(
            start_time=datetime(2024, 1, 1, 17, 30),  # 2h gap
            end_time=datetime(2024, 1, 1, 18, 30),  # 1h nap
            source="test"
        )
        
        # Act
        windows = analyzer.analyze_sleep_episodes([episode1, episode2])
        
        # Assert
        assert len(windows) == 1
        window = windows[0]
        assert window.start_time == episode1.start_time
        assert window.end_time == episode2.end_time
        assert window.total_duration_hours == 2.5  # 1.5 + 1 hours
        assert window.episode_count == 2
        assert window.gap_hours == [2.0]  # One gap of 2 hours
    
    def test_distant_episodes_create_separate_windows(self, analyzer):
        """Episodes > 3.75 hours apart should not merge."""
        # Arrange - night sleep and afternoon nap
        night_sleep = SleepRecord(
            start_time=datetime(2024, 1, 1, 23, 0),
            end_time=datetime(2024, 1, 2, 7, 0),
            source="test"
        )
        afternoon_nap = SleepRecord(
            start_time=datetime(2024, 1, 2, 14, 0),  # 7h gap
            end_time=datetime(2024, 1, 2, 15, 0),
            source="test"
        )
        
        # Act
        windows = analyzer.analyze_sleep_episodes([night_sleep, afternoon_nap])
        
        # Assert
        assert len(windows) == 2
        assert windows[0].total_duration_hours == 8.0
        assert windows[1].total_duration_hours == 1.0
    
    def test_exactly_threshold_gap_merges(self, analyzer):
        """Episodes exactly 3.75 hours apart should merge."""
        # Arrange
        episode1 = SleepRecord(
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 11, 0),
            source="test"
        )
        episode2 = SleepRecord(
            start_time=datetime(2024, 1, 1, 14, 45),  # Exactly 3.75h gap
            end_time=datetime(2024, 1, 1, 15, 45),
            source="test"
        )
        
        # Act
        windows = analyzer.analyze_sleep_episodes([episode1, episode2])
        
        # Assert
        assert len(windows) == 1
        assert windows[0].episode_count == 2
    
    def test_multiple_episodes_complex_merging(self, analyzer):
        """Complex pattern of multiple episodes."""
        # Arrange - simulating fragmented sleep pattern
        episodes = [
            # First window: fragmented night sleep
            SleepRecord(
                start_time=datetime(2024, 1, 1, 22, 0),
                end_time=datetime(2024, 1, 1, 23, 30),  # 1.5h
                source="test"
            ),
            SleepRecord(
                start_time=datetime(2024, 1, 2, 0, 30),   # 1h gap
                end_time=datetime(2024, 1, 2, 3, 0),      # 2.5h
                source="test"
            ),
            SleepRecord(
                start_time=datetime(2024, 1, 2, 4, 0),    # 1h gap
                end_time=datetime(2024, 1, 2, 6, 30),     # 2.5h
                source="test"
            ),
            # Second window: afternoon nap (> 3.75h from morning)
            SleepRecord(
                start_time=datetime(2024, 1, 2, 14, 0),   # 7.5h gap
                end_time=datetime(2024, 1, 2, 15, 30),    # 1.5h
                source="test"
            ),
        ]
        
        # Act
        windows = analyzer.analyze_sleep_episodes(episodes)
        
        # Assert
        assert len(windows) == 2
        
        # First window (fragmented night)
        night_window = windows[0]
        assert night_window.episode_count == 3
        assert night_window.total_duration_hours == 6.5  # 1.5 + 2.5 + 2.5
        assert len(night_window.gap_hours) == 2
        assert night_window.gap_hours == [1.0, 1.0]
        
        # Second window (afternoon nap)
        nap_window = windows[1]
        assert nap_window.episode_count == 1
        assert nap_window.total_duration_hours == 1.5
    
    def test_window_analysis_result(self, analyzer):
        """Test comprehensive analysis results."""
        # Arrange
        episodes = [
            # Short fragmented night (< 6h total)
            SleepRecord(
                start_time=datetime(2024, 1, 1, 23, 0),
                end_time=datetime(2024, 1, 2, 1, 0),  # 2h
                source="test"
            ),
            SleepRecord(
                start_time=datetime(2024, 1, 2, 2, 0),   # 1h gap
                end_time=datetime(2024, 1, 2, 4, 30),    # 2.5h
                source="test"
            ),
            # Long night (> 10h window)
            SleepRecord(
                start_time=datetime(2024, 1, 2, 22, 0),
                end_time=datetime(2024, 1, 3, 8, 30),    # 10.5h
                source="test"
            ),
        ]
        
        # Act
        result = analyzer.get_analysis_summary(episodes, days=2)
        
        # Assert
        assert result.total_windows == 2
        assert result.short_windows == 1  # First night < 6h
        assert result.long_windows == 1   # Second night > 10h
        assert result.fragmented_windows == 1  # First night has 2 episodes
        assert result.short_window_percentage == 50.0
        assert result.long_window_percentage == 50.0
        assert result.average_window_duration_hours == pytest.approx(7.5, 0.1)
    
    def test_empty_episodes_handling(self, analyzer):
        """Handle empty episode list gracefully."""
        # Act
        windows = analyzer.analyze_sleep_episodes([])
        result = analyzer.get_analysis_summary([], days=7)
        
        # Assert
        assert windows == []
        assert result.total_windows == 0
        assert result.short_window_percentage == 0.0
        assert result.long_window_percentage == 0.0
    
    def test_custom_threshold(self):
        """Test analyzer with custom merge threshold."""
        # Arrange - 2 hour threshold
        analyzer = SleepWindowAnalyzer(merge_threshold_hours=2.0)
        
        episodes = [
            SleepRecord(
                start_time=datetime(2024, 1, 1, 10, 0),
                end_time=datetime(2024, 1, 1, 11, 0),
                source="test"
            ),
            SleepRecord(
                start_time=datetime(2024, 1, 1, 13, 30),  # 2.5h gap
                end_time=datetime(2024, 1, 1, 14, 30),
                source="test"
            ),
        ]
        
        # Act
        windows = analyzer.analyze_sleep_episodes(episodes)
        
        # Assert - should NOT merge with 2h threshold
        assert len(windows) == 2
    
    def test_overlapping_episodes_handling(self, analyzer):
        """Handle overlapping sleep episodes (data quality issue)."""
        # Arrange - overlapping episodes
        episodes = [
            SleepRecord(
                start_time=datetime(2024, 1, 1, 22, 0),
                end_time=datetime(2024, 1, 2, 2, 0),
                source="test"
            ),
            SleepRecord(
                start_time=datetime(2024, 1, 2, 1, 0),  # Overlaps!
                end_time=datetime(2024, 1, 2, 3, 0),
                source="test"
            ),
        ]
        
        # Act
        windows = analyzer.analyze_sleep_episodes(episodes)
        
        # Assert - should merge and handle overlap
        assert len(windows) == 1
        assert windows[0].start_time == datetime(2024, 1, 1, 22, 0)
        assert windows[0].end_time == datetime(2024, 1, 2, 3, 0)
        assert windows[0].episode_count == 2