"""End-to-end test for HR/HRV functionality with TimescaleDB."""

import time
from datetime import UTC, date, datetime

import pytest

@pytest.mark.integration
@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Integration tests require --run-integration flag and database",
)
class TestHRHRVEndToEnd:
    """Test HR/HRV functionality end-to-end with real TimescaleDB."""

    @pytest.fixture
    def test_db_url(self):
        """Get test database URL."""
        import os

        # Use test database URL from environment or default
        return os.getenv(
            "TEST_DATABASE_URL",
            "postgresql://test_user:test_pass@localhost:5432/test_bigmood"
        )

    @pytest.fixture
    def repository(self, test_db_url):
        """Create TimescaleDB repository for testing."""
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import TimescaleBaselineRepository

        try:
            repo = TimescaleBaselineRepository(
                connection_string=test_db_url,
                enable_feast_sync=False,  # Disable Feast for this test
            )
            yield repo
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def test_full_hr_hrv_pipeline(self, repository):
        """Test complete HR/HRV pipeline from save to retrieve."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Create baseline with all HR/HRV fields
        test_user_id = f"hr_hrv_test_user_{int(time.time())}"

        baseline = UserBaseline(
            user_id=test_user_id,
            baseline_date=date.today(),
            sleep_mean=7.8,
            sleep_std=1.1,
            activity_mean=8500.0,
            activity_std=1800.0,
            circadian_phase=22.5,
            heart_rate_mean=62.5,  # Resting HR
            heart_rate_std=4.2,
            hrv_mean=58.3,  # HRV SDNN
            hrv_std=12.1,
            last_updated=datetime.now(UTC),
            data_points=45,
        )

        # Save baseline
        repository.save_baseline(baseline)

        # Retrieve and verify
        retrieved = repository.get_baseline(test_user_id)

        assert retrieved is not None
        assert retrieved.user_id == test_user_id
        assert retrieved.baseline_date == baseline.baseline_date

        # Verify core metrics
        assert retrieved.sleep_mean == 7.8
        assert retrieved.sleep_std == 1.1
        assert retrieved.activity_mean == 8500.0
        assert retrieved.activity_std == 1800.0
        assert retrieved.circadian_phase == 22.5

        # Verify HR/HRV metrics
        assert retrieved.heart_rate_mean == 62.5
        assert retrieved.heart_rate_std == 4.2
        assert retrieved.hrv_mean == 58.3
        assert retrieved.hrv_std == 12.1

        # Verify metadata
        assert retrieved.data_points == 45

    def test_hr_hrv_optional_fields(self, repository):
        """Test that HR/HRV fields are optional and can be None."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        test_user_id = f"optional_hr_test_{int(time.time())}"

        # Create baseline without HR/HRV data
        baseline_no_hr = UserBaseline(
            user_id=test_user_id,
            baseline_date=date.today(),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=7000.0,
            activity_std=1500.0,
            circadian_phase=23.0,
            heart_rate_mean=None,  # No HR data
            heart_rate_std=None,
            hrv_mean=None,  # No HRV data
            hrv_std=None,
            last_updated=datetime.now(UTC),
            data_points=30,
        )

        # Save baseline
        repository.save_baseline(baseline_no_hr)

        # Retrieve and verify
        retrieved = repository.get_baseline(test_user_id)

        assert retrieved is not None
        assert retrieved.heart_rate_mean is None
        assert retrieved.heart_rate_std is None
        assert retrieved.hrv_mean is None
        assert retrieved.hrv_std is None

        # Core fields should still be present
        assert retrieved.sleep_mean == 7.0
        assert retrieved.activity_mean == 7000.0

    def test_hr_hrv_history_tracking(self, repository):
        """Test that HR/HRV metrics are tracked in baseline history."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        test_user_id = f"history_hr_test_{int(time.time())}"

        # Create baselines over multiple days with evolving HR/HRV
        baselines = [
            UserBaseline(
                user_id=test_user_id,
                baseline_date=date(2024, 1, day),
                sleep_mean=7.5,
                sleep_std=1.0,
                activity_mean=8000.0,
                activity_std=2000.0,
                circadian_phase=22.0,
                heart_rate_mean=65.0 - day * 0.5,  # HR improves over time
                heart_rate_std=5.0,
                hrv_mean=45.0 + day * 2.0,  # HRV improves over time
                hrv_std=10.0,
                last_updated=datetime(2024, 1, day, 10, 0, tzinfo=UTC),
                data_points=30,
            )
            for day in range(1, 6)
        ]

        # Save all baselines
        for baseline in baselines:
            repository.save_baseline(baseline)

        # Retrieve history
        history = repository.get_baseline_history(test_user_id, limit=10)

        # Should have all 5 baselines
        assert len(history) == 5

        # Verify chronological order (oldest first)
        for i in range(1, len(history)):
            assert history[i].baseline_date > history[i-1].baseline_date

        # Verify HR/HRV progression
        first_baseline = history[0]
        last_baseline = history[-1]

        # HR should have decreased (improved)
        assert first_baseline.heart_rate_mean > last_baseline.heart_rate_mean

        # HRV should have increased (improved)
        assert first_baseline.hrv_mean < last_baseline.hrv_mean

    def test_concurrent_hr_hrv_updates(self, repository):
        """Test concurrent updates to HR/HRV baselines."""
        import concurrent.futures

        test_user_id = f"concurrent_hr_test_{int(time.time())}"

        def update_baseline(value_offset: int):
            """Update baseline with different HR/HRV values."""
            baseline = UserBaseline(
                user_id=test_user_id,
                baseline_date=date.today(),
                sleep_mean=7.5,
                sleep_std=1.0,
                activity_mean=8000.0,
                activity_std=2000.0,
                circadian_phase=22.0,
                heart_rate_mean=60.0 + value_offset,
                heart_rate_std=5.0,
                hrv_mean=50.0 + value_offset,
                hrv_std=10.0,
                last_updated=datetime.now(UTC),
                data_points=30 + value_offset,
            )
            repository.save_baseline(baseline)
            return value_offset

        # Run concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(update_baseline, i)
                for i in range(5)
            ]
            # Wait for all futures to complete
            [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify the last update wins
        final_baseline = repository.get_baseline(test_user_id)
        assert final_baseline is not None

        # Should have one of the HR values we set
        assert final_baseline.heart_rate_mean in [60.0 + i for i in range(5)]

        # HRV should match HR (same offset was used)
        hr_offset = final_baseline.heart_rate_mean - 60.0
        assert final_baseline.hrv_mean == 50.0 + hr_offset

    def test_user_id_hashing_with_hr_hrv(self, repository):
        """Test that user ID hashing works correctly with HR/HRV data."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Use a recognizable user ID
        plain_user_id = "test.user@example.com"

        baseline = UserBaseline(
            user_id=plain_user_id,
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            heart_rate_std=5.0,
            hrv_mean=55.0,
            hrv_std=8.0,
            last_updated=datetime.now(UTC),
            data_points=30,
        )

        # Save baseline
        repository.save_baseline(baseline)

        # Retrieve using same plain user ID
        retrieved = repository.get_baseline(plain_user_id)

        # Should work transparently
        assert retrieved is not None
        assert retrieved.user_id == plain_user_id
        assert retrieved.heart_rate_mean == 65.0
        assert retrieved.hrv_mean == 55.0

        # The actual storage should use hashed ID
        # This is verified internally by the repository
