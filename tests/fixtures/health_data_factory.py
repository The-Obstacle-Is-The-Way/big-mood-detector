"""
Factory classes for generating test health data.

Uses factory-boy pattern for creating realistic test data that mimics clinical scenarios.
"""

import random
from datetime import datetime, timedelta

import factory
import factory.fuzzy


class HealthKitDataFactory(factory.Factory):
    """Factory for generating realistic Apple HealthKit JSON data."""

    class Meta:
        model = dict

    @factory.lazy_attribute
    def data(self):
        """Generate comprehensive HealthKit data structure."""
        start_date = datetime.now() - timedelta(days=30)

        return {
            "data": {
                "metrics": {
                    "step_count": self._generate_step_data(start_date),
                    "heart_rate": self._generate_heart_rate_data(start_date),
                    "sleep_analysis": self._generate_sleep_data(start_date),
                    "active_energy": self._generate_active_energy_data(start_date),
                }
            }
        }

    def _generate_step_data(self, start_date: datetime) -> list[dict]:
        """Generate realistic step count data with bipolar patterns."""
        data = []
        for i in range(30):  # 30 days of data
            date = start_date + timedelta(days=i)

            # Simulate mood state patterns
            if i < 7:  # Stable period
                base_steps = 8000
                variance = 2000
            elif i < 14:  # Manic period (higher activity)
                base_steps = 15000
                variance = 3000
            elif i < 24:  # Depressive period (lower activity)
                base_steps = 3000
                variance = 1000
            else:  # Recovery period
                base_steps = 7000
                variance = 1500

            steps = max(0, int(random.gauss(base_steps, variance)))
            data.append({"date": date.strftime("%Y-%m-%d"), "value": steps})

        return data

    def _generate_heart_rate_data(self, start_date: datetime) -> list[dict]:
        """Generate heart rate data throughout the day."""
        data = []
        for day in range(30):
            date = start_date + timedelta(days=day)

            # Generate multiple readings per day
            for hour in [8, 12, 16, 20]:  # 4 readings per day
                timestamp = date.replace(hour=hour)

                # Mood-based heart rate patterns
                if day < 7:  # Stable
                    base_hr = 70
                elif day < 14:  # Manic (elevated)
                    base_hr = 85
                elif day < 24:  # Depressive (lower)
                    base_hr = 62
                else:  # Recovery
                    base_hr = 72

                hr = max(50, int(random.gauss(base_hr, 8)))
                data.append({"timestamp": timestamp.isoformat() + "Z", "value": hr})

        return data

    def _generate_sleep_data(self, start_date: datetime) -> list[dict]:
        """Generate sleep analysis data."""
        data = []
        for day in range(30):
            date = start_date + timedelta(days=day)

            # Mood-based sleep patterns
            if day < 7:  # Stable
                sleep_duration = 7.5
                bedtime_hour = 23
            elif day < 14:  # Manic (less sleep)
                sleep_duration = 4.5
                bedtime_hour = 1
            elif day < 24:  # Depressive (more sleep, irregular)
                sleep_duration = 10.0
                bedtime_hour = 21
            else:  # Recovery
                sleep_duration = 8.0
                bedtime_hour = 22

            bedtime = date.replace(hour=bedtime_hour, minute=0)
            wake_time = bedtime + timedelta(hours=sleep_duration)

            data.append(
                {
                    "start_date": bedtime.isoformat() + "Z",
                    "end_date": wake_time.isoformat() + "Z",
                    "value": "InBed",
                }
            )

        return data

    def _generate_active_energy_data(self, start_date: datetime) -> list[dict]:
        """Generate active energy burned data."""
        data = []
        for day in range(30):
            date = start_date + timedelta(days=day)

            # Correlate with step count patterns
            if day < 7:  # Stable
                energy = 400
            elif day < 14:  # Manic
                energy = 800
            elif day < 24:  # Depressive
                energy = 150
            else:  # Recovery
                energy = 350

            data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "value": max(0, int(random.gauss(energy, 100))),
                }
            )

        return data


class ClinicalDatasetFactory(factory.Factory):
    """Factory for generating clinical validation datasets."""

    class Meta:
        model = dict

    mood_episode = factory.fuzzy.FuzzyChoice(
        ["stable", "manic", "hypomanic", "depressive", "mixed"]
    )

    @factory.lazy_attribute
    def patient_data(self):
        """Generate patient data with known clinical outcomes."""
        return {
            "patient_id": factory.Faker("uuid4"),
            "age": factory.fuzzy.FuzzyInteger(18, 65).fuzz(),
            "diagnosis": "Bipolar I Disorder",
            "medication_adherence": factory.fuzzy.FuzzyFloat(0.6, 1.0).fuzz(),
            "mood_episode": self.mood_episode,
            "clinical_severity": self._get_severity_score(),
        }

    def _get_severity_score(self) -> float:
        """Generate clinical severity score based on mood episode."""
        severity_map = {
            "stable": 0.0,
            "hypomanic": 0.3,
            "manic": 0.8,
            "depressive": 0.7,
            "mixed": 0.9,
        }
        return severity_map.get(self.mood_episode, 0.0)


class PerformanceTestDataFactory(factory.Factory):
    """Factory for generating large-scale performance test data."""

    class Meta:
        model = dict

    @factory.lazy_attribute
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        return {
            "patients": [HealthKitDataFactory().data for _ in range(100)],
            "time_range": "365_days",
            "total_data_points": 100
            * 365
            * 24,  # 100 patients, 365 days, 24 readings/day
        }
