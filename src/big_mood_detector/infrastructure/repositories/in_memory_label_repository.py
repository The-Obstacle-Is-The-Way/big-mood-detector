"""
In-memory implementation of the label repository.

Useful for testing and development. In production, this would be
replaced with a database-backed implementation.
"""


from big_mood_detector.domain.entities.label import Label
from big_mood_detector.domain.repositories.label_repository import LabelRepository


class InMemoryLabelRepository(LabelRepository):
    """In-memory implementation of label repository."""

    def __init__(self) -> None:
        """Initialize with empty storage and some default labels."""
        self._labels: dict[str, Label] = {}
        self._initialize_default_labels()

    def _initialize_default_labels(self) -> None:
        """Add some default clinical labels."""
        default_labels = [
            Label(
                id="label-depression",
                name="Depression",
                description="Major depressive episode indicators based on DSM-5 criteria",
                color="#5B6C8F",
                category="mood",
                metadata={"dsm5_code": "296.2x", "min_duration_days": 14},
            ),
            Label(
                id="label-mania",
                name="Mania",
                description="Manic episode indicators with elevated mood and energy",
                color="#FF6B6B",
                category="mood",
                metadata={"dsm5_code": "296.4x", "min_duration_days": 7},
            ),
            Label(
                id="label-hypomania",
                name="Hypomania",
                description="Hypomanic episode with less severe mood elevation",
                color="#FFA500",
                category="mood",
                metadata={"dsm5_code": "296.8x", "min_duration_days": 4},
            ),
            Label(
                id="label-sleep-disruption",
                name="Sleep Disruption",
                description="Significant disruption in sleep patterns",
                color="#4ECDC4",
                category="sleep",
                metadata={"threshold_hours": 3.5, "fragmentation_index": 0.3},
            ),
            Label(
                id="label-circadian-shift",
                name="Circadian Rhythm Shift",
                description="Shift in sleep-wake cycle timing",
                color="#95E1D3",
                category="sleep",
                metadata={"shift_threshold_hours": 2},
            ),
            Label(
                id="label-low-activity",
                name="Low Activity",
                description="Significantly reduced physical activity levels",
                color="#DDA0DD",
                category="activity",
                metadata={"percentile_threshold": 20},
            ),
            Label(
                id="label-high-activity",
                name="High Activity",
                description="Elevated physical activity levels",
                color="#98D8C8",
                category="activity",
                metadata={"percentile_threshold": 80},
            ),
        ]

        for label in default_labels:
            self._labels[label.id] = label

    def save(self, label: Label) -> None:
        """Save or update a label."""
        self._labels[label.id] = label

    def find_by_id(self, label_id: str) -> Label | None:
        """Find a label by its ID."""
        return self._labels.get(label_id)

    def find_by_name(self, name: str) -> list[Label]:
        """Find labels by name."""
        return [
            label
            for label in self._labels.values()
            if label.name.lower() == name.lower()
        ]

    def find_by_category(self, category: str) -> list[Label]:
        """Find all labels in a category."""
        return [
            label
            for label in self._labels.values()
            if label.category.lower() == category.lower()
        ]

    def find_all(self) -> list[Label]:
        """Get all labels."""
        return list(self._labels.values())

    def delete(self, label_id: str) -> None:
        """Delete a label by ID."""
        if label_id in self._labels:
            del self._labels[label_id]
