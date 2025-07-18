"""
Label repository interface.

Defines the contract for label persistence following the repository pattern.
"""

from abc import ABC, abstractmethod

from big_mood_detector.domain.entities.label import Label


class LabelRepository(ABC):
    """Abstract interface for label persistence."""

    @abstractmethod
    def save(self, label: Label) -> None:
        """Save or update a label."""
        pass

    @abstractmethod
    def find_by_id(self, label_id: str) -> Label | None:
        """Find a label by its ID."""
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> list[Label]:
        """Find labels by name (should be unique, but returns list for safety)."""
        pass

    @abstractmethod
    def find_by_category(self, category: str) -> list[Label]:
        """Find all labels in a category."""
        pass

    @abstractmethod
    def find_all(self) -> list[Label]:
        """Get all labels."""
        pass

    @abstractmethod
    def delete(self, label_id: str) -> None:
        """Delete a label by ID."""
        pass
