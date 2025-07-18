"""
Label service for managing mood and health labels.

This service orchestrates label operations following Clean Architecture principles.
"""

from uuid import uuid4

from big_mood_detector.domain.entities.label import Label
from big_mood_detector.domain.repositories.label_repository import LabelRepository


class LabelService:
    """Service for managing labels."""

    def __init__(self, label_repository: LabelRepository):
        """Initialize the label service with a repository."""
        self.repository = label_repository

    def create_label(
        self,
        name: str,
        description: str,
        category: str,
        color: str,
        metadata: dict | None = None,
    ) -> Label:
        """
        Create a new label.

        Args:
            name: Label name (must be unique)
            description: Label description
            category: Label category (e.g., mood, sleep, activity)
            color: Label color in hex format
            metadata: Optional metadata dictionary

        Returns:
            Created label

        Raises:
            ValueError: If label with same name already exists
        """
        # Check if label with same name exists
        existing_labels = self.repository.find_by_name(name)
        if existing_labels:
            raise ValueError(f"Label with name '{name}' already exists")

        # Create new label
        label = Label(
            id=f"label-{uuid4().hex[:8]}",
            name=name,
            description=description,
            category=category,
            color=color,
            metadata=metadata or {},
        )

        # Save to repository
        self.repository.save(label)

        return label

    def get_label(self, label_id: str) -> Label | None:
        """Get a label by ID."""
        return self.repository.find_by_id(label_id)

    def list_labels(self, category: str | None = None) -> list[Label]:
        """
        List all labels, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of labels
        """
        if category:
            return self.repository.find_by_category(category)
        return self.repository.find_all()

    def search_labels(self, query: str, limit: int = 10) -> list[Label]:
        """
        Search labels by name or description.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching labels
        """
        all_labels = self.repository.find_all()

        # Simple search implementation
        # In production, this would use a proper search index
        query_lower = query.lower()
        matches = []

        for label in all_labels:
            if (
                query_lower in label.name.lower()
                or query_lower in label.description.lower()
            ):
                matches.append(label)

                if len(matches) >= limit:
                    break

        return matches

    def update_label(
        self,
        label_id: str,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        color: str | None = None,
        metadata: dict | None = None,
    ) -> Label:
        """
        Update an existing label.

        Args:
            label_id: ID of label to update
            name: New name (optional)
            description: New description (optional)
            category: New category (optional)
            color: New color (optional)
            metadata: New metadata (optional)

        Returns:
            Updated label

        Raises:
            ValueError: If label not found
        """
        label = self.repository.find_by_id(label_id)
        if not label:
            raise ValueError(f"Label with ID '{label_id}' not found")

        # Check name uniqueness if changing name
        if name and name != label.name:
            existing = self.repository.find_by_name(name)
            if existing:
                raise ValueError(f"Label with name '{name}' already exists")

        # Create updated label
        updated_label = Label(
            id=label.id,
            name=name or label.name,
            description=description or label.description,
            category=category or label.category,
            color=color or label.color,
            metadata=metadata if metadata is not None else label.metadata,
        )

        # Save updated label
        self.repository.save(updated_label)

        return updated_label

    def delete_label(self, label_id: str) -> bool:
        """
        Delete a label.

        Args:
            label_id: ID of label to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If label is in use (future implementation)
        """
        label = self.repository.find_by_id(label_id)
        if not label:
            return False

        # TODO: Check if label is in use by any records
        # This would involve checking with other repositories

        self.repository.delete(label_id)
        return True
