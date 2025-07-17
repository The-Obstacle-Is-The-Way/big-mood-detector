"""
Label entity for categorizing mood and health states.

Labels represent clinical and behavioral categories that can be applied
to analyzed health data (e.g., "Depression", "Mania", "Sleep Disruption").
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class Label:
    """
    Represents a label for categorizing mood and health states.
    
    Attributes:
        id: Unique identifier for the label
        name: Human-readable name (e.g., "Depression", "Mania")
        description: Detailed description of what this label represents
        color: Hex color code for UI representation
        category: Category grouping (e.g., "mood", "sleep", "activity")
        metadata: Additional metadata (e.g., DSM-5 codes, thresholds)
    """
    
    id: str
    name: str
    description: str
    color: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate label data after initialization."""
        if not self.name:
            raise ValueError("Label name cannot be empty")
        
        if not self.description:
            raise ValueError("Label description cannot be empty")
        
        if not self.category:
            raise ValueError("Label category cannot be empty")
        
        # Validate color format (basic hex validation)
        if not self.color or not self.color.startswith("#"):
            raise ValueError("Label color must be in hex format (e.g., #FF0000)")
        
        # Validate hex color length
        if len(self.color) not in [4, 7]:  # #RGB or #RRGGBB
            raise ValueError("Invalid hex color format")