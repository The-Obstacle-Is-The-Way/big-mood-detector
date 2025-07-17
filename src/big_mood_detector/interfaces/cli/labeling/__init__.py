"""Labeling CLI module for ground truth collection."""

from ..label_group import unified_label_group
from .commands import label_group

__all__ = ["label_group", "unified_label_group"]
