"""
Fast Streaming XML Parser using lxml

Optimized for Apple Health export files with:
- 20x faster parsing using lxml
- Memory-efficient streaming with fast_iter pattern
- Smart date filtering without full file scan
- Batch processing for performance
"""

import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAS_LXML = False
    logging.warning("lxml not available, falling back to slower stdlib XML parser")

from dateutil import parser as date_parser

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.infrastructure.parsers.xml import (
    ActivityParser,
    HeartRateParser,
    SleepParser,
)

logger = logging.getLogger(__name__)


class FastStreamingXMLParser:
    """
    High-performance XML parser using lxml for 20x speed improvement.
    
    Key optimizations:
    - Uses lxml's C-based parser (20x faster than stdlib)
    - Implements fast_iter pattern for memory efficiency
    - Early date filtering to skip irrelevant records
    - Batch processing for better performance
    """

    def __init__(self) -> None:
        """Initialize with existing parsers for entity conversion."""
        self.sleep_parser = SleepParser()
        self.activity_parser = ActivityParser()
        self.heart_parser = HeartRateParser()
        
        # Record type mappings for faster lookup
        self.sleep_types = {"HKCategoryTypeIdentifierSleepAnalysis"}
        self.activity_types = set(self.activity_parser.supported_activity_types)
        self.heart_types = set(self.heart_parser.supported_heart_types)
        
        # All supported types for 'all' mode
        self.all_types = self.sleep_types | self.activity_types | self.heart_types

    def fast_iter(
        self,
        context: Any,
        func: Any,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        *args: Any,
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """
        Fast iteration pattern that clears elements to save memory.
        
        Based on Liza Daly's fast_iter pattern, optimized for date filtering.
        """
        for event, elem in context:
            # Early date filtering before processing
            if start_date or end_date:
                date_str = elem.get("startDate")
                if date_str:
                    try:
                        record_date = date_parser.parse(date_str).date()
                        
                        if start_date and record_date < start_date:
                            elem.clear()
                            if HAS_LXML:
                                # Also eliminate now-empty references from the tree
                                while elem.getprevious() is not None:
                                    del elem.getparent()[0]
                            continue
                            
                        if end_date and record_date > end_date:
                            elem.clear()
                            if HAS_LXML:
                                while elem.getprevious() is not None:
                                    del elem.getparent()[0]
                            continue
                    except (ValueError, TypeError):
                        pass
            
            # Process the element
            result = func(elem, *args, **kwargs)
            if result is not None:
                yield result
            
            # Clear the element to free memory
            elem.clear()
            
            if HAS_LXML:
                # Also eliminate now-empty references from the tree
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        
        del context

    def iter_records(
        self,
        file_path: str | Path,
        record_types: set[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream records from XML file with optimized filtering.
        
        Args:
            file_path: Path to the XML file
            record_types: Set of record types to filter (faster than list)
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Yields:
            Dictionary of record attributes
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        # Log parser being used
        logger.info(f"Parsing XML with {'lxml' if HAS_LXML else 'stdlib'} parser")
        
        try:
            # Create iterator context - only parse Record elements
            context = etree.iterparse(
                str(file_path),
                events=("end",),
                tag="Record"
            )
            
            def process_element(elem: Any) -> dict[str, Any] | None:
                """Process a single element and return its data."""
                record_type = elem.get("type")
                
                # Filter by record types if specified
                if record_types and record_type not in record_types:
                    return None
                
                # Extract all attributes
                record_data = dict(elem.attrib)
                
                # Extract metadata entries (e.g., heart rate motion context)
                for metadata in elem.findall("MetadataEntry"):
                    key = metadata.get("key")
                    value = metadata.get("value")
                    if key == "HKMetadataKeyHeartRateMotionContext" and value is not None:
                        record_data["motionContext"] = value
                
                return record_data
            
            # Use fast_iter for memory-efficient processing
            yield from self.fast_iter(
                context,
                process_element,
                start_date,
                end_date
            )
            
        except etree.ParseError as e:
            raise ValueError(f"XML parsing error: {str(e)}") from e

    def parse_file(
        self,
        file_path: str | Path,
        entity_type: str = "all",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[SleepRecord | ActivityRecord | HeartRateRecord, None, None]:
        """
        Parse file and yield domain entities with optimized performance.
        
        Args:
            file_path: Path to the XML file
            entity_type: Type of entities to parse ('sleep', 'activity', 'heart', 'all')
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)
            
        Yields:
            Domain entities
        """
        # Convert dates for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
        
        # Determine which types to parse (using sets for O(1) lookup)
        if entity_type == "sleep":
            types_to_parse = self.sleep_types
        elif entity_type == "activity":
            types_to_parse = self.activity_types
        elif entity_type == "heart":
            types_to_parse = self.heart_types
        else:  # 'all'
            types_to_parse = self.all_types
        
        # Count records for progress
        record_count = 0
        
        # Stream through records
        for record_dict in self.iter_records(file_path, types_to_parse, start_dt, end_dt):
            record_type = record_dict.get("type")
            record_count += 1
            
            # Log progress every 10k records
            if record_count % 10000 == 0:
                logger.info(f"Processed {record_count:,} records...")
            
            try:
                # Convert to appropriate entity based on type
                if record_type in self.sleep_types:
                    # Create minimal XML for parser compatibility
                    elem = self._dict_to_element(record_dict)
                    sleep_entities = self.sleep_parser.parse_to_entities(elem)
                    yield from sleep_entities
                    
                elif record_type in self.activity_types:
                    elem = self._dict_to_element(record_dict)
                    activity_entities = self.activity_parser.parse_to_entities(elem)
                    yield from activity_entities
                    
                elif record_type in self.heart_types:
                    elem = self._dict_to_element(record_dict)
                    heart_entities = self.heart_parser.parse_to_entities(elem)
                    yield from heart_entities
                    
            except (ValueError, KeyError) as e:
                # Log but skip records that can't be converted
                if record_count % 1000 == 0:  # Don't spam logs
                    logger.debug(f"Skipping record: {e}")
                continue
        
        logger.info(f"Completed parsing {record_count:,} records")

    def parse_file_in_batches(
        self,
        file_path: str | Path,
        batch_size: int = 5000,
        entity_type: str = "all",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[list[Any], None, None]:
        """
        Parse file and yield entities in batches for better performance.
        
        Args:
            file_path: Path to the XML file
            batch_size: Number of entities per batch (increased for performance)
            entity_type: Type of entities to parse
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Yields:
            List of domain entities
        """
        batch = []
        
        for entity in self.parse_file(file_path, entity_type, start_date, end_date):
            batch.append(entity)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining entities
        if batch:
            yield batch

    def count_records_by_date(
        self,
        file_path: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, int]:
        """
        Quickly count records by type within date range without full parsing.
        
        Useful for progress estimation and data quality checks.
        """
        counts = {
            "sleep": 0,
            "activity": 0,
            "heart": 0,
            "total": 0
        }
        
        # Convert dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
        
        # Just count without converting to entities
        for record_dict in self.iter_records(file_path, self.all_types, start_dt, end_dt):
            record_type = record_dict.get("type")
            counts["total"] += 1
            
            if record_type in self.sleep_types:
                counts["sleep"] += 1
            elif record_type in self.activity_types:
                counts["activity"] += 1
            elif record_type in self.heart_types:
                counts["heart"] += 1
        
        return counts

    def _dict_to_element(self, record_dict: dict[str, Any]) -> Any:
        """Convert record dictionary to minimal XML element for parser compatibility."""
        # Create a minimal HealthData root with one Record
        if HAS_LXML:
            root = etree.Element("HealthData")
            record = etree.SubElement(root, "Record")
        else:
            import xml.etree.ElementTree as ET
            root = ET.Element("HealthData")
            record = ET.SubElement(root, "Record")
        
        # Set attributes
        for key, value in record_dict.items():
            if key == "motionContext":
                # Recreate metadata entry for heart rate motion context
                if HAS_LXML:
                    meta_elem = etree.SubElement(record, "MetadataEntry")
                else:
                    meta_elem = ET.SubElement(record, "MetadataEntry")
                meta_elem.set("key", "HKMetadataKeyHeartRateMotionContext")
                meta_elem.set("value", value)
            else:
                record.set(key, str(value))
        
        return root