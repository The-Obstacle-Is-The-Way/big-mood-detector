"""
Apple HealthKit Sleep Data Parser

Clinical-grade sleep data extraction from Apple Health exports.
Based on proven reference implementation.

Following SOLID principles:
- Single Responsibility: Only parses sleep data from XML
- Open/Closed: Extensible for new sleep record types
- Liskov Substitution: Can be substituted with other parsers
- Interface Segregation: Focused interface for sleep parsing
- Dependency Inversion: Depends on abstractions (will add later)
"""

from typing import List, Dict, Any
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError


class SleepParser:
    """Parses Apple HealthKit sleep data from XML exports."""
    
    # Constants following DRY principle
    SLEEP_RECORD_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
    RECORD_TAG = "Record"
    
    def __init__(self):
        """Initialize sleep parser."""
        pass
    
    def parse(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        Parse Apple HealthKit XML and extract sleep records.
        
        Args:
            xml_data: XML string containing HealthKit export data
            
        Returns:
            List of sleep record dictionaries with keys:
            - sourceName: Device that recorded the data
            - startDate: Sleep start timestamp
            - endDate: Sleep end timestamp  
            - value: Sleep state (InBed or Asleep)
            
        Raises:
            ValueError: If XML is invalid or malformed
        """
        try:
            root = ET.fromstring(xml_data)
        except ParseError as e:
            raise ValueError(f"Invalid XML: {str(e)}")
        
        sleep_records = []
        
        # Extract only sleep analysis records
        for record in root.findall(f"./{self.RECORD_TAG}"):
            if record.get("type") == self.SLEEP_RECORD_TYPE:
                sleep_records.append(self._extract_sleep_data(record))
        
        return sleep_records
    
    def _extract_sleep_data(self, element: ET.Element) -> Dict[str, Any]:
        """
        Extract sleep data from a single XML element.
        
        Single Responsibility: This method only extracts data from one element.
        """
        return {
            "sourceName": element.get("sourceName"),
            "startDate": element.get("startDate"),
            "endDate": element.get("endDate"),
            "value": element.get("value"),
        }