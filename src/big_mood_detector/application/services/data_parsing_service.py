"""
Data Parsing Service

Handles all data parsing operations for the mood prediction pipeline.
Extracted from MoodPredictionPipeline to follow Single Responsibility Principle.

Design Patterns:
- Strategy Pattern: Different parsing strategies for different file types
- Factory Pattern: Parser creation based on file type
- Facade Pattern: Unified interface for all parsing operations
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.infrastructure.parsers.json.json_parsers import (
    ActivityJSONParser,
    SleepJSONParser,
)
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)

# Try to import fast parser (20x faster with lxml)
try:
    from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
        FastStreamingXMLParser,
    )

    HAS_FAST_PARSER = True
except ImportError:
    HAS_FAST_PARSER = False


@dataclass
class ParsedHealthData:
    """Container for parsed health data."""

    sleep_records: list[SleepRecord]
    activity_records: list[ActivityRecord]
    heart_rate_records: list[HeartRateRecord]
    errors: list[str] | None = None


@dataclass
class DataValidationResult:
    """Result of data validation."""

    is_valid: bool
    sleep_record_count: int
    activity_record_count: int
    heart_record_count: int
    date_range: tuple[date, date] | None
    warnings: list[str]


@dataclass
class DataSummary:
    """Summary of parsed health data."""

    total_records: int
    sleep_days: int
    activity_days: int
    heart_days: int
    date_range: tuple[date, date] | None
    data_density: float  # 0-1, percentage of days with data


class HealthDataParser(Protocol):
    """Protocol for health data parsers."""

    def parse(self, file_path: Path) -> dict[str, list]:
        """Parse health data from file."""
        ...


class DataParsingService:
    """
    Service responsible for all data parsing operations.

    This service encapsulates:
    - File type detection
    - Parser selection
    - Data validation
    - Error handling
    - Progress reporting
    """

    # File size threshold for streaming (100MB)
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024

    def __init__(
        self,
        xml_parser: StreamingXMLParser | FastStreamingXMLParser | None = None,
        sleep_parser: SleepJSONParser | None = None,
        activity_parser: ActivityJSONParser | None = None,
    ):
        """
        Initialize with parser dependencies.

        Args:
            xml_parser: Parser for XML exports
            sleep_parser: Parser for sleep JSON data
            activity_parser: Parser for activity JSON data
        """
        # Use fast parser if available (20x faster with lxml)
        if HAS_FAST_PARSER and xml_parser is None:
            self._xml_parser: StreamingXMLParser | FastStreamingXMLParser = (
                FastStreamingXMLParser()
            )
        else:
            self._xml_parser = xml_parser or StreamingXMLParser()
        self._sleep_parser = sleep_parser or SleepJSONParser()
        self._activity_parser = activity_parser or ActivityJSONParser()

        # Cache for parsed data
        self._cache: dict[str, ParsedHealthData] = {}

        # Supported data sources
        self._sources = {
            "apple_health_xml": self._xml_parser,
            "health_auto_export_json": {
                "sleep": self._sleep_parser,
                "activity": self._activity_parser,
            },
        }

    def parse_health_data(
        self,
        file_path: Path,
        start_date: date | None = None,
        end_date: date | None = None,
        use_cache: bool = True,
        continue_on_error: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Parse health data from file with optional filtering.

        Args:
            file_path: Path to health data file/directory
            start_date: Optional start date filter
            end_date: Optional end date filter
            use_cache: Whether to use cached results
            continue_on_error: Whether to continue on parsing errors
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with parsed records and metadata
        """
        # Check cache
        cache_key = str(file_path)
        if use_cache and cache_key in self._cache:
            cached_data = self._cache[cache_key]
            return self._format_result(cached_data)

        try:
            # Determine parser type
            if file_path.is_file() and file_path.suffix == ".xml":
                parsed_data = self.parse_xml_export(
                    file_path, progress_callback, start_date, end_date
                )
            elif file_path.is_file() and file_path.suffix == ".zip":
                parsed_data = self.parse_json_zip(file_path, progress_callback)
                # Filter JSON data by date range if specified
                if start_date or end_date:
                    parsed_data = self._filter_by_date_range(
                        parsed_data, start_date, end_date
                    )
            elif file_path.is_dir():
                parsed_data = self.parse_json_export(file_path, progress_callback)
                # Filter JSON data by date range if specified
                if start_date or end_date:
                    parsed_data = self._filter_by_date_range(
                        parsed_data, start_date, end_date
                    )
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Cache result
            if use_cache:
                self._cache[cache_key] = parsed_data

            return self._format_result(parsed_data)

        except Exception as e:
            if continue_on_error:
                return {
                    "sleep_records": [],
                    "activity_records": [],
                    "heart_rate_records": [],
                    "errors": [str(e)],
                }
            raise

    def parse_xml_export(
        self,
        xml_path: Path,
        progress_callback: Callable[[str, float], None] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ParsedHealthData:
        """
        Parse Apple Health XML export efficiently with single-pass parsing.

        Args:
            xml_path: Path to export.xml
            progress_callback: Optional progress callback
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            ParsedHealthData with all records
        """
        sleep_records = []
        activity_records = []
        heart_records = []

        # Convert dates to string format for parser
        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None

        if progress_callback:
            progress_callback("Parsing all records (single pass)", 0.0)

        # Single-pass parsing: parse all entity types in one iteration
        # This is 3x faster than the previous approach that made 3 separate passes
        for entity in self._xml_parser.parse_file(
            xml_path,
            entity_type="all",
            start_date=start_date_str,
            end_date=end_date_str,
            progress_callback=progress_callback,
        ):
            if isinstance(entity, SleepRecord):
                sleep_records.append(entity)
            elif isinstance(entity, ActivityRecord):
                activity_records.append(entity)
            elif isinstance(entity, HeartRateRecord):
                heart_records.append(entity)

        if progress_callback:
            progress_callback("Parsing complete", 1.0)

        return ParsedHealthData(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_rate_records=heart_records,
        )

    def parse_json_export(
        self,
        json_dir: Path,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ParsedHealthData:
        """
        Parse Health Auto Export JSON files.

        Args:
            json_dir: Directory containing JSON files
            progress_callback: Optional progress callback

        Returns:
            ParsedHealthData with all records
        """
        sleep_records = []
        activity_records = []

        # Find and parse sleep files
        if progress_callback:
            progress_callback("Parsing sleep files", 0.0)

        sleep_files = list(
            set(
                list(json_dir.glob("*[Ss]leep*.json"))
                + list(json_dir.glob("[Ss]leep*.json"))
            )
        )
        for i, file in enumerate(sleep_files):
            sleep_data = self._sleep_parser.parse_file(str(file))
            sleep_records.extend(sleep_data)

            if progress_callback:
                progress = (i + 1) / len(sleep_files) if sleep_files else 1.0
                progress_callback("Parsing sleep files", progress)

        # Find and parse activity files
        if progress_callback:
            progress_callback("Parsing activity files", 0.0)

        activity_files = list(
            set(
                list(json_dir.glob("*[Ss]tep*.json"))
                + list(json_dir.glob("[Ss]tep*.json"))
            )
        )
        for i, file in enumerate(activity_files):
            activity_data = self._activity_parser.parse_file(str(file))
            activity_records.extend(activity_data)

            if progress_callback:
                progress = (i + 1) / len(activity_files) if activity_files else 1.0
                progress_callback("Parsing activity files", progress)

        return ParsedHealthData(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_rate_records=[],  # JSON export doesn't include heart rate
        )

    def parse_json_zip(
        self,
        zip_path: Path,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ParsedHealthData:
        """
        Parse Health Auto Export JSON files from a ZIP archive.

        Args:
            zip_path: Path to ZIP file containing JSON exports
            progress_callback: Optional progress callback

        Returns:
            ParsedHealthData with all records
        """
        import tempfile
        import zipfile

        sleep_records = []
        activity_records = []
        heart_rate_records: list[HeartRateRecord] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check if this is an Apple Health XML export or JSON export
            xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
            json_files = [f for f in zf.namelist() if f.endswith(".json")]

            if xml_files and any("export.xml" in f for f in xml_files):
                # This is an Apple Health XML export ZIP
                # Extract and parse the XML file
                for xml_file in xml_files:
                    if "export.xml" in xml_file:
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".xml", delete=False
                        ) as tmp:
                            tmp.write(zf.read(xml_file))
                            tmp_path = Path(tmp.name)

                        # Parse as XML export
                        parsed_data = self.parse_xml_export(tmp_path, progress_callback)
                        tmp_path.unlink()  # Clean up

                        return parsed_data

            elif json_files:
                # This is a Health Auto Export JSON ZIP
                # Process each JSON file
                for i, filename in enumerate(json_files):
                    if progress_callback:
                        progress = i / len(json_files) if json_files else 0
                        progress_callback(f"Processing {filename}", progress)

                    # Read JSON content
                    json_content = zf.read(filename)

                    # Parse based on filename
                    if "sleep" in filename.lower():
                        # Write to temp file for parser
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".json", delete=False
                        ) as tmp:
                            tmp.write(json_content)
                            tmp_path = Path(tmp.name)

                        sleep_data = self._sleep_parser.parse_file(tmp_path)
                        sleep_records.extend(sleep_data)
                        Path(tmp_path).unlink()  # Clean up temp file

                    elif (
                        "step" in filename.lower()
                        or "activity" in filename.lower()
                        or "count" in filename.lower()
                    ):
                        # Write to temp file for parser
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".json", delete=False
                        ) as tmp:
                            tmp.write(json_content)
                            tmp_path = Path(tmp.name)

                        activity_data = self._activity_parser.parse_file(tmp_path)
                        activity_records.extend(activity_data)
                        Path(tmp_path).unlink()  # Clean up temp file

                if progress_callback:
                    progress_callback("Processing complete", 1.0)

            else:
                # No supported files found
                raise ValueError(
                    "ZIP file does not contain supported health data (no export.xml or JSON files found)"
                )

        return ParsedHealthData(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_rate_records=heart_rate_records,
        )

    def filter_records_by_date_range(
        self,
        records: list[Any],
        start_date: date | None,
        end_date: date | None,
        date_extractor: Callable[[Any], date],
    ) -> list[Any]:
        """
        Filter records by date range.

        Args:
            records: List of records to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            date_extractor: Function to extract date from record

        Returns:
            Filtered list of records
        """
        filtered = records

        if start_date:
            filtered = [r for r in filtered if date_extractor(r) >= start_date]

        if end_date:
            filtered = [r for r in filtered if date_extractor(r) <= end_date]

        return filtered

    def validate_parsed_data(
        self, data: dict[str, list] | ParsedHealthData
    ) -> DataValidationResult:
        """
        Validate parsed health data.

        Args:
            data: Dictionary with parsed records or ParsedHealthData object

        Returns:
            Validation result with counts and warnings
        """
        # Convert ParsedHealthData to dict if needed
        if isinstance(data, ParsedHealthData):
            data = self._format_result(data)

        sleep_records = data.get("sleep_records", [])
        activity_records = data.get("activity_records", [])
        heart_records = data.get("heart_rate_records", [])

        warnings = []

        # Calculate date range
        all_dates = []
        if sleep_records:
            all_dates.extend([r.start_date.date() for r in sleep_records])
        if activity_records:
            all_dates.extend([r.start_date.date() for r in activity_records])
        if heart_records:
            all_dates.extend([r.timestamp.date() for r in heart_records])

        date_range = None
        if all_dates:
            date_range = (min(all_dates), max(all_dates))

        # Check for data issues
        if not sleep_records:
            warnings.append("No sleep records found")
        if not activity_records:
            warnings.append("No activity records found")

        # Check data density
        if date_range and sleep_records:
            days_span = (date_range[1] - date_range[0]).days + 1
            sleep_days = len({r.start_date.date() for r in sleep_records})
            density = sleep_days / days_span
            if density < 0.5:
                warnings.append(f"Sparse sleep data: {density:.1%} coverage")

        return DataValidationResult(
            is_valid=len(sleep_records) > 0 or len(activity_records) > 0,
            sleep_record_count=len(sleep_records),
            activity_record_count=len(activity_records),
            heart_record_count=len(heart_records),
            date_range=date_range,
            warnings=warnings,
        )

    def get_data_summary(self, data: dict[str, list]) -> dict[str, Any]:
        """
        Get summary of parsed data.

        Args:
            data: Dictionary with parsed records

        Returns:
            Summary dictionary
        """
        sleep_records = data.get("sleep_records", [])
        activity_records = data.get("activity_records", [])
        heart_records = data.get("heart_rate_records", [])

        # Get unique days
        sleep_days = {r.start_date.date() for r in sleep_records}
        activity_days = {r.start_date.date() for r in activity_records}
        heart_days = {r.timestamp.date() for r in heart_records}

        # Calculate date range
        all_dates = list(sleep_days | activity_days | heart_days)
        date_range = None
        data_density = 0.0

        if all_dates:
            date_range = (min(all_dates), max(all_dates))
            days_span = (date_range[1] - date_range[0]).days + 1
            days_with_data = len(all_dates)
            data_density = days_with_data / days_span

        return {
            "total_records": len(sleep_records)
            + len(activity_records)
            + len(heart_records),
            "sleep_days": len(sleep_days),
            "activity_days": len(activity_days),
            "heart_days": len(heart_days),
            "date_range": date_range,
            "data_density": data_density,
        }

    def get_parser_for_path(self, file_path: Path) -> Any:
        """
        Get appropriate parser for file path.

        Args:
            file_path: Path to data file

        Returns:
            Parser instance

        Raises:
            ValueError: If file type not supported
        """
        if file_path.is_file() and file_path.suffix == ".xml":
            return self._xml_parser
        elif file_path.is_dir():
            # Return a composite parser for JSON directory
            return self  # Self can act as parser
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def get_supported_sources(self) -> list[str]:
        """Get list of supported data sources."""
        return list(self._sources.keys())

    def get_parser_for_source(self, source: str) -> Any:
        """Get parser for specific data source."""
        return self._sources.get(source)

    def parse_large_file(self, file_path: Path) -> ParsedHealthData:
        """
        Parse large files using memory-efficient methods.

        Args:
            file_path: Path to large file

        Returns:
            ParsedHealthData
        """
        # For XML files, StreamingXMLParser already handles this
        # For JSON, would need to implement streaming JSON parser
        return self.parse_xml_export(file_path)

    def clear_cache(self) -> None:
        """Clear the parsing cache."""
        self._cache.clear()

    def _select_parser_type(self, file_path: Path) -> str:
        """Select parser type based on file characteristics."""
        if not file_path.exists():
            return "standard"

        file_size = file_path.stat().st_size
        if file_size > self.LARGE_FILE_THRESHOLD:
            return "streaming"
        return "standard"

    def _filter_by_date_range(
        self, data: ParsedHealthData, start_date: date | None, end_date: date | None
    ) -> ParsedHealthData:
        """Filter parsed data by date range."""
        return ParsedHealthData(
            sleep_records=self.filter_records_by_date_range(
                data.sleep_records, start_date, end_date, lambda r: r.start_date.date()
            ),
            activity_records=self.filter_records_by_date_range(
                data.activity_records,
                start_date,
                end_date,
                lambda r: r.start_date.date(),
            ),
            heart_rate_records=self.filter_records_by_date_range(
                data.heart_rate_records,
                start_date,
                end_date,
                lambda r: r.timestamp.date(),
            ),
            errors=data.errors,
        )

    def _format_result(self, data: ParsedHealthData) -> dict[str, Any]:
        """Format ParsedHealthData as dictionary."""
        result = {
            "sleep_records": data.sleep_records,
            "activity_records": data.activity_records,
            "heart_rate_records": data.heart_rate_records,
        }
        if data.errors:
            result["errors"] = data.errors
        return result
