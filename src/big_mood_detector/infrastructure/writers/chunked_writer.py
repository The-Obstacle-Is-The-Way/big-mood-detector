"""
Chunked Writer for Streaming Output

Writes data in chunks to avoid memory issues with large datasets.
Supports CSV and Parquet formats.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ChunkedWriter:
    """
    Write data in chunks to avoid memory issues.
    
    Useful for processing very large XML files where keeping
    all records in memory would be problematic.
    """
    
    def __init__(
        self,
        output_path: Path,
        format: str = "csv",
        chunk_size: int = 10000,
    ):
        """
        Initialize chunked writer.
        
        Args:
            output_path: Path to output file
            format: Output format ('csv' or 'parquet')
            chunk_size: Records per chunk
        """
        self.output_path = output_path
        self.format = format.lower()
        self.chunk_size = chunk_size
        self.chunk_buffer: list[dict[str, Any]] = []
        self.chunks_written = 0
        self.first_chunk = True
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For parquet, we'll write to temporary files and combine
        if self.format == "parquet":
            self.temp_dir = self.output_path.parent / f".{self.output_path.stem}_chunks"
            self.temp_dir.mkdir(exist_ok=True)
    
    def write_record(self, record: dict[str, Any]) -> None:
        """Add a record to the buffer and write if chunk is full."""
        self.chunk_buffer.append(record)
        
        if len(self.chunk_buffer) >= self.chunk_size:
            self._write_chunk()
    
    def write_records(self, records: list[dict[str, Any]]) -> None:
        """Write multiple records, handling chunking."""
        for record in records:
            self.write_record(record)
    
    def _write_chunk(self) -> None:
        """Write current chunk to disk."""
        if not self.chunk_buffer:
            return
        
        df = pd.DataFrame(self.chunk_buffer)
        
        if self.format == "csv":
            # For CSV, append to existing file
            mode = "w" if self.first_chunk else "a"
            header = self.first_chunk
            df.to_csv(self.output_path, mode=mode, header=header, index=False)
            
        elif self.format == "parquet":
            # For parquet, write to temporary chunk file
            chunk_path = self.temp_dir / f"chunk_{self.chunks_written:06d}.parquet"
            df.to_parquet(chunk_path, index=False)
        
        logger.info(f"Wrote chunk {self.chunks_written} with {len(df)} records")
        
        self.chunks_written += 1
        self.first_chunk = False
        self.chunk_buffer = []
    
    def finalize(self) -> Path:
        """
        Write any remaining records and finalize output.
        
        Returns:
            Path to final output file
        """
        # Write remaining records
        if self.chunk_buffer:
            self._write_chunk()
        
        if self.format == "parquet" and self.chunks_written > 0:
            # Combine all parquet chunks
            self._combine_parquet_chunks()
        
        logger.info(f"Finalized output: {self.output_path}")
        logger.info(f"Total chunks written: {self.chunks_written}")
        
        return self.output_path
    
    def _combine_parquet_chunks(self) -> None:
        """Combine temporary parquet chunks into final file."""
        import pyarrow.parquet as pq
        
        chunk_files = sorted(self.temp_dir.glob("chunk_*.parquet"))
        
        if not chunk_files:
            logger.warning("No parquet chunks to combine")
            return
        
        # Read and combine all chunks
        tables = []
        for chunk_file in chunk_files:
            table = pq.read_table(chunk_file)
            tables.append(table)
        
        # Write combined table
        combined_table = pq.concat_tables(tables)
        pq.write_table(combined_table, self.output_path)
        
        # Clean up temporary files
        for chunk_file in chunk_files:
            chunk_file.unlink()
        self.temp_dir.rmdir()
        
        logger.info(f"Combined {len(chunk_files)} chunks into {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure finalization."""
        self.finalize()


class StreamingFeatureWriter(ChunkedWriter):
    """
    Specialized writer for streaming feature extraction results.
    
    Converts domain entities to feature dictionaries on the fly.
    """
    
    def __init__(
        self,
        output_path: Path,
        format: str = "csv",
        chunk_size: int = 1000,
    ):
        """Initialize with smaller default chunk for features."""
        super().__init__(output_path, format, chunk_size)
        self.dates_seen = set()
    
    def write_features(self, date: str, features: dict[str, float]) -> None:
        """
        Write a day's features.
        
        Args:
            date: Date string (YYYY-MM-DD)
            features: Feature dictionary
        """
        if date not in self.dates_seen:
            record = {"date": date, **features}
            self.write_record(record)
            self.dates_seen.add(date)
    
    def write_clinical_features(self, clinical_features: Any) -> None:
        """
        Write clinical features from domain object.
        
        Args:
            clinical_features: ClinicalFeatureSet object
        """
        record = {
            "date": clinical_features.date,
            "daily_steps": clinical_features.daily_steps,
            "activity_variance": clinical_features.activity_variance,
            "sedentary_hours": clinical_features.sedentary_hours,
            "activity_fragmentation": clinical_features.activity_fragmentation,
            "sedentary_bout_mean": clinical_features.sedentary_bout_mean,
            "activity_intensity_ratio": clinical_features.activity_intensity_ratio,
            "sleep_duration_hours": clinical_features.sleep_duration_hours,
            "sleep_efficiency": clinical_features.sleep_efficiency,
            "sleep_onset_hour": clinical_features.sleep_onset_hour,
            "wake_time_hour": clinical_features.wake_time_hour,
            # Add more fields as needed
        }
        self.write_record(record)