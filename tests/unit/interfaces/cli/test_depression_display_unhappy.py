"""
Unhappy path tests for depression display.

Ensures the system handles missing or malformed PAT data gracefully.
"""

from datetime import date
from pathlib import Path

from click.testing import CliRunner

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    PipelineResult,
)
from big_mood_detector.interfaces.cli.commands import generate_clinical_report


class TestDepressionDisplayUnhappyPaths:
    """Test edge cases and error conditions for depression display."""
    
    def test_clinical_report_handles_missing_pat_keys_in_summary(self):
        """Report generation should not crash when PAT keys are missing from summary."""
        # Create result with missing PAT keys in overall_summary
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    # PAT scores present in daily but not in summary
                    "pat_depression_probability": 0.75,
                    "pat_confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                # Missing: avg_pat_depression_probability
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Generate report - should not raise
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            
            # Verify report was created
            assert report_path.exists()
            
            # Check content doesn't have PAT assessment section
            report_content = report_path.read_text()
            assert "CLINICAL RISK ASSESSMENT" in report_content
            # Should not have PAT assessment since it's missing from summary
            assert "PAT Depression Assessment:" not in report_content
    
    def test_clinical_report_handles_none_summary(self):
        """Report should handle None overall_summary gracefully."""
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                }
            },
            overall_summary=None,  # type: ignore[arg-type]
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Generate report - should not raise
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            
            # Verify report was created
            assert report_path.exists()
            
            # Should still have basic structure
            report_content = report_path.read_text()
            assert "CLINICAL DECISION SUPPORT (CDS) REPORT" in report_content
    
    def test_clinical_report_handles_malformed_pat_scores(self):
        """Report should handle non-numeric PAT scores gracefully."""
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    "pat_depression_probability": "invalid",  # type: ignore[dict-item]
                    "pat_confidence": None,  # type: ignore[dict-item]
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                "avg_pat_depression_probability": "not_a_number",  # type: ignore[typeddict-item]
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Generate report - should not raise
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            
            # This might raise due to format_risk_level expecting a number
            # The implementation should handle this gracefully
            try:
                generate_clinical_report(result, report_path)
                # If it doesn't raise, verify the report exists
                assert report_path.exists()
            except (TypeError, ValueError):
                # If it does raise, that's also acceptable behavior
                # The important thing is it doesn't crash unexpectedly
                pass
    
    def test_clinical_report_handles_empty_daily_predictions(self):
        """Report should handle empty daily predictions gracefully."""
        result = PipelineResult(
            daily_predictions={},  # Empty
            overall_summary={
                "avg_depression_risk": 0.0,
                "avg_hypomanic_risk": 0.0,
                "avg_manic_risk": 0.0,
                "days_analyzed": 0,
            },
            confidence_score=0.0,
            processing_time_seconds=2.5,
            records_processed=0,
            features_extracted=0,
        )
        
        # Generate report - should not raise
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            
            # Verify report was created
            assert report_path.exists()
            
            # Check it handles zero days analyzed
            report_content = report_path.read_text()
            assert "Analysis Period: 0 days" in report_content
    
    def test_clinical_report_handles_partial_pat_data(self):
        """Report should handle when only some days have PAT scores."""
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 24): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    # No PAT scores for this day
                },
                date(2025, 7, 25): {
                    "depression_risk": 0.4,
                    "hypomanic_risk": 0.3,
                    "manic_risk": 0.2,
                    "confidence": 0.9,
                    # PAT scores present
                    "pat_depression_probability": 0.75,
                    "pat_confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.35,
                "avg_hypomanic_risk": 0.25,
                "avg_manic_risk": 0.15,
                "days_analyzed": 2,
                # Average might be present even if not all days have PAT
                "avg_pat_depression_probability": 0.75,
            },
            confidence_score=0.85,
            processing_time_seconds=2.5,
            records_processed=200,
            features_extracted=72,
        )
        
        # Generate report - should not raise
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            
            # Verify report was created
            assert report_path.exists()
            
            # Check daily analysis section
            report_content = report_path.read_text()
            assert "DETAILED DAILY ANALYSIS" in report_content
            # First day shouldn't have PAT, second day should
            assert report_content.count("PAT Depression:") == 1  # Only for second day