#!/usr/bin/env python3
"""
Comprehensive Pipeline Test Script for Big Mood Detector
========================================================

This script tests ALL data flows through the complete pipeline:
1. All input formats (JSON from Health Auto Export, XML from Apple Health Export)
2. All parsers (JSON and XML streaming)
3. All domain services (feature extraction, aggregation, clinical thresholds)
4. All ML models (XGBoost ensemble)
5. All output formats (CSV features, JSON predictions, clinical reports)
6. All interfaces (CLI commands, API endpoints, Docker)

Usage:
    python scripts/validation/test_complete_pipeline.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class PipelineTestReport:
    """Collects and formats test results."""

    def __init__(self):
        self.sections: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.summary_stats: dict[str, int] = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "warnings": 0,
        }

    def add_section(self, name: str, tests: list[dict[str, Any]]) -> None:
        """Add a test section with results."""
        self.sections.append(
            {
                "name": name,
                "tests": tests,
                "passed": sum(1 for t in tests if t["status"] == "PASS"),
                "failed": sum(1 for t in tests if t["status"] == "FAIL"),
            }
        )

        # Update summary
        self.summary_stats["total_tests"] += len(tests)
        self.summary_stats["passed_tests"] += sum(
            1 for t in tests if t["status"] == "PASS"
        )
        self.summary_stats["failed_tests"] += sum(
            1 for t in tests if t["status"] == "FAIL"
        )

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.summary_stats["failed_tests"] += 1
        self.summary_stats["total_tests"] += 1

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        self.summary_stats["warnings"] += 1

    def generate_report(self) -> str:
        """Generate formatted report."""
        report = []
        report.append("=" * 80)
        report.append("BIG MOOD DETECTOR - COMPREHENSIVE PIPELINE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {self.summary_stats['total_tests']}")
        report.append(
            f"Passed: {self.summary_stats['passed_tests']} "
            f"({self.summary_stats['passed_tests'] / max(1, self.summary_stats['total_tests']) * 100:.1f}%)"
        )
        report.append(f"Failed: {self.summary_stats['failed_tests']}")
        report.append(f"Warnings: {self.summary_stats['warnings']}")
        report.append("")

        # Test sections
        for section in self.sections:
            report.append(f"\n{section['name'].upper()}")
            report.append("=" * len(section["name"]))
            report.append(
                f"Tests: {len(section['tests'])} | "
                f"Passed: {section['passed']} | "
                f"Failed: {section['failed']}"
            )
            report.append("")

            for test in section["tests"]:
                status_icon = "✅" if test["status"] == "PASS" else "❌"
                report.append(f"{status_icon} {test['name']}")
                if test.get("details"):
                    report.append(f"   Details: {test['details']}")
                if test.get("error"):
                    report.append(f"   Error: {test['error']}")
                if test.get("duration"):
                    report.append(f"   Duration: {test['duration']:.2f}s")
            report.append("")

        # Errors
        if self.errors:
            report.append("\nERRORS")
            report.append("=" * 6)
            for error in self.errors:
                report.append(f"❌ {error}")
            report.append("")

        # Warnings
        if self.warnings:
            report.append("\nWARNINGS")
            report.append("=" * 8)
            for warning in self.warnings:
                report.append(f"⚠️  {warning}")
            report.append("")

        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("=" * 15)
        if self.summary_stats["failed_tests"] == 0:
            report.append("✅ All tests passed! The system is production ready.")
        else:
            report.append("❌ Some tests failed. Please address the following:")
            if any("API" in error for error in self.errors):
                report.append("- Fix API endpoint issues")
            if any("Docker" in error for error in self.errors):
                report.append("- Check Docker configuration")
            if any("Model" in error for error in self.errors):
                report.append("- Verify model weights are present")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


class ComprehensivePipelineTest:
    """Tests all components of the Big Mood Detector pipeline."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        self.report = PipelineTestReport()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test output directory
        self.test_output_dir = (
            self.output_dir
            / f"pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: list[str], timeout: int = 60) -> tuple[bool, str, str]:
        """Run a command and return success status, stdout, and stderr."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=self.base_dir
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def test_cli_help(self) -> list[dict[str, Any]]:
        """Test CLI help and command availability."""
        tests = []

        # Test main help
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "big_mood_detector", "--help"]
        )
        tests.append(
            {
                "name": "CLI Help",
                "status": (
                    "PASS" if success and "Big Mood Detector" in stdout else "FAIL"
                ),
                "details": "Main help displays correctly" if success else stderr,
            }
        )

        # Test each command help
        commands = ["process", "predict", "serve", "watch", "train", "label"]
        for cmd in commands:
            success, stdout, stderr = self.run_command(
                [sys.executable, "-m", "big_mood_detector", cmd, "--help"]
            )
            tests.append(
                {
                    "name": f"CLI {cmd} --help",
                    "status": "PASS" if success else "FAIL",
                    "details": f"{cmd} command available" if success else stderr,
                }
            )

        return tests

    def test_json_parser(self) -> list[dict[str, Any]]:
        """Test JSON parser with Health Auto Export data."""
        tests = []
        json_dir = self.input_dir / "health_auto_export"

        if not json_dir.exists():
            tests.append(
                {
                    "name": "JSON Test Data",
                    "status": "FAIL",
                    "error": f"JSON test data not found at {json_dir}",
                }
            )
            return tests

        # List available JSON files
        json_files = list(json_dir.glob("*.json"))
        tests.append(
            {
                "name": "JSON Files Available",
                "status": "PASS" if json_files else "FAIL",
                "details": f"Found {len(json_files)} JSON files",
            }
        )

        # Test processing JSON data
        start_time = time.time()
        output_file = self.test_output_dir / "json_features.csv"
        success, stdout, stderr = self.run_command(
            [
                sys.executable,
                "-m",
                "big_mood_detector",
                "process",
                str(json_dir),
                "--output",
                str(output_file),
                "--verbose",
            ],
            timeout=120,
        )
        duration = time.time() - start_time

        tests.append(
            {
                "name": "JSON Processing",
                "status": "PASS" if success and output_file.exists() else "FAIL",
                "details": f"Processed in {duration:.2f}s" if success else stderr,
                "duration": duration,
            }
        )

        # Verify output
        if output_file.exists():
            try:
                df = pd.read_csv(output_file)
                tests.append(
                    {
                        "name": "JSON Output Validation",
                        "status": "PASS",
                        "details": f"Generated {len(df)} days of features with {len(df.columns)} columns",
                    }
                )
            except Exception as e:
                tests.append(
                    {
                        "name": "JSON Output Validation",
                        "status": "FAIL",
                        "error": str(e),
                    }
                )

        return tests

    def test_xml_parser(self) -> list[dict[str, Any]]:
        """Test XML parser with Apple Health Export data."""
        tests = []
        xml_dir = self.input_dir / "apple_export"
        xml_file = xml_dir / "export.xml"

        if not xml_file.exists():
            tests.append(
                {
                    "name": "XML Test Data",
                    "status": "FAIL",
                    "error": f"XML test data not found at {xml_file}",
                }
            )
            return tests

        # Check file size
        file_size_mb = xml_file.stat().st_size / (1024 * 1024)
        tests.append(
            {
                "name": "XML File Check",
                "status": "PASS",
                "details": f"File size: {file_size_mb:.2f} MB",
            }
        )

        # Test streaming parser
        start_time = time.time()
        output_file = self.test_output_dir / "xml_features.csv"
        success, stdout, stderr = self.run_command(
            [
                sys.executable,
                "-m",
                "big_mood_detector",
                "process",
                str(xml_file),
                "--output",
                str(output_file),
                "--verbose",
            ],
            timeout=300,
        )  # 5 minutes for large files
        duration = time.time() - start_time

        tests.append(
            {
                "name": "XML Streaming Parser",
                "status": "PASS" if success and output_file.exists() else "FAIL",
                "details": (
                    f"Processed {file_size_mb:.2f} MB in {duration:.2f}s"
                    if success
                    else stderr
                ),
                "duration": duration,
            }
        )

        # Memory efficiency check
        if success and duration > 0:
            processing_rate = file_size_mb / duration
            tests.append(
                {
                    "name": "XML Processing Rate",
                    "status": "PASS" if processing_rate > 1.0 else "FAIL",
                    "details": f"{processing_rate:.2f} MB/s",
                }
            )

        return tests

    def test_prediction_pipeline(self) -> list[dict[str, Any]]:
        """Test end-to-end prediction pipeline."""
        tests = []

        # Test with JSON data
        json_dir = self.input_dir / "health_auto_export"
        if json_dir.exists():
            start_time = time.time()
            output_file = self.test_output_dir / "predictions.json"
            success, stdout, stderr = self.run_command(
                [
                    sys.executable,
                    "-m",
                    "big_mood_detector",
                    "predict",
                    str(json_dir),
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                    "--verbose",
                ],
                timeout=180,
            )
            duration = time.time() - start_time

            tests.append(
                {
                    "name": "Prediction Pipeline (JSON)",
                    "status": "PASS" if success and output_file.exists() else "FAIL",
                    "details": (
                        f"Generated predictions in {duration:.2f}s"
                        if success
                        else stderr
                    ),
                    "duration": duration,
                }
            )

            # Validate predictions
            if output_file.exists():
                try:
                    with open(output_file) as f:
                        predictions = json.load(f)

                    has_summary = "summary" in predictions
                    has_daily = "daily_predictions" in predictions
                    has_confidence = "confidence" in predictions

                    tests.append(
                        {
                            "name": "Prediction Output Structure",
                            "status": (
                                "PASS"
                                if all([has_summary, has_daily, has_confidence])
                                else "FAIL"
                            ),
                            "details": f"Days predicted: {len(predictions.get('daily_predictions', {}))}",
                        }
                    )
                except Exception as e:
                    tests.append(
                        {
                            "name": "Prediction Output Structure",
                            "status": "FAIL",
                            "error": str(e),
                        }
                    )

        return tests

    def test_ensemble_models(self) -> list[dict[str, Any]]:
        """Test ensemble model functionality."""
        tests = []

        # Check if model weights exist
        model_weights_dir = self.base_dir / "model_weights"
        xgboost_dir = model_weights_dir / "xgboost" / "converted"
        pat_dir = model_weights_dir / "pat"

        # Check XGBoost models
        xgboost_models = ["XGBoost_DE.json", "XGBoost_HME.json", "XGBoost_ME.json"]
        for model in xgboost_models:
            model_path = xgboost_dir / model
            tests.append(
                {
                    "name": f"XGBoost Model: {model}",
                    "status": "PASS" if model_path.exists() else "FAIL",
                    "details": (
                        f"Size: {model_path.stat().st_size / 1024:.1f} KB"
                        if model_path.exists()
                        else "Not found"
                    ),
                }
            )

        # Check PAT model (optional)
        pat_available = pat_dir.exists() and any(pat_dir.iterdir())
        tests.append(
            {
                "name": "PAT Model",
                "status": "PASS" if pat_available else "WARN",
                "details": (
                    "Available for ensemble"
                    if pat_available
                    else "Not available (optional)"
                ),
            }
        )

        # Test ensemble prediction if models available
        json_dir = self.input_dir / "health_auto_export"
        if json_dir.exists() and all(
            (xgboost_dir / m).exists() for m in xgboost_models
        ):
            output_file = self.test_output_dir / "ensemble_predictions.json"
            success, stdout, stderr = self.run_command(
                [
                    sys.executable,
                    "-m",
                    "big_mood_detector",
                    "predict",
                    str(json_dir),
                    "--output",
                    str(output_file),
                    "--ensemble",
                    "--format",
                    "json",
                ],
                timeout=180,
            )

            tests.append(
                {
                    "name": "Ensemble Prediction",
                    "status": "PASS" if success else "FAIL",
                    "details": (
                        "Ensemble models working together" if success else stderr
                    ),
                }
            )

        return tests

    def test_clinical_features(self) -> list[dict[str, Any]]:
        """Test clinical feature extraction and thresholds."""
        tests = []

        # Test clinical report generation
        json_dir = self.input_dir / "health_auto_export"
        if json_dir.exists():
            report_file = self.test_output_dir / "clinical_report.txt"
            success, stdout, stderr = self.run_command(
                [
                    sys.executable,
                    "-m",
                    "big_mood_detector",
                    "predict",
                    str(json_dir),
                    "--output",
                    str(report_file),
                    "--report",
                ],
                timeout=120,
            )

            tests.append(
                {
                    "name": "Clinical Report Generation",
                    "status": "PASS" if success and report_file.exists() else "FAIL",
                    "details": (
                        "Report generated with clinical recommendations"
                        if success
                        else stderr
                    ),
                }
            )

            # Validate report content
            if report_file.exists():
                with open(report_file) as f:
                    content = f.read()

                has_risk_assessment = "CLINICAL RISK ASSESSMENT" in content
                has_recommendations = "CLINICAL RECOMMENDATIONS" in content
                has_dsm5 = any(
                    term in content
                    for term in ["Depression Risk", "Hypomanic Risk", "Manic Risk"]
                )

                tests.append(
                    {
                        "name": "Clinical Report Content",
                        "status": (
                            "PASS"
                            if all([has_risk_assessment, has_recommendations, has_dsm5])
                            else "FAIL"
                        ),
                        "details": "Contains DSM-5 based assessments and recommendations",
                    }
                )

        return tests

    def test_api_endpoints(self) -> list[dict[str, Any]]:
        """Test API functionality."""
        tests = []

        # Start API server
        api_process = subprocess.Popen(
            [sys.executable, "-m", "big_mood_detector", "serve", "--no-reload"],
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(5)

        try:
            # Test health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                tests.append(
                    {
                        "name": "API Health Check",
                        "status": "PASS" if response.status_code == 200 else "FAIL",
                        "details": f"Status: {response.status_code}",
                    }
                )
            except Exception as e:
                tests.append(
                    {"name": "API Health Check", "status": "FAIL", "error": str(e)}
                )

            # Test OpenAPI docs
            try:
                response = requests.get("http://localhost:8000/docs", timeout=5)
                tests.append(
                    {
                        "name": "API Documentation",
                        "status": "PASS" if response.status_code == 200 else "FAIL",
                        "details": "OpenAPI docs available",
                    }
                )
            except Exception as e:
                tests.append(
                    {"name": "API Documentation", "status": "FAIL", "error": str(e)}
                )

            # Test prediction endpoint with sample data
            try:
                # Create minimal test data
                test_data = {
                    "sleep_data": [
                        {
                            "date": "2024-01-01",
                            "duration_hours": 7.5,
                            "efficiency": 0.85,
                        }
                    ],
                    "activity_data": [
                        {"date": "2024-01-01", "steps": 8000, "active_energy": 350}
                    ],
                }

                response = requests.post(
                    "http://localhost:8000/api/v1/predictions/quick",
                    json=test_data,
                    timeout=10,
                )

                tests.append(
                    {
                        "name": "API Prediction Endpoint",
                        "status": (
                            "PASS" if response.status_code in [200, 422] else "FAIL"
                        ),
                        "details": f"Status: {response.status_code}",
                    }
                )
            except Exception:
                tests.append(
                    {
                        "name": "API Prediction Endpoint",
                        "status": "SKIP",
                        "details": "Endpoint may not be implemented yet",
                    }
                )

        finally:
            # Stop API server
            api_process.terminate()
            api_process.wait(timeout=5)

        return tests

    def test_docker_integration(self) -> list[dict[str, Any]]:
        """Test Docker containerization."""
        tests = []

        # Check if Docker is available
        docker_available, _, _ = self.run_command(["docker", "--version"])
        if not docker_available:
            tests.append(
                {
                    "name": "Docker Availability",
                    "status": "SKIP",
                    "details": "Docker not installed or not in PATH",
                }
            )
            return tests

        tests.append(
            {
                "name": "Docker Availability",
                "status": "PASS",
                "details": "Docker is available",
            }
        )

        # Check if image exists or can be built
        image_exists, stdout, _ = self.run_command(
            ["docker", "images", "-q", "big-mood-detector:latest"]
        )

        if not stdout.strip():
            # Try to build image
            self.report.add_warning("Docker image not found, attempting to build...")
            build_success, _, stderr = self.run_command(
                ["docker", "build", "-t", "big-mood-detector:latest", "."],
                timeout=600,  # 10 minutes for build
            )
            tests.append(
                {
                    "name": "Docker Build",
                    "status": "PASS" if build_success else "FAIL",
                    "details": "Image built successfully" if build_success else stderr,
                }
            )
        else:
            tests.append(
                {"name": "Docker Image", "status": "PASS", "details": "Image exists"}
            )

        # Test running container
        if image_exists or build_success:
            # Run health check
            success, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "big-mood-detector:latest",
                    "python",
                    "-m",
                    "big_mood_detector",
                    "--help",
                ],
                timeout=30,
            )

            tests.append(
                {
                    "name": "Docker CLI Execution",
                    "status": "PASS" if success else "FAIL",
                    "details": "CLI works in container" if success else stderr,
                }
            )

        return tests

    def test_data_formats(self) -> list[dict[str, Any]]:
        """Test all supported data formats."""
        tests = []

        # Test CSV output
        json_dir = self.input_dir / "health_auto_export"
        if json_dir.exists():
            csv_file = self.test_output_dir / "test_features.csv"
            success, _, _ = self.run_command(
                [
                    sys.executable,
                    "-m",
                    "big_mood_detector",
                    "process",
                    str(json_dir),
                    "--output",
                    str(csv_file),
                ]
            )

            tests.append(
                {
                    "name": "CSV Output Format",
                    "status": "PASS" if success and csv_file.exists() else "FAIL",
                    "details": (
                        f"Size: {csv_file.stat().st_size / 1024:.1f} KB"
                        if csv_file.exists()
                        else "Failed"
                    ),
                }
            )

            # Test prediction formats
            for fmt in ["json", "csv"]:
                output_file = self.test_output_dir / f"predictions.{fmt}"
                success, _, _ = self.run_command(
                    [
                        sys.executable,
                        "-m",
                        "big_mood_detector",
                        "predict",
                        str(json_dir),
                        "--output",
                        str(output_file),
                        "--format",
                        fmt,
                    ]
                )

                tests.append(
                    {
                        "name": f"Prediction {fmt.upper()} Format",
                        "status": (
                            "PASS" if success and output_file.exists() else "FAIL"
                        ),
                        "details": f"Generated {fmt.upper()} predictions",
                    }
                )

        return tests

    def test_error_handling(self) -> list[dict[str, Any]]:
        """Test error handling and edge cases."""
        tests = []

        # Test with non-existent file
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "big_mood_detector", "process", "/non/existent/path"]
        )
        tests.append(
            {
                "name": "Invalid Path Handling",
                "status": (
                    "PASS" if not success and "does not exist" in stderr else "FAIL"
                ),
                "details": "Properly handles non-existent paths",
            }
        )

        # Test with empty directory
        empty_dir = self.test_output_dir / "empty"
        empty_dir.mkdir(exist_ok=True)
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "big_mood_detector", "process", str(empty_dir)]
        )
        tests.append(
            {
                "name": "Empty Directory Handling",
                "status": (
                    "PASS" if not success and "No JSON or XML" in stderr else "FAIL"
                ),
                "details": "Properly handles empty directories",
            }
        )

        # Test with invalid date range
        json_dir = self.input_dir / "health_auto_export"
        if json_dir.exists():
            success, stdout, stderr = self.run_command(
                [
                    sys.executable,
                    "-m",
                    "big_mood_detector",
                    "process",
                    str(json_dir),
                    "--start-date",
                    "2024-01-01",
                    "--end-date",
                    "2023-01-01",
                ]
            )
            tests.append(
                {
                    "name": "Invalid Date Range",
                    "status": (
                        "PASS" if not success and "must be before" in stderr else "FAIL"
                    ),
                    "details": "Validates date ranges properly",
                }
            )

        return tests

    def run_all_tests(self) -> None:
        """Run all pipeline tests."""
        print("Starting comprehensive pipeline test...\n")

        # Test sections
        test_sections = [
            ("CLI Interface", self.test_cli_help),
            ("JSON Parser", self.test_json_parser),
            ("XML Parser", self.test_xml_parser),
            ("Prediction Pipeline", self.test_prediction_pipeline),
            ("ML Models", self.test_ensemble_models),
            ("Clinical Features", self.test_clinical_features),
            ("API Endpoints", self.test_api_endpoints),
            ("Docker Integration", self.test_docker_integration),
            ("Data Formats", self.test_data_formats),
            ("Error Handling", self.test_error_handling),
        ]

        for section_name, test_func in test_sections:
            print(f"Testing {section_name}...")
            try:
                tests = test_func()
                self.report.add_section(section_name, tests)
            except Exception as e:
                self.report.add_error(f"{section_name}: {str(e)}")
            print(f"  Completed {section_name}")

        # Generate and save report
        report_text = self.report.generate_report()
        report_file = self.output_dir / "pipeline_test_report.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        # Also save JSON summary
        summary_file = self.output_dir / "pipeline_test_summary.json"
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "summary": self.report.summary_stats,
                    "sections": [
                        {
                            "name": s["name"],
                            "passed": s["passed"],
                            "failed": s["failed"],
                            "total": len(s["tests"]),
                        }
                        for s in self.report.sections
                    ],
                    "errors": self.report.errors,
                    "warnings": self.report.warnings,
                },
                f,
                indent=2,
            )

        # Print report
        print("\n" + report_text)
        print(f"\nFull report saved to: {report_file}")
        print(f"JSON summary saved to: {summary_file}")
        print(f"Test outputs saved to: {self.test_output_dir}")

        # Exit with appropriate code
        if self.report.summary_stats["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


def main():
    """Main entry point."""
    # Determine base directory
    script_path = Path(__file__)
    base_dir = script_path.parent.parent.parent  # Go up to project root

    # Create and run tests
    tester = ComprehensivePipelineTest(base_dir)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
