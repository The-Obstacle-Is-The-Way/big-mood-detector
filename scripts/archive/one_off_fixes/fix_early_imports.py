#!/usr/bin/env python3
"""Fix early imports in test files to restore coverage."""

from pathlib import Path

# Comprehensive list of all test files with early imports
FILES_TO_FIX = [
    # conftest files
    "tests/conftest.py",
    # Unit tests
    "tests/unit/infrastructure/test_json_parsers.py",
    "tests/unit/infrastructure/test_heart_rate_parser.py",
    "tests/unit/infrastructure/test_activity_parser.py",
    "tests/unit/infrastructure/parsers/test_flexible_json_parser.py",
    "tests/unit/domain/test_time_period.py",
    "tests/unit/core/test_paths.py",
    "tests/unit/test_date_range_filtering.py",
    "tests/unit/api/test_clinical_endpoint.py",
    "tests/unit/application/test_aggregation_pipeline_sleep.py",
    "tests/unit/application/test_pipeline_ensemble_integration.py",
    "tests/unit/application/test_aggregation_pipeline_fix_verification.py",
    "tests/unit/application/test_no_magic_hr_defaults.py",
    "tests/unit/application/test_mood_pipeline_baseline_di.py",
    "tests/unit/domain/services/test_incremental_stats_property.py",
    "tests/unit/infrastructure/test_di_container_phase2.py",
    "tests/unit/infrastructure/test_user_repository.py",
    "tests/unit/infrastructure/test_local_cache_repository.py",
    "tests/unit/infrastructure/test_file_baseline_repository.py",
    "tests/unit/infrastructure/test_streaming_parser.py",
    "tests/unit/infrastructure/test_db_setup.py",
    "tests/unit/infrastructure/test_models.py",
    "tests/unit/infrastructure/test_heart_repositories.py",
    "tests/unit/infrastructure/test_sleep_parser.py",
    "tests/unit/infrastructure/test_sleep_repository.py",
    "tests/unit/infrastructure/test_activity_repositories.py",
    "tests/unit/infrastructure/test_di_container.py",
    "tests/unit/infrastructure/test_xml_parsers.py",
    "tests/unit/test_file_hash_repository.py",
    # Integration tests
    "tests/integration/test_json_xml_feature_parity.py",
    "tests/integration/test_xml_json_parity.py",
    "tests/integration/test_parser_factory.py",
    "tests/integration/test_full_pipeline_integration.py",
    "tests/integration/test_ensemble_predictions.py",
    "tests/integration/test_json_flexible_parser.py",
    "tests/integration/test_real_xml_parsing.py",
    # Domain tests
    "tests/unit/domain/test_mood_category.py",
    "tests/unit/domain/test_heart_rate_record.py",
    "tests/unit/domain/test_activity_record.py",
    "tests/unit/domain/test_user.py",
    "tests/unit/domain/test_clinical_features.py",
    "tests/unit/domain/test_sleep_record.py",
    "tests/unit/domain/test_mood_prediction.py",
    "tests/unit/domain/test_file_hash.py",
    # Services tests
    "tests/unit/domain/services/test_heart_aggregator.py",
    "tests/unit/domain/services/test_sleep_aggregator.py",
    "tests/unit/domain/services/test_activity_aggregator.py",
    "tests/unit/domain/services/test_circadian_rhythm_analyzer.py",
    "tests/unit/domain/services/test_clinical_thresholds.py",
    "tests/unit/domain/services/test_mood_trend_analyzer.py",
    "tests/unit/domain/services/test_feature_extraction_service.py",
    "tests/unit/domain/services/test_data_validator.py",
    # Application tests
    "tests/unit/application/test_process_health_data_use_case.py",
    "tests/unit/application/test_mood_prediction_pipeline.py",
    "tests/unit/application/test_predict_mood_use_case.py",
    "tests/unit/application/test_aggregation_pipeline.py",
    # API tests
    "tests/unit/api/test_health_data_api.py",
    "tests/unit/api/test_rate_limiting.py",
    "tests/unit/api/test_cors_middleware.py",
    "tests/unit/api/test_monitoring_api.py",
    "tests/unit/api/test_prediction_endpoints.py",
    # CLI tests
    "tests/unit/cli/test_main_cli.py",
    "tests/unit/cli/test_commands.py",
]


def move_imports_to_class(file_path):
    """Move top-level imports inside the first test class or fixture."""
    try:
        content = Path(file_path).read_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    lines = content.split("\n")

    # Find imports to move
    imports_to_move = []
    import_indices = []

    for i, line in enumerate(lines):
        if line.strip().startswith(
            ("from big_mood_detector", "import big_mood_detector")
        ):
            # Check if it's at module level (not indented)
            if not line.startswith((" ", "\t")):
                imports_to_move.append(line)
                import_indices.append(i)

    if not imports_to_move:
        return False

    # Special handling for conftest.py
    if file_path.endswith("conftest.py"):
        # For conftest, move imports inside fixtures
        for idx in reversed(import_indices):
            lines.pop(idx)

        # Find first fixture
        for i, line in enumerate(lines):
            if line.strip().startswith("@pytest.fixture"):
                # Find the function definition after the decorator
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("def "):
                        # Insert imports at the beginning of the fixture
                        for imp in imports_to_move:
                            lines.insert(j + 1, "    " + imp)
                        Path(file_path).write_text("\n".join(lines))
                        return True
        return False

    # Remove the imports from their current location
    for idx in reversed(import_indices):
        lines.pop(idx)

    # Find the first test method, fixture, or setup method
    first_method_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("def test_")
            or stripped.startswith("def setup")
            or stripped.startswith("@pytest.fixture")
        ):
            if stripped.startswith("@"):
                # Find the actual function definition after decorator
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("def "):
                        first_method_idx = j
                        break
            else:
                first_method_idx = i
            break

    if first_method_idx is None:
        # No test method found, put imports at the beginning of the first class
        for i, line in enumerate(lines):
            if line.strip().startswith("class "):
                # Find first method in the class
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("def "):
                        # Insert imports at the beginning of this method
                        indent = "        "  # 8 spaces for method inside class
                        for imp in imports_to_move:
                            lines.insert(j + 1, indent + imp)
                        Path(file_path).write_text("\n".join(lines))
                        return True
                break
    else:
        # Determine proper indentation
        method_line = lines[first_method_idx]
        base_indent = len(method_line) - len(method_line.lstrip())
        indent = " " * (base_indent + 4)

        # Insert at the beginning of the first test method
        insert_pos = first_method_idx + 1
        # Skip any docstring
        if insert_pos < len(lines) and lines[insert_pos].strip().startswith('"""'):
            while insert_pos < len(lines) and not lines[insert_pos].strip().endswith(
                '"""'
            ):
                insert_pos += 1
            insert_pos += 1

        for imp in imports_to_move:
            lines.insert(insert_pos, indent + imp)

    # Write back
    Path(file_path).write_text("\n".join(lines))
    return True


# Fix each file
for file_path in FILES_TO_FIX:
    if Path(file_path).exists():
        if move_imports_to_class(file_path):
            print(f"Fixed: {file_path}")
        else:
            print(f"No changes needed: {file_path}")
    else:
        print(f"File not found: {file_path}")
