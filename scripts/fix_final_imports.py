#!/usr/bin/env python3
"""Fix the final set of early imports identified by grep."""

import re
from pathlib import Path

# All files with early imports found via grep
FILES_TO_FIX = [
    "tests/unit/application/test_predict_mood_ensemble_use_case.py",
    "tests/unit/application/test_prediction_interpreter.py",
    "tests/unit/application/test_clinical_feature_extraction.py",
    "tests/unit/application/test_sleep_duration_regression.py",
    "tests/unit/application/test_aggregation_pipeline_sleep_bug.py",
    "tests/unit/application/test_pipeline_personal_calibration.py",
    "tests/unit/application/test_data_structure_consolidation.py",
    "tests/unit/application/test_data_parsing_service.py",
    "tests/unit/infrastructure/settings/test_settings_config.py",
    "tests/unit/infrastructure/test_pat_equivalence.py",
    "tests/unit/infrastructure/parsers/test_streaming_xml_date_filter.py",
    "tests/unit/infrastructure/test_streaming_adapter.py",
    "tests/unit/infrastructure/test_baseline_repository_factory.py",
    "tests/unit/infrastructure/test_baseline_repository_hr_hrv.py",
    "tests/unit/infrastructure/test_repository_dependency_injection.py",
    "tests/unit/infrastructure/test_file_activity_repository.py",
    "tests/unit/infrastructure/test_pat_model.py",
    "tests/unit/infrastructure/security/test_privacy.py",
    "tests/unit/infrastructure/repositories/test_timescale_baseline_hr_hrv.py",
    "tests/unit/infrastructure/repositories/test_sqlite_episode_repository.py",
    "tests/unit/infrastructure/repositories/test_timescale_concurrency.py",
    "tests/unit/infrastructure/repositories/test_timescale_upsert_fix.py",
    "tests/unit/infrastructure/test_xgboost_models.py",
    "tests/unit/infrastructure/test_di_container_baseline_repository.py",
    "tests/unit/infrastructure/test_timescale_baseline_repository.py",
    "tests/unit/infrastructure/test_repository_factory_integration.py",
    "tests/unit/infrastructure/test_file_sleep_repository.py",
    "tests/unit/infrastructure/test_file_heart_rate_repository.py",
    "tests/unit/domain/test_clinical_feature_extractor.py",
    "tests/unit/domain/test_biomarker_interpreter.py",
    "tests/unit/domain/test_advanced_feature_engineering_with_persistence.py",
    "tests/unit/domain/test_temporal_feature_calculator.py",
    "tests/unit/domain/test_clinical_feature_extractor_with_calibration.py",
    "tests/unit/domain/test_clinical_interpreter_with_config.py",
    "tests/unit/domain/test_sleep_feature_calculator.py",
    "tests/unit/domain/test_sleep_aggregator.py",
    "tests/unit/domain/test_sparse_data_handler.py",
    "tests/unit/domain/test_dlmo_calculator.py",
    "tests/unit/domain/test_feature_engineering_orchestrator.py",
    "tests/unit/domain/test_baseline_repository_interface.py",
    "tests/unit/domain/test_circadian_feature_calculator.py",
    "tests/unit/domain/test_clinical_interpreter.py",
    "tests/unit/domain/test_feature_orchestrator_interface.py",
    "tests/unit/domain/test_heart_rate_aggregator.py",
    "tests/unit/domain/test_advanced_feature_engineering.py",
    "tests/unit/domain/test_episode_interpreter.py",
    "tests/unit/domain/test_sleep_math_debug.py",
    "tests/unit/domain/test_activity_sequence_extractor.py",
    "tests/unit/domain/test_mood_predictor.py",
    "tests/unit/domain/test_feature_extraction_service.py",
    "tests/unit/domain/test_baseline_repository_integration.py",
    "tests/unit/domain/test_circadian_rhythm_analyzer.py",
    "tests/unit/domain/test_interpolation_strategies.py",
    "tests/unit/domain/test_activity_feature_calculator.py",
    "tests/unit/domain/test_early_warning_detector.py",
    "tests/unit/domain/test_clinical_thresholds.py",
    "tests/unit/domain/test_risk_level_assessor.py",
    "tests/unit/domain/test_clinical_interpreter_refactored.py",
    "tests/unit/domain/test_sleep_window_analyzer.py",
    "tests/unit/domain/test_pat_sequence_builder.py",
    "tests/unit/domain/test_user_baseline_hr_hrv.py",
    "tests/unit/domain/test_activity_aggregator.py",
    "tests/unit/domain/services/test_sleep_aggregator_apple_3pm.py",
    "tests/unit/domain/services/test_sleep_aggregator_midnight.py",
    "tests/unit/domain/services/test_sleep_aggregator_regression.py",
    "tests/unit/domain/test_episode_mapper.py",
    "tests/unit/domain/test_treatment_recommender.py",
    "tests/unit/domain/test_dsm5_criteria_evaluator.py",
    "tests/unit/domain/test_clinical_interpreter_migration.py",
    "tests/unit/test_progress_indication.py",
    "tests/unit/interfaces/cli/test_user_id_validation.py",
    "tests/unit/interfaces/cli/labeling/test_progress_simple.py",
    "tests/unit/interfaces/api/test_input_validation.py",
    "tests/unit/interfaces/test_cli.py",
    "tests/integration/test_baseline_persistence_e2e.py",
    "tests/integration/pipeline/test_full_pipeline.py",
    "tests/integration/test_progress_indication_integration.py",
    "tests/integration/features/test_advanced_feature_pipeline.py",
    "tests/integration/test_api_integration.py",
    "tests/integration/test_real_data_sleep_math.py",
    "tests/integration/data_processing/test_streaming_large_files.py",
    "tests/integration/data_processing/test_health_data_integration.py",
    "tests/integration/data_processing/test_real_data_integration.py",
    "tests/integration/data_processing/test_dual_pipeline_validation.py",
    "tests/integration/data_processing/test_xml_date_filtering_integration.py",
    "tests/integration/data_processing/test_end_to_end_data_processing.py",
    "tests/integration/storage/test_repository_integration.py",
    "tests/integration/test_openapi_contract.py",
    "tests/integration/ml/test_xgboost_real_models.py",
    "tests/integration/test_hr_hrv_e2e.py",
    "tests/integration/api/test_features_endpoint_activity.py",
    "tests/integration/test_baseline_persistence_pipeline.py",
    "tests/integration/test_di_smoke.py",
    "tests/integration/test_memory_bounds.py",
    "tests/integration/test_ensemble_pipeline_activity.py",
    "tests/e2e/test_label_workflow.py",
]


def move_imports_in_file(file_path: Path) -> bool:
    """Move early imports in a single file."""
    try:
        content = file_path.read_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    lines = content.split('\n')

    # Find all early imports
    imports_to_move = []
    import_indices = []

    for i, line in enumerate(lines):
        # Check for big_mood_detector imports at module level
        if re.match(r'^(from big_mood_detector|import big_mood_detector)', line):
            imports_to_move.append(line)
            import_indices.append(i)

    if not imports_to_move:
        return False

    # Remove imports from their original positions
    for idx in reversed(import_indices):
        lines.pop(idx)

    # Find first test function or fixture
    insert_idx = None
    base_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for functions, fixtures, or class methods
        if (stripped.startswith('def test_') or
            stripped.startswith('def setup') or
            stripped.startswith('@pytest.fixture') or
            stripped == 'def test_'):

            if stripped.startswith('@'):
                # Find actual function after decorator
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('def '):
                        insert_idx = j
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
            else:
                insert_idx = i
                base_indent = len(line) - len(line.lstrip())

            if insert_idx is not None:
                break

    # If no test function found, look for first class
    if insert_idx is None:
        for i, line in enumerate(lines):
            if line.strip().startswith('class Test'):
                # Find first method in class
                for j in range(i+1, len(lines)):
                    if lines[j].strip().startswith('def '):
                        insert_idx = j
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                break

    if insert_idx is None:
        print(f"Warning: No suitable insertion point found in {file_path}")
        return False

    # Calculate proper indentation
    indent = ' ' * (base_indent + 4)

    # Find where to insert (after function definition, skip docstring)
    insert_pos = insert_idx + 1
    if insert_pos < len(lines) and lines[insert_pos].strip().startswith('"""'):
        # Skip docstring
        insert_pos += 1
        while insert_pos < len(lines) and '"""' not in lines[insert_pos]:
            insert_pos += 1
        if insert_pos < len(lines):
            insert_pos += 1

    # Insert the imports
    for imp in imports_to_move:
        lines.insert(insert_pos, indent + imp)
        insert_pos += 1

    # Write back
    file_path.write_text('\n'.join(lines))
    return True


def main():
    """Fix all identified files."""
    fixed_count = 0

    for file_path_str in FILES_TO_FIX:
        file_path = Path(file_path_str)
        if file_path.exists():
            if move_imports_in_file(file_path):
                print(f"Fixed: {file_path_str}")
                fixed_count += 1
        else:
            print(f"File not found: {file_path_str}")

    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()
