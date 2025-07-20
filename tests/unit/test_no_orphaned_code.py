"""
Test to ensure no orphaned/duplicate code remains in the codebase.

This prevents dead code from accumulating and confusing developers.
"""
from pathlib import Path


class TestNoOrphanedCode:
    """Ensure codebase hygiene - no dead code allowed!"""

    def test_no_clinical_decision_engine_duplicate(self):
        """Ensure clinical_decision_engine.py doesn't exist (it's a duplicate)."""
        src_path = Path("src")

        # Search for the duplicate file
        orphan_files = list(src_path.rglob("*clinical_decision_engine.py"))

        assert len(orphan_files) == 0, (
            f"Found orphaned clinical_decision_engine.py at: {orphan_files}. "
            "This is a duplicate of clinical_interpreter.py and should be deleted!"
        )

    def test_no_orphaned_test_only_services(self):
        """
        Ensure services that exist only in tests are properly marked or removed.

        Based on audit, these services should either:
        1. Be integrated into the main pipeline
        2. Be marked as test fixtures
        3. Be deleted if truly unused
        """
        # Services that should be actively used (not just in tests)
        services_that_should_be_integrated = [
            "feature_engineering_orchestrator",
            "baseline_extractor",
            "personal_calibrator",
        ]

        src_path = Path("src")

        for service_name in services_that_should_be_integrated:
            # Find the service file
            service_files = list(src_path.rglob(f"*{service_name}.py"))

            if service_files:
                # Check if it's imported anywhere outside tests
                non_test_imports = []
                for py_file in src_path.rglob("*.py"):
                    if "test" not in str(py_file):
                        content = py_file.read_text()
                        if service_name in content and "import" in content:
                            non_test_imports.append(py_file)

                # Special case: feature_engineering_orchestrator should be used
                if service_name == "feature_engineering_orchestrator":
                    assert len(non_test_imports) > 0, (
                        f"{service_name} exists but is not imported outside tests! "
                        "It should replace direct clinical_extractor calls in the pipeline."
                    )

    def test_no_todo_fixme_hack_comments(self):
        """Ensure we're not leaving HACK/FIXME comments in production code."""
        src_path = Path("src")

        hack_patterns = ["HACK:", "FIXME:", "XXX:", "TODO: CRITICAL"]
        found_hacks = []

        for py_file in src_path.rglob("*.py"):
            if "test" not in str(py_file):
                content = py_file.read_text()
                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern in hack_patterns:
                        if pattern in line:
                            found_hacks.append(f"{py_file}:{line_num} - {line.strip()}")

        assert len(found_hacks) == 0, (
            f"Found {len(found_hacks)} HACK/FIXME comments:\n" +
            "\n".join(found_hacks[:10])  # Show first 10
        )
