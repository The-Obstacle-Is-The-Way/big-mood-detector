#!/usr/bin/env python3
"""
Generate comprehensive license report for all dependencies.

This script creates a detailed report of all third-party licenses
used in the project, ensuring compliance with open source requirements.
"""

import subprocess
import sys
from pathlib import Path


def generate_license_report():
    """Generate license report using pip-licenses."""

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    licenses_dir = project_root / "LICENSES"

    print("Generating license report...")

    # Install pip-licenses if not present
    try:
        import piplicenses  # noqa: F401
    except ImportError:
        print("Installing pip-licenses...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pip-licenses"], check=True
        )

    # Generate markdown report
    output_file = licenses_dir / "python-dependencies.md"

    cmd = [
        sys.executable,
        "-m",
        "piplicenses",
        "--format=markdown",
        "--with-urls",
        "--with-description",
        "--output-file",
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"License report generated: {output_file}")

        # Add header to the file
        content = output_file.read_text()
        header = """# Python Dependencies License Report

This file is auto-generated. To regenerate, run: `python scripts/generate_licenses.py`

Last generated: {}

---

""".format(subprocess.run(["date"], capture_output=True, text=True).stdout.strip())

        output_file.write_text(header + content)

        # Also generate a CSV for easier processing
        csv_file = licenses_dir / "python-dependencies.csv"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piplicenses",
                "--format=csv",
                "--with-urls",
                "--output-file",
                str(csv_file),
            ]
        )
        print(f"CSV report generated: {csv_file}")

    else:
        print(f"Error generating report: {result.stderr}")
        sys.exit(1)


def check_license_compatibility():
    """Check for license compatibility issues."""

    # Known problematic licenses for Apache 2.0
    incompatible = ["GPL", "AGPL", "LGPL"]  # Simplified check

    project_root = Path(__file__).parent.parent
    csv_file = project_root / "LICENSES" / "python-dependencies.csv"

    if not csv_file.exists():
        print("No CSV file found. Run generate_license_report first.")
        return

    print("\nChecking license compatibility...")

    issues = []
    with open(csv_file) as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            license_name = row.get("License", "")
            package = row.get("Name", "")

            for incomp in incompatible:
                if (
                    incomp in license_name
                    and "LGPL with exceptions" not in license_name
                ):
                    issues.append(f"{package}: {license_name}")

    if issues:
        print("⚠️  Potential license compatibility issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious license compatibility issues found.")


def create_attribution_file():
    """Create attribution file for binary distributions."""

    project_root = Path(__file__).parent.parent
    attribution_file = project_root / "LICENSES" / "ATTRIBUTION.txt"

    content = """Big Mood Detector - Third Party Attributions

This file contains attributions for third-party software included in
Big Mood Detector binary distributions.

================================================================================

1. XGBoost
   Copyright (c) 2015 by Contributors
   Licensed under Apache License 2.0
   https://github.com/dmlc/xgboost

2. TensorFlow
   Copyright 2015 The TensorFlow Authors
   Licensed under Apache License 2.0
   https://github.com/tensorflow/tensorflow

3. NumPy
   Copyright (c) 2005-2024, NumPy Developers
   Licensed under BSD 3-Clause License
   https://numpy.org/

4. Pandas
   Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
   Licensed under BSD 3-Clause License
   https://pandas.pydata.org/

5. scikit-learn
   Copyright (c) 2007-2024 The scikit-learn developers
   Licensed under BSD 3-Clause License
   https://scikit-learn.org/

6. FastAPI
   Copyright (c) 2018 Sebastián Ramírez
   Licensed under MIT License
   https://fastapi.tiangolo.com/

7. Pydantic
   Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors
   Licensed under MIT License
   https://pydantic.dev/

8. Redis-py
   Copyright (c) 2022 Redis Inc.
   Licensed under MIT License
   https://github.com/redis/redis-py

9. SQLAlchemy
   Copyright (c) 2005-2024 Michael Bayer and contributors
   Licensed under MIT License
   https://www.sqlalchemy.org/

10. Prometheus Client
    Copyright 2015 The Prometheus Authors
    Licensed under Apache License 2.0
    https://github.com/prometheus/client_python

================================================================================

For complete license texts, see the individual LICENSE files in this directory.
"""

    attribution_file.write_text(content)
    print(f"\nAttribution file created: {attribution_file}")


if __name__ == "__main__":
    generate_license_report()
    check_license_compatibility()
    create_attribution_file()

    print("\n✅ License documentation complete!")
    print("Remember to:")
    print("1. Review the generated reports for accuracy")
    print("2. Update ATTRIBUTION.txt when adding new dependencies")
    print("3. Include LICENSES/ directory in distributions")
    print("4. Cite research papers when publishing results")
