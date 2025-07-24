#!/usr/bin/env python3
"""
Documentation inventory script - finds all docs and their status.
"""
import json
import pathlib
import re
import subprocess
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs"
OUTPUT = []

def get_last_modified(file_path):
    """Get last git modification date."""
    try:
        result = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--", str(file_path)],
            stderr=subprocess.DEVNULL
        ).strip()
        return datetime.fromtimestamp(int(result)).date().isoformat()
    except:
        return "unknown"

def analyze_doc(md_path):
    """Analyze a single markdown file."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    
    return {
        "path": str(md_path.relative_to(ROOT)),
        "size_kb": round(md_path.stat().st_size / 1024, 1),
        "last_modified": get_last_modified(md_path),
        "has_v0_2": "v0.2" in text,
        "has_v0_3": "v0.3" in text,
        "has_todo": bool(re.search(r"TODO|FIXME|HACK|XXX", text)),
        "mentions_tensorflow": "tensorflow" in text.lower(),
        "mentions_ensemble": "ensemble" in text.lower(),
        "line_count": len(text.splitlines()),
        "title": text.split('\n')[0].strip('# ') if text else "NO TITLE"
    }

# Find all markdown files
for md in sorted(ROOT.rglob("*.md")):
    if not any(skip in str(md) for skip in ["node_modules", ".git"]):
        OUTPUT.append(analyze_doc(md))

# Print summary
print(f"\n📊 DOCUMENTATION INVENTORY - {len(OUTPUT)} files\n")
print(f"{'Path':<60} {'Size':<8} {'Modified':<12} {'Issues'}")
print("=" * 100)

for doc in sorted(OUTPUT, key=lambda x: x["path"]):
    issues = []
    if doc["has_v0_2"]: issues.append("OLD-v0.2")
    if doc["has_v0_3"]: issues.append("OLD-v0.3")
    if doc["mentions_tensorflow"]: issues.append("TF")
    if doc["mentions_ensemble"] and "ensemble" not in doc["path"]: issues.append("ENSEMBLE")
    if doc["has_todo"]: issues.append("TODO")
    if doc["size_kb"] < 0.5: issues.append("STUB")
    if "DRAFT" in doc["title"]: issues.append("DRAFT")
    
    status = "🔴" if len(issues) > 2 else "🟡" if issues else "🟢"
    print(f"{status} {doc['path']:<58} {doc['size_kb']:>6.1f}KB {doc['last_modified']:<12} {','.join(issues)}")

# Save detailed JSON
with open(ROOT.parent / "docs_inventory.json", "w") as f:
    json.dump(OUTPUT, f, indent=2)
    
print(f"\n📁 Full inventory saved to docs_inventory.json")
print(f"\n📈 Summary:")
print(f"  - Outdated (v0.2/v0.3): {sum(1 for d in OUTPUT if d['has_v0_2'] or d['has_v0_3'])}")
print(f"  - Mentions TensorFlow: {sum(1 for d in OUTPUT if d['mentions_tensorflow'])}")
print(f"  - Has TODOs: {sum(1 for d in OUTPUT if d['has_todo'])}")
print(f"  - Small stubs (<0.5KB): {sum(1 for d in OUTPUT if d['size_kb'] < 0.5)}")