#!/bin/bash
# Archive outdated documentation with proper banner

ARCHIVE_DIR="docs/archive/202507"
ARCHIVE_DATE="2025-07-24"

# Function to add archive banner and move file
archive_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "Archiving: $file"
        # Add banner at top
        {
            echo "> **Archived $ARCHIVE_DATE** – No longer relevant as of v0.4.0"
            echo ""
            cat "$file"
        } > "$file.tmp"
        mv "$file.tmp" "$ARCHIVE_DIR/$(basename "$file")"
    fi
}

# Archive outdated ensemble docs
archive_file "docs/models/ensemble/CURRENT_ENSEMBLE_EXPLANATION.md"
archive_file "docs/architecture/bulletproof_pipeline_summary.md"
archive_file "docs/architecture/ensemble_weights_config.md"

# Archive old API structure docs (not actual API docs)
for f in docs/api/python/*.md; do
    archive_file "$f"
done

# Archive all phase planning
find docs/archive/phase* -name "*.md" -type f | while read f; do
    mv "$f" "$ARCHIVE_DIR/$(basename "$f")"
done

# Archive old version docs
for f in docs/archive/*V0.2*.md docs/archive/*V0.3*.md docs/archive/*v0_2*.md docs/archive/*v0_3*.md; do
    [ -f "$f" ] && mv "$f" "$ARCHIVE_DIR/$(basename "$f")"
done

# Archive old roadmaps and plans
for f in docs/archive/ROADMAP*.md docs/archive/*_PLAN.md; do
    [ -f "$f" ] && mv "$f" "$ARCHIVE_DIR/$(basename "$f")"
done

echo "✅ Archived $(ls -1 $ARCHIVE_DIR | wc -l) files to $ARCHIVE_DIR"