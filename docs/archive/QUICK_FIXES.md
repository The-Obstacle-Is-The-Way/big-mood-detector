# Quick Fixes for Common Issues

## 1. Large XML File Timeout

**Problem:** Processing Apple Health export.xml times out after 2 minutes

**Temporary Workarounds:**

### Option A: Use Health Auto Export App (Recommended)
1. Download "Health Auto Export" from App Store
2. Export data as JSON instead of XML
3. Process JSON files: `python3 src/big_mood_detector/main.py process data/`

### Option B: Split Large XML (Advanced)
```bash
# Split by date range (requires xmlstarlet)
xmlstarlet sel -t -c "//Record[@startDate > '2024-01-01']" export.xml > export_2024.xml
```

### Option C: Increase Timeout (May Still Fail)
```python
# In src/big_mood_detector/main.py, add:
import sys
sys.setrecursionlimit(10000)  # Increase if needed
```

## 2. Docker Won't Start

**Problem:** Security validation fails in production mode

**Quick Fix:**
```bash
# Create docker-compose.dev.yml
cat > docker-compose.dev.yml << 'EOF'
services:
  app:
    extends:
      file: docker-compose.yml
      service: app
    environment:
      - ENVIRONMENT=development  # Bypass security check
      - SECRET_KEY=dev-secret-key-only
      - API_KEY_SALT=dev-salt-only
EOF

# Run with:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## 3. API Server Model Path Errors

**Problem:** Models not found at `/src/model_weights/...`

**Fix:** Models are in the correct location, these are non-fatal warnings.

## 4. No Progress Indication

**Problem:** No feedback during long operations

**Workaround:** Watch log output
```bash
# In another terminal:
tail -f logs/*.log | grep -E "(processed|complete|error)"
```

## 5. Memory Issues with Large Files

**Problem:** Process killed or system freezes

**Fix:** Limit memory usage
```bash
# Linux/Mac
ulimit -v 4000000  # Limit to 4GB
python3 src/big_mood_detector/main.py process export.xml

# Or use Docker with memory limits
docker run -m 4g big-mood-detector process /data/export.xml
```

## Recommended Approach for New Users

1. **Start with JSON data** from Health Auto Export app
2. **Use development mode** for Docker
3. **Process recent data only** (last 60-90 days)
4. **Monitor memory usage** during processing

## When to Wait for v0.3.0

If you experience:
- XML files over 200MB timing out
- Need true ensemble predictions
- Want current state assessment (not just 24hr forecast)
- Require production Docker deployment

---

*These are temporary workarounds. Proper fixes coming in v0.3.0*