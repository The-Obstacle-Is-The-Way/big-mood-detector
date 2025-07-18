# Production Deployment Notes

## Thread Safety and Multi-Worker Considerations

### Model Loading
- Models are loaded once at startup via FastAPI's `startup` event
- Stored in `app.state` for sharing across requests
- Each worker process gets its own copy (no shared memory)

### Memory Requirements
- XGBoost models: ~50MB per worker
- PAT-M model: ~600MB per worker (TensorFlow)
- Total per worker: ~650MB with ensemble

### Recommended Deployment

#### Single Machine
```bash
# For CPU-bound workloads (4 workers on 4-core machine)
gunicorn big_mood_detector.interfaces.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --preload
```

#### Multi-Machine (Kubernetes)
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Thread Safety Issues

1. **TensorFlow Global State**
   - TF uses global singleton for GPU/CPU device management
   - Can cause deadlocks with ThreadPoolExecutor
   - Solution: Use `TF_FORCE_GPU_ALLOW_GROWTH=true`

2. **XGBoost Thread Pool**
   - XGBoost creates internal OpenMP threads
   - Can conflict with Python's ThreadPoolExecutor
   - Solution: Set `OMP_NUM_THREADS=1` for API workers

3. **Recommended Environment Variables**
```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TF logging
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Prevent GPU memory issues
export OMP_NUM_THREADS=1  # Prevent XGBoost threading conflicts
export OPENBLAS_NUM_THREADS=1  # Prevent NumPy threading conflicts
```

### Alternative: Process Pool
For heavy workloads, consider using a process pool:

```python
# In ensemble_orchestrator.py
from concurrent.futures import ProcessPoolExecutor

# Replace ThreadPoolExecutor with:
self.executor = ProcessPoolExecutor(max_workers=2)
```

### Monitoring
- Add Prometheus metrics for prediction latency
- Monitor memory usage per worker
- Alert on prediction timeouts > 1s

### Rate Limiting
For public endpoints, add rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/predict/ensemble")
@limiter.limit("10/minute")
async def predict_mood_ensemble(...):
    ...
```