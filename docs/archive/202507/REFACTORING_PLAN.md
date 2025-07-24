# Refactoring Plan

## 1. Split ClinicalInterpreter (Current Task)

### Proposed Structure:
```
domain/services/
├── clinical_interpreter.py (main orchestrator, ~200 lines)
├── episode_interpreter.py (depression/mania/mixed interpretation)
├── biomarker_interpreter.py (sleep/activity/circadian)
├── treatment_recommender.py (recommendations & rules)
├── risk_analyzer.py (early warnings, trends, personalization)
└── clinical_thresholds.py (already done ✅)
```

### Benefits:
- Single Responsibility Principle
- Easier testing
- Better maintainability
- Allows parallel development

## 2. Add Regulatory Logging

### Components to Add:
- `audit_middleware.py` - Captures all API requests/responses
- `clinical_logger.py` - Specialized logger for clinical decisions
- `audit_models.py` - Database models for audit trail
- Update all endpoints to log clinical interpretations

### Reference Implementation:
```python
# Example audit event
{
    "timestamp": "2024-07-16T15:30:45Z",
    "user_id": "clinician_123",
    "action": "clinical_interpretation",
    "resource": "/api/v1/clinical/interpret",
    "patient_id": "hash_12345",  # Never log raw patient ID
    "decision": {
        "risk_level": "moderate",
        "episode_type": "depressive",
        "dsm5_criteria_met": true
    },
    "request_id": "req_abc123",
    "ip_address": "192.168.1.100"
}
```

## 3. Authentication Middleware

### Options:
1. **FastAPI-Users** (recommended for quick start)
   - Full user management
   - JWT + OAuth support
   - Well documented

2. **Custom JWT** (for more control)
   - Based on fastapi-jwt-auth reference
   - Lighter weight
   - Custom claims for clinical roles

## 4. Future: tsfresh Integration

### When to Implement:
- After core refactoring complete
- When we need advanced time series features
- For research/ML improvement phase

### What it Adds:
- 750+ time series features
- Statistical significance testing
- Feature selection algorithms
- Better circadian rhythm analysis

## Execution Order:

1. **Today**: Continue splitting ClinicalInterpreter
2. **Next**: Add regulatory logging (high priority for compliance)
3. **Then**: Authentication (builds on logging)
4. **Later**: tsfresh for enhanced features

## Success Metrics:
- [ ] No file > 400 lines
- [ ] All clinical decisions logged
- [ ] 100% test coverage on critical paths
- [ ] Authentication on all endpoints
- [ ] Audit trail passes compliance review