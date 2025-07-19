# Baseline Persistence Implementation Plan

## Current Situation
- We have `AdvancedFeatureEngineer` already calculating Z-scores with 30-day rolling windows
- We have `PersonalCalibrator` with `BaselineExtractor` for comprehensive baseline extraction
- Missing: Persistence layer to store and retrieve these baselines

## Decision: Start Simple, Plan for Scale

### Phase 4A: Simple File-Based Implementation (NOW)
**Why:** Following YAGNI and Uncle Bob's principles - start with the simplest thing that works

1. **FileBaselineRepository** 
   - JSON files per user: `baselines/{user_id}/baseline_history.json`
   - Stores last 10 baselines with timestamps
   - Simple, testable, works for MVP
   - Can handle hundreds of users easily

2. **Integration Points**
   - Inject repository into `AdvancedFeatureEngineer`
   - Save baselines after feature extraction
   - Load baselines at prediction time

3. **Timeline**: 1-2 hours to complete

### Phase 4B: Production-Grade Implementation (LATER)
**When:** After MVP validation, when we need:
- Multi-user concurrent access
- Real-time updates
- Audit trails
- Scale beyond 1000 users

#### Architecture (Your Excellent Suggestion)
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Feature         │────▶│ Baseline         │────▶│ TimescaleDB │
│ Engineer        │     │ Repository       │     │ (Offline)   │
└─────────────────┘     └──────────────────┘     └─────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────────┐     ┌─────────────┐
                        │ Feast Feature    │────▶│ Redis       │
                        │ Store            │     │ (Online)    │
                        └──────────────────┘     └─────────────┘
```

#### Implementation Steps
1. **Add TimescaleDB Extension**
   ```sql
   CREATE TABLE user_baseline_raw (
       user_id UUID,
       metric TEXT,
       value DOUBLE PRECISION,
       ts TIMESTAMPTZ
   );
   SELECT create_hypertable('user_baseline_raw', 'ts');
   ```

2. **Create Continuous Aggregates**
   ```sql
   CREATE MATERIALIZED VIEW user_baseline_30d
   WITH (timescaledb.continuous) AS
   SELECT
      user_id,
      metric,
      time_bucket('1 day', ts) AS as_of,
      AVG(value) AS mean,
      STDDEV_SAMP(value) AS std,
      COUNT(value) AS n
   FROM user_baseline_raw
   GROUP BY 1,2,3;
   ```

3. **Integrate Feast**
   - Define FeatureView for baselines
   - Configure Redis online store
   - Set up materialization jobs

## Migration Path

### From File to TimescaleDB
1. Keep same `BaselineRepositoryInterface`
2. Create `TimescaleBaselineRepository` implementation
3. Migrate existing JSON files with a script
4. Switch DI registration
5. No other code changes needed!

## Benefits of This Approach
- **Start Simple**: Get working baseline persistence today
- **Clean Architecture**: Repository pattern allows easy swapping
- **No Wasted Work**: File-based repo helps us understand the domain
- **Clear Upgrade Path**: When we need scale, we know exactly what to do

## Recommendation
Let's implement the simple file-based solution now (Phase 4A) and document the production path (Phase 4B) for later. This follows:
- YAGNI: We don't need TimescaleDB yet
- Uncle Bob: Simple, clean, testable
- Incremental: We can upgrade without changing domain code

## Next Steps
1. ✅ Complete FileBaselineRepository implementation
2. ✅ Integrate with AdvancedFeatureEngineer
3. ✅ Add tests for persistence in prediction pipeline
4. 📝 Document TimescaleDB migration for future
5. 🚀 Ship MVP with working baselines!