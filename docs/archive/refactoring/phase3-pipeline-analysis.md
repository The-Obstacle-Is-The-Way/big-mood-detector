# Phase 3.1: MoodPredictionPipeline Analysis

## Overview
The `MoodPredictionPipeline` is a 957-line god class that violates multiple SOLID principles and Clean Architecture boundaries. It orchestrates the complete mood prediction process but has accumulated too many responsibilities.

## Current Issues

### 1. Single Responsibility Principle Violations
The pipeline currently handles:
- Data parsing orchestration (XML and JSON)
- Feature extraction orchestration
- Statistical calculations (mean, std, z-scores)
- Data quality assessment
- File I/O operations
- Export functionality
- Rolling window management
- Date range filtering
- Error handling and reporting

### 2. Open/Closed Principle Violations
- Hard-coded feature calculations in `_calculate_daily_metrics`
- Direct instantiation of parsers instead of dependency injection
- Tight coupling to specific file formats (XML/JSON)
- No extension points for new feature types

### 3. Dependency Inversion Principle Violations
- Direct dependencies on concrete implementations:
  ```python
  self.sleep_parser = SleepJSONParser()
  self.activity_parser = ActivityJSONParser()
  self.xml_parser = StreamingXMLParser()
  ```
- Should depend on abstractions (interfaces)

### 4. Interface Segregation Principle Violations
- The `DailyFeatures` dataclass has 36 fields - clients are forced to depend on all
- No smaller, focused interfaces for different feature domains

### 5. Liskov Substitution Principle Issues
- The pipeline can't easily swap different parser implementations
- No abstraction for different data sources

## Specific Code Smells

### 1. Long Method Anti-pattern
- `_extract_daily_features`: 200+ lines (611-757)
- `_calculate_features_with_stats`: 100+ lines (812-926)
- Multiple nested loops and conditionals

### 2. Feature Envy
- Methods that primarily work with data from other classes:
  ```python
  total_sleep_minutes = sum(w.total_duration_hours * 60 for w in sleep_windows)
  wake_periods = [g for w in sleep_windows for g in w.gap_hours if g > 0]
  ```

### 3. Data Clumps
- Repeated parameter groups:
  ```python
  sleep_records: list,
  activity_records: list,
  heart_records: list,
  ```

### 4. Primitive Obsession
- Using dict[str, list] instead of proper domain types
- Complex nested dictionaries for metrics

### 5. Inappropriate Intimacy
- Direct access to internal structures of domain entities
- Knowledge of specific field names and calculations

## Proposed Decomposition

### Phase 3.2: Extract DataParsingService
Responsibilities:
- Parse different file formats (XML, JSON)
- Validate parsed data
- Convert to domain entities
- Handle file I/O

### Phase 3.3: Extract AggregationPipeline
Responsibilities:
- Orchestrate feature aggregation
- Manage rolling windows
- Calculate statistics (mean, std, z-scores)
- Handle temporal alignment

### Phase 3.4: Implement Dependency Injection
- Create interfaces for all dependencies
- Use constructor injection
- Enable easy testing and extension

## Metrics
- Current lines: 957
- Methods: 12
- Responsibilities: 9+
- Dependencies: 15+ concrete classes

## Refactoring Strategy

1. **Start with DataParsingService** (Phase 3.2)
   - Extract all parsing logic
   - Create Parser interface
   - Implement Factory pattern

2. **Extract AggregationPipeline** (Phase 3.3)
   - Move feature calculation logic
   - Create FeatureAggregator interface
   - Implement Strategy pattern for different aggregations

3. **Implement Dependency Injection** (Phase 3.4)
   - Create interfaces for all services
   - Use constructor injection
   - Remove hard-coded instantiations

4. **Final State**
   - MoodPredictionPipeline becomes a thin orchestrator
   - Each service has a single responsibility
   - Easy to test, extend, and maintain

## Benefits
- Improved testability (can mock dependencies)
- Better separation of concerns
- Easier to add new data sources
- Simpler to modify feature calculations
- Reduced coupling between components