# 🏗️ Big Mood Detector - Architecture Overview

## 📐 System Architecture

Big Mood Detector follows **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaces Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │     CLI     │  │   FastAPI   │  │   Future Web UI    │  │
│  │  (Typer)    │  │   Server    │  │   (Next.js)        │  │
│  └─────────────┘  └─────────────┘  └────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Application Layer                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Use Cases & Services                   │    │
│  │  • ProcessHealthDataUseCase                         │    │
│  │  • PredictMoodEnsembleUseCase                       │    │
│  │  • LabelManagementService                           │    │
│  │  • FileWatcherService                               │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Domain Layer                             │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │    Entities     │  │  Domain Services │                 │
│  │ • SleepRecord   │  │ • SleepWindowAnalyzer            │ │
│  │ • ActivityRecord│  │ • ActivitySequenceExtractor      │ │
│  │ • HeartRateRec  │  │ • CircadianRhythmCalculator     │ │
│  │ • Episode       │  │ • FeatureExtractionService      │ │
│  │ • Label         │  │ • MoodPredictor (interface)     │ │
│  └─────────────────┘  └──────────────────┘                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Infrastructure Layer                         │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Parsers    │  │ ML Models   │  │  Repositories   │   │
│  │ • JSON       │  │ • XGBoost   │  │ • SQLite        │   │
│  │ • XML        │  │ • PAT       │  │ • InMemory      │   │
│  │ • Factory    │  │ • Ensemble  │  │ • File          │   │
│  └──────────────┘  └─────────────┘  └─────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Fine-Tuning  │  │ Background  │  │   Monitoring    │   │
│  │ • Personal   │  │ • TaskQueue │  │ • FileWatcher   │   │
│  │ • Population │  │ • Worker    │  │ • Metrics       │   │
│  │ • NHANES     │  │ • Celery    │  │ • Logging       │   │
│  └──────────────┘  └─────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 🧩 Core Components

### 1. Interfaces Layer
**Purpose**: Entry points for users and external systems

- **CLI (Typer)**: Command-line interface with 6 main commands
  - `process`: Extract features from health data
  - `predict`: Generate mood predictions
  - `label`: Manage ground truth annotations
  - `train`: Fine-tune personalized models
  - `serve`: Start API server
  - `watch`: Monitor directories

- **API (FastAPI)**: RESTful endpoints for integrations
  - File upload and processing
  - Async predictions
  - Result retrieval
  - Webhook support

### 2. Application Layer
**Purpose**: Orchestrate business logic and coordinate between layers

- **Use Cases**: Implement complete workflows
  ```python
  class ProcessHealthDataUseCase:
      def execute(self, input_path: Path) -> ClinicalFeatures:
          # 1. Parse data (infrastructure)
          # 2. Extract features (domain)
          # 3. Store results (infrastructure)
          # 4. Return features
  ```

- **Services**: Cross-cutting concerns
  - Label management with multi-rater support
  - File watching with debouncing
  - Longitudinal analysis

### 3. Domain Layer
**Purpose**: Core business logic, independent of frameworks

- **Entities**: Core data structures
  ```python
  @dataclass
  class SleepRecord:
      start_time: datetime
      end_time: datetime
      sleep_state: SleepState
      heart_rate_samples: List[int]
      
  @dataclass
  class Episode:
      id: UUID
      mood_type: MoodType  # DEPRESSIVE, MANIC, HYPOMANIC
      start_date: date
      end_date: date
      severity: Severity
      confidence: float
  ```

- **Domain Services**: Business rules
  - `SleepWindowAnalyzer`: Merges sleep episodes within 3.75 hours
  - `ActivitySequenceExtractor`: Creates 1440-minute daily sequences
  - `CircadianRhythmCalculator`: Computes phase, amplitude, stability

### 4. Infrastructure Layer
**Purpose**: External dependencies and technical implementations

- **Data Parsers**:
  ```python
  # Dual pipeline support
  XMLParser → StreamingXMLParser (handles 520MB+ files)
  JSONParser → Direct parsing (pre-aggregated data)
  ParserFactory → Auto-detects format
  ```

- **ML Models**:
  ```python
  XGBoostMoodPredictor:
    - depression_model.pkl (AUC 0.80)
    - manic_model.pkl (AUC 0.98)
    - hypomanic_model.pkl (AUC 0.95)
  
  PATModel:
    - PAT-S (285K params)
    - PAT-M (1M params)
    - PAT-L (2M params)
  ```

- **Repositories**: Data persistence
  - SQLite for episodes and labels
  - File-based for features
  - In-memory for caching

## 🔄 Data Flow

### 1. Input Processing
```
Health Data → Parser → Domain Entities → Aggregators → Features
     ↓           ↓            ↓              ↓            ↓
  XML/JSON   Streaming   SleepRecord   SleepWindow   36 Features
```

### 2. Prediction Pipeline
```
Features → ML Models → Risk Scores → Clinical Report
    ↓          ↓            ↓              ↓
36 values  XGBoost+PAT  0.0-1.0    Recommendations
```

### 3. Feedback Loop
```
Predictions → Labels → Fine-tuning → Personalized Model
     ↓          ↓          ↓               ↓
  Results   Ground Truth  Training   Better Accuracy
```

## 🎯 Key Design Decisions

### 1. Clean Architecture
- **Why**: Maintainability, testability, flexibility
- **Benefit**: Can swap ML models without changing business logic
- **Example**: PAT model added without modifying domain layer

### 2. Streaming XML Parser
- **Why**: Handle 500MB+ Apple Health exports
- **How**: SAX parser with <100MB memory usage
- **Performance**: 520MB in 13 seconds

### 3. Dual Model Ensemble
- **Why**: Leverage strengths of both approaches
- **XGBoost**: Interpretable features, fast inference
- **PAT**: Captures temporal patterns, state-of-the-art accuracy

### 4. Repository Pattern
- **Why**: Abstract data access
- **Benefit**: Easy to switch between SQLite, PostgreSQL, cloud storage
- **Testing**: In-memory repositories for unit tests

## 🚀 Performance Characteristics

### Processing Speed
- **XML Parsing**: 40,000 records/second
- **Feature Extraction**: <1 second per year of data
- **ML Inference**: <100ms per prediction
- **API Response**: <200ms average

### Memory Usage
- **Streaming Parser**: <100MB for any file size
- **Feature Storage**: ~1KB per day per user
- **Model Weights**: 47MB total (all models)

### Scalability
- **Concurrent Users**: 100+ via API
- **Background Jobs**: Celery with Redis
- **Horizontal Scaling**: Stateless design

## 🔒 Security Architecture

### Data Protection
```python
# Layered security approach
Input → Validation → Sanitization → Processing → Encryption → Storage
```

### Access Control
- API authentication via JWT tokens
- Role-based permissions (user, clinician, admin)
- Audit logging for all operations

### Privacy by Design
- Local processing option (no cloud required)
- Configurable data retention
- Anonymization for research export

## 🧪 Testing Strategy

### Test Pyramid
```
         E2E Tests (5%)
        /           \
   Integration Tests (25%)
   /                    \
Unit Tests (70% - 695 tests)
```

### Test Coverage
- Domain Layer: 98%
- Application Layer: 92%
- Infrastructure: 87%
- Overall: 91%

## 📦 Deployment Architecture

### Local Development
```bash
big-mood serve
```

### Docker Deployment
```yaml
services:
  api:
    image: big-mood-detector:latest
    ports: ["8000:8000"]
  
  worker:
    image: big-mood-detector:latest
    command: celery worker
  
  redis:
    image: redis:7
  
  postgres:
    image: postgres:15
```

### Cloud Deployment
- **API**: AWS ECS / Google Cloud Run
- **Workers**: Kubernetes Jobs
- **Storage**: S3 / Cloud Storage
- **Database**: RDS / Cloud SQL

## 🔮 Future Architecture Considerations

### Planned Enhancements
1. **Real-time Streaming**: Kafka for continuous data ingestion
2. **GraphQL API**: For flexible client queries
3. **Federated Learning**: Privacy-preserving model updates
4. **Multi-tenancy**: Organization-level deployments

### Extension Points
- Custom ML models via plugin system
- Additional data sources (wearables, EHR)
- Clinical decision support rules engine
- Advanced visualization dashboard

---

For implementation details, see:
- [Model Integration Guide](./model_integration_guide.md)
- [API Reference](./API_REFERENCE.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)