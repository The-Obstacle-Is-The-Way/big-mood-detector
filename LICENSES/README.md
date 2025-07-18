# Third-Party Licenses

This directory contains license information for third-party components used in the Big Mood Detector project.

## Project License

Big Mood Detector is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file in the project root for the full license text.

## Model Weights

### XGBoost Models
- **Source**: Seoul National University Hospital Study
- **Paper**: "Predicting mood episodes in mood disorder patients using sleep and circadian rhythm features"
- **License**: Research use permitted with citation
- **Citation Required**: Yes

### PAT (Pretrained Actigraphy Transformer) Models
- **Source**: Dartmouth College
- **Paper**: "AI Foundation Models for Wearable Movement Data in Mental Health Research"
- **License**: Research use permitted
- **Model Weights**: NHANES pretrained weights
- **Citation Required**: Yes

## Software Dependencies

### Core ML Libraries
- **XGBoost**: Apache License 2.0
- **TensorFlow**: Apache License 2.0
- **scikit-learn**: BSD 3-Clause License
- **NumPy**: BSD 3-Clause License
- **Pandas**: BSD 3-Clause License

### Web Framework
- **FastAPI**: MIT License
- **Uvicorn**: BSD 3-Clause License
- **Pydantic**: MIT License

### Other Key Dependencies
- **Redis**: BSD 3-Clause License
- **PostgreSQL Client (psycopg2)**: LGPL with exceptions
- **Prometheus Client**: Apache License 2.0

## Data Sources

### NHANES (National Health and Nutrition Examination Survey)
- **Source**: CDC (Centers for Disease Control and Prevention)
- **License**: Public Domain
- **Usage**: Pretrained model weights only

## Citation Requirements

When using this software for research, please cite:

1. **Seoul National Study (XGBoost Models)**:
   ```
   Lim et al., "Accurately predicting mood episodes in mood disorder patients 
   using wearable sleep and circadian rhythm features", 
   npj Digital Medicine, 2024
   ```

2. **PAT Models**:
   ```
   Ruan et al., "AI Foundation Models for Wearable Movement Data 
   in Mental Health Research", 2024
   ```

## License Compliance

To ensure compliance:
1. Include citations in any publications
2. Do not redistribute model weights without permission
3. Maintain this LICENSES directory in any forks
4. Update this file when adding new dependencies

## Generating License Report

To generate a full license report for all Python dependencies:

```bash
pip install pip-licenses
pip-licenses --format=markdown --output-file=LICENSES/python-dependencies.md
```