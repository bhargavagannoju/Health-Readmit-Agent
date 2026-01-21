# QUICKSTART - Healthcare AI Systems

## 5-Minute Setup Guide

### Prerequisites
- Python 3.9+
- Git
- Docker & Docker Compose (optional)
- OpenAI API Key (for LLM features)

### Installation

#### Option 1: Local Development (Recommended for Testing)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/healthcare-ai-systems.git
cd healthcare-ai-systems

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Run individual components
```

#### Option 2: Docker Compose (Full Stack)

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/healthcare-ai-systems.git
cd healthcare-ai-systems

# 2. Set environment variables
cp .env.example .env
export OPENAI_API_KEY="sk-..."

# 3. Start all services
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. View logs
docker-compose logs -f api
```

---

## Quick Tests

### Test 1: Problem 1 - ML Risk Prediction

```bash
# Start Python interpreter
python

# Run test
from problem_1_ml_readmission.preprocessing import DataPreprocessor
from problem_1_ml_readmission.models import ModelTrainer
import pandas as pd
import numpy as np

# Create sample data
n_samples = 1000
sample_data = pd.DataFrame({
    'patient_id': [f'PAT_{i:05d}' for i in range(n_samples)],
    'age': np.random.randint(18, 90, n_samples),
    'length_of_stay': np.random.randint(1, 30, n_samples),
    'num_diagnoses': np.random.randint(1, 15, n_samples),
    'num_medications': np.random.randint(0, 20, n_samples),
    'previous_admissions_6m': np.random.randint(0, 5, n_samples),
    'readmitted': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(sample_data)

print(f"✓ Data preprocessed: {X.shape}")
print(f"✓ Class balance: {y.value_counts().to_dict()}")
```

### Test 2: Problem 2 - LLM Clinical Note Extraction

```bash
from problem_2_llm_note_extraction.extractors import ClinicalNoteExtractor

# Sample clinical note
sample_note = """
Patient: 65-year-old male with Type 2 Diabetes and HTN
Current medications: Metformin 500mg BID, Lisinopril 10mg daily
Chief Complaint: Shortness of breath
"""

# Extract
extractor = ClinicalNoteExtractor()

medications = extractor.extract_medications(sample_note)
print("✓ Medications extracted:")
for med in medications:
    print(f"  - {med.name} {med.dose} {med.frequency}")

diagnoses = extractor.extract_diagnoses(sample_note)
print("✓ Diagnoses extracted:")
for diag in diagnoses:
    print(f"  - {diag.text}")
```

### Test 3: Problem 3 - Agentic System

```bash
from problem_3_agentic_system.agents import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# Sample inputs
patient_data = {
    'age': 68,
    'gender': 'M',
    'length_of_stay': 5,
    'num_diagnoses': 3,
    'num_medications': 7,
    'previous_admissions_6m': 2
}

clinical_note = """
Patient admitted with shortness of breath.
PMH: Type 2 DM, HTN, heart failure
Current meds: Metformin 500mg BID, Lisinopril 10mg daily
Discharged on furosemide 40mg daily.
"""

# Execute workflow
result = orchestrator.execute_workflow(
    patient_id='PAT_12345',
    patient_data=patient_data,
    clinical_note=clinical_note
)

print("✓ Agentic workflow completed")
print(f"✓ Risk Score: {result['risk_score']:.2f}")
print(f"✓ Recommendations: {len(result['recommendations'])}")
print(f"✓ Overall Confidence: {result['overall_confidence']:.2f}")
```

---

## API Testing with cURL

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-21T20:30:00Z"
}
```

### 2. Risk Prediction

```bash
curl -X POST http://localhost:8000/api/v1/risk-prediction \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_12345",
    "patient_data": {
      "age": 68,
      "gender": "M",
      "length_of_stay": 5,
      "num_diagnoses": 3,
      "num_medications": 7,
      "previous_admissions_6m": 2
    },
    "model_name": "xgboost"
  }'
```

Response:
```json
{
  "patient_id": "PAT_12345",
  "risk_score": 0.65,
  "risk_category": "high",
  "confidence": 0.88,
  "model_version": "v1.2.1",
  "timestamp": "2024-01-21T20:30:00Z"
}
```

### 3. Clinical Note Extraction

```bash
curl -X POST http://localhost:8000/api/v1/extract-clinical-note \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_12345",
    "clinical_note": "65yo male with Type 2 DM on Metformin 500mg BID..."
  }'
```

### 4. Get Recommendations

```bash
curl -X POST http://localhost:8000/api/v1/get-recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_12345",
    "patient_data": {...},
    "clinical_note": "..."
  }'
```

---

## Using Python Requests

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Risk Prediction
response = requests.post(
    f"{BASE_URL}/api/v1/risk-prediction",
    json={
        "patient_id": "PAT_12345",
        "patient_data": {
            "age": 68,
            "gender": "M",
            "length_of_stay": 5,
            "num_diagnoses": 3,
            "num_medications": 7,
            "previous_admissions_6m": 2
        }
    }
)
print(response.json())

# 2. Clinical Note Extraction
response = requests.post(
    f"{BASE_URL}/api/v1/extract-clinical-note",
    json={
        "patient_id": "PAT_12345",
        "clinical_note": "..."
    }
)
print(response.json())

# 3. Get Recommendations
response = requests.post(
    f"{BASE_URL}/api/v1/get-recommendations",
    json={
        "patient_id": "PAT_12345",
        "patient_data": {...},
        "clinical_note": "..."
    }
)
result = response.json()
print(f"Risk: {result['risk_score']:.2f}")
print(f"Recommendations: {len(result['recommendations'])}")
```

---

## Running Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Navigate to notebooks:
# - problem_1_ml_readmission/notebooks/01_eda_analysis.ipynb
# - problem_2_llm_note_extraction/notebooks/01_llm_exploration.ipynb
# - problem_3_agentic_system/notebooks/01_agent_exploration.ipynb
```

---

## Training ML Models

```bash
python scripts/train_ml_model.py \
  --config problem_1_ml_readmission/config/config.yaml \
  --output-dir ./outputs/models
```

This will:
1. Generate synthetic patient data (or load real data)
2. Perform feature engineering
3. Train all models (RF, XGBoost, LightGBM, etc.)
4. Evaluate with cross-validation
5. Save best model
6. Generate evaluation report

Output:
```
outputs/
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── ...
├── metrics/
│   ├── evaluation_report.json
│   └── metrics_comparison.csv
└── plots/
    ├── roc_curves.png
    ├── precision_recall.png
    ├── feature_importance.png
    └── calibration_curves.png
```

---

## Testing LLM Extraction

```bash
python scripts/test_llm_extraction.py \
  --note-path problem_2_llm_note_extraction/sample_notes/sample_note_1.txt \
  --llm-provider openai  # or anthropic
```

Output:
```
Extraction Results:
✓ Diagnoses: 3 extracted
✓ Medications: 5 extracted
✓ Symptoms: 2 extracted
✓ Care gaps: 2 identified

Confidence Scores:
- diagnoses: 0.92
- medications: 0.87
- symptoms: 0.95
- overall: 0.89

Hallucination Check: PASSED
```

---

## Running Agentic Demo

```bash
python scripts/run_agentic_demo.py \
  --patient-id PAT_001 \
  --include-ml \
  --include-llm \
  --include-guidelines
```

Output:
```
Workflow: WF_20240121_203000
Patient: PAT_001

Agent Execution:
  ✓ RiskScoringAgent - 245ms (risk: 0.65, conf: 0.88)
  ✓ ClinicalUnderstandingAgent - 1850ms (entities: 10, conf: 0.82)
  ✓ GuidelineReasoningAgent - 340ms (violations: 1, conf: 0.85)
  ✓ DecisionAgent - 450ms (recommendations: 3, conf: 0.80)
  ✓ AuditLoggingAgent - 120ms (logged: yes)

Total Duration: 3.005s
Overall Confidence: 0.79

Recommendations:
1. [HIGH] Increase monitoring frequency (ML prediction)
2. [HIGH] Schedule early discharge follow-up (Care pathway)
3. [MEDIUM] Order HbA1c test (Guideline compliance)
```

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
# Install TensorFlow (optional, but required for Deep Learning)
pip install tensorflow>=2.10.0
```

### Issue 2: "OpenAI API Key not found"

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY="sk-your-key-here"
# or in .env file:
OPENAI_API_KEY=sk-...
```

### Issue 3: "Database connection refused"

**Solution:**
```bash
# Make sure PostgreSQL is running
docker-compose up -d db

# or use SQLite for development
DATABASE_URL="sqlite:///healthcare_ai.db"
```

### Issue 4: "Port 8000 already in use"

**Solution:**
```bash
# Use different port
python -m uvicorn ui.backend.app:app --port 8001

# or kill process using port 8000
lsof -i :8000
kill -9 <PID>
```

---

## Next Steps

1. **Read the full documentation:**
   - [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
   - [PROBLEM_1_ML.md](docs/PROBLEM_1_ML.md) - ML details
   - [PROBLEM_2_LLM.md](docs/PROBLEM_2_LLM.md) - LLM details
   - [PROBLEM_3_AGENTS.md](docs/PROBLEM_3_AGENTS.md) - Agentic system

2. **Explore notebooks:**
   - `problem_1_ml_readmission/notebooks/` - ML experiments
   - `problem_2_llm_note_extraction/notebooks/` - LLM tuning
   - `problem_3_agentic_system/notebooks/` - Agent testing

3. **Run tests:**
   ```bash
   pytest tests/ -v --cov
   ```

4. **Deploy:**
   - See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production setup

---

**Happy exploring! 🚀**
