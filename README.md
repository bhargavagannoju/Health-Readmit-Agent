# Healthcare AI Systems - Patient Readmission Risk & Clinical Decision Support

A production-ready implementation of three interconnected healthcare AI systems:
1. **ML-based Patient Readmission Risk Prediction** (Classical ML + Deep Learning)
2. **LLM-powered Clinical Note Understanding** (Structured Data Extraction)
3. **Agentic Decision Support System** (Multi-Agent Orchestration)

## 📋 Project Overview

This repository demonstrates end-to-end healthcare AI engineering with:
- **Classical ML & Deep Learning** for risk prediction with explainability
- **LLM-based** structured extraction from unstructured clinical notes
- **Multi-agent orchestration** combining ML + GenAI for clinical decision-making
- **Production-grade** data pipelines, evaluation, and logging
- **Safety-first** approach with hallucination reduction and uncertainty quantification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Patient Data & Clinical Notes              │
└─────────────────┬──────────────────────────┬────────────────┘
                  │                          │
        ┌─────────▼─────────┐      ┌────────▼─────────┐
        │  Problem 1: ML    │      │ Problem 2: LLM   │
        │  Risk Prediction  │      │ Note Extraction  │
        │  (Readmission)    │      │ (Structured Data)│
        └─────────┬─────────┘      └────────┬─────────┘
                  │                          │
        ┌─────────▼──────────────────────────▼─────────┐
        │    Problem 3: Agentic System                  │
        │  ┌──────────────────────────────────────┐    │
        │  │ • Risk Scoring Agent                 │    │
        │  │ • Clinical Understanding Agent       │    │
        │  │ • Guideline Reasoning Agent          │    │
        │  │ • Decision Agent                     │    │
        │  │ • Audit/Logging Agent                │    │
        │  └──────────────────────────────────────┘    │
        │  Produces: Recommendations + Confidence      │
        └──────────────────┬──────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │  Clinician UI   │
                  │  Dashboard      │
                  └─────────────────┘
```

## 📂 Repository Structure

```
healthcare-ai-systems/
├── README.md                          # This file
├── .gitignore
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
│
├── problem_1_ml_readmission/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_data_generator.py  # Generate synthetic patient data
│   │   ├── data_loader.py              # Data loading utilities
│   │   └── sample_data.csv             # Example dataset (synthetic)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py      # Feature creation & transformation
│   │   ├── missing_value_handler.py    # Missing data imputation
│   │   └── imbalance_handler.py        # Class imbalance (SMOTE, etc)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_models.py         # Logistic Regression, Random Forest, XGBoost
│   │   ├── deep_learning_models.py     # Neural Networks (TensorFlow/PyTorch)
│   │   └── model_registry.py           # Model save/load utilities
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # AUROC, Precision-Recall, Calibration
│   │   ├── explainability.py           # SHAP, Feature Importance
│   │   └── evaluation_report.py        # Generate evaluation reports
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── training_pipeline.py        # End-to-end training pipeline
│   │
│   ├── notebooks/
│   │   ├── 01_eda_analysis.ipynb       # Exploratory Data Analysis
│   │   ├── 02_feature_engineering.ipynb
│   │   └── 03_model_comparison.ipynb
│   │
│   ├── tests/
│   │   ├── test_data_loader.py
│   │   ├── test_feature_engineering.py
│   │   └── test_models.py
│   │
│   ├── config/
│   │   └── config.yaml                 # ML pipeline configuration
│   │
│   └── outputs/
│       ├── models/                     # Saved trained models
│       ├── metrics/                    # Evaluation metrics
│       └── plots/                      # Visualizations
│
├── problem_2_llm_note_extraction/
│   ├── __init__.py
│   ├── prompt_engineering/
│   │   ├── __init__.py
│   │   ├── base_prompts.py             # Core extraction prompts
│   │   ├── prompt_templates.py         # Jinja2 templates
│   │   ├── validation_prompts.py       # Hallucination check prompts
│   │   └── prompt_library.yaml         # Prompt catalog
│   │
│   ├── llm_interface/
│   │   ├── __init__.py
│   │   ├── base_llm.py                 # Abstract LLM interface
│   │   ├── openai_client.py            # OpenAI GPT integration
│   │   ├── anthropic_client.py         # Anthropic Claude integration
│   │   └── llm_config.yaml             # LLM configuration
│   │
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── clinical_note_extractor.py  # Main extraction orchestrator
│   │   ├── entity_extractor.py         # Diagnoses, meds, symptoms
│   │   ├── date_event_extractor.py     # Temporal information
│   │   └── warning_detector.py         # Detect missing/conflicting info
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── hallucination_detector.py   # Detect hallucinated info
│   │   ├── consistency_checker.py      # Cross-validate extracted entities
│   │   ├── medical_validator.py        # Medical knowledge validation
│   │   └── confidence_scorer.py        # Confidence level assignment
│   │
│   ├── data_structures/
│   │   ├── __init__.py
│   │   └── schemas.py                  # Pydantic models for structured output
│   │
│   ├── sample_notes/
│   │   ├── sample_note_1.txt           # Example clinical notes
│   │   ├── sample_note_2.txt
│   │   └── sample_notes_catalog.md     # Note descriptions
│   │
│   ├── tests/
│   │   ├── test_extractors.py
│   │   ├── test_validation.py
│   │   └── test_llm_interface.py
│   │
│   ├── notebooks/
│   │   ├── 01_llm_exploration.ipynb    # LLM testing & prompt tuning
│   │   ├── 02_extraction_pipeline.ipynb
│   │   └── 03_validation_testing.ipynb
│   │
│   └── outputs/
│       ├── extracted_data/             # Sample extraction outputs (JSON)
│       └── validation_reports/         # Validation & confidence reports
│
├── problem_3_agentic_system/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py               # Abstract agent base class
│   │   ├── risk_scoring_agent.py       # Calls ML model (Problem 1)
│   │   ├── clinical_understanding_agent.py  # Uses LLM extraction (Problem 2)
│   │   ├── guideline_reasoning_agent.py    # Matches vs clinical guidelines
│   │   ├── decision_agent.py           # Produces recommendations
│   │   └── audit_logging_agent.py      # Tracks execution & decisions
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py       # Coordinates agent execution
│   │   ├── workflow_engine.py          # Workflow definition & execution
│   │   ├── state_manager.py            # Manages agent/workflow state
│   │   └── error_handler.py            # Graceful failure handling
│   │
│   ├── guidelines/
│   │   ├── clinical_guidelines.md      # Sample clinical guidelines
│   │   ├── guideline_loader.py         # Load & parse guidelines
│   │   └── guideline_matcher.py        # Match patient state vs guidelines
│   │
│   ├── decision_engine/
│   │   ├── __init__.py
│   │   ├── recommendation_generator.py # Generate actionable recommendations
│   │   ├── confidence_calculator.py    # Calculate confidence scores
│   │   └── care_gap_detector.py        # Identify gaps in care
│   │
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── audit_logger.py             # Detailed execution logging
│   │   ├── trace_formatter.py          # Format agent execution traces
│   │   └── log_storage.py              # Persist logs (database/file)
│   │
│   ├── tests/
│   │   ├── test_agents.py
│   │   ├── test_orchestrator.py
│   │   └── test_workflows.py
│   │
│   ├── notebooks/
│   │   ├── 01_agent_exploration.ipynb
│   │   ├── 02_workflow_definition.ipynb
│   │   └── 03_end_to_end_system.ipynb
│   │
│   └── outputs/
│       ├── recommendations/            # Generated recommendations (JSON)
│       └── audit_logs/                 # Agent execution traces & decisions
│
├── ui/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── patient.py              # Patient data endpoints
│   │   │   ├── risk_prediction.py      # ML prediction endpoints
│   │   │   ├── note_extraction.py      # Note extraction endpoints
│   │   │   ├── recommendations.py      # Agentic system endpoints
│   │   │   └── health.py               # Health check endpoints
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── ml_service.py           # ML model serving
│   │   │   ├── llm_service.py          # LLM extraction service
│   │   │   └── agentic_service.py      # Agentic system service
│   │   │
│   │   └── config.py
│   │
│   └── frontend/
│       ├── public/
│       │   └── index.html
│       ├── src/
│       │   ├── App.tsx
│       │   ├── components/
│       │   │   ├── PatientDashboard.tsx
│       │   │   ├── RiskPanel.tsx
│       │   │   ├── ClinicalNotesPanel.tsx
│       │   │   ├── RecommendationsPanel.tsx
│       │   │   ├── AgentActivityPanel.tsx
│       │   │   └── ExplanationPanel.tsx
│       │   │
│       │   ├── services/
│       │   │   ├── api.ts
│       │   │   └── types.ts
│       │   │
│       │   ├── styles/
│       │   │   └── index.css
│       │   │
│       │   └── index.tsx
│       │
│       ├── package.json
│       └── tsconfig.json
│
├── docs/
│   ├── ARCHITECTURE.md                 # Detailed architecture
│   ├── PROBLEM_1_ML.md                 # ML model documentation
│   ├── PROBLEM_2_LLM.md                # LLM extraction documentation
│   ├── PROBLEM_3_AGENTS.md             # Agentic system documentation
│   ├── API_REFERENCE.md                # API documentation
│   ├── DEPLOYMENT.md                   # Deployment guide
│   └── SAFETY_CONSIDERATIONS.md        # Safety & ethics
│
├── scripts/
│   ├── train_ml_model.py               # Train readmission model
│   ├── test_llm_extraction.py          # Test LLM extraction
│   ├── run_agentic_demo.py             # Run agent system demo
│   ├── generate_sample_data.py          # Generate synthetic data
│   └── evaluate_system.py               # Comprehensive system evaluation
│
├── docker/
│   ├── Dockerfile.ml                   # ML service container
│   ├── Dockerfile.llm                  # LLM service container
│   ├── Dockerfile.api                  # API service container
│   ├── docker-compose.yml              # Multi-container orchestration
│   └── .env.example                    # Environment variables template
│
├── tests/
│   ├── integration_tests.py            # End-to-end integration tests
│   └── test_suite.py                   # Run all tests
│
├── .github/
│   └── workflows/
│       ├── ci_cd.yml                   # CI/CD pipeline
│       └── tests.yml                   # Automated testing
│
├── LICENSE
├── .gitignore
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-ai-systems.git
cd healthcare-ai-systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Start

#### Problem 1: Train ML Model
```bash
python scripts/train_ml_model.py --config problem_1_ml_readmission/config/config.yaml
```

#### Problem 2: Test LLM Extraction
```bash
python scripts/test_llm_extraction.py --note-path problem_2_llm_note_extraction/sample_notes/sample_note_1.txt
```

#### Problem 3: Run Agentic Demo
```bash
python scripts/run_agentic_demo.py --patient-id PATIENT_001
```

#### Launch UI
```bash
# Terminal 1: Backend
cd ui/backend
python -m uvicorn app:app --reload --port 8000

# Terminal 2: Frontend
cd ui/frontend
npm install
npm start
```

## 📊 Problem-Specific Details

### Problem 1: ML Readmission Risk Prediction

**Key Features:**
- ✅ Feature engineering (handcrafted + automated)
- ✅ Missing value handling (KNN imputation, forward fill)
- ✅ Class imbalance correction (SMOTE, class weights)
- ✅ Multiple model training (Logistic Regression, Random Forest, XGBoost, Neural Networks)
- ✅ Comprehensive evaluation (AUROC, Precision-Recall, F1, Calibration)
- ✅ Explainability (SHAP values, Feature importance)

**Models Implemented:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Deep Neural Network (TensorFlow)
- TabNet (Deep Learning for tabular data)

**Key Metrics:**
```
- AUROC (Area Under ROC Curve)
- Precision-Recall Curve
- F1-Score
- Calibration Curve
- Feature Importance (Gini, Permutation)
- SHAP Feature Impact
```

**Configuration Example:**
```yaml
model:
  algorithm: xgboost
  hyperparameters:
    max_depth: 6
    learning_rate: 0.05
    n_estimators: 100

preprocessing:
  missing_value_strategy: knn
  imbalance_strategy: smote
  feature_selection: mutual_information

evaluation:
  test_size: 0.2
  cv_folds: 5
  metrics: [auroc, pr_auc, f1, calibration]
```

---

### Problem 2: LLM Clinical Note Extraction

**Key Features:**
- ✅ Robust prompt engineering with error handling
- ✅ Structured output extraction (Diagnoses, Medications, Symptoms, Dates, Warnings)
- ✅ Hallucination detection & reduction
- ✅ Confidence scoring
- ✅ Validation & consistency checking
- ✅ Medical knowledge grounding

**Extraction Entities:**
```
- Diagnoses (ICD codes & descriptions)
- Medications (name, dose, frequency, route)
- Symptoms & Chief Complaints
- Vital Signs
- Lab Results
- Dates & Important Events
- Care Gaps & Clinical Warnings
```

**Hallucination Reduction Strategy:**

1. **Input Grounding**
   - Extract only from explicit text spans
   - Quote original text for all entities
   - Flag inferred vs. explicit information

2. **Output Validation**
   - Cross-validate medications against standard formularies
   - Check diagnosis consistency with symptoms
   - Verify date logic (admission < discharge)

3. **Confidence Scoring**
   - Explicit mention: 1.0
   - Clear inference: 0.8
   - Implicit/uncertain: 0.5
   - Hallucinated/unsupported: 0.0 (flagged)

4. **Fallback Strategy**
   - Return partial results if extraction fails
   - Explicit "uncertain" tags
   - Request human review for low-confidence extractions

**Sample Output:**
```json
{
  "extraction_metadata": {
    "confidence": 0.87,
    "warnings": [],
    "timestamp": "2024-01-21T20:30:00Z"
  },
  "diagnoses": [
    {
      "text": "Type 2 Diabetes Mellitus",
      "icd_code": "E11.9",
      "confidence": 0.95,
      "source_span": "diagnosed with Type 2 Diabetes"
    }
  ],
  "medications": [
    {
      "name": "Metformin",
      "dose": "500mg",
      "frequency": "twice daily",
      "confidence": 0.9,
      "source_span": "Metformin 500mg BID"
    }
  ],
  "care_gaps": [
    {
      "issue": "No HbA1c recent lab result",
      "severity": "medium",
      "confidence": 0.8
    }
  ]
}
```

---

### Problem 3: Agentic Decision Support System

**Architecture:**

```
Patient Data + Clinical Notes
        │
        ├──→ Risk Scoring Agent
        │    └─→ Calls ML model → Risk score (0-1)
        │
        ├──→ Clinical Understanding Agent
        │    └─→ Calls LLM extractor → Structured data
        │
        ├──→ Guideline Reasoning Agent
        │    └─→ Matches vs guidelines → Care gaps
        │
        ├──→ Decision Agent
        │    └─→ Synthesizes → Recommendations
        │
        └──→ Audit Logging Agent
             └─→ Logs all steps, reasoning, confidence

Final Output:
{
  "patient_id": "...",
  "risk_score": 0.72,
  "clinical_facts": {...},
  "care_gaps": [...],
  "recommendations": [...],
  "confidence": 0.78,
  "execution_trace": [...]
}
```

**Agent Responsibilities:**

| Agent | Responsibility | Inputs | Outputs |
|-------|---|---|---|
| **Risk Scoring** | Call ML model, return risk score | Structured patient data | Risk score, confidence |
| **Clinical Understanding** | Extract entities from notes | Clinical note text | Structured entities |
| **Guideline Reasoning** | Match patient vs guidelines | Patient state, guidelines | Care gaps, guideline violations |
| **Decision** | Generate recommendations | All agent outputs | Ranked recommendations |
| **Audit Logging** | Track execution & decisions | All agent data | Execution trace, confidence |

**Error Handling:**
- Missing patient data → Return partial results with warnings
- LLM extraction failure → Use fallback templates
- Model serving error → Return risk score with confidence=0
- Guideline not found → Skip guideline matching gracefully
- All failures logged with retry logic

**Execution Trace Example:**
```json
{
  "workflow_id": "WF_20240121_001",
  "patient_id": "PAT_12345",
  "timestamp": "2024-01-21T20:35:00Z",
  "agents_executed": [
    {
      "agent_name": "risk_scoring_agent",
      "status": "success",
      "duration_ms": 245,
      "output": {
        "risk_score": 0.72,
        "confidence": 0.88,
        "model_version": "v1.2.1"
      }
    },
    {
      "agent_name": "clinical_understanding_agent",
      "status": "success",
      "duration_ms": 1850,
      "output": {
        "diagnoses": 3,
        "medications": 5,
        "care_gaps": 2,
        "confidence": 0.82
      }
    },
    {
      "agent_name": "guideline_reasoning_agent",
      "status": "success",
      "duration_ms": 340,
      "output": {
        "guideline_matches": 5,
        "violations": 1,
        "confidence": 0.85
      }
    },
    {
      "agent_name": "decision_agent",
      "status": "success",
      "duration_ms": 450,
      "output": {
        "recommendations": 3,
        "primary_recommendation": "Increase monitoring frequency",
        "confidence": 0.8
      }
    },
    {
      "agent_name": "audit_logging_agent",
      "status": "success",
      "duration_ms": 120
    }
  ],
  "overall_status": "success",
  "total_duration_ms": 3005,
  "final_confidence": 0.79,
  "user_id": "DR_456"
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest problem_1_ml_readmission/tests/

# Run with coverage
pytest --cov=. tests/

# Run integration tests
pytest tests/integration_tests.py -v
```

## 📈 Evaluation

Generate comprehensive evaluation report:

```bash
python scripts/evaluate_system.py \
  --output-dir results/ \
  --include-ml \
  --include-llm \
  --include-agents
```

This produces:
- ML model performance metrics & plots
- LLM extraction quality report
- Agent execution traces & success rates
- System-level performance dashboard

## 🐳 Docker Deployment

```bash
# Build all images
docker-compose build

# Run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Run tests in container
docker-compose exec ml pytest tests/
```

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design & component details
- **[PROBLEM_1_ML.md](docs/PROBLEM_1_ML.md)** - ML model documentation
- **[PROBLEM_2_LLM.md](docs/PROBLEM_2_LLM.md)** - LLM extraction documentation
- **[PROBLEM_3_AGENTS.md](docs/PROBLEM_3_AGENTS.md)** - Agentic system documentation
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - API endpoints & usage
- **[SAFETY_CONSIDERATIONS.md](docs/SAFETY_CONSIDERATIONS.md)** - Safety & ethics

## 🔒 Safety & Ethics

### Key Safety Measures

1. **Hallucination Detection**
   - All LLM outputs validated against source text
   - Confidence scores tied to verifiability
   - Medical knowledge grounding against clinical databases

2. **Explainability**
   - Every prediction explains its reasoning
   - SHAP values show feature contribution
   - Agent execution traces show decision steps

3. **Human-in-the-Loop**
   - System designed to support, not replace, clinical judgment
   - Confidence scores guide when to escalate to human review
   - Audit logs enable full transparency

4. **Class Imbalance Handling**
   - SMOTE for synthetic minority oversampling
   - Threshold optimization for precision-recall tradeoff
   - Calibration curves to ensure probability estimates are valid

5. **Data Privacy**
   - No real patient data in repository
   - Synthetic data only
   - HIPAA compliance framework (production ready)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

MIT License - See [LICENSE](LICENSE) file

## 👨‍💻 Author & Maintainer

**Your Name** | [Email] | [LinkedIn]

---

## 🆘 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check [Documentation](docs/)
- Review [Notebook Examples](problem_1_ml_readmission/notebooks/)

---

## 🗺️ Roadmap

- [ ] Production deployment on Kubernetes
- [ ] Real-world clinical data integration (MIMIC-III)
- [ ] Multi-language LLM support
- [ ] Advanced visualizations in UI
- [ ] Mobile app for clinicians
- [ ] Continuous model monitoring & retraining
- [ ] Integration with EHR systems

---

**Last Updated:** January 2026
**Status:** Production Ready ✅
