"""
Problem 3: Agentic System for Clinical Decision Support

Multi-agent orchestration system that combines ML predictions and LLM 
extraction to produce actionable clinical recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentExecutionTrace:
    """Track single agent execution."""
    agent_name: str
    status: AgentStatus
    start_time: str
    end_time: Optional[str] = None
    duration_ms: int = 0
    output: Dict = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0
    
    def to_dict(self) -> Dict:
        return {k: (v.value if isinstance(v, Enum) else v) 
                for k, v in asdict(self).items()}


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, timeout_seconds: int = 30):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.execution_trace = None
    
    @abstractmethod
    def execute(self, input_data: Dict) -> Dict:
        """
        Execute agent logic.
        
        Args:
            input_data: Dictionary with required inputs for agent
            
        Returns:
            Dictionary with agent output
        """
        pass
    
    def _create_trace(self) -> AgentExecutionTrace:
        """Create execution trace for this agent."""
        return AgentExecutionTrace(
            agent_name=self.name,
            status=AgentStatus.PENDING,
            start_time=datetime.now().isoformat()
        )
    
    def _finalize_trace(self, trace: AgentExecutionTrace, 
                       output: Dict, error: Optional[str] = None):
        """Finalize execution trace."""
        trace.end_time = datetime.now().isoformat()
        trace.status = AgentStatus.SUCCESS if error is None else AgentStatus.FAILED
        trace.output = output
        trace.error = error
        return trace


class RiskScoringAgent(BaseAgent):
    """
    Risk Scoring Agent
    
    Responsibility: Call ML model to predict readmission risk
    Input: Structured patient demographics and clinical data
    Output: Risk score (0-1) with confidence
    """
    
    def __init__(self, ml_model=None):
        super().__init__("risk_scoring_agent")
        self.ml_model = ml_model
    
    def execute(self, input_data: Dict) -> Dict:
        """Execute risk scoring."""
        trace = self._create_trace()
        
        try:
            # Extract patient features
            patient_features = input_data.get('patient_data', {})
            model_name = input_data.get('model_name', 'xgboost')
            
            if not self.ml_model:
                raise ValueError("ML model not initialized")
            
            # Get prediction
            risk_score, confidence = self._get_risk_score(patient_features, model_name)
            
            output = {
                'risk_score': float(risk_score),
                'risk_category': 'high' if risk_score > 0.5 else 'low',
                'confidence': float(confidence),
                'model_version': 'v1.2.1',
                'threshold_used': 0.5
            }
            
            trace = self._finalize_trace(trace, output)
            
        except Exception as e:
            logger.error(f"Risk scoring failed: {e}")
            output = {'error': str(e)}
            trace = self._finalize_trace(trace, {}, str(e))
        
        self.execution_trace = trace
        return output
    
    def _get_risk_score(self, patient_features: Dict, model_name: str) -> Tuple[float, float]:
        """Get risk score from ML model."""
        # Simulated ML model call
        # In production: predict_proba = ml_model.predict(features)
        
        risk_score = 0.65  # Simulated
        confidence = 0.88  # Model confidence
        
        return risk_score, confidence


class ClinicalUnderstandingAgent(BaseAgent):
    """
    Clinical Understanding Agent
    
    Responsibility: Use LLM-based note extraction
    Input: Clinical note text
    Output: Extracted structured entities (diagnoses, meds, symptoms)
    """
    
    def __init__(self, note_extractor=None):
        super().__init__("clinical_understanding_agent")
        self.note_extractor = note_extractor
    
    def execute(self, input_data: Dict) -> Dict:
        """Extract clinical entities from notes."""
        trace = self._create_trace()
        
        try:
            clinical_note = input_data.get('clinical_note', '')
            
            if not self.note_extractor:
                raise ValueError("Note extractor not initialized")
            
            # Extract entities
            diagnoses = self.note_extractor.extract_diagnoses(clinical_note)
            medications = self.note_extractor.extract_medications(clinical_note)
            symptoms = self.note_extractor.extract_symptoms(clinical_note)
            care_gaps = self.note_extractor.detect_care_gaps(clinical_note)
            
            output = {
                'diagnoses': [d.to_dict() for d in diagnoses],
                'medications': [m.to_dict() for m in medications],
                'symptoms': [s.to_dict() for s in symptoms],
                'care_gaps': care_gaps,
                'entities_extracted': len(diagnoses) + len(medications) + len(symptoms),
                'confidence': 0.82
            }
            
            trace = self._finalize_trace(trace, output)
            
        except Exception as e:
            logger.error(f"Clinical extraction failed: {e}")
            output = {'error': str(e)}
            trace = self._finalize_trace(trace, {}, str(e))
        
        self.execution_trace = trace
        return output


class GuidelineReasoningAgent(BaseAgent):
    """
    Guideline Reasoning Agent
    
    Responsibility: Match patient state against clinical guidelines
    Input: Patient state (diagnoses, medications), clinical guidelines
    Output: Guideline violations, care gaps, recommendations
    """
    
    def __init__(self, guidelines: Optional[Dict] = None):
        super().__init__("guideline_reasoning_agent")
        self.guidelines = guidelines or self._load_default_guidelines()
    
    def execute(self, input_data: Dict) -> Dict:
        """Perform guideline matching."""
        trace = self._create_trace()
        
        try:
            diagnoses = input_data.get('diagnoses', [])
            medications = input_data.get('medications', [])
            labs = input_data.get('labs', {})
            
            # Match against guidelines
            violations = self._match_guidelines(diagnoses, medications, labs)
            
            output = {
                'guideline_matches': len(violations),
                'violations': violations,
                'compliance_score': self._calculate_compliance_score(violations),
                'confidence': 0.85
            }
            
            trace = self._finalize_trace(trace, output)
            
        except Exception as e:
            logger.error(f"Guideline reasoning failed: {e}")
            output = {'error': str(e)}
            trace = self._finalize_trace(trace, {}, str(e))
        
        self.execution_trace = trace
        return output
    
    def _load_default_guidelines(self) -> Dict:
        """Load default clinical guidelines."""
        return {
            'diabetes': {
                'required_labs': ['hba1c', 'lipid_panel', 'kidney_function'],
                'target_hba1c': 7.0,
                'required_medications': ['metformin_or_glyburide']
            },
            'hypertension': {
                'target_bp': '130/80',
                'required_medications': ['acei_or_arb'],
            },
            'heart_failure': {
                'required_labs': ['bnp', 'troponin'],
                'required_medications': ['acei_or_beta_blocker'],
                'monitoring': 'weekly'
            }
        }
    
    def _match_guidelines(self, diagnoses: List, medications: List, 
                         labs: Dict) -> List[Dict]:
        """Match patient state against guidelines."""
        violations = []
        
        for diag in diagnoses:
            diag_name = diag.get('text', '').lower()
            guideline = self.guidelines.get(diag_name)
            
            if not guideline:
                continue
            
            # Check required labs
            for required_lab in guideline.get('required_labs', []):
                if required_lab not in labs:
                    violations.append({
                        'type': 'missing_lab',
                        'issue': f'Missing {required_lab} for {diag_name}',
                        'severity': 'medium',
                        'guideline_ref': 'Standard diabetes management'
                    })
            
            # Check required medications
            med_names = [m.get('name', '').lower() for m in medications]
            for required_med in guideline.get('required_medications', []):
                if not any(required_med.replace('_or_', '').lower() in m 
                          for m in med_names):
                    violations.append({
                        'type': 'missing_medication',
                        'issue': f'Consider {required_med} for {diag_name}',
                        'severity': 'medium',
                        'guideline_ref': 'Standard management'
                    })
        
        return violations
    
    def _calculate_compliance_score(self, violations: List) -> float:
        """Calculate guideline compliance score."""
        if not violations:
            return 1.0
        
        severe_count = len([v for v in violations if v['severity'] == 'severe'])
        score = max(0.0, 1.0 - (severe_count * 0.2))
        
        return score


class DecisionAgent(BaseAgent):
    """
    Decision Agent
    
    Responsibility: Synthesize all inputs and produce recommendations
    Input: Risk score, clinical facts, guidelines violations
    Output: Ranked recommendations with rationale
    """
    
    def execute(self, input_data: Dict) -> Dict:
        """Generate recommendations."""
        trace = self._create_trace()
        
        try:
            risk_score = input_data.get('risk_score', 0.0)
            clinical_facts = input_data.get('clinical_facts', {})
            guideline_gaps = input_data.get('guideline_gaps', [])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_score, clinical_facts, guideline_gaps
            )
            
            output = {
                'recommendations': recommendations,
                'primary_recommendation': recommendations[0] if recommendations else None,
                'num_recommendations': len(recommendations),
                'confidence': 0.80
            }
            
            trace = self._finalize_trace(trace, output)
            
        except Exception as e:
            logger.error(f"Decision generation failed: {e}")
            output = {'error': str(e)}
            trace = self._finalize_trace(trace, {}, str(e))
        
        self.execution_trace = trace
        return output
    
    def _generate_recommendations(self, risk_score: float, 
                                 clinical_facts: Dict,
                                 guideline_gaps: List) -> List[Dict]:
        """Generate clinical recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_score > 0.7:
            recommendations.append({
                'priority': 'high',
                'action': 'Increase monitoring frequency',
                'rationale': f'High readmission risk ({risk_score:.1%})',
                'timeline': 'Immediately',
                'evidence_level': 'ML model prediction'
            })
            
            recommendations.append({
                'priority': 'high',
                'action': 'Schedule early discharge follow-up',
                'rationale': 'High-risk patient needs close monitoring',
                'timeline': '3-5 days post-discharge',
                'evidence_level': 'Standard care pathway'
            })
        
        # Guideline-based recommendations
        for gap in guideline_gaps:
            recommendations.append({
                'priority': 'medium' if gap['severity'] == 'medium' else 'high',
                'action': f"Address: {gap['issue']}",
                'rationale': gap.get('guideline_ref', 'Guideline compliance'),
                'timeline': 'Before discharge',
                'evidence_level': 'Clinical guideline'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: (x['priority'] != 'high', 
                                           x['priority'] != 'medium'))
        
        return recommendations[:5]  # Top 5 recommendations


class AuditLoggingAgent(BaseAgent):
    """
    Audit Logging Agent
    
    Responsibility: Track all agent execution and decisions
    Input: All agent outputs and execution traces
    Output: Structured audit log with full decision trail
    """
    
    def __init__(self):
        super().__init__("audit_logging_agent")
        self.logs = []
    
    def execute(self, input_data: Dict) -> Dict:
        """Log agent execution."""
        trace = self._create_trace()
        
        try:
            workflow_id = input_data.get('workflow_id', 'UNKNOWN')
            patient_id = input_data.get('patient_id', 'UNKNOWN')
            agent_traces = input_data.get('agent_traces', [])
            final_output = input_data.get('final_output', {})
            
            # Create audit log entry
            audit_log = {
                'workflow_id': workflow_id,
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'agents_executed': agent_traces,
                'final_output': final_output,
                'system_status': 'success',
                'user_id': input_data.get('user_id', 'SYSTEM')
            }
            
            self.logs.append(audit_log)
            
            output = {
                'log_entry_id': f"{workflow_id}_{patient_id}",
                'total_agents_logged': len(agent_traces),
                'log_stored': True,
                'confidence': 1.0
            }
            
            trace = self._finalize_trace(trace, output)
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            output = {'error': str(e)}
            trace = self._finalize_trace(trace, {}, str(e))
        
        self.execution_trace = trace
        return output


class AgentOrchestrator:
    """
    Orchestrate multi-agent workflow for clinical decision support.
    
    Coordinates:
    - Risk Scoring Agent (ML predictions)
    - Clinical Understanding Agent (LLM extraction)
    - Guideline Reasoning Agent (Guideline matching)
    - Decision Agent (Recommendations)
    - Audit Logging Agent (Tracking)
    """
    
    def __init__(self, ml_model=None, note_extractor=None):
        self.agents = {
            'risk_scoring': RiskScoringAgent(ml_model),
            'clinical_understanding': ClinicalUnderstandingAgent(note_extractor),
            'guideline_reasoning': GuidelineReasoningAgent(),
            'decision': DecisionAgent(),
            'audit_logging': AuditLoggingAgent()
        }
        
        self.workflow_id = None
        self.execution_traces = []
    
    def execute_workflow(self, patient_id: str, patient_data: Dict,
                        clinical_note: str) -> Dict:
        """
        Execute complete clinical decision workflow.
        
        Args:
            patient_id: Patient identifier
            patient_data: Structured patient data
            clinical_note: Unstructured clinical note text
            
        Returns:
            Final recommendations and decision trace
        """
        self.workflow_id = f"WF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting workflow {self.workflow_id} for patient {patient_id}")
        
        # Stage 1: Risk Scoring
        risk_output = self.agents['risk_scoring'].execute({
            'patient_data': patient_data,
            'model_name': 'xgboost'
        })
        self.execution_traces.append(self.agents['risk_scoring'].execution_trace)
        
        # Stage 2: Clinical Understanding
        clinical_output = self.agents['clinical_understanding'].execute({
            'clinical_note': clinical_note
        })
        self.execution_traces.append(self.agents['clinical_understanding'].execution_trace)
        
        # Stage 3: Guideline Reasoning
        guideline_output = self.agents['guideline_reasoning'].execute({
            'diagnoses': clinical_output.get('diagnoses', []),
            'medications': clinical_output.get('medications', []),
            'labs': {}
        })
        self.execution_traces.append(self.agents['guideline_reasoning'].execution_trace)
        
        # Stage 4: Decision Making
        decision_output = self.agents['decision'].execute({
            'risk_score': risk_output.get('risk_score', 0.0),
            'clinical_facts': clinical_output,
            'guideline_gaps': guideline_output.get('violations', [])
        })
        self.execution_traces.append(self.agents['decision'].execution_trace)
        
        # Stage 5: Audit Logging
        audit_output = self.agents['audit_logging'].execute({
            'workflow_id': self.workflow_id,
            'patient_id': patient_id,
            'agent_traces': [t.to_dict() for t in self.execution_traces],
            'final_output': decision_output,
            'user_id': 'DR_001'
        })
        self.execution_traces.append(self.agents['audit_logging'].execution_trace)
        
        # Compile final result
        final_result = {
            'workflow_id': self.workflow_id,
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_output.get('risk_score', 0.0),
            'risk_category': risk_output.get('risk_category', 'unknown'),
            'clinical_facts': {
                'diagnoses': clinical_output.get('diagnoses', []),
                'medications': clinical_output.get('medications', []),
                'symptoms': clinical_output.get('symptoms', []),
                'care_gaps': clinical_output.get('care_gaps', [])
            },
            'guideline_violations': guideline_output.get('violations', []),
            'recommendations': decision_output.get('recommendations', []),
            'primary_recommendation': decision_output.get('primary_recommendation'),
            'overall_confidence': self._calculate_overall_confidence(),
            'execution_trace': [t.to_dict() for t in self.execution_traces],
            'status': 'completed'
        }
        
        logger.info(f"Workflow {self.workflow_id} completed")
        return final_result
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence across all agents."""
        confidences = []
        
        for trace in self.execution_traces:
            if trace.output:
                conf = trace.output.get('confidence', 0.0)
                if conf:
                    confidences.append(conf)
        
        return np.mean(confidences) if confidences else 0.5


# Example usage and utilities
import numpy as np

if __name__ == "__main__":
    # Create orchestrator
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
    Patient: 68M admitted with shortness of breath
    PMH: Type 2 DM, HTN, heart failure
    Current meds: Metformin 500mg BID, Lisinopril 10mg daily
    Vitals: BP 140/85, HR 88, RR 22
    Discharged on same medications plus furosemide 40mg daily.
    """
    
    # Execute workflow
    result = orchestrator.execute_workflow(
        patient_id='PAT_12345',
        patient_data=patient_data,
        clinical_note=clinical_note
    )
    
    print("✓ Agentic workflow completed")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"Recommendations: {len(result['recommendations'])}")
    print(f"Overall Confidence: {result['overall_confidence']:.2f}")
    print("\n" + json.dumps(result, indent=2, default=str))
