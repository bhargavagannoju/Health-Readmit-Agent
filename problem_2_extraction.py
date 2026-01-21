"""
Problem 2: LLM-based Clinical Note Extraction

Extract structured medical insights from unstructured clinical notes
using LLM-based extraction with hallucination detection and confidence scoring.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import re

logger = logging.getLogger(__name__)


class ExtractionConfidence(str, Enum):
    """Confidence levels for extracted entities."""
    EXPLICIT = "explicit"        # Direct mention in text
    INFERRED = "inferred"        # Logical inference
    UNCERTAIN = "uncertain"      # Ambiguous/implicit
    HALLUCINATED = "hallucinated" # Not grounded in text


@dataclass
class MedicationEntity:
    """Structured medication representation."""
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = "oral"
    confidence: ExtractionConfidence = ExtractionConfidence.EXPLICIT
    source_span: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {k: (v.value if isinstance(v, Enum) else v) 
                for k, v in asdict(self).items()}


@dataclass
class DiagnosisEntity:
    """Structured diagnosis representation."""
    text: str
    icd_code: Optional[str] = None
    confidence: ExtractionConfidence = ExtractionConfidence.EXPLICIT
    source_span: str = ""
    onset_date: Optional[str] = None
    status: str = "active"  # active, resolved, history
    
    def to_dict(self) -> Dict:
        return {k: (v.value if isinstance(v, Enum) else v) 
                for k, v in asdict(self).items()}


@dataclass
class SymptomEntity:
    """Structured symptom representation."""
    text: str
    severity: Optional[str] = None  # mild, moderate, severe
    onset_date: Optional[str] = None
    confidence: ExtractionConfidence = ExtractionConfidence.EXPLICIT
    source_span: str = ""
    
    def to_dict(self) -> Dict:
        return {k: (v.value if isinstance(v, Enum) else v) 
                for k, v in asdict(self).items()}


class ClinicalNoteExtractor:
    """
    Extract structured entities from clinical notes.
    
    Features:
    - Medication extraction (name, dose, frequency)
    - Diagnosis extraction with ICD codes
    - Symptom and vital extraction
    - Date and event temporal information
    - Care gap detection
    - Hallucination detection
    """
    
    # Common medication patterns
    MEDICATION_PATTERNS = [
        r'(?P<med>\w+)\s+(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|ml|units)',
        r'(?P<med>\w+)\s+(?P<dose>\d+(?:\.\d+)?%)',
        r'(?P<med>\w+)\s+(?P<dose>\d+)?(?:\s+)?(?P<freq>daily|BID|TID|QID|once|twice)',
    ]
    
    # Common diagnosis keywords
    DIAGNOSIS_KEYWORDS = {
        'diabetes': 'E11',
        'hypertension': 'I10',
        'pneumonia': 'J18',
        'heart failure': 'I50',
        'stroke': 'I63',
        'copd': 'J43',
        'asthma': 'J45',
        'uti': 'N39.0',
        'infection': 'B99',
        'sepsis': 'R65',
    }
    
    # Symptom keywords
    SYMPTOM_KEYWORDS = {
        'pain': ['chest pain', 'abdominal pain', 'back pain', 'headache'],
        'respiratory': ['shortness of breath', 'cough', 'dyspnea', 'wheezing'],
        'digestive': ['nausea', 'vomiting', 'diarrhea', 'constipation'],
        'neuro': ['dizziness', 'confusion', 'weakness', 'numbness'],
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize extractor.
        
        Args:
            llm_client: Optional LLM client (OpenAI/Anthropic)
        """
        self.llm_client = llm_client
        self.extracted_text_spans = {}  # Track source text for grounding
    
    def extract_medications(self, text: str) -> List[MedicationEntity]:
        """
        Extract medications from clinical note.
        
        Uses regex patterns and LLM if available for complex cases.
        """
        medications = []
        
        # Pattern-based extraction
        for pattern in self.MEDICATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                med_name = match.group('med')
                dose = match.group('dose') if 'dose' in match.groupdict() else None
                freq = match.group('freq') if 'freq' in match.groupdict() else None
                
                # Validate against common medications (basic check)
                confidence = self._assess_medication_confidence(med_name)
                
                medications.append(MedicationEntity(
                    name=med_name,
                    dose=dose,
                    frequency=freq,
                    confidence=confidence,
                    source_span=match.group(0)
                ))
        
        # LLM-based extraction for complex cases
        if self.llm_client and len(medications) < 3:
            llm_meds = self._extract_medications_llm(text)
            medications.extend(llm_meds)
        
        # Remove duplicates
        medications = self._deduplicate_medications(medications)
        
        return medications
    
    def extract_diagnoses(self, text: str) -> List[DiagnosisEntity]:
        """
        Extract diagnoses from clinical note.
        
        Maps to ICD codes where possible.
        """
        diagnoses = []
        
        # Keyword-based extraction
        for condition, icd_code in self.DIAGNOSIS_KEYWORDS.items():
            pattern = rf'\b{condition}\b'
            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(pattern, text, re.IGNORECASE)
                
                diagnoses.append(DiagnosisEntity(
                    text=condition.title(),
                    icd_code=icd_code,
                    confidence=ExtractionConfidence.INFERRED,
                    source_span=match.group(0) if match else condition
                ))
        
        # LLM-based extraction for comprehensive diagnosis list
        if self.llm_client:
            llm_diagnoses = self._extract_diagnoses_llm(text)
            diagnoses.extend(llm_diagnoses)
        
        # Remove duplicates
        diagnoses = self._deduplicate_diagnoses(diagnoses)
        
        return diagnoses
    
    def extract_symptoms(self, text: str) -> List[SymptomEntity]:
        """
        Extract symptoms and presenting complaints.
        """
        symptoms = []
        
        # Keyword matching
        for symptom_category, keywords in self.SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    match = re.search(rf'\b{keyword}\b', text, re.IGNORECASE)
                    symptoms.append(SymptomEntity(
                        text=keyword.title(),
                        confidence=ExtractionConfidence.EXPLICIT,
                        source_span=match.group(0) if match else keyword
                    ))
        
        return symptoms
    
    def extract_dates_and_events(self, text: str) -> Dict[str, str]:
        """
        Extract temporal information from clinical note.
        
        Returns:
            Dictionary with key dates and events
        """
        events = {}
        
        # Date patterns
        date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        
        # Find admission/discharge dates
        if 'admission' in text.lower():
            match = re.search(f'admission.*?({date_pattern})', text, re.IGNORECASE)
            if match:
                events['admission_date'] = match.group(1)
        
        if 'discharge' in text.lower():
            match = re.search(f'discharge.*?({date_pattern})', text, re.IGNORECASE)
            if match:
                events['discharge_date'] = match.group(1)
        
        return events
    
    def detect_care_gaps(self, text: str) -> List[Dict]:
        """
        Identify potential gaps in care from clinical note.
        
        Examples:
        - Missing recent lab work
        - Medication not mentioned
        - Unresolved symptoms
        """
        gaps = []
        
        # Missing HbA1c for diabetic
        if 'diabetes' in text.lower() and 'hba1c' not in text.lower():
            gaps.append({
                'issue': 'No recent HbA1c lab result',
                'severity': 'medium',
                'recommendation': 'Order HbA1c test'
            })
        
        # Missing BNP for heart failure
        if 'heart failure' in text.lower() and 'bnp' not in text.lower():
            gaps.append({
                'issue': 'No recent BNP lab result',
                'severity': 'medium',
                'recommendation': 'Order BNP test'
            })
        
        # Incomplete medication review
        if 'medication' not in text.lower().split('discharge')[0]:
            gaps.append({
                'issue': 'Medication list incomplete at discharge',
                'severity': 'high',
                'recommendation': 'Reconcile all medications'
            })
        
        return gaps
    
    def _assess_medication_confidence(self, med_name: str) -> ExtractionConfidence:
        """Assess confidence level for medication extraction."""
        # Common medications - high confidence
        common_meds = ['metformin', 'lisinopril', 'atorvastatin', 'omeprazole', 
                      'aspirin', 'metoprolol', 'amlodipine']
        
        if med_name.lower() in common_meds:
            return ExtractionConfidence.EXPLICIT
        
        # Unknown medication - uncertain
        return ExtractionConfidence.UNCERTAIN
    
    def _extract_medications_llm(self, text: str) -> List[MedicationEntity]:
        """Use LLM to extract medications (requires LLM client)."""
        if not self.llm_client:
            return []
        
        prompt = f"""
        Extract all medications from this clinical note.
        For each medication, provide: name, dose, frequency.
        Format as JSON list.
        
        Clinical note:
        {text}
        """
        
        try:
            response = self.llm_client.query(prompt)
            # Parse JSON response
            meds_data = json.loads(response)
            
            medications = []
            for med in meds_data:
                medications.append(MedicationEntity(
                    name=med.get('name', ''),
                    dose=med.get('dose'),
                    frequency=med.get('frequency'),
                    confidence=ExtractionConfidence.INFERRED,
                    source_span=med.get('text', '')
                ))
            
            return medications
        
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
    
    def _extract_diagnoses_llm(self, text: str) -> List[DiagnosisEntity]:
        """Use LLM to extract diagnoses."""
        if not self.llm_client:
            return []
        
        prompt = f"""
        Extract all diagnoses/conditions from this clinical note.
        For each, provide diagnosis name and estimate ICD-10 code if possible.
        Format as JSON list.
        
        Clinical note:
        {text}
        """
        
        try:
            response = self.llm_client.query(prompt)
            diagnoses_data = json.loads(response)
            
            diagnoses = []
            for diag in diagnoses_data:
                diagnoses.append(DiagnosisEntity(
                    text=diag.get('diagnosis', ''),
                    icd_code=diag.get('icd_code'),
                    confidence=ExtractionConfidence.INFERRED,
                    source_span=diag.get('text', '')
                ))
            
            return diagnoses
        
        except Exception as e:
            logger.error(f"LLM diagnosis extraction failed: {e}")
            return []
    
    def _deduplicate_medications(self, meds: List[MedicationEntity]) -> List[MedicationEntity]:
        """Remove duplicate medications."""
        seen = {}
        for med in meds:
            key = med.name.lower()
            if key not in seen:
                seen[key] = med
        return list(seen.values())
    
    def _deduplicate_diagnoses(self, diags: List[DiagnosisEntity]) -> List[DiagnosisEntity]:
        """Remove duplicate diagnoses."""
        seen = {}
        for diag in diags:
            key = diag.text.lower()
            if key not in seen:
                seen[key] = diag
        return list(seen.values())


class HallucinationDetector:
    """
    Detect and reduce hallucinations in LLM extractions.
    
    Strategies:
    1. Ground all extractions in source text
    2. Validate against medical knowledge bases
    3. Check consistency with other extracted entities
    4. Flag low-confidence or unsupported claims
    """
    
    def __init__(self):
        # Known medication database
        self.known_medications = {
            'metformin', 'lisinopril', 'atorvastatin', 'omeprazole',
            'aspirin', 'metoprolol', 'amlodipine', 'sertraline',
            'levothyroxine', 'albuterol', 'fluoxetine', 'warfarin'
        }
        
        # Known diagnoses
        self.known_diagnoses = {
            'diabetes', 'hypertension', 'heart failure', 'asthma',
            'copd', 'pneumonia', 'stroke', 'uti', 'sepsis',
            'acute myocardial infarction', 'angina'
        }
    
    def validate_extraction(self, text: str, extracted_entity: str, 
                           entity_type: str) -> Tuple[bool, float, str]:
        """
        Validate if extracted entity is grounded in text.
        
        Returns:
            (is_valid, confidence_score, reason)
        """
        text_lower = text.lower()
        entity_lower = extracted_entity.lower()
        
        # Check 1: Is entity explicitly mentioned?
        if entity_lower in text_lower:
            return True, 0.95, "Explicit mention in text"
        
        # Check 2: Is it a known entity?
        if entity_type == 'medication' and entity_lower in self.known_medications:
            return True, 0.70, "Known medication (inferred)"
        
        if entity_type == 'diagnosis' and entity_lower in self.known_diagnoses:
            return True, 0.70, "Known diagnosis (inferred)"
        
        # Check 3: Is it medically plausible?
        is_plausible = self._check_medical_plausibility(extracted_entity, entity_type)
        if is_plausible:
            return True, 0.50, "Medically plausible but not explicitly mentioned"
        
        # Likely hallucination
        return False, 0.0, "Not found in text, unknown entity, implausible"
    
    def _check_medical_plausibility(self, entity: str, entity_type: str) -> bool:
        """Check if entity is medically plausible."""
        # Basic sanity checks
        if len(entity) < 2:
            return False
        
        # No random characters
        if any(char in entity for char in ['@', '#', '$', '!', '?']):
            return False
        
        return True
    
    def flag_suspicious_extractions(self, extraction_result: Dict) -> List[Dict]:
        """
        Identify potentially hallucinated extractions.
        
        Returns list of suspicious items with confidence < threshold.
        """
        suspicious = []
        confidence_threshold = 0.65
        
        for med in extraction_result.get('medications', []):
            if med['confidence'] in ['uncertain', 'hallucinated']:
                suspicious.append({
                    'type': 'medication',
                    'entity': med['name'],
                    'reason': 'Low confidence extraction',
                    'recommendation': 'Verify with clinical team'
                })
        
        for diag in extraction_result.get('diagnoses', []):
            if diag['confidence'] in ['uncertain', 'hallucinated']:
                suspicious.append({
                    'type': 'diagnosis',
                    'entity': diag['text'],
                    'reason': 'Low confidence extraction',
                    'recommendation': 'Verify with clinical team'
                })
        
        return suspicious


# Example usage
if __name__ == "__main__":
    # Sample clinical note
    sample_note = """
    HISTORY OF PRESENT ILLNESS:
    Patient is a 65-year-old male with diabetes mellitus type 2 and 
    hypertension admitted with shortness of breath. On admission, patient
    was on metformin 500mg twice daily and lisinopril 10mg daily.
    
    HOSPITAL COURSE:
    Patient improved with diuretics. Discharged on metformin 500mg BID,
    lisinopril 10mg daily, and newly started on furosemide 40mg daily.
    
    DISCHARGE SUMMARY:
    Patient tolerating diet well. No HbA1c lab during this admission.
    Follow up with cardiologist in 2 weeks.
    """
    
    # Extract
    extractor = ClinicalNoteExtractor()
    
    medications = extractor.extract_medications(sample_note)
    print("Medications extracted:")
    for med in medications:
        print(f"  - {med.name} {med.dose} {med.frequency}")
    
    diagnoses = extractor.extract_diagnoses(sample_note)
    print("\nDiagnoses extracted:")
    for diag in diagnoses:
        print(f"  - {diag.text} ({diag.icd_code})")
    
    symptoms = extractor.extract_symptoms(sample_note)
    print("\nSymptoms extracted:")
    for sym in symptoms:
        print(f"  - {sym.text}")
    
    gaps = extractor.detect_care_gaps(sample_note)
    print("\nCare gaps detected:")
    for gap in gaps:
        print(f"  - {gap['issue']}")
    
    print("\n✓ Extraction complete")
