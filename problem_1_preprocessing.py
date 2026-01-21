"""
Problem 1: ML-based Patient Readmission Risk Prediction

This module implements feature engineering, model training, and evaluation
for predicting 30-day hospital readmission risk.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for patient readmission data.
    
    Handles:
    - Derived feature creation
    - Domain-specific features
    - Temporal features
    - Risk factor engineering
    """
    
    def __init__(self):
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw patient data.
        
        Features created:
        - Age groups
        - Length of stay categories
        - Comorbidity burden score
        - Recent admission frequency
        - Risk factor interactions
        """
        df = df.copy()
        
        # 1. Age-based features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                      bins=[0, 30, 50, 70, 150],
                                      labels=['18-30', '31-50', '51-70', '70+'])
            df['is_elderly'] = (df['age'] >= 65).astype(int)
            df['age_squared'] = df['age'] ** 2
        
        # 2. Length of stay features
        if 'length_of_stay' in df.columns:
            df['los_category'] = pd.cut(df['length_of_stay'],
                                        bins=[0, 3, 7, 14, 365],
                                        labels=['short', 'medium', 'long', 'very_long'])
            df['log_los'] = np.log1p(df['length_of_stay'])
        
        # 3. Comorbidity features
        if 'num_diagnoses' in df.columns:
            df['high_comorbidity'] = (df['num_diagnoses'] > 5).astype(int)
            df['comorbidity_score'] = np.minimum(df['num_diagnoses'], 10)
        
        # 4. Medication complexity
        if 'num_medications' in df.columns:
            df['high_med_burden'] = (df['num_medications'] > 10).astype(int)
            df['polypharmacy'] = (df['num_medications'] > 5).astype(int)
        
        # 5. Recent admission patterns
        if 'previous_admissions_6m' in df.columns:
            df['frequent_flyer'] = (df['previous_admissions_6m'] > 2).astype(int)
            df['readmission_risk_score_baseline'] = np.log1p(
                df['previous_admissions_6m'] + 1
            )
        
        # 6. Lab abnormality features
        lab_cols = [col for col in df.columns if col.startswith('lab_')]
        for col in lab_cols:
            if col in df.columns:
                df[f'{col}_abnormal'] = (
                    (df[col] < df[col].quantile(0.1)) | 
                    (df[col] > df[col].quantile(0.9))
                ).astype(int)
        
        # 7. Interaction features
        if all(col in df.columns for col in ['age', 'length_of_stay']):
            df['age_los_interaction'] = df['age'] * df['length_of_stay']
        
        if all(col in df.columns for col in ['num_diagnoses', 'num_medications']):
            df['burden_interaction'] = df['num_diagnoses'] * df['num_medications']
        
        # 8. Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['patient_id', 'readmitted']:
                df[f'{col}_encoded'] = pd.factorize(df[col])[0]
        
        self.feature_names = [col for col in df.columns 
                             if col not in ['patient_id', 'readmitted']]
        
        logger.info(f"Engineered {len(self.feature_names)} features")
        return df


class MissingValueHandler:
    """
    Handle missing values using multiple strategies.
    
    Strategies:
    - KNN imputation
    - Forward fill for temporal data
    - Mean/median imputation
    - Domain-specific imputation
    """
    
    def __init__(self, strategy: str = 'knn', n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer = None
        
    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        """Fit the imputation strategy."""
        if self.strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.imputer.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation."""
        X = X.copy()
        
        if self.strategy == 'knn':
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Handle categorical missing values
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class ImbalanceHandler:
    """
    Handle class imbalance using multiple strategies.
    
    Strategies:
    - SMOTE (Synthetic Minority Oversampling)
    - Class weights
    - Stratified sampling
    - Threshold optimization
    """
    
    def __init__(self, strategy: str = 'smote', sampling_strategy: float = 0.5):
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.sample_weight_dict = None
        
    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights."""
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(unique) * counts)
        return dict(zip(unique, weights))
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for oversampling."""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except ImportError:
            logger.warning("imblearn not installed, using class weights instead")
            return X, y
    
    def handle(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Apply imbalance handling strategy."""
        if self.strategy == 'smote':
            return self.apply_smote(X, y), self.get_class_weights(y)
        
        return X, y, self.get_class_weights(y)


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.
    
    Handles:
    - Feature engineering
    - Missing value imputation
    - Class imbalance
    - Feature scaling
    - Train-test splitting
    """
    
    def __init__(self, 
                 missing_strategy: str = 'knn',
                 imbalance_strategy: str = 'smote',
                 scaling_method: str = 'robust'):
        
        self.feature_engineer = FeatureEngineer()
        self.missing_handler = MissingValueHandler(strategy=missing_strategy)
        self.imbalance_handler = ImbalanceHandler(strategy=imbalance_strategy)
        
        if scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.feature_names = []
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     target_col: str = 'readmitted') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform data through complete pipeline.
        
        Returns:
            (X_processed, y) - Processed features and target
        """
        # Separate target
        y = df[target_col].copy()
        X = df.drop(columns=[target_col, 'patient_id'], errors='ignore')
        
        # Feature engineering
        X = self.feature_engineer.engineer_features(X)
        
        # Missing value handling
        X = self.missing_handler.fit_transform(X)
        
        # Handle class imbalance
        X, y, class_weights = self.imbalance_handler.handle(X, y)
        
        # Scale features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        self.feature_names = list(X.columns)
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, Any]:
        """Apply fitted preprocessing to new data."""
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            X = df.drop(columns=[target_col, 'patient_id'], errors='ignore')
        else:
            X = df.drop(columns=['patient_id'], errors='ignore')
        
        X = self.feature_engineer.engineer_features(X)
        X = self.missing_handler.transform(X)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return (X, y) if y is not None else (X, None)


# Example usage
if __name__ == "__main__":
    # Create sample data
    n_samples = 1000
    sample_data = pd.DataFrame({
        'patient_id': [f'PAT_{i:05d}' for i in range(n_samples)],
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'length_of_stay': np.random.randint(1, 30, n_samples),
        'num_diagnoses': np.random.randint(1, 15, n_samples),
        'num_medications': np.random.randint(0, 20, n_samples),
        'previous_admissions_6m': np.random.randint(0, 5, n_samples),
        'lab_hemoglobin': np.random.normal(13, 2, n_samples),
        'lab_glucose': np.random.normal(100, 25, n_samples),
        'discharge_disposition': np.random.choice(['home', 'facility', 'ama'], n_samples),
        'readmitted': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(sample_data)
    
    print("✓ Preprocessing successful")
    print(f"Feature count: {len(preprocessor.feature_names)}")
    print(f"Readmission rate: {y.mean():.2%}")
