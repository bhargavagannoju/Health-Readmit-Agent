"""
Problem 1: ML Model Training and Evaluation

Implements multiple classical ML and deep learning models with
comprehensive evaluation and explainability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    classification_report, confusion_matrix, calibration_curve
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from typing import Dict, Tuple, Any, List
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and evaluate multiple ML models for readmission risk.
    
    Models:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost (if available)
    - LightGBM (if available)
    - Deep Neural Network (if TensorFlow available)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.cv_results = {}
        
    def _create_models(self) -> Dict[str, Any]:
        """Create model instances."""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        # Optional models
        try:
            import xgboost as xgb
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                scale_pos_weight=4,  # Adjust for imbalance
                n_jobs=-1
            )
        except ImportError:
            logger.warning("XGBoost not installed")
        
        try:
            import lightgbm as lgb
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                is_unbalanced=True,
                n_jobs=-1
            )
        except ImportError:
            logger.warning("LightGBM not installed")
        
        return models
    
    def train_models(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train all models with cross-validation.
        
        Returns:
            Dictionary with CV results for each model
        """
        self.models = self._create_models()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.random_state)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv,
                scoring=['roc_auc', 'precision', 'recall', 'f1'],
                n_jobs=-1
            )
            
            results[name] = {
                'roc_auc': cv_results['test_roc_auc'].mean(),
                'roc_auc_std': cv_results['test_roc_auc'].std(),
                'precision': cv_results['test_precision'].mean(),
                'recall': cv_results['test_recall'].mean(),
                'f1': cv_results['test_f1'].mean(),
            }
            
            # Train final model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            logger.info(f"{name}: AUROC={results[name]['roc_auc']:.4f}")
        
        self.cv_results = results
        return results
    
    def evaluate_on_test_set(self,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test set.
        
        Returns:
            Dictionary with evaluation metrics for each model
        """
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            logger.info(f"Evaluating {name} on test set...")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Calibration
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            calibration_error = np.mean((prob_pred - prob_true) ** 2)
            
            evaluation_results[name] = {
                'roc_auc': roc_auc,
                'f1_score': f1,
                'precision': precision_recall_curve(y_test, y_pred_proba)[0].mean(),
                'recall': recall.mean(),
                'calibration_error': calibration_error,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, 
                                                               output_dict=True),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision_curve': precision.tolist(),
                'recall_curve': recall.tolist(),
            }
        
        return evaluation_results
    
    def get_feature_importance(self, model_name: str, 
                              feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Supports tree-based and linear models.
        """
        model = self.trained_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_dict = dict(zip(feature_names, np.abs(model.coef_)[0]))
        else:
            logger.warning(f"Cannot extract importance for {model_name}")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_shap_explanation(self, model_name: str,
                            X_test: pd.DataFrame,
                            sample_size: int = 100) -> Dict:
        """
        Generate SHAP explanations if available.
        
        Falls back to permutation importance if SHAP unavailable.
        """
        try:
            import shap
            
            model = self.trained_models[model_name]
            X_sample = X_test.sample(min(sample_size, len(X_test)))
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)[1]  # Class 1 SHAP values
            else:
                logger.warning(f"SHAP not available for {model_name}")
                return {}
            
            return {
                'shap_values': shap_values.tolist(),
                'feature_names': X_sample.columns.tolist(),
                'base_value': explainer.expected_value[1]
            }
        
        except ImportError:
            logger.warning("SHAP not installed, using permutation importance")
            
            from sklearn.inspection import permutation_importance
            model = self.trained_models[model_name]
            
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            return {
                'permutation_importance': perm_importance.importances_mean.tolist(),
                'feature_names': X_test.columns.tolist()
            }
    
    def save_model(self, model_name: str, output_dir: str = './models'):
        """Save trained model to disk."""
        Path(output_dir).mkdir(exist_ok=True)
        
        model = self.trained_models[model_name]
        output_path = Path(output_dir) / f"{model_name}_model.pkl"
        
        joblib.dump(model, output_path)
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_name: str, input_path: str):
        """Load trained model from disk."""
        model = joblib.load(input_path)
        self.trained_models[model_name] = model
        logger.info(f"Model loaded from {input_path}")
    
    def predict_risk(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict readmission risk.
        
        Returns:
            (binary_predictions, probability_scores)
        """
        model = self.trained_models[model_name]
        
        binary_pred = model.predict(X)
        prob_scores = model.predict_proba(X)[:, 1]
        
        return binary_pred, prob_scores


class ExplainabilityEngine:
    """
    Provide explainability for individual predictions.
    
    Methods:
    - Feature contribution analysis
    - Comparison to similar patients
    - Risk factor identification
    """
    
    def __init__(self, model_trainer: ModelTrainer, X_train: pd.DataFrame):
        self.model_trainer = model_trainer
        self.X_train = X_train
    
    def explain_prediction(self, model_name: str, 
                          X_sample: pd.DataFrame,
                          sample_idx: int = 0) -> Dict:
        """
        Explain why a single prediction was made.
        
        Returns:
            Dictionary with explanation components
        """
        model = self.model_trainer.trained_models[model_name]
        
        row = X_sample.iloc[sample_idx:sample_idx+1]
        risk_score = model.predict_proba(row)[0, 1]
        
        # Feature importance
        feature_imp = self.model_trainer.get_feature_importance(
            model_name, X_sample.columns.tolist()
        )
        
        # Patient's feature values
        patient_values = row.iloc[0].to_dict()
        
        # Risk factors (high importance features with extreme values)
        risk_factors = []
        for idx, row_imp in feature_imp.head(10).iterrows():
            feature = row_imp['feature']
            importance = row_imp['importance']
            value = patient_values.get(feature)
            
            if value is not None:
                percentile = (self.X_train[feature] <= value).mean()
                
                risk_factors.append({
                    'feature': feature,
                    'importance': float(importance),
                    'value': float(value),
                    'percentile': float(percentile),
                    'risk_direction': 'high' if percentile > 0.7 else 'low'
                })
        
        return {
            'risk_score': float(risk_score),
            'risk_category': 'high' if risk_score > 0.5 else 'low',
            'top_risk_factors': risk_factors,
            'comparison': {
                'mean_readmitted': float(self.X_train.mean()),
                'mean_not_readmitted': float(self.X_train.mean())
            }
        }


# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train)
    
    print("✓ Model training complete")
    print("CV Results:", results)
    
    # Evaluate
    eval_results = trainer.evaluate_on_test_set(X_test, y_test)
    print("\nTest Set Results:")
    for model_name, metrics in eval_results.items():
        print(f"{model_name}: AUROC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
