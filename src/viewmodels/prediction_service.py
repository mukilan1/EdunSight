"""
Prediction ViewModel for EdunSight Application

This module handles prediction logic, model management, and result interpretation.
Following MVVM pattern - this is part of the ViewModel layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import joblib
import yaml

from ..models.data_models import PredictionResult, StudentRecord, ModelMetrics
from ..models.ml_models import BaseModel, ONNXModel, ModelFactory
from .data_processor import DataProcessor


class PredictionService:
    """Handles prediction operations and model management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.model = None
        self.onnx_model = None
        self.data_processor = DataProcessor(config_path)
        self.model_version = "1.0.0"
        self.use_onnx = self.config.get('performance', {}).get('enable_onnx', False)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def load_model(self, model_path: str, onnx_path: Optional[str] = None) -> None:
        """Load trained model for predictions"""
        try:
            # Load the main model
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                # New format with metadata
                self.model = model_data['model']
                self.model_version = model_data.get('version', self.model_version)
            else:
                # Legacy format - just the model
                self.model = model_data
            
            self.logger.info(f"Model loaded from {model_path}")
            
            # Load ONNX model if available and enabled
            if self.use_onnx and onnx_path and Path(onnx_path).exists():
                self.onnx_model = ONNXModel(onnx_path)
                self.onnx_model.load()
                self.logger.info(f"ONNX model loaded from {onnx_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, student_data: Dict[str, Any]) -> PredictionResult:
        """Make prediction for a single student"""
        # Convert to DataFrame for processing
        df = pd.DataFrame([student_data])
        
        # Process the data
        df_processed = self._preprocess_for_prediction(df)
        
        # Make prediction
        if self.use_onnx and self.onnx_model:
            probabilities = self.onnx_model.predict_proba(df_processed.values)
        else:
            if not self.model:
                raise ValueError("No model loaded")
            probabilities = self.model.predict_proba(df_processed)
        
        # Extract probabilities
        prob_fail = float(probabilities[0][0])
        prob_pass = float(probabilities[0][1])
        
        # Determine risk category
        risk_category = self._determine_risk_category(prob_fail)
        
        # Calculate confidence (how far from 0.5)
        confidence_score = float(abs(max(prob_fail, prob_pass) - 0.5) * 2)
        
        # Get contributing factors
        contributing_factors = self._get_contributing_factors(df_processed)
        
        return PredictionResult(
            student_id=student_data.get('student_id', 'unknown'),
            probability_pass=prob_pass,
            probability_fail=prob_fail,
            risk_category=risk_category,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            prediction_timestamp=datetime.now(),
            model_version=self.model_version
        )
    
    def predict_batch(self, students_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Make predictions for multiple students"""
        results = []
        
        # Convert to DataFrame
        df = pd.DataFrame(students_data)
        
        # Process the data
        df_processed = self._preprocess_for_prediction(df)
        
        # Make batch predictions
        if self.use_onnx and self.onnx_model:
            probabilities = self.onnx_model.predict_proba(df_processed.values)
        else:
            if not self.model:
                raise ValueError("No model loaded")
            probabilities = self.model.predict_proba(df_processed)
        
        # Process results
        for i, student_data in enumerate(students_data):
            prob_fail = float(probabilities[i][0])
            prob_pass = float(probabilities[i][1])
            
            risk_category = self._determine_risk_category(prob_fail)
            confidence_score = float(abs(max(prob_fail, prob_pass) - 0.5) * 2)
            contributing_factors = self._get_contributing_factors(df_processed.iloc[[i]])
            
            results.append(PredictionResult(
                student_id=student_data.get('student_id', f'student_{i}'),
                probability_pass=prob_pass,
                probability_fail=prob_fail,
                risk_category=risk_category,
                confidence_score=confidence_score,
                contributing_factors=contributing_factors,
                prediction_timestamp=datetime.now(),
                model_version=self.model_version
            ))
        
        self.logger.info(f"Batch prediction completed for {len(results)} students")
        return results
    
    def predict_from_csv(self, csv_path: str) -> List[PredictionResult]:
        """Make predictions from CSV file"""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        students_data = df.to_dict('records')
        
        # Make predictions
        return self.predict_batch(students_data)
    
    def _preprocess_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for prediction"""
        # Apply the same preprocessing as training
        df_processed = self.data_processor.clean_data(df)
        df_processed = self.data_processor.engineer_features(df_processed)
        df_processed = self.data_processor.encode_categorical_features(df_processed)
        df_processed = self.data_processor.scale_features(df_processed)
        
        # Ensure we have the same features as training
        if hasattr(self.model, 'feature_names_'):
            expected_features = self.model.feature_names_
            current_features = df_processed.columns.tolist()
            
            # Add missing features
            for feature in expected_features:
                if feature not in current_features:
                    df_processed[feature] = 0
            
            # Select only expected features in correct order
            df_processed = df_processed[expected_features]
        
        return df_processed
    
    def _determine_risk_category(self, prob_fail: float) -> str:
        """Determine risk category based on failure probability"""
        if prob_fail <= 0.3:
            return "Low"
        elif prob_fail <= 0.6:
            return "Medium"
        else:
            return "High"
    
    def _get_contributing_factors(self, df_processed: pd.DataFrame) -> Dict[str, float]:
        """Get the most important contributing factors for the prediction"""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_importance = self.model.feature_importances_
        feature_names = df_processed.columns.tolist()
        
        # Get feature values for this prediction
        feature_values = df_processed.iloc[0].values
        
        # Calculate contribution (importance * normalized value)
        contributions = {}
        for i, (name, importance, value) in enumerate(zip(feature_names, feature_importance, feature_values)):
            # Normalize value to 0-1 range for interpretation
            normalized_value = min(max(value, 0), 1) if not np.isnan(value) else 0
            contribution = importance * normalized_value
            contributions[name] = float(contribution)
        
        # Return top 5 contributing factors
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_contributions[:5])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "model_version": self.model_version,
            "onnx_enabled": self.use_onnx and self.onnx_model is not None,
            "features_count": len(getattr(self.model, 'feature_names_', [])),
            "created_at": datetime.now().isoformat()
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info["max_depth"] = self.model.max_depth
        
        return info
    
    def validate_input(self, student_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate input data for prediction"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['student_id']
        for field in required_fields:
            if field not in student_data or student_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check data types and ranges
        if 'age' in student_data:
            age = student_data['age']
            if not isinstance(age, (int, float)) or age < 0 or age > 100:
                warnings.append("Age should be between 0 and 100")
        
        if 'attendance_rate' in student_data:
            attendance = student_data['attendance_rate']
            if not isinstance(attendance, (int, float)) or attendance < 0 or attendance > 1:
                warnings.append("Attendance rate should be between 0 and 1")
        
        return {"errors": errors, "warnings": warnings}
    
    def explain_prediction(self, prediction_result: PredictionResult) -> Dict[str, Any]:
        """Provide detailed explanation of the prediction"""
        explanation = {
            "prediction_summary": {
                "risk_level": prediction_result.risk_category,
                "pass_probability": f"{prediction_result.probability_pass:.2%}",
                "confidence": f"{prediction_result.confidence_score:.2%}"
            },
            "key_factors": prediction_result.contributing_factors,
            "recommendations": self._generate_recommendations(prediction_result),
            "interpretation": self._interpret_prediction(prediction_result)
        }
        
        return explanation
    
    def _generate_recommendations(self, prediction_result: PredictionResult) -> List[str]:
        """Generate actionable recommendations based on prediction"""
        recommendations = []
        
        if prediction_result.risk_category == "High":
            recommendations.extend([
                "Immediate intervention recommended",
                "Schedule one-on-one meeting with academic advisor",
                "Consider additional tutoring support",
                "Monitor attendance and engagement closely"
            ])
        elif prediction_result.risk_category == "Medium":
            recommendations.extend([
                "Provide additional learning resources",
                "Encourage participation in study groups",
                "Regular check-ins with instructor",
                "Consider supplementary assignments"
            ])
        else:
            recommendations.extend([
                "Continue current learning approach",
                "Consider peer mentoring opportunities",
                "Maintain engagement levels"
            ])
        
        # Factor-specific recommendations
        factors = prediction_result.contributing_factors
        if 'attendance_rate' in factors and factors['attendance_rate'] > 0.1:
            recommendations.append("Focus on improving attendance")
        if 'submission_delays' in factors and factors['submission_delays'] > 0.1:
            recommendations.append("Work on time management and deadline adherence")
        
        return recommendations
    
    def _interpret_prediction(self, prediction_result: PredictionResult) -> str:
        """Provide human-readable interpretation of the prediction"""
        risk = prediction_result.risk_category
        confidence = prediction_result.confidence_score
        
        if confidence > 0.8:
            confidence_text = "very confident"
        elif confidence > 0.6:
            confidence_text = "confident"
        else:
            confidence_text = "somewhat confident"
        
        interpretation = (
            f"The model is {confidence_text} that this student has a "
            f"{risk.lower()} risk of academic failure. "
        )
        
        if risk == "High":
            interpretation += "Immediate attention and intervention are strongly recommended."
        elif risk == "Medium":
            interpretation += "Some additional support may be beneficial."
        else:
            interpretation += "The student appears to be on track for success."
        
        return interpretation
