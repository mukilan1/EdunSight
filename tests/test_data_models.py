"""
Test Suite for EdunSight Data Models

Tests for data structures and model schemas.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.data_models import (
    StudentRecord, PredictionResult, ModelMetrics, DatasetInfo,
    FeatureSchema, ModelConfig, AppConstants
)


class TestStudentRecord:
    """Test StudentRecord data model"""
    
    def test_student_record_creation(self):
        """Test creating a student record"""
        record = StudentRecord(
            student_id="STU001",
            age=20,
            gender="M",
            course_id="AAA"
        )
        
        assert record.student_id == "STU001"
        assert record.age == 20
        assert record.gender == "M"
        assert record.course_id == "AAA"
    
    def test_student_record_to_dict(self):
        """Test converting student record to dictionary"""
        record = StudentRecord(
            student_id="STU001",
            age=20,
            gender="M",
            attendance_rate=0.85
        )
        
        data_dict = record.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["student_id"] == "STU001"
        assert data_dict["age"] == 20
        assert data_dict["gender"] == "M"
        assert data_dict["attendance_rate"] == 0.85


class TestPredictionResult:
    """Test PredictionResult data model"""
    
    def test_prediction_result_creation(self):
        """Test creating a prediction result"""
        result = PredictionResult(
            student_id="STU001",
            probability_pass=0.75,
            probability_fail=0.25,
            risk_category="Low",
            confidence_score=0.8,
            contributing_factors={"attendance": 0.3, "grades": 0.5},
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        assert result.student_id == "STU001"
        assert result.probability_pass == 0.75
        assert result.probability_fail == 0.25
        assert result.risk_category == "Low"
    
    def test_risk_level_property(self):
        """Test risk level property calculation"""
        # Low risk
        result_low = PredictionResult(
            student_id="STU001",
            probability_pass=0.8,
            probability_fail=0.2,
            risk_category="Low",
            confidence_score=0.8,
            contributing_factors={},
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"
        )
        assert result_low.risk_level == "Low Risk"
        
        # High risk
        result_high = PredictionResult(
            student_id="STU002",
            probability_pass=0.2,
            probability_fail=0.8,
            risk_category="High",
            confidence_score=0.9,
            contributing_factors={},
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"
        )
        assert result_high.risk_level == "High Risk"


class TestFeatureSchema:
    """Test FeatureSchema functionality"""
    
    def test_get_all_features(self):
        """Test getting all feature names"""
        features = FeatureSchema.get_all_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'age' in features
        assert 'gender' in features
    
    def test_validate_dataframe_valid(self):
        """Test dataframe validation with valid data"""
        # Create a valid dataframe
        data = {
            'age': [20, 25, 30],
            'gender': ['M', 'F', 'M'],
            'attendance_rate': [0.8, 0.9, 0.7],
            'final_result': ['Pass', 'Pass', 'Fail']
        }
        df = pd.DataFrame(data)
        
        issues = FeatureSchema.validate_dataframe(df)
        
        assert isinstance(issues, dict)
        assert 'missing_columns' in issues
        assert 'extra_columns' in issues
        assert 'type_mismatches' in issues
    
    def test_validate_dataframe_missing_columns(self):
        """Test dataframe validation with missing columns"""
        # Create dataframe with missing required columns
        data = {
            'age': [20, 25, 30],
            'gender': ['M', 'F', 'M']
            # Missing many required columns
        }
        df = pd.DataFrame(data)
        
        issues = FeatureSchema.validate_dataframe(df)
        
        assert len(issues['missing_columns']) > 0


class TestModelConfig:
    """Test ModelConfig constants"""
    
    def test_model_types(self):
        """Test model type constants"""
        assert ModelConfig.LIGHTGBM == "lightgbm"
        assert ModelConfig.RANDOM_FOREST == "random_forest"
        assert ModelConfig.LOGISTIC_REGRESSION == "logistic_regression"
    
    def test_performance_thresholds(self):
        """Test performance threshold constants"""
        assert ModelConfig.MIN_ACCURACY > 0
        assert ModelConfig.MIN_AUC > 0
        assert ModelConfig.MAX_INFERENCE_TIME_MS > 0


class TestAppConstants:
    """Test AppConstants"""
    
    def test_file_paths(self):
        """Test file path constants"""
        assert ModelConfig.RAW_DATA_DIR == "data/raw"
        assert ModelConfig.PROCESSED_DATA_DIR == "data/processed"
        assert ModelConfig.MODELS_DIR == "models"
    
    def test_supported_file_types(self):
        """Test supported file types"""
        assert '.csv' in AppConstants.SUPPORTED_FILE_TYPES
        assert '.xlsx' in AppConstants.SUPPORTED_FILE_TYPES


if __name__ == "__main__":
    pytest.main([__file__])
