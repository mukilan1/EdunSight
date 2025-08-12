"""
Test Suite for EdunSight Prediction Service

Tests for prediction functionality and model management.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.viewmodels.prediction_service import PredictionService
from src.models.data_models import PredictionResult


@pytest.fixture
def sample_student_data():
    """Create sample student data for testing"""
    return {
        'student_id': 'STU001',
        'age': 20,
        'gender': 'M',
        'course_id': 'AAA',
        'attendance_rate': 0.85,
        'time_spent_online': 120.5,
        'submission_delays': 1,
        'previous_attempts': 0
    }


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing"""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])  # [fail_prob, pass_prob]
    model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
    model.feature_names_ = ['age', 'gender', 'attendance_rate', 'course_id']
    return model


@pytest.fixture
def prediction_service_with_mock(mock_model):
    """Create PredictionService with mock model"""
    service = PredictionService()
    service.model = mock_model
    service.model_version = "test_1.0.0"
    return service


class TestPredictionService:
    """Test PredictionService functionality"""
    
    def test_initialization(self):
        """Test PredictionService initialization"""
        service = PredictionService()
        
        assert service is not None
        assert hasattr(service, 'model')
        assert hasattr(service, 'data_processor')
        assert hasattr(service, 'model_version')
    
    @patch('joblib.load')
    def test_load_model_new_format(self, mock_joblib_load):
        """Test loading model with new format"""
        mock_model_data = {
            'model': Mock(),
            'version': '2.0.0',
            'metadata': {'test': 'data'}
        }
        mock_joblib_load.return_value = mock_model_data
        
        service = PredictionService()
        service.load_model('test_model.joblib')
        
        assert service.model == mock_model_data['model']
        assert service.model_version == '2.0.0'
        mock_joblib_load.assert_called_once_with('test_model.joblib')
    
    def test_predict_single(self, prediction_service_with_mock, sample_student_data):
        """Test single student prediction"""
        with patch.object(prediction_service_with_mock, '_preprocess_for_prediction') as mock_preprocess:
            # Mock preprocessing to return DataFrame
            mock_preprocess.return_value = pd.DataFrame([[20, 1, 0.85, 1]], 
                                                       columns=['age', 'gender', 'attendance_rate', 'course_id'])
            
            result = prediction_service_with_mock.predict_single(sample_student_data)
            
            assert isinstance(result, PredictionResult)
            assert result.student_id == 'STU001'
            assert result.probability_pass == 0.7
            assert result.probability_fail == 0.3
            assert result.risk_category in ['Low', 'Medium', 'High']
            assert 0 <= result.confidence_score <= 1
            assert isinstance(result.contributing_factors, dict)
    
    def test_predict_batch(self, prediction_service_with_mock, sample_student_data):
        """Test batch prediction"""
        batch_data = [sample_student_data, {**sample_student_data, 'student_id': 'STU002'}]
        
        with patch.object(prediction_service_with_mock, '_preprocess_for_prediction') as mock_preprocess:
            # Mock preprocessing
            mock_preprocess.return_value = pd.DataFrame(
                [[20, 1, 0.85, 1], [22, 0, 0.90, 2]], 
                columns=['age', 'gender', 'attendance_rate', 'course_id']
            )
            
            # Mock model to return batch predictions
            prediction_service_with_mock.model.predict_proba.return_value = np.array([
                [0.3, 0.7], [0.2, 0.8]
            ])
            
            results = prediction_service_with_mock.predict_batch(batch_data)
            
            assert len(results) == 2
            assert all(isinstance(r, PredictionResult) for r in results)
            assert results[0].student_id == 'STU001'
            assert results[1].student_id == 'STU002'
    
    def test_determine_risk_category(self, prediction_service_with_mock):
        """Test risk category determination"""
        # Test low risk
        assert prediction_service_with_mock._determine_risk_category(0.2) == "Low"
        
        # Test medium risk
        assert prediction_service_with_mock._determine_risk_category(0.5) == "Medium"
        
        # Test high risk
        assert prediction_service_with_mock._determine_risk_category(0.8) == "High"
    
    def test_get_contributing_factors(self, prediction_service_with_mock):
        """Test contributing factors extraction"""
        df_processed = pd.DataFrame([[20, 1, 0.85, 1]], 
                                   columns=['age', 'gender', 'attendance_rate', 'course_id'])
        
        factors = prediction_service_with_mock._get_contributing_factors(df_processed)
        
        assert isinstance(factors, dict)
        assert len(factors) <= 5  # Should return top 5 factors
        
        # Check that all values are floats
        for factor, value in factors.items():
            assert isinstance(value, float)
    
    def test_get_model_info_with_model(self, prediction_service_with_mock):
        """Test getting model info when model is loaded"""
        info = prediction_service_with_mock.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'model_version' in info
        assert 'onnx_enabled' in info
        assert info['model_version'] == 'test_1.0.0'
    
    def test_get_model_info_without_model(self):
        """Test getting model info when no model is loaded"""
        service = PredictionService()
        info = service.get_model_info()
        
        assert 'error' in info
        assert info['error'] == 'No model loaded'
    
    def test_validate_input_valid(self, prediction_service_with_mock, sample_student_data):
        """Test input validation with valid data"""
        validation = prediction_service_with_mock.validate_input(sample_student_data)
        
        assert 'errors' in validation
        assert 'warnings' in validation
        assert len(validation['errors']) == 0  # Should be no errors
    
    def test_validate_input_missing_required(self, prediction_service_with_mock):
        """Test input validation with missing required fields"""
        invalid_data = {'age': 20, 'gender': 'M'}  # Missing student_id
        
        validation = prediction_service_with_mock.validate_input(invalid_data)
        
        assert len(validation['errors']) > 0
        assert any('student_id' in error for error in validation['errors'])
    
    def test_validate_input_invalid_ranges(self, prediction_service_with_mock):
        """Test input validation with invalid value ranges"""
        invalid_data = {
            'student_id': 'STU001',
            'age': -5,  # Invalid age
            'attendance_rate': 1.5  # Invalid attendance rate
        }
        
        validation = prediction_service_with_mock.validate_input(invalid_data)
        
        assert len(validation['warnings']) > 0
    
    def test_explain_prediction(self, prediction_service_with_mock, sample_student_data):
        """Test prediction explanation"""
        # Create a sample prediction result
        prediction_result = PredictionResult(
            student_id='STU001',
            probability_pass=0.7,
            probability_fail=0.3,
            risk_category='Low',
            confidence_score=0.8,
            contributing_factors={'attendance_rate': 0.4, 'age': 0.2},
            prediction_timestamp=datetime.now(),
            model_version='test_1.0.0'
        )
        
        explanation = prediction_service_with_mock.explain_prediction(prediction_result)
        
        assert isinstance(explanation, dict)
        assert 'prediction_summary' in explanation
        assert 'key_factors' in explanation
        assert 'recommendations' in explanation
        assert 'interpretation' in explanation
        
        # Check prediction summary
        summary = explanation['prediction_summary']
        assert 'risk_level' in summary
        assert 'pass_probability' in summary
        assert 'confidence' in summary
    
    def test_generate_recommendations_high_risk(self, prediction_service_with_mock):
        """Test recommendation generation for high risk student"""
        prediction_result = PredictionResult(
            student_id='STU001',
            probability_pass=0.2,
            probability_fail=0.8,
            risk_category='High',
            confidence_score=0.9,
            contributing_factors={'attendance_rate': 0.6, 'submission_delays': 0.4},
            prediction_timestamp=datetime.now(),
            model_version='test_1.0.0'
        )
        
        recommendations = prediction_service_with_mock._generate_recommendations(prediction_result)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('intervention' in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_low_risk(self, prediction_service_with_mock):
        """Test recommendation generation for low risk student"""
        prediction_result = PredictionResult(
            student_id='STU001',
            probability_pass=0.9,
            probability_fail=0.1,
            risk_category='Low',
            confidence_score=0.8,
            contributing_factors={},
            prediction_timestamp=datetime.now(),
            model_version='test_1.0.0'
        )
        
        recommendations = prediction_service_with_mock._generate_recommendations(prediction_result)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('continue' in rec.lower() for rec in recommendations)
    
    def test_interpret_prediction(self, prediction_service_with_mock):
        """Test prediction interpretation"""
        prediction_result = PredictionResult(
            student_id='STU001',
            probability_pass=0.7,
            probability_fail=0.3,
            risk_category='Medium',
            confidence_score=0.75,
            contributing_factors={},
            prediction_timestamp=datetime.now(),
            model_version='test_1.0.0'
        )
        
        interpretation = prediction_service_with_mock._interpret_prediction(prediction_result)
        
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert 'medium risk' in interpretation.lower()
        assert 'confident' in interpretation.lower()


class TestPredictionServiceEdgeCases:
    """Test edge cases for PredictionService"""
    
    def test_predict_without_model(self):
        """Test prediction without loaded model"""
        service = PredictionService()
        sample_data = {'student_id': 'STU001', 'age': 20}
        
        with pytest.raises(ValueError):
            service.predict_single(sample_data)
    
    def test_predict_with_missing_features(self, prediction_service_with_mock):
        """Test prediction with missing features"""
        # Mock preprocessing to return DataFrame with missing features
        with patch.object(prediction_service_with_mock, '_preprocess_for_prediction') as mock_preprocess:
            # Return DataFrame with fewer features than expected
            mock_preprocess.return_value = pd.DataFrame([[20, 1]], columns=['age', 'gender'])
            
            sample_data = {'student_id': 'STU001', 'age': 20}
            
            # Should handle missing features gracefully
            result = prediction_service_with_mock.predict_single(sample_data)
            assert isinstance(result, PredictionResult)


if __name__ == "__main__":
    pytest.main([__file__])
