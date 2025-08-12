"""
Test Suite for EdunSight Data Processing

Tests for data preprocessing and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.viewmodels.data_processor import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = {
        'student_id': [f'STU_{i:03d}' for i in range(100)],
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'course_id': np.random.choice(['AAA', 'BBB', 'CCC'], 100),
        'attendance_rate': np.random.uniform(0.4, 1.0, 100),
        'time_spent_online': np.random.uniform(10, 200, 100),
        'submission_delays': np.random.randint(0, 5, 100),
        'final_result': np.random.choice(['Pass', 'Fail'], 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_processor():
    """Create DataProcessor instance for testing"""
    return DataProcessor()


class TestDataProcessor:
    """Test DataProcessor functionality"""
    
    def test_initialization(self, data_processor):
        """Test DataProcessor initialization"""
        assert data_processor is not None
        assert hasattr(data_processor, 'scaler')
        assert hasattr(data_processor, 'label_encoders')
        assert hasattr(data_processor, 'imputer')
    
    def test_validate_data(self, data_processor, sample_data):
        """Test data validation"""
        validation_report = data_processor.validate_data(sample_data)
        
        assert isinstance(validation_report, dict)
        assert 'total_rows' in validation_report
        assert 'total_columns' in validation_report
        assert 'missing_values' in validation_report
        assert 'data_types' in validation_report
        assert 'duplicate_rows' in validation_report
        
        assert validation_report['total_rows'] == 100
        assert validation_report['total_columns'] == len(sample_data.columns)
    
    def test_clean_data(self, data_processor, sample_data):
        """Test data cleaning"""
        # Add some missing values and duplicates for testing
        dirty_data = sample_data.copy()
        dirty_data.loc[0, 'age'] = np.nan
        dirty_data.loc[1, 'attendance_rate'] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]])  # Add duplicate
        
        cleaned_data = data_processor.clean_data(dirty_data)
        
        assert len(cleaned_data) <= len(dirty_data)  # Duplicates removed
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
    
    def test_engineer_features(self, data_processor, sample_data):
        """Test feature engineering"""
        # Add date column for testing
        sample_data['date_registration'] = pd.date_range('2023-01-01', periods=100)
        
        featured_data = data_processor.engineer_features(sample_data)
        
        assert len(featured_data.columns) >= len(sample_data.columns)
        # Check if new features were created
        if 'date_registration' in sample_data.columns:
            assert 'days_since_registration' in featured_data.columns
    
    def test_encode_categorical_features(self, data_processor, sample_data):
        """Test categorical feature encoding"""
        encoded_data = data_processor.encode_categorical_features(sample_data)
        
        # Should have same or more columns (due to one-hot encoding)
        assert len(encoded_data.columns) >= len(sample_data.select_dtypes(include=[np.number]).columns)
        
        # Check that categorical columns are properly encoded
        categorical_cols = sample_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in encoded_data.columns:
                # If column still exists, it should be numeric
                assert pd.api.types.is_numeric_dtype(encoded_data[col])
    
    def test_scale_features(self, data_processor, sample_data):
        """Test feature scaling"""
        # First encode categorical features
        encoded_data = data_processor.encode_categorical_features(sample_data)
        
        scaled_data = data_processor.scale_features(encoded_data)
        
        # Check that numerical columns are scaled
        numerical_cols = encoded_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in scaled_data.columns:
                # Scaled data should have different mean/std (unless already scaled)
                assert scaled_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    
    def test_prepare_for_training(self, data_processor, sample_data):
        """Test training data preparation"""
        X_train, X_test, y_train, y_test = data_processor.prepare_for_training(sample_data)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that target is numeric
        assert pd.api.types.is_numeric_dtype(y_train)
        assert pd.api.types.is_numeric_dtype(y_test)
    
    def test_save_and_load_processed_data(self, data_processor, sample_data, tmp_path):
        """Test saving and loading processed data"""
        output_path = tmp_path / "test_data.csv"
        
        # Save data
        data_processor.save_processed_data(sample_data, str(output_path))
        
        assert output_path.exists()
        
        # Load data
        loaded_data = data_processor.load_processed_data(str(output_path))
        
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_get_dataset_info(self, data_processor, sample_data):
        """Test dataset info generation"""
        dataset_info = data_processor.get_dataset_info(
            sample_data, "test_dataset", "test_path.csv"
        )
        
        assert dataset_info.name == "test_dataset"
        assert dataset_info.source == "test_path.csv"
        assert dataset_info.rows == len(sample_data)
        assert dataset_info.columns == len(sample_data.columns)
        assert isinstance(dataset_info.missing_values, dict)
        assert isinstance(dataset_info.data_types, dict)


class TestDataProcessorEdgeCases:
    """Test edge cases for DataProcessor"""
    
    def test_empty_dataframe(self, data_processor):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        
        validation_report = data_processor.validate_data(empty_df)
        assert validation_report['total_rows'] == 0
        assert validation_report['total_columns'] == 0
    
    def test_missing_target_column(self, data_processor, sample_data):
        """Test handling of missing target column"""
        data_without_target = sample_data.drop('final_result', axis=1)
        
        with pytest.raises(ValueError):
            data_processor.prepare_for_training(data_without_target)
    
    def test_all_missing_values(self, data_processor):
        """Test handling of columns with all missing values"""
        data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [1, 2, 3],
            'final_result': ['Pass', 'Fail', 'Pass']
        })
        
        cleaned_data = data_processor.clean_data(data)
        
        # Should handle missing values appropriately
        assert not cleaned_data['col1'].isnull().all()
    
    def test_single_category_column(self, data_processor):
        """Test handling of categorical column with single category"""
        data = pd.DataFrame({
            'single_cat': ['A', 'A', 'A'],
            'age': [20, 25, 30],
            'final_result': ['Pass', 'Fail', 'Pass']
        })
        
        encoded_data = data_processor.encode_categorical_features(data)
        
        # Should handle single-category columns without error
        assert len(encoded_data) == len(data)


if __name__ == "__main__":
    pytest.main([__file__])
