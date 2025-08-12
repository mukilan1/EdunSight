"""
Test Configuration for EdunSight

Provides pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_training_data():
    """Create sample training dataset"""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'student_id': [f'STU_{i:04d}' for i in range(n_samples)],
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'course_id': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], n_samples),
        'region': np.random.choice(['East', 'West', 'North', 'South'], n_samples),
        'highest_education': np.random.choice(['A Level', 'HE Qualification'], n_samples),
        'disability': np.random.choice(['Y', 'N'], n_samples, p=[0.1, 0.9]),
        'total_clicks': np.random.randint(50, 2000, n_samples),
        'attendance_rate': np.random.uniform(0.3, 1.0, n_samples),
        'time_spent_online': np.random.uniform(10, 300, n_samples),
        'submission_delays': np.random.randint(0, 8, n_samples),
        'previous_attempts': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05]),
        'forum_posts': np.random.poisson(3, n_samples),
        'assessment_1': np.random.normal(65, 20, n_samples),
        'assessment_2': np.random.normal(68, 18, n_samples),
        'assessment_3': np.random.normal(70, 22, n_samples)
    }
    
    # Create realistic target based on features
    risk_score = (
        -0.4 * (data['attendance_rate'] - 0.5) +
        0.3 * (data['submission_delays'] / 8) +
        -0.3 * ((data['assessment_1'] + data['assessment_2'] + data['assessment_3']) / 300 - 0.5) +
        np.random.normal(0, 0.15, n_samples)
    )
    
    data['final_result'] = ['Pass' if score < 0 else 'Fail' for score in risk_score]
    
    return pd.DataFrame(data)


@pytest.fixture
def config_data():
    """Sample configuration data"""
    return {
        'project': {
            'name': 'EdunSight',
            'version': '1.0.0'
        },
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'max_features': 50,
            'chunk_size': 1000
        },
        'model': {
            'artifacts_dir': 'models',
            'lgb_params': {
                'n_estimators': 50,  # Smaller for testing
                'max_depth': 3,
                'random_state': 42
            },
            'test_size': 0.2,
            'random_state': 42
        },
        'performance': {
            'enable_onnx': False,  # Disable for testing
            'cache_size': 100
        }
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test discovery patterns
pytest_collection_modifyitems_hooks = []


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Fixtures for mock objects
@pytest.fixture
def mock_config_file(tmp_path, config_data):
    """Create a temporary config file"""
    import yaml
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_file)


@pytest.fixture
def mock_csv_file(tmp_path, sample_training_data):
    """Create a temporary CSV file"""
    csv_file = tmp_path / "test_data.csv"
    sample_training_data.to_csv(csv_file, index=False)
    return str(csv_file)
