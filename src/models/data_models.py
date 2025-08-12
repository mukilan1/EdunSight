"""
Data Models for EdunSight Application

This module contains data structures and schemas used throughout the application.
Following MVVM pattern - this represents the Model layer.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class StudentRecord:
    """Individual student record data structure"""
    student_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    course_id: Optional[str] = None
    assessment_scores: Optional[List[float]] = None
    attendance_rate: Optional[float] = None
    time_spent_online: Optional[float] = None
    submission_delays: Optional[int] = None
    previous_attempts: Optional[int] = None
    final_result: Optional[str] = None  # Pass/Fail/Withdrawn/Distinction
    
    # Additional fields for comprehensive student modeling
    previous_grades: Optional[List[float]] = None
    assignment_scores: Optional[List[float]] = None
    participation_score: Optional[float] = None
    study_hours_per_week: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing"""
        return {
            'student_id': self.student_id,
            'age': self.age,
            'gender': self.gender,
            'course_id': self.course_id,
            'assessment_scores': self.assessment_scores,
            'attendance_rate': self.attendance_rate,
            'time_spent_online': self.time_spent_online,
            'submission_delays': self.submission_delays,
            'previous_attempts': self.previous_attempts,
            'final_result': self.final_result
        }


@dataclass
class PredictionResult:
    """Prediction result data structure"""
    student_id: str
    probability_pass: float
    probability_fail: float
    risk_category: str  # Low/Medium/High
    confidence_score: float
    contributing_factors: Dict[str, float]
    prediction_timestamp: datetime
    model_version: str
    
    @property
    def risk_level(self) -> str:
        """Determine risk level based on probability"""
        if self.probability_fail <= 0.3:
            return "Low Risk"
        elif self.probability_fail <= 0.6:
            return "Medium Risk"
        else:
            return "High Risk"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    training_time: float
    inference_time: float
    model_size_mb: float


@dataclass
class DatasetInfo:
    """Dataset information and metadata"""
    name: str
    source: str
    rows: int
    columns: int
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    target_distribution: Dict[str, int]
    last_updated: datetime
    file_path: str


class FeatureSchema:
    """Schema definition for features"""
    
    # Numerical features
    NUMERICAL_FEATURES = [
        'age',
        'attendance_rate',
        'time_spent_online',
        'avg_assessment_score',
        'submission_delays',
        'previous_attempts',
        'days_since_last_access',
        'total_clicks',
        'forum_posts'
    ]
    
    # Categorical features
    CATEGORICAL_FEATURES = [
        'gender',
        'course_id',
        'region',
        'highest_education',
        'disability',
        'age_band'
    ]
    
    # Target variable
    TARGET_VARIABLE = 'final_result'
    
    # Target mapping
    TARGET_MAPPING = {
        'Pass': 1,
        'Distinction': 1,
        'Fail': 0,
        'Withdrawn': 0
    }
    
    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get all feature names"""
        return cls.NUMERICAL_FEATURES + cls.CATEGORICAL_FEATURES
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate dataframe against schema"""
        issues = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }
        
        expected_cols = set(cls.get_all_features() + [cls.TARGET_VARIABLE])
        actual_cols = set(df.columns)
        
        issues['missing_columns'] = list(expected_cols - actual_cols)
        issues['extra_columns'] = list(actual_cols - expected_cols)
        
        return issues


class ModelConfig:
    """Model configuration constants"""
    
    # Model types
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    
    # File extensions
    JOBLIB_EXT = ".joblib"
    ONNX_EXT = ".onnx"
    PICKLE_EXT = ".pkl"
    
    # Performance thresholds
    MIN_ACCURACY = 0.75
    MIN_AUC = 0.80
    MAX_INFERENCE_TIME_MS = 100
    
    # Feature selection
    MAX_FEATURES = 50
    MIN_FEATURE_IMPORTANCE = 0.001


class AppConstants:
    """Application constants"""
    
    # File paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 1000
    
    # UI settings
    MAX_UPLOAD_SIZE_MB = 200
    SUPPORTED_FILE_TYPES = ['.csv', '.xlsx', '.json']
    
    # Performance settings
    CHUNK_SIZE = 10000
    N_JOBS = -1
