"""
ML Model Classes for EdunSight Application

This module contains machine learning model implementations.
Following MVVM pattern - this is part of the Model layer.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import onnxruntime as ort

# Make ONNX conversion optional
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("skl2onnx not available. ONNX conversion features will be disabled.")

from .data_models import ModelMetrics, PredictionResult, ModelConfig


class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.metrics = None
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance"""
        start_time = time.time()
        predictions = self.predict(X)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        probabilities = self.predict_proba(X)
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, average='weighted'),
            recall=recall_score(y, predictions, average='weighted'),
            f1_score=f1_score(y, predictions, average='weighted'),
            auc_roc=roc_auc_score(y, probabilities[:, 1]),
            confusion_matrix=self._compute_confusion_matrix(y, predictions),
            feature_importance=self.get_feature_importance(),
            training_time=0.0,  # Set during training
            inference_time=inference_time,
            model_size_mb=self._get_model_size()
        )
        
        self.metrics = metrics
        return metrics
    
    def save(self, file_path: str) -> None:
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, file_path)
        self.logger.info(f"Model saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """Load model from file"""
        data = joblib.load(file_path)
        self.model = data['model']
        self.model_name = data['model_name']
        self.config = data['config']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        self.logger.info(f"Model loaded from {file_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained or self.feature_names is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, importance))
        else:
            return {}
    
    def _compute_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> List[List[int]]:
        """Compute confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()
    
    def _get_model_size(self) -> float:
        """Estimate model size in MB"""
        try:
            import sys
            return sys.getsizeof(self.model) / (1024 * 1024)
        except:
            return 0.0


class LightGBMModel(BaseModel):
    """LightGBM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LightGBM", config)
        self.model = lgb.LGBMClassifier(**config.get('lgb_params', {}))
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LightGBM model"""
        start_time = time.time()
        self.logger.info("Starting LightGBM training...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_trained = True
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        if self.metrics:
            self.metrics.training_time = training_time
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RandomForest", config)
        rf_params = config.get('rf_params', {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        })
        self.model = RandomForestClassifier(**rf_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Random Forest model"""
        start_time = time.time()
        self.logger.info("Starting Random Forest training...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_trained = True
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        if self.metrics:
            self.metrics.training_time = training_time
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LogisticRegression", config)
        lr_params = config.get('lr_params', {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        })
        self.model = LogisticRegression(**lr_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Logistic Regression model"""
        start_time = time.time()
        self.logger.info("Starting Logistic Regression training...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_trained = True
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        if self.metrics:
            self.metrics.training_time = training_time
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)


class ONNXModel:
    """ONNX Runtime model for fast inference"""
    
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.feature_names = None
        self.logger = logging.getLogger(f"{__name__}.ONNXModel")
    
    def load(self) -> None:
        """Load ONNX model"""
        self.session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.logger.info(f"ONNX model loaded from {self.onnx_path}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction using ONNX Runtime"""
        if self.session is None:
            raise ValueError("ONNX model must be loaded first")
        
        # Ensure input is float32
        X = X.astype(np.float32)
        result = self.session.run(None, {self.input_name: X})
        return result[1]  # Probabilities
    
    @staticmethod
    def convert_sklearn_to_onnx(sklearn_model: BaseModel, X_sample: pd.DataFrame, 
                              output_path: str) -> None:
        """Convert sklearn model to ONNX format"""
        if not ONNX_AVAILABLE:
            raise ImportError("skl2onnx is not available. Cannot convert model to ONNX format.")
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(sklearn_model.model, initial_types=initial_type)
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logging.info(f"Model converted to ONNX and saved to {output_path}")


class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
        """Create model instance based on type"""
        model_map = {
            ModelConfig.LIGHTGBM: LightGBMModel,
            ModelConfig.RANDOM_FOREST: RandomForestModel,
            ModelConfig.LOGISTIC_REGRESSION: LogisticRegressionModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_map[model_type](config)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types"""
        return [
            ModelConfig.LIGHTGBM,
            ModelConfig.RANDOM_FOREST,
            ModelConfig.LOGISTIC_REGRESSION
        ]
