"""
Training ViewModel for EdunSight Application

This module handles model training, evaluation, and management.
Following MVVM pattern - this is part of the ViewModel layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import time
import joblib
import yaml
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from ..models.data_models import ModelMetrics, ModelConfig
from ..models.ml_models import BaseModel, ModelFactory, ONNXModel
from .data_processor import DataProcessor


class TrainingService:
    """Handles model training operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor(config_path)
        self.models = {}
        self.best_model = None
        self.training_history = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   hyperparameter_tuning: bool = False) -> ModelMetrics:
        """Train a single model"""
        start_time = time.time()
        
        self.logger.info(f"Starting training for {model_type}")
        
        # Create model
        model = ModelFactory.create_model(model_type, self.config)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, X_train, y_train)
        
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        metrics.training_time = time.time() - start_time
        
        # Store model
        self.models[model_type] = model
        
        # Log training results
        self.logger.info(f"Training completed for {model_type}:")
        self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        self.logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
        self.logger.info(f"  Training time: {metrics.training_time:.2f}s")
        
        # Add to training history
        self.training_history.append({
            'model_type': model_type,
            'timestamp': time.time(),
            'metrics': metrics,
            'hyperparameter_tuning': hyperparameter_tuning
        })
        
        return metrics
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            model_types: Optional[List[str]] = None,
                            hyperparameter_tuning: bool = False) -> Dict[str, ModelMetrics]:
        """Train multiple models and compare performance"""
        if model_types is None:
            model_types = ModelFactory.get_available_models()
        
        results = {}
        
        for model_type in model_types:
            try:
                metrics = self.train_model(
                    model_type, X_train, y_train, X_test, y_test,
                    hyperparameter_tuning=hyperparameter_tuning
                )
                results[model_type] = metrics
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
                continue
        
        # Select best model
        self._select_best_model(results)
        
        return results
    
    def _tune_hyperparameters(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Perform hyperparameter tuning"""
        self.logger.info(f"Tuning hyperparameters for {model.model_name}")
        
        param_grids = {
            ModelConfig.LIGHTGBM: {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.05, 0.1, 0.2]
            },
            ModelConfig.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelConfig.LOGISTIC_REGRESSION: {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model.model_name in param_grids:
            param_grid = param_grids[model.model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                model.model, param_grid,
                cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            model.model = grid_search.best_estimator_
            model.config.update(grid_search.best_params_)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return model
    
    def _select_best_model(self, results: Dict[str, ModelMetrics]) -> None:
        """Select the best model based on performance metrics"""
        if not results:
            return
        
        # Primary metric: AUC-ROC
        # Secondary metrics: accuracy, inference time
        best_model_type = None
        best_score = 0
        
        for model_type, metrics in results.items():
            # Composite score: weighted combination of metrics
            score = (0.6 * metrics.auc_roc + 
                    0.3 * metrics.accuracy + 
                    0.1 * min(1.0, ModelConfig.MAX_INFERENCE_TIME_MS / max(metrics.inference_time, 1)))
            
            if score > best_score:
                best_score = score
                best_model_type = model_type
        
        if best_model_type:
            self.best_model = self.models[best_model_type]
            self.logger.info(f"Best model selected: {best_model_type} (score: {best_score:.4f})")
    
    def cross_validate(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                      cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        model = ModelFactory.create_model(model_type, self.config)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model.model, X, y,
            cv=cv_folds, scoring='roc_auc',
            n_jobs=-1
        )
        
        results = {
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'scores': cv_scores.tolist()
        }
        
        self.logger.info(f"Cross-validation results for {model_type}:")
        self.logger.info(f"  Mean AUC: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        
        return results
    
    def generate_detailed_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Generate detailed training report"""
        report = {
            'training_summary': {
                'models_trained': len(self.models),
                'best_model': self.best_model.model_name if self.best_model else None,
                'training_time_total': sum(h['metrics'].training_time for h in self.training_history)
            },
            'model_comparison': {},
            'best_model_details': {},
            'training_history': self.training_history
        }
        
        # Model comparison
        for model_type, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            report['model_comparison'][model_type] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc,
                'inference_time_ms': metrics.inference_time,
                'model_size_mb': metrics.model_size_mb
            }
        
        # Best model details
        if self.best_model:
            y_pred = self.best_model.predict(X_test)
            y_proba = self.best_model.predict_proba(X_test)
            
            report['best_model_details'] = {
                'model_type': self.best_model.model_name,
                'feature_importance': self.best_model.get_feature_importance(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'prediction_distribution': {
                    'low_risk': int(np.sum(y_proba[:, 1] >= 0.7)),
                    'medium_risk': int(np.sum((y_proba[:, 1] >= 0.3) & (y_proba[:, 1] < 0.7))),
                    'high_risk': int(np.sum(y_proba[:, 1] < 0.3))
                }
            }
        
        return report
    
    def save_model(self, model_type: Optional[str] = None, 
                   output_dir: str = "models",
                   convert_to_onnx: bool = True) -> Dict[str, str]:
        """Save trained model(s) to disk"""
        if model_type:
            models_to_save = {model_type: self.models[model_type]}
        else:
            models_to_save = self.models
        
        saved_files = {}
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for name, model in models_to_save.items():
            # Save main model
            model_path = Path(output_dir) / f"{name}_model.joblib"
            
            # Create model package with metadata
            model_package = {
                'model': model.model,
                'model_name': model.model_name,
                'config': model.config,
                'feature_names': model.feature_names,
                'metrics': model.metrics,
                'version': '1.0.0',
                'timestamp': time.time()
            }
            
            joblib.dump(model_package, model_path)
            saved_files[f"{name}_joblib"] = str(model_path)
            
            # Convert to ONNX if requested and model supports it
            if convert_to_onnx and hasattr(model.model, 'predict_proba'):
                try:
                    onnx_path = Path(output_dir) / f"{name}_model.onnx"
                    
                    # Create sample input for ONNX conversion
                    if model.feature_names:
                        sample_input = pd.DataFrame(
                            np.random.random((1, len(model.feature_names))),
                            columns=model.feature_names
                        )
                        
                        ONNXModel.convert_sklearn_to_onnx(model, sample_input, str(onnx_path))
                        saved_files[f"{name}_onnx"] = str(onnx_path)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert {name} to ONNX: {e}")
        
        self.logger.info(f"Models saved to {output_dir}")
        return saved_files
    
    def load_model(self, model_path: str) -> BaseModel:
        """Load a previously trained model"""
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            # New format with metadata
            model = ModelFactory.create_model(model_data['model_name'], model_data['config'])
            model.model = model_data['model']
            model.feature_names = model_data['feature_names']
            model.metrics = model_data['metrics']
            model.is_trained = True
        else:
            # Legacy format
            raise ValueError("Legacy model format not supported")
        
        self.logger.info(f"Model loaded from {model_path}")
        return model
    
    def get_training_recommendations(self, dataset_size: int, 
                                   feature_count: int) -> Dict[str, Any]:
        """Get training recommendations based on dataset characteristics"""
        recommendations = {
            'suggested_models': [],
            'hyperparameter_tuning': True,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'expected_training_time': 'unknown'
        }
        
        # Model recommendations based on dataset size
        if dataset_size < 1000:
            recommendations['suggested_models'] = [ModelConfig.LOGISTIC_REGRESSION, ModelConfig.RANDOM_FOREST]
            recommendations['hyperparameter_tuning'] = False
            recommendations['expected_training_time'] = '< 1 minute'
        elif dataset_size < 10000:
            recommendations['suggested_models'] = [ModelConfig.RANDOM_FOREST, ModelConfig.LIGHTGBM]
            recommendations['expected_training_time'] = '1-5 minutes'
        else:
            recommendations['suggested_models'] = [ModelConfig.LIGHTGBM]
            recommendations['cross_validation_folds'] = 3
            recommendations['expected_training_time'] = '5-15 minutes'
        
        # Feature-based recommendations
        if feature_count > 50:
            recommendations['feature_selection'] = True
            recommendations['suggested_models'] = [ModelConfig.LIGHTGBM]
        
        return recommendations
