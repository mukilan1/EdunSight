"""
Model Training Script for EdunSight

This script handles the complete training pipeline including:
- Data downloading and preprocessing
- Model training and evaluation
- Model saving and ONNX conversion
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.viewmodels.data_processor import DataProcessor
from src.viewmodels.training_service import TrainingService
from src.utils.data_downloader import DataDownloader
from src.utils.logging_config import get_logger, PerformanceLogger
from src.models.data_models import ModelConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train EdunSight ML models")
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data CSV file"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=[ModelConfig.LIGHTGBM, ModelConfig.RANDOM_FOREST, ModelConfig.LOGISTIC_REGRESSION],
        default=ModelConfig.LIGHTGBM,
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download sample datasets before training"
    )
    
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all available model types"
    )
    
    return parser.parse_args()


def download_and_prepare_data(downloader: DataDownloader, 
                            data_processor: DataProcessor) -> Path:
    """Download and prepare training data"""
    logger = get_logger(__name__)
    
    with PerformanceLogger("Data Download", logger):
        # Download datasets
        datasets = downloader.download_all_datasets()
        
        # Use sample OULAD data as primary dataset
        if datasets.get('sample_oulad'):
            data_path = datasets['sample_oulad'][0]
        elif datasets.get('uci_student_performance'):
            data_path = datasets['uci_student_performance'][0]
        else:
            raise ValueError("No datasets available for training")
        
        logger.info(f"Using dataset: {data_path}")
    
    return data_path


def prepare_training_data(data_processor: DataProcessor, 
                         data_path: Path) -> tuple:
    """Prepare data for training"""
    logger = get_logger(__name__)
    
    with PerformanceLogger("Data Preparation", logger):
        # Load data
        df = data_processor.load_data(str(data_path))
        logger.info(f"Loaded {len(df)} records")
        
        # Validate data
        validation_report = data_processor.validate_data(df)
        logger.info(f"Data validation: {validation_report['total_rows']} rows, "
                   f"{validation_report['total_columns']} columns")
        
        if validation_report['schema_validation']['missing_columns']:
            logger.warning(f"Missing expected columns: {validation_report['schema_validation']['missing_columns']}")
        
        # Clean and preprocess data
        df_clean = data_processor.clean_data(df)
        df_features = data_processor.engineer_features(df_clean)
        df_encoded = data_processor.encode_categorical_features(df_features)
        df_scaled = data_processor.scale_features(df_encoded)
        
        # Prepare for training
        X_train, X_test, y_train, y_test = data_processor.prepare_for_training(df_scaled)
        
        logger.info(f"Training data prepared: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Feature count: {len(X_train.columns)}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        
    return X_train, X_test, y_train, y_test


def train_models(training_service: TrainingService, 
                X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                model_types: list, hyperparameter_tuning: bool) -> dict:
    """Train specified models"""
    logger = get_logger(__name__)
    
    with PerformanceLogger("Model Training", logger):
        if len(model_types) == 1:
            # Train single model
            model_type = model_types[0]
            metrics = training_service.train_model(
                model_type, X_train, y_train, X_test, y_test,
                hyperparameter_tuning=hyperparameter_tuning
            )
            results = {model_type: metrics}
        else:
            # Train multiple models
            results = training_service.train_multiple_models(
                X_train, y_train, X_test, y_test,
                model_types=model_types,
                hyperparameter_tuning=hyperparameter_tuning
            )
        
        # Generate detailed report
        report = training_service.generate_detailed_report(X_test, y_test)
        
        # Log results
        logger.info("Training Results:")
        for model_type, metrics in results.items():
            logger.info(f"  {model_type}:")
            logger.info(f"    Accuracy: {metrics.accuracy:.4f}")
            logger.info(f"    AUC-ROC: {metrics.auc_roc:.4f}")
            logger.info(f"    Training Time: {metrics.training_time:.2f}s")
            logger.info(f"    Inference Time: {metrics.inference_time:.2f}ms")
        
        if training_service.best_model:
            logger.info(f"Best model: {training_service.best_model.model_name}")
    
    return results, report


def save_models_and_artifacts(training_service: TrainingService,
                            output_dir: str, report: dict) -> None:
    """Save trained models and training artifacts"""
    logger = get_logger(__name__)
    
    with PerformanceLogger("Model Saving", logger):
        # Save models
        saved_files = training_service.save_model(
            output_dir=output_dir,
            convert_to_onnx=True
        )
        
        logger.info("Saved files:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  {file_type}: {file_path}")
        
        # Save training report
        report_path = Path(output_dir) / "training_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Training report saved to {report_path}")


def main():
    """Main training function"""
    args = parse_arguments()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting EdunSight model training")
    logger.info(f"Configuration: {args}")
    
    try:
        # Initialize services
        data_processor = DataProcessor(args.config)
        training_service = TrainingService(args.config)
        
        # Download data if requested
        if args.download_data or not args.data_path:
            downloader = DataDownloader(args.config)
            data_path = download_and_prepare_data(downloader, data_processor)
        else:
            data_path = Path(args.data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Prepare training data
        X_train, X_test, y_train, y_test = prepare_training_data(data_processor, data_path)
        
        # Determine which models to train
        if args.all_models:
            model_types = training_service.ModelFactory.get_available_models()
        else:
            model_types = [args.model_type]
        
        logger.info(f"Training models: {model_types}")
        
        # Get training recommendations
        recommendations = training_service.get_training_recommendations(
            len(X_train), len(X_train.columns)
        )
        logger.info(f"Training recommendations: {recommendations}")
        
        # Train models
        results, report = train_models(
            training_service, X_train, y_train, X_test, y_test,
            model_types, args.hyperparameter_tuning
        )
        
        # Save models and artifacts
        save_models_and_artifacts(training_service, args.output_dir, report)
        
        logger.info("✅ Training completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("🎯 TRAINING SUMMARY")
        print("="*50)
        
        for model_type, metrics in results.items():
            print(f"\n📊 {model_type}:")
            print(f"   Accuracy: {metrics.accuracy:.1%}")
            print(f"   AUC-ROC:  {metrics.auc_roc:.1%}")
            print(f"   Training: {metrics.training_time:.1f}s")
            print(f"   Inference: {metrics.inference_time:.1f}ms")
        
        if training_service.best_model:
            print(f"\n🏆 Best Model: {training_service.best_model.model_name}")
        
        print(f"\n📁 Models saved to: {args.output_dir}")
        print("\n🚀 Ready to run: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
