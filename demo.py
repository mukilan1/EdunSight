"""
Quick Demo Script for EdunSight

Demonstrates the complete ML pipeline in a simple script.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_downloader import DataDownloader
from src.viewmodels.data_processor import DataProcessor
from src.viewmodels.training_service import TrainingService
from src.viewmodels.prediction_service import PredictionService
from src.models.data_models import ModelConfig


def main():
    """Run quick demo"""
    print("🚀 EdunSight Quick Demo")
    print("=" * 40)
    
    try:
        # 1. Download sample data
        print("\n📥 Step 1: Downloading sample data...")
        downloader = DataDownloader()
        datasets = downloader.download_all_datasets()
        
        if datasets.get('sample_oulad'):
            data_path = datasets['sample_oulad'][0]
        else:
            print("❌ No sample data available")
            return
        
        print(f"✅ Data downloaded: {data_path}")
        
        # 2. Process data
        print("\n🔧 Step 2: Processing data...")
        data_processor = DataProcessor()
        
        # Load and clean data
        df_raw = data_processor.load_data(str(data_path))
        df_clean = data_processor.clean_data(df_raw)
        df_features = data_processor.engineer_features(df_clean)
        df_encoded = data_processor.encode_categorical_features(df_features)
        df_scaled = data_processor.scale_features(df_encoded)
        
        # Prepare for training
        X_train, X_test, y_train, y_test = data_processor.prepare_for_training(df_scaled)
        
        print(f"✅ Data processed: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # 3. Train model
        print("\n🤖 Step 3: Training LightGBM model...")
        training_service = TrainingService()
        
        metrics = training_service.train_model(
            ModelConfig.LIGHTGBM,
            X_train, y_train,
            X_test, y_test
        )
        
        print(f"✅ Model trained:")
        print(f"   Accuracy: {metrics.accuracy:.3f}")
        print(f"   AUC-ROC: {metrics.auc_roc:.3f}")
        print(f"   Training time: {metrics.training_time:.1f}s")
        
        # 4. Save model
        print("\n💾 Step 4: Saving model...")
        saved_files = training_service.save_model(output_dir="models")
        model_path = [f for f in saved_files.values() if f.endswith('.joblib')][0]
        print(f"✅ Model saved: {model_path}")
        
        # 5. Make predictions
        print("\n🎯 Step 5: Making predictions...")
        prediction_service = PredictionService()
        prediction_service.load_model(model_path)
        
        # Test with sample students
        sample_students = [
            {
                'student_id': 'DEMO_001',
                'age': 22,
                'gender': 'F',
                'course_id': 'AAA',
                'attendance_rate': 0.95,
                'time_spent_online': 150.0,
                'submission_delays': 0,
                'previous_attempts': 0
            },
            {
                'student_id': 'DEMO_002',
                'age': 35,
                'gender': 'M',
                'course_id': 'BBB',
                'attendance_rate': 0.60,
                'time_spent_online': 45.0,
                'submission_delays': 3,
                'previous_attempts': 1
            }
        ]
        
        predictions = prediction_service.predict_batch(sample_students)
        
        print("✅ Predictions made:")
        for pred in predictions:
            print(f"   {pred.student_id}: {pred.risk_category} risk ({pred.probability_pass:.1%} pass probability)")
        
        # 6. Show demo summary
        print("\n🎉 Demo completed successfully!")
        print("\n📋 What was demonstrated:")
        print("   ✅ Data download and preprocessing")
        print("   ✅ Feature engineering")
        print("   ✅ Model training with LightGBM")
        print("   ✅ Model evaluation and saving")
        print("   ✅ Prediction service")
        print("   ✅ Risk assessment")
        
        print("\n🚀 Ready to run the full application:")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
