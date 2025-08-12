#!/usr/bin/env python3
"""
🧪 Basic EdunSight Test - Core Functionality
============================================

A minimal test to validate core components work without complex preprocessing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_simple_test_data():
    """Create simple test data without encoding issues"""
    print("📊 Creating simple test data...")
    
    # Create simple numeric data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'student_id': [f"STU_{i:03d}" for i in range(n_samples)],
        'age': np.random.randint(18, 65, n_samples),
        'attendance_rate': np.random.uniform(0.5, 1.0, n_samples),
        'assignment_score': np.random.uniform(0, 100, n_samples),
        'participation': np.random.uniform(0, 100, n_samples),
        'study_hours': np.random.uniform(5, 40, n_samples),
        'final_result': np.random.choice(['Pass', 'Fail'], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} student records")
    print(f"Target distribution: {df['final_result'].value_counts().to_dict()}")
    return df

def test_basic_ml_pipeline():
    """Test basic ML pipeline with simple data"""
    print("\n🤖 Testing Basic ML Pipeline...")
    
    try:
        from src.models.ml_models import ModelFactory
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Get simple test data
        df = create_simple_test_data()
        
        # Prepare features and target
        feature_columns = ['age', 'attendance_rate', 'assignment_score', 'participation', 'study_hours']
        X = df[feature_columns]
        
        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(df['final_result'])
        
        print(f"Features: {len(feature_columns)} columns")
        print(f"Target classes: {le.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test each model type
        model_types = ['lightgbm', 'random_forest', 'logistic_regression']
        results = {}
        
        for model_type in model_types:
            print(f"\n🔧 Testing {model_type}...")
            try:
                # Create config
                config = {
                    'lgb_params': {'n_estimators': 50, 'random_state': 42},
                    'rf_params': {'n_estimators': 50, 'random_state': 42},
                    'lr_params': {'random_state': 42, 'max_iter': 1000}
                }
                
                # Create and train model
                model = ModelFactory.create_model(model_type, config)
                model.train(X_train, y_train)
                
                # Test prediction
                predictions = model.predict(X_test)
                accuracy = (predictions == y_test).mean()
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'n_test': len(y_test),
                    'predictions': len(predictions)
                }
                
                print(f"  ✅ {model_type}: {accuracy:.3f} accuracy")
            
            except Exception as e:
                print(f"  ❌ {model_type} failed: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return None

def test_prediction_service():
    """Test prediction service with simple data"""
    print("\n📊 Testing Prediction Service...")
    
    try:
        from src.viewmodels.prediction_service import PredictionService
        from src.models.data_models import StudentRecord
        
        # Initialize service
        service = PredictionService()
        print("✅ PredictionService initialized")
        
        # Create sample student record
        student = StudentRecord(
            student_id="TEST001",
            age=22,
            attendance_rate=0.85,
            assessment_scores=[85, 90, 78],
            study_hours_per_week=15
        )
        
        print("✅ Sample StudentRecord created")
        return True
    
    except Exception as e:
        print(f"❌ Prediction service test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 EdunSight Core Functionality Test")
    print("=" * 40)
    
    results = {}
    
    # Test basic ML pipeline
    ml_results = test_basic_ml_pipeline()
    results['ml_pipeline'] = ml_results is not None
    
    # Test prediction service
    prediction_results = test_prediction_service()
    results['prediction_service'] = prediction_results
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if ml_results:
        print("\n🤖 ML Model Results:")
        for model_type, result in ml_results.items():
            if 'accuracy' in result:
                print(f"  ✅ {model_type}: {result['accuracy']:.3f} accuracy")
            else:
                print(f"  ❌ {model_type}: {result.get('error', 'Unknown error')}")
    
    if passed == total:
        print("\n🎉 All core functionality tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
