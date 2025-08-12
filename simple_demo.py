#!/usr/bin/env python3
"""
🧪 Simple EdunSight Demo - Step by Step Testing
===============================================

A simplified demo to test each component individually.
"""

import sys
from pathlib import Path
import pandas as pd
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading():
    """Test data loading"""
    print("📊 Testing data loading...")
    
    try:
        # Check if data exists
        data_path = Path("data/raw/sample_oulad_data.csv")
        if not data_path.exists():
            print("❌ Data file not found")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Target distribution: {df['final_result'].value_counts().to_dict()}")
        return df
    
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_data_processor():
    """Test data processor"""
    print("\n🔧 Testing data processor...")
    
    try:
        from src.viewmodels.data_processor import DataProcessor
        
        processor = DataProcessor()
        print("✅ DataProcessor initialized")
        
        # Test basic processing
        df = test_data_loading()
        if df is None:
            return None
        
        # Clean data
        df_clean = processor.clean_data(df)
        print(f"✅ Data cleaned: {len(df_clean)} rows")
        
        # Engineer features  
        df_features = processor.engineer_features(df_clean)
        print(f"✅ Features engineered: {len(df_features.columns)} columns")
        
        # Check target variable before encoding
        print(f"Target before encoding: {df_features['final_result'].unique()}")
        print(f"Target type: {type(df_features['final_result'].iloc[0])}")
        
        return df_features
    
    except Exception as e:
        print(f"❌ Data processor failed: {e}")
        traceback.print_exc()
        return None

def test_basic_training():
    """Test basic training without full pipeline"""
    print("\n🤖 Testing basic training...")
    
    try:
        from src.models.ml_models import ModelFactory
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Get processed data
        df = test_data_processor()
        if df is None:
            return False
        
        # Simple encoding for demonstration
        df_simple = df.copy()
        
        # Convert target to numbers
        le = LabelEncoder()
        df_simple['target_numeric'] = le.fit_transform(df_simple['final_result'])
        
        # Select only numeric columns for features
        numeric_columns = df_simple.select_dtypes(include=['number']).columns
        feature_columns = [col for col in numeric_columns if col not in ['target_numeric', 'student_id']]
        
        X = df_simple[feature_columns]
        y = df_simple['target_numeric']
        
        print(f"Features: {len(feature_columns)} columns")
        print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        config = {'lgb_params': {'n_estimators': 100, 'random_state': 42}}  # Simple config dict
        model = ModelFactory.create_model('lightgbm', config)
        model.train(X_train, y_train)
        
        # Test prediction
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        print(f"✅ Model trained and tested. Accuracy: {accuracy:.3f}")
        return True
    
    except Exception as e:
        print(f"❌ Basic training failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("🚀 EdunSight Simple Demo")
    print("=" * 30)
    
    # Test each component
    success = all([
        test_data_loading() is not None,
        test_data_processor() is not None,
        test_basic_training()
    ])
    
    if success:
        print("\n🎉 All tests passed! The MVVM pipeline is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
