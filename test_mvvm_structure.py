#!/usr/bin/env python3
"""
🧪 EdunSight MVVM Structure Verification
========================================

Comprehensive test to verify the MVVM architecture implementation
and validate that all components can be imported and initialized correctly.
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all MVVM imports to ensure proper structure"""
    print("🔍 Testing MVVM Import Structure...")
    
    # Add the project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    tests = []
    
    # Test Model Layer
    print("\n📊 Testing Model Layer...")
    try:
        from src.models.data_models import StudentRecord, PredictionResult, ModelMetrics
        print("  ✅ Data models imported successfully")
        tests.append(("Data Models", True, None))
    except Exception as e:
        print(f"  ❌ Data models failed: {e}")
        tests.append(("Data Models", False, str(e)))
    
    try:
        from src.models.ml_models import BaseModel, LightGBMModel, ModelFactory
        print("  ✅ ML models imported successfully")
        tests.append(("ML Models", True, None))
    except Exception as e:
        print(f"  ❌ ML models failed: {e}")
        tests.append(("ML Models", False, str(e)))
    
    # Test ViewModel Layer
    print("\n🔄 Testing ViewModel Layer...")
    try:
        from src.viewmodels.data_processor import DataProcessor
        print("  ✅ Data processor imported successfully")
        tests.append(("Data Processor", True, None))
    except Exception as e:
        print(f"  ❌ Data processor failed: {e}")
        tests.append(("Data Processor", False, str(e)))
    
    try:
        from src.viewmodels.prediction_service import PredictionService
        print("  ✅ Prediction service imported successfully")
        tests.append(("Prediction Service", True, None))
    except Exception as e:
        print(f"  ❌ Prediction service failed: {e}")
        tests.append(("Prediction Service", False, str(e)))
    
    try:
        from src.viewmodels.training_service import TrainingService
        print("  ✅ Training service imported successfully")
        tests.append(("Training Service", True, None))
    except Exception as e:
        print(f"  ❌ Training service failed: {e}")
        tests.append(("Training Service", False, str(e)))
    
    # Test View Layer
    print("\n👁️ Testing View Layer...")
    try:
        from src.views.dashboard import DashboardView
        print("  ✅ Dashboard view imported successfully")
        tests.append(("Dashboard View", True, None))
    except Exception as e:
        print(f"  ❌ Dashboard view failed: {e}")
        tests.append(("Dashboard View", False, str(e)))
    
    # Test Utilities
    print("\n🛠️ Testing Utilities...")
    try:
        from src.utils.data_downloader import DataDownloader
        print("  ✅ Data downloader imported successfully")
        tests.append(("Data Downloader", True, None))
    except Exception as e:
        print(f"  ❌ Data downloader failed: {e}")
        tests.append(("Data Downloader", False, str(e)))
    
    try:
        from src.utils.logging_config import setup_logging
        print("  ✅ Logging config imported successfully")
        tests.append(("Logging Config", True, None))
    except Exception as e:
        print(f"  ❌ Logging config failed: {e}")
        tests.append(("Logging Config", False, str(e)))
    
    return tests

def test_basic_initialization():
    """Test basic initialization of key components"""
    print("\n🏗️ Testing Basic Component Initialization...")
    
    tests = []
    
    try:
        from src.models.data_models import StudentRecord
        # Test creating a basic student record
        student = StudentRecord(
            student_id="TEST123",
            age=20,
            gender="M",
            previous_grades=[85, 90, 78],
            attendance_rate=0.95,
            assignment_scores=[88, 92, 85],
            participation_score=85,
            study_hours_per_week=15
        )
        print("  ✅ StudentRecord creation successful")
        tests.append(("StudentRecord Creation", True, None))
    except Exception as e:
        print(f"  ❌ StudentRecord creation failed: {e}")
        tests.append(("StudentRecord Creation", False, str(e)))
    
    try:
        from src.viewmodels.data_processor import DataProcessor
        processor = DataProcessor()
        print("  ✅ DataProcessor initialization successful")
        tests.append(("DataProcessor Init", True, None))
    except Exception as e:
        print(f"  ❌ DataProcessor initialization failed: {e}")
        tests.append(("DataProcessor Init", False, str(e)))
    
    try:
        from src.models.ml_models import ModelFactory
        # Test factory pattern
        model_types = ModelFactory.get_available_models()
        print(f"  ✅ ModelFactory works - Available models: {model_types}")
        tests.append(("ModelFactory", True, None))
    except Exception as e:
        print(f"  ❌ ModelFactory failed: {e}")
        tests.append(("ModelFactory", False, str(e)))
    
    return tests

def check_file_structure():
    """Verify the file structure is correctly set up"""
    print("\n📂 Checking File Structure...")
    
    project_root = Path(__file__).parent
    expected_structure = {
        "src": True,
        "src/models": True,
        "src/models/__init__.py": True,
        "src/models/data_models.py": True,
        "src/models/ml_models.py": True,
        "src/viewmodels": True,
        "src/viewmodels/__init__.py": True,
        "src/viewmodels/data_processor.py": True,
        "src/viewmodels/prediction_service.py": True,
        "src/viewmodels/training_service.py": True,
        "src/views": True,
        "src/views/__init__.py": True,
        "src/views/dashboard.py": True,
        "src/utils": True,
        "src/utils/__init__.py": True,
        "src/utils/data_downloader.py": True,
        "src/utils/logging_config.py": True,
        "tests": True,
        "app.py": True,
        "train.py": True,
        "config.yaml": True,
        "requirements.txt": True
    }
    
    missing_files = []
    for file_path, required in expected_structure.items():
        full_path = project_root / file_path
        if required and not full_path.exists():
            missing_files.append(str(file_path))
        elif full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def main():
    """Main test runner"""
    print("🚀 EdunSight MVVM Architecture Verification")
    print("=" * 50)
    
    # Check file structure first
    structure_ok = check_file_structure()
    
    # Test imports
    import_tests = test_imports()
    
    # Test basic initialization
    init_tests = test_basic_initialization()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    all_tests = import_tests + init_tests
    passed = sum(1 for _, success, _ in all_tests if success)
    total = len(all_tests)
    
    print(f"Structure Check: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Import Tests: {passed}/{total} passed")
    
    if passed == total and structure_ok:
        print("\n🎉 All tests passed! MVVM structure is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        
        # Show failed tests
        failed_tests = [name for name, success, error in all_tests if not success]
        if failed_tests:
            print(f"\nFailed tests: {', '.join(failed_tests)}")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
