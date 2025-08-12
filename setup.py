"""
Setup script for EdunSight

Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def setup_environment():
    """Set up the EdunSight environment"""
    print("🚀 Setting up EdunSight environment...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install requirements
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing requirements"):
        return False
    
    # Create necessary directories
    directories = ["data/raw", "data/processed", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Download data: python src/utils/data_downloader.py")
    print("  2. Train model: python train.py --download-data")
    print("  3. Run app: streamlit run app.py")
    print("  4. Run tests: python run_tests.py")
    
    return True


def quick_start():
    """Quick start with data download and model training"""
    print("\n🚀 Running quick start...")
    
    # Download data
    if not run_command([sys.executable, "src/utils/data_downloader.py"], 
                      "Downloading sample data"):
        return False
    
    # Train a quick model
    if not run_command([sys.executable, "train.py", "--download-data", "--model-type", "lightgbm"], 
                      "Training LightGBM model"):
        return False
    
    print("\n🎉 Quick start completed!")
    print("  ✅ Data downloaded")
    print("  ✅ Model trained")
    print("  🚀 Ready to run: streamlit run app.py")
    
    return True


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup EdunSight environment")
    parser.add_argument(
        "--quick-start",
        action="store_true",
        help="Run quick start (setup + data download + model training)"
    )
    
    args = parser.parse_args()
    
    # Always run basic setup
    if not setup_environment():
        sys.exit(1)
    
    # Run quick start if requested
    if args.quick_start:
        if not quick_start():
            sys.exit(1)


if __name__ == "__main__":
    main()
