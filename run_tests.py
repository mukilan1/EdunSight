"""
Test runner script for EdunSight

Runs all tests with coverage reporting.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage"""
    
    # Change to project directory
    project_dir = Path(__file__).parent
    
    print("🧪 Running EdunSight Test Suite")
    print("=" * 50)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=html:coverage_html",
        "--cov-report=term-missing",
        "--cov-fail-under=70",
        "-x"  # Stop on first failure
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_dir, check=True)
        print("\n✅ All tests passed!")
        print("\n📊 Coverage report generated in coverage_html/")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_tests(test_pattern):
    """Run specific tests matching pattern"""
    
    project_dir = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-k", test_pattern
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_dir, check=True)
        print(f"\n✅ Tests matching '{test_pattern}' passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EdunSight tests")
    parser.add_argument(
        "--pattern", "-k",
        help="Run tests matching pattern"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_true", 
        help="Run only integration tests"
    )
    
    args = parser.parse_args()
    
    if args.pattern:
        success = run_specific_tests(args.pattern)
    elif args.unit:
        success = run_specific_tests("unit")
    elif args.integration:
        success = run_specific_tests("integration")
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
