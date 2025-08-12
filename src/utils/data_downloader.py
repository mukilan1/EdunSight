"""
Data Download Utility for EdunSight Application

This module handles downloading and preparing datasets.
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import yaml


class DataDownloader:
    """Downloads and prepares datasets for training"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config.get('data', {}).get('raw_dir', 'data/raw'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def download_file(self, url: str, filename: str, force_download: bool = False) -> Path:
        """Download file from URL"""
        file_path = self.data_dir / filename
        
        if file_path.exists() and not force_download:
            self.logger.info(f"File {filename} already exists. Skipping download.")
            return file_path
        
        try:
            self.logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Successfully downloaded {filename}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading {filename}: {e}")
            raise
    
    def extract_zip(self, zip_path: Path, extract_to: Optional[Path] = None) -> Path:
        """Extract ZIP file"""
        if extract_to is None:
            extract_to = zip_path.parent / zip_path.stem
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            self.logger.info(f"Extracted {zip_path} to {extract_to}")
            return extract_to
            
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {e}")
            raise
    
    def download_uci_student_performance(self) -> List[Path]:
        """Download UCI Student Performance dataset"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        
        # Download ZIP file
        zip_path = self.download_file(url, "student_performance.zip")
        
        # Extract ZIP
        extract_dir = self.extract_zip(zip_path)
        
        # Find CSV files
        csv_files = list(extract_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        self.logger.info(f"Found {len(csv_files)} CSV files in UCI dataset")
        return csv_files
    
    def create_sample_oulad_data(self) -> Path:
        """Create a sample OULAD-like dataset for demonstration"""
        self.logger.info("Creating sample OULAD-like dataset...")
        
        # Generate sample data
        np.random.seed(42)
        n_students = 1000
        
        data = {
            'student_id': [f'STU_{i:04d}' for i in range(n_students)],
            'age': np.random.randint(18, 65, n_students),
            'gender': np.random.choice(['M', 'F'], n_students),
            'course_id': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'], n_students),
            'region': np.random.choice(['East', 'West', 'North', 'South', 'Central'], n_students),
            'highest_education': np.random.choice(['A Level', 'HE Qualification', 'Lower Than A Level'], n_students),
            'disability': np.random.choice(['Y', 'N'], n_students, p=[0.1, 0.9]),
            'age_band': np.random.choice(['0-35', '35-55', '55<='], n_students),
            'total_clicks': np.random.randint(50, 2000, n_students),
            'attendance_rate': np.random.beta(8, 2, n_students),  # Skewed towards higher attendance
            'time_spent_online': np.random.gamma(2, 25, n_students),  # Hours
            'submission_delays': np.random.poisson(1.5, n_students),
            'previous_attempts': np.random.choice([0, 1, 2, 3], n_students, p=[0.7, 0.2, 0.08, 0.02]),
            'forum_posts': np.random.poisson(5, n_students),
            'assessment_1': np.random.normal(70, 15, n_students),
            'assessment_2': np.random.normal(68, 18, n_students),
            'assessment_3': np.random.normal(72, 16, n_students)
        }
        
        # Create derived features
        data['days_since_last_access'] = np.random.exponential(3, n_students)
        data['clicks_per_day'] = data['total_clicks'] / (data['days_since_last_access'] + 1)
        data['avg_assessment_score'] = (data['assessment_1'] + data['assessment_2'] + data['assessment_3']) / 3
        data['assessment_consistency'] = np.std([data['assessment_1'], data['assessment_2'], data['assessment_3']], axis=0)
        
        # Create target variable based on features (more realistic)
        risk_score = (
            -0.3 * data['attendance_rate'] +
            0.2 * (data['submission_delays'] / 5) +
            -0.2 * (data['avg_assessment_score'] / 100) +
            0.1 * (data['days_since_last_access'] / 10) +
            -0.1 * (data['time_spent_online'] / 100) +
            np.random.normal(0, 0.1, n_students)  # Add noise
        )
        
        # Convert to Pass/Fail (reverse the logic - higher risk_score means more likely to fail)
        data['final_result'] = ['Fail' if score > 0 else 'Pass' for score in risk_score]
        
        # Adjust some edge cases for realism
        for i in range(n_students):
            if data['avg_assessment_score'][i] > 85 and np.random.random() < 0.9:
                data['final_result'][i] = 'Pass'
            elif data['avg_assessment_score'][i] < 40 and np.random.random() < 0.8:
                data['final_result'][i] = 'Fail'
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = self.data_dir / "sample_oulad_data.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Created sample OULAD dataset with {len(df)} students")
        self.logger.info(f"Target distribution: {df['final_result'].value_counts().to_dict()}")
        
        return output_path
    
    def download_all_datasets(self) -> Dict[str, List[Path]]:
        """Download all available datasets"""
        datasets = {}
        
        try:
            # UCI Student Performance
            uci_files = self.download_uci_student_performance()
            datasets['uci_student_performance'] = uci_files
        except Exception as e:
            self.logger.error(f"Failed to download UCI dataset: {e}")
            datasets['uci_student_performance'] = []
        
        try:
            # Sample OULAD data
            oulad_file = self.create_sample_oulad_data()
            datasets['sample_oulad'] = [oulad_file]
        except Exception as e:
            self.logger.error(f"Failed to create sample OULAD dataset: {e}")
            datasets['sample_oulad'] = []
        
        return datasets
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about available datasets"""
        info = {}
        
        # Check for existing files
        for file_path in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path, nrows=5)  # Just peek at the data
                info[file_path.name] = {
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'columns': list(df.columns),
                    'sample_data': df.head().to_dict('records')
                }
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")
        
        return info


def main():
    """Main function for running as script"""
    import numpy as np
    
    logging.basicConfig(level=logging.INFO)
    
    downloader = DataDownloader()
    
    print("📥 Downloading datasets...")
    datasets = downloader.download_all_datasets()
    
    print("\n📊 Dataset Summary:")
    for name, files in datasets.items():
        print(f"  {name}: {len(files)} files")
        for file_path in files:
            print(f"    - {file_path}")
    
    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
