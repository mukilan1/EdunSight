"""
Data Processing ViewModel for EdunSight Application

This module handles data preprocessing, feature engineering, and data validation.
Following MVVM pattern - this is part of the ViewModel layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import yaml

from ..models.data_models import FeatureSchema, DatasetInfo, AppConstants


class DataProcessor:
    """Handles all data processing operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def load_data(self, file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Load data from CSV file with chunking support"""
        try:
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                self.logger.info(f"Loaded {len(df)} rows in chunks from {file_path}")
            else:
                df = pd.read_csv(file_path)
                self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and structure"""
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'schema_validation': FeatureSchema.validate_dataframe(df)
        }
        
        # Check for high missing value columns
        missing_threshold = 0.5
        high_missing_cols = [
            col for col, missing_count in validation_report['missing_values'].items()
            if missing_count / len(df) > missing_threshold
        ]
        validation_report['high_missing_columns'] = high_missing_cols
        
        self.logger.info(f"Data validation completed. {validation_report['total_rows']} rows, "
                        f"{validation_report['total_columns']} columns")
        
        return validation_report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data"""
        df_clean = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        # Standardize column names
        df_clean = self._standardize_column_names(df_clean)
        
        self.logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].dtype in ['int64', 'float64']:
                # Numerical columns: fill with median
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            else:
                # Categorical columns: fill with mode or 'Unknown'
                mode_value = df_filled[col].mode()
                if len(mode_value) > 0:
                    df_filled[col] = df_filled[col].fillna(mode_value[0])
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
        
        return df_filled
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_no_outliers = df.copy()
        numerical_cols = df_no_outliers.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_no_outliers[col] = df_no_outliers[col].clip(lower_bound, upper_bound)
        
        return df_no_outliers
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df_standard = df.copy()
        df_standard.columns = (df_standard.columns
                              .str.lower()
                              .str.replace(' ', '_')
                              .str.replace('[^a-zA-Z0-9_]', '', regex=True))
        return df_standard
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        df_features = df.copy()
        
        # Time-based features
        if 'date_registration' in df_features.columns:
            df_features['days_since_registration'] = (
                pd.to_datetime('today') - pd.to_datetime(df_features['date_registration'])
            ).dt.days
        
        # Engagement features
        if 'total_clicks' in df_features.columns and 'days_since_registration' in df_features.columns:
            df_features['clicks_per_day'] = (
                df_features['total_clicks'] / (df_features['days_since_registration'] + 1)
            )
        
        # Assessment performance features
        assessment_cols = [col for col in df_features.columns if 'assessment' in col.lower()]
        if assessment_cols:
            df_features['avg_assessment_score'] = df_features[assessment_cols].mean(axis=1)
            df_features['assessment_consistency'] = df_features[assessment_cols].std(axis=1)
        
        # Risk indicators
        if 'submission_delays' in df_features.columns:
            df_features['high_delay_risk'] = (df_features['submission_delays'] > 2).astype(int)
        
        if 'attendance_rate' in df_features.columns:
            df_features['low_attendance_risk'] = (df_features['attendance_rate'] < 0.7).astype(int)
        
        self.logger.info(f"Feature engineering completed. Added {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical features"""
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude target variable from categorical encoding
        if FeatureSchema.TARGET_VARIABLE in categorical_cols:
            categorical_cols.remove(FeatureSchema.TARGET_VARIABLE)
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Use label encoding for binary categories, one-hot for others
                unique_values = df_encoded[col].nunique()
                
                if unique_values == 2:
                    # Binary encoding
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                    else:
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                
                elif unique_values <= 10:  # One-hot encode if not too many categories
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                
                else:
                    # Target encoding for high cardinality categories
                    df_encoded[col] = self._target_encode(df_encoded, col)
        
        return df_encoded
    
    def _target_encode(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Simple target encoding for high cardinality categorical features"""
        if FeatureSchema.TARGET_VARIABLE in df.columns:
            target = df[FeatureSchema.TARGET_VARIABLE]
            
            # Convert target to numeric if it's categorical
            if target.dtype == 'object':
                target = target.map(FeatureSchema.TARGET_MAPPING)
            
            mean_target = target.mean()
            
            # Create a temporary dataframe with encoded target for groupby
            temp_df = df.copy()
            temp_df[FeatureSchema.TARGET_VARIABLE] = target
            
            encoded = df[column].map(temp_df.groupby(column)[FeatureSchema.TARGET_VARIABLE].mean())
            return encoded.fillna(mean_target)
        else:
            # If no target available, use frequency encoding
            return df[column].map(df[column].value_counts())
    
    def scale_features(self, df: pd.DataFrame, 
                      numerical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features"""
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_scaled = df.copy()
        
        if numerical_cols:
            if not self.is_fitted:
                df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
                self.is_fitted = True
            else:
                df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        return df_scaled
    
    def prepare_for_training(self, df: pd.DataFrame, 
                           target_col: str = FeatureSchema.TARGET_VARIABLE,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series]:
        """Prepare data for model training"""
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Apply target mapping if needed
        if y.dtype == 'object':
            y = y.map(FeatureSchema.TARGET_MAPPING)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Data split completed. Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self, df: pd.DataFrame, dataset_name: str, 
                        file_path: str) -> DatasetInfo:
        """Get dataset information and metadata"""
        from datetime import datetime
        
        return DatasetInfo(
            name=dataset_name,
            source=file_path,
            rows=len(df),
            columns=len(df.columns),
            missing_values=df.isnull().sum().to_dict(),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            target_distribution=(
                df[FeatureSchema.TARGET_VARIABLE].value_counts().to_dict() 
                if FeatureSchema.TARGET_VARIABLE in df.columns else {}
            ),
            last_updated=datetime.now(),
            file_path=file_path
        )
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to file"""
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV for compatibility, or Feather for speed
        if output_path.endswith('.feather'):
            df.to_feather(output_path)
        else:
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """Load previously processed data"""
        if file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            df = pd.read_csv(file_path)
        
        self.logger.info(f"Loaded processed data from {file_path}")
        return df
