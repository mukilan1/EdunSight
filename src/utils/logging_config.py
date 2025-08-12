"""
Logging Configuration Utility for EdunSight Application

This module provides centralized logging configuration.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any
import yaml


def setup_logging(config_path: str = "config.yaml", 
                  default_level: int = logging.INFO,
                  log_dir: str = "logs") -> None:
    """Setup logging configuration"""
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Default logging configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'detailed',
                'filename': f'{log_dir}/edusight.log',
                'mode': 'a'
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.FileHandler',
                'formatter': 'detailed',
                'filename': f'{log_dir}/errors.log',
                'mode': 'a'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'streamlit': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    # Try to load logging config from YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging_config = config.get('logging', default_config)
    except (FileNotFoundError, KeyError):
        logging_config = default_config
    
    # Configure logging
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation_name: str, logger: logging.Logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f} seconds")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f} seconds: {exc_val}")


# Initialize logging when module is imported
setup_logging()
