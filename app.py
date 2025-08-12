"""
Main Streamlit Application for EdunSight

This is the entry point for the EdunSight web application.
Integrates all MVVM components for a complete ML pipeline.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.data_models import AppConstants
from src.viewmodels.prediction_service import PredictionService
from src.views.dashboard import DashboardView, render_footer
from src.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def load_prediction_service() -> PredictionService:
    """Load and initialize the prediction service"""
    try:
        prediction_service = PredictionService()
        
        # Try to load a pre-trained model
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*_model.joblib"))
            if model_files:
                # Load the first available model
                model_path = model_files[0]
                onnx_path = model_path.with_suffix('.onnx')
                
                prediction_service.load_model(
                    str(model_path),
                    str(onnx_path) if onnx_path.exists() else None
                )
                logger.info(f"Loaded model from {model_path}")
            else:
                st.warning("⚠️ No trained models found. Please train a model first.")
                logger.warning("No trained models found")
        else:
            st.warning("⚠️ Models directory not found. Please train a model first.")
            logger.warning("Models directory not found")
        
        return prediction_service
        
    except Exception as e:
        st.error(f"❌ Error loading prediction service: {str(e)}")
        logger.error(f"Error loading prediction service: {e}")
        return PredictionService()


def main():
    """Main application function"""
    try:
        # Initialize prediction service
        prediction_service = load_prediction_service()
        
        # Initialize dashboard view
        dashboard = DashboardView(prediction_service)
        
        # Render header
        dashboard.render_header()
        
        # Render sidebar and get user settings
        settings = dashboard.render_sidebar()
        
        # Route to appropriate page based on selection
        page = settings["page"]
        
        if page == "Single Prediction":
            dashboard.render_single_prediction_page(settings)
        
        elif page == "Batch Prediction":
            dashboard.render_batch_prediction_page(settings)
        
        elif page == "Model Analytics":
            dashboard.render_model_analytics_page()
        
        elif page == "Data Upload":
            dashboard.render_data_upload_page()
        
        # Render footer
        render_footer()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show error details in expander for debugging
        with st.expander("Error Details"):
            st.exception(e)


def setup_page_config():
    """Setup initial page configuration"""
    st.set_page_config(
        page_title="EdunSight - Student Performance Predictor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/edusight',
            'Report a bug': 'https://github.com/your-repo/edusight/issues',
            'About': """
            # EdunSight v1.0.0
            
            A lightweight, real-time ML pipeline that predicts student performance 
            from CSV/LMS logs and serves fast predictions via an interactive web app.
            
            Built with:
            - **ML**: LightGBM, scikit-learn, ONNX Runtime
            - **Web**: Streamlit, Plotly
            - **Architecture**: MVVM Pattern
            """
        }
    )


if __name__ == "__main__":
    # Setup page configuration
    setup_page_config()
    
    # Run the main application
    main()
