"""
Dashboard View Components for EdunSight Application

This module contains Streamlit UI components for the main dashboard.
Following MVVM pattern - this is part of the View layer.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import numpy as np

from ..models.data_models import PredictionResult, ModelMetrics
from ..viewmodels.prediction_service import PredictionService


class DashboardView:
    """Main dashboard view components"""
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
    
    def render_header(self) -> None:
        """Render dashboard header"""
        st.set_page_config(
            page_title="EdunSight - Student Performance Predictor",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📚 EdunSight - Student Performance Predictor")
        st.markdown("---")
        
        # Quick stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Status", "✅ Ready" if self.prediction_service.model else "❌ Not Loaded")
        
        with col2:
            model_info = self.prediction_service.get_model_info()
            st.metric("Model Type", model_info.get("model_type", "N/A"))
        
        with col3:
            st.metric("ONNX Enabled", "Yes" if model_info.get("onnx_enabled") else "No")
        
        with col4:
            st.metric("Features", model_info.get("features_count", "N/A"))
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar controls and return user inputs"""
        st.sidebar.header("Navigation")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["Single Prediction", "Batch Prediction", "Model Analytics", "Data Upload"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("Quick Actions")
        
        # Model info
        if st.sidebar.button("🔄 Refresh Model"):
            st.rerun()
        
        # Settings
        st.sidebar.markdown("---")
        st.sidebar.header("Settings")
        
        show_explanations = st.sidebar.checkbox("Show Explanations", value=True)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        
        return {
            "page": page,
            "show_explanations": show_explanations,
            "confidence_threshold": confidence_threshold
        }
    
    def render_single_prediction_page(self, settings: Dict[str, Any]) -> None:
        """Render single student prediction page"""
        st.header("Single Student Prediction")
        
        # Input form
        with st.form("student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.text_input("Student ID", value="STU001")
                age = st.number_input("Age", min_value=16, max_value=100, value=20)
                gender = st.selectbox("Gender", ["M", "F", "Other"])
                attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.8)
                
            with col2:
                course_id = st.selectbox("Course", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"])
                time_spent_online = st.number_input("Time Spent Online (hours)", min_value=0.0, value=50.0)
                submission_delays = st.number_input("Submission Delays", min_value=0, value=1)
                previous_attempts = st.number_input("Previous Attempts", min_value=0, value=0)
            
            submitted = st.form_submit_button("Predict Performance")
        
        if submitted:
            # Prepare student data
            student_data = {
                "student_id": student_id,
                "age": age,
                "gender": gender,
                "course_id": course_id,
                "attendance_rate": attendance_rate,
                "time_spent_online": time_spent_online,
                "submission_delays": submission_delays,
                "previous_attempts": previous_attempts
            }
            
            # Validate input
            validation = self.prediction_service.validate_input(student_data)
            
            if validation["errors"]:
                st.error("Please fix the following errors:")
                for error in validation["errors"]:
                    st.error(f"• {error}")
                return
            
            if validation["warnings"]:
                st.warning("Please note:")
                for warning in validation["warnings"]:
                    st.warning(f"• {warning}")
            
            # Make prediction
            try:
                with st.spinner("Making prediction..."):
                    prediction = self.prediction_service.predict_single(student_data)
                
                # Display results
                self._display_prediction_result(prediction, settings)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    def render_batch_prediction_page(self, settings: Dict[str, Any]) -> None:
        """Render batch prediction page"""
        st.header("Batch Student Predictions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with student data",
            type=['csv'],
            help="CSV should contain columns: student_id, age, gender, course_id, etc."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                st.info(f"Loaded {len(df)} students for prediction")
                
                if st.button("Run Batch Predictions"):
                    with st.spinner("Running batch predictions..."):
                        # Convert to list of dictionaries
                        students_data = df.to_dict('records')
                        
                        # Make predictions
                        predictions = self.prediction_service.predict_batch(students_data)
                        
                        # Display results
                        self._display_batch_results(predictions, settings)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def render_model_analytics_page(self) -> None:
        """Render model analytics and performance metrics"""
        st.header("Model Analytics")
        
        model_info = self.prediction_service.get_model_info()
        
        if not self.prediction_service.model:
            st.warning("No model loaded. Please load a model first.")
            return
        
        # Model Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            info_df = pd.DataFrame([
                {"Metric": "Model Type", "Value": model_info.get("model_type", "N/A")},
                {"Metric": "Version", "Value": model_info.get("model_version", "N/A")},
                {"Metric": "Features Count", "Value": model_info.get("features_count", "N/A")},
                {"Metric": "ONNX Enabled", "Value": "Yes" if model_info.get("onnx_enabled") else "No"}
            ])
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Performance Metrics")
            # This would be populated from actual model metrics
            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy", "Value": "85.2%"},
                {"Metric": "Precision", "Value": "83.1%"},
                {"Metric": "Recall", "Value": "87.4%"},
                {"Metric": "F1-Score", "Value": "85.2%"},
                {"Metric": "AUC-ROC", "Value": "89.6%"}
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Feature Importance (if available)
        if hasattr(self.prediction_service.model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance = self.prediction_service.model.feature_importances_
            feature_names = getattr(self.prediction_service.model, 'feature_names_', 
                                  [f"Feature_{i}" for i in range(len(importance))])
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_prediction_result(self, prediction: PredictionResult, 
                                  settings: Dict[str, Any]) -> None:
        """Display single prediction result"""
        st.subheader("Prediction Results")
        
        # Main result card
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Risk level with color coding
            risk_color = {
                "Low": "green",
                "Medium": "orange", 
                "High": "red"
            }.get(prediction.risk_category, "gray")
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {risk_color}; background-color: #f8f9fa;">
                <h3 style="margin: 0; color: {risk_color};">Risk Level</h3>
                <h2 style="margin: 0;">{prediction.risk_category}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Pass Probability",
                f"{prediction.probability_pass:.1%}",
                delta=f"{prediction.probability_pass - 0.5:.1%}" if prediction.probability_pass != 0.5 else None
            )
        
        with col3:
            st.metric(
                "Confidence",
                f"{prediction.confidence_score:.1%}",
                delta=None
            )
        
        # Probability visualization
        st.subheader("Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Pass', 'Fail'],
                y=[prediction.probability_pass, prediction.probability_fail],
                marker_color=['green', 'red'],
                text=[f"{prediction.probability_pass:.1%}", f"{prediction.probability_fail:.1%}"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Pass/Fail Probability",
            yaxis_title="Probability",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Contributing factors
        if prediction.contributing_factors and settings.get("show_explanations", True):
            st.subheader("Key Contributing Factors")
            
            factors_df = pd.DataFrame([
                {"Factor": factor, "Contribution": contribution}
                for factor, contribution in prediction.contributing_factors.items()
            ]).sort_values("Contribution", ascending=False)
            
            fig = px.bar(factors_df, x='Contribution', y='Factor', 
                        orientation='h', title="Factors Influencing Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            explanation = self.prediction_service.explain_prediction(prediction)
            
            st.subheader("Interpretation")
            st.info(explanation["interpretation"])
            
            st.subheader("Recommendations")
            for rec in explanation["recommendations"]:
                st.write(f"• {rec}")
    
    def _display_batch_results(self, predictions: List[PredictionResult], 
                              settings: Dict[str, Any]) -> None:
        """Display batch prediction results"""
        st.subheader("Batch Prediction Results")
        
        # Summary statistics
        total_students = len(predictions)
        high_risk = sum(1 for p in predictions if p.risk_category == "High")
        medium_risk = sum(1 for p in predictions if p.risk_category == "Medium")
        low_risk = sum(1 for p in predictions if p.risk_category == "Low")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", total_students)
        with col2:
            st.metric("High Risk", high_risk, delta=f"{high_risk/total_students:.1%}")
        with col3:
            st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total_students:.1%}")
        with col4:
            st.metric("Low Risk", low_risk, delta=f"{low_risk/total_students:.1%}")
        
        # Risk distribution chart
        risk_data = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [high_risk, medium_risk, low_risk],
            'Percentage': [high_risk/total_students*100, medium_risk/total_students*100, low_risk/total_students*100]
        })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level', 
                    title="Risk Distribution",
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Results")
        
        results_df = pd.DataFrame([
            {
                "Student ID": p.student_id,
                "Risk Level": p.risk_category,
                "Pass Probability": f"{p.probability_pass:.1%}",
                "Confidence": f"{p.confidence_score:.1%}",
                "Top Factor": max(p.contributing_factors.items(), key=lambda x: x[1])[0] if p.contributing_factors else "N/A"
            }
            for p in predictions
        ])
        
        # Filter by confidence threshold if set
        confidence_threshold = settings.get("confidence_threshold", 0.0)
        if confidence_threshold > 0:
            filtered_df = results_df[
                results_df["Confidence"].str.rstrip('%').astype(float) / 100 >= confidence_threshold
            ]
            st.info(f"Showing {len(filtered_df)} students with confidence >= {confidence_threshold:.1%}")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
    
    def render_data_upload_page(self) -> None:
        """Render data upload and management page"""
        st.header("Data Upload & Management")
        
        st.subheader("Upload Training Data")
        
        uploaded_file = st.file_uploader(
            "Upload training dataset",
            type=['csv'],
            help="CSV should contain student features and outcomes for model training"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Data quality report
                st.subheader("Data Quality Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Statistics:**")
                    st.write(f"• Rows: {len(df):,}")
                    st.write(f"• Columns: {len(df.columns)}")
                    st.write(f"• Missing Values: {df.isnull().sum().sum():,}")
                    st.write(f"• Duplicates: {df.duplicated().sum():,}")
                
                with col2:
                    st.write("**Data Types:**")
                    for dtype, count in df.dtypes.value_counts().items():
                        st.write(f"• {dtype}: {count} columns")
                
                # Missing values visualization
                if df.isnull().sum().sum() > 0:
                    st.subheader("Missing Values by Column")
                    missing_data = df.isnull().sum().sort_values(ascending=False)
                    missing_data = missing_data[missing_data > 0]
                    
                    fig = px.bar(x=missing_data.values, y=missing_data.index, 
                               orientation='h', title="Missing Values Count")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def render_footer() -> None:
    """Render application footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            EdunSight v1.0.0 | Built with Streamlit | 
            <a href="https://github.com" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )
