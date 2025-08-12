# EdunSight - Student Performance Prediction ML Pipeline

A lightweight, real-time ML pipeline that predicts student performance from CSV/LMS logs and serves fast predictions via an interactive web app.

## Features

- **Fast Training & Inference**: Uses LightGBM and ONNX Runtime for optimal performance
- **MVVM Architecture**: Clean separation of concerns for maintainable code
- **Real-time Predictions**: Streamlit-based interactive dashboard
- **Free Datasets**: Uses OULAD and UCI Student Performance datasets
- **Production Ready**: Includes monitoring, testing, and deployment scripts

## Quick Start

1. **Setup Environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

2. **Download Data**

```bash
python src/utils/data_downloader.py
```

3. **Train Model**

```bash
python src/models/train.py
```

4. **Run Dashboard**

```bash
streamlit run app.py
```

## Project Structure

```
EdunSight/
├── src/
│   ├── models/          # Data models and ML models (Model layer)
│   ├── viewmodels/      # Business logic and data processing (ViewModel layer)
│   ├── views/           # UI components and Streamlit pages (View layer)
│   └── utils/           # Utilities and helpers
├── data/                # Raw and processed datasets
├── models/              # Trained model artifacts
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for exploration
└── app.py               # Main Streamlit application
```

## Architecture (MVVM)

- **Models**: Data structures, ML models, database interfaces
- **ViewModels**: Business logic, data preprocessing, prediction logic
- **Views**: Streamlit UI components, dashboards, forms

## Performance Optimizations

- LightGBM for fast gradient boosting
- ONNX Runtime for ultra-fast inference
- Chunked data loading for large datasets
- Feature selection for model efficiency
- Caching for repeated operations

## Datasets

- **OULAD**: Open University Learning Analytics Dataset
- **UCI Student Performance**: University student grades dataset

## Tech Stack

- **ML**: scikit-learn, LightGBM, ONNX Runtime
- **Web App**: Streamlit, Plotly
- **Data**: Pandas, NumPy
- **Model Interpretation**: SHAP

## License

MIT License
