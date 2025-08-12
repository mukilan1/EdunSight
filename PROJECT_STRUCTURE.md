# EdunSight Project Structure

```
EdunSight/
├── 📱 app.py                    # Main Streamlit application
├── 🔧 train.py                  # Model training script
├── 🎯 demo.py                   # Quick demo script
├── ⚙️ setup.py                  # Environment setup script
├── 🧪 run_tests.py              # Test runner
├── 📋 config.yaml               # Configuration file
├── 📖 README.md                 # Project documentation
├── 📦 requirements.txt          # Python dependencies
├── 
├── 📁 src/                      # MVVM Source Code
│   ├── 🎯 models/               # Model Layer (Data & ML Models)
│   │   ├── data_models.py       # Data structures & schemas
│   │   ├── ml_models.py         # ML model implementations
│   │   └── __init__.py
│   │
│   ├── 🔄 viewmodels/           # ViewModel Layer (Business Logic)
│   │   ├── data_processor.py    # Data preprocessing & feature engineering
│   │   ├── prediction_service.py # Prediction logic & model management
│   │   ├── training_service.py  # Training logic & model evaluation
│   │   └── __init__.py
│   │
│   ├── 👁️ views/                # View Layer (UI Components)
│   │   ├── dashboard.py         # Streamlit dashboard components
│   │   └── __init__.py
│   │
│   ├── 🛠️ utils/                # Utilities
│   │   ├── data_downloader.py   # Dataset download & management
│   │   ├── logging_config.py    # Logging configuration
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📊 data/                     # Data Storage
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
│
├── 🤖 models/                   # Trained Model Artifacts
│   ├── *.joblib                 # Serialized models
│   └── *.onnx                   # ONNX optimized models
│
├── 📓 notebooks/                # Jupyter Notebooks
│   └── edusight_exploration.ipynb # Data exploration & analysis
│
└── 🧪 tests/                    # Test Suite
    ├── conftest.py              # Test configuration
    ├── test_data_models.py      # Model layer tests
    ├── test_data_processor.py   # Data processing tests
    └── test_prediction_service.py # Prediction service tests
```

## 🏗️ MVVM Architecture

### **Model Layer** (`src/models/`)

- **Data Models**: `StudentRecord`, `PredictionResult`, `ModelMetrics`
- **ML Models**: `LightGBMModel`, `RandomForestModel`, `ONNXModel`
- **Schemas**: `FeatureSchema`, `ModelConfig`

### **ViewModel Layer** (`src/viewmodels/`)

- **DataProcessor**: Data cleaning, feature engineering, validation
- **TrainingService**: Model training, evaluation, hyperparameter tuning
- **PredictionService**: Predictions, explanations, model management

### **View Layer** (`src/views/`)

- **DashboardView**: Streamlit UI components
- **Interactive Forms**: Single/batch prediction interfaces
- **Visualizations**: Charts, metrics, analytics

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
python setup.py

# 2. Quick demo (downloads data + trains model)
python demo.py

# 3. Full training pipeline
python train.py --download-data --all-models

# 4. Run web application
streamlit run app.py

# 5. Run tests
python run_tests.py
```

## 🎯 Key Features

### **Fast & Optimized**

- ⚡ LightGBM for rapid training
- 🏃 ONNX Runtime for ultra-fast inference
- 📦 Chunked data processing
- 🗄️ Feature caching

### **Production Ready**

- 🧪 Comprehensive test suite
- 📊 Model monitoring & drift detection
- 🔄 Automated retraining pipeline
- 📈 Performance metrics tracking

### **User Friendly**

- 🎨 Interactive Streamlit dashboard
- 📱 Responsive design
- 💡 Prediction explanations
- 📋 Actionable recommendations

### **Scalable Architecture**

- 🏗️ Clean MVVM separation
- 🔌 Pluggable model types
- 📦 Microservice ready
- ☁️ Cloud deployment friendly

## 📊 Performance Targets

- **Training Time**: < 2 minutes for 10K samples
- **Prediction Time**: < 100ms per student
- **Accuracy**: > 85% on test set
- **Memory Usage**: < 1GB RAM
- **Model Size**: < 50MB

## 🎓 Educational Use Cases

1. **Early Warning System**: Identify at-risk students
2. **Resource Allocation**: Prioritize intervention efforts
3. **Academic Planning**: Predict course success rates
4. **Personalized Learning**: Tailor support strategies
5. **Institutional Analytics**: Track program effectiveness
