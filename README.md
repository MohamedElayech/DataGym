# DataGym

A machine learning model training and evaluation platform with an interactive Gradio interface. This project provides automated data preprocessing, model training with hyperparameter tuning, and AI-powered result interpretation.

## Project Structure

### Main Application (`DataGym/`)
The core application files that power the interactive ML training interface:

- **`app.py`** - Main Gradio application with web interface for model training
- **`data_processing.py`** - Data preprocessing pipeline (handling missing values, encoding, scaling)
- **`model_utils.py`** - Model definitions and evaluation utilities
- **`ai_interpretation.py`** - AI-powered interpretation of model results using Hugging Face models

### Notebooks (Testing & Development)
These Jupyter notebooks are for **testing, experimentation, and understanding the logic** behind the core functions:

- **`data_exploration.ipynb`** - Exploratory data analysis and understanding dataset characteristics
- **`data_preprocesing.ipynb`** - Testing and validating preprocessing logic before integration
- **`testing_algorithms.ipynb`** - Experimenting with different ML algorithms and understanding their behavior

> **Note:** The notebooks serve as a development sandbox to prototype and validate the essential needs of the functions before they are implemented in the production code within the `DataGym/` directory.

## Features

- **Automatic Target Detection** - Intelligently identifies the target column in your dataset
- **Data Preprocessing** - Handles missing values, categorical encoding, and feature scaling
- **Multiple ML Models** - Support for Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost
- **Hyperparameter Tuning** - Automated Grid Search CV for optimal model parameters
- **Model Evaluation** - Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Confusion Matrix Visualization** - Visual representation of model performance
- **AI Interpretation** - Natural language explanations of model results

## Setup & Usage

### Prerequisites
```bash
pip install gradio pandas numpy scikit-learn matplotlib seaborn transformers torch xgboost
```

### Running the Application
```bash
cd DataGym
python app.py
```

### Environment Variables
Set your Hugging Face token for AI interpretation features:
change the content of the variable with your Hugging Face token
```python
hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## How to Use

1. Upload your CSV dataset
2. Select or confirm the auto-detected target column
3. Choose a machine learning model
4. Select evaluation metrics
5. Click "Train Model" to start training
6. View results, metrics, and AI interpretation
