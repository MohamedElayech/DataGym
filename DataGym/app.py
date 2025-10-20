import os
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import json
import traceback
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processing import detect_target_column, process_data
from model_utils import get_model_param_grid, evaluate_model
from ai_interpretation import generate_ai_interpretation

# Read Hugging Face token from environment (do not hardcode secrets)
# Set it in your shell before running the app, e.g., $Env:HF_TOKEN = "<token>" on PowerShell
hf_token = os.getenv("HF_TOKEN")

def train_and_evaluate(file_obj, target_column, model_name, metrics, hf_token=None):
    """Main function to process data, train model and evaluate results"""
    # Process data
    try:
        processed_data, original_df, preprocessing_report = process_data(file_obj, target_column)
        if processed_data is None:  # Error in processing
            return preprocessing_report, None, None, None
        
        X_train, X_test, y_train, y_test = processed_data
        
        # Get model and parameter grid
        model, param_grid = get_model_param_grid(model_name)
        if model is None:
            return f"Error: Unknown model '{model_name}'.", None, None, None
        
        # Train with Grid Search CV
        progress_text = f"\n## Training Progress\n\n- Initializing {model_name} model with Grid Search CV\n"
        progress_text += f"- Searching best parameters from: {json.dumps(param_grid, indent=2)}\n"
        progress_text += f"- Data shape: X_train={X_train.shape}, y_train={y_train.shape}\n"
        progress_text += f"- Target classes: {len(np.unique(y_train))}\n"
        
        # Use error_score='raise' to get detailed error messages
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', 
            verbose=0, error_score='raise', n_jobs=1
        )
        
        try:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            progress_text += f"- Best parameters found: {best_params}\n"
        except Exception as fit_error:
            error_details = traceback.format_exc()
            return preprocessing_report + f"\n\n**Training Error**: {str(fit_error)}\n\n```\n{error_details}\n```", None, None, None
        
        # Evaluate model
        metrics_results, cm_plot_path, class_report = evaluate_model(best_model, X_test, y_test, metrics)
        metrics_text = "\n".join([f"- **{k}**: {v:.4f}" for k, v in metrics_results.items()])
        results_report = f"""## Model Results\n\n### Metrics\n{metrics_text}\n\n### Best Hyperparameters\n```json\n{json.dumps(best_params, indent=2)}\n```\n"""
        
        # Generate AI interpretation
        interpretation = generate_ai_interpretation(model_name, metrics_results, class_report, hf_token=hf_token)
        
        # Final report
        final_report = f"{preprocessing_report}\n{progress_text}\n{results_report}\n## AI Interpretation\n{interpretation}"
        
        return final_report, cm_plot_path, metrics_results, best_params
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, None, None, None

def create_interface():
    global hf_token  # Use the token set earlier
    with gr.Blocks(title="ML Model Trainer") as app:
        gr.Markdown("# Machine Learning Model Trainer")
        gr.Markdown("Upload your dataset, select parameters, and train machine learning models")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                file_input = gr.File(label="Upload Dataset (CSV)")
                
                # Auto-detection info and dropdown list
                target_info = gr.Markdown("Upload a dataset to auto-detect target column")
                target_column = gr.Dropdown(
                    label="Target Column (Auto-detected)",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False
                )
                
                model_choice = gr.Dropdown(
                    choices=["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "XGBoost"],
                    label="Select Model",
                    value="Logistic Regression"
                )
                metrics_choice = gr.CheckboxGroup(
                    choices=["accuracy", "precision", "recall", "f1"],
                    label="Evaluation Metrics",
                    value=["accuracy"]
                )
                train_button = gr.Button("Train Model")
            
            with gr.Column(scale=2):
                # Output components
                output_report = gr.Markdown(label="Report")
                with gr.Row():
                    confusion_matrix_plot = gr.Image(label="Confusion Matrix", type="filepath")
        
        # Auto-detect target and populate dropdown when file is uploaded
        def on_file_upload(file):
            if file is None:
                return "Upload a dataset to auto-detect target column", gr.Dropdown(choices=[], value=None)
            
            # Get all columns and detected target
            try:
                if hasattr(file, 'name'):
                    df = pd.read_csv(file.name)
                else:
                    content = file.decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                
                all_columns = df.columns.tolist()
                detected_col, message = detect_target_column(file)
                
                # Return message and updated dropdown with all columns
                return message, gr.Dropdown(choices=all_columns, value=detected_col)
            except Exception as e:
                return f"Error reading file: {str(e)}", gr.Dropdown(choices=[], value=None)
        
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[target_info, target_column]
        )
        
        # Connect the button to the train_and_evaluate function, passing hf_token
        train_button.click(
            fn=lambda file_obj, target_column, model_name, metrics: train_and_evaluate(
                file_obj, target_column, model_name, metrics, hf_token=hf_token
            ),
            inputs=[file_input, target_column, model_choice, metrics_choice],
            outputs=[output_report, confusion_matrix_plot, gr.JSON(visible=False), gr.JSON(visible=False)]
        )
        
    return app

# Create and launch the interface
if __name__ == "__main__":
    # Install required packages if not already installed
    # Uncomment the following line if you want to install packages automatically
    # import os
    # os.system("pip install -q gradio pandas numpy scikit-learn matplotlib transformers torch xgboost")
    
    app = create_interface()
    app.launch()