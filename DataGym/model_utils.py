import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

def get_model_param_grid(model_name):
    """Return model and parameter grid based on model name"""
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
        }
    elif model_name == "KNN":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif model_name == "SVM":
        model = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    elif model_name == "XGBoost":
        # use_label_encoder is deprecated and ignored in recent XGBoost versions
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    else:
        return None, None
    
    return model, param_grid

def evaluate_model(model, X_test, y_test, metrics):
    """Evaluate the model on test data"""
    try:
        y_pred = model.predict(X_test)
        
        results = {}
        if "accuracy" in metrics:
            results["Accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics:
            results["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0) 
        if "recall" in metrics:
            results["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        if "f1" in metrics:
            results["F1 Score"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_path = 'confusion_matrix.png'
        plt.savefig(cm_plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Get detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return results, cm_plot_path, class_report
    
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return {}, None, {}