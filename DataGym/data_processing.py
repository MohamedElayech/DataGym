import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def detect_target_column(file_obj):
    """Automatically detect the most likely target column"""
    try:
        # Read the uploaded file
        if hasattr(file_obj, 'name'):
            df = pd.read_csv(file_obj.name)
        else:
            content = file_obj.decode('utf-8')
            df = pd.read_csv(StringIO(content))
        
        if df.empty or len(df.columns) == 0:
            return None, "Error: Empty dataset"
        
        # Common target column names
        common_targets = [
            'target', 'label', 'class', 'y', 'output', 'result', 
            'deposit', 'diagnosis', 'species', 'quality', 'price',
            'salary', 'income', 'churn', 'survived', 'outcome',
            'category', 'type', 'status', 'approved', 'default'
        ]
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in common_targets:
                return col, f"Auto-detected target: **{col}**"
        
        # Look for partial matches
        for col in df.columns:
            for target_name in common_targets:
                if target_name in col.lower():
                    return col, f"Auto-detected target: **{col}**"
        
        # Heuristic: Last column is often the target
        last_col = df.columns[-1]
        
        # Additional heuristic: column with fewer unique values (likely categorical target)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            unique_counts = {col: df[col].nunique() for col in df.columns}
            # Find column with reasonable number of unique values (2-20 for classification)
            candidates = {col: count for col, count in unique_counts.items() 
                         if 2 <= count <= 20}
            if candidates:
                # Prefer the last column if it's in candidates
                if last_col in candidates:
                    return last_col, f"Auto-detected target: **{last_col}** (last column with {candidates[last_col]} unique values)"
                else:
                    best_col = min(candidates, key=candidates.get)
                    return best_col, f"Auto-detected target: **{best_col}** ({candidates[best_col]} unique values)"
        
        # Fallback: use last column
        return last_col, f"Auto-detected target: **{last_col}** (last column - please verify)"
    
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def process_data(file_obj, target_column):
    """Process the uploaded dataset"""
    try:
        # Read the uploaded file
        if hasattr(file_obj, 'name'):
            df = pd.read_csv(file_obj.name)
        else:
            content = file_obj.decode('utf-8')
            df = pd.read_csv(StringIO(content))
        
        # Store original dataframe
        original_df = df.copy()
        
        # Check if target column exists
        if target_column not in df.columns:
            return None, None, f"Error: Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        # Separate features and target - FIXED: removed inplace=True
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Handle missing values in features BEFORE checking
        missing_report = []
        
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            missing_report.append(f"Found missing values in {len(null_cols)} columns: {', '.join(null_cols)}")
            
            # Identify column types
            categorical_cols_with_nulls = X[null_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols_with_nulls = X[null_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Handle numerical missing values - fill with median
            for col in numerical_cols_with_nulls:
                missing_count = X[col].isnull().sum()
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                missing_report.append(f"  - {col}: Filled {missing_count} missing values with median ({median_val:.2f})")
            
            # Handle categorical missing values - fill with mode (most frequent)
            for col in categorical_cols_with_nulls:
                missing_count = X[col].isnull().sum()
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col].fillna(mode_val, inplace=True)
                missing_report.append(f"  - {col}: Filled {missing_count} missing values with mode ('{mode_val}')")
        else:
            missing_report.append("No missing values found in features.")
        
        # Handle missing values in target
        if y.isnull().any():
            missing_count = y.isnull().sum()
            missing_report.append(f"Warning: Target column '{target_column}' has {missing_count} missing values - removing these rows.")
            
            # Remove rows where target is null
            valid_indices = ~y.isnull()
            y = y[valid_indices]
            X = X[valid_indices]
            
            missing_report.append(f"Removed {missing_count} rows. New dataset size: {len(X)} rows")
        else:
            missing_report.append(f"No missing values in target column '{target_column}'.")
        
        # Create missing values report
        missing_values_report = "\n".join(missing_report)
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y))
            target_encoded = True
            target_type = "Categorical (Encoded to integers)"
        else:
            # Check if target is continuous (float) or discrete (int)
            unique_values = len(y.unique())
            y_range = y.max() - y.min()
            
            # If target has many unique values and appears continuous, convert to classes
            if y.dtype in ['float64', 'float32'] or (unique_values > 10 and y_range > 20):
                # Convert continuous target to binary classification using median
                median_value = y.median()
                y = pd.Series((y > median_value).astype('int64'))
                target_encoded = False
                target_type = f"Continuous (Converted to binary: 0 if ≤ {median_value:.2f}, 1 if > {median_value:.2f})"
            else:
                # Target is already discrete, ensure it's integer
                y = pd.Series(y).astype('int64')
                target_encoded = False
                target_type = "Discrete (Integer classes)"
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Process categorical features with Label Encoding
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Ensure all features are numeric
        X = X.astype('float64')
        
        # Normalize numerical features
        scaler = StandardScaler()
        if numerical_cols:
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Final validation
        if X.shape[0] == 0 or X.shape[1] == 0:
            return None, None, "Error: Dataset has no valid features or samples after processing."
        
        if len(np.unique(y)) < 2:
            return None, None, f"Error: Target variable must have at least 2 unique classes. Found only {len(np.unique(y))}."
        
        # Split data into train and test sets with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as ve:
            # If stratification fails (e.g., too few samples), split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Create a report of the preprocessing steps
        preprocessing_report = f"""## Data Preprocessing Report
    
- **Original Dataset Shape**: {original_df.shape[0]} rows, {original_df.shape[1]} columns
- **Final Dataset Shape**: {X.shape[0]} rows, {X.shape[1]} features
- **Target Column**: {target_column}
- **Target Classes**: {len(np.unique(y))} (Values: {sorted(np.unique(y).tolist())})

### Missing Values Handling
{missing_values_report}

### Feature Encoding & Normalization
- **Train-Test Split**: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)
"""
        
        return (X_train, X_test, y_train, y_test), original_df, preprocessing_report
    
    except Exception as e:
        error_msg = f"Error in data processing: {str(e)}"
        import traceback
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return None, None, error_msg