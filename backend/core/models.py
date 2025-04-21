# core/models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple
import joblib
import os
import logging
import json # <-- Import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache"

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

def train_evaluate_baseline(
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str],
    feature_columns: List[str] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    # --- Start of function is the same (loading, cleaning, splitting) ---
    # ... (Keep all the loading, cleaning, feature selection, splitting logic as before) ...
    # ... Ensure valid_sensitive_columns and feature_columns are correctly defined ...
    # ... Ensure X_train_full, X_test_full, y_train, y_test are created ...

    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        df = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
        logger.info(f"Data loaded successfully from {filepath}, treating '?' as NaN.")
    except Exception as e:
        logger.exception(f"Error reading CSV file: {e}")
        raise ValueError(f"Error reading CSV file: {e}")

    # --- Clean Target Column ---
    if target_column not in df.columns:
         raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    if df[target_column].isnull().any():
        logger.warning(f"Target column '{target_column}' contains {df[target_column].isnull().sum()} missing values. Dropping rows with missing target.")
        df.dropna(subset=[target_column], inplace=True)
    if df[target_column].dtype == 'object':
        unique_vals = df[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K'
            negative_class_marker = '<=50K'
            if positive_class_marker in unique_vals and negative_class_marker in unique_vals:
                 df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int)
                 logger.info(f"Target column '{target_column}' cleaned to 0/1 (1 represents '{positive_class_marker}').")
            else:
                 logger.warning(f"Expected markers '{positive_class_marker}'/'{negative_class_marker}' not found. Assuming '{unique_vals[0]}' as positive class (1).")
                 df[target_column] = df[target_column].apply(lambda x: 1 if x == unique_vals[0] else 0).astype(int)
        else:
            raise ValueError(f"Target column '{target_column}' is object type but does not have exactly 2 unique values after dropping NaNs. Found: {unique_vals}")
    elif pd.api.types.is_numeric_dtype(df[target_column]):
         unique_numeric_vals = df[target_column].unique()
         if set(unique_numeric_vals) == {0, 1}:
             logger.info(f"Target column '{target_column}' is already numeric binary (0/1).")
             df[target_column] = df[target_column].astype(int)
         else:
             logger.warning(f"Target column '{target_column}' is numeric but not binary 0/1. Values found: {unique_numeric_vals}.")
    else:
         raise ValueError(f"Target column '{target_column}' has an unsupported data type: {df[target_column].dtype}")

    # --- Feature Selection ---
    valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df.columns]
    if len(valid_sensitive_columns) != len(sensitive_attribute_columns):
         missing_sens_cols = set(sensitive_attribute_columns) - set(valid_sensitive_columns)
         logger.warning(f"Provided sensitive attributes not found in data: {missing_sens_cols}. Proceeding without them.")

    if feature_columns is None:
        feature_columns = [
            col for col in df.columns
            if col != target_column and col not in valid_sensitive_columns
        ]
        logger.info(f"Using automatically selected features for training: {feature_columns}")
    else:
        missing_feats = [col for col in feature_columns if col not in df.columns]
        if missing_feats:
             raise ValueError(f"Provided feature columns not found in data: {missing_feats}")
        logger.info(f"Using provided features for training: {feature_columns}")

    # --- Data Split ---
    cols_to_keep = feature_columns + valid_sensitive_columns + [target_column]
    df_subset = df[[col for col in cols_to_keep if col in df.columns]].copy() # Ensure columns exist
    try:
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            df_subset.drop(columns=[target_column]),
            df_subset[target_column],
            test_size=test_size,
            random_state=random_state,
            stratify=df_subset[target_column]
        )
        logger.info(f"Data split into training ({len(X_train_full)} samples) and testing ({len(X_test_full)} samples).")
    except Exception as e:
        logger.exception(f"Error during train_test_split: {e}")
        raise ValueError(f"Error during train_test_split: Ensure target column '{target_column}' exists and is suitable for stratification. {e}")

    # --- Preprocessing Setup ---
    X_train_features_only = X_train_full[feature_columns]
    numerical_features_train = X_train_features_only.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only.select_dtypes(exclude=np.number).columns.tolist()
    logger.info(f"Identified numerical features for preprocessing: {numerical_features_train}")
    logger.info(f"Identified categorical features for preprocessing: {categorical_features_train}")
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_train),
            ('cat', categorical_transformer, categorical_features_train)],
        remainder='drop')

    # --- Model ---
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced')

    # --- Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # --- Training ---
    logger.info("Starting model training...")
    try:
        pipeline.fit(X_train_full, y_train)
    except Exception as e:
         logger.exception(f"Error during pipeline fitting: {e}")
         # ... (error logging) ...
         raise e
    logger.info("Model training complete.")

    # --- Evaluation ---
    # ... (Keep evaluation logic as before) ...
    y_pred_test = pipeline.predict(X_test_full)
    results = {"metrics": {}, "model_info": {}}
    # ... calculate overall metrics ...
    # ... calculate disaggregated metrics ...
    try:
        results["metrics"]["overall"] = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall": recall_score(y_test, y_pred_test, zero_division=0),
            "f1": f1_score(y_test, y_pred_test, zero_division=0),
        }
        logger.info(f"Overall Test Metrics: {results['metrics']['overall']}")
    except Exception as e:
         logger.exception(f"Error calculating overall metrics: {e}")
         results["metrics"]["overall"] = {"error": str(e)}
    # Disaggregated metrics calculation ...
    results["metrics"]["by_sensitive_group"] = {}
    for sens_col in valid_sensitive_columns:
        # ... (logic for calculating metrics per group) ...
        pass # Placeholder for brevity, keep the full logic from previous version

    # --- Save Model Pipeline ---
    pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)
    try:
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Pipeline saved to {pipeline_path}")
    except Exception as e:
        logger.exception(f"Error saving pipeline to {pipeline_path}: {e}")
        raise IOError(f"Could not save pipeline: {e}")

    # --- *** NEW: SAVE METADATA *** ---
    metadata_filename = pipeline_path.replace('.joblib', '.meta.json')
    metadata_to_save = {
        "pipeline_path": pipeline_path, # Keep path for reference if needed
        "original_filename": filename,
        "features_used_in_training": feature_columns, # Crucial list
        "sensitive_attributes_present": valid_sensitive_columns, # Actual sensitive cols found/used
        "target_column": target_column,
        "model_type": type(model).__name__,
        "training_random_state": random_state,
        # Add any other info needed by later steps
    }
    try:
        with open(metadata_filename, 'w') as f:
            json.dump(metadata_to_save, f, indent=4)
        logger.info(f"Metadata saved to {metadata_filename}")
    except Exception as e:
        logger.exception(f"Error saving metadata to {metadata_filename}: {e}")
        # Don't raise error here? Or should we? If metadata fails, explain will fail. Let's raise it.
        raise IOError(f"Could not save metadata: {e}")
    # --- *** END NEW *** ---


    # Include paths and key info in the results returned by the API endpoint
    results["model_info"] = {
        "pipeline_path": pipeline_path,
        "metadata_path": metadata_filename, # Add path to metadata file
        "features_used_in_training": feature_columns,
        "sensitive_attributes_present": valid_sensitive_columns,
        "target_column": target_column,
        "model_type": type(model).__name__,
    }

    return results