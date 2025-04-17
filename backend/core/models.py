import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Tuple
import joblib
import os
import logging # Added logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache" # Store trained models/preprocessors here

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

def train_evaluate_baseline(
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str],
    feature_columns: List[str] = None, # Optional: specify features, else use all non-target/sensitive
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a baseline model, evaluates it overall and per sensitive group,
    and saves the model and preprocessor.
    """
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, skipinitialspace=True)
        logger.info(f"Data loaded successfully from {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # --- Clean Target Column (same logic as analysis) ---
    if df[target_column].dtype == 'object':
        if df[target_column].nunique() == 2:
            unique_vals = df[target_column].unique()
            positive_class_marker = unique_vals[0]
            df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0)
            logger.info(f"Target column '{target_column}' cleaned to 0/1.")
        else:
            logger.warning(f"Target column '{target_column}' is object type but not binary. Model training might fail.")
            # Consider raising an error here instead of just warning

    # --- Feature Selection ---
    if feature_columns is None:
        feature_columns = [
            col for col in df.columns
            if col != target_column and col not in sensitive_attribute_columns
        ]
        logger.info(f"Using automatically selected features: {feature_columns}")

    # --- Data Split ---
    X = df[feature_columns + sensitive_attribute_columns] # Keep sensitive for now, handle in pipeline if needed or for post-hoc analysis
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
         raise ValueError("Target column is not numeric. Please ensure it's cleaned/encoded properly.")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify helps with imbalanced datasets
    )
    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # --- Preprocessing ---
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove sensitive attributes from features used for *training* if they are categorical/numerical
    # We keep them in X_test for evaluation based on groups
    numerical_features_train = [f for f in numerical_features if f not in sensitive_attribute_columns]
    categorical_features_train = [f for f in categorical_features if f not in sensitive_attribute_columns]


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_train),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_train)
        ],
        remainder='passthrough' # Keep other columns (like sensitive attributes if not num/cat)
    )

    # --- Model ---
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced') # Example

    # --- Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- Training ---
    logger.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete.")

    # --- Evaluation ---
    y_pred_test = pipeline.predict(X_test)

    results = {"metrics": {}, "model_info": {}}

    # Overall metrics
    results["metrics"]["overall"] = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
    }
    logger.info(f"Overall Test Metrics: {results['metrics']['overall']}")


    # Disaggregated metrics
    results["metrics"]["by_sensitive_group"] = {}
    for sens_col in sensitive_attribute_columns:
        if sens_col in X_test.columns:
            results["metrics"]["by_sensitive_group"][sens_col] = {}
            unique_groups = X_test[sens_col].unique()
            for group in unique_groups:
                group_mask = X_test[sens_col] == group
                y_test_group = y_test[group_mask]
                y_pred_group = y_pred_test[group_mask]

                if len(y_test_group) > 0: # Avoid division by zero if a group has no samples in test set
                    results["metrics"]["by_sensitive_group"][sens_col][group] = {
                         "sample_count": len(y_test_group),
                         "accuracy": accuracy_score(y_test_group, y_pred_group),
                         "precision": precision_score(y_test_group, y_pred_group, zero_division=0),
                         "recall": recall_score(y_test_group, y_pred_group, zero_division=0),
                         "f1": f1_score(y_test_group, y_pred_group, zero_division=0),
                    }
                else:
                     results["metrics"]["by_sensitive_group"][sens_col][group] = {"sample_count": 0, "message": "No samples in test set for this group."}

            logger.info(f"Disaggregated metrics calculated for '{sens_col}'.")
        else:
            logger.warning(f"Sensitive column '{sens_col}' not found in X_test columns for disaggregated metrics.")


    # --- Save Model & Preprocessor ---
    # Use a unique identifier based on filename/timestamp later if needed
    model_filename = f"{os.path.splitext(filename)[0]}_model.joblib"
    preprocessor_filename = f"{os.path.splitext(filename)[0]}_preprocessor.joblib"

    model_path = os.path.join(MODEL_DIRECTORY, model_filename)
    preprocessor_path = os.path.join(MODEL_DIRECTORY, preprocessor_filename)

    joblib.dump(pipeline, model_path) # Save the whole pipeline
    #joblib.dump(preprocessor, preprocessor_path) # No need if saving pipeline

    results["model_info"]["pipeline_path"] = model_path
    #results["model_info"]["preprocessor_path"] = preprocessor_path
    results["model_info"]["features_used"] = numerical_features_train + categorical_features_train
    results["model_info"]["sensitive_attributes_present"] = sensitive_attribute_columns

    logger.info(f"Pipeline saved to {model_path}")

    return results