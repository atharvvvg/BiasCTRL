import pandas as pd
import numpy as np # Added numpy import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer # Added imputer import
from typing import Dict, Any, List, Tuple
import joblib
import os
import logging

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
    and saves the model and preprocessor. Handles '?' missing values.
    """
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # --- MODIFICATION: Handle '?' as NaN ---
        df = pd.read_csv(filepath, skipinitialspace=True, na_values='?') # Add na_values
        logger.info(f"Data loaded successfully from {filepath}, treating '?' as NaN.")
    except Exception as e:
        logger.exception(f"Error reading CSV file: {e}") # Use logger.exception
        raise ValueError(f"Error reading CSV file: {e}")

    # --- Clean Target Column (robustly) ---
    if target_column not in df.columns:
         raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Handle potential missing values in target before cleaning
    if df[target_column].isnull().any():
        logger.warning(f"Target column '{target_column}' contains {df[target_column].isnull().sum()} missing values. Dropping rows with missing target.")
        df.dropna(subset=[target_column], inplace=True)

    if df[target_column].dtype == 'object':
        unique_vals = df[target_column].unique()
        logger.info(f"Unique values in object-type target column '{target_column}': {unique_vals}")
        if len(unique_vals) == 2:
            # Explicitly define positive class for Adult dataset common case
            positive_class_marker = '>50K'
            negative_class_marker = '<=50K' # Assuming this is the other value
            if positive_class_marker in unique_vals and negative_class_marker in unique_vals:
                 df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0)
                 df[target_column] = df[target_column].astype(int) # Ensure integer type
                 logger.info(f"Target column '{target_column}' cleaned to 0/1 (1 represents '{positive_class_marker}').")
            else:
                 # Fallback if expected markers aren't present, use first unique value as positive
                 logger.warning(f"Expected markers '{positive_class_marker}'/'{negative_class_marker}' not found. Assuming '{unique_vals[0]}' as positive class (1).")
                 df[target_column] = df[target_column].apply(lambda x: 1 if x == unique_vals[0] else 0).astype(int)
        else:
            raise ValueError(f"Target column '{target_column}' is object type but does not have exactly 2 unique values after dropping NaNs. Found: {unique_vals}")
    elif pd.api.types.is_numeric_dtype(df[target_column]):
         # If already numeric, ensure it's binary 0/1 if possible or log warning
         unique_numeric_vals = df[target_column].unique()
         if set(unique_numeric_vals) == {0, 1}:
             logger.info(f"Target column '{target_column}' is already numeric binary (0/1).")
             df[target_column] = df[target_column].astype(int) # Ensure integer type
         else:
             logger.warning(f"Target column '{target_column}' is numeric but not binary 0/1. Values found: {unique_numeric_vals}. Model performance might be affected.")
             # Consider raising error if strictly binary classification is required
    else:
         raise ValueError(f"Target column '{target_column}' has an unsupported data type: {df[target_column].dtype}")


    # --- Feature Selection ---
    if feature_columns is None:
        # Exclude target AND sensitive cols from FEATURES used for PREDICTION
        # Ensure sensitive columns exist before trying to exclude them
        valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df.columns]
        if len(valid_sensitive_columns) != len(sensitive_attribute_columns):
             missing_sens_cols = set(sensitive_attribute_columns) - set(valid_sensitive_columns)
             logger.warning(f"Provided sensitive attributes not found in data: {missing_sens_cols}. Proceeding without them.")

        feature_columns = [
            col for col in df.columns
            if col != target_column and col not in valid_sensitive_columns
        ]
        logger.info(f"Using automatically selected features for training: {feature_columns}")
    else:
         # Validate provided feature columns
         missing_feats = [col for col in feature_columns if col not in df.columns]
         if missing_feats:
             raise ValueError(f"Provided feature columns not found in data: {missing_feats}")
         logger.info(f"Using provided features for training: {feature_columns}")

    # --- Data Split ---
    # Ensure all columns for split actually exist in the dataframe
    cols_for_split = feature_columns + valid_sensitive_columns + [target_column]
    missing_split_cols = [col for col in cols_for_split if col not in df.columns]
    if missing_split_cols:
        # This should ideally not happen after checks above, but as safeguard:
        raise ValueError(f"Internal Error: Columns needed for data split missing: {missing_split_cols}")

    df_subset = df[cols_for_split].copy() # Work on a copy

    # X contains features + sensitive attributes (sensitive needed for grouping later)
    # Y contains only the target
    X = df_subset[feature_columns + valid_sensitive_columns]
    y = df_subset[target_column]

    # Final check on y type before split
    if not pd.api.types.is_numeric_dtype(y) or not set(y.unique()).issubset({0, 1}):
         raise ValueError("Target column is not numeric binary (0/1) before train/test split. Check cleaning logic.")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")


    # --- Preprocessing with Imputation ---
    # Identify types based on X_train[feature_columns] - features ACTUALLY used for modeling
    X_train_features_only = X_train[feature_columns] # Use only the training features to determine types/transformers

    numerical_features_train = X_train_features_only.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only.select_dtypes(exclude=np.number).columns.tolist() # Assume others are categorical

    logger.info(f"Identified numerical features for preprocessing: {numerical_features_train}")
    logger.info(f"Identified categorical features for preprocessing: {categorical_features_train}")

    # Define preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Impute missing numerical with median
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Set sparse_output=False to debug/avoid sparse error
    ])

    # Create the preprocessor ColumnTransformer
    # Apply transformers ONLY to the appropriate *feature columns*
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_train),
            ('cat', categorical_transformer, categorical_features_train)
        ],
        remainder='passthrough' # Keep other columns (i.e., the sensitive attributes if they weren't in feature_columns)
                                # This ensures sensitive attributes are available in the output of preprocessor if needed downstream
                                # Important: If a sensitive attribute IS ALSO a feature, it gets transformed.
    )

    # --- Model ---
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced') # Example

    # --- Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- Training ---
    logger.info("Starting model training...")
    try:
        # Fit the pipeline on X_train (which includes features + sensitive cols) and y_train
        # The preprocessor inside pipeline correctly selects features based on its definition
        pipeline.fit(X_train, y_train)
    except Exception as e:
         logger.exception(f"Error during pipeline fitting: {e}") # Use logger.exception
         # Provide more context if possible
         logger.error(f"X_train shape: {X_train.shape}")
         logger.error(f"y_train shape: {y_train.shape}, unique values: {y_train.unique()}")
         logger.error(f"X_train dtypes:\n{X_train.dtypes}")
         raise e # Re-raise the error
    logger.info("Model training complete.")

    # --- Evaluation ---
    # Predict needs the same columns as fit
    y_pred_test = pipeline.predict(X_test)

    results = {"metrics": {}, "model_info": {}}

    # Overall metrics
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


    # Disaggregated metrics
    results["metrics"]["by_sensitive_group"] = {}
    # Use the original valid_sensitive_columns list identified earlier
    for sens_col in valid_sensitive_columns:
        if sens_col in X_test.columns:
            results["metrics"]["by_sensitive_group"][sens_col] = {}
            # Handle potential NaN values introduced by split/preprocessing in sensitive columns of X_test
            unique_groups = X_test[sens_col].dropna().unique()
            logger.info(f"Calculating disaggregated metrics for '{sens_col}', groups found: {unique_groups}")

            for group in unique_groups:
                group_mask = X_test[sens_col] == group
                y_test_group = y_test[group_mask]
                y_pred_group = y_pred_test[group_mask]

                group_metrics = {"sample_count": len(y_test_group)}
                if len(y_test_group) > 0: # Avoid division by zero if a group has no samples in test set
                    try:
                        group_metrics["accuracy"] = accuracy_score(y_test_group, y_pred_group)
                        group_metrics["precision"] = precision_score(y_test_group, y_pred_group, zero_division=0)
                        group_metrics["recall"] = recall_score(y_test_group, y_pred_group, zero_division=0)
                        group_metrics["f1"] = f1_score(y_test_group, y_pred_group, zero_division=0)
                    except Exception as e:
                        logger.exception(f"Error calculating metrics for group '{group}' in column '{sens_col}': {e}")
                        group_metrics["error"] = str(e)
                else:
                     group_metrics["message"] = "No samples in test set for this group."

                # Use group label directly as key, converting to string if necessary (e.g., if numeric)
                results["metrics"]["by_sensitive_group"][sens_col][str(group)] = group_metrics

            # Check for NaN group if NaNs were present
            nan_mask = X_test[sens_col].isnull()
            if nan_mask.any():
                 y_test_nan_group = y_test[nan_mask]
                 y_pred_nan_group = y_pred_test[nan_mask]
                 nan_group_metrics = {"sample_count": len(y_test_nan_group)}
                 if len(y_test_nan_group) > 0:
                     try:
                        nan_group_metrics["accuracy"] = accuracy_score(y_test_nan_group, y_pred_nan_group)
                        nan_group_metrics["precision"] = precision_score(y_test_nan_group, y_pred_nan_group, zero_division=0)
                        nan_group_metrics["recall"] = recall_score(y_test_nan_group, y_pred_nan_group, zero_division=0)
                        nan_group_metrics["f1"] = f1_score(y_test_nan_group, y_pred_nan_group, zero_division=0)
                     except Exception as e:
                        logger.exception(f"Error calculating metrics for NaN group in column '{sens_col}': {e}")
                        nan_group_metrics["error"] = str(e)
                 else:
                     nan_group_metrics["message"] = "No samples in test set for NaN group."
                 results["metrics"]["by_sensitive_group"][sens_col]["__NaN__"] = nan_group_metrics # Use specific key for NaN group


            logger.info(f"Disaggregated metrics calculation attempted for '{sens_col}'.")
        else:
            # This should not happen if valid_sensitive_columns is used correctly
            logger.error(f"Internal Error: Sensitive column '{sens_col}' expected but not found in X_test columns for disaggregated metrics.")


    # --- Save Model & Preprocessor ---
    # Use a unique identifier based on filename
    model_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib" # Save pipeline instead of just model
    model_path = os.path.join(MODEL_DIRECTORY, model_filename)

    try:
        joblib.dump(pipeline, model_path) # Save the whole pipeline
        logger.info(f"Pipeline saved to {model_path}")
    except Exception as e:
        logger.exception(f"Error saving pipeline to {model_path}: {e}")
        raise IOError(f"Could not save pipeline: {e}")


    results["model_info"] = {
        "pipeline_path": model_path,
        "features_used_in_training": feature_columns, # Columns selected as features
        "sensitive_attributes_present": valid_sensitive_columns, # Sensitive attributes present in data/used for grouping
        "target_column": target_column,
        "model_type": type(model).__name__,
        "preprocessing_steps": [name for name, _ in pipeline.named_steps['preprocessor'].transformers] + ['remainder'],
        "training_random_state": random_state,
    }

    return results