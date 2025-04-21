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
    """ ... """
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
    # (Keep the robust cleaning logic from the previous version)
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
        # Automatically select features (excluding target AND sensitive)
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
    # Split the *entire* dataframe first, keeping relevant columns
    cols_to_keep = feature_columns + valid_sensitive_columns + [target_column]
    df_subset = df[cols_to_keep].copy()

    try:
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            df_subset.drop(columns=[target_column]), # X contains features + sensitive
            df_subset[target_column],               # y is the target
            test_size=test_size,
            random_state=random_state,
            stratify=df_subset[target_column] # Stratify on target
        )
        logger.info(f"Data split into training ({len(X_train_full)} samples) and testing ({len(X_test_full)} samples).")
    except Exception as e:
        logger.exception(f"Error during train_test_split: {e}")
        raise ValueError(f"Error during train_test_split: Ensure target column '{target_column}' exists and is suitable for stratification. {e}")


    # --- Preprocessing Setup (Applied ONLY to Features) ---
    # Identify feature types based on the feature columns ONLY within X_train_full
    X_train_features_only = X_train_full[feature_columns]

    numerical_features_train = X_train_features_only.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only.select_dtypes(exclude=np.number).columns.tolist()

    logger.info(f"Identified numerical features for preprocessing: {numerical_features_train}")
    logger.info(f"Identified categorical features for preprocessing: {categorical_features_train}")

    # Define preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the preprocessor ColumnTransformer - **IMPORTANT CHANGE HERE**
    # Apply transformers ONLY to the appropriate feature columns
    # Set remainder='drop' because we don't want the sensitive attributes passed to the model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_train),
            ('cat', categorical_transformer, categorical_features_train)
        ],
        remainder='drop' # <--- KEY CHANGE: Drop columns not specified (i.e., sensitive attributes)
    )

    # --- Model ---
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced')

    # --- Create the FULL Training Pipeline ---
    # The input to this pipeline will be X_train_full (features + sensitive)
    # The preprocessor step will select ONLY the feature columns and transform them
    # The output of the preprocessor (only transformed features) goes to the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- Training ---
    logger.info("Starting model training...")
    try:
        # Fit the pipeline on X_train_full and y_train
        pipeline.fit(X_train_full, y_train)
    except Exception as e:
         logger.exception(f"Error during pipeline fitting: {e}")
         logger.error(f"X_train_full shape: {X_train_full.shape}")
         logger.error(f"y_train shape: {y_train.shape}, unique values: {y_train.unique()}")
         logger.error(f"X_train_full dtypes:\n{X_train_full.dtypes}")
         raise e
    logger.info("Model training complete.")

    # --- Evaluation ---
    # Predict using the fitted pipeline on X_test_full
    # The pipeline will automatically apply the same preprocessing (selecting features, transforming)
    y_pred_test = pipeline.predict(X_test_full)

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
    # Use X_test_full here as it contains the sensitive attribute columns needed for grouping
    for sens_col in valid_sensitive_columns:
        if sens_col in X_test_full.columns: # Check existence in the test split data
            results["metrics"]["by_sensitive_group"][sens_col] = {}
            unique_groups = X_test_full[sens_col].dropna().unique()
            logger.info(f"Calculating disaggregated metrics for '{sens_col}', groups found: {unique_groups}")

            for group in unique_groups:
                # Get mask from X_test_full which contains the sensitive column
                group_mask = X_test_full[sens_col] == group
                # Apply mask to y_test and y_pred_test
                y_test_group = y_test[group_mask]
                y_pred_group = y_pred_test[group_mask] # y_pred_test corresponds row-wise to X_test_full

                group_metrics = {"sample_count": len(y_test_group)}
                if len(y_test_group) > 0:
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
                results["metrics"]["by_sensitive_group"][sens_col][str(group)] = group_metrics

            # Handle NaN group (using X_test_full for mask)
            nan_mask = X_test_full[sens_col].isnull()
            if nan_mask.any():
                 y_test_nan_group = y_test[nan_mask]
                 y_pred_nan_group = y_pred_test[nan_mask]
                 nan_group_metrics = {"sample_count": len(y_test_nan_group)}
                 if len(y_test_nan_group) > 0:
                     try:
                        nan_group_metrics["accuracy"] = accuracy_score(y_test_nan_group, y_pred_nan_group)
                        # ... other metrics ...
                     except Exception as e:
                        logger.exception(f"Error calculating metrics for NaN group in column '{sens_col}': {e}")
                        nan_group_metrics["error"] = str(e)
                 else:
                     nan_group_metrics["message"] = "No samples in test set for NaN group."
                 results["metrics"]["by_sensitive_group"][sens_col]["__NaN__"] = nan_group_metrics

            logger.info(f"Disaggregated metrics calculation attempted for '{sens_col}'.")
        else:
            logger.error(f"Internal Error: Sensitive column '{sens_col}' expected but not found in X_test_full columns.")


    # --- Save Model & Metadata ---
    model_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
    model_path = os.path.join(MODEL_DIRECTORY, model_filename)

    try:
        joblib.dump(pipeline, model_path)
        logger.info(f"Pipeline saved to {model_path}")
    except Exception as e:
        logger.exception(f"Error saving pipeline to {model_path}: {e}")
        raise IOError(f"Could not save pipeline: {e}")

    # Store info needed later
    results["model_info"] = {
        "pipeline_path": model_path,
        "features_used_in_training": feature_columns, # The list of feature names input to preprocessor
        "sensitive_attributes_present": valid_sensitive_columns,
        "target_column": target_column,
        "model_type": type(model).__name__,
        "training_random_state": random_state,
    }

    return results