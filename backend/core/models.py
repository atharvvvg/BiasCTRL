# core/models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # sklearn's Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple
import joblib
import os
import logging
import json
from imblearn.over_sampling import RandomOverSampler # For train_evaluate_oversampled
from imblearn.pipeline import Pipeline as ImbPipeline # For train_evaluate_oversampled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache"

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

# --- Helper Function to Calculate Disaggregated Metrics ---
def _calculate_disaggregated_metrics(
    X_test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    sensitive_attributes: List[str]
) -> Dict[str, Dict[str, Any]]:
    disaggregated_results = {}
    # ...(rest of the function as provided before)...
    logger.info(f"Calculating disaggregated metrics for attributes: {sensitive_attributes}")
    for sens_col in sensitive_attributes:
        if sens_col in X_test_df.columns:
            disaggregated_results[sens_col] = {}
            unique_groups = X_test_df[sens_col].dropna().unique()
            logger.debug(f"Processing sensitive attribute '{sens_col}', groups found: {unique_groups}")
            for group in unique_groups:
                group_mask = X_test_df[sens_col] == group
                y_test_group = y_test[group_mask]
                y_pred_group = y_pred_test[group_mask]
                group_metrics = {"sample_count": int(len(y_test_group))}
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
                disaggregated_results[sens_col][str(group)] = group_metrics
            nan_mask = X_test_df[sens_col].isnull()
            if nan_mask.any():
                 y_test_nan_group = y_test[nan_mask]
                 y_pred_nan_group = y_pred_test[nan_mask]
                 nan_group_metrics = {"sample_count": int(len(y_test_nan_group))}
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
                 disaggregated_results[sens_col]["__NaN__"] = nan_group_metrics
            logger.info(f"Disaggregated metrics calculation attempted for '{sens_col}'.")
        else:
            logger.error(f"Internal Error: Sensitive column '{sens_col}' expected but not found in X_test_df columns.")
    return disaggregated_results


# --- Baseline Training Function ---
def train_evaluate_baseline(
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str],
    feature_columns: List[str] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    # ...(rest of the function as provided before)...
    logger.info(f"--- Starting Baseline Training for '{filename}' ---")
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    try: df = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
    except Exception as e: logger.exception(f"Error reading CSV file: {e}"); raise ValueError(f"Error reading CSV file: {e}")
    if target_column not in df.columns: raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    if df[target_column].isnull().any(): logger.warning(f"Target column '{target_column}' contains {df[target_column].isnull().sum()} missing values. Dropping rows with missing target."); df.dropna(subset=[target_column], inplace=True)
    if df[target_column].dtype == 'object':
        unique_vals = df[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K'; negative_class_marker = '<=50K'
            if positive_class_marker in unique_vals and negative_class_marker in unique_vals: df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int); logger.info(f"Target column '{target_column}' cleaned to 0/1 (1 represents '{positive_class_marker}').")
            else: logger.warning(f"Expected markers '{positive_class_marker}'/'{negative_class_marker}' not found. Assuming '{unique_vals[0]}' as positive class (1)."); df[target_column] = df[target_column].apply(lambda x: 1 if x == unique_vals[0] else 0).astype(int)
        else: raise ValueError(f"Target column '{target_column}' is object type but does not have exactly 2 unique values after dropping NaNs. Found: {unique_vals}")
    elif pd.api.types.is_numeric_dtype(df[target_column]):
         unique_numeric_vals = df[target_column].unique()
         if set(unique_numeric_vals) == {0, 1}: logger.info(f"Target column '{target_column}' is already numeric binary (0/1)."); df[target_column] = df[target_column].astype(int)
         else: logger.warning(f"Target column '{target_column}' is numeric but not binary 0/1. Values found: {unique_numeric_vals}.")
    else: raise ValueError(f"Target column '{target_column}' has an unsupported data type: {df[target_column].dtype}")
    valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df.columns]
    if len(valid_sensitive_columns) != len(sensitive_attribute_columns): missing_sens_cols = set(sensitive_attribute_columns) - set(valid_sensitive_columns); logger.warning(f"Provided sensitive attributes not found in data: {missing_sens_cols}. Proceeding without them.")
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column and col not in valid_sensitive_columns]; logger.info(f"Using automatically selected features for training: {feature_columns}")
    else:
        missing_feats = [col for col in feature_columns if col not in df.columns];
        if missing_feats:
            raise ValueError(f"Provided feature columns not found in data: {missing_feats}"); logger.info(f"Using provided features for training: {feature_columns}")
    cols_to_keep = feature_columns + valid_sensitive_columns + [target_column]
    df_subset = df[[col for col in cols_to_keep if col in df.columns]].copy()
    try: X_train_full, X_test_full, y_train, y_test = train_test_split(df_subset.drop(columns=[target_column]), df_subset[target_column], test_size=test_size, random_state=random_state, stratify=df_subset[target_column]); logger.info(f"Data split into training ({len(X_train_full)} samples) and testing ({len(X_test_full)} samples).")
    except Exception as e: logger.exception(f"Error during train_test_split: {e}"); raise ValueError(f"Error during train_test_split: {e}")
    X_train_features_only = X_train_full[feature_columns]
    numerical_features_train = X_train_features_only.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only.select_dtypes(exclude=np.number).columns.tolist()
    logger.info(f"Identified numerical features for preprocessing: {numerical_features_train}"); logger.info(f"Identified categorical features for preprocessing: {categorical_features_train}")
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features_train), ('cat', categorical_transformer, categorical_features_train)], remainder='drop')
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    logger.info("Starting baseline model training...")
    try: pipeline.fit(X_train_full, y_train)
    except Exception as e: logger.exception("Error during pipeline fitting"); raise e
    logger.info("Baseline model training complete.")
    y_pred_test = pipeline.predict(X_test_full)
    results = {"metrics": {}, "model_info": {}}
    try: results["metrics"]["overall"] = {"accuracy": accuracy_score(y_test, y_pred_test), "precision": precision_score(y_test, y_pred_test, zero_division=0), "recall": recall_score(y_test, y_pred_test, zero_division=0), "f1": f1_score(y_test, y_pred_test, zero_division=0)}
    except Exception as e: results["metrics"]["overall"] = {"error": str(e)}
    logger.info(f"Baseline Model - Overall Test Metrics: {results['metrics']['overall']}")
    results["metrics"]["by_sensitive_group"] = _calculate_disaggregated_metrics(X_test_full, y_test, y_pred_test, valid_sensitive_columns)
    pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)
    metadata_filename = pipeline_path.replace('.joblib', '.meta.json')
    try: joblib.dump(pipeline, pipeline_path)
    except Exception as e: raise IOError(f"Could not save pipeline: {e}")
    metadata_to_save = {"pipeline_path": pipeline_path, "original_filename": filename, "features_used_in_training": feature_columns, "sensitive_attributes_present": valid_sensitive_columns, "target_column": target_column, "model_type": type(model).__name__, "training_random_state": random_state, "mitigation_applied": "None"}
    try:
        with open(metadata_filename, 'w') as f: json.dump(metadata_to_save, f, indent=4); logger.info(f"Baseline metadata saved to {metadata_filename}")
    except Exception as e:
        raise IOError(f"Could not save baseline metadata: {e}")
    results["model_info"] = {"pipeline_path": pipeline_path, "metadata_path": metadata_filename, "features_used_in_training": feature_columns, "sensitive_attributes_present": valid_sensitive_columns, "target_column": target_column, "model_type": type(model).__name__, "mitigation_applied": "None"}
    logger.info(f"--- Finished Baseline Training for '{filename}' ---")
    return results


# --- Reweighing Weight Calculation Function ---
def calculate_reweighing_weights(
    X_train_df: pd.DataFrame, y_train: pd.Series, sensitive_attribute: str
) -> np.ndarray:
    # ...(rest of the function as provided before)...
    logger.info(f"Calculating reweighing weights based on sensitive attribute: '{sensitive_attribute}'")
    if sensitive_attribute not in X_train_df.columns: raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in training data for weight calculation.")
    if len(X_train_df) != len(y_train): raise ValueError("Length mismatch between X_train_df and y_train for weight calculation.")
    combined_train = pd.DataFrame({'sensitive': X_train_df[sensitive_attribute], 'target': y_train})
    n_samples = len(combined_train)
    joint_prob = combined_train.groupby(['sensitive', 'target']).size().unstack(fill_value=0) / n_samples
    prob_a = combined_train['sensitive'].value_counts() / n_samples
    prob_y = combined_train['target'].value_counts() / n_samples
    weights_map = {}; epsilon = 1e-9
    for sens_val, group_joint_probs in joint_prob.iterrows():
        weights_map[sens_val] = {}
        for target_val, p_a_y in group_joint_probs.items():
            p_a = prob_a.get(sens_val, 0); p_y = prob_y.get(target_val, 0)
            if p_a_y < epsilon: weight = 1.0
            else: weight = (p_a * p_y) / p_a_y
            weights_map[sens_val][target_val] = weight
    sample_weights = np.ones(n_samples)
    for i in range(n_samples):
         sens_val = combined_train.iloc[i]['sensitive']; target_val = combined_train.iloc[i]['target']
         if sens_val in weights_map and target_val in weights_map[sens_val]: sample_weights[i] = weights_map[sens_val][target_val]
    logger.info(f"Reweighing weights calculated. Min: {np.min(sample_weights):.4f}, Max: {np.max(sample_weights):.4f}, Mean: {np.mean(sample_weights):.4f}")
    return sample_weights


# --- Reweighed Training Function ---
def train_evaluate_reweighed(
    filename: str, target_column: str, sensitive_attribute_columns: List[str],
    reweigh_attribute: str, feature_columns: List[str] = None,
    test_size: float = 0.3, random_state: int = 42
) -> Dict[str, Any]:
    # ...(rest of the function as provided before)...
    logger.info(f"--- Starting Reweighed Training for '{filename}' based on '{reweigh_attribute}' ---")
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    try: df = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
    except Exception as e: raise ValueError(f"Error reading CSV file: {e}")
    if target_column not in df.columns: raise ValueError(f"Target column '{target_column}' not found.")
    if df[target_column].isnull().any(): df.dropna(subset=[target_column], inplace=True)
    if df[target_column].dtype == 'object':
        unique_vals = df[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K'; negative_class_marker = '<=50K'
            if positive_class_marker in unique_vals and negative_class_marker in unique_vals: df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int)
            else: df[target_column] = df[target_column].apply(lambda x: 1 if x == unique_vals[0] else 0).astype(int)
        else: raise ValueError(f"Target column '{target_column}' not binary.")
    elif pd.api.types.is_numeric_dtype(df[target_column]):
        if set(df[target_column].unique()) == {0, 1}: df[target_column] = df[target_column].astype(int)
        else: logger.warning(f"Numeric target not binary 0/1.")
    else: raise ValueError(f"Unsupported target dtype: {df[target_column].dtype}")
    valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df.columns]
    if reweigh_attribute not in valid_sensitive_columns: raise ValueError(f"Attribute '{reweigh_attribute}' not found in data or sensitive list.")
    if feature_columns is None: feature_columns = [col for col in df.columns if col != target_column and col not in valid_sensitive_columns]
    logger.info(f"Using features for reweighed training: {feature_columns}")
    cols_to_keep = feature_columns + valid_sensitive_columns + [target_column]
    df_subset = df[[col for col in cols_to_keep if col in df.columns]].copy()
    try: X_train_full, X_test_full, y_train, y_test = train_test_split(df_subset.drop(columns=[target_column]), df_subset[target_column], test_size=test_size, random_state=random_state, stratify=df_subset[target_column])
    except Exception as e: raise ValueError(f"Error during train_test_split: {e}")
    try: sample_weights = calculate_reweighing_weights(X_train_full, y_train, reweigh_attribute)
    except Exception as e: logger.exception("Failed to calculate reweighing weights."); raise ValueError(f"Failed to calculate reweighing weights: {e}")
    X_train_features_only = X_train_full[feature_columns]
    numerical_features_train = X_train_features_only.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only.select_dtypes(exclude=np.number).columns.tolist()
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features_train), ('cat', categorical_transformer, categorical_features_train)], remainder='drop')
    model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight=None)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    logger.info("Starting reweighed model training...")
    try: fit_params = {'classifier__sample_weight': sample_weights}; pipeline.fit(X_train_full, y_train, **fit_params)
    except Exception as e: logger.exception("Error during reweighed pipeline fitting"); raise e
    logger.info("Reweighed model training complete.")
    y_pred_test = pipeline.predict(X_test_full)
    results = {"metrics": {}, "model_info": {}}
    try: results["metrics"]["overall"] = {"accuracy": accuracy_score(y_test, y_pred_test), "precision": precision_score(y_test, y_pred_test, zero_division=0), "recall": recall_score(y_test, y_pred_test, zero_division=0), "f1": f1_score(y_test, y_pred_test, zero_division=0)}
    except Exception as e: results["metrics"]["overall"] = {"error": str(e)}
    logger.info(f"Reweighed Model - Overall Test Metrics: {results['metrics']['overall']}")
    results["metrics"]["by_sensitive_group"] = _calculate_disaggregated_metrics(X_test_full, y_test, y_pred_test, valid_sensitive_columns)
    mitigation_suffix = f"_reweighed_{reweigh_attribute.replace(' ', '_').lower()}"
    pipeline_filename = f"{os.path.splitext(filename)[0]}{mitigation_suffix}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)
    metadata_filename = pipeline_path.replace('.joblib', '.meta.json')
    try: joblib.dump(pipeline, pipeline_path)
    except Exception as e: raise IOError(f"Could not save reweighed pipeline: {e}")
    metadata_to_save = {"pipeline_path": pipeline_path, "original_filename": filename, "features_used_in_training": feature_columns, "sensitive_attributes_present": valid_sensitive_columns, "target_column": target_column, "model_type": type(model).__name__, "training_random_state": random_state, "mitigation_applied": "Reweighing", "mitigation_target_attribute": reweigh_attribute, "mitigation_goal": "Demographic Parity (approx)"}
    try:
        with open(metadata_filename, 'w') as f: json.dump(metadata_to_save, f, indent=4); logger.info(f"Reweighed metadata saved to {metadata_filename}")
    except Exception as e:
        raise IOError(f"Could not save reweighed metadata: {e}")
    results["model_info"] = {"pipeline_path": pipeline_path, "metadata_path": metadata_filename, "mitigation_applied": metadata_to_save["mitigation_applied"], "mitigation_target_attribute": metadata_to_save["mitigation_target_attribute"]}
    logger.info(f"--- Finished Reweighed Training for '{filename}' based on '{reweigh_attribute}' ---")
    return results

# --- Oversampled Training Function ---
def train_evaluate_oversampled(
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str],
    feature_columns: List[str] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    # ...(rest of the function as provided before)...
    logger.info(f"--- Starting Random Oversampled Training for '{filename}' ---")
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    try: df = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
    except Exception as e: raise ValueError(f"Error reading CSV file: {e}")
    if target_column not in df.columns: raise ValueError(f"Target column '{target_column}' not found.")
    if df[target_column].isnull().any(): df.dropna(subset=[target_column], inplace=True)
    if df[target_column].dtype == 'object':
        unique_vals = df[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K'; negative_class_marker = '<=50K'
            if positive_class_marker in unique_vals and negative_class_marker in unique_vals: df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int)
            else: df[target_column] = df[target_column].apply(lambda x: 1 if x == unique_vals[0] else 0).astype(int)
        else: raise ValueError(f"Target column '{target_column}' not binary.")
    elif pd.api.types.is_numeric_dtype(df[target_column]):
        if set(df[target_column].unique()) == {0, 1}: df[target_column] = df[target_column].astype(int)
        else: logger.warning(f"Numeric target not binary 0/1.")
    else: raise ValueError(f"Unsupported target dtype: {df[target_column].dtype}")
    valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df.columns]
    if feature_columns is None: feature_columns = [col for col in df.columns if col != target_column and col not in valid_sensitive_columns]
    logger.info(f"Using features for oversampled training: {feature_columns}")
    cols_to_keep = feature_columns + valid_sensitive_columns + [target_column]
    df_subset = df[[col for col in cols_to_keep if col in df.columns]].copy()
    try: X_train_full, X_test_full, y_train, y_test = train_test_split( df_subset.drop(columns=[target_column]), df_subset[target_column], test_size=test_size, random_state=random_state, stratify=df_subset[target_column])
    except Exception as e: raise ValueError(f"Error during train_test_split: {e}")
    X_train_features_only_for_types = X_train_full[feature_columns]
    numerical_features_train = X_train_features_only_for_types.select_dtypes(include=np.number).columns.tolist()
    categorical_features_train = X_train_features_only_for_types.select_dtypes(exclude=np.number).columns.tolist()
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features_train), ('cat', categorical_transformer, categorical_features_train)], remainder='drop')
    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    oversampler_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor), ('oversampler', RandomOverSampler(random_state=random_state)), ('classifier', model)])
    logger.info("Starting oversampled model training...")
    try: oversampler_pipeline.fit(X_train_full, y_train)
    except Exception as e: logger.exception(f"Error during oversampled pipeline fitting: {e}"); raise e
    logger.info("Oversampled model training complete.")
    y_pred_test = oversampler_pipeline.predict(X_test_full)
    results = {"metrics": {}, "model_info": {}}
    try: results["metrics"]["overall"] = {"accuracy": accuracy_score(y_test, y_pred_test), "precision": precision_score(y_test, y_pred_test, zero_division=0), "recall": recall_score(y_test, y_pred_test, zero_division=0), "f1": f1_score(y_test, y_pred_test, zero_division=0)}
    except Exception as e: results["metrics"]["overall"] = {"error": str(e)}
    logger.info(f"Oversampled Model - Overall Test Metrics: {results['metrics']['overall']}")
    results["metrics"]["by_sensitive_group"] = _calculate_disaggregated_metrics(X_test_full, y_test, y_pred_test, valid_sensitive_columns)
    mitigation_suffix = "_oversampled"
    pipeline_filename = f"{os.path.splitext(filename)[0]}{mitigation_suffix}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)
    metadata_filename = pipeline_path.replace('.joblib', '.meta.json')
    try: joblib.dump(oversampler_pipeline, pipeline_path)
    except Exception as e: raise IOError(f"Could not save oversampled pipeline: {e}")
    metadata_to_save = {"pipeline_path": pipeline_path, "original_filename": filename, "features_used_in_training": feature_columns, "sensitive_attributes_present": valid_sensitive_columns, "target_column": target_column, "model_type": type(model).__name__, "training_random_state": random_state, "mitigation_applied": "RandomOversampling", "mitigation_details": "Applied RandomOverSampler from imblearn to balance target classes in training."}
    try: 
        with open(metadata_filename, 'w') as f: json.dump(metadata_to_save, f, indent=4); logger.info(f"Oversampled metadata saved to {metadata_filename}")
    except Exception as e:
        raise IOError(f"Could not save oversampled metadata: {e}")
    results["model_info"] = {"pipeline_path": pipeline_path, "metadata_path": metadata_filename, "mitigation_applied": metadata_to_save["mitigation_applied"]}
    logger.info(f"--- Finished Random Oversampled Training for '{filename}' ---")
    return results