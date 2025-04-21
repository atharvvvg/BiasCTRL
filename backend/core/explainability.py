# core/explainability.py
import pandas as pd
import shap
import joblib
import os
import logging
from typing import Dict, Any, List
import numpy as np
import json # <-- Import json

logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache" # Defined elsewhere

def explain_baseline_model(
    pipeline_path: str, # Expect the direct path to the pipeline file
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Loads a trained pipeline and its metadata, uses SHAP KernelExplainer
    to explain feature importance.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    # --- Load Metadata ---
    metadata_filename = pipeline_path.replace('.joblib', '.meta.json')
    if not os.path.exists(metadata_filename):
        raise FileNotFoundError(f"Metadata file not found: {metadata_filename}. Was the model trained successfully?")

    try:
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded successfully from {metadata_filename}")
    except Exception as e:
        logger.exception(f"Error loading metadata from {metadata_filename}: {e}")
        raise ValueError(f"Could not load metadata: {e}")

    # Extract needed info from metadata
    features_used_in_training = metadata.get('features_used_in_training')
    target_column = metadata.get('target_column')
    sensitive_attributes_present = metadata.get('sensitive_attributes_present', [])
    original_filename = metadata.get('original_filename')

    if not features_used_in_training:
        raise ValueError(f"Metadata file {metadata_filename} is missing 'features_used_in_training'.")
    if not target_column:
        raise ValueError(f"Metadata file {metadata_filename} is missing 'target_column'.")
    if not original_filename:
         raise ValueError(f"Metadata file {metadata_filename} is missing 'original_filename'.")

    # --- Load Pipeline and Data ---
    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")

        data_filepath = os.path.join(UPLOAD_DIRECTORY, original_filename)
        if not os.path.exists(data_filepath):
             raise FileNotFoundError(f"Original data file '{original_filename}' specified in metadata not found at {data_filepath}")
        df = pd.read_csv(data_filepath, skipinitialspace=True, na_values='?')
        logger.info(f"Data loaded from {data_filepath} for SHAP.")

    except Exception as e:
        logger.exception(f"Error loading pipeline or data: {e}")
        raise ValueError(f"Error loading pipeline or data: {e}")


    # --- Prepare Data for SHAP Transform ---
    # Reconstruct the list of columns the pipeline's preprocessor expects as input
    cols_to_keep_for_transform = features_used_in_training + sensitive_attributes_present
    cols_to_keep_for_transform = list(dict.fromkeys(cols_to_keep_for_transform)) # Remove duplicates

    missing_cols = [col for col in cols_to_keep_for_transform if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns required for transform {cols_to_keep_for_transform} missing from loaded data '{original_filename}': {missing_cols}")

    X_original_for_transform = df[cols_to_keep_for_transform].copy()
    logger.info(f"Prepared original data for transform. Shape: {X_original_for_transform.shape}")


    # --- Apply Preprocessing ---
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        X_processed = preprocessor.transform(X_original_for_transform)
        logger.info(f"Data preprocessing applied for SHAP. Shape: {X_processed.shape}")
        if hasattr(X_processed, "toarray"): # Should not happen with sparse_output=False, but check
             X_processed = X_processed.toarray()
    except Exception as e:
        logger.error(f"Error applying preprocessor. Input data shape: {X_original_for_transform.shape}")
        logger.error(f"Input data columns: {X_original_for_transform.columns.tolist()}")
        logger.error(f"Input data sample:\n{X_original_for_transform.head().to_string()}")
        logger.exception(f"Error applying preprocessor to data for SHAP: {e}")
        raise ValueError(f"Error applying preprocessor to data for SHAP: {e}")


    # --- Get Feature Names After Preprocessing ---
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        logger.info(f"Successfully retrieved {len(feature_names_out)} feature names after preprocessing.")
        if len(feature_names_out) != X_processed.shape[1]:
             logger.warning(f"Mismatch! Feature names count ({len(feature_names_out)}) != processed columns count ({X_processed.shape[1]}). Naming might be incorrect.")
             # Allow proceeding but with warning

    except Exception as e:
        logger.warning(f"Could not reliably call get_feature_names_out() on preprocessor: {e}. Using generic feature names.")
        feature_names_out = None


    # --- Create SHAP Explainer ---
    classifier = pipeline.named_steps['classifier']
    def predict_proba_pipeline(data_processed):
        """Wrapper function to get probabilities from the classifier step."""
        try:
            if hasattr(classifier, "predict_proba"):
                probas = classifier.predict_proba(data_processed)
                return np.array(probas)
            else: # Handle models without predict_proba
                preds = classifier.predict(data_processed)
                if len(preds.shape) == 1: # Binary
                    return np.array([1 - preds, preds]).T
                else: # Multiclass
                    return np.array(preds)
        except Exception as e:
             logger.error(f"Error inside predict_proba_pipeline wrapper: {e}")
             raise

    # --- Sample Background Data ---
    n_samples_actual = min(n_samples, X_processed.shape[0])
    if n_samples_actual <= 0:
         raise ValueError("n_samples must be positive.")
    if n_samples_actual < X_processed.shape[0]:
         try: # Prefer SHAP's sampling
              background_indices = shap.utils.sample(X_processed, n_samples_actual, random_state=42)
              background_data = X_processed[background_indices]
              logger.info(f"Using {n_samples_actual} samples (via shap.utils.sample) as background data.")
         except Exception as e: # Fallback if SHAP sample fails
              logger.warning(f"shap.utils.sample failed ({e}), using np.random.choice.")
              background_indices = np.random.choice(X_processed.shape[0], n_samples_actual, replace=False)
              background_data = X_processed[background_indices]
    else: # Use all data
         background_data = X_processed
         logger.info(f"Using all {X_processed.shape[0]} samples as background data.")

    # --- Initialize SHAP Explainer ---
    try:
        explainer = shap.KernelExplainer(predict_proba_pipeline, background_data)
        logger.info("SHAP KernelExplainer initialized.")
    except Exception as e:
         logger.exception(f"Error initializing SHAP KernelExplainer: {e}")
         raise ValueError(f"Failed to initialize SHAP Explainer: {e}")

    # --- Calculate SHAP Values ---
    logger.info(f"Calculating SHAP values for {background_data.shape[0]} samples...")
    try:
        shap_values = explainer.shap_values(background_data, nsamples='auto')
        logger.info("SHAP values calculated.")
        logger.info(f"Raw SHAP values type: {type(shap_values)}, "
                    f"Shape/Length: {getattr(shap_values, 'shape', len(shap_values) if isinstance(shap_values, list) else 'N/A')}")
    except Exception as e:
        logger.exception(f"Error calculating SHAP values: {e}")
        raise ValueError(f"Error during SHAP value calculation: {e}")


    # --- Process SHAP Value Structure ---
    try:
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Standard case for binary classification with KernelExplainer returning list
            shap_values_positive_class = shap_values[1]
            logger.info("Extracted SHAP values for the positive class (index 1) from list.")
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Case where explainer returns a 3D array (samples, features, classes)
            if shap_values.shape[-1] == 2: # Check if last dimension size is 2 (binary)
                 shap_values_positive_class = shap_values[:, :, 1] # Select slice for class 1
                 logger.info(f"Extracted SHAP values for the positive class (index 1) from 3D array. New shape: {shap_values_positive_class.shape}")
            else:
                 raise ValueError(f"Received 3D SHAP array, but last dimension size is {shap_values.shape[-1]}, expected 2 for binary classification.")
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
             # Case where explainer might directly return 2D array for positive class
             shap_values_positive_class = shap_values
             logger.info(f"SHAP values received as 2D numpy array with shape {shap_values.shape}. Assuming positive class.")
        else:
             shape_info = getattr(shap_values, 'shape', len(shap_values) if isinstance(shap_values, list) else 'N/A')
             logger.error(f"Unexpected SHAP values structure: type {type(shap_values)}, shape/length {shape_info}")
             raise ValueError(f"Cannot process unexpected SHAP values structure.")

    except Exception as e:
         logger.exception(f"Error processing SHAP value structure: {e}")
         raise ValueError(f"Could not process SHAP values structure: {e}")


    # --- Global Feature Importance (Mean Absolute SHAP) ---
    try:
        # Check if processed object is a 2D numpy array
        if not isinstance(shap_values_positive_class, np.ndarray):
             raise TypeError(f"After processing, expected shap_values_positive_class to be a numpy array, but got {type(shap_values_positive_class)}")
        if len(shap_values_positive_class.shape) != 2:
             raise ValueError(f"After processing, expected shap_values_positive_class to be a 2D array (samples, features), but got shape {shap_values_positive_class.shape}")

        # Calculate mean absolute value across samples (axis=0)
        mean_abs_shap = np.abs(shap_values_positive_class).mean(axis=0)
        logger.info(f"Calculated mean absolute SHAP values. Shape: {mean_abs_shap.shape}")

    except Exception as e:
        logger.exception(f"Error calculating mean absolute SHAP values: {e}")
        shape_info = getattr(shap_values_positive_class, 'shape', 'N/A')
        raise ValueError(f"Could not calculate mean absolute SHAP from processed object with shape {shape_info}: {e}")


    # --- Map importance back to feature names ---
    feature_importance = {}
    if feature_names_out is not None and len(feature_names_out) == len(mean_abs_shap):
        try:
            feature_importance = dict(zip(feature_names_out, mean_abs_shap))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            logger.info("Mapped SHAP values to feature names successfully.")
        except Exception as e:
            logger.exception(f"Error mapping feature names to SHAP values: {e}")
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(mean_abs_shap)}
            logger.warning("Mapping SHAP values to feature names failed; using generic names.")
    else:
         logger.warning(f"Feature name count ({len(feature_names_out) if feature_names_out else 'None'}) != SHAP values count ({len(mean_abs_shap)}). Using generic names.")
         feature_importance = {f"feature_{i}": imp for i, imp in enumerate(mean_abs_shap)}


    # --- Prepare Results ---
    results = {
        "explanation_info": {
            "method": "SHAP KernelExplainer",
            "n_background_samples": background_data.shape[0],
            "n_explained_samples": background_data.shape[0],
            "positive_class_index": 1 # Assuming binary, positive class is 1
        },
        # Convert numpy floats to standard Python floats for JSON serialization
        "global_feature_importance": {k: float(v) for k, v in feature_importance.items()}
    }
    return results