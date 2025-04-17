import pandas as pd
import shap
import joblib
import os
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
# MODEL_DIRECTORY defined elsewhere

def explain_baseline_model(
    pipeline_path: str,
    filename: str, # Data filename (used for background/sampling)
    target_column: str, # Needed to drop from explanation data
    sensitive_attribute_columns: List[str], # Needed to drop
    n_samples: int = 100 # Number of samples for SHAP background/explanation
) -> Dict[str, Any]:
    """
    Loads a trained pipeline and uses SHAP to explain feature importance.
    Focuses on global importance for Phase 1.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        df = pd.read_csv(filepath, skipinitialspace=True)
        logger.info(f"Data loaded from {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading pipeline or data: {e}")

    # --- Prepare Data for SHAP ---
    # SHAP needs the data in the format *after* preprocessing
    # We need to apply the preprocessor part of the pipeline

    # Define feature columns used by the preprocessor
    # This info should ideally be saved with the model or inferred from it
    try:
        # Attempt to get feature names after transformation
        preprocessor = pipeline.named_steps['preprocessor']
        # Get feature names out. This can be complex depending on transformers.
        # Simplified approach: Assume OneHotEncoder adds names, StandardScaler keeps originals
        transformers_info = preprocessor.transformers_
        feature_names_out = []

        for name, transformer, features in transformers_info:
            if hasattr(transformer, 'get_feature_names_out'):
                 # OneHotEncoder, etc.
                processed_names = transformer.get_feature_names_out(features)
                feature_names_out.extend(list(processed_names))
            elif name != 'remainder': # StandardScaler doesn't change names
                 feature_names_out.extend(features)

        # Handle remainder='passthrough' columns if any
        if hasattr(preprocessor, 'get_feature_names_out'):
             # Newer sklearn versions might simplify this
             try:
                 all_feature_names = preprocessor.get_feature_names_out()
                 feature_names_out = list(all_feature_names) # Overwrite with simpler method if available
             except Exception: # Fallback to manual reconstruction
                 if preprocessor.remainder == 'passthrough' and hasattr(preprocessor, 'feature_names_in_'):
                    input_features = preprocessor.feature_names_in_
                    processed_features = set()
                    for _, _, features in transformers_info:
                         processed_features.update(features)
                    remainder_features = [f for f in input_features if f not in processed_features]
                    feature_names_out.extend(remainder_features)

        logger.info(f"Attempting to infer feature names after preprocessing: {len(feature_names_out)} features.")
        if not feature_names_out:
             logger.warning("Could not reliably determine feature names after preprocessing. SHAP output might lack labels.")
             # Fallback: generate generic names? Might be misleading.


    except Exception as e:
        logger.warning(f"Could not extract feature names from preprocessor: {e}. SHAP output might lack proper labels.")
        feature_names_out = None # Indicate names are unknown


    # Prepare X by dropping target and sensitive columns before preprocessing
    cols_to_drop = [target_column] #+ sensitive_attribute_columns # Decide if sensitive cols were used in training features
    X = df.drop(columns=cols_to_drop, errors='ignore')

    # Apply the preprocessing step ONLY
    try:
        X_processed = preprocessor.transform(X)
        logger.info("Data preprocessing applied for SHAP.")
         # If X_processed is sparse, convert to dense for some SHAP explainers
        if hasattr(X_processed, "toarray"):
             X_processed = X_processed.toarray()
    except Exception as e:
        raise ValueError(f"Error applying preprocessor to data for SHAP: {e}")

    # --- Create SHAP Explainer ---
    # We need the prediction function of the *classifier* part of the pipeline
    classifier = pipeline.named_steps['classifier']

    # Use KernelExplainer for model-agnostic method, works with pipelines
    # It needs a function that takes the *preprocessed* data and returns predictions (usually probabilities)
    def predict_proba_pipeline(data):
        # Input 'data' here is already preprocessed
        if hasattr(classifier, "predict_proba"):
            return classifier.predict_proba(data)
        else:
            # Handle classifiers without predict_proba (like SVC without probability=True)
            # Return pseudo-probabilities or just predictions if necessary
            preds = classifier.predict(data)
            # Convert to shape (n_samples, n_classes) if binary
            return np.vstack([1 - preds, preds]).T if len(preds.shape) == 1 else preds


    # Sample background data (important for KernelExplainer performance and accuracy)
    if n_samples > X_processed.shape[0]:
        n_samples = X_processed.shape[0] # Don't sample more than available

    if n_samples > 0:
         background_data = shap.sample(X_processed, n_samples)
         logger.info(f"Using {n_samples} samples as background data for SHAP KernelExplainer.")
    else:
         logger.warning("n_samples is 0, SHAP explainer might not work as expected.")
         background_data = X_processed # Use all data if sampling is 0, can be slow


    explainer = shap.KernelExplainer(predict_proba_pipeline, background_data)
    logger.info("SHAP KernelExplainer initialized.")

    # --- Calculate SHAP Values ---
    # Explain a subset of the data (or all if small)
    # Using the same background data sample for explanation for simplicity here
    logger.info(f"Calculating SHAP values for {n_samples} samples...")
    shap_values = explainer.shap_values(background_data) # Explain the same sample used for background
    logger.info("SHAP values calculated.")

    # For binary classification, shap_values might be a list [shap_for_class_0, shap_for_class_1]
    # We usually care about the positive class (class 1)
    shap_values_positive_class = shap_values[1] if isinstance(shap_values, list) else shap_values


    # --- Global Feature Importance (Mean Absolute SHAP) ---
    mean_abs_shap = np.abs(shap_values_positive_class).mean(axis=0)

    # Try to map importance back to feature names
    if feature_names_out and len(feature_names_out) == len(mean_abs_shap):
         feature_importance = dict(zip(feature_names_out, mean_abs_shap))
         # Sort by importance
         feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    else:
         feature_importance = {f"feature_{i}": imp for i, imp in enumerate(mean_abs_shap)}
         logger.warning("Mapping SHAP values to feature names failed; using generic names.")


    results = {
        "explanation_info": {
            "method": "SHAP KernelExplainer",
            "n_samples_explained": n_samples,
            "positive_class_index": 1 # Assuming binary, positive class is 1
        },
        "global_feature_importance": feature_importance
        # Later: Add local explanations (SHAP for individual instances)
        # Later: Add SHAP summary plots (requires matplotlib potentially saved to file)
    }

    return results