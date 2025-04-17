import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate
from sklearn.metrics import accuracy_score, recall_score # Use sklearn metrics within MetricFrame
from typing import Dict, Any, List
import joblib
import os
import logging

logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
# MODEL_DIRECTORY is defined in models.py, assume accessible or redefine

def calculate_fairness_metrics(
    pipeline_path: str,
    filename: str, # Test data filename
    target_column: str,
    sensitive_attribute_columns: List[str]
) -> Dict[str, Any]:
    """
    Loads a trained pipeline and test data to calculate fairness metrics.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}")

    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        df_test = pd.read_csv(filepath, skipinitialspace=True)
        logger.info(f"Test data loaded from {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading pipeline or test data: {e}")

     # --- Clean Target Column (ensure consistency) ---
    if df_test[target_column].dtype == 'object':
        if df_test[target_column].nunique() == 2:
            unique_vals = df_test[target_column].unique()
            positive_class_marker = unique_vals[0]
            df_test[target_column] = df_test[target_column].apply(lambda x: 1 if x == positive_class_marker else 0)
        else:
            logger.warning(f"Target column '{target_column}' in test data is object type but not binary.")


    # --- Prepare Data ---
    # Assume feature columns are implicitly handled by the loaded pipeline's preprocessor
    # We need the sensitive features and the true labels (y_true) from the test set
    if not all(col in df_test.columns for col in sensitive_attribute_columns):
        raise ValueError("Sensitive attribute columns specified are not all in the test data.")

    X_test = df_test # Pipeline expects all columns it was trained on (or handles missing ones)
    y_true = df_test[target_column]

    if not pd.api.types.is_numeric_dtype(y_true):
         raise ValueError("Target column in test data is not numeric.")


    sensitive_features = df_test[sensitive_attribute_columns]

    # --- Get Predictions ---
    try:
        y_pred = pipeline.predict(X_test)
        logger.info("Predictions generated using the loaded pipeline.")
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}. Ensure test data has necessary columns.")


    # --- Calculate Metrics using MetricFrame ---
    # Define metrics to compute
    # Use functions directly, MetricFrame handles grouping
    metrics = {
        'accuracy': accuracy_score,
        'recall': recall_score,  # Assumes positive label is 1
        'selection_rate': selection_rate, # Rate at which model predicts positive class (1)
        'count': count
    }

    # Grouped on single sensitive feature for now, can extend later
    # For multiple sensitive features, MetricFrame can handle tuples, or run separately
    grouped_on_feature = sensitive_attribute_columns[0] # Start with the first one
    logger.info(f"Calculating metrics grouped by: {grouped_on_feature}")

    try:
        # control_features argument is deprecated/removed, use sensitive_features directly
        grouped_on = sensitive_features[grouped_on_feature]
        metric_frame = MetricFrame(metrics=metrics,
                                y_true=y_true,
                                y_pred=y_pred,
                                sensitive_features=grouped_on) # Use the specific column

        results = {"fairness_metrics": {}}
        results["fairness_metrics"]["by_group"] = metric_frame.by_group.to_dict()
        results["fairness_metrics"]["overall"] = metric_frame.overall.to_dict()

        # Calculate common fairness disparities
        results["fairness_metrics"]["disparities"] = {
             # Difference between the max and min group metric values
            "accuracy_difference": metric_frame.difference(metric='accuracy'),
            "recall_difference (Equal Opportunity Difference)": metric_frame.difference(metric='recall'),
            "selection_rate_difference (Demographic Parity Difference)": metric_frame.difference(metric='selection_rate'),
            # Ratio of min to max group metric values
            # "demographic_parity_ratio": metric_frame.ratio(metric='selection_rate'), # Requires >0 rates
        }
        logger.info("Fairness metrics calculated successfully.")

    except Exception as e:
        logger.error(f"Error during MetricFrame calculation: {e}")
        raise ValueError(f"Could not calculate fairness metrics: {e}")


    return results