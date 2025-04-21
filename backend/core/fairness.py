import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate, make_derived_metric, true_positive_rate, false_positive_rate # Import more base metrics if needed for specific disparities
from sklearn.metrics import accuracy_score, recall_score, precision_score # Use sklearn metrics within MetricFrame
from typing import Dict, Any, List
import joblib
import os
import logging

logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache" # Assuming access

def calculate_fairness_metrics(
    pipeline_path: str,
    filename: str, # Test data filename (used to load data for evaluation)
    target_column: str,
    sensitive_attribute_columns: List[str]
) -> Dict[str, Any]:
    """
    Loads a trained pipeline and test data to calculate fairness metrics using MetricFrame.
    Handles potential '?' missing values in test data.
    Calculates metrics based on the FIRST sensitive attribute provided.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}")

    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        # Load test data, handling '?' same way as training
        df_test = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
        logger.info(f"Test data loaded from {filepath}")
    except Exception as e:
        logger.exception(f"Error loading pipeline or test data: {e}")
        raise ValueError(f"Error loading pipeline or test data: {e}")

    # --- Clean Target Column (ensure consistency) ---
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test dataset.")
    # Handle potential missing values in target before cleaning
    if df_test[target_column].isnull().any():
        logger.warning(f"Target column '{target_column}' in test data contains {df_test[target_column].isnull().sum()} missing values. Dropping rows with missing target.")
        df_test.dropna(subset=[target_column], inplace=True)

    if df_test[target_column].dtype == 'object':
        unique_vals = df_test[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K' # Explicitly define
            df_test[target_column] = df_test[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int)
            # logger.info("Test data target column cleaned.") # Less verbose logging
        else:
            logger.error(f"Target column '{target_column}' in test data is object type but not binary. Found: {unique_vals}")
            raise ValueError(f"Target column '{target_column}' in test data is object type but not binary.")
    elif pd.api.types.is_numeric_dtype(df_test[target_column]):
         if set(df_test[target_column].unique()) == {0, 1}:
             df_test[target_column] = df_test[target_column].astype(int)
         else:
            logger.warning(f"Target column '{target_column}' in test data is numeric but not binary 0/1.")
    else:
         raise ValueError(f"Target column '{target_column}' in test data has an unsupported data type: {df_test[target_column].dtype}")


    # --- Prepare Data for Prediction and Fairness Calculation ---
    # Check sensitive attributes exist
    valid_sensitive_columns = [col for col in sensitive_attribute_columns if col in df_test.columns]
    if len(valid_sensitive_columns) != len(sensitive_attribute_columns):
        missing_sens_cols = set(sensitive_attribute_columns) - set(valid_sensitive_columns)
        logger.warning(f"Provided sensitive attributes not found in test data: {missing_sens_cols}. Proceeding without them for fairness calculation.")
        if not valid_sensitive_columns: # Cannot proceed if NO valid sensitive columns
             raise ValueError("No valid sensitive attribute columns found in the test data to calculate fairness.")

    # X_test needs all columns the pipeline expects (features + potentially sensitive if passthrough was used, though we dropped them now)
    # The loaded pipeline's preprocessor knows which columns to use for transforming features
    X_eval = df_test # Pass the whole dataframe subset relevant for evaluation
    y_true = df_test[target_column]

    # Get sensitive features directly from the test dataframe for MetricFrame
    sensitive_features_eval = df_test[valid_sensitive_columns]

    # --- Get Predictions ---
    try:
        y_pred = pipeline.predict(X_eval)
        logger.info("Predictions generated using the loaded pipeline.")
    except Exception as e:
        logger.exception(f"Error during prediction on test data: {e}")
        raise ValueError(f"Error during prediction: {e}. Ensure test data has necessary columns for the pipeline.")


    # --- Calculate Metrics using MetricFrame ---
    # Define base metrics required for common fairness calculations
    # Use fairlearn's built-in metrics where possible for consistency
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'false_positive_rate': false_positive_rate,
        'true_positive_rate': true_positive_rate, # recall is same as TPR
        'selection_rate': selection_rate,
        'count': count
    }

    # --- IMPORTANT: Group on ONE sensitive feature at a time ---
    # MetricFrame calculates metrics *disaggregated* by groups in sensitive_features.
    # Disparity metrics (difference, ratio) are then calculated *across these groups*.
    # Let's calculate for the FIRST valid sensitive attribute provided.
    # A more advanced version could loop through all or handle combined attributes.

    grouped_on_feature_name = valid_sensitive_columns[0] # Use the first valid one
    grouped_on_values = sensitive_features_eval[grouped_on_feature_name]
    logger.info(f"Calculating fairness metrics grouped by: '{grouped_on_feature_name}'")

    try:
        # Initialize MetricFrame
        metric_frame = MetricFrame(metrics=metrics,
                                   y_true=y_true,
                                   y_pred=y_pred,
                                   sensitive_features=grouped_on_values) # Pass the specific Series

        results = {"fairness_metrics": {}}
        # Get overall metrics (averaged over all data)
        results["fairness_metrics"]["overall"] = metric_frame.overall.to_dict()
        # Get metrics disaggregated by group
        results["fairness_metrics"]["by_group"] = metric_frame.by_group.to_dict()

        # --- MODIFICATION: Calculate common fairness disparities ---
        # The .difference() and .ratio() methods are called *after* calculation,
        # and operate on the results stored within the metric_frame object for ALL metrics.
        # We extract the specific disparity we need after calling the method.

        differences = metric_frame.difference(method='between_groups') # Max difference between any two groups
        ratios = metric_frame.ratio(method='between_groups') # Min ratio between any two groups

        results["fairness_metrics"]["disparities"] = {
            # Extract difference for specific base metrics
            "accuracy_difference": differences['accuracy'],
            "precision_difference": differences['precision'],
            "recall_difference (true_positive_rate_difference)": differences['recall'], # or differences['true_positive_rate']
            "false_positive_rate_difference": differences['false_positive_rate'],
            "selection_rate_difference (demographic_parity_difference)": differences['selection_rate'],

            # Extract ratios for specific base metrics
             # Handle potential division by zero or NaN results from ratio calculation if needed
            "accuracy_ratio": ratios.get('accuracy', None), # Use .get for safety
            "precision_ratio": ratios.get('precision', None),
            "recall_ratio": ratios.get('recall', None),
            "selection_rate_ratio (disparate_impact)": ratios.get('selection_rate', None),
        }

        # --- Common Named Fairness Metrics ---
        # These combine differences of base metrics
        results["fairness_metrics"]["standard_definitions"] = {
            "demographic_parity_difference": differences['selection_rate'],
            "demographic_parity_ratio": ratios.get('selection_rate', None),
            "equalized_odds_difference": max(differences['true_positive_rate'], differences['false_positive_rate']), # Max diff in TPR and FPR
            "equal_opportunity_difference": differences['true_positive_rate'], # Difference in TPR (Recall)
        }


        logger.info("Fairness metrics and disparities calculated successfully.")

    except Exception as e:
        logger.exception(f"Error during MetricFrame calculation or disparity processing: {e}") # Use logger.exception
        raise ValueError(f"Could not calculate fairness metrics: {e}")


    results["fairness_info"] = {
         "grouped_on_sensitive_attribute": grouped_on_feature_name
    }

    return results