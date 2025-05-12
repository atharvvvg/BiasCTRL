import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate, true_positive_rate, false_positive_rate
from sklearn.metrics import accuracy_score, recall_score, precision_score
from typing import Dict, Any, List
import joblib
import os
import logging

logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"
# MODEL_DIRECTORY = "models_cache"

def calculate_fairness_metrics(
    pipeline_path: str,
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str] 
) -> Dict[str, Any]:
    """
    Loads a trained pipeline and test data to calculate fairness metrics using MetricFrame.
    Calculates metrics for EACH sensitive attribute provided in the list.
    Returns a dictionary keyed by sensitive attribute name.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}")

    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        df_test = pd.read_csv(filepath, skipinitialspace=True, na_values='?')
        logger.info(f"Test data loaded from {filepath}")
    except Exception as e:
        logger.exception(f"Error loading pipeline or test data: {e}")
        raise ValueError(f"Error loading pipeline or test data: {e}")

    # Clean Target Column 
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test dataset.")
    if df_test[target_column].isnull().any():
        df_test.dropna(subset=[target_column], inplace=True)
    if df_test[target_column].dtype == 'object':
        unique_vals = df_test[target_column].unique()
        if len(unique_vals) == 2:
            positive_class_marker = '>50K'
            df_test[target_column] = df_test[target_column].apply(lambda x: 1 if x == positive_class_marker else 0).astype(int)
        else: raise ValueError(f"Target column '{target_column}' in test data is object type but not binary.")
    elif pd.api.types.is_numeric_dtype(df_test[target_column]):
         if not set(df_test[target_column].unique()).issubset({0, 1}): 
             logger.warning(f"Target column '{target_column}' in test data is numeric but not strictly binary 0/1.")
         df_test[target_column] = df_test[target_column].astype(int) 
    else: raise ValueError(f"Target column '{target_column}' in test data has an unsupported data type.")


    # Prepare Data for Prediction 
    X_eval = df_test
    y_true = df_test[target_column]

    try:
        y_pred = pipeline.predict(X_eval)
        logger.info("Predictions generated using the loaded pipeline.")
    except Exception as e:
        logger.exception(f"Error during prediction on test data: {e}")
        raise ValueError(f"Error during prediction: {e}.")

    # Define Base Metrics 
    base_metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'false_positive_rate': false_positive_rate,
        'true_positive_rate': true_positive_rate,
        'selection_rate': selection_rate,
        'count': count
    }

    # Initialize overall results dictionary 
    all_fairness_results = {}

    valid_sensitive_columns_from_input = [col for col in sensitive_attribute_columns if col in df_test.columns]
    if not valid_sensitive_columns_from_input:
        raise ValueError("None of the provided sensitive attribute columns found in the test data.")

    # Loop Through Each Provided Sensitive Attribute 
    for sens_attr_name in valid_sensitive_columns_from_input:
        logger.info(f"Calculating fairness metrics grouped by: '{sens_attr_name}'")
        sensitive_features_for_metricframe = df_test[sens_attr_name]

        try:
            metric_frame = MetricFrame(metrics=base_metrics,
                                       y_true=y_true,
                                       y_pred=y_pred,
                                       sensitive_features=sensitive_features_for_metricframe)

            current_attribute_results = {"fairness_metrics": {}}
            current_attribute_results["fairness_metrics"]["overall"] = metric_frame.overall.to_dict()
            current_attribute_results["fairness_metrics"]["by_group"] = metric_frame.by_group.to_dict()

            differences = metric_frame.difference(method='between_groups')
            ratios = metric_frame.ratio(method='between_groups')

            current_attribute_results["fairness_metrics"]["disparities"] = {
                "accuracy_difference": differences['accuracy'],
                "precision_difference": differences['precision'],
                "recall_difference (true_positive_rate_difference)": differences['recall'],
                "false_positive_rate_difference": differences['false_positive_rate'],
                "selection_rate_difference (demographic_parity_difference)": differences['selection_rate'],
                "accuracy_ratio": ratios.get('accuracy', None),
                "precision_ratio": ratios.get('precision', None),
                "recall_ratio": ratios.get('recall', None),
                "selection_rate_ratio (disparate_impact)": ratios.get('selection_rate', None),
            }
            current_attribute_results["fairness_metrics"]["standard_definitions"] = {
                "demographic_parity_difference": differences['selection_rate'],
                "demographic_parity_ratio": ratios.get('selection_rate', None),
                "equalized_odds_difference": max(differences['true_positive_rate'], differences['false_positive_rate']),
                "equal_opportunity_difference": differences['true_positive_rate'],
            }
            current_attribute_results["fairness_info"] = {
                 "grouped_on_sensitive_attribute": sens_attr_name
            }
            logger.info(f"Fairness metrics for '{sens_attr_name}' calculated successfully.")
            all_fairness_results[sens_attr_name] = current_attribute_results

        except Exception as e:
            logger.exception(f"Error during MetricFrame calculation for sensitive attribute '{sens_attr_name}': {e}")
            all_fairness_results[sens_attr_name] = {"error": f"Could not calculate fairness for {sens_attr_name}: {e}"}

    if not all_fairness_results: 
        return {"error": "No fairness results could be calculated."}

    return all_fairness_results 