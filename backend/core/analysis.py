import pandas as pd
from typing import Dict, Any, List
import os

UPLOAD_DIRECTORY = "uploads"

def load_and_analyze_data(
    filename: str,
    target_column: str,
    sensitive_attribute_columns: List[str]
) -> Dict[str, Any]:
    """
    Loads data from the uploaded file and performs basic analysis focused on
    target and sensitive attributes.
    """
    filepath = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, skipinitialspace=True) # skipinitialspace helps with datasets like Adult
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    results = {"filename": filename, "analysis": {}}

    # --- Basic Validation ---
    all_cols = [target_column] + sensitive_attribute_columns
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- Clean Target Column (Common for Adult Dataset) ---
    # Example: Convert '>50K'/'<=50K' to 1/0 if necessary
    if df[target_column].dtype == 'object':
         # Simple check, might need adjustment based on actual dataset values
        if df[target_column].nunique() == 2:
            unique_vals = df[target_column].unique()
            positive_class_marker = unique_vals[0] # Assume first value is positive class (e.g., '>50K')
            df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class_marker else 0)
            results['analysis']['target_cleaned'] = f"Assumed '{positive_class_marker}' as positive class (1)"
        else:
             results['analysis']['target_warning'] = "Target column is object type but not binary. Skipping cleaning."


    # --- Analysis ---
    results['analysis']['row_count'] = len(df)
    results['analysis']['column_names'] = df.columns.tolist()

    # Target distribution
    if pd.api.types.is_numeric_dtype(df[target_column]):
        results['analysis']['target_distribution'] = df[target_column].value_counts().to_dict()
    else:
         results['analysis']['target_distribution'] = "Target not numeric after cleaning attempt"

    # Sensitive attribute distributions
    sensitive_analysis = {}
    for col in sensitive_attribute_columns:
        dist = df[col].value_counts().to_dict()
        sensitive_analysis[col] = {
            "distribution": dist,
            "unique_values": len(dist)
        }
        # Target distribution per group
        if pd.api.types.is_numeric_dtype(df[target_column]):
             target_by_group = df.groupby(col)[target_column].value_counts(normalize=True).unstack().fillna(0)
             sensitive_analysis[col]["target_distribution_by_group"] = target_by_group.to_dict()

    results['analysis']['sensitive_attributes'] = sensitive_analysis

    return results