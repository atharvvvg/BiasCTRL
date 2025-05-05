# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import json # Keep json import if needed elsewhere

# Import core functions
from core.analysis import load_and_analyze_data
from core.models import train_evaluate_baseline, train_evaluate_reweighed
from core.fairness import calculate_fairness_metrics
from core.explainability import explain_baseline_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ethical AI Bias Mitigation Workbench API",
    description="Phase 1: Core backend for data analysis, model training, fairness & explainability.",
    version="0.1.2", # Incremented version
)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(MODEL_DIRECTORY, exist_ok=True)

# --- REMOVED State Management Cache ---
# analysis_results_cache = {}
# model_results_cache = {}

# --- Helper Function for Filename (Keep) ---
def get_safe_filename(filename: str) -> str:
    basename, ext = os.path.splitext(filename)
    safe_basename = "".join(c if c.isalnum() else "_" for c in basename)
    return f"{safe_basename}{ext}"

# --- Helper Function to Parse Sensitive Attributes (Keep) ---
def parse_sensitive_attributes(sensitive_attributes_str: str) -> List[str]:
    if not sensitive_attributes_str: return []
    sensitive_attribute_columns = [s.strip() for s in sensitive_attributes_str.split(',') if s.strip()]
    if not sensitive_attribute_columns:
        raise ValueError("Sensitive attribute columns string was provided but resulted in an empty list after parsing.")
    return sensitive_attribute_columns

# --- API Endpoints ---

@app.post("/upload", summary="Upload CSV dataset")
async def upload_dataset(file: UploadFile = File(...)):
    safe_filename = get_safe_filename(file.filename)
    file_location = os.path.join(UPLOAD_DIRECTORY, safe_filename)
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"File '{safe_filename}' uploaded successfully.")
        # --- REMOVED Cache Clearing ---
        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded successfully", "filename": safe_filename}
        )
    except Exception as e:
        logger.exception(f"Error uploading file {safe_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        if file and not file.file.closed:
             await file.close()

@app.post("/analyze", summary="Analyze uploaded dataset")
async def analyze_dataset_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)")
):
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Analyzing data for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}")
        # Call core function directly
        results = load_and_analyze_data(filename, target_column, parsed_sensitive_columns)
        # --- REMOVED Storing to Cache ---
        logger.info(f"Analysis complete for {filename}.")
        return JSONResponse(status_code=200, content=results) # Return results directly
    # ... (error handling as before) ...
    except ValueError as e:
        logger.error(f"Analysis failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/train_baseline", summary="Train and evaluate baseline model")
async def train_baseline_model_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)")
):
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Training baseline model for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}")
        # Call core function directly
        results = train_evaluate_baseline(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        # --- REMOVED Storing to Cache ---
        logger.info(f"Baseline model training complete for {filename}.")
        return JSONResponse(status_code=200, content=results) # Return results directly
    # ... (error handling as before) ...
    except ValueError as e:
        logger.error(f"Model training failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/calculate_fairness", summary="Calculate fairness metrics for trained model")
async def calculate_fairness_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)")
):
    # --- Construct expected pipeline path ---
    pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)

    # --- Check if pipeline file exists ---
    if not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at {pipeline_path}. Was the model trained successfully for '{filename}'?")

    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Calculating fairness for model from {filename}, pipeline: {pipeline_path}, sensitive: {parsed_sensitive_columns}")
        # Call core function directly, passing the pipeline path
        results = calculate_fairness_metrics(
            pipeline_path=pipeline_path,
            filename=filename, # Still need filename to load test data in core function
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        # --- REMOVED Storing to Cache ---
        logger.info(f"Fairness calculation complete for {filename}.")
        return JSONResponse(status_code=200, content=results) # Return results directly
    # ... (error handling as before) ...
    except ValueError as e:
        logger.error(f"Fairness calculation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e: # Catch file not found from core function too
        logger.error(f"Fairness calculation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during fairness calculation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/explain_model", summary="Explain baseline model using SHAP")
async def explain_model_endpoint(
    filename: str = Form(...), # Keep filename to find the corresponding pipeline
    # target_column: str = Form(...), # No longer needed, get from metadata
    # sensitive_attribute_columns: str = Form(...), # No longer needed, get from metadata
    n_samples: int = Form(100, description="Number of samples for SHAP background/explanation")
):
    # --- Construct expected pipeline path ---
    pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline_filename)

    # --- Check if pipeline file exists ---
    if not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at {pipeline_path}. Was the model trained successfully for '{filename}'?")

    try:
        # Note: Removed target_column and sensitive_attribute_columns from Form input
        # The core function now gets them from the metadata file.
        logger.info(f"Explaining model associated with {filename}, using pipeline: {pipeline_path}")
        # Call core function directly, passing only pipeline path and n_samples
        results = explain_baseline_model(
            pipeline_path=pipeline_path,
            n_samples=n_samples
        )
        # --- REMOVED Storing to Cache ---
        logger.info(f"Model explanation complete for {filename}.")
        return JSONResponse(status_code=200, content=results) # Return results directly
    # ... (error handling as before) ...
    except ValueError as e:
        logger.error(f"Model explanation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e: # Catch file not found from core function (e.g., metadata)
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model explanation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model explanation: {e}")


@app.post("/mitigate_reweigh", summary="Train model using Reweighing mitigation")
async def mitigate_reweigh_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)"),
    reweigh_attribute: str = Form(..., description="The specific sensitive attribute to base reweighing on (must be one of the columns above)")
):
    """
    Trains a new model using Reweighing mitigation based on the specified attribute.
    Returns the performance and fairness metrics of the *mitigated* model.
    """
    try:
        # Parse sensitive attributes
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        if reweigh_attribute not in parsed_sensitive_columns:
             # Check if the target attribute is actually in the list provided
              raise HTTPException(status_code=400, detail=f"Reweigh attribute '{reweigh_attribute}' must be one of the provided sensitive columns: {parsed_sensitive_columns}")

        logger.info(f"Starting Reweighing mitigation for: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}, reweigh_on: {reweigh_attribute}")

        # Call the new core function
        results = train_evaluate_reweighed(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns,
            reweigh_attribute=reweigh_attribute # Pass the specific attribute
        )

        logger.info(f"Reweighing mitigation training complete for {filename} based on {reweigh_attribute}.")
        # Return the results for the mitigated model
        return JSONResponse(status_code=200, content=results)

    except FileNotFoundError as e:
        logger.error(f"Reweighing failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Reweighing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during Reweighing mitigation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Root Endpoint (Keep) ---
@app.get("/", summary="Root endpoint", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Ethical AI Bias Mitigation Workbench API (Phase 1 - Metadata File Version)"}