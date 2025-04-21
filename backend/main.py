from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import shutil
import logging

# Import core functions
from core.analysis import load_and_analyze_data
from core.models import train_evaluate_baseline
from core.fairness import calculate_fairness_metrics
from core.explainability import explain_baseline_model

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ethical AI Bias Mitigation Workbench API",
    description="Phase 1: Core backend for data analysis, model training, fairness & explainability.",
    version="0.1.1", # Incremented version
)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache" # Ensure consistency

# Create directories if they don't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(MODEL_DIRECTORY, exist_ok=True)

# --- State Management (Simple In-Memory for Phase 1) ---
# WARNING: This is NOT suitable for production. State will be lost on restart.
# A proper solution would involve databases or more robust file tracking.
analysis_results_cache = {}
model_results_cache = {}

# --- Helper Function for Filename ---
def get_safe_filename(filename: str) -> str:
    # Basic sanitization, replace spaces, keep extension
    basename, ext = os.path.splitext(filename)
    safe_basename = "".join(c if c.isalnum() else "_" for c in basename)
    return f"{safe_basename}{ext}"

# --- Helper Function to Parse Sensitive Attributes String ---
def parse_sensitive_attributes(sensitive_attributes_str: str) -> List[str]:
    """Parses comma-separated string into a list of stripped strings."""
    if not sensitive_attributes_str: # Handle empty string case
        return []
    sensitive_attribute_columns = [s.strip() for s in sensitive_attributes_str.split(',') if s.strip()]
    if not sensitive_attribute_columns:
        raise ValueError("Sensitive attribute columns string was provided but resulted in an empty list after parsing.")
    return sensitive_attribute_columns

# --- API Endpoints ---

@app.post("/upload", summary="Upload CSV dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a CSV file to the server for analysis.
    """
    safe_filename = get_safe_filename(file.filename)
    file_location = os.path.join(UPLOAD_DIRECTORY, safe_filename)
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"File '{safe_filename}' uploaded successfully.")
        # Clear any old cache related to this filename on new upload
        if safe_filename in analysis_results_cache:
            del analysis_results_cache[safe_filename]
        if safe_filename in model_results_cache:
            del model_results_cache[safe_filename]
        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded successfully", "filename": safe_filename}
        )
    except Exception as e:
        logger.exception(f"Error uploading file {safe_filename}: {e}") # Use logger.exception for stacktrace
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        if file and not file.file.closed: # Check if file exists and is not closed before closing
             await file.close() # Ensure file handle is closed


@app.post("/analyze", summary="Analyze uploaded dataset")
@app.post("/analyze", summary="Analyze uploaded dataset")
async def analyze_dataset_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    # --- CHANGE THIS LINE ---
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)") # Renamed parameter back, kept type str
):
    """
    Performs initial data analysis... Input sensitive attributes as a comma-separated string.
    """
    try:
        # --- Parse the input string using the helper ---
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns) # Parse the input variable

        logger.info(f"Analyzing data for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}") # Use the parsed list
        results = load_and_analyze_data(filename, target_column, parsed_sensitive_columns) # Pass the parsed list
        analysis_results_cache[filename] = results
        logger.info(f"Analysis complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    # ... (rest of the function and error handling remains the same) ...
    except ValueError as e: # Catch parsing errors too
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
    """
    Trains a baseline classification model... Input sensitive attributes as a comma-separated string.
    """
    try:
        # --- Parse the input string using the helper ---
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)

        logger.info(f"Training baseline model for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}")
        results = train_evaluate_baseline(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns # Pass the parsed list
        )
        model_results_cache[filename] = results
        logger.info(f"Baseline model training complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    # ... (rest of the function and error handling remains the same) ...
    except ValueError as e: # Catch parsing errors too
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
    filename: str = Form(...), # Filename of the *original* data used for training/test split reference
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)")
):
    """
    Calculates fairness metrics... Input sensitive attributes as a comma-separated string...
    """
    if filename not in model_results_cache:
         raise HTTPException(status_code=404, detail=f"No trained model found associated with filename: '{filename}'. Please run /train_baseline first.")
    # ... (rest of the checks for model_info and pipeline_path) ...
    model_info = model_results_cache[filename].get('model_info', {})
    pipeline_path = model_info.get('pipeline_path')
    if not pipeline_path or not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline path not found or file does not exist for {filename} at {pipeline_path}. Was training successful?")

    try:
        # --- Parse the input string using the helper ---
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)

        # --- Optional: Check against saved sensitive columns (as before) ---
        saved_sensitive_cols = model_info.get('sensitive_attributes_present', [])
        if set(parsed_sensitive_columns) != set(saved_sensitive_cols):
            logger.warning(f"Sensitive attributes provided {parsed_sensitive_columns} do not exactly match those used during training {saved_sensitive_cols} for file {filename}.")

        logger.info(f"Calculating fairness for model from {filename}, pipeline: {pipeline_path}, sensitive: {parsed_sensitive_columns}")
        results = calculate_fairness_metrics(
            pipeline_path=pipeline_path,
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns # Pass the parsed list
        )
        model_results_cache[filename]['fairness'] = results
        logger.info(f"Fairness calculation complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    # ... (rest of the function and error handling remains the same) ...
    except ValueError as e: # Catch parsing errors too
        logger.error(f"Fairness calculation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Fairness calculation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during fairness calculation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/explain_model", summary="Explain baseline model using SHAP")
async def explain_model_endpoint(
    filename: str = Form(...), # Filename of the original data
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)"),
    n_samples: int = Form(100, description="Number of samples for SHAP background/explanation")
):
    """
    Generates SHAP explanations... Input sensitive attributes as a comma-separated string.
    """
    if filename not in model_results_cache:
         raise HTTPException(status_code=404, detail=f"No trained model found associated with filename: '{filename}'. Please run /train_baseline first.")
    # ... (rest of the checks for model_info and pipeline_path) ...
    model_info = model_results_cache[filename].get('model_info', {})
    pipeline_path = model_info.get('pipeline_path')
    if not pipeline_path or not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline path not found or file does not exist for {filename} at {pipeline_path}. Was training successful?")

    try:
        # --- Parse the input string using the helper ---
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)

        logger.info(f"Explaining model from {filename}, pipeline: {pipeline_path}, sensitive: {parsed_sensitive_columns}")
        results = explain_baseline_model(
            pipeline_path=pipeline_path,
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns, # Pass the parsed list
            n_samples=n_samples
        )
        model_results_cache[filename]['explanation'] = results
        logger.info(f"Model explanation complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    # ... (rest of the function and error handling remains the same) ...
    except ValueError as e: # Catch parsing errors too
        logger.error(f"Model explanation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model explanation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model explanation: {e}")

# --- Root Endpoint ---
@app.get("/", summary="Root endpoint", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Ethical AI Bias Mitigation Workbench API (Phase 1)"}


# --- How to Run ---
# Save this file as main.py in the root of your bias_workbench directory.
# Open your terminal in the bias_workbench directory.
# Make sure your virtual environment is activated.
# Run the server: uvicorn main:app --reload
# Access the API docs at http://127.0.0.1:8000/docs