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
    version="0.1.0",
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
        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded successfully", "filename": safe_filename}
        )
    except Exception as e:
        logger.error(f"Error uploading file {safe_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        await file.close() # Ensure file handle is closed

@app.post("/analyze", summary="Analyze uploaded dataset")
async def analyze_dataset(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: List[str] = Form(...) # Input as multiple form fields with same name
):
    """
    Performs initial data analysis on the specified uploaded CSV file.
    Identifies target and sensitive attributes.
    """
    try:
        logger.info(f"Analyzing data for file: {filename}, target: {target_column}, sensitive: {sensitive_attribute_columns}")
        results = load_and_analyze_data(filename, target_column, sensitive_attribute_columns)
        # Store results in cache (simple approach)
        analysis_results_cache[filename] = results
        logger.info(f"Analysis complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except FileNotFoundError as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/train_baseline", summary="Train and evaluate baseline model")
async def train_baseline_model_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: List[str] = Form(...)
    # Optional: Add feature_columns: List[str] = Form(None) if needed
):
    """
    Trains a baseline classification model on the specified data,
    evaluates performance (overall and per group), and saves the model.
    """
    try:
        logger.info(f"Training baseline model for file: {filename}, target: {target_column}, sensitive: {sensitive_attribute_columns}")
        results = train_evaluate_baseline(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=sensitive_attribute_columns
            # Pass other parameters like feature_columns if added
        )
        # Store model path and other info for later use
        model_results_cache[filename] = results # Overwrites previous if same filename used
        logger.info(f"Baseline model training complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except FileNotFoundError as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/calculate_fairness", summary="Calculate fairness metrics for trained model")
async def calculate_fairness_endpoint(
    filename: str = Form(...), # Filename of the *original* data used for training/test split reference
    target_column: str = Form(...),
    sensitive_attribute_columns: List[str] = Form(...)
):
    """
    Calculates fairness metrics (like Demographic Parity, Equalized Odds difference)
    for the previously trained baseline model associated with the filename.
    Requires the model to have been trained via /train_baseline first.
    """
    if filename not in model_results_cache:
         raise HTTPException(status_code=404, detail=f"No trained model found for filename: {filename}. Please run /train_baseline first.")

    model_info = model_results_cache[filename]['model_info']
    pipeline_path = model_info.get('pipeline_path')

    if not pipeline_path or not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline path not found or invalid for {filename}.")

    try:
        logger.info(f"Calculating fairness for model from {filename}, pipeline: {pipeline_path}")
        # Use the same filename for test data source - fairness function re-reads it
        results = calculate_fairness_metrics(
            pipeline_path=pipeline_path,
            filename=filename, # Use the original filename to load test data within the function
            target_column=target_column,
            sensitive_attribute_columns=sensitive_attribute_columns
        )
        # Optionally merge fairness results into the cache
        model_results_cache[filename]['fairness'] = results
        logger.info(f"Fairness calculation complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except FileNotFoundError as e:
        logger.error(f"Fairness calculation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Fairness calculation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during fairness calculation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/explain_model", summary="Explain baseline model using SHAP")
async def explain_model_endpoint(
    filename: str = Form(...), # Filename of the original data
    target_column: str = Form(...),
    sensitive_attribute_columns: List[str] = Form(...),
    n_samples: int = Form(100) # Number of samples for SHAP background/explanation
):
    """
    Generates SHAP explanations (global feature importance) for the
    previously trained baseline model associated with the filename.
    """
    if filename not in model_results_cache:
         raise HTTPException(status_code=404, detail=f"No trained model found for filename: {filename}. Please run /train_baseline first.")

    model_info = model_results_cache[filename]['model_info']
    pipeline_path = model_info.get('pipeline_path')

    if not pipeline_path or not os.path.exists(pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline path not found or invalid for {filename}.")

    try:
        logger.info(f"Explaining model from {filename}, pipeline: {pipeline_path}")
        results = explain_baseline_model(
            pipeline_path=pipeline_path,
            filename=filename, # Original data file used for sampling/background
            target_column=target_column,
            sensitive_attribute_columns=sensitive_attribute_columns,
            n_samples=n_samples
        )
        # Optionally merge explanation results into the cache
        model_results_cache[filename]['explanation'] = results
        logger.info(f"Model explanation complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except FileNotFoundError as e:
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model explanation: {e}")
        # SHAP can sometimes have cryptic errors, provide generic message
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