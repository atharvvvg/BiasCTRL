# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional # Ensure Optional is imported
import os
import shutil
import logging
import json

# Import core functions
from core.analysis import load_and_analyze_data
from core.models import train_evaluate_baseline, train_evaluate_reweighed
from core.fairness import calculate_fairness_metrics
from core.explainability import explain_baseline_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ethical AI Bias Mitigation Workbench API",
    description="Phase 1: Core backend with flexible fairness/explainability.",
    version="0.1.3", # Incremented version
)

UPLOAD_DIRECTORY = "uploads"
MODEL_DIRECTORY = "models_cache"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(MODEL_DIRECTORY, exist_ok=True)

def get_safe_filename(filename: str) -> str:
    basename, ext = os.path.splitext(filename)
    safe_basename = "".join(c if c.isalnum() else "_" for c in basename)
    return f"{safe_basename}{ext}"

def parse_sensitive_attributes(sensitive_attributes_str: str) -> List[str]:
    if not sensitive_attributes_str: return []
    sensitive_attribute_columns = [s.strip() for s in sensitive_attributes_str.split(',') if s.strip()]
    if not sensitive_attribute_columns:
        raise ValueError("Sensitive attribute columns string was provided but resulted in an empty list after parsing.")
    return sensitive_attribute_columns

# --- API Endpoints ---

@app.post("/upload", summary="Upload CSV dataset")
async def upload_dataset(file: UploadFile = File(...)):
    # ... (no changes here) ...
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
    # ... (no changes here) ...
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Analyzing data for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}")
        results = load_and_analyze_data(filename, target_column, parsed_sensitive_columns)
        logger.info(f"Analysis complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
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
    # ... (no changes here) ...
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Training baseline model for file: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}")
        results = train_evaluate_baseline(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        logger.info(f"Baseline model training complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except ValueError as e:
        logger.error(f"Model training failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/mitigate_reweigh", summary="Train model using Reweighing mitigation")
async def mitigate_reweigh_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)"),
    reweigh_attribute: str = Form(..., description="The specific sensitive attribute to base reweighing on (must be one of the columns above)")
):
    # ... (no changes here) ...
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        if reweigh_attribute not in parsed_sensitive_columns:
              raise HTTPException(status_code=400, detail=f"Reweigh attribute '{reweigh_attribute}' must be one of the provided sensitive columns: {parsed_sensitive_columns}")
        logger.info(f"Starting Reweighing mitigation for: {filename}, target: {target_column}, sensitive: {parsed_sensitive_columns}, reweigh_on: {reweigh_attribute}")
        results = train_evaluate_reweighed(
            filename=filename, target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns,
            reweigh_attribute=reweigh_attribute
        )
        logger.info(f"Reweighing mitigation training complete for {filename} based on {reweigh_attribute}.")
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


# --- MODIFIED /calculate_fairness Endpoint ---
@app.post("/calculate_fairness", summary="Calculate fairness metrics for a trained model")
async def calculate_fairness_endpoint(
    filename: str = Form(..., description="Filename of the *original* data used for evaluation (e.g., adult.csv)"),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender)"),
    pipeline_path_param: Optional[str] = Form(None, alias="pipeline_path", description="Optional: Full path to a specific .joblib pipeline file. If not provided, uses baseline path from 'filename'.") # Added optional parameter
):
    """
    Calculates fairness metrics for a trained model.
    If `pipeline_path` is provided, it uses that specific model.
    Otherwise, it assumes the baseline model associated with `filename`.
    """
    actual_pipeline_path: str

    if pipeline_path_param:
        # Use the provided path directly
        if not os.path.isabs(pipeline_path_param) and not pipeline_path_param.startswith(MODEL_DIRECTORY):
            # Assume it's a relative path within MODEL_DIRECTORY if not absolute
            actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(pipeline_path_param))
            logger.info(f"Using provided relative pipeline path, resolved to: {actual_pipeline_path}")
        else:
            actual_pipeline_path = pipeline_path_param
            logger.info(f"Using provided absolute pipeline path: {actual_pipeline_path}")
    else:
        # Construct baseline pipeline path from filename
        base_pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
        actual_pipeline_path = os.path.join(MODEL_DIRECTORY, base_pipeline_filename)
        logger.info(f"No explicit pipeline_path provided, using baseline path: {actual_pipeline_path}")

    # --- Check if the determined pipeline file exists ---
    if not os.path.exists(actual_pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at '{actual_pipeline_path}'. Was the model trained successfully?")

    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Calculating fairness for model using pipeline: '{actual_pipeline_path}', data from '{filename}', sensitive: {parsed_sensitive_columns}")

        results = calculate_fairness_metrics(
            pipeline_path=actual_pipeline_path, # Use the determined path
            filename=filename, # Original data filename to load test data
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        logger.info(f"Fairness calculation complete for pipeline '{actual_pipeline_path}'.")
        return JSONResponse(status_code=200, content=results)
    except ValueError as e:
        logger.error(f"Fairness calculation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Fairness calculation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during fairness calculation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- MODIFIED /explain_model Endpoint (similar logic for pipeline_path) ---
@app.post("/explain_model", summary="Explain a trained model using SHAP")
async def explain_model_endpoint(
    filename: str = Form(..., description="Filename of the *original* data (needed if metadata is used for data loading, e.g., adult.csv)"),
    pipeline_path_param: Optional[str] = Form(None, alias="pipeline_path", description="Optional: Full path to a specific .joblib pipeline file. If not provided, uses baseline path from 'filename'."),
    n_samples: int = Form(100, description="Number of samples for SHAP background/explanation")
):
    """
    Generates SHAP explanations for a trained model.
    If `pipeline_path` is provided, it uses that specific model.
    Otherwise, it assumes the baseline model associated with `filename`.
    """
    actual_pipeline_path: str

    if pipeline_path_param:
        if not os.path.isabs(pipeline_path_param) and not pipeline_path_param.startswith(MODEL_DIRECTORY):
            actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(pipeline_path_param))
        else:
            actual_pipeline_path = pipeline_path_param
        logger.info(f"Using explicit pipeline path for explanation: {actual_pipeline_path}")
    else:
        base_pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
        actual_pipeline_path = os.path.join(MODEL_DIRECTORY, base_pipeline_filename)
        logger.info(f"No explicit pipeline_path provided for explanation, using baseline path: {actual_pipeline_path}")

    if not os.path.exists(actual_pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at '{actual_pipeline_path}'. Was the model trained successfully?")

    try:
        # Note: explain_baseline_model now gets target/sensitive from metadata associated with pipeline_path
        logger.info(f"Explaining model using pipeline: '{actual_pipeline_path}'")
        results = explain_baseline_model(
            pipeline_path=actual_pipeline_path, # Pass determined path
            n_samples=n_samples
        )
        logger.info(f"Model explanation complete for pipeline '{actual_pipeline_path}'.")
        return JSONResponse(status_code=200, content=results)
    except ValueError as e:
        logger.error(f"Model explanation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e: # Catch file not found (e.g., metadata or data file)
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model explanation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model explanation: {e}")


# --- Root Endpoint (Keep) ---
@app.get("/", summary="Root endpoint", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Ethical AI Bias Mitigation Workbench API (v0.1.3)"}