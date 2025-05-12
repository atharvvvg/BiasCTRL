# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional # Ensure Optional is imported
import os
import shutil
import logging
import json


# Import core functions
from core.analysis import load_and_analyze_data
from core.models import train_evaluate_baseline, train_evaluate_reweighed, train_evaluate_oversampled
from core.fairness import calculate_fairness_metrics
from core.explainability import explain_baseline_model
# No need to import imblearn components here as they are used within core.models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ethical AI Bias Mitigation Workbench API",
    description="Phase 1: Core backend with flexible fairness/explainability.",
    version="0.1.5",
)

# CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:5173", # Vite default React dev server
    # Add your deployed frontend URL here in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/mitigate_oversample", summary="Train model using Random Oversampling mitigation")
async def mitigate_oversample_endpoint(
    filename: str = Form(...),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attribute column names (e.g., race,gender). Used for evaluation.")
):
    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        logger.info(f"Starting Random Oversampling mitigation for: {filename}, target: {target_column}, sensitive_for_eval: {parsed_sensitive_columns}")
        results = train_evaluate_oversampled(
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        logger.info(f"Random Oversampling mitigation training complete for {filename}.")
        return JSONResponse(status_code=200, content=results)
    except FileNotFoundError as e:
        logger.error(f"Oversampling failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Oversampling failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during Oversampling mitigation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/calculate_fairness", summary="Calculate fairness metrics for a trained model")
async def calculate_fairness_endpoint(
    filename: str = Form(..., description="Filename of the *original* data for evaluation"),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attributes (e.g., race,gender)"),
    pipeline_path: Optional[str] = Form(None, description="Optional: Path to a specific .joblib pipeline file")
):
    actual_pipeline_path: str
    if pipeline_path:
        if not os.path.isabs(pipeline_path) and not pipeline_path.startswith(MODEL_DIRECTORY) and MODEL_DIRECTORY in pipeline_path:
            actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(pipeline_path))
        elif os.path.isabs(pipeline_path): actual_pipeline_path = pipeline_path
        else: actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(pipeline_path))
    else:
        base_pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
        actual_pipeline_path = os.path.join(MODEL_DIRECTORY, base_pipeline_filename)
    logger.info(f"Determined pipeline path for fairness calculation: {actual_pipeline_path}")

    if not os.path.exists(actual_pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at '{actual_pipeline_path}'.")

    try:
        parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
        if not parsed_sensitive_columns:
             raise HTTPException(status_code=400, detail="No sensitive attributes provided for fairness calculation.")

        logger.info(f"Calculating fairness for pipeline: '{actual_pipeline_path}', data: '{filename}', sensitive attrs: {parsed_sensitive_columns}")
        results_by_sensitive_attr = calculate_fairness_metrics(
            pipeline_path=actual_pipeline_path,
            filename=filename,
            target_column=target_column,
            sensitive_attribute_columns=parsed_sensitive_columns
        )
        logger.info(f"Fairness calculation complete for pipeline '{actual_pipeline_path}'.")
        return JSONResponse(status_code=200, content=results_by_sensitive_attr)

    except ValueError as e:
        logger.error(f"Fairness calculation failed (ValueError): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Fairness calculation failed (FileNotFound): {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during fairness calculation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/explain_model", summary="Explain a trained model using SHAP")
async def explain_model_endpoint(
    filename: str = Form(..., description="Filename of the *original* data (needed if metadata is used for data loading, e.g., adult.csv)"),
    pipeline_path: Optional[str] = Form(None, description="Optional: Full path to a specific .joblib pipeline file. If not provided, uses baseline path from 'filename'."), # Corrected parameter name
    n_samples: int = Form(100, description="Number of samples for SHAP background/explanation")
):
    actual_pipeline_path: str
    if pipeline_path: # Use pipeline_path directly
        if not os.path.isabs(pipeline_path) and not pipeline_path.startswith(MODEL_DIRECTORY) and MODEL_DIRECTORY in pipeline_path:
            actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(pipeline_path))
        else:
            actual_pipeline_path = pipeline_path
        logger.info(f"Using explicit pipeline path for explanation: {actual_pipeline_path}")
    else:
        base_pipeline_filename = f"{os.path.splitext(filename)[0]}_pipeline.joblib"
        actual_pipeline_path = os.path.join(MODEL_DIRECTORY, base_pipeline_filename)
        logger.info(f"No explicit pipeline_path provided for explanation, using baseline path: {actual_pipeline_path}")

    if not os.path.exists(actual_pipeline_path):
         raise HTTPException(status_code=404, detail=f"Model pipeline file not found at '{actual_pipeline_path}'. Was the model trained successfully?")
    try:
        logger.info(f"Explaining model using pipeline: '{actual_pipeline_path}'")
        results = explain_baseline_model(
            pipeline_path=actual_pipeline_path,
            n_samples=n_samples
        )
        logger.info(f"Model explanation complete for pipeline '{actual_pipeline_path}'.")
        return JSONResponse(status_code=200, content=results)
    except ValueError as e:
        logger.error(f"Model explanation failed due to value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model explanation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during model explanation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model explanation: {e}")

@app.post("/compare_models", summary="Compare fairness of two models across specified sensitive attributes")
async def compare_models_endpoint(
    baseline_pipeline_path: str = Form(..., description="Path to the baseline .joblib pipeline file"),
    mitigated_pipeline_path: str = Form(..., description="Path to the mitigated .joblib pipeline file"),
    filename: str = Form(..., description="Filename of the *original* data for evaluation"),
    target_column: str = Form(...),
    sensitive_attribute_columns: str = Form(..., description="Comma-separated sensitive attributes for fairness eval")
):
    comparison_results_by_attr = {}
    parsed_sensitive_columns = parse_sensitive_attributes(sensitive_attribute_columns)
    if not parsed_sensitive_columns:
        raise HTTPException(status_code=400, detail="No sensitive attributes provided for comparison.")

    for model_name_key, path_param_value in [("baseline", baseline_pipeline_path), ("mitigated", mitigated_pipeline_path)]:
        actual_pipeline_path: str
        if not os.path.isabs(path_param_value) and not path_param_value.startswith(MODEL_DIRECTORY) and MODEL_DIRECTORY in path_param_value:
            actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(path_param_value))
        elif os.path.isabs(path_param_value): actual_pipeline_path = path_param_value
        else: actual_pipeline_path = os.path.join(MODEL_DIRECTORY, os.path.basename(path_param_value))
        logger.info(f"Resolving path for {model_name_key} model comparison: input '{path_param_value}', resolved to '{actual_pipeline_path}'")

        if not os.path.exists(actual_pipeline_path):
            for sens_attr in parsed_sensitive_columns:
                if sens_attr not in comparison_results_by_attr: comparison_results_by_attr[sens_attr] = {}
                comparison_results_by_attr[sens_attr][model_name_key] = {"error": f"Model pipeline file not found at '{actual_pipeline_path}'.", "pipeline_path": actual_pipeline_path}
            continue

        logger.info(f"Calculating fairness for {model_name_key} model: '{actual_pipeline_path}' across sensitive attributes: {parsed_sensitive_columns}")
        try:
            fairness_data_for_model = calculate_fairness_metrics(
                pipeline_path=actual_pipeline_path,
                filename=filename,
                target_column=target_column,
                sensitive_attribute_columns=parsed_sensitive_columns
            )

            for sens_attr, metrics_for_sens_attr in fairness_data_for_model.items():
                if sens_attr not in comparison_results_by_attr:
                    comparison_results_by_attr[sens_attr] = {}

                if "error" in metrics_for_sens_attr:
                    comparison_results_by_attr[sens_attr][model_name_key] = metrics_for_sens_attr
                    continue

                comparison_results_by_attr[sens_attr][model_name_key] = {
                    "pipeline_path": actual_pipeline_path,
                    "overall_accuracy_from_fairness_eval": metrics_for_sens_attr.get("fairness_metrics", {}).get("overall", {}).get("accuracy"),
                    "fairness_disparities": metrics_for_sens_attr.get("fairness_metrics", {}).get("disparities", {}),
                    "standard_fairness_definitions": metrics_for_sens_attr.get("fairness_metrics", {}).get("standard_definitions", {})
                }
        except Exception as e:
            logger.exception(f"Error processing {model_name_key} model at '{actual_pipeline_path}': {e}")
            for sens_attr in parsed_sensitive_columns:
                if sens_attr not in comparison_results_by_attr: comparison_results_by_attr[sens_attr] = {}
                comparison_results_by_attr[sens_attr][model_name_key] = {"error": str(e), "pipeline_path": actual_pipeline_path}

    return JSONResponse(status_code=200, content={"comparison_by_sensitive_attribute": comparison_results_by_attr})

@app.get("/", summary="Root endpoint", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Ethical AI Bias Mitigation Workbench API (v0.1.5)"}