import axios from 'axios';

// Ensure your FastAPI backend is running and accessible at this URL
const API_BASE_URL = 'http://127.0.0.1:8000';

// --- Upload Endpoint ---
export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data; // { message: "...", filename: "..." }
    } catch (error) {
        console.error("Error uploading file:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("File upload failed or server unreachable.");
    }
};

// --- Analyze Endpoint ---
export const analyzeData = async (filename, targetColumn, sensitiveAttributeColumns) => {
    const params = new URLSearchParams();
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns); // Comma-separated string

    try {
        const response = await axios.post(`${API_BASE_URL}/analyze`, params, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        });
        return response.data; // Full analysis results
    } catch (error) {
        console.error("Error analyzing data:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Data analysis failed or server unreachable.");
    }
};

// --- Train Baseline Endpoint ---
export const trainBaselineModel = async (filename, targetColumn, sensitiveAttributeColumns) => {
    const params = new URLSearchParams();
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns);

    try {
        const response = await axios.post(`${API_BASE_URL}/train_baseline`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data; // { metrics: {...}, model_info: { pipeline_path: "..."} }
    } catch (error) {
        console.error("Error training baseline model:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Baseline training failed.");
    }
};

// --- Mitigate Reweigh Endpoint ---
export const mitigateReweigh = async (filename, targetColumn, sensitiveAttributeColumns, reweighAttribute) => {
    const params = new URLSearchParams();
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns);
    params.append('reweigh_attribute', reweighAttribute);

    try {
        const response = await axios.post(`${API_BASE_URL}/mitigate_reweigh`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data;
    } catch (error) {
        console.error("Error during reweighing mitigation:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Reweighing mitigation failed.");
    }
};

// --- Mitigate Oversample Endpoint ---
export const mitigateOversample = async (filename, targetColumn, sensitiveAttributeColumns) => {
    const params = new URLSearchParams();
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns);

    try {
        const response = await axios.post(`${API_BASE_URL}/mitigate_oversample`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data;
    } catch (error) {
        console.error("Error during oversampling mitigation:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Oversampling mitigation failed.");
    }
};

// --- Calculate Fairness Endpoint ---
export const calculateFairness = async (filename, targetColumn, sensitiveAttributeColumns, pipelinePath = null) => {
    const params = new URLSearchParams();
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns);
    if (pipelinePath) {
        params.append('pipeline_path', pipelinePath);
    }

    try {
        const response = await axios.post(`${API_BASE_URL}/calculate_fairness`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data; // Dict keyed by sensitive attribute
    } catch (error) {
        console.error("Error calculating fairness:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Fairness calculation failed.");
    }
};

// --- Explain Model Endpoint ---
export const explainModel = async (filename, pipelinePath = null, nSamples = 100) => {
    const params = new URLSearchParams();
    params.append('filename', filename); // Still needed for original data loading if metadata points to it
    if (pipelinePath) {
        params.append('pipeline_path', pipelinePath);
    }
    params.append('n_samples', nSamples);

    try {
        const response = await axios.post(`${API_BASE_URL}/explain_model`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data;
    } catch (error) {
        console.error("Error explaining model:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Model explanation failed.");
    }
};

// --- Compare Models Endpoint ---
export const compareModels = async (baselinePipelinePath, mitigatedPipelinePath, filename, targetColumn, sensitiveAttributeColumns) => {
    const params = new URLSearchParams();
    params.append('baseline_pipeline_path', baselinePipelinePath);
    params.append('mitigated_pipeline_path', mitigatedPipelinePath);
    params.append('filename', filename);
    params.append('target_column', targetColumn);
    params.append('sensitive_attribute_columns', sensitiveAttributeColumns);

    try {
        const response = await axios.post(`${API_BASE_URL}/compare_models`, params, {
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        });
        return response.data; // { comparison_by_sensitive_attribute: {...} }
    } catch (error) {
        console.error("Error comparing models:", error.response ? error.response.data : error.message);
        throw error.response ? error.response.data : new Error("Model comparison failed.");
    }
};