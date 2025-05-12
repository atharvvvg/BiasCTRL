import React, { useState, useCallback } from 'react';
import './App.css'; 
import * as api from './services/api'; 
import MetricsTable from './components/MetricsTable';
import FairnessDisplay from './components/FairnessDisplay';
import ShapChart from './components/ShapChart';

// Simple display components (can be moved to separate files later)
const JsonDisplay = ({ data, title }) => (
  data && (
    <div className="results-card">
      <h3>{title}</h3>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
);

const ModelInfoDisplay = ({ modelInfo, title }) => (
  modelInfo && (
    <div className="results-card model-info-card">
      <h3>{title}</h3>
      <p><strong>Pipeline Path:</strong> {modelInfo.pipeline_path}</p>
      <p><strong>Metadata Path:</strong> {modelInfo.metadata_path}</p>
      {modelInfo.mitigation_applied && modelInfo.mitigation_applied !== "None" && (
        <p><strong>Mitigation:</strong> {modelInfo.mitigation_applied}
          {modelInfo.mitigation_target_attribute && ` (on ${modelInfo.mitigation_target_attribute})`}
        </p>
      )}
    </div>
  )
);


function App() {
  // File and Config State
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [targetColumn, setTargetColumn] = useState('income');
  const [sensitiveAttrs, setSensitiveAttrs] = useState('race,gender');
  const [reweighAttribute, setReweighAttribute] = useState('race'); 

  // Model Training State
  const [baselineModelInfo, setBaselineModelInfo] = useState(null);
  const [reweighedModelInfo, setReweighedModelInfo] = useState(null);
  const [oversampledModelInfo, setOversampledModelInfo] = useState(null);

  // Results State
  const [analysisResults, setAnalysisResults] = useState(null);
  const [fairnessResults, setFairnessResults] = useState(null); // For single model fairness
  const [shapResults, setShapResults] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);

  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [currentAction, setCurrentAction] = useState(''); // To show specific loading message
  const [error, setError] = useState('');

  const [selectedModelForEval, setSelectedModelForEval] = useState('');
  
  const handleApiCall = async (actionMessage, apiFunc, ...args) => {
    setIsLoading(true);
    setCurrentAction(actionMessage);
    setError('');
    try {
      const result = await apiFunc(...args);
      return result;
    } catch (err) {
      const errorMessage = err.detail || (err.response && err.response.data && err.response.data.detail) || err.message || `Action "${actionMessage}" failed.`;
      setError(errorMessage);
      console.error(`${actionMessage} Error:`, err);
      throw err; // Re-throw for specific handling if needed
    } finally {
      setIsLoading(false);
      setCurrentAction('');
    }
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadedFilename(''); setAnalysisResults(null); setBaselineModelInfo(null);
    setReweighedModelInfo(null); setOversampledModelInfo(null); setFairnessResults(null);
    setShapResults(null); setComparisonResults(null); setError('');
  };

  const handleUpload = async () => {
    if (!selectedFile) { setError('Please select a file.'); return; }
    const result = await handleApiCall('Uploading file...', api.uploadFile, selectedFile);
    if (result) setUploadedFilename(result.filename);
  };

  const handleAnalyze = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    const result = await handleApiCall('Analyzing data...', api.analyzeData, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) setAnalysisResults(result);
  };

  const handleTrainBaseline = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    const result = await handleApiCall('Training baseline model...', api.trainBaselineModel, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) {
        setBaselineModelInfo(result.model_info);
    }
  };

  const handleMitigateReweigh = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!reweighAttribute) { setError('Select an attribute for reweighing.'); return; }
    const result = await handleApiCall(`Applying Reweighing (on ${reweighAttribute})...`, api.mitigateReweigh, uploadedFilename, targetColumn, sensitiveAttrs, reweighAttribute);
    if (result) setReweighedModelInfo(result.model_info);
  };

  const handleMitigateOversample = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    const result = await handleApiCall('Applying Oversampling...', api.mitigateOversample, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) setOversampledModelInfo(result.model_info);
  };

  const handleCalculateFairness = async (pipelinePathToUse, modelNameForTitle) => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Model pipeline path is missing.'); return; }
    const result = await handleApiCall(
      `Calculating fairness for ${modelNameForTitle}...`,
      api.calculateFairness,
      uploadedFilename, targetColumn, sensitiveAttrs, pipelinePathToUse
    );
    if (result) {
      setFairnessResults({ title: `Fairness Results for ${modelNameForTitle}`, data: result, forPipeline: pipelinePathToUse });
      setSelectedModelForEval(pipelinePathToUse); 
    }
  };

  const handleExplainModel = async (pipelinePathToUse, modelNameForTitle) => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Model pipeline path is missing.'); return; }
    const result = await handleApiCall(
      `Generating SHAP for ${modelNameForTitle}...`,
      api.explainModel,
      uploadedFilename, pipelinePathToUse, 100
    );
    if (result) {
      setShapResults({ title: `SHAP Explanation for ${modelNameForTitle}`, data: result, forPipeline: pipelinePathToUse });
      setSelectedModelForEval(pipelinePathToUse); 
    }
  };

  const handleCompareModels = async () => {
    if (!baselineModelInfo || !(reweighedModelInfo || oversampledModelInfo)) {
      setError('Train at least a baseline and one mitigated model to compare.');
      return;
    }
    const mitigatedPath = reweighedModelInfo ? reweighedModelInfo.pipeline_path : (oversampledModelInfo ? oversampledModelInfo.pipeline_path : null);
    if (!mitigatedPath) { setError('No mitigated model found for comparison.'); return; }

    const result = await handleApiCall('Comparing models...', api.compareModels, baselineModelInfo.pipeline_path, mitigatedPath, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) {
        setComparisonResults(result);
        setFairnessResults(null);
        setShapResults(null);
    }
  };

  const getModelName = (modelInfo) => {
    if (!modelInfo || !modelInfo.pipeline_path) return "Unknown Model";
    const parts = modelInfo.pipeline_path.split(/[\\/]/).pop().split('_');
    if (parts.includes("baseline")) return "Baseline";
    if (parts.includes("reweighed")) return `Reweighed (by ${modelInfo.mitigation_target_attribute || parts[parts.indexOf("reweighed")+1]})`;
    if (parts.includes("oversampled")) return "Oversampled";
    return modelInfo.pipeline_path.split(/[\\/]/).pop();
  };

  const trainedModels = [baselineModelInfo, reweighedModelInfo, oversampledModelInfo].filter(Boolean);

  return (
    <div className="App">
      <header><h1>AI Bias Mitigation Workbench</h1></header>
      {isLoading && <div className="loading-overlay">Loading: {currentAction}</div>}
      {error && <p className="error-message">Error: {error}</p>}

      <main>
        {/* --- Section 1: Dataset & Config --- */}
        <section className="card">
          <h2>1. Dataset & Configuration</h2>
          {/* ... (file upload, target, sensitive inputs as before) ... */}
          <input type="file" onChange={handleFileChange} accept=".csv" />
          <button onClick={handleUpload} disabled={isLoading || !selectedFile}>Upload CSV</button>
          {uploadedFilename && <p>Uploaded: <strong>{uploadedFilename}</strong></p>}

          {uploadedFilename && (<>
            <div>
              <label htmlFor="targetCol">Target Column: </label>
              <input type="text" id="targetCol" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} />
            </div>
            <div>
              <label htmlFor="sensitiveAttrs">Sensitive Attributes (comma-separated): </label>
              <input type="text" id="sensitiveAttrs" value={sensitiveAttrs} onChange={(e) => setSensitiveAttrs(e.target.value)} />
            </div>
            <button onClick={handleAnalyze} disabled={isLoading}>Analyze Data</button>
          </>)}
        </section>

        {/* Display Analysis Results using JsonDisplay (or a new component) */}
        {analysisResults && (
             <div className="card results-card">
                <h3>Data Analysis Overview for {analysisResults.filename}</h3>
                <pre>{JSON.stringify(analysisResults.analysis, null, 2)}</pre>
            </div>
        )}


        {/* --- Section 2: Train Models --- */}
        {uploadedFilename && (
          <section className="card">
            <h2>2. Train Models</h2>
            <div className="model-training-action">
                <button onClick={handleTrainBaseline} disabled={isLoading}>Train Baseline Model</button>
                <ModelInfoDisplay modelInfo={baselineModelInfo} title="Baseline Model Artifacts" />
            </div>
            <hr />
            <h3>Mitigation Techniques:</h3>
            <div className="model-training-action">
              <h4>Reweighing</h4>
              <label htmlFor="reweighAttr">Reweigh by Attribute: </label>
              <select id="reweighAttr" value={reweighAttribute} onChange={(e) => setReweighAttribute(e.target.value)}>
                {sensitiveAttrs.split(',').map(attr => attr.trim()).filter(attr => attr).map(attr => (
                  <option key={attr} value={attr}>{attr}</option>
                ))}
              </select>
              <button onClick={handleMitigateReweigh} disabled={isLoading || !reweighAttribute}>Apply Reweighing</button>
              <ModelInfoDisplay modelInfo={reweighedModelInfo} title="Reweighed Model Artifacts" />
            </div>
            <div className="model-training-action">
              <h4>Random Oversampling</h4>
              <button onClick={handleMitigateOversample} disabled={isLoading}>Apply Oversampling</button>
              <ModelInfoDisplay modelInfo={oversampledModelInfo} title="Oversampled Model Artifacts" />
            </div>
          </section>
        )}

        {/* --- Section 3: Evaluate & Explain Individual Models --- */}
        {trainedModels.length > 0 && (
          <section className="card">
            <h2>3. Evaluate & Explain Individual Models</h2>
            <p>Select a trained model to evaluate its fairness or get SHAP explanations.</p>
            <select
              value={selectedModelForEval}
              onChange={(e) => {
                setSelectedModelForEval(e.target.value);
                setFairnessResults(null); // Clear previous results when model changes
                setShapResults(null);
              }}
              disabled={trainedModels.length === 0}
            >
              <option value="">-- Select a Trained Model --</option>
              {trainedModels.map(model => (
                model && model.pipeline_path && (
                  <option key={model.pipeline_path} value={model.pipeline_path}>
                    {getModelName(model)} ({model.pipeline_path.split(/[\\/]/).pop()})
                  </option>
                )
              ))}
            </select>

            {selectedModelForEval && (
                <div className="model-actions">
                    <button onClick={() => handleCalculateFairness(selectedModelForEval, getModelName(trainedModels.find(m=>m.pipeline_path === selectedModelForEval)) )} disabled={isLoading}>Calculate Fairness</button>
                    <button onClick={() => handleExplainModel(selectedModelForEval, getModelName(trainedModels.find(m=>m.pipeline_path === selectedModelForEval)))} disabled={isLoading}>Generate SHAP Explanation</button>
                </div>
            )}

            {/* Display Fairness Results for the selected model */}
            {fairnessResults && fairnessResults.forPipeline === selectedModelForEval && (
                <div className="results-display">
                    <h3>{fairnessResults.title}</h3>
                    {Object.entries(fairnessResults.data).map(([sensAttr, data]) => {
                        if (data.error) return <p key={sensAttr}>Error for {sensAttr}: {data.error}</p>;
                        return (
                            <div key={sensAttr} className="fairness-results-per-attribute">
                                <MetricsTable metrics={data.fairness_metrics?.overall} title={`Overall Metrics (evaluating on ${sensAttr})`} />
                                <FairnessDisplay fairnessData={data.fairness_metrics} sensitiveAttribute={sensAttr} />
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Display SHAP Results for the selected model */}
            {shapResults && shapResults.forPipeline === selectedModelForEval && (
                 <div className="results-display">
                    {/* <h3>{shapResults.title}</h3> Now handled by ShapChart title */}
                    <ShapChart shapData={shapResults.data} title={shapResults.title} />
                 </div>
            )}
          </section>
        )}


        {/* --- Section 4: Compare Models --- */}
        {baselineModelInfo && trainedModels.length > 1 && (
          <section className="card">
            <h2>4. Compare Models</h2>
            <p>Compares baseline with the first available mitigated model (Reweighed or Oversampled).</p>
            <button onClick={handleCompareModels} disabled={isLoading}>Compare Baseline vs. A Mitigated Model</button>

            {comparisonResults && comparisonResults.comparison_by_sensitive_attribute && (
                <div className="results-display comparison-view">
                    <h3>Model Comparison Results</h3>
                    {Object.entries(comparisonResults.comparison_by_sensitive_attribute).map(([sensAttr, comparisonData]) => (
                        <div key={sensAttr} className="comparison-per-attribute">
                            <h4>Comparison for Sensitive Attribute: "{sensAttr}"</h4>
                            <div className="comparison-columns">
                                {["baseline", "mitigated"].map(modelType => (
                                    comparisonData[modelType] && !comparisonData[modelType].error ? (
                                        <div key={modelType} className="comparison-column">
                                            <h5>{modelType.charAt(0).toUpperCase() + modelType.slice(1)} Model</h5>
                                            <small>({comparisonData[modelType].pipeline_path.split(/[\\/]/).pop()})</small>
                                            <MetricsTable metrics={{accuracy: comparisonData[modelType].overall_accuracy_from_fairness_eval}} title="Overall Accuracy" />
                                            {/* <FairnessDisplay fairnessData={comparisonData[modelType]} sensitiveAttribute={sensAttr} /> */}
                                            <FairnessDisplay
                                              fairnessData={{
                                                  disparities: comparisonData[modelType].fairness_disparities,
                                                  standard_definitions: comparisonData[modelType].standard_fairness_definitions
                                              }}
                                              sensitiveAttribute={sensAttr}
                                            />
                                        </div>
                                    ) : (
                                        <div key={modelType} className="comparison-column">
                                             <h5>{modelType.charAt(0).toUpperCase() + modelType.slice(1)} Model</h5>
                                             <p>Error or no data: {comparisonData[modelType]?.error || "Data unavailable"}</p>
                                        </div>
                                    )
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;