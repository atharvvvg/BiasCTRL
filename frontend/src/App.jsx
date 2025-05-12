import React, { useState, useCallback } from 'react';
import './App.css'; // We'll create this for basic styling
import * as api from './services/api'; // Import all API functions

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
  const [reweighAttribute, setReweighAttribute] = useState('race'); // Default for reweighing

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
        // Optionally, auto-fetch fairness for baseline
        // await handleCalculateFairness(result.model_info.pipeline_path, 'Baseline Model Fairness');
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

  const handleCalculateFairness = async (pipelinePathToUse, titlePrefix = "Fairness Results") => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Train a model or provide pipeline path.'); return; }
    const result = await handleApiCall(`Calculating fairness for ${pipelinePathToUse}...`, api.calculateFairness, uploadedFilename, targetColumn, sensitiveAttrs, pipelinePathToUse);
    if (result) setFairnessResults({title: titlePrefix, data: result});
  };

  const handleExplainModel = async (pipelinePathToUse) => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Train a model or provide pipeline path.'); return; }
    const result = await handleApiCall(`Generating SHAP for ${pipelinePathToUse}...`, api.explainModel, uploadedFilename, pipelinePathToUse, 100);
    if (result) setShapResults({title: `SHAP for ${pipelinePathToUse}`, data: result});
  };

  const handleCompareModels = async () => {
    if (!baselineModelInfo || !(reweighedModelInfo || oversampledModelInfo)) {
      setError('Train at least a baseline and one mitigated model to compare.');
      return;
    }
    // For simplicity, compare baseline with the first available mitigated model
    const mitigatedPath = reweighedModelInfo ? reweighedModelInfo.pipeline_path : (oversampledModelInfo ? oversampledModelInfo.pipeline_path : null);
    if (!mitigatedPath) { setError('No mitigated model found for comparison.'); return; }

    const result = await handleApiCall('Comparing models...', api.compareModels, baselineModelInfo.pipeline_path, mitigatedPath, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) setComparisonResults(result);
  };

  return (
    <div className="App">
      <header><h1>Ethical AI Bias Mitigation Workbench</h1></header>
      {isLoading && <div className="loading-overlay">Loading: {currentAction}</div>}
      {error && <p className="error-message">Error: {error}</p>}

      <main>
        <section className="card">
          <h2>1. Dataset & Configuration</h2>
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

        <JsonDisplay data={analysisResults?.analysis} title="Data Analysis Overview" />

        {uploadedFilename && (
          <section className="card">
            <h2>2. Train Models</h2>
            <button onClick={handleTrainBaseline} disabled={isLoading}>Train Baseline Model</button>
            <ModelInfoDisplay modelInfo={baselineModelInfo} title="Baseline Model Trained" />

            <hr />
            <h3>Mitigation Techniques:</h3>
            <div>
              <h4>Reweighing</h4>
              <label htmlFor="reweighAttr">Reweigh by Attribute: </label>
              <select id="reweighAttr" value={reweighAttribute} onChange={(e) => setReweighAttribute(e.target.value)}>
                {sensitiveAttrs.split(',').map(attr => attr.trim()).filter(attr => attr).map(attr => (
                  <option key={attr} value={attr}>{attr}</option>
                ))}
              </select>
              <button onClick={handleMitigateReweigh} disabled={isLoading || !reweighAttribute}>Apply Reweighing</button>
              <ModelInfoDisplay modelInfo={reweighedModelInfo} title="Reweighed Model Trained" />
            </div>

            <div>
              <h4>Random Oversampling</h4>
              <button onClick={handleMitigateOversample} disabled={isLoading}>Apply Oversampling</button>
              <ModelInfoDisplay modelInfo={oversampledModelInfo} title="Oversampled Model Trained" />
            </div>
          </section>
        )}

        {(baselineModelInfo || reweighedModelInfo || oversampledModelInfo) && (
          <section className="card">
            <h2>3. Evaluate & Explain Individual Models</h2>
            {/* Simplified: allow choosing which model to evaluate/explain */}
            {[baselineModelInfo, reweighedModelInfo, oversampledModelInfo].filter(Boolean).map(model => (
              model && model.pipeline_path && (
                <div key={model.pipeline_path} className="model-actions">
                  <h4>Actions for: {model.pipeline_path.split(/[\\/]/).pop()}</h4>
                  <button onClick={() => handleCalculateFairness(model.pipeline_path, `Fairness for ${model.pipeline_path.split(/[\\/]/).pop()}`)} disabled={isLoading}>Calculate Fairness</button>
                  <button onClick={() => handleExplainModel(model.pipeline_path)} disabled={isLoading}>Generate SHAP Explanation</button>
                </div>
              )
            ))}
          </section>
        )}

        <JsonDisplay data={fairnessResults?.data} title={fairnessResults?.title} />
        <JsonDisplay data={shapResults?.data} title={shapResults?.title} />


        {(baselineModelInfo && (reweighedModelInfo || oversampledModelInfo)) && (
          <section className="card">
            <h2>4. Compare Models</h2>
            <p>Compares baseline with the first available mitigated model.</p>
            <button onClick={handleCompareModels} disabled={isLoading}>Compare Baseline vs. Mitigated</button>
          </section>
        )}
        <JsonDisplay data={comparisonResults} title="Model Comparison Results" />

      </main>
    </div>
  );
}

export default App;