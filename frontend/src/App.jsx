import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css'; 
import * as api from './services/api'; 
import MetricsTable from './components/MetricsTable';
import FairnessDisplay from './components/FairnessDisplay';
import ShapChart from './components/ShapChart';
import AnalysisDisplay from './components/AnalysisDisplay';
import { ThemeProvider } from './context/ThemeContext';
import ThemeToggle from './components/ThemeToggle';

// Simple display components (can be moved to separate files later)
const JsonDisplay = ({ data, title }) => (
  data && (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 space-y-4"
    >
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">{title}</h3>
      </div>
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96">
        <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    </motion.div>
  )
);

const ModelInfoDisplay = ({ modelInfo, title, metrics }) => (
  modelInfo && (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 space-y-4"
    >
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">{title}</h3>
      
      {/* Model metadata */}
      <div className="space-y-2 mb-4">
        <p className="text-gray-600 dark:text-gray-400">
          <span className="font-medium">Pipeline Path:</span> {modelInfo.pipeline_path}
        </p>
        <p className="text-gray-600 dark:text-gray-400">
          <span className="font-medium">Metadata Path:</span> {modelInfo.metadata_path}
        </p>
        {modelInfo.mitigation_applied && modelInfo.mitigation_applied !== "None" && (
          <p className="text-gray-600 dark:text-gray-400">
            <span className="font-medium">Mitigation:</span> {modelInfo.mitigation_applied}
            {modelInfo.mitigation_target_attribute && ` (on ${modelInfo.mitigation_target_attribute})`}
          </p>
        )}
      </div>
      
      {/* Performance metrics if available */}
      {metrics && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
          <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3">Model Performance</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics.accuracy !== undefined && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Accuracy</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{(metrics.accuracy).toFixed(4)}</p>
              </div>
            )}
            {metrics.precision !== undefined && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Precision</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{(metrics.precision).toFixed(4)}</p>
              </div>
            )}
            {metrics.recall !== undefined && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Recall</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{(metrics.recall).toFixed(4)}</p>
              </div>
            )}
            {metrics.f1_score !== undefined && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">F1 Score</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{(metrics.f1_score).toFixed(4)}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
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
  const [baselineMetrics, setBaselineMetrics] = useState(null);
  const [reweighedModelInfo, setReweighedModelInfo] = useState(null);
  const [reweighedMetrics, setReweighedMetrics] = useState(null);
  const [oversampledModelInfo, setOversampledModelInfo] = useState(null);
  const [oversampledMetrics, setOversampledMetrics] = useState(null);

  // Results State
  const [analysisResults, setAnalysisResults] = useState(null);
  const [fairnessResults, setFairnessResults] = useState(null); // For single model fairness
  const [shapResults, setShapResults] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);

  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [currentAction, setCurrentAction] = useState(''); // To show specific loading message
  const [error, setError] = useState('');
  const [showRawData, setShowRawData] = useState(false);

  const [selectedModelForEval, setSelectedModelForEval] = useState('');
  
  const handleApiCall = async (actionMessage, apiFunc, ...args) => {
    setIsLoading(true);
    setCurrentAction(actionMessage);
    setError('');
    try {
      const result = await apiFunc(...args);
      console.log(`${actionMessage} Result:`, result); // Log the result for debugging
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
    setBaselineMetrics(null); setReweighedMetrics(null); setOversampledMetrics(null);
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
        if (result.metrics) {
          setBaselineMetrics(result.metrics);
        }
    }
  };

  const handleMitigateReweigh = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!reweighAttribute) { setError('Select an attribute for reweighing.'); return; }
    const result = await handleApiCall(`Applying Reweighing (on ${reweighAttribute})...`, api.mitigateReweigh, uploadedFilename, targetColumn, sensitiveAttrs, reweighAttribute);
    if (result) {
      setReweighedModelInfo(result.model_info);
      if (result.metrics) {
        setReweighedMetrics(result.metrics);
      }
    }
  };

  const handleMitigateOversample = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    const result = await handleApiCall('Applying Oversampling...', api.mitigateOversample, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) {
      setOversampledModelInfo(result.model_info);
      if (result.metrics) {
        setOversampledMetrics(result.metrics);
      }
    }
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
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
        <ThemeToggle />
        <header className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-800 dark:to-indigo-900 text-white py-8 px-4 shadow-lg">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-center"
          >
            AI Bias Mitigation Workbench
          </motion.h1>
        </header>

        <AnimatePresence>
          {isLoading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-40"
            >
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-xl">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
                <p className="mt-4 text-gray-700 dark:text-gray-300">{currentAction}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-100 dark:bg-red-900 border-l-4 border-red-500 text-red-700 dark:text-red-200 p-4 mx-4 my-4 rounded"
          >
            <p className="font-medium">Error: {error}</p>
          </motion.div>
        )}

        <main className="container mx-auto px-4 py-8 space-y-8">
          {/* Section 1: Dataset & Config */}
          <motion.section 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
          >
            <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6">1. Dataset & Configuration</h2>
            
            <div className="space-y-6">
              <div className="flex items-center space-x-4">
                <label className="flex-1">
                  <span className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Upload Dataset</span>
                  <input 
                    type="file" 
                    onChange={handleFileChange} 
                    accept=".csv"
                    className="block w-full text-sm text-gray-500 dark:text-gray-400
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-blue-50 file:text-blue-700
                      dark:file:bg-blue-900 dark:file:text-blue-300
                      hover:file:bg-blue-100 dark:hover:file:bg-blue-800"
                  />
                </label>
                <button 
                  onClick={handleUpload} 
                  disabled={isLoading || !selectedFile}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Upload CSV
                </button>
              </div>

              {uploadedFilename && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-4"
                >
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Uploaded: <span className="font-medium">{uploadedFilename}</span>
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="targetCol" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Target Column
                      </label>
                      <input 
                        type="text" 
                        id="targetCol" 
                        value={targetColumn} 
                        onChange={(e) => setTargetColumn(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label htmlFor="sensitiveAttrs" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Sensitive Attributes (comma-separated)
                      </label>
                      <input 
                        type="text" 
                        id="sensitiveAttrs" 
                        value={sensitiveAttrs} 
                        onChange={(e) => setSensitiveAttrs(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100"
                      />
                    </div>
                  </div>

                  <button 
                    onClick={handleAnalyze} 
                    disabled={isLoading}
                    className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Analyze Data
                  </button>
                </motion.div>
              )}
            </div>
          </motion.section>

          <AnimatePresence>
            {analysisResults && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <AnalysisDisplay
                  data={analysisResults}
                  filename={analysisResults.filename}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Section 2: Train Models */}
          {uploadedFilename && (
            <motion.section 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
            >
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6">2. Train Models</h2>
              
              <div className="space-y-8">
                <div className="model-training-action">
                  <button 
                    onClick={handleTrainBaseline} 
                    disabled={isLoading}
                    className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-800 disabled:opacity-50 disabled:cursor-not-allowed mb-4"
                  >
                    Train Baseline Model
                  </button>
                  {baselineModelInfo && (
                    <>
                      <ModelInfoDisplay 
                        modelInfo={baselineModelInfo} 
                        title="Baseline Model Artifacts" 
                        metrics={baselineMetrics}
                      />
                      {showRawData && <JsonDisplay data={baselineModelInfo} title="Raw Baseline Model Info" />}
                    </>
                  )}
                </div>

                <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
                  <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-6">Mitigation Techniques</h3>
                  
                  <div className="space-y-8">
                    <div className="model-training-action">
                      <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">Reweighing</h4>
                      <div className="space-y-4">
                        <div>
                          <label htmlFor="reweighAttr" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Reweigh by Attribute
                          </label>
                          <select 
                            id="reweighAttr" 
                            value={reweighAttribute} 
                            onChange={(e) => setReweighAttribute(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100"
                          >
                            {sensitiveAttrs.split(',').map(attr => attr.trim()).filter(attr => attr).map(attr => (
                              <option key={attr} value={attr}>{attr}</option>
                            ))}
                          </select>
                        </div>
                        <button 
                          onClick={handleMitigateReweigh} 
                          disabled={isLoading || !reweighAttribute}
                          className="w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 dark:bg-purple-700 dark:hover:bg-purple-800 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Apply Reweighing
                        </button>
                      </div>
                      {reweighedModelInfo && (
                        <>
                          <ModelInfoDisplay 
                            modelInfo={reweighedModelInfo} 
                            title="Reweighed Model Artifacts" 
                            metrics={reweighedMetrics}
                          />
                          {showRawData && <JsonDisplay data={reweighedModelInfo} title="Raw Reweighed Model Info" />}
                        </>
                      )}
                    </div>

                    <div className="model-training-action">
                      <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">Random Oversampling</h4>
                      <button 
                        onClick={handleMitigateOversample} 
                        disabled={isLoading}
                        className="w-full px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 dark:bg-orange-700 dark:hover:bg-orange-800 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Apply Oversampling
                      </button>
                      {oversampledModelInfo && (
                        <>
                          <ModelInfoDisplay 
                            modelInfo={oversampledModelInfo} 
                            title="Oversampled Model Artifacts" 
                            metrics={oversampledMetrics}
                          />
                          {showRawData && <JsonDisplay data={oversampledModelInfo} title="Raw Oversampled Model Info" />}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.section>
          )}

          {/* Section 3: Evaluate & Explain Individual Models */}
          {trainedModels.length > 0 && (
            <motion.section 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
            >
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6">3. Evaluate & Explain Individual Models</h2>
              
              <div className="space-y-6">
                <p className="text-gray-600 dark:text-gray-400">Select a trained model to evaluate its fairness or get SHAP explanations.</p>
                
                <select
                  value={selectedModelForEval}
                  onChange={(e) => {
                    setSelectedModelForEval(e.target.value);
                    setFairnessResults(null);
                    setShapResults(null);
                  }}
                  disabled={trainedModels.length === 0}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100"
                >
                  <option value="">Select a model...</option>
                  {trainedModels.map((model) => (
                    <option key={model.pipeline_path} value={model.pipeline_path}>
                      {getModelName(model)}
                    </option>
                  ))}
                </select>

                {selectedModelForEval && (
                  <div className="flex space-x-4">
                    <button
                      onClick={() => handleCalculateFairness(selectedModelForEval, getModelName(trainedModels.find(m => m.pipeline_path === selectedModelForEval)))}
                      disabled={isLoading}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Calculate Fairness
                    </button>
                    <button
                      onClick={() => handleExplainModel(selectedModelForEval, getModelName(trainedModels.find(m => m.pipeline_path === selectedModelForEval)))}
                      disabled={isLoading}
                      className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Generate SHAP
                    </button>
                  </div>
                )}
              </div>

              <AnimatePresence>
                {fairnessResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <FairnessDisplay
                      title={fairnessResults.title}
                      data={fairnessResults.data}
                    />
                    {showRawData && <JsonDisplay data={fairnessResults.data} title="Raw Fairness Data" />}
                  </motion.div>
                )}

                {shapResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <ShapChart
                      title={shapResults.title}
                      data={shapResults.data.features}
                    />
                    {showRawData && <JsonDisplay data={shapResults.data} title="Raw SHAP Data" />}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>
          )}

          {/* Section 4: Compare Models */}
          {trainedModels.length >= 2 && (
            <motion.section 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
            >
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6">4. Compare Models</h2>
              
              <button
                onClick={handleCompareModels}
                disabled={isLoading || !baselineModelInfo || !(reweighedModelInfo || oversampledModelInfo)}
                className="w-full px-4 py-2 bg-teal-600 text-white rounded-md hover:bg-teal-700 dark:bg-teal-700 dark:hover:bg-teal-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Compare Models
              </button>

              <AnimatePresence>
                {comparisonResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <MetricsTable 
                      title="Model Comparison" 
                      data={comparisonResults} 
                    />
                    {showRawData && <JsonDisplay data={comparisonResults} title="Raw Comparison Data" />}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>
          )}

          {/* Global Controls */}
          <div className="flex justify-end">
            <button
              onClick={() => setShowRawData(!showRawData)}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 dark:bg-gray-700 dark:hover:bg-gray-800"
            >
              {showRawData ? "Hide Raw JSON Data" : "Show Raw JSON Data"}
            </button>
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;