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
import GenericResultsDisplay from './components/GenericResultsDisplay';

function App() {
  // File and Config State
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [targetColumn, setTargetColumn] = useState('income');
  const [sensitiveAttrs, setSensitiveAttrs] = useState('race,gender');
  const [reweighAttribute, setReweighAttribute] = useState('race'); 
  const [shapNSamples, setShapNSamples] = useState(100); // Added for SHAP n_samples

  // Model Training State
  const [baselineModelResults, setBaselineModelResults] = useState(null);
  const [reweighedModelResults, setReweighedModelResults] = useState(null);
  const [oversampledModelResults, setOversampledModelResults] = useState(null);

  // Results State
  const [analysisResults, setAnalysisResults] = useState(null);
  // fairnessResults is for the specific FairnessDisplay component, if we choose to use it.
  // For now, /calculate_fairness populates fairnessCalculationResults for GenericResultsDisplay.
  const [fairnessResults, setFairnessResults] = useState(null); 
  const [shapResults, setShapResults] = useState(null); // For ShapChart
  const [comparisonResults, setComparisonResults] = useState(null); // For GenericResultsDisplay
  const [fairnessCalculationResults, setFairnessCalculationResults] = useState(null); // For GenericResultsDisplay
  const [explanationResults, setExplanationResults] = useState(null); // For GenericResultsDisplay (fallback for explain)

  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [currentAction, setCurrentAction] = useState(''); 
  const [error, setError] = useState('');

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
    setUploadedFilename(''); 
    setAnalysisResults(null); 
    setBaselineModelResults(null);
    setReweighedModelResults(null);
    setOversampledModelResults(null);
    setFairnessResults(null); // Clear this too, though not directly used for GenericDisplay
    setFairnessCalculationResults(null);
    setShapResults(null); 
    setExplanationResults(null);
    setComparisonResults(null); 
    setError('');
    setSelectedModelForEval(''); // Reset selected model
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
    if (result) setBaselineModelResults(result);
  };

  const handleMitigateReweigh = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!reweighAttribute) { setError('Select an attribute for reweighing.'); return; }
    const result = await handleApiCall(`Applying Reweighing (on ${reweighAttribute})...`, api.mitigateReweigh, uploadedFilename, targetColumn, sensitiveAttrs, reweighAttribute);
    if (result) setReweighedModelResults(result);
  };

  const handleMitigateOversample = async () => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    const result = await handleApiCall('Applying Oversampling...', api.mitigateOversample, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) setOversampledModelResults(result);
  };

  const handleCalculateFairness = async (pipelinePathToUse, modelNameForTitle) => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Model pipeline path is missing.'); return; }
    // Clear previous results for this section
    setFairnessCalculationResults(null); 
    setShapResults(null);
    setExplanationResults(null);

    const result = await handleApiCall(
      `Calculating fairness for ${modelNameForTitle}...`,
      api.calculateFairness,
      uploadedFilename, targetColumn, sensitiveAttrs, pipelinePathToUse
    );
    if (result) {
      setFairnessCalculationResults({ title: `Fairness Calculation: ${modelNameForTitle} (${pipelinePathToUse.split(/[\\/]/).pop()})`, data: result, forPipeline: pipelinePathToUse });
    }
  };

  const handleExplainModel = async (pipelinePathToUse, modelNameForTitle) => {
    if (!uploadedFilename) { setError('Upload a file first.'); return; }
    if (!pipelinePathToUse) { setError('Model pipeline path is missing.'); return; }
     // Clear previous results for this section
    setFairnessCalculationResults(null);
    setShapResults(null);
    setExplanationResults(null);

    const result = await handleApiCall(
      `Generating explanation for ${modelNameForTitle} (Samples: ${shapNSamples})...`,
      api.explainModel,
      uploadedFilename, pipelinePathToUse, shapNSamples // Pass shapNSamples
    );
    if (result) {
      // Heuristic to check if it's SHAP data for ShapChart, otherwise use GenericResultsDisplay
      // The ShapChart itself has normalization, so passing the raw result might be okay.
      // Common SHAP outputs might include 'shap_values', 'feature_names', 'base_value', or an array of {feature, value}
      // For this example, we assume if it's not an empty object and has some common shap keys or is an array.
      // A more robust check might be needed based on actual backend output structure.
      if (result && (result.shap_values || result.feature_names || result.base_value || Array.isArray(result.features) || (Array.isArray(result) && result.length > 0 && (result[0].feature || result[0].name)))) {
        setShapResults({ title: `SHAP Explanation: ${modelNameForTitle} (${pipelinePathToUse.split(/[\\/]/).pop()})`, data: result, forPipeline: pipelinePathToUse });
      } else {
        setExplanationResults({ title: `Model Explanation: ${modelNameForTitle} (${pipelinePathToUse.split(/[\\/]/).pop()})`, data: result, forPipeline: pipelinePathToUse });
      }
    }
  };
  
  const handleCompareModels = async () => {
    const baselinePipelinePath = baselineModelResults?.model_info?.pipeline_path;
    const mitigatedModels = [reweighedModelResults, oversampledModelResults].filter(Boolean);
    
    if (!baselinePipelinePath) {
        setError('Baseline model not trained or results not available.');
        return;
    }
    if (mitigatedModels.length === 0) {
        setError('No mitigated models trained or results not available for comparison.');
        return;
    }
    // For simplicity, this example compares with the first available mitigated model.
    // A more complex UI could allow choosing which mitigated model to compare.
    const mitigatedPipelinePath = mitigatedModels[0]?.model_info?.pipeline_path;

    if (!mitigatedPipelinePath) {
      setError('Mitigated model pipeline path not found. Ensure a mitigated model was trained successfully.');
      return;
    }

    const result = await handleApiCall('Comparing models...', api.compareModels, baselinePipelinePath, mitigatedPipelinePath, uploadedFilename, targetColumn, sensitiveAttrs);
    if (result) {
        setComparisonResults(result); 
    }
  };

  const getModelName = (modelResults) => {
    if (!modelResults || !modelResults.model_info || !modelResults.model_info.pipeline_path) return "Unknown Model";
    const { pipeline_path, mitigation_applied, mitigation_target_attribute } = modelResults.model_info;
    const fileName = pipeline_path.split(/[\\/]/).pop();
    
    if (mitigation_applied && mitigation_applied !== "None" && mitigation_applied !== "baseline") {
        let name = mitigation_applied.charAt(0).toUpperCase() + mitigation_applied.slice(1);
        if (mitigation_target_attribute) {
            name += ` (by ${mitigation_target_attribute})`;
        }
        return name;
    }
    if (fileName.toLowerCase().includes("baseline")) return "Baseline";
    return fileName;
  };

  const getTrainedModelInfoForDisplay = (modelResults, modelName) => {
    if (!modelResults || !modelResults.model_info) return null;
    return {
        name: modelName,
        pipelinePath: modelResults.model_info.pipeline_path,
        // Include other details if needed by the select dropdown or UI
        metrics: modelResults.metrics // Useful for quick reference if displayed in selection
    };
  }

  const trainedModelDetailsForSelection = [
    baselineModelResults && getTrainedModelInfoForDisplay(baselineModelResults, getModelName(baselineModelResults)),
    reweighedModelResults && getTrainedModelInfoForDisplay(reweighedModelResults, getModelName(reweighedModelResults)),
    oversampledModelResults && getTrainedModelInfoForDisplay(oversampledModelResults, getModelName(oversampledModelResults)),
  ].filter(Boolean);

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
                    disabled={isLoading || !uploadedFilename}
                    className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading && currentAction.startsWith('Analyzing') ? 'Analyzing...' : 'Analyze Data'}
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
                  {baselineModelResults && (
                    <motion.div initial={{ opacity: 0, y:10 }} animate={{ opacity: 1, y:0 }} className="mt-4">
                      <GenericResultsDisplay data={baselineModelResults} title={getModelName(baselineModelResults) + " - Full Results"} />
                    </motion.div>
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
                      {reweighedModelResults && (
                        <motion.div initial={{ opacity: 0, y:10 }} animate={{ opacity: 1, y:0 }} className="mt-4">
                          <GenericResultsDisplay data={reweighedModelResults} title={getModelName(reweighedModelResults) + " - Full Results"} />
                        </motion.div>
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
                      {oversampledModelResults && (
                        <motion.div initial={{ opacity: 0, y:10 }} animate={{ opacity: 1, y:0 }} className="mt-4">
                          <GenericResultsDisplay data={oversampledModelResults} title={getModelName(oversampledModelResults) + " - Full Results"} />
                        </motion.div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.section>
          )}

          {/* Section 3: Evaluate & Explain Individual Models */}
          {trainedModelDetailsForSelection.length > 0 && (
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
                    // Clear previous results when model selection changes
                    setFairnessCalculationResults(null);
                    setShapResults(null);
                    setExplanationResults(null);
                  }}
                  disabled={trainedModelDetailsForSelection.length === 0}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100"
                >
                  <option value="">Select a model...</option>
                  {trainedModelDetailsForSelection.map((model) => (
                    <option key={model.pipelinePath} value={model.pipelinePath}>
                      {model.name} ({model.pipelinePath.split(/[\\/]/).pop()})
                    </option>
                  ))}
                </select>

                {selectedModelForEval && (
                  <div className="space-y-4 mt-4">
                    <div className="flex items-center space-x-4">
                      <button
                        onClick={() => {
                            const modelDetails = trainedModelDetailsForSelection.find(m => m.pipelinePath === selectedModelForEval);
                            const modelName = modelDetails ? modelDetails.name : "Selected Model";
                            handleCalculateFairness(selectedModelForEval, modelName);
                        }}
                        disabled={isLoading || !selectedModelForEval}
                        className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Calculate Fairness
                      </button>
                      <button
                        onClick={() => {
                            const modelDetails = trainedModelDetailsForSelection.find(m => m.pipelinePath === selectedModelForEval);
                            const modelName = modelDetails ? modelDetails.name : "Selected Model";
                            handleExplainModel(selectedModelForEval, modelName);
                        }}
                        disabled={isLoading || !selectedModelForEval}
                        className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Generate SHAP
                      </button>
                    </div>
                    <div>
                      <label htmlFor="shapSamples" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        SHAP Samples (for Generate SHAP):
                      </label>
                      <input
                        type="number"
                        id="shapSamples"
                        value={shapNSamples}
                        onChange={(e) => setShapNSamples(Math.max(1, parseInt(e.target.value, 10) || 100))}
                        className="mt-1 w-full md:w-1/2 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-gray-100"
                        placeholder="e.g., 100"
                      />
                    </div>
                  </div>
                )}
              </div>

              <AnimatePresence>
                {fairnessCalculationResults && fairnessCalculationResults.data && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <GenericResultsDisplay
                      title={fairnessCalculationResults.title}
                      data={fairnessCalculationResults.data}
                    />
                  </motion.div>
                )}
                
                {explanationResults && explanationResults.data && (
                   <motion.div
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <GenericResultsDisplay
                      title={explanationResults.title}
                      data={explanationResults.data}
                    />
                  </motion.div>
                )}

                {shapResults && shapResults.data && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <ShapChart
                      title={shapResults.title}
                      data={shapResults.data} // ShapChart expects the full data object from explain_model
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>
          )}

          {/* Section 4: Compare Models */}
          {trainedModelDetailsForSelection.length >= 2 && (
            <motion.section 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
            >
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6">4. Compare Models</h2>
              
              <button
                onClick={handleCompareModels}
                disabled={isLoading || !baselineModelResults || !(reweighedModelResults || oversampledModelResults)}
                className="w-full px-4 py-2 bg-teal-600 text-white rounded-md hover:bg-teal-700 dark:bg-teal-700 dark:hover:bg-teal-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Compare Models
              </button>

              <AnimatePresence>
                {comparisonResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <GenericResultsDisplay data={comparisonResults} title="Model Comparison Results" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;