import React, { useState } from 'react';
import { motion } from 'framer-motion';

const AnalysisDisplay = ({ analysisData, filename, data }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  
  // Use either the specific analysisData prop or the general data prop
  const displayData = analysisData || (data && data.analysis) || data;
  
  if (!displayData) return null;

  const formatValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(2);
    }
    return value;
  };

  const formatPercentage = (value) => {
    if (typeof value === 'number') {
      return `${(value * 100).toFixed(1)}%`;
    }
    return value;
  };

  const renderHeader = () => (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="gradient-header"
    >
      <h2 className="text-2xl font-bold mb-2">Analysis Results</h2>
      <p className="text-blue-100">File: {displayData.filename || filename}</p>
      {displayData.target_cleaned && (
        <p className="text-blue-100 mt-2">{displayData.target_cleaned}</p>
      )}
      <div className="mt-4 flex justify-end">
        <button
          onClick={() => setShowRawJson(!showRawJson)}
          className="px-3 py-1 bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-300 text-sm rounded-md hover:bg-opacity-90 transition-all"
        >
          {showRawJson ? "Hide Raw Data" : "Show Raw Data"}
        </button>
      </div>
    </motion.div>
  );

  const renderDatasetOverview = () => {
    if (!displayData.row_count && !displayData.column_names) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Dataset Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {displayData.row_count && (
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Total Rows</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{displayData.row_count.toLocaleString()}</p>
            </div>
          )}
          
          {displayData.column_names && (
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Total Columns</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{displayData.column_names.length}</p>
            </div>
          )}
          
          {displayData.target_distribution && (
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Target Distribution</p>
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Class 0 (Negative)</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {formatPercentage(displayData.target_distribution[0] / displayData.row_count)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Class 1 (Positive)</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {formatPercentage(displayData.target_distribution[1] / displayData.row_count)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  const renderSensitiveAttributes = () => {
    if (!displayData.sensitive_attributes) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Sensitive Attributes Analysis</h3>
        <div className="space-y-8">
          {Object.entries(displayData.sensitive_attributes).map(([attribute, data]) => (
            <div key={attribute} className="space-y-4">
              <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300">
                {attribute.charAt(0).toUpperCase() + attribute.slice(1)}
              </h4>
              
              {/* Distribution */}
              {data.distribution && (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
                  <h5 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Distribution</h5>
                  <div className="space-y-3">
                    {Object.entries(data.distribution).map(([value, count]) => (
                      <div key={value} className="space-y-1">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
                          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {formatPercentage(count / displayData.row_count)}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${(count / displayData.row_count) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Target Distribution by Group */}
              {data.target_distribution_by_group && (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
                  <h5 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Target Distribution by Group</h5>
                  <div className="space-y-4">
                    {Object.entries(data.target_distribution_by_group).map(([targetClass, distributions]) => (
                      <div key={targetClass} className="space-y-2">
                        <h6 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                          Class {targetClass} ({targetClass === '0' ? 'Negative' : 'Positive'})
                        </h6>
                        <div className="space-y-2">
                          {Object.entries(distributions).map(([group, percentage]) => (
                            <div key={group} className="space-y-1">
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600 dark:text-gray-400">{group}</span>
                                <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                  {formatPercentage(percentage)}
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${targetClass === '0' ? 'bg-red-500' : 'bg-green-500'}`}
                                  style={{ width: `${percentage * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  const renderColumnList = () => {
    if (!displayData.column_names) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Dataset Columns</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {displayData.column_names.map((column) => (
            <div
              key={column}
              className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600"
            >
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">{column}</p>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  const renderRawJson = () => {
    if (!showRawJson) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Raw Analysis Data</h3>
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96">
          <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {JSON.stringify(displayData, null, 2)}
          </pre>
        </div>
      </motion.div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {renderHeader()}
      {renderRawJson()}
      {renderDatasetOverview()}
      {renderSensitiveAttributes()}
      {renderColumnList()}
    </motion.div>
  );
};

export default AnalysisDisplay;