import React, { useState } from 'react';
import { motion } from 'framer-motion';

const FairnessDisplay = ({ title, data }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  
  if (!data) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-6"
      >
        <div className="gradient-header">
          <h2 className="text-2xl font-bold mb-2">{title || "Fairness Results"}</h2>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <p className="text-gray-600 dark:text-gray-400 text-center">No fairness data available.</p>
        </div>
      </motion.div>
    );
  }

  const formatValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  const getFairnessStatus = (value, threshold = 0.1) => {
    if (typeof value !== 'number') return 'unknown';
    if (value <= threshold) return 'good';
    if (value <= threshold * 1.5) return 'warning';
    return 'bad';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'good':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'bad':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
    }
  };

  const renderHeader = () => (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="gradient-header"
    >
      <h2 className="text-2xl font-bold mb-2">{title || "Fairness Results"}</h2>
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

  const renderRawJson = () => {
    if (!showRawJson) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Raw Fairness Data</h3>
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96">
          <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </motion.div>
    );
  };

  const renderMetricSummary = (metrics) => {
    if (!metrics) return null;
    
    // First look for top-level metrics like disparate_impact, statistical_parity_difference, etc.
    const topMetrics = {};
    Object.entries(data).forEach(([key, value]) => {
      if (typeof value === 'number' && !['row_count', 'column_count'].includes(key)) {
        topMetrics[key] = value;
      }
    });
    
    if (Object.keys(topMetrics).length === 0) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Fairness Metrics Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(topMetrics).map(([metric, value]) => {
            const status = getFairnessStatus(value);
            return (
              <div
                key={metric}
                className={`rounded-lg p-4 border ${getStatusColor(status)}`}
              >
                <p className="text-sm font-medium mb-1">
                  {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </p>
                <p className="text-lg font-semibold">
                  {formatValue(value)}
                </p>
                {typeof value === 'number' && (
                  <>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            status === 'good' ? 'bg-green-500' :
                            status === 'warning' ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                    <p className="text-xs mt-2">
                      {status === 'good' ? 'Fair' : status === 'warning' ? 'Moderate Bias' : 'Significant Bias'}
                    </p>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </motion.div>
    );
  };

  const renderDisparities = (disparities) => {
    if (!disparities) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Fairness Disparities</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(disparities).map(([metric, value]) => {
            const status = getFairnessStatus(value);
            return (
              <div
                key={metric}
                className={`rounded-lg p-4 border ${getStatusColor(status)}`}
              >
                <p className="text-sm font-medium mb-1">
                  {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </p>
                <p className="text-lg font-semibold">
                  {formatValue(value)}
                </p>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        status === 'good' ? 'bg-green-500' :
                        status === 'warning' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <p className="text-xs mt-2">
                  {status === 'good' ? 'Fair' : status === 'warning' ? 'Moderate Bias' : 'Significant Bias'}
                </p>
              </div>
            );
          })}
        </div>
      </motion.div>
    );
  };

  const renderGroupMetrics = (groupMetrics) => {
    if (!groupMetrics) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Metrics by Group</h3>
        <div className="space-y-6">
          {Object.entries(groupMetrics).map(([group, metrics]) => (
            <div key={group} className="space-y-4">
              <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700 pb-2">
                {group.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {Object.entries(metrics).map(([metric, value]) => (
                  <div
                    key={`${group}-${metric}`}
                    className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600"
                  >
                    <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                      {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                      {formatValue(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  const renderDefinitions = (definitions) => {
    if (!definitions) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Fairness Definitions</h3>
        <div className="space-y-4">
          {Object.entries(definitions).map(([metric, definition]) => (
            <div
              key={metric}
              className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600"
            >
              <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </h4>
              <p className="text-gray-600 dark:text-gray-400">{definition}</p>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  const renderDetailedMetrics = (metrics) => {
    if (!metrics) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Detailed Metrics</h3>
        <div className="space-y-6">
          {Object.entries(metrics).map(([category, values]) => (
            <div key={category} className="space-y-4">
              <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300">
                {category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(values).map(([key, value]) => (
                  <div
                    key={key}
                    className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600"
                  >
                    <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                      {formatValue(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };
  
  // Look for group metrics in the data (certain APIs might structure them differently)
  const groupMetricsKeys = Object.keys(data).filter(key => 
    key.includes('by_group') || 
    key.includes('group_metrics') || 
    (typeof data[key] === 'object' && data[key] !== null && 
     Object.keys(data[key]).length > 0 && 
     typeof data[key][Object.keys(data[key])[0]] === 'object')
  );
  
  const groupMetrics = groupMetricsKeys.length > 0 ? 
    data[groupMetricsKeys[0]] : 
    (data.group_metrics || null);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {renderHeader()}
      {renderRawJson()}
      {renderMetricSummary(data)}
      {data.disparities && renderDisparities(data.disparities)}
      {groupMetrics && renderGroupMetrics(groupMetrics)}
      {data.detailed_metrics && renderDetailedMetrics(data.detailed_metrics)}
      {data.standard_definitions && renderDefinitions(data.standard_definitions)}
    </motion.div>
  );
};

export default FairnessDisplay;