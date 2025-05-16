import React, { useState } from 'react';
import { motion } from 'framer-motion';

const MetricsTable = ({ title, data }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  
  if (!data) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-6"
      >
        <div className="gradient-header">
          <h2 className="text-2xl font-bold mb-2">{title || "Model Metrics"}</h2>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <p className="text-gray-600 dark:text-gray-400 text-center">No metrics data available.</p>
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

  const getMetricColor = (value, type) => {
    if (type === 'accuracy' || type === 'precision' || type === 'recall' || type === 'f1') {
      if (value >= 0.8) return 'text-green-600 dark:text-green-400';
      if (value >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
      return 'text-red-600 dark:text-red-400';
    }
    return 'text-gray-900 dark:text-gray-100';
  };
  
  const renderHeader = () => (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="gradient-header"
    >
      <h2 className="text-2xl font-bold mb-2">{title || "Model Metrics"}</h2>
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
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Raw Metrics Data</h3>
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96">
          <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </motion.div>
    );
  };

  const renderMetricCard = (label, value, type = 'default') => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
          {label}
        </h3>
        <div className="flex items-center justify-between">
          <p className={`text-2xl font-bold ${getMetricColor(value, type)}`}>
            {formatValue(value)}
          </p>
          {type !== 'default' && (
            <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
              <div
                className={`h-2 rounded-full ${
                  type === 'accuracy' || type === 'precision' || type === 'recall' || type === 'f1'
                    ? value >= 0.8
                      ? 'bg-green-500'
                      : value >= 0.6
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                    : 'bg-blue-500'
                }`}
                style={{ width: `${Math.min(value * 100, 100)}%` }}
              />
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  const renderConfusionMatrix = (matrix) => {
    if (!matrix) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">
          Confusion Matrix
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">True Negatives</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">
              {matrix.true_negatives || matrix.tn || 0}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">False Positives</p>
            <p className="text-2xl font-bold text-red-600 dark:text-red-400">
              {matrix.false_positives || matrix.fp || 0}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">False Negatives</p>
            <p className="text-2xl font-bold text-red-600 dark:text-red-400">
              {matrix.false_negatives || matrix.fn || 0}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">True Positives</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">
              {matrix.true_positives || matrix.tp || 0}
            </p>
          </div>
        </div>
      </motion.div>
    );
  };
  
  const renderModelComparison = () => {
    // Check if this is a model comparison with baseline and mitigated models
    if (!data.baseline || !data.mitigated) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-6">Model Comparison</h3>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Metric
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Baseline
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Mitigated
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Difference
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {['accuracy', 'precision', 'recall', 'f1_score'].map(metric => (
                <tr key={metric}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                    {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {formatValue(data.baseline[metric])}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {formatValue(data.mitigated[metric])}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                    data.mitigated[metric] > data.baseline[metric] 
                      ? 'text-green-600 dark:text-green-400' 
                      : data.mitigated[metric] < data.baseline[metric]
                        ? 'text-red-600 dark:text-red-400'
                        : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {data.mitigated[metric] > data.baseline[metric] ? '+' : ''}
                    {formatValue(data.mitigated[metric] - data.baseline[metric])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    );
  };
  
  const renderFairnessComparison = () => {
    // Check if there's fairness metrics for comparison
    if (!data.fairness_comparison && 
        !(data.baseline && data.mitigated && 
          (data.baseline.disparities || data.mitigated.disparities))) {
      return null;
    }
    
    const fairnessData = data.fairness_comparison || {
      baseline: data.baseline.disparities || {},
      mitigated: data.mitigated.disparities || {}
    };
    
    const metrics = Object.keys(fairnessData.baseline || {});
    
    if (metrics.length === 0) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-6">Fairness Comparison</h3>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Metric
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Baseline
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Mitigated
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Improvement
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {metrics.map(metric => {
                const baselineValue = fairnessData.baseline[metric];
                const mitigatedValue = fairnessData.mitigated[metric];
                // For fairness metrics, lower is usually better
                const improvement = Math.abs(baselineValue) - Math.abs(mitigatedValue);
                
                return (
                  <tr key={metric}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                      {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatValue(baselineValue)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatValue(mitigatedValue)}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                      improvement > 0 
                        ? 'text-green-600 dark:text-green-400' 
                        : improvement < 0
                          ? 'text-red-600 dark:text-red-400'
                          : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {improvement > 0 ? '+' : ''}
                      {formatValue(improvement)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
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
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">
          Detailed Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(metrics).map(([key, value]) => (
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
      
      {/* Handle model comparison data */}
      {renderModelComparison()}
      {renderFairnessComparison()}
      
      {/* Handle regular metrics */}
      {data.metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {data.metrics.accuracy !== undefined && renderMetricCard('Accuracy', data.metrics.accuracy, 'accuracy')}
          {data.metrics.precision !== undefined && renderMetricCard('Precision', data.metrics.precision, 'precision')}
          {data.metrics.recall !== undefined && renderMetricCard('Recall', data.metrics.recall, 'recall')}
          {data.metrics.f1_score !== undefined && renderMetricCard('F1 Score', data.metrics.f1_score, 'f1')}
        </div>
      )}
      
      {/* Handle directly available metrics */}
      {!data.metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {data.accuracy !== undefined && renderMetricCard('Accuracy', data.accuracy, 'accuracy')}
          {data.precision !== undefined && renderMetricCard('Precision', data.precision, 'precision')}
          {data.recall !== undefined && renderMetricCard('Recall', data.recall, 'recall')}
          {data.f1_score !== undefined && renderMetricCard('F1 Score', data.f1_score, 'f1')}
        </div>
      )}

      {data.confusion_matrix && renderConfusionMatrix(data.confusion_matrix)}
      {data.detailed_metrics && renderDetailedMetrics(data.detailed_metrics)}
    </motion.div>
  );
};

export default MetricsTable;