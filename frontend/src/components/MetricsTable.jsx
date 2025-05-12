import React from 'react';
import './MetricsTable.css'; 

const MetricsTable = ({ metrics, title }) => {
  if (!metrics || Object.keys(metrics).length === 0) {
    return null; // Don't render if no metrics
  }

  // Convert metric names to a more readable format
  const formatMetricName = (name) => {
    return name
      .replace(/_/g, ' ') // Replace underscores with spaces
      .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize first letter of each word
  };

  return (
    <div className="metrics-table-container">
      <h4>{title || 'Overall Performance Metrics'}</h4>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([key, value]) => (
            <tr key={key}>
              <td>{formatMetricName(key)}</td>
              <td>{typeof value === 'number' ? value.toFixed(4) : String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MetricsTable;