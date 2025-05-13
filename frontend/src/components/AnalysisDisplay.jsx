import React from 'react';
import './AnalysisDisplay.css';

const AnalysisDisplay = ({ analysisData, filename }) => {
  if (!analysisData || Object.keys(analysisData).length === 0) {
    return <p>No analysis data to display.</p>;
  }

  const {
    target_cleaned,
    row_count,
    column_names,
    target_distribution,
    sensitive_attributes,
  } = analysisData;

  return (
    <div className="analysis-display card"> 
      <h3>Data Analysis Overview for: {filename}</h3>

      {target_cleaned && <p className="analysis-note"><strong>Note on Target:</strong> {target_cleaned}</p>}

      <div className="analysis-section">
        <h4>Dataset Summary</h4>
        <ul>
          <li><strong>Total Rows:</strong> {row_count?.toLocaleString()}</li>
          <li><strong>Total Columns:</strong> {column_names?.length}</li>
        </ul>
      </div>

      {column_names && (
        <div className="analysis-section">
          <h4>Column Names ({column_names.length})</h4>
          <ul className="column-list">
            {column_names.map((col, index) => (
              <li key={index}>{col}</li>
            ))}
          </ul>
        </div>
      )}

      {target_distribution && (
        <div className="analysis-section">
          <h4>Target Variable Distribution</h4>
          <table>
            <thead>
              <tr>
                <th>Class Label</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(target_distribution).map(([label, count]) => (
                <tr key={label}>
                  <td>{label === "0" ? "Class 0 (e.g., <=50K)" : (label === "1" ? "Class 1 (e.g., >50K)" : label)}</td>
                  <td>{count?.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {sensitive_attributes && Object.keys(sensitive_attributes).length > 0 && (
        <div className="analysis-section">
          <h4>Sensitive Attributes Analysis</h4>
          {Object.entries(sensitive_attributes).map(([attrName, attrData]) => (
            <div key={attrName} className="sensitive-attribute-details">
              <h5>Sensitive Attribute: "{attrName}" (Unique Values: {attrData.unique_values})</h5>

              <h6>Distribution of "{attrName}":</h6>
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(attrData.distribution || {}).map(([group, count]) => (
                    <tr key={group}>
                      <td>{group}</td>
                      <td>{count?.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {attrData.target_distribution_by_group && (
                <div className="target-by-group">
                  <h6>Target Distribution within "{attrName}" Groups (Proportions):</h6>
                  {Object.entries(attrData.target_distribution_by_group["0"] || {}).map(([groupName]) => (
                    <div key={groupName} className="target-by-group-item">
                      <strong>Group: {groupName}</strong>
                      <ul>
                        <li>
                          Class 0:
                          <strong> {((attrData.target_distribution_by_group["0"]?.[groupName] || 0) * 100).toFixed(2)}%</strong>
                        </li>
                        <li>
                          Class 1:
                          <strong> {((attrData.target_distribution_by_group["1"]?.[groupName] || 0) * 100).toFixed(2)}%</strong>
                        </li>
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AnalysisDisplay;