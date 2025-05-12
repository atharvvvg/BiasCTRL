import React from 'react';
import './FairnessDisplay.css'; // Basic CSS

const FairnessDisplay = ({ fairnessData, sensitiveAttribute }) => {
  if (!fairnessData || (!fairnessData.disparities && !fairnessData.standard_definitions)) {
    return <p>No fairness data to display for {sensitiveAttribute}.</p>;
  }

  const { disparities, standard_definitions, by_group } = fairnessData;

  const formatKey = (key) => {
    return key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase())
      .replace(/\(.*\)/, match => `<small>${match}</small>`); // Smaller text for parenthetical
  };

  return (
    <div className="fairness-display">
      <h5>Fairness Metrics for "{sensitiveAttribute}"</h5>
      {disparities && (
        <div className="fairness-section">
          <h6>Key Disparities (Between Groups)</h6>
          <ul>
            {Object.entries(disparities).map(([key, value]) => (
              <li key={key}>
                <span dangerouslySetInnerHTML={{ __html: formatKey(key) }} />:
                <strong> {typeof value === 'number' ? value.toFixed(4) : String(value)}</strong>
              </li>
            ))}
          </ul>
        </div>
      )}
      {standard_definitions && (
        <div className="fairness-section">
          <h6>Standard Fairness Definitions</h6>
          <ul>
            {Object.entries(standard_definitions).map(([key, value]) => (
              <li key={key}>
                <span dangerouslySetInnerHTML={{ __html: formatKey(key) }} />:
                <strong> {typeof value === 'number' ? value.toFixed(4) : String(value)}</strong>
              </li>
            ))}
          </ul>
        </div>
      )}
       {/* Optionally, display by_group data here or in a separate component */}
       {by_group && (
         <div className="fairness-section by-group-details">
            <h6>Metrics By Group (Details)</h6>
            {/* You might want a more structured table for by_group */}
            <pre>{JSON.stringify(by_group, null, 2)}</pre>
         </div>
       )}
    </div>
  );
};

export default FairnessDisplay;