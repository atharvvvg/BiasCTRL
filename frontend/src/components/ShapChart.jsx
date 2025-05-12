import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ShapChart.css';

const ShapChart = ({ shapData, title }) => {
  if (!shapData || !shapData.global_feature_importance || Object.keys(shapData.global_feature_importance).length === 0) {
    return null;
  }

  const importanceData = Object.entries(shapData.global_feature_importance)
    .map(([name, value]) => ({ name, importance: value }))
    .sort((a, b) => b.importance - a.importance) // Sort by importance
    .slice(0, 20); // Display top 20 features for readability

  return (
    <div className="shap-chart-container">
      <h4>{title || 'Global Feature Importance (SHAP)'}</h4>
      <ResponsiveContainer width="100%" height={400 + importanceData.length * 10}> {/* Adjust height dynamically */}
        <BarChart
          data={importanceData}
          layout="vertical" // Horizontal bars
          margin={{
            top: 5,
            right: 30,
            left: 150, // Increase left margin for long feature names
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={150} interval={0} /> {/* Show all labels */}
          <Tooltip formatter={(value) => value.toFixed(4)} />
          <Legend />
          <Bar dataKey="importance" fill="#82ca9d" barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ShapChart;