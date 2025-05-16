import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

const ShapChart = ({ data, title }) => {
  const svgRef = useRef(null);
  const [hasError, setHasError] = useState(false);
  const [showRawData, setShowRawData] = useState(false);
  
  // Normalize the data to ensure we have a consistent format
  const normalizeData = (inputData) => {
    // If it's already an array, check if items have feature and value properties
    if (Array.isArray(inputData)) {
      // Check if the array items have the expected format
      if (inputData.length > 0 && (inputData[0].feature || inputData[0].name) && 'value' in inputData[0]) {
        return inputData.map(item => ({
          feature: item.feature || item.name,
          value: item.value
        }));
      }
    }
    
    // If it's an object, it might be in various formats
    if (inputData && typeof inputData === 'object') {
      // Check if it has features property that is an array
      if (Array.isArray(inputData.features)) {
        return normalizeData(inputData.features);
      }
      
      // Check if it has a values property with feature names as keys
      if (inputData.values && typeof inputData.values === 'object') {
        return Object.entries(inputData.values).map(([feature, value]) => ({
          feature,
          value: typeof value === 'object' ? value.value || 0 : value
        }));
      }
      
      // Check if the object itself has feature names as keys
      if (Object.keys(inputData).length > 0 && typeof inputData[Object.keys(inputData)[0]] === 'number') {
        return Object.entries(inputData).map(([feature, value]) => ({
          feature,
          value
        }));
      }
    }
    
    // If we reach this point, we couldn't recognize the format
    return [];
  };
  
  const normalizedData = normalizeData(data);
  
  useEffect(() => {
    if (!normalizedData || normalizedData.length === 0 || !svgRef.current) {
      setHasError(true);
      return;
    }
    
    setHasError(false);
    
    try {
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();

      const margin = { top: 40, right: 20, bottom: 60, left: 120 };
      const width = svgRef.current.clientWidth - margin.left - margin.right;
      const height = Math.max(400, normalizedData.length * 30) - margin.top - margin.bottom;

      const g = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // Sort data by absolute SHAP value
      const sortedData = [...normalizedData].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
      // Take only top 15 features if there are too many
      const displayData = sortedData.slice(0, 15);

      // Create scales
      const xExtent = d3.extent(displayData, d => d.value);
      const absMax = Math.max(Math.abs(xExtent[0]), Math.abs(xExtent[1]));
      const x = d3
        .scaleLinear()
        .domain([-absMax, absMax])
        .range([0, width])
        .nice();

      const y = d3
        .scaleBand()
        .domain(displayData.map(d => d.feature))
        .range([0, height])
        .padding(0.3);

      // Add axes
      const xAxis = g
        .append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5));

      const yAxis = g
        .append('g')
        .call(d3.axisLeft(y));

      // Style axes
      xAxis.selectAll('text')
        .style('fill', 'currentColor')
        .style('font-size', '12px');

      yAxis.selectAll('text')
        .style('fill', 'currentColor')
        .style('font-size', '12px');

      // Add grid lines
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
        )
        .style('stroke', 'currentColor')
        .style('stroke-opacity', 0.1);
        
      // Add vertical line at x=0
      g.append('line')
        .attr('x1', x(0))
        .attr('x2', x(0))
        .attr('y1', 0)
        .attr('y2', height)
        .style('stroke', 'currentColor')
        .style('stroke-opacity', 0.5)
        .style('stroke-dasharray', '4');

      // Add bars
      g.selectAll('.bar')
        .data(displayData)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => x(Math.min(0, d.value)))
        .attr('y', d => y(d.feature))
        .attr('width', d => Math.abs(x(d.value) - x(0)))
        .attr('height', y.bandwidth())
        .attr('fill', d => d.value > 0 ? '#10B981' : '#EF4444')
        .attr('opacity', 0.8)
        .on('mouseover', function(event, d) {
          d3.select(this)
            .attr('opacity', 1)
            .attr('stroke', 'currentColor')
            .attr('stroke-width', 2);

          // Add tooltip
          const tooltip = g.append('g')
            .attr('class', 'tooltip')
            .style('pointer-events', 'none');

          tooltip.append('rect')
            .attr('x', x(d.value) + (d.value > 0 ? 10 : -130))
            .attr('y', y(d.feature) - 20)
            .attr('width', 120)
            .attr('height', 40)
            .attr('fill', 'currentColor')
            .attr('opacity', 0.9)
            .attr('rx', 4);

          tooltip.append('text')
            .attr('x', x(d.value) + (d.value > 0 ? 20 : -120))
            .attr('y', y(d.feature))
            .attr('fill', 'white')
            .text(`Value: ${d.value.toFixed(4)}`);

          tooltip.append('text')
            .attr('x', x(d.value) + (d.value > 0 ? 20 : -120))
            .attr('y', y(d.feature) + 20)
            .attr('fill', 'white')
            .text(`Feature: ${d.feature}`);
        })
        .on('mouseout', function() {
          d3.select(this)
            .attr('opacity', 0.8)
            .attr('stroke', 'none');

          // Remove tooltip
          g.selectAll('.tooltip').remove();
        });

      // Add title
      svg.append('text')
        .attr('x', width / 2 + margin.left)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .style('fill', 'currentColor')
        .text(title || 'SHAP Values');
        
      // Add impact labels
      svg.append('text')
        .attr('x', margin.left + width - 50)
        .attr('y', margin.top - 10)
        .attr('text-anchor', 'end')
        .style('font-size', '12px')
        .style('fill', 'currentColor')
        .text('Increases prediction');
        
      svg.append('text')
        .attr('x', margin.left + 50)
        .attr('y', margin.top - 10)
        .attr('text-anchor', 'start')
        .style('font-size', '12px')
        .style('fill', 'currentColor')
        .text('Decreases prediction');
        
    } catch (error) {
      console.error("Error rendering SHAP chart:", error);
      setHasError(true);
    }
  }, [normalizedData, title]);

  const renderRawDataView = () => {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96"
      >
        <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
          {JSON.stringify(data, null, 2)}
        </pre>
      </motion.div>
    );
  };

  if (!data || (Array.isArray(normalizedData) && normalizedData.length === 0) || hasError) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
      >
        <div className="text-center py-8">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">
            {title || 'SHAP Values'}
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            No feature data available for visualization. Please ensure the model has been properly explained.
          </p>
          {data && (
            <button
              onClick={() => setShowRawData(!showRawData)}
              className="mt-4 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800"
            >
              {showRawData ? "Hide Raw Data" : "Show Raw Data"}
            </button>
          )}
          {showRawData && data && renderRawDataView()}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
          {title || 'SHAP Values'}
        </h3>
        <button
          onClick={() => setShowRawData(!showRawData)}
          className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-all"
        >
          {showRawData ? "Hide Raw Data" : "Show Raw Data"}
        </button>
      </div>
      
      <div className="w-full overflow-x-auto">
        <svg
          ref={svgRef}
          width="100%"
          height={Math.max(400, normalizedData.length * 30)}
          className="text-gray-900 dark:text-gray-100"
        />
      </div>
      
      <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
        <p>Showing top {Math.min(normalizedData.length, 15)} features by importance. Positive values (green) increase the prediction, negative values (red) decrease it.</p>
      </div>
      
      {showRawData && renderRawDataView()}
    </motion.div>
  );
};

export default ShapChart;