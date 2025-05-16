import React, { useState } from 'react';
import { motion } from 'framer-motion';

const renderValue = (value, keyPrefix) => {
  if (typeof value === 'object' && value !== null) {
    if (Array.isArray(value)) {
      return (
        <div className="ml-4 pl-4 border-l border-gray-200 dark:border-gray-600">
          {value.map((item, index) => (
            <div key={`${keyPrefix}-${index}`} className="mb-1">
              <span className="text-sm font-medium text-indigo-500 dark:text-indigo-400">{index}: </span>
              {renderValue(item, `${keyPrefix}-${index}`)}
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className="ml-4 pl-4 border-l border-gray-200 dark:border-gray-600">
        {Object.entries(value).map(([key, val]) => (
          <div key={`${keyPrefix}-${key}`} className="mb-2">
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">{key}:</span>
            <div className="ml-2">{renderValue(val, `${keyPrefix}-${key}`)}</div>
          </div>
        ))}
      </div>
    );
  }
  return <span className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{String(value)}</span>;
};

const GenericResultsDisplay = ({ data, title }) => {
  const [showRawJson, setShowRawJson] = useState(false);

  if (!data) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 space-y-4 mb-6"
    >
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">{title}</h3>
        <button
          onClick={() => setShowRawJson(!showRawJson)}
          aria-label={showRawJson ? "Hide raw JSON data" : "Show raw JSON data"}
          tabIndex="0"
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowRawJson(!showRawJson); }}
          className="px-3 py-1 bg-blue-500 hover:bg-blue-600 dark:bg-indigo-600 dark:hover:bg-indigo-700 text-white text-sm rounded-md transition-all focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-indigo-400"
        >
          {showRawJson ? 'Hide Raw JSON' : 'Show Raw JSON'}
        </button>
      </div>

      {showRawJson ? (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 overflow-auto max-h-96">
          <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      ) : (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 space-y-3">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="mb-2">
              <span className="text-md font-semibold text-gray-700 dark:text-gray-300 capitalize">{key.replace(/_/g, ' ')}:</span>
              <div className="ml-2 mt-1">{renderValue(value, key)}</div>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default GenericResultsDisplay; 