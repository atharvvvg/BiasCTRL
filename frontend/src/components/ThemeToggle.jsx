import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';

const ThemeToggle = () => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleTheme}
      className="fixed top-4 right-4 z-50 p-2 rounded-full bg-gray-200 dark:bg-gray-700 shadow-lg"
      aria-label="Toggle theme"
    >
      <div className="relative w-12 h-6">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
          initial={false}
          animate={{ opacity: isDarkMode ? 1 : 0 }}
        />
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full"
          initial={false}
          animate={{ opacity: isDarkMode ? 0 : 1 }}
        />
        <motion.div
          className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
          initial={false}
          animate={{ x: isDarkMode ? 24 : 0 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        >
          {isDarkMode ? (
            <MoonIcon className="w-4 h-4 text-gray-800" />
          ) : (
            <SunIcon className="w-4 h-4 text-yellow-500" />
          )}
        </motion.div>
      </div>
    </motion.button>
  );
};

export default ThemeToggle; 