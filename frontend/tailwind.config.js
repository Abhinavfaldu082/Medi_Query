// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  // Note: v3 uses module.exports, v4 might use ES module syntax
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // This covers common React file types
  ],
  theme: {
    extend: {
      // Optional: Add custom animations or other theme extensions here
      animation: {
        fadeIn: "fadeIn 0.5s ease-in-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
