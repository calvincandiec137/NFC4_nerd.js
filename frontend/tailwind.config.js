/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#9b8b6f',
        secondary: '#e8dcc8',
        accent: '#6ba549',
        text: '#333333',
        'text-light': '#666666',
        'bg-light': '#f5f3f0',
        'bg-white': '#ffffff',
        'bg-tan': '#d9cfc2',
      }
    },
  },
  plugins: [],
}
