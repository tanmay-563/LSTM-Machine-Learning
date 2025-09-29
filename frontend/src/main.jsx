import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'     // <-- correct if App.jsx is in the same folder
import './index.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
