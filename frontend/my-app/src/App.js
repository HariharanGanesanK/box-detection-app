import React, { useState } from 'react';
import './App.css'; // Import the new CSS file

function App() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMessage = async () => {
    setIsLoading(true);
    setError(null);
    setMessage('');

    // This logic handles both deployment and local development.
    const apiUrl = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

    try {
      const response = await fetch(apiUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMessage(data.message);
    } catch (e) {
      console.error("Fetch error:", e);
      setError("Failed to connect to the backend. Make sure it's running and accessible.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>React + FastAPI on Render</h1>
        <p>A simple full-stack app to demonstrate deployment.</p>
      </header>
      <main>
        <button onClick={fetchMessage} disabled={isLoading}>
          {isLoading ? 'Connecting...' : 'Connect to Backend'}
        </button>
        {message && <p className="message success">{message}</p>}
        {error && <p className="message error">{error}</p>}
      </main>
      <footer>
        <p>Push to GitHub to automatically deploy changes.</p>
      </footer>
    </div>
  );
}

export default App;
