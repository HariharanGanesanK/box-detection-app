import React, { useState, useEffect } from 'react';

// The CSS is embedded directly into the component.
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

  :root {
    --background-color: #f0f2f5;
    --card-background: #ffffff;
    --primary-color: #007bff;
    --primary-hover: #0056b3;
    --text-color: #333;
    --border-color: #dee2e6;
    --success-color: #28a745;
    --error-color: #dc3545;
  }

  body {
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-color: var(--background-color);
    color: var(--text-color);
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
  }

  .container {
    background-color: var(--card-background);
    padding: 2rem 3rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    text-align: center;
    max-width: 500px;
    width: 90%;
  }

  header h1 {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
  }

  header p {
    color: #6c757d;
    margin-bottom: 2rem;
  }

  button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease-in-out, transform 0.1s ease;
  }

  button:hover {
    background-color: var(--primary-hover);
    transform: translateY(-2px);
  }

  button:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  .message {
    margin-top: 1.5rem;
    padding: 0.8rem;
    border-radius: 8px;
    font-weight: 500;
  }

  .success {
    background-color: #e9f7ef;
    color: var(--success-color);
    border: 1px solid var(--success-color);
  }

  .error {
    background-color: #fdeaea;
    color: var(--error-color);
    border: 1px solid var(--error-color);
  }

  footer {
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #6c757d;
  }
`;

function App() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // This useEffect hook injects the styles into the document's head
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.innerHTML = globalStyles;
    document.head.appendChild(styleElement);

    // Cleanup function to remove the style when the component unmounts
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);


  const fetchMessage = async () => {
    setIsLoading(true);
    setError(null);
    setMessage('');
    
    // This logic handles both deployment and local development.
    // On Render, `import.meta.env.VITE_BACKEND_URL` will be set.
    // Locally, it will be undefined, so it falls back to the local server address.
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
