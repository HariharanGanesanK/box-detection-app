import React, { useState, useRef } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const ws = useRef(null);

  // ✅ Use environment variable or fallback
  const API_URL =
    process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

  // ✅ WebSocket URL (auto converts http -> ws / https -> wss)
  const WS_URL = `${API_URL.replace("http", "ws")}/ws`;

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch(`${API_URL}/predict/`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  const connectWebSocket = () => {
    if (ws.current) ws.current.close();

    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      console.log("✅ WebSocket connected");
    };

    ws.current.onmessage = (event) => {
      console.log("📩 Message:", event.data);
    };

    ws.current.onclose = () => {
      console.log("❌ WebSocket closed");
    };

    ws.current.onerror = (error) => {
      console.error("⚠️ WebSocket error:", error);
    };
  };

  return (
    <div>
      <h1>📦 Box Detection App</h1>

      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Detect</button>
      <button onClick={connectWebSocket}>Connect WebSocket</button>

      {result && (
        <pre style={{ textAlign: "left" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default App;
