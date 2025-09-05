import React, { useState, useRef } from "react";
import "./App.css";

const BACKEND_URL = "https://box-detection-app-3.onrender.com";

function App() {
  const [output, setOutput] = useState("Output will appear here...");
  const wsRef = useRef(null);

  const log = (msg) => {
    setOutput((prev) => prev + "\n" + msg);
  };

  const pingBackend = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/ping`);
      const data = await res.json();
      log("🔎 Ping response: " + JSON.stringify(data));
    } catch (err) {
      log("❌ Error: " + err.message);
    }
  };

  const connectWebSocket = () => {
    wsRef.current = new WebSocket(BACKEND_URL.replace("https", "wss") + "/ws");
    wsRef.current.onopen = () => log("✅ WebSocket connected!");
    wsRef.current.onmessage = (event) =>
      log("📩 WS message: " + event.data);
    wsRef.current.onclose = () => log("❌ WebSocket closed");
  };

  const sendMessage = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("Hello from frontend!");
      log("➡️ Sent: Hello from frontend!");
    } else {
      log("⚠️ WebSocket not connected.");
    }
  };

  return (
    <div className="App">
      <h2>📦 Backend Communication Test</h2>
      <div className="button-group">
        <button onClick={pingBackend}>Ping Backend</button>
        <button onClick={connectWebSocket}>Connect WebSocket</button>
        <button onClick={sendMessage}>Send WS Message</button>
      </div>
      <pre className="output">{output}</pre>
    </div>
  );
}

export default App;
