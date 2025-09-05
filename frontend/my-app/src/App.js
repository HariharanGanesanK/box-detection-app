import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [videoPath, setVideoPath] = useState('');
  const [status, setStatus] = useState('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [counts, setCounts] = useState({});

  const [vehicles, setVehicles] = useState(['TN 66 J 0001', 'TN 66 J 0002', 'TN 66 J 0003']);
  const [supervisors, setSupervisors] = useState(['John', 'Hari', 'Chandra']);
  const [selectedVehicle, setSelectedVehicle] = useState(vehicles[0]);
  const [selectedSupervisor, setSelectedSupervisor] = useState(supervisors[0]);
  
  const [selectedModel, setSelectedModel] = useState('multi_box_counter');
  
  const ws = useRef(null);

  useEffect(() => {
    if (status === 'processing') {
      ws.current = new WebSocket(`${API_URL.replace('http', 'ws')}/ws`);
      ws.current.onopen = () => console.log("WebSocket connected");
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.counts) {
          setCounts(data.counts);
        }
      };
      ws.current.onclose = () => console.log("WebSocket disconnected");
      return () => ws.current && ws.current.close();
    }
  }, [status]);

  const handleAddItem = (type) => {
    const newItem = prompt(`Enter new ${type} name:`);
    if (newItem && newItem.trim() !== '') {
      if (type === 'vehicle') {
        setVehicles(prev => [...prev, newItem]);
        setSelectedVehicle(newItem);
      } else {
        setSupervisors(prev => [...prev, newItem]);
        setSelectedSupervisor(newItem);
      }
    }
  };

  const handleStart = async () => {
    if (!videoPath) {
      setErrorMessage('Please provide the full path to the video file.');
      setStatus('error');
      return;
    }
    
    setStatus('starting');
    setErrorMessage('');
    setCounts({});

    try {
      const response = await axios.post(`${API_URL}/start_processing`, {
        video_path: videoPath,
        vehicle_number: selectedVehicle,
        supervisor_name: selectedSupervisor,
        model_name: selectedModel,
      });

      if (response.data.error) {
        setErrorMessage(response.data.error);
        setStatus('error');
      } else {
        setStatus('processing');
      }
    } catch (err) {
      setErrorMessage('Failed to connect to the backend. Is it running?');
      setStatus('error');
    }
  };

  const handleStop = async () => {
    try {
      await axios.post(`${API_URL}/stop_processing`);
      setStatus('idle');
      setVideoPath(''); 
    } catch (err) {
      setErrorMessage('Failed to send stop command to the backend.');
      setStatus('error');
    }
  };

  const isProcessingOrStarting = status === 'processing' || status === 'starting';

  const calculateTotal = () => {
      if (selectedModel === 'multi_box_counter') {
          return Object.entries(counts).reduce((sum, [boxType, count]) => {
              const numMatch = boxType.match(/\d+/);
              if (numMatch) {
                  return sum + (parseInt(numMatch[0], 10) * count);
              }
              return sum;
          }, 0);
      }
      // For both LTR and RTL counters
      return Object.values(counts).reduce((sum, count) => sum + count, 0);
  };

  const total = calculateTotal();

  const getStatusText = () => {
      if (selectedModel === 'multi_box_counter') return `Total Value: <strong>${total}</strong>`;
      return `Total Count: <strong>${total}</strong>`;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>📦 Object Detection System</h1>
      </header>
      <main className="container">
        <div className="controls">
          <h2>Controls & Information</h2>

          {status === 'processing' && (
            <div className='status-box'>
              <p><strong>Status:</strong> Processing...</p>
              <div className='live-counts'>
                {Object.entries(counts).map(([boxType, count]) => (
                  <p key={boxType}>{boxType}: <strong>{count}</strong></p>
                ))}
                {Object.keys(counts).length > 0 && <hr />}
                <p dangerouslySetInnerHTML={{ __html: getStatusText() }} />
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="error-box">
              <strong>Error:</strong> {errorMessage}
            </div>
          )}
          
          <div className="form-group">
            <label>Select Model</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} disabled={isProcessingOrStarting}>
              <option value="multi_box_counter">Multi Box Counter</option>
              <option value="box_counter_rtl">Box Counter (Right to Left)</option>
              <option value="box_counter_ltr">Box Counter (Left to Right)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Full Video File Path</label>
            <input
              type="text"
              placeholder="Use Shift + Right-Click -> Copy as path"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              disabled={isProcessingOrStarting}
            />
          </div>

          <div className="form-group">
            <label>Vehicle Number</label>
            <div className="dropdown-container">
              <select value={selectedVehicle} onChange={(e) => setSelectedVehicle(e.target.value)} disabled={isProcessingOrStarting}>
                {vehicles.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
              <button onClick={() => handleAddItem('vehicle')} disabled={isProcessingOrStarting}>+</button>
            </div>
          </div>

          <div className="form-group">
            <label>Supervisor Name</label>
            <div className="dropdown-container">
              <select value={selectedSupervisor} onChange={(e) => setSelectedSupervisor(e.target.value)} disabled={isProcessingOrStarting}>
                {supervisors.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <button onClick={() => handleAddItem('supervisor')} disabled={isProcessingOrStarting}>+</button>
            </div>
          </div>
          
          <div className="button-group">
            <button onClick={handleStart} disabled={isProcessingOrStarting} className="start-btn">
              {status === 'starting' ? 'Starting...' : '▶️ Start Processing'}
            </button>
            <button onClick={handleStop} disabled={!isProcessingOrStarting} className="stop-btn">
              ⏹️ Stop Processing
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
