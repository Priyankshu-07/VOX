import React, { useState } from 'react';
import './App.css';

interface Prediction {
  intent: string;
  confidence: number;
  slots: Record<string, string>;
  response: string;
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      
      if (!res.ok) throw new Error('Failed to fetch prediction');
      
      const data = await res.json();
      setResult(data);
      setText('');
    } catch (err) {
      setError('Connection to VOX engine failed. Is the backend running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Mock voice interaction injecting random test data
  const handleVoice = () => {
    const examples = [
      "bhai location bhej de",
      "traffic ki wajah se 10 min late honga",
      "customer phone ni utha rha",
      "order damage hai cancel karo",
      "map stuck ho gaya hai"
    ];
    setText(examples[Math.floor(Math.random() * examples.length)]);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-title">EdgeAssist.</div>
        <div className="status-badge">
          <div className="status-dot"></div>
          ONNX Engine
        </div>
      </header>

      {/* Main Content Area */}
      <main className="app-main">
        {error && (
          <div className="response-message" style={{ borderLeftColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.1)' }}>
            {error}
          </div>
        )}

        {!result && !loading && !error && (
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '40px' }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '16px', opacity: 0.5 }}>
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
            <p>Ready for offline commands.</p>
            <p style={{ fontSize: '0.85rem', marginTop: '8px', opacity: 0.7 }}>Try typing: "traffic ki wajah se 10 min late honga"</p>
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center', marginTop: '40px' }}>
            <div style={{ width: '24px', height: '24px', border: '2px solid var(--accent-primary)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }}></div>
          </div>
        )}

        {result && !loading && (
          <>
            <div className="intent-panel active">
              <div className="intent-label">Detected Intent</div>
              <div className="intent-value">
                {result.intent.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
              </div>
              
              <div className="confidence-bar-bg">
                <div 
                  className="confidence-bar-fill" 
                  style={{ width: `${Math.round(result.confidence * 100)}%` }}
                ></div>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '8px', textAlign: 'right' }}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>

            {Object.keys(result.slots).length > 0 && (
              <div className="slots-panel">
                {Object.entries(result.slots).map(([key, value]) => (
                  <div key={key} className="slot-chip">
                    <span className="slot-key">{key}:</span>
                    <span className="slot-value">"{value}"</span>
                  </div>
                ))}
              </div>
            )}

            <div className="response-message">
              {result.response}
            </div>
          </>
        )}
      </main>

      {/* Input Area */}
      <form className="input-area" onSubmit={handlePredict}>
        <button type="button" className="action-btn voice" onClick={handleVoice} title="Simulate Voice Input">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
        </button>
        <input 
          type="text" 
          className="text-input" 
          placeholder="Type command..." 
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          autoComplete="off"
        />
        <button type="submit" className="action-btn" disabled={!text.trim() || loading}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" x2="11" y1="2" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </form>

      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

export default App;
