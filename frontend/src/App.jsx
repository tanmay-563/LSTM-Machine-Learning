import React, { useEffect, useState, useRef } from 'react'
import AlarmHeader from './components/AlarmHeader'
import VitalsChart from './components/VitalsChart'

export default function App() {
  const [state, setState] = useState({
    mode: 'vital',
    anomaly_score: 0,
    hr: 0,
    rr: 0,
    spo2: 0
  })
  const [lastWindow, setLastWindow] = useState([])
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WS open')
      setWsConnected(true)
    }

    ws.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data)
        setState(prev => ({
          ...prev,
          mode: d.mode ?? prev.mode,
          anomaly_score: Number(d.anomaly_score ?? prev.anomaly_score),
          hr: Number(d.hr ?? prev.hr),
          rr: Number(d.rr ?? prev.rr),
          spo2: Number(d.spo2 ?? prev.spo2)
        }))
        if (d.window) setLastWindow(d.window)
      } catch (err) {
        console.error('WS parse error', err)
      }
    }

    ws.onclose = () => {
      console.log('WS closed')
      setWsConnected(false)
    }

    return () => {
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  function scoreToRisk(score) {
    if (score >= 0.9) return 'high'
    if (score >= 0.6) return 'medium'
    return 'low'
  }

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: 24, fontFamily: 'Inter, Arial' }}>
      {/* Header */}
      <AlarmHeader
        mode={state.mode}
        score={state.anomaly_score}
        risk={scoreToRisk(state.anomaly_score)}
      />

      {/* Connection status */}
      <div style={{
        marginTop: 16,
        padding: 12,
        border: '1px solid #ddd',
        borderRadius: 8,
        background: wsConnected ? '#f6ffed' : '#fffbe6'
      }}>
        <strong>WebSocket:</strong>{' '}
        {wsConnected
          ? <span style={{ color: 'green' }}>Connected</span>
          : <span style={{ color: 'orange' }}>Disconnected</span>}
      </div>

      {/* Content */}
      <div style={{ marginTop: 20, display: 'flex', gap: 20 }}>
        {/* Left: Chart */}
        <div style={{
          flex: 1,
          background: '#fff',
          padding: 16,
          borderRadius: 8,
          boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{ marginBottom: 12 }}>Patient Vitals (latest window)</h3>
          <VitalsChart windowData={lastWindow} anomalyScore={state.anomaly_score} />
        </div>

        {/* Right: Latest state */}
        <div style={{
          width: 280,
          background: '#fff',
          padding: 16,
          borderRadius: 8,
          boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
        }}>
          <h4>Latest State</h4>
          <div style={{ marginTop: 8 }}>
            <div><strong>Mode:</strong> {state.mode}</div>
            <div><strong>Alarm Score:</strong> {state.anomaly_score.toFixed(3)}</div>
            <div><strong>Risk Level:</strong> {scoreToRisk(state.anomaly_score)}</div>
            <div><strong>HR:</strong> {state.hr}</div>
            <div><strong>RR:</strong> {state.rr}</div>
            <div><strong>SpO₂:</strong> {state.spo2}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
