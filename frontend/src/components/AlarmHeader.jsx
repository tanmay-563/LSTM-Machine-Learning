import React from 'react'

export default function AlarmHeader({ mode, score, risk }) {
  const getRiskStyle = () => {
    if (risk === 'high') {
      return { background: '#ffe6e6', color: '#b42318', border: '1px solid #f5c2c7' }
    }
    if (risk === 'medium') {
      return { background: '#fff9db', color: '#ad8b00', border: '1px solid #ffe58f' }
    }
    return { background: '#e6fffb', color: '#006d75', border: '1px solid #87e8de' }
  }

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '16px 20px',
      borderRadius: 10,
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      background: '#fafafa',
      marginBottom: 20
    }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '1.4rem' }}>Digital Epidemiologist Dashboard</h2>
        <p style={{ margin: 0, color: '#666' }}>Monitoring patient vitals in real-time</p>
      </div>

      <div style={{
        padding: '10px 16px',
        borderRadius: 8,
        fontWeight: 'bold',
        minWidth: 140,
        textAlign: 'center',
        ...getRiskStyle()
      }}>
        {mode.toUpperCase()} MODE<br />
        Risk: {risk.toUpperCase()}<br />
        Score: {score.toFixed(3)}
      </div>
    </div>
  )
}
