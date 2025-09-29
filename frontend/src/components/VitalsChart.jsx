import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter
} from 'recharts'

export default function VitalsChart({ windowData = [], anomalyScore = 0 }) {
  // Map windowData into chart format
  const data = windowData.map((row, i) => ({
    idx: i,
    hr: row[0],
    rr: row[1],
    spo2: row[2],
    isAnom: anomalyScore >= 0.5 // mark anomalies if score > 0.5
  }))

  return (
    <div style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="idx" label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }} />
          <YAxis />
          <Tooltip />
          <Legend />

          {/* Heart Rate */}
          <Line type="monotone" dataKey="hr" stroke="#1677ff" dot={false} strokeWidth={2} />
          {/* Respiration Rate */}
          <Line type="monotone" dataKey="rr" stroke="#52c41a" dot={false} strokeWidth={2} />
          {/* SpO2 */}
          <Line type="monotone" dataKey="spo2" stroke="#faad14" dot={false} strokeWidth={2} />

          {/* Mark anomalies */}
          {anomalyScore >= 0.5 && (
            <Scatter data={data.filter(d => d.isAnom)} dataKey="hr" fill="red" shape="circle" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
