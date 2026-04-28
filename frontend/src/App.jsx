import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const API_BASE = "http://localhost:8000";

const C = {
  blue:   "#3b82f6",
  green:  "#10b981",
  amber:  "#f59e0b",
  red:    "#ef4444",
  purple: "#8b5cf6",
  bg:     "#0f172a",
  card:   "#1e293b",
  border: "#334155",
  text:   "#f1f5f9",
  muted:  "#94a3b8",
};

function KpiCard({ label, value, unit, color, icon }) {
  return (
    <div style={{ background: C.card, border: "1px solid " + C.border, borderRadius: 12, padding: "20px 24px", display: "flex", alignItems: "center", gap: 16 }}>
      <div style={{ width: 48, height: 48, borderRadius: 12, background: (color || C.blue) + "22", display: "grid", placeItems: "center", fontSize: 20, fontWeight: 700, color: color || C.blue }}>{icon}</div>
      <div>
        <div style={{ color: C.muted, fontSize: 13, marginBottom: 4 }}>{label}</div>
        <div style={{ color: C.text, fontSize: 26, fontWeight: 700 }}>{value} <span style={{ fontSize: 14, color: C.muted }}>{unit}</span></div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab]               = useState("forecast");
  const [health, setHealth]         = useState(null);
  const [forecast, setForecast]     = useState([]);
  const [anomalies, setAnomalies]   = useState({});
  const [suggestions, setSuggestions] = useState({});
  const [metrics, setMetrics]       = useState([]);
  const [horizon, setHorizon]       = useState(24);
  const [loading, setLoading]       = useState({});
  const [error, setError]           = useState(null);

  useEffect(() => {
    fetch(API_BASE + "/health")
      .then(r => r.json()).then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
    fetch(API_BASE + "/model/info")
      .then(r => r.json()).then(d => setMetrics(d.metrics || []))
      .catch(() => {});
  }, []);

  const fetchForecast = useCallback(async () => {
    setLoading(l => ({ ...l, forecast: true }));
    setError(null);
    try {
      const res = await fetch(API_BASE + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon: horizon, start_time: new Date().toISOString() }),
      });
      const data = await res.json();
      setForecast((data.predictions || []).map((p, i) => ({
        hour: "+" + (i + 1) + "h",
        kwh: parseFloat(p.predicted_kwh.toFixed(4)),
      })));
    } catch (e) {
      setError("API unreachable. Make sure api.py is running on port 8000.");
    }
    setLoading(l => ({ ...l, forecast: false }));
  }, [horizon]);

  const fetchAnomalies = useCallback(async () => {
    setLoading(l => ({ ...l, anomaly: true }));
    const readings = Array.from({ length: 48 }, (_, i) =>
      0.4 + 0.15 * Math.sin(i * Math.PI / 12) + (i === 20 || i === 35 ? 1.8 : 0) + Math.random() * 0.05
    );
    try {
      const res = await fetch(API_BASE + "/anomalies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ readings: readings }),
      });
      const data = await res.json();
      setAnomalies({
        total: data.total_readings,
        count: data.anomalies_detected,
        items: data.anomalies || [],
        chartData: readings.map((v, i) => ({
          idx: i,
          value: parseFloat(v.toFixed(4)),
          anomaly: (data.anomalies || []).some(a => Math.abs(a.value - v) < 0.01) ? v : null,
        })),
      });
    } catch (e) {
      setError("API unreachable.");
    }
    setLoading(l => ({ ...l, anomaly: false }));
  }, []);

  const fetchOptimize = useCallback(async () => {
    setLoading(l => ({ ...l, opt: true }));
    const hourlyAvg = Array.from({ length: 24 }, (_, h) =>
      h >= 18 && h <= 21 ? 0.85 + Math.random() * 0.1 : 0.35 + Math.random() * 0.05
    );
    try {
      const res = await fetch(API_BASE + "/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hourly_avg: hourlyAvg, anomaly_count: 2, has_weekend_spike: true }),
      });
      const data = await res.json();
      setSuggestions({ items: data.suggestions });
    } catch (e) {
      setError("API unreachable.");
    }
    setLoading(l => ({ ...l, opt: false }));
  }, []);

  const tabBtn = (t, label) => (
    <button key={t} onClick={() => setTab(t)} style={{
      padding: "8px 20px", borderRadius: 8, cursor: "pointer", fontSize: 14,
      fontWeight: tab === t ? 700 : 400,
      background: tab === t ? C.blue : "transparent",
      color: tab === t ? "#fff" : C.muted,
      border: "none",
    }}>{label}</button>
  );

  const card = { background: C.card, border: "1px solid " + C.border, borderRadius: 12, padding: 24, marginBottom: 24 };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "Inter, sans-serif", padding: 24 }}>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>Energy Forecasting Dashboard</h1>
          <p style={{ margin: "4px 0 0", color: C.muted, fontSize: 14 }}>Smart Meter Analytics - SARIMA + LSTM</p>
        </div>
        <div style={{ padding: "6px 14px", borderRadius: 20, background: health?.status === "ok" ? C.green + "22" : C.red + "22", color: health?.status === "ok" ? C.green : C.red, fontSize: 13, fontWeight: 600 }}>
          {health?.status === "ok" ? "API Live - " + health.best_model : "API Offline"}
        </div>
      </div>

      {error && (
        <div style={{ background: C.red + "22", border: "1px solid " + C.red, borderRadius: 8, padding: "12px 16px", marginBottom: 24, color: C.red, fontSize: 14 }}>
          {error}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 32 }}>
        <KpiCard label="Model" value={health?.best_model || "-"} unit="" color={C.blue} icon="AI" />
        <KpiCard label="Lookback" value={health?.lookback || "-"} unit="hrs" color={C.purple} icon="T" />
        {metrics[0] && <KpiCard label="SARIMA RMSE" value={typeof metrics[0].RMSE === "number" ? metrics[0].RMSE.toFixed(3) : "-"} unit="" color={C.amber} icon="S" />}
        {metrics[1] && <KpiCard label="LSTM RMSE" value={typeof metrics[1].RMSE === "number" ? metrics[1].RMSE.toFixed(3) : "-"} unit="" color={C.green} icon="L" />}
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 24, background: C.card, padding: 6, borderRadius: 12, width: "fit-content" }}>
        {tabBtn("forecast", "Forecast")}
        {tabBtn("anomaly", "Anomalies")}
        {tabBtn("optimize", "Optimize")}
      </div>

      {tab === "forecast" && (
        <div style={card}>
          <h2 style={{ margin: "0 0 16px", fontSize: 18 }}>Energy Consumption Forecast</h2>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
            <label style={{ color: C.muted, fontSize: 14 }}>Horizon:</label>
            <select value={horizon} onChange={e => setHorizon(Number(e.target.value))}
              style={{ background: C.bg, color: C.text, border: "1px solid " + C.border, borderRadius: 8, padding: "6px 12px", fontSize: 14 }}>
              {[1, 6, 12, 24, 48, 72, 168].map(h => <option key={h} value={h}>{h}h</option>)}
            </select>
            <button onClick={fetchForecast} disabled={loading.forecast}
              style={{ padding: "8px 20px", background: C.blue, color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600 }}>
              {loading.forecast ? "Loading..." : "Run Forecast"}
            </button>
          </div>
          {forecast.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={forecast}>
                <defs>
                  <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.blue} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={C.blue} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="hour" tick={{ fill: C.muted, fontSize: 12 }} interval={Math.floor(horizon / 6)} />
                <YAxis tick={{ fill: C.muted, fontSize: 12 }} />
                <Tooltip contentStyle={{ background: C.card, border: "1px solid " + C.border, borderRadius: 8, color: C.text }} />
                <Area type="monotone" dataKey="kwh" stroke={C.blue} fill="url(#bg)" strokeWidth={2} dot={false} name="kWh" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 300, display: "grid", placeItems: "center", color: C.muted }}>
              Click "Run Forecast" to load predictions
            </div>
          )}
        </div>
      )}

      {tab === "anomaly" && (
        <div style={card}>
          <h2 style={{ margin: "0 0 16px", fontSize: 18 }}>Anomaly Detection</h2>
          <button onClick={fetchAnomalies} disabled={loading.anomaly}
            style={{ padding: "8px 20px", background: C.red, color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600, marginBottom: 20 }}>
            {loading.anomaly ? "Detecting..." : "Detect Anomalies"}
          </button>
          {anomalies.chartData && (
            <div>
              <div style={{ display: "flex", gap: 24, marginBottom: 16 }}>
                <span style={{ color: C.muted, fontSize: 14 }}>Total: <b style={{ color: C.text }}>{anomalies.total}</b></span>
                <span style={{ color: C.red, fontSize: 14 }}>Anomalies: <b>{anomalies.count}</b></span>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={anomalies.chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="idx" tick={{ fill: C.muted, fontSize: 11 }} />
                  <YAxis tick={{ fill: C.muted, fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: C.card, border: "1px solid " + C.border, borderRadius: 8, color: C.text }} />
                  <Line type="monotone" dataKey="value" stroke={C.blue} strokeWidth={2} dot={false} name="Usage" />
                  <Line type="monotone" dataKey="anomaly" stroke={C.red} strokeWidth={0} dot={{ r: 6, fill: C.red }} name="Anomaly" connectNulls={false} />
                </LineChart>
              </ResponsiveContainer>
              {(anomalies.items || []).map((a, i) => (
                <div key={i} style={{ padding: "8px 12px", background: C.red + "11", border: "1px solid " + C.red + "44", borderRadius: 8, marginTop: 8, fontSize: 13 }}>
                  Value {a.value.toFixed(4)} at {a.timestamp} - z-score: {a.z_score}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === "optimize" && (
        <div style={card}>
          <h2 style={{ margin: "0 0 16px", fontSize: 18 }}>Optimization Suggestions</h2>
          <button onClick={fetchOptimize} disabled={loading.opt}
            style={{ padding: "8px 20px", background: C.green, color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600, marginBottom: 20 }}>
            {loading.opt ? "Analyzing..." : "Get Suggestions"}
          </button>
          {(suggestions.items || []).map((s, i) => (
            <div key={i} style={{ padding: "14px 16px", background: C.bg, border: "1px solid " + C.border, borderRadius: 10, marginBottom: 10, fontSize: 14, lineHeight: 1.6 }}>
              {s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}