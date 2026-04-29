"""
=============================================================================
  Smart Meter Energy Forecasting — Prediction API  (US-07)
  FastAPI server exposing: /predict, /anomalies, /optimize, /health
=============================================================================
"""

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
from backend import database
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Load saved model artifacts ────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

def load_artifacts():
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise RuntimeError("models/model_metadata.json not found. Run train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    best = meta["best_model"]
    log.info("Loading best model: %s", best)

    if best == "SARIMA":
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        scaler = None
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODELS_DIR / "best_model.keras"))
        with open(MODELS_DIR / "lstm_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, meta

try:
    database.init_db()
    MODEL, SCALER, META = load_artifacts()
    BEST_NAME    = META["best_model"]
    LOOKBACK     = META["lookback"]
    FEATURE_COLS = META["feature_cols"]
    TARGET_COL   = META["target_col"]
    MODEL_READY  = True
except Exception as exc:
    log.warning("Could not load model: %s — /predict will return 503", exc)
    MODEL, SCALER, META = None, None, {}
    BEST_NAME = "UNKNOWN"
    LOOKBACK, FEATURE_COLS, TARGET_COL = 24, [], "Electricity_Consumed"
    MODEL_READY = False


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="SARIMA + LSTM based energy forecasting with anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=168, description="Hours to forecast (1–168)")
    temperature: Optional[float] = Field(default=0.5, ge=0, le=1)
    humidity:    Optional[float] = Field(default=0.5, ge=0, le=1)
    wind_speed:  Optional[float] = Field(default=0.3, ge=0, le=1)
    start_time:  Optional[str]   = Field(default=None, description="ISO-8601 start timestamp")

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_kwh: float

class ForecastResponse(BaseModel):
    model_used: str
    horizon_hours: int
    predictions: List[ForecastPoint]

class AnomalyRequest(BaseModel):
    readings: List[float] = Field(..., description="List of hourly energy readings (kWh)")
    timestamps: Optional[List[str]] = None

class OptimizationRequest(BaseModel):
    hourly_avg: List[float] = Field(..., description="24-element list of avg hourly usage")
    has_weekend_spike: Optional[bool] = False
    anomaly_count: Optional[int] = 0


# ── Helper: build a single feature row for LSTM ───────────────────────────────
def _make_feature_row(ts: datetime, temp: float, hum: float, wind: float,
                      prev_val: float, lag_24: float, roll6: float) -> dict:
    season_map = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
    return {
        "Temperature":          temp,
        "Humidity":             hum,
        "Wind_Speed":           wind,
        "Avg_Past_Consumption": prev_val,
        "hour_sin":  np.sin(2*np.pi*ts.hour/24),
        "hour_cos":  np.cos(2*np.pi*ts.hour/24),
        "month_sin": np.sin(2*np.pi*ts.month/12),
        "month_cos": np.cos(2*np.pi*ts.month/12),
        "is_weekend": int(ts.weekday() >= 5),
        "season":     season_map.get(ts.month, 2),
        "lag_1h":     prev_val,
        "lag_24h":    lag_24,
        "rolling_mean_6h": roll6,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY else "model_not_loaded",
        "best_model": BEST_NAME,
        "lookback": LOOKBACK,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if not MODEL_READY:
        raise HTTPException(503, "Model not loaded. Run train.py first.")

    start = datetime.fromisoformat(req.start_time.replace("Z", "")) if req.start_time else datetime.utcnow()
    horizon = req.horizon

    predictions = []

    if BEST_NAME == "SARIMA":
        try:
            forecast = MODEL.forecast(steps=horizon)
            vals = np.clip(np.array(forecast), 0, None)
        except Exception as e:
            raise HTTPException(500, f"SARIMA forecast error: {e}")

        for i, v in enumerate(vals):
            ts = start + timedelta(hours=i)
            kwh_val = round(float(v), 4)
            predictions.append(ForecastPoint(
                timestamp=ts.isoformat(),
                predicted_kwh=kwh_val
            ))
            database.insert_prediction(ts.isoformat(), BEST_NAME, horizon, kwh_val)

    else:  # LSTM
        # Build synthetic seed window
        seed_val = 0.5
        lag24 = seed_val
        roll6 = seed_val
        window_rows = []

        for i in range(LOOKBACK):
            ts = start - timedelta(hours=LOOKBACK - i)
            row = _make_feature_row(ts, req.temperature, req.humidity,
                                    req.wind_speed, seed_val, lag24, roll6)
            window_rows.append([row.get(c, 0.0) for c in FEATURE_COLS] + [seed_val])

        window = np.array(window_rows, dtype=np.float32)
        window_scaled = SCALER.transform(window)

        for i in range(horizon):
            ts = start + timedelta(hours=i)
            x = window_scaled[np.newaxis, :, :-1]
            p = float(MODEL.predict(x, verbose=0)[0, 0])

            # inverse transform
            dummy = np.zeros((1, len(FEATURE_COLS) + 1))
            dummy[0, -1] = p
            val = float(np.clip(SCALER.inverse_transform(dummy)[0, -1], 0, None))

            predictions.append(ForecastPoint(
                timestamp=ts.isoformat(),
                predicted_kwh=round(val, 4)
            ))
            database.insert_prediction(ts.isoformat(), BEST_NAME, horizon, round(val, 4))

            # slide window
            new_row = window_scaled[-1].copy()
            new_row[-1] = p
            window_scaled = np.vstack([window_scaled[1:], new_row])

    return ForecastResponse(
        model_used=BEST_NAME,
        horizon_hours=horizon,
        predictions=predictions,
    )


@app.post("/anomalies")
def detect_anomalies(req: AnomalyRequest):
    """Z-score anomaly detection on provided readings."""
    readings = np.array(req.readings, dtype=float)
    if len(readings) < 3:
        raise HTTPException(400, "Need at least 3 readings for anomaly detection.")

    mean, std = readings.mean(), readings.std()
    z_scores = np.abs((readings - mean) / (std + 1e-9))
    flags = z_scores > 3.0

    n = len(readings)
    timestamps = req.timestamps or [
        (datetime.utcnow() - timedelta(hours=n-i)).isoformat()
        for i in range(n)
    ]

    anomalies = [
        {"timestamp": timestamps[i], "value": float(readings[i]),
         "z_score": round(float(z_scores[i]), 3)}
        for i in range(n) if flags[i]
    ]

    for a in anomalies:
        database.insert_anomaly(a["timestamp"], a["value"], a["z_score"])

    return {
        "total_readings": n,
        "anomalies_detected": int(flags.sum()),
        "anomalies": anomalies,
    }


@app.post("/optimize")
def optimize(req: OptimizationRequest):
    """Rule-based optimization suggestions."""
    avg = np.array(req.hourly_avg)
    if len(avg) != 24:
        raise HTTPException(400, "hourly_avg must have exactly 24 values.")

    suggestions = []
    peak_hour = int(np.argmax(avg))
    off_peak  = int(np.argmin(avg))
    
    evening   = avg[18:23].mean() if len(avg[18:23]) > 0 else 0
    night     = avg[0:6].mean() if len(avg[0:6]) > 0 else 0
    morning   = avg[6:12].mean() if len(avg[6:12]) > 0 else 0
    
    overall_mean = avg.mean()
    variance = avg.var()

    if variance > (overall_mean * 0.5) ** 2 + 1e-6:
        suggestions.append("📉 Usage is highly variable throughout the day. Consider installing a smart thermostat or automated load balancing.")

    if morning > evening * 1.2:
        suggestions.append(f"🌅 Morning consumption is dominant. Shift non-essential morning loads (e.g., water heating) to the {off_peak:02d}:00 off-peak window.")
    elif evening > 1.5 * night + 1e-6:
        suggestions.append(f"⚡ Evening consumption is significantly higher. Distribute load away from the 18:00–22:00 window to avoid peak pricing.")

    if peak_hour in range(18, 23):
        suggestions.append(f"🔴 Critical peak demand at {peak_hour:02d}:00. Actively shift heavy appliance usage to {off_peak:02d}:00.")

    if night > overall_mean * 0.6:
        suggestions.append("🌙 Nighttime base load is unusually high. Check for appliances left on or phantom energy draws while sleeping.")

    if req.has_weekend_spike:
        suggestions.append("📅 Weekend usage is 20%+ above weekdays. Review weekend scheduling of HVAC, lighting, or facility systems.")
        
    if req.anomaly_count and req.anomaly_count > 0:
        suggestions.append(f"🔍 {req.anomaly_count} anomalous spikes detected. Promptly inspect equipment for inefficiencies or faults.")

    suggestions.append(f"✅ Optimal off-peak window: {off_peak:02d}:00–{(off_peak+2)%24:02d}:00. Schedule EV charging, laundry, and dishwashers during this time.")

    return {"suggestions": suggestions, "peak_hour": peak_hour, "off_peak_hour": off_peak}


@app.get("/model/info")
def model_info():
    if not MODEL_READY:
        raise HTTPException(503, "Model not loaded.")
    return {
        "best_model": BEST_NAME,
        "feature_cols": FEATURE_COLS,
        "lookback": LOOKBACK,
        "metrics": META.get("all_metrics", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)