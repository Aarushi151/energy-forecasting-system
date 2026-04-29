import itertools
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from train import load_data, handle_missing, convert_timestamp, resample_hourly, engineer_features, split_data, FEATURE_COLS, TARGET_COL, build_sequences, build_contextual_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

def evaluate_sarima(train_series, val_series, order, seasonal_order):
    try:
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=50)
        forecast = fitted.forecast(steps=len(val_series))
        preds = np.clip(np.array(forecast), 0, None)
        mae = mean_absolute_error(val_series, preds)
        return mae
    except Exception:
        return float('inf')

def tune_sarima(train_df, val_df):
    log.info("Starting SARIMA Hyperparameter Tuning...")
    train_series = train_df[TARGET_COL]
    val_series = val_df[TARGET_COL]
    
    # Grid search for (p,d,q) and (P,D,Q,s)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in pdq]
    
    best_score = float('inf')
    best_params = None
    
    # Just try a few combinations to keep it fast
    for param in pdq[:3]:
        for param_seasonal in seasonal_pdq[:2]:
            try:
                mae = evaluate_sarima(train_series, val_series, param, param_seasonal)
                log.info(f"SARIMA{param}x{param_seasonal} - MAE: {mae:.4f}")
                if mae < best_score:
                    best_score = mae
                    best_params = (param, param_seasonal)
            except Exception:
                continue
                
    log.info(f"Best SARIMA Params: {best_params} with MAE: {best_score:.4f}")
    return best_params

def tune_lstm(train_df, val_df):
    log.info("Starting LSTM Hyperparameter Tuning...")
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    cols = available_features + [TARGET_COL]
    
    train_vals = train_df[cols].values
    val_vals = val_df[cols].values
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    val_scaled = scaler.transform(val_vals)
    
    lookback = 24
    X_train, y_train = build_sequences(train_scaled, lookback)
    X_val, y_val = build_contextual_sequences(train_scaled, val_scaled, lookback)
    
    hyperparams = [
        {"units": 32, "dropout": 0.1, "lr": 0.001, "batch_size": 32},
        {"units": 64, "dropout": 0.2, "lr": 0.001, "batch_size": 32},
        {"units": 64, "dropout": 0.2, "lr": 0.005, "batch_size": 64}
    ]
    
    best_loss = float('inf')
    best_params = None
    n_features = X_train.shape[2]
    
    for hp in hyperparams:
        model = Sequential([
            Input(shape=(lookback, n_features)),
            LSTM(hp["units"], return_sequences=True),
            Dropout(hp["dropout"]),
            LSTM(hp["units"] // 2, return_sequences=False),
            Dropout(hp["dropout"]),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp["lr"])
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        
        log.info(f"Training LSTM with {hp}")
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=10, batch_size=hp["batch_size"], callbacks=[early_stopping], verbose=0
        )
        
        val_loss = min(history.history["val_loss"])
        log.info(f"LSTM {hp} - Val Loss: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = hp
            
    log.info(f"Best LSTM Params: {best_params} with Val Loss: {best_loss:.6f}")
    return best_params

def main():
    log.info("Loading Data for Tuning...")
    df = load_data(Path(__file__).parent.parent / "data" / "smart_meter_data.csv")
    df = handle_missing(df)
    df = convert_timestamp(df)
    df = resample_hourly(df)
    df = engineer_features(df)
    
    if "Anomaly_Label" in df.columns:
        df = df.drop(columns=["Anomaly_Label"])
        
    train_df, val_df, test_df = split_data(df)
    
    log.info("--- Starting Tuning Pipeline ---")
    tune_sarima(train_df, val_df)
    tune_lstm(train_df, val_df)
    log.info("--- Tuning Complete ---")

if __name__ == "__main__":
    main()
