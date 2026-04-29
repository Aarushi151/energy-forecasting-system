import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data.db"

def init_db():
    """Initialises the SQLite database and creates necessary tables if they don't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Table for Raw Data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT UNIQUE,
                    electricity_consumed REAL,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL
                )
            ''')
            
            # Table for Predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_used TEXT,
                    horizon_hours INTEGER,
                    predicted_kwh REAL
                )
            ''')
            
            # Table for Anomalies
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    value REAL,
                    z_score REAL
                )
            ''')
            
            conn.commit()
            log.info("Database initialized at %s", DB_PATH)
    except Exception as e:
        log.error("Failed to initialize database: %s", e)

def insert_raw_data(timestamp: str, electricity: float, temp: float, humidity: float, wind: float):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO raw_data (timestamp, electricity_consumed, temperature, humidity, wind_speed)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, electricity, temp, humidity, wind))
            conn.commit()
    except Exception as e:
        log.error("Failed to insert raw data: %s", e)

def insert_prediction(timestamp: str, model_used: str, horizon: int, predicted_kwh: float):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, model_used, horizon_hours, predicted_kwh)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, model_used, horizon, predicted_kwh))
            conn.commit()
    except Exception as e:
        log.error("Failed to insert prediction: %s", e)

def insert_anomaly(timestamp: str, value: float, z_score: float):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO anomalies (timestamp, value, z_score)
                VALUES (?, ?, ?)
            ''', (timestamp, value, z_score))
            conn.commit()
    except Exception as e:
        log.error("Failed to insert anomaly: %s", e)
