import os
import joblib
import json
import logging

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

model = None
scaler = None
metadata = None


def load_model():
    global model, scaler, metadata

    try:
        log.info("Loading model...")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        log.info("Model loaded successfully")
        return True

    except Exception as e:
        import traceback
        err_msg = f"Error loading model: {e}\n{traceback.format_exc()}"
        log.error(err_msg)
        print("CRITICAL ERROR IN LOAD_MODEL:", err_msg)
        return False


def get_model():
    return model


def get_scaler():
    return scaler


def get_metadata():
    return metadata