import os
import gdown
import logging

log = logging.getLogger(__name__)

# Ensure path is relative to this file
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FILES = {
    "best_model.pkl": "1MTyun6KuoWSdapkQwUAa6xxrPPSAHq_k",
    "lstm_scaler.pkl": "1LvUfKqsdzaYHvpdSt79Fn2LM_DV9D4KJ",
    "model_metadata.json": "15GRlvS6AY1VuKabXywESRlNIYcZN2XZm"
}

def download_file_from_drive(file_id, destination):
    gdown.download(id=file_id, output=destination, quiet=False)

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            log.info(f"Downloading {filename}...")
            try:
                download_file_from_drive(file_id, path)
                log.info(f"{filename} downloaded successfully")
            except Exception as e:
                log.error(f"Failed to download {filename}: {e}")
        else:
            log.info(f"{filename} already exists")