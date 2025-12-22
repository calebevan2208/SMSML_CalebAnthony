"""
7.inference.py
--------------
Script inference sederhana menggunakan Flask.
Memuat model Deep Learning yang telah dilatih dan scaler untuk melakukan prediksi real-time.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)

# --- KONFIGURASI PATH ---
BASE_DIR = Path(__file__).resolve().parent.parent / 'Membangun_model'
MODEL_PATH = BASE_DIR / 'artifacts' / 'baseline_model.h5'
SCALER_PATH = BASE_DIR / 'artifacts' / 'scaler.pkl'

# --- LOAD ARTIFACTS ---
# Karena model dan scaler mungkin belum ada di path relatif saat development,
# kita tambahkan error handling untuk dummy execution.
model = None
scaler = None

try:
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
    else:
        print(f"Warning: Scaler not found at {SCALER_PATH}")
except Exception as e:
    print(f"Error loading artifacts: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or Scaler not initialized'}), 500

    try:
        data = request.get_json(force=True)
        # Asumsi data dikirim sebagai list of values atau dictionary yang sesuai
        # Untuk simplifikasi, kita anggap input adalah list of features
        input_data = np.array(data['features']).reshape(1, -1)
        
        # Preprocessing (Scaling)
        scaled_data = scaler.transform(input_data)
        
        # Prediction
        prediction_prob = model.predict(scaled_data)
        prediction_class = (prediction_prob > 0.5).astype(int)[0][0]
        
        return jsonify({
            'prediction_class': int(prediction_class),
            'prediction_prob': float(prediction_prob[0][0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
