from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import json
import time
import socket
from prometheus_client import Gauge, Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# -----------------------------
# Prometheus metrics
# -----------------------------
metrics = PrometheusMetrics(app, path="/metrics")  # auto-instrumentation
HOSTNAME = socket.gethostname()

# Custom metrics
MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["field"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency seconds",
                               buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0))

# -----------------------------
# Load model and metadata
# -----------------------------
model = None
model_name = "Unknown"
model_metrics = {}

try:
    model_path = os.path.join('models', 'best_model.pkl')
    fallback_model_path = 'best_model.pkl'
    metadata_path = os.path.join('models', 'model_metadata.json')

    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Checking for model in: {model_path} or {fallback_model_path}")
    print(f"[DEBUG] Checking for metadata in: {metadata_path}")

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        current_model_name = os.path.basename(model_path)
        print(f"[✔] Model loaded from {model_path}")
    elif os.path.exists(fallback_model_path):
        with open(fallback_model_path, 'rb') as f:
            model = pickle.load(f)
        current_model_name = os.path.basename(fallback_model_path)
        print(f"[✔] Model loaded from fallback {fallback_model_path}")
    else:
        print("[✘] Model not found in 'models/' or project root.")
        current_model_name = "none"

    # Set Prometheus gauge
    MODEL_INFO.labels(model_name=current_model_name, model_version="v1", host=HOSTNAME).set(1)

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            model_name = meta.get("model_name", "Unknown")
            model_metrics = meta.get("metrics", {})
        print("[✔] Loaded model metadata.")
    else:
        print("[⚠] Metadata file not found. Proceeding without metrics.")

except Exception as e:
    print(f"[❌] Error loading model or metadata: {e}")

# -----------------------------
# Input ranges
# -----------------------------
valid_ranges = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "BloodPressure": (40, 140),
    "SkinThickness": (10, 100),
    "Insulin": (15, 846),
    "BMI": (15, 50),
    "DiabetesPedigreeFunction": (0.1, 2.5),
    "Age": (15, 100)
}

non_diabetic_ranges = {
    "Pregnancies": (0, 5),
    "Glucose": (70, 99),
    "BloodPressure": (60, 80),
    "SkinThickness": (10, 30),
    "Insulin": (3, 25),
    "BMI": (18.5, 24.9),
    "DiabetesPedigreeFunction": (0.1, 0.5),
    "Age": (15, 100)
}

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('form.html',
                           prediction=None,
                           probability=None,
                           error_messages=[],
                           non_diabetic_warnings=[],
                           model_name=model_name,
                           model_metrics=model_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if model is None:
        return render_template('form.html',
                               prediction="Error",
                               probability="Model not loaded",
                               error_messages=["System error: Prediction model unavailable."],
                               non_diabetic_warnings=[],
                               model_name=model_name,
                               model_metrics=model_metrics)

    input_data = []
    error_messages = []
    non_diabetic_warnings = []

    # Validate inputs
    for key in valid_ranges.keys():
        try:
            value = float(request.form[key])
            if not (valid_ranges[key][0] <= value <= valid_ranges[key][1]):
                error_messages.append(f"{key} must be between {valid_ranges[key][0]} and {valid_ranges[key][1]}")
                INPUT_VALIDATION_ERRORS.labels(field=key).inc()
            elif key in non_diabetic_ranges and not (non_diabetic_ranges[key][0] <= value <= non_diabetic_ranges[key][1]):
                non_diabetic_warnings.append(
                    f"{key} is outside typical non-diabetic range ({non_diabetic_ranges[key][0]} - {non_diabetic_ranges[key][1]})"
                )
            input_data.append(value)
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")
            INPUT_VALIDATION_ERRORS.labels(field=key).inc()

    if error_messages:
        return render_template('form.html',
                               prediction=None,
                               probability=None,
                               error_messages=error_messages,
                               non_diabetic_warnings=[],
                               model_name=model_name,
                               model_metrics=model_metrics)

    try:
        sample = np.array([input_data])
        with PREDICTION_LATENCY.time():
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5

        outcome = "diabetic" if prediction == 1 else "not_diabetic"
        PREDICTION_COUNT.labels(outcome=outcome).inc()

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        if prediction == 1 and len(non_diabetic_warnings) == 0:
            non_diabetic_warnings.append("Model predicted diabetic despite mostly normal inputs. Please consult a doctor.")

        return render_template('form.html',
                               prediction=result,
                               probability=f"{probability:.1%}",
                               error_messages=[],
                               non_diabetic_warnings=non_diabetic_warnings,
                               model_name=model_name,
                               model_metrics=model_metrics)
    except Exception as e:
        return render_template('form.html',
                               prediction=None,
                               probability=None,
                               error_messages=[f"Prediction error: {str(e)}"],
                               non_diabetic_warnings=[],
                               model_name=model_name,
                               model_metrics=model_metrics)


# -----------------------------
# Start server
# -----------------------------
if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(host='0.0.0.0', port=5000)
