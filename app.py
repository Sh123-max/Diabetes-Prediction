from flask import Flask, request, render_template, Response
import pickle
import numpy as np
import os
import time
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_flask_exporter import PrometheusMetrics
import socket

app = Flask(__name__)

# Prometheus metrics
metrics = PrometheusMetrics(app, path="/metrics")  # auto-instrumentation
HOSTNAME = socket.gethostname()

# Custom metrics
MODEL_INFO = Gauge("model_version_info", "Info about loaded model", ["model_name", "model_version", "host"])
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["outcome"])
INPUT_VALIDATION_ERRORS = Counter("input_validation_errors_total", "Input validation errors counted", ["field"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency seconds", buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0))

# Load model (same as your code)
try:
    model_path = os.path.join('models', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    current_model_name = os.path.basename(model_path)
    MODEL_INFO.labels(model_name=current_model_name, model_version="v1", host=HOSTNAME).set(1)
except:
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        current_model_name = os.path.basename('best_model.pkl')
        MODEL_INFO.labels(model_name=current_model_name, model_version="v1", host=HOSTNAME).set(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        current_model_name = "none"

# valid_ranges / non_diabetic_ranges remain same...

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    if model is None:
        return render_template('form.html', 
                            prediction="Error",
                            probability="Model not loaded",
                            error_messages=["System error: Prediction model unavailable"])

    input_data = []
    error_messages = []
    non_diabetic_warnings = []

    # Validate inputs and increment validation counters
    for key in valid_ranges.keys():
        try:
            value = float(request.form[key])
            if value < valid_ranges[key][0] or value > valid_ranges[key][1]:
                error_messages.append(f"{key} must be between {valid_ranges[key][0]} and {valid_ranges[key][1]}")
                INPUT_VALIDATION_ERRORS.labels(field=key).inc()
            elif key in non_diabetic_ranges and (value < non_diabetic_ranges[key][0] or value > non_diabetic_ranges[key][1]):
                non_diabetic_warnings.append(f"{key} is outside typical non-diabetic range ({non_diabetic_ranges[key][0]}-{non_diabetic_ranges[key][1]})")
            input_data.append(value)
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")
            INPUT_VALIDATION_ERRORS.labels(field=key).inc()

    if error_messages:
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages)

    sample = np.array([input_data])

    try:
        with PREDICTION_LATENCY.time():
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5

        outcome = "diabetic" if prediction == 1 else "not_diabetic"
        PREDICTION_COUNT.labels(outcome=outcome).inc()

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        if prediction == 1 and len(non_diabetic_warnings) == 0:
            non_diabetic_warnings.append("Model predicted diabetic despite normal values - please verify with a doctor")

        return render_template('form.html',
                            prediction=result,
                            probability=f"{probability:.1%}",
                            non_diabetic_warnings=non_diabetic_warnings,
                            error_messages=[])
    except Exception as e:
        error_messages.append(f"Prediction error: {str(e)}")
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages)

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(host='0.0.0.0', port=5000)
