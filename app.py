from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import json

app = Flask(__name__)

# Load model and its metadata (best model info, metrics)
model = None
model_name = "Unknown"
model_metrics = {}

try:
    model_path = os.path.join('models', 'best_model.pkl')
    metadata_path = os.path.join('models', 'model_metadata.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            model_name = meta.get("model_name", "Unknown")
            model_metrics = meta.get("metrics", {})
except Exception as e:
    print(f"Error loading model or metadata: {e}")

# Define validation ranges
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

# Normal non-diabetic ranges
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

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[],
                           model_name=model_name, model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('form.html', prediction="Error", probability="Model not loaded",
                               error_messages=["Prediction model not available."],
                               model_name=model_name, model_metrics=model_metrics)

    input_data = []
    error_messages = []
    non_diabetic_warnings = []

    for key in valid_ranges.keys():
        try:
            value = float(request.form[key])

            if value < valid_ranges[key][0] or value > valid_ranges[key][1]:
                error_messages.append(f"{key} must be between {valid_ranges[key][0]} and {valid_ranges[key][1]}")
            elif key in non_diabetic_ranges and (value < non_diabetic_ranges[key][0] or value > non_diabetic_ranges[key][1]):
                non_diabetic_warnings.append(f"{key} is outside normal non-diabetic range ({non_diabetic_ranges[key][0]} - {non_diabetic_ranges[key][1]})")

            input_data.append(value)
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")

    if error_messages:
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages,
                               model_name=model_name, model_metrics=model_metrics)

    try:
        sample = np.array([input_data])
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        if prediction == 1 and len(non_diabetic_warnings) == 0:
            non_diabetic_warnings.append("Model predicted diabetic despite mostly normal inputs. Please consult a doctor.")

        return render_template('form.html', prediction=result, probability=f"{probability:.1%}",
                               non_diabetic_warnings=non_diabetic_warnings, error_messages=[],
                               model_name=model_name, model_metrics=model_metrics)

    except Exception as e:
        error_messages.append(f"Prediction error: {str(e)}")
        return render_template('form.html', prediction=None, probability=None, error_messages=error_messages,
                               model_name=model_name, model_metrics=model_metrics)

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')

    app.run(host='0.0.0.0', port=5000)
