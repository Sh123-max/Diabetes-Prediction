import os
import pickle
import traceback
import matplotlib.pyplot as plt
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc 
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

# Show working directory
print("Current working directory:", os.getcwd())

# Create directory for models
try:
    os.makedirs('models', exist_ok=True)
except Exception as e:
    print(" Failed to create 'models' directory:", e)
    traceback.print_exc()

# Load processed data
try:
    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
except Exception as e:
    print(" Error loading data:", e)
    traceback.print_exc()
    exit(1)

# Initialize classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression()
)
models['Stacking Ensemble'] = stacking_model

# Metric Weights for Healthcare Priority
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# Initialize tracking
best_model = None
best_score = 0.0
best_model_name = ""

print("\nTraining and evaluating models...\n")

# Start MLflow experiment
mlflow.start_run()

# Log model hyperparameters and weights
mlflow.log_param("accuracy_weight", weights["Accuracy"])
mlflow.log_param("precision_weight", weights["Precision"])
mlflow.log_param("recall_weight", weights["Recall"])
mlflow.log_param("f1_weight", weights["F1-Score"])
mlflow.log_param("roc_auc_weight", weights["ROC-AUC"])

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Needed for ROC-AUC

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Weighted Score
    score = (weights['Accuracy'] * acc +
             weights['Precision'] * prec +
             weights['Recall'] * rec +
             weights['F1-Score'] * f1 +
             weights['ROC-AUC'] * auc)

    # Log metrics to MLflow
    mlflow.log_metric(f"{name}_accuracy", acc)
    mlflow.log_metric(f"{name}_precision", prec)
    mlflow.log_metric(f"{name}_recall", rec)
    mlflow.log_metric(f"{name}_f1", f1)
    mlflow.log_metric(f"{name}_roc_auc", auc)
    mlflow.log_metric(f"{name}_weighted_score", score)

    # Display results
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Weighted Score: {score:.4f}")

    # Track best model
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

# Save the best model to MLflow
mlflow.sklearn.log_model(best_model, "best_model")

# End MLflow run
mlflow.end_run()

print("\n Model training complete.")
print("\nBest Model Based on Weighted Healthcare Metrics:")
print(f"{best_model_name} with Weighted Score: {best_score:.4f}")

# Save best model locally
model_save_path = 'models/best_model.pkl'
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
except Exception as e:
    print(" Error saving model:", e)
    traceback.print_exc()
    exit(1)

print(f"\nSaved model at: {os.path.abspath(model_save_path)}")

# Final file existence check
print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
