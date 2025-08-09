import os
import pickle
import traceback
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Show working directory
print("Current working directory:", os.getcwd())

# Create directory for models
os.makedirs('models', exist_ok=True)

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
    print("Error loading data:", e)
    traceback.print_exc()
    exit(1)

# Define all models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Stacking Ensemble': StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svm', SVC(probability=True))
        ],
        final_estimator=LogisticRegression()
    )
}

# Healthcare weights for scoring
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# MLflow experiment setup
mlflow.set_experiment("Healthcare_Model_Comparison")
best_model = None
best_score = 0.0
best_model_name = ""

print("\nTraining and evaluating models...\n")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Healthcare_Model_Comparison")
with mlflow.start_run():
    mlflow.log_params(weights)

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Weighted healthcare score
        weighted_score = (
            weights['Accuracy'] * acc +
            weights['Precision'] * prec +
            weights['Recall'] * rec +
            weights['F1-Score'] * f1 +
            weights['ROC-AUC'] * roc_auc
        )

        # Log metrics under model name
        mlflow.log_metrics({
            f"{name}_accuracy": acc,
            f"{name}_precision": prec,
            f"{name}_recall": rec,
            f"{name}_f1": f1,
            f"{name}_roc_auc": roc_auc,
            f"{name}_weighted_score": weighted_score
        })

        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Precision    : {prec:.4f}")
        print(f"  Recall       : {rec:.4f}")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  ROC-AUC      : {roc_auc:.4f}")
        print(f"  Weighted Score: {weighted_score:.4f}")

        # Track best model
        if weighted_score > best_score:
            best_score = weighted_score
            best_model = model
            best_model_name = name

    # Save best model to MLflow
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")
    mlflow.set_tag("best_model", best_model_name)
    mlflow.log_metric("best_weighted_score", best_score)

print("\nModel training complete.")
print(f"Best Model Based on Healthcare Metrics: {best_model_name} (Score: {best_score:.4f})")

# Save best model locally
model_save_path = 'models/best_model.pkl'
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
except Exception as e:
    print("Error saving best model locally:", e)
    traceback.print_exc()

print(f"Saved model locally at: {os.path.abspath(model_save_path)}")
print(f"Model file exists: {os.path.exists(model_save_path)}")
