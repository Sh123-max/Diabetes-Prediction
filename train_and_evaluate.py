import os
import shutil
import pickle
import traceback
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Current working directory:", os.getcwd())

# ✅ Dynamically get MLflow Tracking URI from environment variable
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

# --- FAIL FAST if MLflow is pointing to local file:// store ---
if mlflow_uri.startswith("file://"):
    raise RuntimeError(
        f"[ERROR] MLFLOW_TRACKING_URI is set to a local path ({mlflow_uri}). "
        "Please point it to the persistent MLflow server, e.g., http://127.0.0.1:5001"
    )

print(f"[INFO] Using MLflow Tracking URI: {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Healthcare_Model_Comparison")

# Clean models folder
model_dir = os.path.join(os.getcwd(), "models")
if os.path.exists(model_dir):
    try:
        shutil.rmtree(model_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: could not remove models directory: {e}")
os.makedirs(model_dir, exist_ok=True)

# Load processed datasets
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

# Ensure DataFrame format
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)

# Models and hyperparameters
tuning_models = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        {"clf__penalty": ["l1", "l2"], "clf__C": [0.1, 1, 10], "clf__solver": ["liblinear"]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [None, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 4]}
    ),
    "Support Vector Machine": (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"], "clf__gamma": [0.001, 0.01, 1]}
    ),
    "K-Nearest Neighbors": (
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [3, 5, 7]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric='logloss', random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    )
}

# Metric weights
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# Autolog for sklearn and xgboost
mlflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
    log_datasets=True,
    exclusive=False,
    disable=False
)

model_results = {}
best_model = None
best_score = 0.0
best_model_name = ""

print("\nStarting GridSearchCV tuning and MLflow logging...\n")

# Loop through models
for name, (model, param_grid) in tuning_models.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("mlflow.runName", name)

        exp_id = mlflow.get_experiment_by_name("Healthcare_Model_Comparison").experiment_id
        run_link = f"{mlflow.get_tracking_uri()}/#/experiments/{exp_id}/runs/{run.info.run_id}"
        print(f"Training with GridSearchCV: {name}")
        print(f"MLflow Run Link: {run_link}")

        grid = GridSearchCV(model, param_grid, scoring='recall', cv=5)
        grid.fit(X_train, y_train)

        best_model_gs = grid.best_estimator_
        best_params = grid.best_params_

        y_pred = best_model_gs.predict(X_test)
        y_proba = best_model_gs.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        }
        weighted_score = sum(weights[m] * metrics[m] for m in weights)
        metrics["Weighted Score"] = weighted_score

        model_results[name] = {"Model": best_model_gs, "Parameters": best_params, "Metrics": metrics}

        mlflow.log_metric("Weighted Score", weighted_score)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, best_model_gs.predict_proba(X_test))
        mlflow.sklearn.log_model(
            sk_model=best_model_gs,
            name="model",
            input_example=input_example,
            signature=signature
        )

        if weighted_score > best_score:
            best_score = weighted_score
            best_model = best_model_gs
            best_model_name = name

# Save best model locally
model_save_path = os.path.join(model_dir, "best_model.pkl")
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
except Exception as e:
    print("Error saving best model locally:", e)
    traceback.print_exc()

print(f"Best Model: {best_model_name} (Score: {best_score:.4f})")
print(f"Saved best model at: {os.path.abspath(model_save_path)}")

# Comparison plots
comparison_df = pd.DataFrame({
    model: {
        "Accuracy": res["Metrics"]["Accuracy"],
        "Precision": res["Metrics"]["Precision"],
        "Recall": res["Metrics"]["Recall"],
        "F1-Score": res["Metrics"]["F1-Score"],
        "ROC-AUC": res["Metrics"]["ROC-AUC"],
        "Weighted Score": res["Metrics"]["Weighted Score"]
    }
    for model, res in model_results.items()
}).T

for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Weighted Score"]:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df.reset_index(), x="index", y=metric, palette="viridis", legend=False)
    plt.title(f"Model {metric} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_file = f"{metric}_comparison.png"
    plt.savefig(plot_file)
    mlflow.log_artifact(plot_file)
    os.remove(plot_file)

print("\nGridSearchCV tuning and MLflow logging complete.")
