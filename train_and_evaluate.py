import os
import pickle
import traceback
import mlflow
import mlflow.sklearn
from pathlib import Path
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

# ------------------------------
# Models + Parameter Grids
# ------------------------------
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
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    )
}

# Healthcare metric weights
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# ------------------------------
# MLflow Setup
# ------------------------------
mlflow.set_tracking_uri("file:///var/lib/jenkins/workspace/train_and_evaluate/mlruns")
mlflow.set_experiment("Healthcare_Model_Comparison")

model_results = {}
best_model = None
best_score = 0.0
best_model_name = ""

# ------------------------------
# GridSearchCV Training + Logging
# ------------------------------
print("\nStarting GridSearchCV tuning and MLflow logging...\n")

with mlflow.start_run(run_name="mlflow_gridsearch_with_plots"):
    mlflow.log_params(weights)

    for name, (model, param_grid) in tuning_models.items():
        print(f"Training with GridSearchCV: {name}")
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

        model_results[name] = {
            "Model": best_model_gs,
            "Parameters": best_params,
            "Metrics": metrics
        }

        mlflow.log_param(f"{name}_model", name)
        for param, val in best_params.items():
            mlflow.log_param(f"{name}_{param}", val)
        for metric, val in metrics.items():
            mlflow.log_metric(f"{name}_{metric}", val)

        if weighted_score > best_score:
            best_score = weighted_score
            best_model = best_model_gs
            best_model_name = name

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, best_model_gs.predict_proba(X_test))
        mlflow.sklearn.log_model(
            sk_model=best_model_gs,
            artifact_path=f"{name}_gridsearch_model",
            input_example=input_example,
            signature=signature
        )

    # Save best tuned model locally
    model_save_path = 'models/best_model.pkl'
    try:
        with open(model_save_path, 'wb') as f:
            pickle.dump(best_model, f)
    except Exception as e:
        print("Error saving best model locally:", e)
        traceback.print_exc()
    print(f"Best Model: {best_model_name} (Score: {best_score:.4f})")
    print(f"Saved best model at: {os.path.abspath(model_save_path)}")

    # Create comparison dataframe
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

    # Plot and log all metrics
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Weighted Score"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=comparison_df.reset_index(), x="index", y=metric, palette="viridis")
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
