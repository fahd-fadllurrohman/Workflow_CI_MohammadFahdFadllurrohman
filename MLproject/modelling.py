"""
modelling.py - untuk MLproject (Kriteria 3)
Heart Disease Dataset
Tracking URI dibaca dari environment variable.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import os

# ── Setup MLflow ─────────────────────────────────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Latihan Heart Disease Classification")

# ── Load Data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
X_test  = pd.read_csv(f'{DATA_DIR}/X_test.csv')
y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv').squeeze()
y_test  = pd.read_csv(f'{DATA_DIR}/y_test.csv').squeeze()

print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

# ── Training ──────────────────────────────────────────────────────────────────
with mlflow.start_run():

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "min_samples_split": 2,
        "criterion": "gini"
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_score_weighted", f1)

    # Artifact: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Sakit', 'Sakit'],
                yticklabels=['Tidak Sakit', 'Sakit'])
    ax.set_title('Confusion Matrix - Heart Disease')
    plt.tight_layout()
    plt.savefig('training_confusion_matrix.png', dpi=150)
    plt.close()
    mlflow.log_artifact('training_confusion_matrix.png')

    # Artifact: Feature Importance
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    feat_imp.sort_values().plot(kind='barh', ax=ax2, color='#e74c3c')
    ax2.set_title('Feature Importances - Heart Disease')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    mlflow.log_artifact('feature_importance.png')

    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
