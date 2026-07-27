"""
Model Trainer Module
====================
Trains 10 ML algorithms and evaluates each with comprehensive metrics.
Supports cross-validation, hyperparameter tuning, and model persistence.
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_model_definitions() -> dict:
    """Return all model definitions with default hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42, min_samples_split=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            probability=True, kernel="rbf", C=1.0, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean"
        ),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
            activation="relu", alpha=0.001, early_stopping=True, n_iter_no_change=15
        ),
    }


@st.cache_resource(show_spinner=False)
def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all 10 models and compute comprehensive metrics.
    Returns: results_df, trained_models, curves_data
    """
    models = get_model_definitions()
    results = []
    trained_models = {}
    curves_data = {}

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():
        start_time = time.time()

        # Train
        model.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 3)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc, 4),
            "CV Mean": round(cv_scores.mean(), 4),
            "CV Std": round(cv_scores.std(), 4),
            "Train Time (s)": train_time,
        })

        trained_models[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": cm,
            "cr": cr,
            "cv_scores": cv_scores,
        }

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr, "tpr": tpr, "roc_auc": roc,
            "precision_curve": prec_curve, "recall_curve": rec_curve,
            "avg_precision": avg_prec,
        }

        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, trained_models, curves_data


def get_best_model_name(results_df: pd.DataFrame) -> str:
    """Return the name of the best model by F1 Score."""
    return results_df["F1 Score"].idxmax()


def get_model_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by composite score (weighted average of all metrics)."""
    df = results_df.copy()
    df["Score"] = (
        df["Accuracy"] * 0.2
        + df["Precision"] * 0.2
        + df["Recall"] * 0.2
        + df["F1 Score"] * 0.2
        + df["ROC-AUC"] * 0.2
    )
    df = df.sort_values("Score", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def get_learning_curves(model, X_train, y_train, model_name: str):
    """Compute learning curves for the given model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1
    )
    return {
        "train_sizes": train_sizes,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
    }


def load_saved_model(model_name: str):
    """Load a saved model from disk."""
    path = os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_')}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def predict_patient(model, scaler, feature_names: list, patient_data: dict):
    """
    Predict heart disease for a single patient.
    Returns: prediction (0/1), probability, risk level
    """
    patient_df = pd.DataFrame([patient_data])[feature_names]
    patient_scaled = scaler.transform(patient_df)
    pred = model.predict(patient_scaled)[0]
    prob = model.predict_proba(patient_scaled)[0][1]

    if prob < 0.35:
        risk_level = "Low"
    elif prob < 0.65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return int(pred), float(prob), risk_level
