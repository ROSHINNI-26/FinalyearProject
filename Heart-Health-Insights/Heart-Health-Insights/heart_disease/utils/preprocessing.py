"""
Data Preprocessing Module
==========================
Handles missing values, outliers, encoding, scaling, and train-test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
    1. Missing value imputation
    2. Outlier capping (IQR method)
    3. Feature encoding
    4. Standard scaling
    5. Stratified train-test split
    
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names, preprocessing_report
    """
    report = {}
    df_processed = df.copy()

    # ── 1. Missing Values ────────────────────────────────────────────────────
    missing_before = df_processed.isnull().sum()
    report["missing_before"] = missing_before.to_dict()

    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if df_processed[col].dtype in [np.float64, np.int64]:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
            else:
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)

    missing_after = df_processed.isnull().sum()
    report["missing_after"] = missing_after.to_dict()

    # ── 2. Outlier Detection & Capping (IQR) ────────────────────────────────
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    outlier_report = {}
    for col in continuous_cols:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df_processed[col] < lower) | (df_processed[col] > upper)).sum()
            outlier_report[col] = {
                "count": int(n_outliers),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2)
            }
            df_processed[col] = df_processed[col].clip(lower, upper)

    report["outliers"] = outlier_report

    # ── 3. Ensure Correct Data Types ────────────────────────────────────────
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)

    # ── 4. Feature / Target Split ────────────────────────────────────────────
    feature_cols = [c for c in df_processed.columns if c != "target"]
    X = df_processed[feature_cols].copy()
    y = df_processed["target"].copy()

    report["feature_names"] = feature_cols
    report["n_features"] = len(feature_cols)
    report["class_distribution"] = y.value_counts().to_dict()

    # ── 5. Train-Test Split (Stratified) ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    report["train_size"] = len(X_train)
    report["test_size"] = len(X_test)
    report["train_disease_pct"] = round(y_train.mean() * 100, 1)
    report["test_disease_pct"] = round(y_test.mean() * 100, 1)

    # ── 6. Standard Scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    report["scaling"] = {
        col: {"mean": round(float(scaler.mean_[i]), 4), "std": round(float(scaler.scale_[i]), 4)}
        for i, col in enumerate(feature_cols)
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, report


def get_preprocessing_summary(df_raw: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Build a before/after comparison dataframe for display."""
    rows = []
    for col in report.get("feature_names", []):
        row = {"Feature": col}
        if col in report.get("missing_before", {}):
            row["Missing (Before)"] = report["missing_before"][col]
            row["Missing (After)"] = report["missing_after"][col]
        else:
            row["Missing (Before)"] = 0
            row["Missing (After)"] = 0

        if col in report.get("outliers", {}):
            row["Outliers Capped"] = report["outliers"][col]["count"]
        else:
            row["Outliers Capped"] = 0

        if "scaling" in report and col in report["scaling"]:
            row["Scaled Mean"] = report["scaling"][col]["mean"]
            row["Scaled Std"] = report["scaling"][col]["std"]

        rows.append(row)
    return pd.DataFrame(rows)
