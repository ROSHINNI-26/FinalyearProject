"""
Data Loader Module
==================
Loads and caches the UCI Heart Disease dataset from multiple sources.
"""

import os
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st

# Column definitions matching Kaggle heart-disease-data
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

COLUMN_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (0-2)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0-3)",
    "thal": "Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)",
    "target": "Heart Disease (1=Disease, 0=No Disease)"
}

CP_LABELS = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
SLOPE_LABELS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_LABELS = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}
RESTECG_LABELS = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}

DATASET_URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv",
    "https://raw.githubusercontent.com/ronitf/heart-disease-uci/master/heart.csv",
]


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None) -> pd.DataFrame:
    """
    Load the heart disease dataset.
    Priority: uploaded file > local file > remote URL > synthetic UCI data
    """
    # 1. User-uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = _clean_and_validate(df)
        return df

    # 2. Local data file
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
    local_path = os.path.abspath(local_path)
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df = _clean_and_validate(df)
        return df

    # 3. Remote URLs
    for url in DATASET_URLS:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                df = _clean_and_validate(df)
                # Save locally for future use
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                df.to_csv(local_path, index=False)
                return df
        except Exception:
            continue

    # 4. Fallback: generate synthetic UCI-based data
    return _generate_synthetic_data()


def _clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and validate dataset structure."""
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle UCI format where target > 0 means disease
    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only known columns if they exist
    existing = [c for c in COLUMNS if c in df.columns]
    df = df[existing]

    return df.reset_index(drop=True)


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic UCI Heart Disease data as fallback."""
    np.random.seed(42)
    n = 303

    age = np.random.normal(54, 9, n).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n)
    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    chol = np.random.normal(246, 51, n).clip(126, 564).astype(int)
    fbs = np.random.binomial(1, 0.15, n)
    restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.49, 0.03])
    thalach = np.random.normal(150, 23, n).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.33, n)
    oldpeak = np.random.exponential(1.04, n).clip(0, 6.2).round(1)
    slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([0, 1, 2, 3], n, p=[0.01, 0.18, 0.54, 0.27])

    # Logical target based on risk factors
    risk = (
        (age > 55).astype(float) * 0.3
        + (sex == 1).astype(float) * 0.15
        + (cp == 0).astype(float) * 0.3
        + (thalach < 140).astype(float) * 0.25
        + (oldpeak > 1.5).astype(float) * 0.2
        + (exang == 1).astype(float) * 0.25
        + (ca > 0).astype(float) * 0.3
        + (chol > 250).astype(float) * 0.1
        + np.random.normal(0, 0.1, n)
    )
    target = (risk > 0.7).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target
    })
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics about the dataset."""
    return {
        "n_records": len(df),
        "n_features": len(df.columns) - 1,
        "n_disease": int(df["target"].sum()),
        "n_healthy": int((df["target"] == 0).sum()),
        "disease_pct": round(df["target"].mean() * 100, 1),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "age_range": f"{int(df['age'].min())} – {int(df['age'].max())}",
        "avg_age": round(df["age"].mean(), 1),
        "pct_male": round((df["sex"] == 1).mean() * 100, 1),
    }
