Explainable AI-Based Heart Disease Risk Prediction and Remote Monitoring System

# Explainable AI-Based Wearable Heart Disease Risk Prediction and Remote Monitoring System

## Overview

Cardiovascular disease (CVD) is one of the leading causes of death worldwide. Early detection and continuous monitoring can significantly improve patient outcomes. This project proposes an **IoT-based wearable heart disease risk prediction system** that combines physiological sensor data and machine learning with **Explainable Artificial Intelligence (XAI)** to provide accurate, transparent, and real-time cardiovascular risk assessment.

The wearable device continuously collects health parameters such as **Heart Rate, SpO₂, and ECG**, which are analyzed using machine learning models. The prediction results are explained using **SHAP**, allowing healthcare professionals to understand the factors influencing each prediction. A web-based dashboard enables real-time monitoring and alert generation.

---

## Problem Statement

Most existing heart disease prediction systems focus only on prediction accuracy and lack continuous monitoring, explainability, and integration with wearable devices. Many AI models operate as black boxes, making it difficult for clinicians to interpret predictions.

This project aims to develop an **Explainable AI-based wearable monitoring system** that provides accurate heart disease risk prediction, transparent decision-making, and real-time remote patient monitoring.

---

## Objectives

- Develop an IoT-based wearable device for continuous health monitoring.
- Predict heart disease risk using machine learning models.
- Improve prediction performance through data preprocessing and feature engineering.
- Integrate SHAP for explainable AI.
- Enable remote patient monitoring through a dashboard.
- Generate early risk alerts for patients and healthcare professionals.

---

## Proposed Architecture

```text
Wearable Device (ESP32 + Sensors)
              ↓
      Data Collection
              ↓
     Data Preprocessing
              ↓
 Heart Disease Prediction
 (Random Forest / XGBoost)
              ↓
     SHAP Explainability
              ↓
 Remote Monitoring Dashboard
              ↓
      Risk Alerts
```

---

## Technologies Used

### Hardware
- ESP32
- MAX30102 (Heart Rate & SpO₂)
- AD8232 ECG Sensor (Optional)

### Software
- Python
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- SHAP
- Pandas
- NumPy
- Streamlit / Flask

---

## Expected Outcomes

- Real-time wearable health monitoring.
- Accurate heart disease risk prediction.
- Explainable AI-based clinical decision support.
- Continuous remote patient monitoring.
- Early detection of cardiovascular risk.

---

## Future Enhancements

- Federated Learning
- Mobile Application
- Cloud Deployment
- Smartwatch Integration
- Blockchain-based Medical Data Security
