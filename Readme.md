AI-Based Heart Disease Risk Prediction and Remote Monitoring System


## Overview

Develop an IoT-based wearable device for continuous health monitoring.
Predict heart disease risk using machine learning models.
Improve prediction performance through data preprocessing and feature engineering.
Integrate SHAP for Explainable AI.
Enable remote patient monitoring through a dashboard.
Generate early risk alerts for patients and healthcare professionals.


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

                 START
                   ↓
        Wearable Device Setup
       (ESP32 + Sensors)
                   ↓
            Data Collection
       (HR + SpO₂ + ECG*)
                   ↓
         Data Preprocessing
                   ↓
         Feature Engineering
                   ↓
       Feature Extraction
                   ↓
      Heart Disease Prediction
                   ↓
        Risk Classification
      (Low / Medium / High)
                   ↓
          SHAP Explainability
                   ↓
      Remote Monitoring
           Dashboard
                   ↓
            Risk Alerts
                   ↓
                  END

## Technologies Used

**Hardware**
ESP32
Main microcontroller of the wearable system.
Handles sensor data acquisition.
Provides wireless communication for remote monitoring.
MAX30102
Used for Heart Rate measurement.
Used for SpO₂ measurement.
AD8232 ECG Sensor — Optional
Used for ECG signal acquisition.
Can provide additional cardiac information for prediction.

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

1. Real-Time Wearable Health Monitoring
The system should enable continuous monitoring of physiological parameters through the wearable device.
2. Heart Disease Risk Prediction
Machine-learning models should analyze the collected/processed health data and provide a heart-disease risk prediction.
3. Explainable AI-Based Clinical Decision Support
SHAP should provide explanations for the model's predictions, helping users understand the contribution of different features.
4. Continuous Remote Patient Monitoring
The dashboard should allow health information and prediction results to be monitored remotely.
5. Early Detection of Cardiovascular Risk
The system is intended to identify potentially elevated cardiovascular risk early and generate alerts based on predefined criteria.
---

## Future Enhancements

--Federated Learning
Multiple healthcare devices or institutions could train models collaboratively without directly sharing their raw patient data.
--Mobile Application
The dashboard functionality could be extended into an Android/iOS mobile application.
--Cloud Deployment
The system could be deployed on cloud infrastructure for scalable remote monitoring and data storage.
--Smartwatch Integration
The wearable system could be integrated with smartwatch platforms for easier continuous health monitoring.
--Blockchain-Based Medical Data Security
Blockchain could be explored for improving the security, integrity and controlled sharing of medical data.
