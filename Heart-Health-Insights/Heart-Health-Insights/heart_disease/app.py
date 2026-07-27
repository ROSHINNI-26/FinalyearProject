"""
Heart Disease Prediction Dashboard
====================================
A professional Streamlit dashboard for heart disease prediction using
the UCI Heart Disease Dataset. Covers EDA, preprocessing, 10 ML models,
explainable AI, and interactive patient risk prediction.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_loader import (
    load_dataset, get_dataset_info,
    CP_LABELS, SLOPE_LABELS, THAL_LABELS, RESTECG_LABELS, COLUMN_DESCRIPTIONS
)
from utils.preprocessing import preprocess_data, get_preprocessing_summary
from utils.model_trainer import (
    train_all_models, get_best_model_name, get_model_ranking,
    get_learning_curves, predict_patient
)
from utils.plots import (
    plot_target_distribution, plot_age_distribution, plot_sex_distribution,
    plot_chest_pain_distribution, plot_correlation_heatmap, plot_missing_values,
    plot_cholesterol_distribution, plot_bp_distribution, plot_heart_rate_distribution,
    plot_feature_histograms, plot_pairplot, plot_confusion_matrix,
    plot_metrics_comparison, plot_roc_curves, plot_pr_curves, plot_radar_chart,
    plot_learning_curve, plot_feature_importance, plot_risk_gauge
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Home",
    "📊 Dataset Overview",
    "🔍 Exploratory Data Analysis",
    "⚙️ Data Preprocessing",
    "🤖 Model Training",
    "🏆 Model Comparison",
    "🧠 Explainable AI",
    "🫀 Patient Prediction",
    "📝 Conclusions",
]

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:12px 0 8px 0;'>
            <span style='font-size:2.8rem;'>❤️</span>
            <h2 style='margin:4px 0 0 0; font-size:1.15rem; letter-spacing:0.02em;'>
                Heart Disease AI
            </h2>
            <p style='font-size:0.78rem; color:#AAA; margin:2px 0 0 0;'>
                UCI Dataset · 10 ML Models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()

    # Dataset upload
    st.markdown("**📁 Upload Dataset**")
    uploaded_file = st.file_uploader(
        "heart.csv (optional)", type=["csv"], label_visibility="collapsed"
    )
    st.caption("Leave empty to use the UCI Heart Disease dataset automatically.")

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem; color:#777; text-align:center;'>"
        "BE/BTech Final-Year Project<br>Heart Disease Prediction · 2024"
        "</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading dataset…"):
    df = load_dataset(uploaded_file)

info = get_dataset_info(df)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (cached — needed in multiple pages)
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test, scaler, feature_names, prep_report = preprocess_data(df)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #1a0a0a 0%, #2d1515 50%, #1a0a0a 100%);
                    border-radius:16px; padding:40px 48px; margin-bottom:24px;
                    border:1px solid #4a1a1a;'>
            <h1 style='margin:0; font-size:2.4rem; letter-spacing:-0.02em;'>
                ❤️ Heart Disease Prediction
            </h1>
            <p style='margin:12px 0 0 0; font-size:1.1rem; color:#CCC; max-width:640px;'>
                An end-to-end machine learning pipeline that detects cardiovascular disease
                risk using the UCI Heart Disease dataset — covering EDA, 10 algorithms,
                explainable AI, and real-time patient risk assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{info['n_records']:,}")
    c2.metric("Features", info["n_features"])
    c3.metric("Disease Cases", f"{info['n_disease']:,}")
    c4.metric("Disease Rate", f"{info['disease_pct']}%")
    c5.metric("Missing Values", info["missing_values"])

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📌 Project Overview")
        st.markdown("""
        This project demonstrates a **complete ML lifecycle** for clinical decision support:

        | Phase | Details |
        |---|---|
        | **Dataset** | UCI Heart Disease (Cleveland) |
        | **Records** | 303–1,025 rows, 13 features |
        | **Task** | Binary Classification (Disease / No Disease) |
        | **Models** | 10 algorithms (LR, DT, RF, SVM, KNN, NB, XGB, GB, Ada, MLP) |
        | **Explainability** | SHAP, Feature Importance, Permutation Importance |
        | **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, CV |

        **Use the sidebar** to navigate through each stage of the pipeline.
        """)

    with col2:
        st.subheader("📋 Dataset Features")
        feat_df = pd.DataFrame([
            {"Feature": k, "Description": v}
            for k, v in COLUMN_DESCRIPTIONS.items()
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=380)

    st.divider()
    st.subheader("🗺️ Pipeline Overview")
    steps = [
        ("1", "📥", "Data Loading", "UCI Heart Disease CSV"),
        ("2", "🔍", "EDA", "Distributions, Correlations"),
        ("3", "⚙️", "Preprocessing", "Imputation, Scaling, Split"),
        ("4", "🤖", "Model Training", "10 ML Algorithms"),
        ("5", "🏆", "Comparison", "Leaderboard, ROC, Radar"),
        ("6", "🧠", "Explainability", "SHAP, Feature Importance"),
        ("7", "🫀", "Prediction", "Interactive Risk Assessment"),
    ]
    cols = st.columns(len(steps))
    for col, (num, icon, title, desc) in zip(cols, steps):
        col.markdown(
            f"""<div style='background:#1A1F2E; border-radius:10px; padding:16px 10px;
                        text-align:center; border:1px solid #2A3050;'>
                <div style='font-size:1.6rem;'>{icon}</div>
                <div style='font-size:0.7rem; color:#E84855; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.05em;'>Step {num}</div>
                <div style='font-weight:700; margin:4px 0 2px 0; font-size:0.9rem;'>{title}</div>
                <div style='font-size:0.72rem; color:#AAA;'>{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ██████  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['n_records']:,}")
    c2.metric("Features", info["n_features"])
    c3.metric("Missing Values", info["missing_values"])
    c4.metric("Duplicates", info["duplicates"])

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📄 Interactive Viewer", "📈 Statistics", "📐 Data Types"])

    with tab1:
        # Search & filter
        col1, col2, col3 = st.columns([2, 1, 1])
        search_col = col1.selectbox("Filter by column", ["All"] + list(df.columns))
        target_filter = col2.selectbox("Target", ["All", "Disease (1)", "No Disease (0)"])
        n_show = col3.slider("Rows to display", 10, len(df), 50)

        df_view = df.copy()
        if target_filter == "Disease (1)":
            df_view = df_view[df_view["target"] == 1]
        elif target_filter == "No Disease (0)":
            df_view = df_view[df_view["target"] == 0]

        st.dataframe(df_view.head(n_show), use_container_width=True, height=420)
        st.caption(f"Showing {min(n_show, len(df_view))} of {len(df_view)} records")

    with tab2:
        st.subheader("Statistical Summary")
        stats = df.describe().T.round(3)
        stats.insert(0, "dtype", df.dtypes)
        stats.insert(1, "missing", df.isnull().sum())
        st.dataframe(stats, use_container_width=True)

    with tab3:
        st.subheader("Data Types & Null Counts")
        dtype_df = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target & Demographics",
        "📊 Feature Distributions",
        "🔗 Correlations",
        "🔵 Advanced Plots",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_target_distribution(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_sex_distribution(df), use_container_width=True)
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
        st.plotly_chart(plot_chest_pain_distribution(df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_cholesterol_distribution(df), use_container_width=True)
        st.plotly_chart(plot_bp_distribution(df), use_container_width=True)
        st.plotly_chart(plot_heart_rate_distribution(df), use_container_width=True)
        st.plotly_chart(plot_feature_histograms(df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_missing_values(df), use_container_width=True)
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        st.markdown("""
        **Key Correlations with Heart Disease (target):**
        - 🔴 `cp` (chest pain type), `thalach` (max heart rate), `slope` — positive correlation
        - 🔵 `exang` (exercise angina), `oldpeak`, `ca` (vessels), `thal` — negative correlation
        """)

    with tab4:
        with st.spinner("Generating pairplot (this may take a moment)…"):
            st.plotly_chart(plot_pairplot(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Data Preprocessing":
    st.title("⚙️ Data Preprocessing")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Samples", prep_report["train_size"])
    c2.metric("Test Samples", prep_report["test_size"])
    c3.metric("Train Disease %", f"{prep_report['train_disease_pct']}%")
    c4.metric("Test Disease %", f"{prep_report['test_disease_pct']}%")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩹 Missing Values", "📐 Outliers", "📏 Scaling", "🔀 Train/Test Split"
    ])

    with tab1:
        st.subheader("Missing Value Imputation")
        missing_df = pd.DataFrame({
            "Feature": list(prep_report["missing_before"].keys()),
            "Before Imputation": list(prep_report["missing_before"].values()),
            "After Imputation": list(prep_report["missing_after"].values()),
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        st.info("ℹ️ Strategy: Median imputation for numeric features, mode for categorical features.")

    with tab2:
        st.subheader("Outlier Detection (IQR Method)")
        outlier_rows = []
        for feat, info_o in prep_report.get("outliers", {}).items():
            outlier_rows.append({
                "Feature": feat,
                "Outliers Capped": info_o["count"],
                "Lower Bound": info_o["lower_bound"],
                "Upper Bound": info_o["upper_bound"],
            })
        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)
        st.info("ℹ️ Strategy: IQR-based capping (Winsorization). Values beyond Q1−1.5×IQR and Q3+1.5×IQR are capped.")

    with tab3:
        st.subheader("Standard Scaling (Z-Score Normalization)")
        scaling_rows = [
            {"Feature": feat, "Mean (μ)": v["mean"], "Std Dev (σ)": v["std"]}
            for feat, v in prep_report.get("scaling", {}).items()
        ]
        st.dataframe(pd.DataFrame(scaling_rows), use_container_width=True, hide_index=True)
        st.info("ℹ️ StandardScaler fitted on training data; same parameters applied to test data.")

    with tab4:
        st.subheader("Stratified Train-Test Split (80/20)")
        col1, col2 = st.columns(2)
        with col1:
            split_data = pd.DataFrame({
                "Split": ["Train", "Test"],
                "Samples": [prep_report["train_size"], prep_report["test_size"]],
                "Disease %": [prep_report["train_disease_pct"], prep_report["test_disease_pct"]],
            })
            st.dataframe(split_data, hide_index=True, use_container_width=True)
            st.success("✅ Stratified sampling preserves class ratios in both splits.")
        with col2:
            fig = px.pie(split_data, values="Samples", names="Split",
                         title="Train/Test Split",
                         color_discrete_sequence=["#3A86FF", "#E84855"])
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    # Full preprocessing summary
    st.divider()
    st.subheader("📋 Full Preprocessing Report")
    summary_df = get_preprocessing_summary(df, prep_report)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    st.info(
        "Training 10 machine learning algorithms on the preprocessed dataset. "
        "Results are cached — refresh the page to retrain."
    )

    with st.spinner("Training all 10 models… (first run may take ~30 seconds)"):
        results_df, trained_models, curves_data = train_all_models(
            X_train.values, X_test.values, y_train.values, y_test.values, feature_names
        )

    st.success(f"✅ All 10 models trained successfully! Best model: **{get_best_model_name(results_df)}**")
    st.divider()

    # Model selector
    selected_model = st.selectbox("Select a model to inspect:", list(trained_models.keys()))
    model_data = trained_models[selected_model]

    c1, c2, c3, c4, c5 = st.columns(5)
    row = results_df.loc[selected_model]
    c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
    c2.metric("Precision", f"{row['Precision']:.4f}")
    c3.metric("Recall", f"{row['Recall']:.4f}")
    c4.metric("F1 Score", f"{row['F1 Score']:.4f}")
    c5.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔢 Confusion Matrix", "📄 Classification Report",
        "📉 Learning Curve", "🔄 Cross-Validation"
    ])

    with tab1:
        st.plotly_chart(
            plot_confusion_matrix(model_data["cm"], selected_model),
            use_container_width=True
        )

    with tab2:
        st.subheader(f"Classification Report — {selected_model}")
        st.code(model_data["cr"], language="text")

    with tab3:
        with st.spinner("Computing learning curve…"):
            lc = get_learning_curves(
                model_data["model"],
                X_train.values, y_train.values,
                selected_model
            )
        st.plotly_chart(plot_learning_curve(lc, selected_model), use_container_width=True)

    with tab4:
        cv = model_data["cv_scores"]
        st.subheader(f"5-Fold Cross-Validation — {selected_model}")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv))],
            "Accuracy": cv.round(4)
        })
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(cv_df, hide_index=True, use_container_width=True)
            st.metric("CV Mean ± Std", f"{cv.mean():.4f} ± {cv.std():.4f}")
        with col2:
            fig = px.bar(cv_df, x="Fold", y="Accuracy",
                         title="Cross-Validation Scores",
                         color="Accuracy", color_continuous_scale="Blues",
                         text=cv_df["Accuracy"])
            fig.add_hline(y=cv.mean(), line_dash="dash", line_color="#E84855",
                          annotation_text=f"Mean={cv.mean():.4f}")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📊 All Models Summary")
    st.dataframe(results_df.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV Mean"],
        color="#2a3a2a"
    ).highlight_min(
        subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV Mean"],
        color="#3a2a2a"
    ), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")

    with st.spinner("Loading model results…"):
        results_df, trained_models, curves_data = train_all_models(
            X_train.values, X_test.values, y_train.values, y_test.values, feature_names
        )

    best = get_best_model_name(results_df)
    worst = results_df["F1 Score"].idxmin()
    ranking = get_model_ranking(results_df)
    most_balanced = (results_df["Precision"] - results_df["Recall"]).abs().idxmin()

    col1, col2, col3 = st.columns(3)
    col1.success(f"🥇 **Best Model:** {best}  \nF1 = {results_df.loc[best, 'F1 Score']:.4f}")
    col2.error(f"🔴 **Needs Improvement:** {worst}  \nF1 = {results_df.loc[worst, 'F1 Score']:.4f}")
    col3.info(f"⚖️ **Most Balanced:** {most_balanced}  \nPrec≈Rec")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏅 Leaderboard", "📊 Metric Charts", "🕸️ Radar Chart",
        "📈 ROC Curves", "📉 Precision-Recall Curves"
    ])

    with tab1:
        st.subheader("🏅 Algorithm Leaderboard (Composite Score)")
        ranking_display = ranking[["Rank", "Accuracy", "Precision", "Recall",
                                   "F1 Score", "ROC-AUC", "CV Mean", "Score"]].copy()
        ranking_display = ranking_display.reset_index()
        ranking_display.columns = [c.replace("Model", "Algorithm") for c in ranking_display.columns]

        def style_rank(row):
            if row["Rank"] == 1:
                return ["background-color: rgba(232,72,85,0.2)"] * len(row)
            elif row["Rank"] == len(ranking_display):
                return ["background-color: rgba(100,100,100,0.1)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            ranking_display.style.apply(style_rank, axis=1),
            use_container_width=True, hide_index=True
        )

    with tab2:
        metric_choice = st.selectbox(
            "Select metric:", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        )
        st.plotly_chart(
            plot_metrics_comparison(results_df, metric_choice),
            use_container_width=True
        )

    with tab3:
        st.plotly_chart(plot_radar_chart(results_df), use_container_width=True)
        st.caption("Radar chart comparing all models across 5 key metrics. Larger area = better overall performance.")

    with tab4:
        st.plotly_chart(plot_roc_curves(curves_data), use_container_width=True)

    with tab5:
        st.plotly_chart(plot_pr_curves(curves_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████  EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI")
    st.markdown(
        "Understand **why** the model makes each prediction. "
        "Uses SHAP (SHapley Additive exPlanations) and permutation-based importance."
    )

    with st.spinner("Training models for explainability…"):
        results_df, trained_models, curves_data = train_all_models(
            X_train.values, X_test.values, y_train.values, y_test.values, feature_names
        )

    best_name = get_best_model_name(results_df)
    best_model = trained_models[best_name]["model"]

    st.info(f"🥇 Explaining: **{best_name}** (best model by F1 Score = {results_df.loc[best_name, 'F1 Score']:.4f})")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🌍 Global Importance", "🔬 SHAP Analysis", "🔄 Permutation Importance"])

    with tab1:
        st.subheader("Global Feature Importance")
        importances = {}
        if hasattr(best_model, "feature_importances_"):
            for feat, imp in zip(feature_names, best_model.feature_importances_):
                importances[feat] = round(float(imp), 5)
        elif hasattr(best_model, "coef_"):
            for feat, coef in zip(feature_names, best_model.coef_[0]):
                importances[feat] = round(abs(float(coef)), 5)
        else:
            # Permutation importance fallback
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(best_model, X_test.values, y_test.values,
                                        n_repeats=10, random_state=42)
            for feat, imp in zip(feature_names, pi.importances_mean):
                importances[feat] = round(max(0, float(imp)), 5)

        st.plotly_chart(
            plot_feature_importance(importances, f"Feature Importance — {best_name}"),
            use_container_width=True
        )

        st.markdown("""
        **Medical Interpretation of Top Features:**

        | Feature | Medical Significance |
        |---|---|
        | `cp` (Chest Pain) | Asymptomatic chest pain is a strong indicator of silent CAD |
        | `thalach` (Max HR) | Lower maximum HR → reduced cardiac reserve → disease risk |
        | `ca` (Vessels) | More blocked vessels → higher severity of coronary artery disease |
        | `oldpeak` (ST Dep.) | Greater depression → more severe ischemia during stress |
        | `thal` (Thalassemia) | Reversible defects indicate inadequate blood supply to heart muscle |
        | `exang` (Ex. Angina) | Exercise-induced angina is a classic symptom of ischemia |
        """)

    with tab2:
        st.subheader("SHAP Analysis")
        try:
            import shap
            import matplotlib.pyplot as plt
            import io
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            from xgboost import XGBClassifier

            TREE_TYPES = (
                RandomForestClassifier, GradientBoostingClassifier,
                DecisionTreeClassifier, XGBClassifier,
            )

            with st.spinner("Computing SHAP values…"):
                # Use a consistent slice — 50 samples is enough for both explainers
                X_shap = X_test.values[:50]

                if isinstance(best_model, TREE_TYPES):
                    explainer = shap.TreeExplainer(best_model)
                    raw = explainer.shap_values(X_shap)
                    # sklearn tree models return list[2]; XGBoost returns array
                    shap_vals = raw[1] if isinstance(raw, list) else raw
                    base_val = (
                        explainer.expected_value[1]
                        if isinstance(explainer.expected_value, (list, np.ndarray))
                        else float(explainer.expected_value)
                    )
                else:
                    # Model-agnostic — works for AdaBoost, SVM, KNN, MLP, LR …
                    background = shap.sample(X_train.values, 50)
                    explainer = shap.KernelExplainer(best_model.predict_proba, background)
                    raw = explainer.shap_values(X_shap, nsamples=100)
                    shap_vals = raw[1] if isinstance(raw, list) else raw
                    ev = explainer.expected_value
                    base_val = (
                        float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)
                    )

            # ── Summary Plot ────────────────────────────────────────────────
            st.subheader("SHAP Summary Plot")
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            fig_shap.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            shap.summary_plot(shap_vals, X_shap, feature_names=feature_names,
                              show=False, plot_size=None)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0E1117")
            plt.close()
            buf.seek(0)
            st.image(buf, use_container_width=True)

            # ── Waterfall for a single patient ───────────────────────────────
            st.subheader("SHAP Waterfall — Single Patient Explanation")
            n_available = len(shap_vals)
            patient_idx = st.slider("Select patient index", 0, n_available - 1, 0)

            shap_exp = shap.Explanation(
                values=shap_vals[patient_idx],
                base_values=base_val,
                data=X_shap[patient_idx],
                feature_names=feature_names,
            )
            fig_wf, ax2 = plt.subplots(figsize=(10, 5))
            fig_wf.patch.set_facecolor("#0E1117")
            ax2.set_facecolor("#0E1117")
            shap.waterfall_plot(shap_exp, show=False)
            buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0E1117")
            plt.close()
            buf2.seek(0)
            st.image(buf2, use_container_width=True)

            # ── Mean |SHAP| feature ranking ──────────────────────────────────
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importance = dict(sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda x: x[1], reverse=True
            ))
            st.plotly_chart(
                plot_feature_importance(shap_importance, "Mean |SHAP Value| — Global Importance"),
                use_container_width=True,
            )

        except Exception as e:
            st.warning(f"SHAP computation error: {e}")
            st.info("Falling back to standard feature importance visualization above.")

    with tab3:
        st.subheader("Permutation Feature Importance")
        with st.spinner("Computing permutation importance…"):
            from sklearn.inspection import permutation_importance
            pi_result = permutation_importance(
                best_model, X_test.values, y_test.values,
                n_repeats=15, random_state=42, n_jobs=-1
            )
        perm_imp = {
            feat: round(float(imp), 5)
            for feat, imp in zip(feature_names, pi_result.importances_mean)
        }
        st.plotly_chart(
            plot_feature_importance(perm_imp, "Permutation Importance — Feature Contribution to Accuracy"),
            use_container_width=True
        )

        perm_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Importance": pi_result.importances_mean.round(5),
            "Std Dev": pi_result.importances_std.round(5),
        }).sort_values("Mean Importance", ascending=False)
        st.dataframe(perm_df, hide_index=True, use_container_width=True)
        st.caption(
            "Permutation importance = drop in accuracy when a feature's values are randomly shuffled. "
            "Higher value → feature is more important."
        )


# ══════════════════════════════════════════════════════════════════════════════
# ██████  PATIENT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🫀 Patient Prediction":
    st.title("🫀 Patient Risk Prediction")
    st.markdown(
        "Enter patient clinical values to predict the **probability of heart disease** "
        "and receive a personalized risk assessment."
    )

    with st.spinner("Preparing prediction models…"):
        results_df, trained_models, curves_data = train_all_models(
            X_train.values, X_test.values, y_train.values, y_test.values, feature_names
        )

    best_name = get_best_model_name(results_df)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("👤 Patient Input Form")
        with st.form("prediction_form"):
            st.markdown("**Demographics**")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (years)", min_value=20, max_value=90, value=55, step=1)
            sex = c2.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])

            st.markdown("**Cardiac Measurements**")
            c3, c4 = st.columns(2)
            trestbps = c3.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=130)
            chol = c4.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=245)
            thalach = st.slider("Max Heart Rate (bpm)", min_value=60, max_value=210, value=150)
            oldpeak = st.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

            st.markdown("**Clinical Indicators**")
            c5, c6 = st.columns(2)
            cp = c5.selectbox("Chest Pain Type", list(CP_LABELS.items()), format_func=lambda x: f"{x[0]}: {x[1]}")
            slope = c6.selectbox("ST Slope", list(SLOPE_LABELS.items()), format_func=lambda x: f"{x[0]}: {x[1]}")

            c7, c8 = st.columns(2)
            restecg = c7.selectbox("Rest ECG", list(RESTECG_LABELS.items()), format_func=lambda x: f"{x[0]}: {x[1]}")
            thal = c8.selectbox("Thalassemia", list(THAL_LABELS.items()), format_func=lambda x: f"{x[0]}: {x[1]}")

            c9, c10, c11 = st.columns(3)
            fbs = c9.selectbox("Fasting Blood Sugar >120", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            exang = c10.selectbox("Exercise Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            ca = c11.number_input("Major Vessels (ca)", min_value=0, max_value=3, value=0, step=1)

            model_choice = st.selectbox("Prediction Model", [best_name] + [m for m in trained_models if m != best_name])
            submitted = st.form_submit_button("🔬 Predict Heart Disease Risk", use_container_width=True)

    with col_right:
        st.subheader("📊 Prediction Result")

        if submitted:
            patient_data = {
                "age": age, "sex": sex[1], "cp": cp[0], "trestbps": trestbps,
                "chol": chol, "fbs": fbs[1], "restecg": restecg[0],
                "thalach": thalach, "exang": exang[1], "oldpeak": oldpeak,
                "slope": slope[0], "ca": ca, "thal": thal[0]
            }

            selected_model_obj = trained_models[model_choice]["model"]
            prediction, probability, risk_level = predict_patient(
                selected_model_obj, scaler, feature_names, patient_data
            )

            # Risk gauge
            st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)

            # Result card
            if prediction == 1:
                st.error(
                    f"🚨 **Heart Disease DETECTED**\n\n"
                    f"Risk Probability: **{probability*100:.1f}%**  \n"
                    f"Risk Level: **{risk_level}**"
                )
            else:
                st.success(
                    f"✅ **No Heart Disease Detected**\n\n"
                    f"Risk Probability: **{probability*100:.1f}%**  \n"
                    f"Risk Level: **{risk_level}**"
                )

            st.metric("Model Used", model_choice)
            st.metric("Model Accuracy", f"{results_df.loc[model_choice, 'Accuracy']*100:.2f}%")

            # Top contributing features
            model_obj = trained_models[model_choice]["model"]
            if hasattr(model_obj, "feature_importances_"):
                imp_pairs = sorted(
                    zip(feature_names, model_obj.feature_importances_),
                    key=lambda x: x[1], reverse=True
                )[:5]
                st.subheader("🔑 Top Contributing Features")
                for rank, (feat, imp) in enumerate(imp_pairs, 1):
                    desc = COLUMN_DESCRIPTIONS.get(feat, feat)
                    st.markdown(f"**{rank}.** `{feat}` — {desc}  \nImportance: `{imp:.4f}`")

            st.divider()
            st.markdown("**⚕️ Clinical Note:**  \n"
                        "_This tool provides decision support only. "
                        "All clinical decisions must be verified by a licensed medical professional._")
        else:
            st.markdown(
                """
                <div style='background:#1A1F2E; border-radius:12px; padding:32px;
                            text-align:center; border:1px solid #2A3050; margin-top:20px;'>
                    <div style='font-size:3rem;'>🫀</div>
                    <h3 style='margin:12px 0 8px 0;'>Ready to Predict</h3>
                    <p style='color:#AAA;'>Fill in the patient form on the left<br>
                    and click <strong>Predict Heart Disease Risk</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Sample patients for quick testing
    st.divider()
    st.subheader("📋 Sample Patients for Quick Testing")
    samples = pd.DataFrame([
        {"Scenario": "High Risk Male", "age": 67, "sex": 1, "cp": 0, "trestbps": 160,
         "chol": 286, "fbs": 0, "restecg": 0, "thalach": 108, "exang": 1,
         "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2},
        {"Scenario": "Low Risk Female", "age": 45, "sex": 0, "cp": 2, "trestbps": 118,
         "chol": 195, "fbs": 0, "restecg": 1, "thalach": 172, "exang": 0,
         "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 0},
        {"Scenario": "Moderate Risk Male", "age": 58, "sex": 1, "cp": 1, "trestbps": 140,
         "chol": 241, "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0,
         "oldpeak": 0.5, "slope": 1, "ca": 1, "thal": 1},
    ])
    st.dataframe(samples, use_container_width=True, hide_index=True)
    st.caption("Use these values in the input form above for a quick demonstration.")


# ══════════════════════════════════════════════════════════════════════════════
# ██████  CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Conclusions":
    st.title("📝 Conclusions & Final Report")

    with st.spinner("Generating final report…"):
        results_df, trained_models, curves_data = train_all_models(
            X_train.values, X_test.values, y_train.values, y_test.values, feature_names
        )

    best = get_best_model_name(results_df)
    ranking = get_model_ranking(results_df)
    best_row = results_df.loc[best]

    st.subheader("🏆 Best Algorithm")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Algorithm", best)
    c2.metric("Accuracy", f"{best_row['Accuracy']*100:.2f}%")
    c3.metric("F1 Score", f"{best_row['F1 Score']:.4f}")
    c4.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}")
    c5.metric("CV Score", f"{best_row['CV Mean']:.4f} ± {best_row['CV Std']:.4f}")

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Final Ranking")
        rank_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Score"]
        st.dataframe(
            ranking[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Score"]],
            use_container_width=True
        )

    with col2:
        st.subheader("🔑 Important Features")
        model_obj = trained_models[best]["model"]
        if hasattr(model_obj, "feature_importances_"):
            top_features = sorted(
                zip(feature_names, model_obj.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:6]
            for rank, (feat, imp) in enumerate(top_features, 1):
                st.markdown(
                    f"**{rank}. `{feat}`** ({COLUMN_DESCRIPTIONS.get(feat, '')}): `{imp:.4f}`"
                )

    st.divider()
    st.subheader("🏥 Medical Interpretation")
    st.markdown(f"""
    The **{best}** model achieved **{best_row['Accuracy']*100:.2f}% accuracy** in detecting
    heart disease, demonstrating the power of ensemble methods in clinical decision support.

    **Key Clinical Findings:**

    - **Chest pain type (cp)** emerged as a top predictor. Asymptomatic chest pain paradoxically
      carries high risk — patients without classic symptoms may be undiagnosed.
    - **Maximum heart rate (thalach)** inversely correlates with disease risk. A lower peak rate
      during stress testing indicates impaired cardiac reserve.
    - **ST depression (oldpeak)** and **ST slope** are strong ECG markers of ischemia during exercise.
    - **Number of major vessels (ca)** directly reflects coronary artery disease severity.
    - **Thalassemia type** (reversible defect) indicates intermittent reduced blood supply.
    - **Exercise-induced angina (exang)** is a classic symptom of obstructive coronary disease.

    **Risk Stratification:**

    | Risk Level | Probability Range | Recommended Action |
    |---|---|---|
    | Low | 0 – 35% | Routine annual screening |
    | Moderate | 35 – 65% | Stress test + lipid panel |
    | High | 65 – 100% | Immediate cardiology referral |
    """)

    st.divider()
    st.subheader("🚀 Future Scope")
    st.markdown("""
    1. **Deep Learning** — Implement CNN/LSTM on ECG time-series for temporal pattern detection
    2. **Multi-class Classification** — Predict disease severity (0-4 scale) instead of binary
    3. **Federated Learning** — Train across multiple hospitals without sharing patient data
    4. **Real-time EHR Integration** — Connect to Electronic Health Record systems via HL7/FHIR
    5. **Explainability Enhancement** — Integrate LIME for model-agnostic local explanations
    6. **Uncertainty Quantification** — Add confidence intervals using Monte Carlo Dropout
    7. **Clinical Validation** — Prospective study with cardiologist review for validation
    8. **Mobile App** — React Native app for point-of-care risk assessment in rural clinics
    """)

    st.divider()
    st.subheader("📚 References")
    st.markdown("""
    1. Detrano, R., et al. (1989). *International application of a new probability algorithm for the
       diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304-310.
    2. UCI Machine Learning Repository: Heart Disease Dataset.
       https://archive.ics.uci.edu/ml/datasets/heart+disease
    3. Lundberg, S.M., & Lee, S.I. (2017). *A unified approach to interpreting model predictions.*
       NeurIPS 2017.
    4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD 2016.
    """)

    st.divider()
    st.markdown(
        """
        <div style='background:#1A1F2E; border-radius:12px; padding:24px 32px;
                    text-align:center; border:1px solid #2A3050;'>
            <h3 style='margin:0 0 8px 0;'>❤️ Heart Disease Prediction System</h3>
            <p style='color:#AAA; margin:0;'>
                BE/BTech Final-Year Project · Machine Learning in Healthcare Analytics<br>
                Built with Python · scikit-learn · XGBoost · SHAP · Streamlit · Plotly
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
