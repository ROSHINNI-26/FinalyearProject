"""
Visualization Module
====================
All Plotly charts for the Heart Disease Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

PALETTE = px.colors.qualitative.Set1
DISEASE_COLORS = {"No Disease": "#2ECC71", "Disease": "#E84855"}

# ──────────────────────────────────────────────────────────────────────────────
# EDA CHARTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["target"].value_counts().reset_index()
    counts.columns = ["target", "count"]
    counts["label"] = counts["target"].map({0: "No Disease", 1: "Heart Disease"})
    fig = px.pie(
        counts, values="count", names="label",
        title="Target Distribution: Heart Disease vs. No Disease",
        color="label",
        color_discrete_map={"No Disease": "#2ECC71", "Heart Disease": "#E84855"},
        hole=0.45
    )
    fig.update_traces(textinfo="percent+label+value", textfont_size=14)
    fig.update_layout(showlegend=True, height=420)
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="age", color=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        nbins=20, barmode="overlay", opacity=0.75,
        title="Age Distribution by Heart Disease Status",
        labels={"color": "Status", "age": "Age (years)"},
        color_discrete_map=DISEASE_COLORS
    )
    fig.update_layout(height=420, bargap=0.05)
    return fig


def plot_sex_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["sex", "target"]).size().reset_index(name="count")
    counts["sex_label"] = counts["sex"].map({0: "Female", 1: "Male"})
    counts["target_label"] = counts["target"].map({0: "No Disease", 1: "Heart Disease"})
    fig = px.bar(
        counts, x="sex_label", y="count", color="target_label",
        barmode="group",
        title="Heart Disease Prevalence by Sex",
        labels={"sex_label": "Sex", "count": "Count", "target_label": "Status"},
        color_discrete_map=DISEASE_COLORS
    )
    fig.update_layout(height=420)
    return fig


def plot_chest_pain_distribution(df: pd.DataFrame) -> go.Figure:
    cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"}
    counts = df.groupby(["cp", "target"]).size().reset_index(name="count")
    counts["cp_label"] = counts["cp"].map(cp_map)
    counts["target_label"] = counts["target"].map({0: "No Disease", 1: "Heart Disease"})
    fig = px.bar(
        counts, x="cp_label", y="count", color="target_label",
        barmode="group",
        title="Chest Pain Type vs. Heart Disease",
        labels={"cp_label": "Chest Pain Type", "count": "Count", "target_label": "Status"},
        color_discrete_map=DISEASE_COLORS
    )
    fig.update_layout(height=420)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=550,
        xaxis={"tickangle": -30},
    )
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    fig = px.bar(
        missing, x="Feature", y="Missing Count",
        color="Missing %",
        title="Missing Values by Feature",
        text="Missing Count",
        color_continuous_scale="Reds"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400)
    return fig


def plot_cholesterol_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        y="chol", color=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        title="Cholesterol Distribution by Heart Disease Status",
        labels={"x": "Status", "chol": "Cholesterol (mg/dl)", "color": "Status"},
        color_discrete_map=DISEASE_COLORS,
        points="outliers"
    )
    fig.update_layout(height=420)
    return fig


def plot_bp_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df, x=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        y="trestbps", color=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        box=True, points="all",
        title="Resting Blood Pressure Distribution by Status",
        labels={"x": "Status", "trestbps": "Resting BP (mm Hg)", "color": "Status"},
        color_discrete_map=DISEASE_COLORS
    )
    fig.update_layout(height=420)
    return fig


def plot_heart_rate_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="thalach", nbins=20,
        color=df["target"].map({0: "No Disease", 1: "Heart Disease"}),
        barmode="overlay", opacity=0.7,
        title="Maximum Heart Rate Distribution",
        labels={"thalach": "Max Heart Rate (bpm)", "color": "Status"},
        color_discrete_map=DISEASE_COLORS
    )
    fig.update_layout(height=420)
    return fig


def plot_feature_histograms(df: pd.DataFrame) -> go.Figure:
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]
    for idx, col in enumerate(numeric_cols):
        r, c = positions[idx]
        for target_val, label, color in [(0,"No Disease","#2ECC71"),(1,"Heart Disease","#E84855")]:
            data = df[df["target"] == target_val][col]
            fig.add_trace(
                go.Histogram(x=data, name=label, marker_color=color,
                             opacity=0.65, showlegend=(idx == 0)),
                row=r, col=c
            )
    fig.update_layout(barmode="overlay", height=550,
                      title_text="Feature Histograms by Disease Status")
    return fig


def plot_pairplot(df: pd.DataFrame) -> go.Figure:
    cols = ["age", "thalach", "chol", "oldpeak", "trestbps"]
    available = [c for c in cols if c in df.columns]
    df_plot = df[available + ["target"]].copy()
    df_plot["Status"] = df_plot["target"].map({0: "No Disease", 1: "Heart Disease"})
    fig = px.scatter_matrix(
        df_plot, dimensions=available, color="Status",
        color_discrete_map=DISEASE_COLORS,
        title="Pairplot: Key Continuous Features",
        opacity=0.55, height=700
    )
    fig.update_traces(diagonal_visible=False, marker_size=3)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION CHARTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["No Disease", "Heart Disease"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont={"size": 20},
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=380
    )
    return fig


def plot_metrics_comparison(results_df: pd.DataFrame, metric: str) -> go.Figure:
    df = results_df[[metric]].reset_index()
    df = df.sort_values(metric, ascending=False)
    colors = [
        "#E84855" if i == 0 else "#3A86FF" if i == len(df) - 1 else "#6C757D"
        for i in range(len(df))
    ]
    fig = px.bar(
        df, x="Model", y=metric,
        title=f"{metric} Comparison Across Models",
        text=df[metric].round(4),
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=450, yaxis_range=[0, 1.05],
        showlegend=False,
        xaxis={"tickangle": -20}
    )
    return fig


def plot_roc_curves(curves_data: dict) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, (name, data) in enumerate(curves_data.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f'{name} (AUC={data["roc_auc"]:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Random Classifier", showlegend=True
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=550,
        legend=dict(x=0.55, y=0.05)
    )
    return fig


def plot_pr_curves(curves_data: dict) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, (name, data) in enumerate(curves_data.items()):
        fig.add_trace(go.Scatter(
            x=data["recall_curve"], y=data["precision_curve"],
            mode="lines",
            name=f'{name} (AP={data["avg_precision"]:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=550,
        legend=dict(x=0.01, y=0.05)
    )
    return fig


def plot_radar_chart(results_df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, model_name in enumerate(results_df.index):
        row = results_df.loc[model_name]
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model_name,
            line=dict(color=colors[i % len(colors)]),
            opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart: Algorithm Comparison",
        height=600,
        showlegend=True
    )
    return fig


def plot_learning_curve(lc_data: dict, model_name: str) -> go.Figure:
    sizes = lc_data["train_sizes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=lc_data["train_mean"],
        mode="lines+markers", name="Training Score",
        line=dict(color="#E84855", width=2),
        error_y=dict(type="data", array=lc_data["train_std"], visible=True)
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=lc_data["val_mean"],
        mode="lines+markers", name="Validation Score",
        line=dict(color="#3A86FF", width=2),
        error_y=dict(type="data", array=lc_data["val_std"], visible=True)
    ))
    fig.update_layout(
        title=f"Learning Curve — {model_name}",
        xaxis_title="Training Samples",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        height=420
    )
    return fig


def plot_feature_importance(importance_dict: dict, title: str = "Feature Importance") -> go.Figure:
    df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
    df = df.sort_values("Importance", ascending=True)
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=500, coloraxis_showscale=False)
    return fig


def plot_risk_gauge(probability: float) -> go.Figure:
    """Render a gauge chart showing patient risk level."""
    pct = probability * 100
    if pct < 35:
        color = "#2ECC71"
        risk = "Low Risk"
    elif pct < 65:
        color = "#F39C12"
        risk = "Moderate Risk"
    else:
        color = "#E84855"
        risk = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Heart Disease Risk — <b>{risk}</b>", "font": {"size": 20}},
        delta={"reference": 50, "increasing": {"color": "#E84855"}, "decreasing": {"color": "#2ECC71"}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 35], "color": "rgba(46,204,113,0.15)"},
                {"range": [35, 65], "color": "rgba(243,156,18,0.15)"},
                {"range": [65, 100], "color": "rgba(232,72,85,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": pct
            },
        }
    ))
    fig.update_layout(height=350, margin=dict(t=60, b=10, l=20, r=20))
    return fig
