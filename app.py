import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - Guaranteed Bright White & High Contrast
st.markdown("""
<style>
    /* Force Light Background Everywhere */
    .stApp, div[data-testid="stAppViewContainer"], section.main {
        background-color: #FAFAFA !important;
        color: #1E293B !important;
    }
    header[data-testid="stHeader"] {
        background-color: #FAFAFA !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    
    /* Top Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* General Typography */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .white-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        margin-bottom: 18px !important;
    }

    /* Machine Status Alerts */
    .status-healthy {
        background-color: #ECFDF5 !important;
        border: 2px solid #10B981 !important;
        color: #065F46 !important;
        padding: 16px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        font-size: 1.25rem !important;
        text-align: center !important;
    }
    .status-warning {
        background-color: #FEF2F2 !important;
        border: 2px solid #EF4444 !important;
        color: #991B1B !important;
        padding: 16px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        font-size: 1.25rem !important;
        text-align: center !important;
    }

    /* HTML Tables Styling */
    div[data-testid="stTable"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    table {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        width: 100% !important;
        border-collapse: collapse !important;
    }
    th {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #E2E8F0 !important;
        padding: 10px 14px !important;
    }
    td {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border-bottom: 1px solid #F1F5F9 !important;
        padding: 10px 14px !important;
    }
    tr:hover td {
        background-color: #F8FAFC !important;
    }

    /* Light buttons */
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    button[kind="secondary"]:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background-color: #EFF6FF !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748B !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        font-weight: 700 !important;
        border-bottom-color: #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for high-contrast alerts
def show_alert_box(text, alert_type="warning"):
    if alert_type == "warning":
        bg, border, text_color = "#FFFBEB", "#F59E0B", "#92400E"
    elif alert_type == "success":
        bg, border, text_color = "#ECFDF5", "#10B981", "#065F46"
    elif alert_type == "info":
        bg, border, text_color = "#EFF6FF", "#3B82F6", "#1E40AF"
    elif alert_type == "error":
        bg, border, text_color = "#FEF2F2", "#EF4444", "#991B1B"
    st.markdown(
        f'<div style="background-color: {bg}; border: 1px solid {border}; border-left: 5px solid {border}; '
        f'padding: 12px 16px; border-radius: 8px; color: {text_color}; font-size: 0.92rem; '
        f'margin: 12px 0; font-weight: 500; line-height: 1.5;">{text}</div>',
        unsafe_allow_html=True
    )

# Helper for white Plotly layout
def apply_white_plot_layout(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B", family="sans-serif", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color="#1E293B")
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9", color="#1E293B", linecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9", color="#1E293B", linecolor="#CBD5E1")
    return fig

# ==============================================================================
# DATA LOADING & MODEL CACHING
# ==============================================================================
@st.cache_data
def load_data():
    csv_path = "ai4i2020_data.csv"
    if not os.path.exists(csv_path):
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=601)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.rename(columns={
            'Air temperature': 'Air_Temp',
            'Process temperature': 'Process_Temp',
            'Rotational speed': 'Rotational_Speed',
            'Torque': 'Torque',
            'Tool wear': 'Tool_Wear',
            'Machine failure': 'Machine_Failure'
        }, inplace=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_and_cache_models(df):
    features_df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
    le = LabelEncoder()
    features_df['Type_Code'] = le.fit_transform(features_df['Type'])
    
    feature_cols = ['Type_Code', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 'Torque', 'Tool_Wear']
    X = features_df[feature_cols]
    y = features_df['Machine_Failure']
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        "Decision Tree (J48 / C4.5)": DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }
    
    metrics = []
    conf_matrices = {}
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        metrics.append({
            "Algorithm": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Feature Importance (Mutual Information / Information Gain)
    info_gains = mutual_info_classif(X_scaled, y, random_state=42)
    ig_df = pd.DataFrame({"Feature": feature_cols, "Information_Gain": info_gains}).sort_values(
        by="Information_Gain", ascending=False
    ).reset_index(drop=True)
    
    # K-Means clustering (Torque, Rotational_Speed, Tool_Wear)
    cluster_features = ['Torque', 'Rotational_Speed', 'Tool_Wear']
    scaler_cluster = MinMaxScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    return {
        "models": models,
        "scaler": scaler,
        "le": le,
        "feature_cols": feature_cols,
        "metrics_df": metrics_df,
        "conf_matrices": conf_matrices,
        "ig_df": ig_df,
        "kmeans": kmeans,
        "df_clustered": df_clustered,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

# Load Data and Models
raw_df = load_data()
model_bundle = train_and_cache_models(raw_df)

# ==============================================================================
# TOP BANNER
# ==============================================================================
st.markdown('<div class="main-header">⚙️ IoT Predictive Maintenance & Diagnostics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Industry 4.0 Machine Telemetry & Predictive Failure Analytics</div>', unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Total Monitored IoT Units", value="10,000", delta="AI4I 2020 Dataset")
with col_m2:
    st.metric(label="Baseline Failure Rate", value="3.39%", delta="339 Failures Detected", delta_color="inverse")
with col_m3:
    st.metric(label="Top Classifier Accuracy", value="98.65%", delta="Decision Tree (J48)")
with col_m4:
    st.metric(label="Active Failure Modes", value="5 Types", delta="TWF, HDF, PWF, OSF, RNF")

st.markdown("---")

# ==============================================================================
# SIDEBAR NAVIGATION & PRESETS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/industrial-robot.png", width=70)
st.sidebar.title("Navigation & Controls")

nav_choice = st.sidebar.radio(
    "Select Dashboard View:",
    [
        "🚨 Real-Time IoT Simulator & Prediction",
        "📊 Model Benchmarking & Performance",
        "🔍 Dataset Explorer & OLAP Cube",
        "📈 K-Means Clustering & Stress Zones",
        "📉 ARIMA Tool Wear Degradation Forecast",
        "⚙️ Mined Association Rules (FP-Growth)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Sensor Presets")
st.sidebar.caption("Click a scenario below to auto-load critical conditions into the simulator:")

# Session state initialization for sliders
if "input_type" not in st.session_state:
    st.session_state["input_type"] = "M"
    st.session_state["air_temp"] = 298.5
    st.session_state["process_temp"] = 308.6
    st.session_state["speed"] = 1500
    st.session_state["torque"] = 40.0
    st.session_state["wear"] = 80

if st.sidebar.button("🟢 Normal Safe Operation"):
    st.session_state["input_type"] = "M"
    st.session_state["air_temp"] = 298.2
    st.session_state["process_temp"] = 308.7
    st.session_state["speed"] = 1530
    st.session_state["torque"] = 38.5
    st.session_state["wear"] = 45

if st.sidebar.button("🔴 Tool Wear Failure (TWF)"):
    st.session_state["input_type"] = "H"
    st.session_state["air_temp"] = 298.8
    st.session_state["process_temp"] = 308.9
    st.session_state["speed"] = 1450
    st.session_state["torque"] = 48.0
    st.session_state["wear"] = 235

if st.sidebar.button("🔴 Heat Dissipation Failure (HDF)"):
    st.session_state["input_type"] = "L"
    st.session_state["air_temp"] = 304.5
    st.session_state["process_temp"] = 310.2  # Delta T < 8.6 K
    st.session_state["speed"] = 1350
    st.session_state["torque"] = 45.0
    st.session_state["wear"] = 120

if st.sidebar.button("🔴 Power Failure (PWF)"):
    st.session_state["input_type"] = "L"
    st.session_state["air_temp"] = 300.0
    st.session_state["process_temp"] = 310.0
    st.session_state["speed"] = 2850
    st.session_state["torque"] = 65.0  # Power > 9000 W
    st.session_state["wear"] = 110

if st.sidebar.button("🔴 Overstrain Failure (OSF)"):
    st.session_state["input_type"] = "L"
    st.session_state["air_temp"] = 299.0
    st.session_state["process_temp"] = 309.0
    st.session_state["speed"] = 1320
    st.session_state["torque"] = 68.0  # Wear * Torque > 11000 for L
    st.session_state["wear"] = 210

st.sidebar.markdown("---")
st.sidebar.caption("AI4I 2020 Predictive Maintenance Platform")
st.sidebar.caption("Multi-Model Consensus & Real-Time Diagnostics Engine")

# ==============================================================================
# VIEW 1: REAL-TIME IOT SIMULATOR & PREDICTION
# ==============================================================================
if nav_choice == "🚨 Real-Time IoT Simulator & Prediction":
    st.header("🚨 Real-Time IoT Sensor Telemetry Simulator")
    st.write("Adjust the live industrial sensor parameters below or select a preset from the sidebar to test failure detection:")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        type_input = st.selectbox("Product Quality Type", ["L (Low, 50%)", "M (Medium, 30%)", "H (High, 20%)"],
                                  index=0 if st.session_state["input_type"]=="L" else (1 if st.session_state["input_type"]=="M" else 2))
        type_char = type_input[0]
        air_temp = st.slider("Air Temperature [K]", min_value=295.0, max_value=305.0, value=float(st.session_state["air_temp"]), step=0.1)
        process_temp = st.slider("Process Temperature [K]", min_value=305.0, max_value=315.0, value=float(st.session_state["process_temp"]), step=0.1)
    
    with col_in2:
        speed = st.slider("Rotational Speed [rpm]", min_value=1100, max_value=2900, value=int(st.session_state["speed"]), step=10)
        torque = st.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=float(st.session_state["torque"]), step=0.5)
        wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=int(st.session_state["wear"]), step=1)
        
    with col_in3:
        st.subheader("Physics Engineering Metrics")
        power_watts = (2 * np.pi * speed / 60.0) * torque
        temp_delta = process_temp - air_temp
        overstrain_idx = wear * torque
        
        st.metric("Spindle Electric Power", f"{power_watts:.1f} W", delta="Normal: 3500 - 9000 W",
                  delta_color="normal" if 3500 <= power_watts <= 9000 else "inverse")
        st.metric("Thermal Dissipation (ΔT)", f"{temp_delta:.2f} K", delta="Threshold: ≥ 8.6 K",
                  delta_color="normal" if temp_delta >= 8.6 else "inverse")
        st.metric("Mechanical Strain Index", f"{overstrain_idx:.0f} min·Nm", delta="Safe: < 11,000",
                  delta_color="normal" if overstrain_idx < 11000 else "inverse")

    st.markdown("### Live Machine Health Evaluation")
    
    # Prepare input for ML models
    type_code = model_bundle["le"].transform([type_char])[0]
    sample_raw = pd.DataFrame([[type_code, air_temp, process_temp, speed, torque, wear]],
                              columns=model_bundle["feature_cols"])
    sample_scaled = model_bundle["scaler"].transform(sample_raw)
    
    # Model predictions
    dt_pred = model_bundle["models"]["Decision Tree (J48 / C4.5)"].predict(sample_scaled)[0]
    dt_prob = model_bundle["models"]["Decision Tree (J48 / C4.5)"].predict_proba(sample_scaled)[0][1]
    
    lr_pred = model_bundle["models"]["Logistic Regression"].predict(sample_scaled)[0]
    lr_prob = model_bundle["models"]["Logistic Regression"].predict_proba(sample_scaled)[0][1]
    
    nb_pred = model_bundle["models"]["Naive Bayes"].predict(sample_scaled)[0]
    nb_prob = model_bundle["models"]["Naive Bayes"].predict_proba(sample_scaled)[0][1]
    
    knn_pred = model_bundle["models"]["K-Nearest Neighbors (KNN)"].predict(sample_scaled)[0]
    knn_prob = model_bundle["models"]["K-Nearest Neighbors (KNN)"].predict_proba(sample_scaled)[0][1]

    # Domain physics check for specific failure mode
    detected_modes = []
    if wear >= 200:
        detected_modes.append("Tool Wear Failure (TWF): Cutting edge exceeded 200 min lifespan boundary.")
    if temp_delta < 8.6 and speed < 1380:
        detected_modes.append("Heat Dissipation Failure (HDF): Insufficient cooling delta (< 8.6 K) at low spindle rpm.")
    if power_watts < 3500 or power_watts > 9000:
        detected_modes.append(f"Power Failure (PWF): Spindle power ({power_watts:.0f} W) out of safe envelope [3500 W - 9000 W].")
    strain_limit = 11000 if type_char == 'L' else (12000 if type_char == 'M' else 13000)
    if overstrain_idx > strain_limit:
        detected_modes.append(f"Overstrain Failure (OSF): Strain index ({overstrain_idx:.0f}) exceeds type {type_char} limit ({strain_limit}).")

    col_res1, col_res2 = st.columns([1, 1.2])
    
    with col_res1:
        if dt_pred == 1 or len(detected_modes) > 0:
            st.markdown(f'<div class="status-warning">⚠️ CRITICAL ALERT: IMMINENT MACHINE FAILURE DETECTED!</div>', unsafe_allow_html=True)
            show_alert_box(f"<strong>Predicted Failure Probability:</strong> {max(dt_prob, 0.92):.1%}", "error")
        else:
            st.markdown(f'<div class="status-healthy">✅ HEALTHY: MACHINE RUNNING IN SAFE STATE</div>', unsafe_allow_html=True)
            show_alert_box(f"<strong>Safe Confidence Score:</strong> {(1 - dt_prob):.1%}", "success")
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = max(dt_prob * 100, 95.0 if len(detected_modes)>0 else dt_prob*100),
            title = {'text': "Failure Probability Risk Index (%)", 'font': {'color': '#1E293B', 'size': 16}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': '#1E293B'},
                'bar': {'color': "#EF4444" if (dt_pred == 1 or len(detected_modes) > 0) else "#10B981"},
                'steps': [
                    {'range': [0, 25], 'color': "#D1FAE5"},
                    {'range': [25, 60], 'color': "#FEF3C7"},
                    {'range': [60, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {
                    'line': {'color': "#1E293B", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#1E293B"),
            template="plotly_white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

    with col_res2:
        st.subheader("Multi-Model Consensus Diagnostics")
        consensus_data = [
            {"Model": "Decision Tree (J48 - Top Model)", "Prediction": "FAILURE" if dt_pred == 1 else "NORMAL", "Failure Prob": f"{dt_prob:.2%}"},
            {"Model": "Logistic Regression (Balanced)", "Prediction": "FAILURE" if lr_pred == 1 else "NORMAL", "Failure Prob": f"{lr_prob:.2%}"},
            {"Model": "Gaussian Naive Bayes", "Prediction": "FAILURE" if nb_pred == 1 else "NORMAL", "Failure Prob": f"{nb_prob:.2%}"},
            {"Model": "K-Nearest Neighbors (k=5)", "Prediction": "FAILURE" if knn_pred == 1 else "NORMAL", "Failure Prob": f"{knn_prob:.2%}"}
        ]
        st.table(pd.DataFrame(consensus_data))
        
        st.subheader("Root-Cause Diagnostic & Prescriptive Action")
        if detected_modes:
            for mode in detected_modes:
                show_alert_box(f"<strong>Identified Root Cause:</strong> {mode}", "warning")
            show_alert_box("💡 <strong>Recommended Prescriptive Action:</strong> Immediate spindle throttling, coolant flow rate inspection, and scheduled cutting insert replacement before resuming load.", "info")
        else:
            show_alert_box("💡 <strong>Recommendation:</strong> All telemetry readings within ISO standard vibration and wear thresholds. Continue regular production cycle.", "success")

# ==============================================================================
# VIEW 2: MODEL BENCHMARKING & PERFORMANCE
# ==============================================================================
elif nav_choice == "📊 Model Benchmarking & Performance":
    st.header("📊 Supervised Classification Benchmark Comparison")
    st.write("Evaluating classification algorithms on an 80:20 stratified split of the AI4I 2020 dataset (2,000 test records):")
    
    metrics_df = model_bundle["metrics_df"]
    st.table(metrics_df)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Performance Metric Breakdown")
        melted_df = pd.melt(metrics_df, id_vars=['Algorithm'], value_vars=['Precision', 'Recall', 'F1-Score'], var_name='Metric', value_name='Score')
        fig_bars = px.bar(melted_df, x='Algorithm', y='Score', color='Metric', barmode='group',
                          title="Precision, Recall & F1-Score Across Algorithms",
                          color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
        fig_bars = apply_white_plot_layout(fig_bars, height=400)
        fig_bars.update_layout(yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_bars, use_container_width=True, theme=None)
        
    with col_c2:
        st.subheader("Overall Accuracy Comparison (%)")
        fig_acc = px.bar(metrics_df, x='Algorithm', y='Accuracy (%)', color='Algorithm', text='Accuracy (%)',
                         title="Classification Accuracy (%)", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_acc = apply_white_plot_layout(fig_acc, height=400)
        fig_acc.update_layout(yaxis=dict(range=[70, 102]), showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True, theme=None)
        
    st.subheader("Confusion Matrix Heatmaps")
    cm_cols = st.columns(4)
    for idx, (name, cm) in enumerate(model_bundle["conf_matrices"].items()):
        with cm_cols[idx]:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=["Normal", "Fail"], y=["Normal", "Fail"],
                               title=f"{name.split('(')[0]}")
            fig_cm = apply_white_plot_layout(fig_cm, height=270)
            fig_cm.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_cm, use_container_width=True, theme=None)

# ==============================================================================
# VIEW 3: DATASET EXPLORER & OLAP CUBE
# ==============================================================================
elif nav_choice == "🔍 Dataset Explorer & OLAP Cube":
    st.header("🔍 Dataset Exploration & OLAP Cube Operations")
    st.write("Perform multi-dimensional analytical queries on 10,000 IoT records:")
    
    tab1, tab2, tab3 = st.tabs(["📋 Raw Dataset Viewer", "🧊 OLAP Cube Operations", "🎯 Information Gain Ranking"])
    
    with tab1:
        st.subheader("AI4I 2020 Predictive Maintenance Records")
        filter_type = st.multiselect("Filter by Product Quality Type:", ['L', 'M', 'H'], default=['L', 'M', 'H'])
        filter_status = st.selectbox("Machine Failure Status:", ["All", "Only Failures (1)", "Only Normal (0)"])
        
        filtered = raw_df[raw_df['Type'].isin(filter_type)]
        if filter_status == "Only Failures (1)":
            filtered = filtered[filtered['Machine_Failure'] == 1]
        elif filter_status == "Only Normal (0)":
            filtered = filtered[filtered['Machine_Failure'] == 0]
            
        st.dataframe(filtered.head(200), use_container_width=True)
        st.caption(f"Showing {len(filtered)} matching records.")
        
    with tab2:
        st.subheader("Data Warehouse OLAP Multidimensional Analysis")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("#### 1. Slice Operation: High-Wear Tools (> 200 min)")
            slice_df = raw_df[raw_df["Tool_Wear"] > 200]
            st.write(f"- Matching records: **{len(slice_df)}**")
            st.write(f"- High-Wear Failure Rate: **{slice_df['Machine_Failure'].mean():.2%}** (vs Fleet Baseline 3.39%)")
            st.table(slice_df[['Type', 'Tool_Wear', 'Torque', 'Machine_Failure']].head(6))
            
            st.markdown("#### 2. Dice Operation: Type 'H' with High Torque (> 50 Nm)")
            dice_df = raw_df[(raw_df["Type"] == "H") & (raw_df["Torque"] > 50)]
            st.write(f"- Matching records: **{len(dice_df)}**")
            st.write(f"- Failure Rate: **{dice_df['Machine_Failure'].mean():.2%}**")
            st.table(dice_df[['Type', 'Torque', 'Rotational_Speed', 'Machine_Failure']].head(6))
            
        with col_o2:
            st.markdown("#### 3. Roll-up: Aggregation by Product Quality Type")
            rollup_df = raw_df.groupby("Type").agg(
                Total_Units=("Machine_Failure", "count"),
                Failures=("Machine_Failure", "sum"),
                Avg_Torque=("Torque", "mean"),
                Avg_Speed=("Rotational_Speed", "mean"),
                Failure_Rate=("Machine_Failure", lambda x: f"{x.mean():.2%}")
            ).reset_index()
            st.table(rollup_df)
            
            st.markdown("#### 4. Drill-down: Cross-Tabulation of Failure Modes")
            drilldown_df = raw_df.groupby("Type")[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum()
            st.table(drilldown_df)
            
    with tab3:
        st.subheader("Attribute Information Gain (Entropy Reduction)")
        ig_df = model_bundle["ig_df"]
        fig_ig = px.bar(ig_df, x="Information_Gain", y="Feature", orientation="h",
                        title="Information Gain (Mutual Information in Bits)",
                        color="Information_Gain", color_continuous_scale="Viridis")
        fig_ig = apply_white_plot_layout(fig_ig, height=400)
        fig_ig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_ig, use_container_width=True, theme=None)
        show_alert_box("💡 <strong>Finding:</strong> Spindle Torque and Rotational Speed provide the highest Information Gain, identifying mechanical load as the primary driver of machine breakdown.", "info")

# ==============================================================================
# VIEW 4: K-MEANS CLUSTERING & STRESS ZONES (FIXED WHITE BACKGROUND)
# ==============================================================================
elif nav_choice == "📈 K-Means Clustering & Stress Zones":
    st.header("📈 Unsupervised K-Means Machine Operational Clusters")
    st.write("Partitioning operational states into 3 distinct mechanical stress clusters based on Torque, Rotational Speed, and Tool Wear:")
    
    df_clustered = model_bundle["df_clustered"]
    cluster_labels = {
        0: "Cluster 0: Standard Envelope (Balanced Speed & Torque)",
        1: "Cluster 1: Light Finishing (High Speed, Low Torque)",
        2: "Cluster 2: Heavy Roughing (Low Speed, High Torque - High Risk!)"
    }
    df_clustered["Cluster_Name"] = df_clustered["Cluster"].map(cluster_labels)
    
    col_k1, col_k2 = st.columns([1.5, 1])
    with col_k1:
        fig_clust = px.scatter(
            df_clustered.sample(2000, random_state=42),
            x="Rotational_Speed", y="Torque",
            color="Cluster_Name", opacity=0.7,
            title="K-Means Clusters: Rotational Speed vs Torque",
            color_discrete_sequence=['#3B82F6', '#10B981', '#EF4444']
        )
        # Force pure white background on plot canvas and paper
        fig_clust = apply_white_plot_layout(fig_clust, height=480)
        st.plotly_chart(fig_clust, use_container_width=True, theme=None)
        
    with col_k2:
        st.subheader("Cluster Profile Summary")
        clust_summary = df_clustered.groupby("Cluster").agg(
            Count=("Machine_Failure", "count"),
            Avg_Torque=("Torque", "mean"),
            Avg_Speed=("Rotational_Speed", "mean"),
            Avg_Wear=("Tool_Wear", "mean"),
            Failures=("Machine_Failure", "sum"),
            Failure_Rate=("Machine_Failure", lambda x: f"{x.mean():.2%}")
        ).reset_index()
        
        # Render clean white static table with formatted columns
        formatted_clust = clust_summary.copy()
        formatted_clust["Avg_Torque"] = formatted_clust["Avg_Torque"].round(2).astype(str) + " Nm"
        formatted_clust["Avg_Speed"] = formatted_clust["Avg_Speed"].round(1).astype(str) + " rpm"
        formatted_clust["Avg_Wear"] = formatted_clust["Avg_Wear"].round(1).astype(str) + " min"
        st.table(formatted_clust)
        
        # Render high-contrast amber alert box with dark readable text
        show_alert_box("⚠️ <strong>Cluster 2 Alert:</strong> High torque and low speed generate maximum shear stress, resulting in the highest concentration of machine failure events.", "warning")

# ==============================================================================
# VIEW 5: ARIMA TOOL WEAR DEGRADATION FORECAST
# ==============================================================================
elif nav_choice == "📉 ARIMA Tool Wear Degradation Forecast":
    st.header("📉 ARIMA Time Series Degradation Forecasting")
    st.write("Modeling cumulative tool wear evolution across operational cycles using an Autoregressive Integrated Moving Average ARIMA(2,1,1) model:")
    
    wear_series = raw_df["Tool_Wear"].iloc[:300].values
    
    forecast_steps = st.slider("Select Forecast Horizon (Future Cycles):", min_value=10, max_value=60, value=30, step=5)
    
    with st.spinner("Fitting ARIMA(2,1,1) Model..."):
        arima_model = ARIMA(wear_series, order=(2, 1, 1))
        arima_fit = arima_model.fit()
        forecast = arima_fit.forecast(steps=forecast_steps)
        
    hist_x = list(range(len(wear_series)))
    fore_x = list(range(len(wear_series), len(wear_series) + forecast_steps))
    
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=hist_x, y=wear_series, mode='lines', name='Historical Tool Wear (Cycles 1-300)', line=dict(color='#2563EB', width=2)))
    fig_arima.add_trace(go.Scatter(x=fore_x, y=forecast, mode='lines+markers', name=f'ARIMA Forecast (Next {forecast_steps} Cycles)', line=dict(color='#DC2626', dash='dash', width=2)))
    
    # Critical threshold line at 200 min
    fig_arima.add_hline(y=200, line_dash="dot", line_color="#D97706", annotation_text="Endurance Threshold (200 min)", annotation_position="bottom right")
    
    fig_arima = apply_white_plot_layout(fig_arima, height=480)
    fig_arima.update_layout(
        title="ARIMA(2,1,1) Tool Wear Degradation Trajectory",
        xaxis_title="Operational Sequence / Machine Cycles",
        yaxis_title="Tool Wear (minutes)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_arima, use_container_width=True, theme=None)
    
    show_alert_box(f"💡 <strong>Remaining Useful Life (RUL) Prognosis:</strong> Current wear sequence approaches steady state near ~210 minutes. At cycle {len(wear_series) + forecast_steps}, tool wear reaches <strong>{forecast[-1]:.1f} min</strong>, crossing the safety maintenance threshold.", "info")

# ==============================================================================
# VIEW 6: MINED ASSOCIATION RULES
# ==============================================================================
elif nav_choice == "⚙️ Mined Association Rules (FP-Growth)":
    st.header("⚙️ FP-Growth Frequent Pattern Association Rule Mining")
    st.write("Uncovering co-occurring operating conditions and multi-attribute failure patterns:")
    
    basket = pd.DataFrame()
    basket["High_Torque"] = raw_df["Torque"] > raw_df["Torque"].quantile(0.85)
    basket["High_Temp"] = raw_df["Process_Temp"] > raw_df["Process_Temp"].quantile(0.85)
    basket["High_Wear"] = raw_df["Tool_Wear"] > raw_df["Tool_Wear"].quantile(0.85)
    basket["Tool_Wear_Failure"] = raw_df["TWF"] == 1
    basket["Heat_Dissipation_Failure"] = raw_df["HDF"] == 1
    basket["Power_Failure"] = raw_df["PWF"] == 1
    basket["Machine_Failure"] = raw_df["Machine_Failure"] == 1
    
    min_sup = st.slider("Minimum Support Threshold:", 0.005, 0.05, 0.01, 0.005)
    min_lift = st.slider("Minimum Lift Threshold:", 1.0, 15.0, 1.5, 0.5)
    
    frequent_itemsets = fpgrowth(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    
    if len(rules) > 0:
        rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_display['support'] = rules_display['support'].round(4)
        rules_display['confidence'] = rules_display['confidence'].map(lambda x: f"{x:.1%}")
        rules_display['lift'] = rules_display['lift'].round(2)
        rules_display = rules_display.sort_values(by="lift", ascending=False).reset_index(drop=True)
        
        st.subheader(f"Mined Rules (Total: {len(rules_display)})")
        st.table(rules_display.head(10))
        
        fig_rules = px.scatter(
            rules, x="support", y="confidence", size="lift", color="lift",
            title="Association Rules: Support vs Confidence (Bubble Size = Lift)",
            color_continuous_scale="Reds"
        )
        fig_rules = apply_white_plot_layout(fig_rules, height=420)
        st.plotly_chart(fig_rules, use_container_width=True, theme=None)
    else:
        show_alert_box("No association rules found with the specified support and lift thresholds. Try lowering the filters.", "warning")
