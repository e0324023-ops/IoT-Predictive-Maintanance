# ⚙️ IoT Predictive Maintenance & Real-Time Failure Diagnostics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-oriented **Industrial IoT (IIoT) Predictive Maintenance and Failure Diagnostics Framework** designed for CNC milling and high-precision machining systems. Built on the **AI4I 2020 Predictive Maintenance Benchmark Dataset (UCI ID: 601)**, this platform pairs thermodynamic and electromechanical physics feature engineering with multi-model machine learning consensus inference, unsupervised stress regime clustering, association rule mining, and time-series degradation forecasting.

---

## 📌 Architecture & Diagnostic Workflow

```
+---------------------------------------------------------------------------------------------------+
|                                  PHYSICAL MACHINING ENCLOSURE                                     |
|  [Air Temp]   [Process Temp]   [Spindle Rotational Speed]   [Cutting Torque]   [Tool Wear Cycles] |
+-------------------------------------------------+-------------------------------------------------+
                                                  | Multi-Sensor Streaming Ingestion
                                                  v
+---------------------------------------------------------------------------------------------------+
|                            STAGE 1: THERMODYNAMIC & KINETIC PHYSICS ENGINE                        |
|  - Spindle Cutting Power: P = (2 * pi * Speed * Torque) / 60 [Watts]                             |
|  - Thermal Dissipation Margin: Delta_T = Process_Temp - Air_Temp [Kelvin]                         |
|  - Mechanical Overstrain Product: S = Tool_Wear * Torque [min * Nm]                               |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                            STAGE 2: PREPROCESSING & LEAKAGE ISOLATION                             |
|  - Product Variant Encoding (L: 60%, M: 30%, H: 10%)                                              |
|  - Min-Max Feature Scaling [0, 1] across dynamic physical channels                                |
|  - Failure Mode Isolation (TWF, HDF, PWF, OSF, RNF strictly shielded from training vector)        |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                            STAGE 3: MULTI-MODEL CONSENSUS INFERENCE ENGINE                        |
|   +--------------------+  +--------------------+  +--------------------+  +-------------------+   |
|   |   Decision Tree    |  | Logistic Regression|  |    Naive Bayes     |  |    KNN (k = 5)    |   |
|   |    (J48 / C4.5)    |  |  (Balanced Class)  |  |     (Gaussian)     |  | (Euclidean Metric)|   |
|   |  Accuracy: 98.65%  |  |   Recall: 83.82%   |  |   Lightweight NB   |  | Local Clustering  |   |
|   +---------+----------+  +---------+----------+  +---------+----------+  +---------+---------+   |
|             |                       |                       |                       |             |
|             +-----------------------+-----------+-----------+-----------------------+             |
|                                                 |                                                 |
|                                                 v                                                 |
|                   Weighted Consensus Risk Score: P_risk = Sum(w_i * P_fail_i)                    |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        STAGE 4: REAL-TIME HEALTH GAUGE & PRESCRIPTIVE REMEDIES                    |
|  - Safe Zone (P_risk < 20%): Normal Operation, nominal ISO vibration/thermal margins             |
|  - Caution Zone (20% <= P_risk < 50%): Preventative maintenance alert, schedule tool check       |
|  - Critical Zone (P_risk >= 50%): Automated emergency shutdown advisory + root-cause remedy       |
+---------------------------------------------------------------------------------------------------+
```

---

## 🚀 Key Modules & Capabilities

### 1. 🚨 Interactive Real-Time Sensor Telemetry Simulator
* Dynamically adjust environmental air temperature, cutting process temperature, spindle speed, torque load, and cumulative tool wear.
* Instant calculation of derived physical metrics: **Spindle Power (Watts)**, **Thermal Margin $\Delta T$ (K)**, and **Mechanical Overstrain ($min \cdot Nm$)**.
* Live health gauge with instantaneous consensus breakdown risk probability and prescribed engineering actions.

![Simulator Interface](assets/dash_simulator.png)

---

### 2. 📊 Supervised Machine Learning Benchmark
Benchmarked across an 80/20 stratified train-test partition (2,000 holdout records) preserving realistic industrial failure class imbalance (3.39% positive failure rate):

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree (J48/C4.5)** | **98.65%** | **83.61%** | 75.00% | **0.7907** | **0.8722** | Production Line Classification |
| **Logistic Regression (Balanced)** | 84.80% | 16.76% | **83.82%** | 0.2794 | 0.9023 | Maximum Sensitivity Fail-Safe |
| **Gaussian Naive Bayes** | 94.65% | 31.43% | 48.53% | 0.3815 | 0.8872 | Ultra-Low-Power Edge Microcontrollers |
| **K-Nearest Neighbors ($k=5$)** | 97.45% | 76.92% | 29.41% | 0.4255 | 0.8124 | Neighborhood Outlier Detection |

![Model Benchmarks](assets/dash_models.png)

---

### 3. 🔍 Feature Relevance (Shannon Mutual Information)
Quantification of predictive power via Shannon Entropy-based Information Gain identifies **Torque (0.0460 bits)** and **Rotational Speed (0.0364 bits)** as primary leading indicators of failure, followed by **Tool Wear (0.0135 bits)**.

![Feature Importance](assets/feature_importance_info_gain.png)

---

### 4. 🔬 Unsupervised Stress Regime Profiling (K-Means)
Identifies distinct operating regimes by minimizing within-cluster sum of squares ($k=3$):
* **Cluster 0 (Low Stress):** Baseline nominal operations ($1,500\text{--}1,800\text{ rpm}$, torque $< 40\text{ Nm}$).
* **Cluster 1 (Moderate Strain):** Transitional loading states.
* **Cluster 2 (High Mechanical Strain):** Elevated torque ($> 50\text{ Nm}$) and high wear ($> 180\text{ min}$) — the primary vulnerability zone for tool breakage and overstrain.

![Operational Stress Clusters](assets/dash_clusters.png)

---

### 5. ⚡ Association Rule Mining (FP-Growth & Apriori)
Extracts actionable, compound operational preconditions triggering equipment failure:
* `{Heat_Dissipation_Failure} => {Machine_Failure}`: **Confidence: 100.0% | Lift: 29.50**
* `{High_Torque} => {Machine_Failure}`: **Confidence: 14.04% | Lift: 4.14**
* `{High_Wear} => {Machine_Failure}`: **Confidence: 15.49% | Lift: 4.57**

![Association Rules](assets/dash_rules.png)

---

### 6. 📈 Remaining Useful Life (RUL) Forecasting (ARIMA)
An **ARIMA(2, 1, 1)** time-series model fits historical machining degradation sequences to forecast wear progression over future cutting cycles, establishing a predictable schedule before crossing the critical 200-minute wear boundary.

![ARIMA Forecast](assets/dash_arima.png)

---

## 📂 Repository Structure

```
.
├── app.py                      # Interactive Streamlit dashboard application
├── DM_Project.ipynb            # Complete analytical pipeline (Data prep, OLAP, Models, ARIMA)
├── ai4i2020_data.csv           # AI4I 2020 Predictive Maintenance benchmark dataset
├── dashboard.html              # Standalone client-side HTML/Chart.js interactive dashboard
├── run_dashboard.bat           # One-click Windows batch launcher for Streamlit
├── requirements.txt            # Python package dependencies
├── assets/                     # High-resolution architectural and result visualizations
│   ├── pipeline_architecture.png
│   ├── dash_simulator.png
│   ├── dash_models.png
│   ├── dash_clusters.png
│   ├── dash_rules.png
│   ├── dash_arima.png
│   ├── dash_olap.png
│   ├── feature_importance_info_gain.png
│   ├── model_benchmark_comparison.png
│   ├── confusion_matrices.png
│   ├── kmeans_operational_clusters.png
│   └── arima_tool_wear_forecast.png
└── .gitignore                  # Clean gitignore excluding reports and temporary files
```

---

## 🛠️ Quickstart & Installation

### Prerequisites
* Python 3.10 or higher
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/m-eshan/IoT-Predictive-Maintenance.git
cd IoT-Predictive-Maintenance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*Or on Windows, simply double-click `run_dashboard.bat`.*

### 4. Run the Jupyter Notebook
```bash
jupyter notebook DM_Project.ipynb
```

---

## 📊 Dataset Details

* **Name:** AI4I 2020 Predictive Maintenance Dataset
* **Source:** UCI Machine Learning Repository ([Dataset ID 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset))
* **Fleet Records:** 10,000 operational instances
* **Sensor Channels:**
  * Air Temperature $[K]$
  * Process Temperature $[K]$
  * Rotational Speed $[rpm]$
  * Torque $[Nm]$
  * Tool Wear $[min]$
  * Product Variant Tier ($L, M, H$)
* **Failure Modes Monitored:**
  * `TWF` (Tool Wear Failure)
  * `HDF` (Heat Dissipation Failure)
  * `PWF` (Power Failure)
  * `OSF` (Overstrain Failure)
  * `RNF` (Random Failure)

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
