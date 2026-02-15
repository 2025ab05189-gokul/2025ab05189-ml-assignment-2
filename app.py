"""
app.py — Streamlit Web Application
Heart Disease Classification — ML Assignment 2

Uses the UCI Heart Statlog + Cleveland + Hungary combined dataset.
Primary: loads from bundled local CSV. Fallback: loads from URL.
"""

import os
import ssl

# Fix macOS SSL certificate issue BEFORE any network calls
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classification — ML Assignment 2",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Prediction — ML Classification")
st.markdown(
    "**M.Tech (AIML/DSE) — Machine Learning Assignment 2**  \n"
    "Compare 6 classification models on the Heart Failure Prediction dataset."
)
st.divider()

# ──────────────────────────────────────────────
# Helper: load default dataset
# ──────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/erdenahmet11/Heart-Disease-Prediction/main/heart_statlog_cleveland_hungary_final.csv"
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "data", "heart.csv")

COLUMN_MAP = {
    "age": "age",
    "sex": "sex",
    "chest pain type": "chest_pain_type",
    "chest_pain_type": "chest_pain_type",
    "resting bp s": "resting_bp",
    "resting_bp": "resting_bp",
    "cholesterol": "cholesterol",
    "fasting blood sugar": "fasting_bs",
    "fasting_bs": "fasting_bs",
    "resting ecg": "resting_ecg",
    "resting_ecg": "resting_ecg",
    "max heart rate": "max_hr",
    "max_hr": "max_hr",
    "exercise angina": "exercise_angina",
    "exercise_angina": "exercise_angina",
    "oldpeak": "oldpeak",
    "ST slope": "st_slope",
    "st_slope": "st_slope",
    "target": "target",
    "HeartDisease": "target",
}


@st.cache_data
def load_default_dataset():
    """Try URL first (full dataset), then local CSV as fallback."""
    df = None

    # Try URL first (full 1190-row dataset)
    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        pass

    # Fallback: local file
    if df is None and os.path.exists(LOCAL_CSV):
        try:
            df = pd.read_csv(LOCAL_CSV)
        except Exception:
            pass

    if df is None:
        st.error("Could not load dataset. Please upload a CSV manually.")
        st.stop()

    # Normalise column names
    df.columns = [COLUMN_MAP.get(c.strip(), c.strip()) for c in df.columns]

    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


@st.cache_data
def preprocess(df):
    """Return X, y (all numeric)."""
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=10
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=200, random_state=42, max_depth=5,
            learning_rate=0.1, use_label_encoder=False, eval_metric="logloss"
        ),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        results[name] = {
            "model": clf,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        }
    return results, scaler


# ──────────────────────────────────────────────
# Sidebar — Dataset upload
# ──────────────────────────────────────────────
st.sidebar.header("📁 Dataset")
upload_option = st.sidebar.radio(
    "Choose data source:",
    ["Use default Heart Disease dataset", "Upload your own CSV"],
)

if upload_option == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (must contain a 'target' column)", type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [COLUMN_MAP.get(c.strip(), c.strip()) for c in df.columns]
        if "target" not in df.columns:
            st.error(
                "Uploaded CSV must contain a 'target' column "
                "(or 'HeartDisease'). Please check your file."
            )
            st.stop()
    else:
        st.info("⬆️ Please upload a CSV file or switch to the default dataset.")
        st.stop()
else:
    df = load_default_dataset()

st.sidebar.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "🤖 Model Comparison", "🔍 Individual Model Analysis", "ℹ️ About"]
)

# ── Tab 1 ──
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Statistics**")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.markdown("**Target Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        df["target"].value_counts().plot.bar(
            ax=ax, color=["#4CAF50", "#F44336"], edgecolor="black"
        )
        ax.set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown("**Column Info**")
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Unique": df.nunique().values,
    })
    st.dataframe(info, use_container_width=True)

# ── Preprocess & train ──
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with st.spinner("Training all 6 models..."):
    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test)

# ── Tab 2 ──
with tab2:
    st.subheader("📋 Evaluation Metrics — All Models")

    comp = []
    for name, r in results.items():
        comp.append({
            "Model": name,
            "Accuracy": r["Accuracy"],
            "AUC": r["AUC"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"],
            "MCC": r["MCC"],
        })
    comp_df = pd.DataFrame(comp)
    st.dataframe(
        comp_df.style.highlight_max(
            axis=0, subset=comp_df.columns[1:], color="#c8e6c9"
        ),
        use_container_width=True,
    )

    # Bar charts
    st.subheader("📊 Visual Comparison")
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = sns.color_palette("Set2", n_colors=6)
    for idx, m in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        bars = ax.barh(comp_df["Model"], comp_df[m], color=colors, edgecolor="black")
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, comp_df[m]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # ROC
    st.subheader("📈 ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={r['AUC']:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All Models")
    ax_roc.legend(loc="lower right", fontsize=8)
    st.pyplot(fig_roc)

    # Observations
    st.subheader("🔎 Model Observations")
    observations = {
        "Logistic Regression": (
            "Solid baseline with good interpretability. Performs well on linearly separable aspects "
            "of the data. Competitive AUC indicates reasonable generalization. "
            "May underperform with complex non-linear feature interactions."
        ),
        "Decision Tree": (
            "Captures non-linear decision boundaries and is highly interpretable. "
            "Prone to overfitting even with max_depth control. "
            "Generally lower AUC compared to ensemble methods due to high variance."
        ),
        "kNN": (
            "Instance-based lazy learner relying on local similarity. Sensitive to k and distance metric. "
            "Performance degrades with irrelevant features. Scaling is critical — "
            "after standard scaling, results improve notably."
        ),
        "Naive Bayes": (
            "Very fast to train. Works well when feature independence assumption roughly holds. "
            "Often produces well-calibrated probabilities. The strong independence assumption "
            "can limit performance on datasets with correlated features like medical data."
        ),
        "Random Forest (Ensemble)": (
            "Significantly reduces variance via bagging many decorrelated trees. "
            "Typically achieves high accuracy and AUC. Robust to outliers and handles "
            "feature interactions well. Less interpretable than single models."
        ),
        "XGBoost (Ensemble)": (
            "Sequential gradient boosting with regularization. Often yields top performance. "
            "Built-in L1/L2 regularization prevents overfitting. Generally achieves "
            "the best or near-best metrics with excellent generalization."
        ),
    }
    obs_df = pd.DataFrame([{"Model": k, "Observation": v} for k, v in observations.items()])
    st.table(obs_df)

# ── Tab 3 ──
with tab3:
    st.subheader("🔍 Detailed Analysis — Select a Model")

    model_name = st.selectbox("Select Model", list(results.keys()))
    r = results[model_name]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{r['Accuracy']:.4f}")
    c2.metric("AUC Score", f"{r['AUC']:.4f}")
    c3.metric("F1 Score", f"{r['F1']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{r['Precision']:.4f}")
    c5.metric("Recall", f"{r['Recall']:.4f}")
    c6.metric("MCC", f"{r['MCC']:.4f}")

    # Confusion matrix
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, r["y_pred"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig_cm)

    # Classification report
    st.markdown("**Classification Report**")
    report = classification_report(
        y_test, r["y_pred"],
        target_names=["No Disease", "Disease"],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # Individual ROC
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    fig_r, ax_r = plt.subplots(figsize=(6, 4))
    ax_r.plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC = {r['AUC']:.4f}")
    ax_r.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_r.fill_between(fpr, tpr, alpha=0.1, color="#1976D2")
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title(f"ROC Curve — {model_name}")
    ax_r.legend()
    st.pyplot(fig_r)

# ── Tab 4 ──
with tab4:
    st.subheader("About This Project")
    st.markdown("""
    **Assignment:** Machine Learning — Assignment 2  
    **Programme:** M.Tech (AIML / DSE), BITS Pilani — WILP  

    **Dataset:** Heart Statlog + Cleveland + Hungary Combined  
    - **Source:** [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
    - **Origin:** 5 combined heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog)
    - **Features:** 11 clinical features + 1 binary target  

    **Models Implemented:**
    1. Logistic Regression  
    2. Decision Tree Classifier  
    3. K-Nearest Neighbors (kNN)  
    4. Gaussian Naive Bayes  
    5. Random Forest (Ensemble)  
    6. XGBoost (Ensemble)  

    **Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1 Score, MCC
    """)
