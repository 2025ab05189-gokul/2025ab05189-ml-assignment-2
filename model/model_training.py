"""
model_training.py
------------------
Train all 6 classification models on the Heart Disease dataset,
evaluate them, and print results.

Dataset: Heart Statlog + Cleveland + Hungary combined
Target : 'target' — 1 = heart disease, 0 = no heart disease
"""

import os
import ssl
import pickle
import warnings

# Fix macOS SSL certificate issue BEFORE any network calls
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/erdenahmet11/Heart-Disease-Prediction/main/heart_statlog_cleveland_hungary_final.csv"
LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "heart.csv")

COLUMN_MAP = {
    "age": "age", "sex": "sex",
    "chest pain type": "chest_pain_type", "chest_pain_type": "chest_pain_type",
    "resting bp s": "resting_bp", "resting_bp": "resting_bp",
    "cholesterol": "cholesterol",
    "fasting blood sugar": "fasting_bs", "fasting_bs": "fasting_bs",
    "resting ecg": "resting_ecg", "resting_ecg": "resting_ecg",
    "max heart rate": "max_hr", "max_hr": "max_hr",
    "exercise angina": "exercise_angina", "exercise_angina": "exercise_angina",
    "oldpeak": "oldpeak",
    "ST slope": "st_slope", "st_slope": "st_slope",
    "target": "target", "HeartDisease": "target",
}

print("Loading dataset...")
df = None

# Try URL first (full 1190-row dataset)
try:
    df = pd.read_csv(DATA_URL)
    print(f"  Loaded from URL: {DATA_URL}")
except Exception as e:
    print(f"  URL failed ({e}), trying local...")

# Fallback: local CSV
if df is None:
    if os.path.exists(LOCAL_CSV):
        try:
            df = pd.read_csv(LOCAL_CSV)
            print(f"  Loaded from local: {LOCAL_CSV}")
        except Exception as e2:
            print(f"  ERROR: Could not load dataset: {e2}")
            exit(1)
    else:
        print("  ERROR: No data source available.")
        exit(1)

# Normalise columns and dedup
df.columns = [COLUMN_MAP.get(c.strip(), c.strip()) for c in df.columns]
df = df.drop_duplicates().reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# ──────────────────────────────────────────────
# 2. Preprocessing (all features are already numeric)
# ──────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTraining set: {X_train_sc.shape}")
print(f"Test set:     {X_test_sc.shape}")

# ──────────────────────────────────────────────
# 3. Define models
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# 4. Train & evaluate
# ──────────────────────────────────────────────
results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

# ──────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
print(results_df.to_string(index=False))
print("\nDone!")
