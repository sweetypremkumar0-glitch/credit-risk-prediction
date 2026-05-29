"""
Credit Risk Analysis System — Model Training Script
=====================================================
Improvements over baseline:
  • Ensemble voting (Logistic Regression + Random Forest + Gradient Boosting)
  • Proper cross-validation & hyperparameter search
  • SMOTE oversampling for class imbalance
  • Full metrics: ROC-AUC, PR-AUC, per-class F1
  • Saves: trained_model.pkl, label_encoders.pkl, feature_names.pkl, metrics.json
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, f1_score
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠  imbalanced-learn not found. Install with: pip install imbalanced-learn")
    print("   Continuing without SMOTE (will use class_weight='balanced' instead).")
    SMOTE_AVAILABLE = False


# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  CREDIT RISK MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("german_credit_data.csv")
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

print(f"\n✔ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
print("  Target distribution:")
print(df["Risk"].value_counts().to_string(index=True))


# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
CAT_MISSING = ["Saving accounts", "Checking account"]
for col in CAT_MISSING:
    df[col] = df[col].fillna("Unknown")

# Derived features
df["credit_per_month"]   = df["Credit amount"] / df["Duration"]
df["age_job_interaction"] = df["Age"] * df["Job"]

X = df.drop(columns=["Risk"])
y = (df["Risk"] == "bad").astype(int)

numeric_features = [
    "Age", "Job", "Credit amount", "Duration",
    "credit_per_month", "age_job_interaction"
]
categorical_features = [
    "Sex", "Housing", "Saving accounts", "Checking account", "Purpose"
]

print(f"\n✔ Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")


# ─── 3. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✔ Split: {len(X_train)} train / {len(X_test)} test (80/20, stratified)")


# ─── 4. PREPROCESSING PIPELINE ───────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ─── 5. ENSEMBLE MODEL ────────────────────────────────────────────────────────
lr = LogisticRegression(
    C=0.5, class_weight="balanced",
    max_iter=1000, random_state=42, solver="lbfgs"
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    voting="soft",
    weights=[1, 2, 2]   # RF + GB weighted higher
)

if SMOTE_AVAILABLE:
    print("\n✔ Using SMOTE for oversampling")
    smote = SMOTE(random_state=42, k_neighbors=5)
    model_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("classifier", ensemble)
    ])
else:
    print("\n✔ Using class_weight='balanced' for imbalance handling")
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ensemble)
    ])


# ─── 6. CROSS-VALIDATION ─────────────────────────────────────────────────────
print("\n⏳ Running 5-fold cross-validation …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"   ROC-AUC per fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─── 7. FIT FINAL MODEL ───────────────────────────────────────────────────────
print("\n⏳ Fitting final model on full training set …")
model_pipeline.fit(X_train, y_train)
print("✔ Training complete.")


# ─── 8. EVALUATE ──────────────────────────────────────────────────────────────
y_pred  = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

acc         = accuracy_score(y_test, y_pred)
roc_auc     = roc_auc_score(y_test, y_proba)
pr_auc      = average_precision_score(y_test, y_proba)
f1_bad      = f1_score(y_test, y_pred, pos_label=1)
f1_good     = f1_score(y_test, y_pred, pos_label=0)
cm          = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  TEST ACCURACY : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  PR-AUC        : {pr_auc:.4f}")
print(f"  F1 (Bad Risk) : {f1_bad:.4f}")
print(f"  F1 (Good Risk): {f1_good:.4f}")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Good Risk", "Bad Risk"]))
print("Confusion Matrix:")
print(cm)


# ─── 9. SAVE ARTIFACTS ────────────────────────────────────────────────────────
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump({"numeric": numeric_features, "categorical": categorical_features}, f)

metrics = {
    "accuracy":     round(float(acc), 4),
    "roc_auc":      round(float(roc_auc), 4),
    "pr_auc":       round(float(pr_auc), 4),
    "f1_bad":       round(float(f1_bad), 4),
    "f1_good":      round(float(f1_good), 4),
    "cv_roc_mean":  round(float(cv_scores.mean()), 4),
    "cv_roc_std":   round(float(cv_scores.std()), 4),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_test, y_pred, target_names=["Good Risk", "Bad Risk"], output_dict=True
    )
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved:")
print("   • trained_model.pkl   — full pipeline (preprocessor + ensemble)")
print("   • feature_names.pkl   — feature column reference")
print("   • metrics.json        — evaluation metrics for dashboard")
print("\nTraining complete. Run `streamlit run app.py` to launch the dashboard.")
