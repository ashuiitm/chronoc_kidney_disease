"""
02_train.py
Feature engineering, model training, evaluation, and explainability
for the Renalyx-style "Point-of-Care CKD Risk Screening" model.

Framing: Renalyx's Point-of-Care Solution screens diabetic/hypertensive
patients for early-stage CKD. This model plays that role -- given
routine vitals + basic labs (the kind collected at a screening visit,
not a full nephrology workup), flag patients who need referral/dialysis
pathway review.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, RocCurveDisplay, classification_report
)
from xgboost import XGBClassifier
import shap

DATA_PATH = "data/kidney_disease_clean.csv"
FIG_DIR = "figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

def engineer_features(df):
    df = df.copy()
    # BUN:Creatinine ratio -- classic nephrology signal for prerenal vs intrinsic renal issues
    df["bun_creatinine_ratio"] = df["bu"] / df["sc"].replace(0, np.nan)
    # Comorbidity burden -- diabetes + hypertension + coronary artery disease are the exact
    # population Renalyx's point-of-care screening targets
    df["comorbidity_count"] = df[["htn", "dm", "cad"]].sum(axis=1, skipna=True)
    # Anemia flag -- CKD-associated anemia, hemoglobin < 11 g/dL threshold
    df["anemia_flag"] = (df["hemo"] < 11).astype(float)
    # Significant albuminuria -- albumin grade >= 2 is a stronger CKD signal than trace
    df["significant_albuminuria"] = (df["al"] >= 2).astype(float)
    # Low specific gravity -- impaired concentrating ability
    df["low_specific_gravity"] = (df["sg"] <= 1.010).astype(float)
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df["classification"]
    X = df.drop(columns=["classification"])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Baseline: Logistic Regression (interpretable, matches "explainable" requirement) ---
    baseline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]

    # --- Main model: XGBoost (handles missingness + nonlinearity natively) ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    )
    # XGBoost handles NaN natively -- no imputation needed for this one
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    # --- 5-fold CV on training set (robustness given small n=400) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv_auc = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="roc_auc")

    results = {
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_logreg": {
            "roc_auc": round(roc_auc_score(y_test, baseline_proba), 4),
            "f1": round(f1_score(y_test, baseline_pred), 4),
            "precision": round(precision_score(y_test, baseline_pred), 4),
            "recall": round(recall_score(y_test, baseline_pred), 4),
        },
        "xgboost": {
            "roc_auc": round(roc_auc_score(y_test, xgb_proba), 4),
            "f1": round(f1_score(y_test, xgb_pred), 4),
            "precision": round(precision_score(y_test, xgb_pred), 4),
            "recall": round(recall_score(y_test, xgb_pred), 4),
            "cv_auc_mean": round(xgb_cv_auc.mean(), 4),
            "cv_auc_std": round(xgb_cv_auc.std(), 4),
        },
    }
    with open("results_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, xgb_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CKD", "CKD"], yticklabels=["No CKD", "CKD"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("XGBoost — Confusion Matrix (held-out test set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC curve, both models ---
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, baseline_proba, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, xgb_proba, name="XGBoost", ax=ax)
    plt.title("ROC Curve — CKD Screening Models")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # --- SHAP explainability (this is what you talk through in the interview) ---
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save feature importance table too (simpler artifact for the resume/README)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv("feature_importance.csv", index=False)
    print("\nTop features by SHAP importance:")
    print(importance.head(10).to_string(index=False))

    # Save classification report text
    with open("classification_report.txt", "w") as f:
        f.write("=== Logistic Regression ===\n")
        f.write(classification_report(y_test, baseline_pred, target_names=["No CKD", "CKD"]))
        f.write("\n=== XGBoost ===\n")
        f.write(classification_report(y_test, xgb_pred, target_names=["No CKD", "CKD"]))

if __name__ == "__main__":
    main()
