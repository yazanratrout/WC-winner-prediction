"""
Match prediction model training.

Trains Logistic Regression, Random Forest, and XGBoost on historical match
features, calibrates probabilities with isotonic regression, and saves the
best model to models/match_predictor.pkl.

Run:
  python -m src.models.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

FEATURES_PATH  = Path("data/processed/features.parquet")
MODEL_OUT      = Path("models/match_predictor.pkl")
METADATA_OUT   = Path("models/model_metadata.json")

# Feature columns used for training (all diff_* + meta)
_EXCLUDE = {"date", "home_team", "away_team", "outcome", "home_score", "away_score"}

# Train / val / test split boundaries
_VAL_START  = "2022-01-01"
_TEST_START = "2024-01-01"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    # Fill nulls with 0 — means "no differential info" for teams without player data
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    df[feature_cols] = df[feature_cols].fillna(0)

    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] <  _VAL_START]
    val   = df[(df["date"] >= _VAL_START) & (df["date"] < _TEST_START)]
    test  = df[df["date"] >= _TEST_START]

    def split(subset):
        X = subset[feature_cols].values.astype(np.float32)
        y = subset["outcome"]
        return X, y

    X_train, y_train = split(train)
    X_val,   y_val   = split(val)
    X_test,  y_test  = split(test)

    print(f"  Train: {len(train):,} matches ({train['date'].min().date()} – {train['date'].max().date()})")
    print(f"  Val:   {len(val):,}  matches ({val['date'].min().date()} – {val['date'].max().date()})")
    print(f"  Test:  {len(test):,} matches ({test['date'].min().date()} – {test['date'].max().date()})")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train outcome dist: {dict(y_train.value_counts())}")

    # Scale features (helps LR; tree models are unaffected)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

_DRAW_BOOST = 1.8  # extra multiplier on the draw class weight

def _class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = dict(zip(classes, weights))
    if "D" in cw:
        cw["D"] *= _DRAW_BOOST
    return cw


def train_logistic(X_train, y_train):
    cw = _class_weights(y_train)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=cw,
        C=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    cw = _class_weights(y_train)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    cw = _class_weights(y_train)
    sample_w = np.array([cw[label] for label in y_train])

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_enc := X_train, y_enc, sample_weight=sample_w)
    model._le = le
    return model


def _xgb_predict(model, X):
    """Return string labels and probabilities, in original class order."""
    y_enc = model.predict(X)
    proba = model.predict_proba(X)  # columns ordered by encoded integer
    labels = model._le.inverse_transform(y_enc)
    return labels, proba, model._le.classes_


# ---------------------------------------------------------------------------
# Calibration + evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_val, y_val, X_test, y_test):
    le = getattr(model, "_le", None)
    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    p_val   = model.predict_proba(X_val)
    p_test  = model.predict_proba(X_test)

    if le is not None:
        # XGB returns integer predictions — decode to strings
        y_pred_val  = le.inverse_transform(y_pred_val.astype(int))
        y_pred_test = le.inverse_transform(y_pred_test.astype(int))
        classes = le.classes_
        # y_val / y_test are string — encode for log_loss
        ll_val  = log_loss(le.transform(y_val),  p_val)
        ll_test = log_loss(le.transform(y_test), p_test)
    else:
        classes = model.classes_
        le_eval = LabelEncoder().fit(classes)
        ll_val  = log_loss(le_eval.transform(y_val),  p_val)
        ll_test = log_loss(le_eval.transform(y_test), p_test)
    acc_val  = accuracy_score(y_val,  y_pred_val)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"\n  [{name}]")
    print(f"    Val  — log_loss: {ll_val:.4f}  accuracy: {acc_val:.4f}")
    print(f"    Test — log_loss: {ll_test:.4f}  accuracy: {acc_test:.4f}")
    print(f"    Test classification report:")
    report = classification_report(y_test, y_pred_test, target_names=classes)
    for line in report.splitlines():
        print("      " + line)

    return {"val_logloss": ll_val, "val_acc": acc_val,
            "test_logloss": ll_test, "test_acc": acc_test}


def calibrate(model, X_val, y_val, method="isotonic"):
    """Wrap model in isotonic calibration using the validation set."""
    y_fit = model._le.transform(y_val) if hasattr(model, "_le") else y_val
    cal = CalibratedClassifierCV(model, method=method, cv=None)
    cal.fit(X_val, y_fit)
    if hasattr(model, "_le"):
        cal._le = model._le
    return cal


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

def log_feature_importances(model, feature_cols: list[str], name: str) -> None:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0)
    else:
        return

    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
    print(f"\n  Top-15 feature importances [{name}]:")
    for feat, score in pairs:
        print(f"    {feat:<40s} {score:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("MATCH PREDICTION MODEL TRAINING")
    print("=" * 60)

    print("\n--- Build Training Dataset ---")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler = load_dataset()

    print("\n--- Train Models ---")
    print("  Training Logistic Regression...")
    lr   = train_logistic(X_train, y_train)

    print("  Training Random Forest...")
    rf   = train_random_forest(X_train, y_train)

    print("  Training XGBoost...")
    xgb  = train_xgboost(X_train, y_train)

    print("\n--- Evaluate (uncalibrated) ---")
    lr_metrics  = evaluate("Logistic Regression", lr,  X_val, y_val, X_test, y_test)
    rf_metrics  = evaluate("Random Forest",       rf,  X_val, y_val, X_test, y_test)
    xgb_metrics = evaluate("XGBoost",             xgb, X_val, y_val, X_test, y_test)

    # Calibrate all three on val set
    print("\n--- Calibrating probabilities (isotonic) ---")
    lr_cal  = calibrate(lr,  X_val, y_val)
    rf_cal  = calibrate(rf,  X_val, y_val)
    xgb_cal = calibrate(xgb, X_val, y_val)

    print("\n--- Evaluate (calibrated) ---")
    lr_cal_m  = evaluate("LR  (calibrated)", lr_cal,  X_val, y_val, X_test, y_test)
    rf_cal_m  = evaluate("RF  (calibrated)", rf_cal,  X_val, y_val, X_test, y_test)
    xgb_cal_m = evaluate("XGB (calibrated)", xgb_cal, X_val, y_val, X_test, y_test)

    # Feature importances from best tree model
    log_feature_importances(rf,  feature_cols, "Random Forest")
    log_feature_importances(xgb, feature_cols, "XGBoost")

    # Select best by val log-loss (calibrated)
    candidates = [
        ("LR",  lr_cal,  lr_cal_m),
        ("RF",  rf_cal,  rf_cal_m),
        ("XGB", xgb_cal, xgb_cal_m),
    ]
    best_name, best_model, best_metrics = min(candidates, key=lambda x: x[2]["val_logloss"])
    print(f"\n  Best model: {best_name}  (val log_loss = {best_metrics['val_logloss']:.4f})")

    # Save
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    le = getattr(best_model, "_le", None)
    classes_out = list(le.classes_) if le is not None else list(best_model.classes_)
    payload = {
        "model":        best_model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "classes":      classes_out,
    }
    joblib.dump(payload, MODEL_OUT)
    print(f"  Saved → {MODEL_OUT}")

    metadata = {
        "best_model":    best_name,
        "feature_cols":  feature_cols,
        "classes":       classes_out,
        "val_logloss":   best_metrics["val_logloss"],
        "val_acc":       best_metrics["val_acc"],
        "test_logloss":  best_metrics["test_logloss"],
        "test_acc":      best_metrics["test_acc"],
        "train_cutoff":  _VAL_START,
        "val_cutoff":    _TEST_START,
        "all_results": {
            "LR_cal":  lr_cal_m,
            "RF_cal":  rf_cal_m,
            "XGB_cal": xgb_cal_m,
        },
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2))
    print(f"  Metadata → {METADATA_OUT}")

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)

    return best_model, feature_cols


if __name__ == "__main__":
    run()
