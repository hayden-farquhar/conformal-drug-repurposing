#!/usr/bin/env python3
"""
Script 04: Train XGBoost classifier for drug-disease indication prediction.

- Stratified 5-fold CV hyperparameter tuning on train split
- Final model trained on full train split
- Evaluation on calibration and test splits: AUROC, AUPRC, Brier score
- Saves model and predicted probabilities for conformal prediction (Script 06)
"""

import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    """Load labelled dataset and prepare feature matrix."""
    df = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")

    # Load datasource list to identify feature columns
    with open(PROCESSED_DIR / "datasource_list.json") as f:
        datasources = json.load(f)

    # Feature columns: datasource scores + n_targets + enrichment features
    # EXCLUDE clinical_precedence (encodes clinical trial/approval history — circular)
    # EXCLUDE overall_score (aggregates clinical_precedence — contaminated)
    EXCLUDE = {"clinical_precedence", "overall_score"}
    ENRICHMENT_COLS = [
        "disease_ontology_max_sim", "disease_ontology_mean_sim", "disease_ontology_n_related",
        "target_degree_max", "target_degree_mean", "target_mean_interaction_score",
        "target_disease_network_overlap",
        "drug_fp_sim_max", "drug_fp_sim_mean", "drug_has_smiles",
    ]
    feature_cols = [c for c in datasources if c not in EXCLUDE] + ["n_targets"]
    feature_cols += [c for c in ENRICHMENT_COLS if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Excluded (clinical leakage): {sorted(EXCLUDE)}")

    return df, feature_cols


def split_data(df, feature_cols):
    """Split into train/calibration/test sets."""
    splits = {}
    for split_name in ["train", "calibration", "test"]:
        mask = df["split"] == split_name
        X = df.loc[mask, feature_cols].values.astype(np.float32)
        y = df.loc[mask, "label"].values.astype(int)
        splits[split_name] = (X, y)
        print(f"{split_name}: {X.shape[0]:,} samples, "
              f"{y.sum():,} positive ({y.mean():.4f} prevalence)")
    return splits


def tune_xgboost(X_train, y_train):
    """
    Hyperparameter tuning via manual random search with per-iteration progress.
    Tunes on a stratified 20% subsample, then retrains best params on full data.
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.utils import check_random_state
    import os

    param_options = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.3, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    # ── Step 1: Subsample for tuning ─────────────────────────────────────
    TUNE_FRAC = 0.2
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=TUNE_FRAC,
        stratify=y_train, random_state=RANDOM_STATE,
    )
    tune_pos = y_tune.sum()
    tune_neg = len(y_tune) - tune_pos
    tune_scale_pos = tune_neg / tune_pos if tune_pos > 0 else 1.0

    print(f"\n  Tuning on {TUNE_FRAC:.0%} stratified subsample: "
          f"{len(y_tune):,} samples ({tune_pos:,} pos)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_iter = 30
    rng = check_random_state(RANDOM_STATE)

    # Sample parameter combinations upfront
    param_combos = []
    for _ in range(n_iter):
        combo = {k: rng.choice(v) for k, v in param_options.items()}
        # numpy types → Python native for XGBoost
        combo = {k: int(v) if isinstance(v, (np.integer,)) else float(v) for k, v in combo.items()}
        param_combos.append(combo)

    print(f"  Search: {n_iter} iterations × 5-fold CV")
    t_start = time.time()

    best_score = -1
    best_params = None
    best_idx = -1

    for i, params in enumerate(param_combos):
        t_iter = time.time()
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=tune_scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        scores = cross_val_score(
            model, X_tune, y_tune, cv=cv,
            scoring="average_precision", n_jobs=-1,
        )
        mean_score = scores.mean()
        iter_sec = time.time() - t_iter
        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (n_iter - i - 1)

        marker = " ★" if mean_score > best_score else ""
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_idx = i

        print(f"    [{i+1:2d}/{n_iter}] AUPRC={mean_score:.4f} "
              f"(best={best_score:.4f} @{best_idx+1}) "
              f"depth={params['max_depth']} n_est={params['n_estimators']} "
              f"lr={params['learning_rate']} — "
              f"{iter_sec:.1f}s (ETA {fmt_elapsed(eta)}){marker}",
              flush=True)

    tune_elapsed = time.time() - t_start
    print(f"\n  Tuning complete in {fmt_elapsed(tune_elapsed)}")
    print(f"  Best CV AUPRC: {best_score:.4f}")
    print(f"  Best params: {best_params}")

    # ── Step 2: Retrain best params on full training set ─────────────────
    print(f"\n  Retraining best params on full training set ({len(y_train):,} samples)...")
    t_retrain = time.time()
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    final_model.fit(X_train, y_train)
    print(f"  Full retrain complete in {fmt_elapsed(time.time() - t_retrain)}")

    return final_model, best_params, best_score


def evaluate(model, X, y, split_name):
    """Compute classification metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auroc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": average_precision_score(y, y_prob) if y.sum() > 0 else float("nan"),
        "brier": brier_score_loss(y, y_prob),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
    }

    print(f"\n── {split_name} ──")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  N={metrics['n_samples']:,} (pos={metrics['n_positive']:,})")
    print(classification_report(y, y_pred, digits=3, zero_division=0))

    return metrics, y_prob


def fmt_elapsed(seconds):
    """Format elapsed time as Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main():
    t_total = time.time()
    print("Script 04: XGBoost training for drug-disease indication prediction")
    print("=" * 70)

    # Phase 1: Load data
    print("\n[1/4] Loading data...")
    t = time.time()
    df, feature_cols = load_data()
    splits = split_data(df, feature_cols)
    print(f"  Done ({fmt_elapsed(time.time() - t)})")

    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    # Phase 2: Tune and train
    print("\n[2/4] Hyperparameter tuning...")
    model, best_params, cv_score = tune_xgboost(X_train, y_train)

    # Phase 3: Evaluate on all splits
    print("\n[3/4] Evaluating model...")
    t = time.time()
    results = {}
    probs = {}
    for split_name, (X, y) in splits.items():
        metrics, y_prob = evaluate(model, X, y, split_name)
        results[split_name] = metrics
        probs[split_name] = y_prob
    print(f"  Evaluation done ({fmt_elapsed(time.time() - t)})")

    # ── Save outputs ──────────────────────────────────────────────────────
    print("\n[4/4] Saving outputs...")
    t = time.time()
    # Model
    model_path = OUTPUT_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    # Predicted probabilities (needed for conformal prediction)
    prob_df = df[["drug_id", "disease_id", "label", "split"]].copy()
    prob_df["y_prob"] = np.nan
    for split_name, y_prob in probs.items():
        mask = df["split"] == split_name
        prob_df.loc[mask, "y_prob"] = y_prob
    prob_path = PROCESSED_DIR / "xgboost_predictions.parquet"
    prob_df.to_parquet(prob_path, index=False)
    print(f"Predictions saved: {prob_path}")

    # Metrics
    results["best_params"] = best_params
    results["cv_auprc"] = cv_score
    results["feature_cols"] = feature_cols
    metrics_path = OUTPUT_DIR / "xgboost_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Metrics saved: {metrics_path}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)
    print(f"\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    print(f"  Outputs saved ({fmt_elapsed(time.time() - t)})")
    print(f"\n{'=' * 70}")
    print(f"Script 04 complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
