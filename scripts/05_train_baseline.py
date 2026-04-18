#!/usr/bin/env python3
"""
Script 05: Baseline model — logistic regression on Open Targets features.

Simpler comparator to XGBoost. Uses the same feature matrix but with
L2-regularised logistic regression. This establishes a lower bound
and demonstrates that the conformal prediction framework is model-agnostic.

Optional: node2vec embeddings can be added as additional features if the
interaction graph is built (requires networkx + node2vec library).
"""

import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42


def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")
    with open(PROCESSED_DIR / "datasource_list.json") as f:
        datasources = json.load(f)
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
    return df, feature_cols


def evaluate(model, X, y, split_name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auroc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": average_precision_score(y, y_prob) if y.sum() > 0 else float("nan"),
        "brier": brier_score_loss(y, y_prob),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
    }

    print(f"\n── {split_name} ──")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(classification_report(y, y_pred, digits=3, zero_division=0))

    return metrics, y_prob


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main():
    t_total = time.time()
    print("Script 05: Logistic regression baseline")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    t = time.time()
    df, feature_cols = load_data()
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    splits = {}
    for split_name in ["train", "calibration", "test"]:
        mask = df["split"] == split_name
        X = df.loc[mask, feature_cols].values.astype(np.float32)
        y = df.loc[mask, "label"].values.astype(int)
        splits[split_name] = (X, y)
        print(f"  {split_name}: {X.shape[0]:,} samples, {y.sum():,} pos")
    print(f"  Done ({fmt_elapsed(time.time() - t)})")

    X_train, y_train = splits["train"]

    # Handle class imbalance
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])

    print("\n[2/4] Training logistic regression...")
    t = time.time()
    model.fit(X_train, y_train)
    print(f"  Done ({fmt_elapsed(time.time() - t)})")

    print("\n[3/4] Evaluating model...")
    results = {}
    probs = {}
    for split_name, (X, y) in splits.items():
        metrics, y_prob = evaluate(model, X, y, split_name)
        results[split_name] = metrics
        probs[split_name] = y_prob

    # Save
    print("\n[4/4] Saving outputs...")
    model_path = OUTPUT_DIR / "baseline_lr_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save predictions for conformal prediction comparison
    prob_df = df[["drug_id", "disease_id", "label", "split"]].copy()
    prob_df["y_prob"] = np.nan
    for split_name, y_prob in probs.items():
        mask = df["split"] == split_name
        prob_df.loc[mask, "y_prob"] = y_prob
    prob_df.to_parquet(PROCESSED_DIR / "baseline_predictions.parquet", index=False)

    results["feature_cols"] = feature_cols
    with open(OUTPUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Coefficient analysis
    lr = model.named_steps["lr"]
    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr.coef_[0],
    }).sort_values("coefficient", ascending=False)
    coefs.to_csv(OUTPUT_DIR / "baseline_coefficients.csv", index=False)
    print(f"\nTop 10 positive coefficients:")
    print(coefs.head(10).to_string(index=False))

    print(f"\nModel saved: {model_path}")
    print(f"\n{'=' * 70}")
    print(f"Script 05 complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
