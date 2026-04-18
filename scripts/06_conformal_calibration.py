#!/usr/bin/env python3
"""
Script 06: Conformal prediction calibration — marginal and Mondrian.

Core methodological contribution of the paper.

Methods:
  1. Marginal split CP: single threshold across all drug-disease pairs
  2. Mondrian CP: group-conditional thresholds by therapeutic area
  3. Evaluation: empirical coverage, set size distributions, efficiency

Non-conformity score: s(x, y) = 1 - f(x)_y
  - For y=1 (positive): s = 1 - P(indication)
  - For y=0 (negative): s = P(indication)  [i.e., 1 - P(not indication)]

At coverage level α (e.g., 0.90), the prediction set for a new x includes
class y if s(x, y) ≤ q̂, where q̂ is the ⌈(n+1)α⌉/n quantile of
calibration non-conformity scores.

Inputs:
  - data/processed/xgboost_predictions.parquet
  - data/processed/drug_disease_labelled.parquet (for therapeutic area groups)

Outputs:
  - outputs/conformal_results.json (coverage, set sizes, per-group metrics)
  - data/processed/conformal_prediction_sets.parquet
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_LEVELS = [0.80, 0.85, 0.90, 0.95]  # nominal coverage levels
PRIMARY_ALPHA = 0.90
RANDOM_STATE = 42

# ── Therapeutic area grouping for Mondrian CP ─────────────────────────────
# Map detailed OT therapeutic areas to 7 broad groups
TA_GROUP_MAP = {
    "neoplasm": "oncology",
    "cancer": "oncology",
    "tumor": "oncology",
    "carcinoma": "oncology",
    "nervous system disease": "neurological",
    "neurodegenerat": "neurological",
    "psychiatric": "neurological",
    "mental": "neurological",
    "cardiovascular": "cardiovascular",
    "heart": "cardiovascular",
    "infectious disease": "infectious",
    "viral": "infectious",
    "bacterial": "infectious",
    "metabolic disease": "metabolic",
    "diabetes": "metabolic",
    "obesity": "metabolic",
    "rare disease": "rare_disease",
    "orphan": "rare_disease",
    "genetic disorder": "rare_disease",
    "genetic, familial or congenital": "rare_disease",
    "congenital": "rare_disease",
}


def map_therapeutic_area(ta_name: str) -> str:
    """Map a therapeutic area name to a broad Mondrian group."""
    if pd.isna(ta_name):
        return "other"
    ta_lower = ta_name.lower()
    for pattern, group in TA_GROUP_MAP.items():
        if pattern in ta_lower:
            return group
    return "other"


# ── Core conformal prediction functions ───────────────────────────────────

def nonconformity_scores(y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute non-conformity scores for each sample.
    s(x, y) = 1 - f(x)_y
    """
    scores = np.where(y_true == 1, 1 - y_prob, y_prob)
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal quantile threshold.
    q̂ = ⌈(n+1)α⌉ / n -th quantile of scores.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * alpha) / n
    level = min(level, 1.0)  # cap at 1.0
    return np.quantile(scores, level)


def prediction_set(y_prob: float, threshold: float) -> list:
    """
    Construct prediction set for a single sample.
    Include class y if its non-conformity score ≤ threshold.
    """
    pset = []
    # Class 1 (indication): score = 1 - y_prob
    if (1 - y_prob) <= threshold:
        pset.append(1)
    # Class 0 (no indication): score = y_prob
    if y_prob <= threshold:
        pset.append(0)
    return pset


# ── Marginal split conformal prediction ──────────────────────────────────

def marginal_cp(cal_prob, cal_y, test_prob, test_y, alpha=PRIMARY_ALPHA):
    """
    Standard split conformal prediction with a single global threshold.
    """
    # Calibration: compute non-conformity scores and threshold
    cal_scores = nonconformity_scores(cal_prob, cal_y)
    threshold = conformal_quantile(cal_scores, alpha)

    # Test: construct prediction sets
    test_sets = [prediction_set(p, threshold) for p in test_prob]
    set_sizes = np.array([len(s) for s in test_sets])

    # Coverage: fraction of test samples where true label is in prediction set
    covered = np.array([test_y[i] in test_sets[i] for i in range(len(test_y))])
    coverage = covered.mean()

    # Bootstrap CI for coverage
    rng = np.random.RandomState(RANDOM_STATE)
    boot_coverages = []
    for _ in range(1000):
        idx = rng.choice(len(covered), size=len(covered), replace=True)
        boot_coverages.append(covered[idx].mean())
    ci_low, ci_high = np.percentile(boot_coverages, [2.5, 97.5])

    return {
        "alpha": alpha,
        "threshold": float(threshold),
        "coverage": float(coverage),
        "coverage_ci_low": float(ci_low),
        "coverage_ci_high": float(ci_high),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "empty_sets": int((set_sizes == 0).sum()),
        "singleton_sets": int((set_sizes == 1).sum()),
        "both_classes_sets": int((set_sizes == 2).sum()),
        "n_calibration": len(cal_y),
        "n_test": len(test_y),
        "test_sets": test_sets,
        "covered": covered,
    }


# ── Mondrian (group-conditional) conformal prediction ────────────────────

def mondrian_cp(cal_prob, cal_y, cal_groups, test_prob, test_y, test_groups, alpha=PRIMARY_ALPHA):
    """
    Group-conditional conformal prediction.
    Separate threshold per therapeutic area group.
    """
    unique_groups = sorted(set(cal_groups) | set(test_groups))
    group_thresholds = {}
    group_results = {}

    # Compute per-group thresholds from calibration data
    for group in unique_groups:
        cal_mask = np.array([g == group for g in cal_groups])
        if cal_mask.sum() == 0:
            group_thresholds[group] = float("inf")  # no calibration data → include everything
            continue

        cal_scores_g = nonconformity_scores(cal_prob[cal_mask], cal_y[cal_mask])
        group_thresholds[group] = conformal_quantile(cal_scores_g, alpha)

    # Test: construct prediction sets using group-specific thresholds
    test_sets = []
    for i in range(len(test_y)):
        g = test_groups[i]
        thresh = group_thresholds.get(g, float("inf"))
        test_sets.append(prediction_set(test_prob[i], thresh))

    set_sizes = np.array([len(s) for s in test_sets])
    covered = np.array([test_y[i] in test_sets[i] for i in range(len(test_y))])

    # Per-group coverage
    for group in unique_groups:
        test_mask = np.array([g == group for g in test_groups])
        if test_mask.sum() == 0:
            continue
        g_covered = covered[test_mask]
        g_sizes = set_sizes[test_mask]
        cal_mask = np.array([g == group for g in cal_groups])

        group_results[group] = {
            "threshold": float(group_thresholds.get(group, float("inf"))),
            "coverage": float(g_covered.mean()),
            "mean_set_size": float(g_sizes.mean()),
            "n_calibration": int(cal_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_positive_test": int(test_y[test_mask].sum()),
        }

    overall_coverage = covered.mean()

    return {
        "alpha": alpha,
        "overall_coverage": float(overall_coverage),
        "mean_set_size": float(set_sizes.mean()),
        "group_thresholds": {g: float(t) for g, t in group_thresholds.items()},
        "group_results": group_results,
        "n_groups": len(unique_groups),
        "test_sets": test_sets,
        "covered": covered,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main():
    t_total = time.time()
    print("Script 06: Conformal prediction calibration")
    print("=" * 70)

    # Load predictions and metadata
    print("\n[1/5] Loading data...")
    preds = pd.read_parquet(PROCESSED_DIR / "xgboost_predictions.parquet")
    labelled = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")

    # Add therapeutic area group
    if "therapeutic_area_name" in labelled.columns:
        labelled["ta_group"] = labelled["therapeutic_area_name"].apply(map_therapeutic_area)
    else:
        labelled["ta_group"] = "other"

    # Merge TA group into predictions
    preds = preds.merge(
        labelled[["drug_id", "disease_id", "ta_group"]].drop_duplicates(),
        on=["drug_id", "disease_id"],
        how="left",
    )
    preds["ta_group"] = preds["ta_group"].fillna("other")

    # Split
    cal = preds[preds["split"] == "calibration"].dropna(subset=["y_prob"])
    test = preds[preds["split"] == "test"].dropna(subset=["y_prob"])

    cal_prob = cal["y_prob"].values
    cal_y = cal["label"].values
    cal_groups = cal["ta_group"].values.tolist()

    test_prob = test["y_prob"].values
    test_y = test["label"].values
    test_groups = test["ta_group"].values.tolist()

    print(f"Calibration set: {len(cal):,} samples ({cal_y.sum():,} positive)")
    print(f"Test set: {len(test):,} samples ({test_y.sum():,} positive)")
    print(f"Therapeutic area groups: {sorted(set(cal_groups))}")

    results = {"marginal": {}, "mondrian": {}}

    # ── Marginal CP at multiple alpha levels ──────────────────────────────
    print(f"\n[2/5] Marginal split conformal prediction...")
    print("=" * 70)

    for alpha in ALPHA_LEVELS:
        res = marginal_cp(cal_prob, cal_y, test_prob, test_y, alpha)
        results["marginal"][str(alpha)] = {
            k: v for k, v in res.items() if k not in ("test_sets", "covered")
        }
        print(f"\nα = {alpha:.2f}:")
        print(f"  Threshold: {res['threshold']:.4f}")
        print(f"  Coverage: {res['coverage']:.4f} "
              f"(95% CI: {res['coverage_ci_low']:.4f}–{res['coverage_ci_high']:.4f})")
        print(f"  Mean set size: {res['mean_set_size']:.2f}")
        print(f"  Empty/singleton/both: "
              f"{res['empty_sets']}/{res['singleton_sets']}/{res['both_classes_sets']}")

    # ── Mondrian CP at primary alpha ──────────────────────────────────────
    print(f"\n[3/5] Mondrian (group-conditional) conformal prediction (α = {PRIMARY_ALPHA})...")
    print("=" * 70)

    mondrian_res = mondrian_cp(
        cal_prob, cal_y, cal_groups,
        test_prob, test_y, test_groups,
        alpha=PRIMARY_ALPHA,
    )

    results["mondrian"][str(PRIMARY_ALPHA)] = {
        k: v for k, v in mondrian_res.items() if k not in ("test_sets", "covered")
    }

    print(f"\nOverall coverage: {mondrian_res['overall_coverage']:.4f}")
    print(f"Mean set size: {mondrian_res['mean_set_size']:.2f}")
    print(f"\nPer-group results:")
    for group, gres in sorted(mondrian_res["group_results"].items()):
        print(f"  {group:20s}: coverage={gres['coverage']:.4f}, "
              f"set_size={gres['mean_set_size']:.2f}, "
              f"n_cal={gres['n_calibration']}, n_test={gres['n_test']}")

    # ── Coverage gap analysis ─────────────────────────────────────────────
    print(f"\n[4/5] Coverage gap analysis: marginal vs Mondrian...")

    marginal_90 = marginal_cp(cal_prob, cal_y, test_prob, test_y, PRIMARY_ALPHA)

    # Per-group coverage under marginal CP
    print(f"\nPer-group coverage under MARGINAL CP (single threshold):")
    for group in sorted(set(test_groups)):
        mask = np.array([g == group for g in test_groups])
        if mask.sum() == 0:
            continue
        g_covered_marginal = marginal_90["covered"][mask].mean()
        g_covered_mondrian = mondrian_res["covered"][mask].mean()
        gap = g_covered_mondrian - g_covered_marginal
        print(f"  {group:20s}: marginal={g_covered_marginal:.4f}, "
              f"mondrian={g_covered_mondrian:.4f}, gap={gap:+.4f}")

    results["coverage_gap"] = {
        "marginal_overall": float(marginal_90["coverage"]),
        "mondrian_overall": float(mondrian_res["overall_coverage"]),
    }

    # ── Save prediction sets ──────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")
    test_df = test.copy()
    test_df["marginal_set"] = marginal_90["test_sets"]
    test_df["marginal_set_size"] = [len(s) for s in marginal_90["test_sets"]]
    test_df["marginal_covered"] = marginal_90["covered"]
    test_df["mondrian_set"] = mondrian_res["test_sets"]
    test_df["mondrian_set_size"] = [len(s) for s in mondrian_res["test_sets"]]
    test_df["mondrian_covered"] = mondrian_res["covered"]

    cp_path = PROCESSED_DIR / "conformal_prediction_sets.parquet"
    # Convert list columns to strings for parquet compatibility
    test_df["marginal_set"] = test_df["marginal_set"].apply(str)
    test_df["mondrian_set"] = test_df["mondrian_set"].apply(str)
    test_df.to_parquet(cp_path, index=False)
    print(f"\nPrediction sets saved: {cp_path}")

    # ── Save results ──────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "conformal_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # ── Key finding summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print("=" * 70)
    m90 = results["marginal"][str(PRIMARY_ALPHA)]
    print(f"H1 (marginal coverage ≈ 90%): "
          f"{m90['coverage']:.4f} "
          f"(CI: {m90['coverage_ci_low']:.4f}–{m90['coverage_ci_high']:.4f})")

    if mondrian_res["group_results"]:
        coverages = [g["coverage"] for g in mondrian_res["group_results"].values()
                     if g["n_test"] > 0]
        if coverages:
            coverage_range = max(coverages) - min(coverages)
            print(f"H2 (coverage uniformity): range = {coverage_range:.4f} "
                  f"(min={min(coverages):.4f}, max={max(coverages):.4f})")

    print(f"Mean prediction set size: {m90['mean_set_size']:.2f}")

    print(f"\n{'=' * 70}")
    print(f"Script 06 complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
