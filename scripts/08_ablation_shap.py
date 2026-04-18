#!/usr/bin/env python3
"""
Script 08: SHAP feature importance and single-source ablation.

Analyses:
  1. SHAP values for XGBoost model — which evidence sources drive predictions?
  2. Single-source ablation: retrain with each source removed, measure AUROC drop
  3. Sensitivity to negative-sample ratio
  4. Sensitivity to temporal split boundaries

Outputs:
  - outputs/figures/feature_importance.png
  - outputs/figures/shap_summary.png (if shap available)
  - outputs/tables/shap_importance.csv
  - outputs/tables/ablation_results.csv
  - outputs/tables/sensitivity_results.json
"""

import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import duckdb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not available (numba/llvmlite missing on Python 3.14)")
    print("  SHAP analysis will use XGBoost built-in importance instead.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_model_and_data():
    model_path = OUTPUT_DIR / "xgboost_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")
    with open(PROCESSED_DIR / "datasource_list.json") as f:
        datasources = json.load(f)

    # Load tuned hyperparameters from training
    with open(OUTPUT_DIR / "xgboost_metrics.json") as f:
        metrics = json.load(f)
    best_params = metrics["best_params"]

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
    return model, df, feature_cols, datasources, best_params


# ── SHAP Analysis ─────────────────────────────────────────────────────────

def _pretty_feature_name(feat):
    """Convert internal feature names to readable labels for figures."""
    LABELS = {
        "europepmc": "Literature mining (Europe PMC)",
        "gwas_credible_sets": "GWAS credible sets",
        "cancer_gene_census": "Cancer Gene Census",
        "cancer_biomarkers": "Cancer biomarkers",
        "expression_atlas": "Expression Atlas",
        "eva": "ClinVar (EVA)",
        "eva_somatic": "ClinVar somatic",
        "impc": "Mouse phenotyping (IMPC)",
        "intogen": "IntOGen (cancer drivers)",
        "orphanet": "Orphanet (rare disease)",
        "genomics_england": "Genomics England",
        "reactome": "Reactome pathways",
        "gene2phenotype": "Gene2Phenotype",
        "gene_burden": "Gene burden",
        "uniprot_literature": "UniProt literature",
        "uniprot_variants": "UniProt variants",
        "clingen": "ClinGen",
        "crispr": "Project Score (CRISPR)",
        "crispr_screen": "CRISPR screen",
        "n_targets": "Number of drug targets",
        "target_degree_max": "Target degree (max)",
        "target_degree_mean": "Target degree (mean)",
        "target_mean_interaction_score": "Target interaction score",
        "target_disease_network_overlap": "Target–disease network overlap",
        "disease_ontology_max_sim": "Disease ontology sim. (max)",
        "disease_ontology_mean_sim": "Disease ontology sim. (mean)",
        "disease_ontology_n_related": "Related diseases (count)",
        "drug_fp_sim_max": "Drug fingerprint sim. (max)",
        "drug_fp_sim_mean": "Drug fingerprint sim. (mean)",
        "drug_has_smiles": "Drug has SMILES",
    }
    return LABELS.get(feat, feat.replace("_", " ").title())


def _plot_importance_bar(shap_df, importance_col, fig_path):
    """Publication-quality horizontal bar chart of feature importance."""
    plot_df = shap_df.head(20).copy()
    plot_df = plot_df.iloc[::-1]  # reverse for horizontal bar (top at top)
    plot_df["label"] = plot_df["feature"].apply(_pretty_feature_name)

    # Colour by feature category
    def _colour(feat):
        if feat.startswith(("target_", "n_targets")):
            return "#2196F3"   # blue — network/target
        if feat.startswith(("disease_ontology",)):
            return "#4CAF50"   # green — ontology
        if feat.startswith(("drug_fp", "drug_has")):
            return "#FF9800"   # orange — chemical
        return "#78909C"       # grey — OT evidence sources

    colours = [_colour(f) for f in plot_df["feature"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_df["label"], plot_df[importance_col], color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(importance_col.replace("_", " ").title(), fontsize=11)
    ax.set_title("Feature importance (XGBoost gain)", fontsize=13, pad=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for colour categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#78909C", label="Open Targets evidence"),
        Patch(facecolor="#2196F3", label="Network / target features"),
        Patch(facecolor="#4CAF50", label="Disease ontology features"),
        Patch(facecolor="#FF9800", label="Drug chemical features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot: {fig_path}")


def shap_analysis(model, df, feature_cols):
    """Compute SHAP values or fall back to XGBoost built-in importance."""
    print("\n── Feature Importance ──")

    if HAS_SHAP:
        # Use test set for SHAP (most relevant for understanding model behaviour)
        test = df[df["split"] == "test"]
        X_test = test[feature_cols].values.astype(np.float32)

        # TreeExplainer is exact and fast for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        print("\nTop 15 features by mean |SHAP|:")
        print(shap_df.head(15).to_string(index=False))
        shap_df.to_csv(TABLE_DIR / "shap_importance.csv", index=False)

        # SHAP beeswarm plot
        shap.summary_plot(
            shap_values, X_test,
            feature_names=[_pretty_feature_name(f) for f in feature_cols],
            show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  SHAP summary plot: {FIG_DIR / 'shap_summary.png'}")

        # Also generate bar chart for consistency
        _plot_importance_bar(shap_df, "mean_abs_shap", FIG_DIR / "feature_importance.png")
    else:
        # Fallback: XGBoost gain-based importance
        print("  Using XGBoost gain-based feature importance (SHAP unavailable)")
        importance = model.feature_importances_
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        print("\nTop 15 features by XGBoost importance:")
        print(shap_df.head(15).to_string(index=False))
        shap_df.to_csv(TABLE_DIR / "shap_importance.csv", index=False)

        # Publication-quality bar chart (always generated)
        _plot_importance_bar(shap_df, "importance", FIG_DIR / "feature_importance.png")

    return shap_df


# ── Single-source ablation ────────────────────────────────────────────────

def source_ablation(df, feature_cols, datasources, best_params):
    """
    Retrain XGBoost with each feature removed one at a time.
    Uses tuned hyperparameters from Script 04 for consistency.
    """
    all_ablation_features = list(feature_cols)
    n_ablations = len(all_ablation_features)
    print(f"\n── Single-feature ablation ({n_ablations} features) ──")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train = train["label"].values
    y_test = test["label"].values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    model_kwargs = {
        **best_params,
        "scale_pos_weight": scale_pos,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
    }

    # Baseline: full model performance on test
    print("  Training baseline (all features)...")
    t_base = time.time()
    X_train_full = train[feature_cols].values.astype(np.float32)
    X_test_full = test[feature_cols].values.astype(np.float32)

    base_model = xgb.XGBClassifier(
        **model_kwargs,
        early_stopping_rounds=10,
    )
    base_model.fit(
        X_train_full, y_train,
        eval_set=[(X_test_full, y_test)],
        verbose=False,
    )
    base_prob = base_model.predict_proba(X_test_full)[:, 1]
    base_auroc = roc_auc_score(y_test, base_prob)
    base_auprc = average_precision_score(y_test, base_prob)
    base_sec = time.time() - t_base

    print(f"  Baseline: AUROC={base_auroc:.4f}, AUPRC={base_auprc:.4f} ({fmt_elapsed(base_sec)})")
    est_total = base_sec * n_ablations
    print(f"  Estimated ablation time: {fmt_elapsed(est_total)} "
          f"({n_ablations} retrains × ~{base_sec:.0f}s each)")

    results = []
    t_start = time.time()

    for i, source in enumerate(all_ablation_features):
        t_iter = time.time()
        ablated_cols = [c for c in feature_cols if c != source]
        X_train_abl = train[ablated_cols].values.astype(np.float32)
        X_test_abl = test[ablated_cols].values.astype(np.float32)

        abl_model = xgb.XGBClassifier(
            **model_kwargs,
            early_stopping_rounds=10,
        )
        abl_model.fit(
            X_train_abl, y_train,
            eval_set=[(X_test_abl, y_test)],
            verbose=False,
        )
        abl_prob = abl_model.predict_proba(X_test_abl)[:, 1]
        abl_auroc = roc_auc_score(y_test, abl_prob)
        abl_auprc = average_precision_score(y_test, abl_prob)

        delta_auroc = abl_auroc - base_auroc
        delta_auprc = abl_auprc - base_auprc
        iter_sec = time.time() - t_iter
        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (n_ablations - i - 1)

        results.append({
            "removed_source": source,
            "auroc": abl_auroc,
            "auprc": abl_auprc,
            "delta_auroc": delta_auroc,
            "delta_auprc": delta_auprc,
        })

        print(f"    [{i+1:2d}/{n_ablations}] −{source}: "
              f"AUROC={abl_auroc:.4f} (Δ{delta_auroc:+.4f}) — "
              f"{iter_sec:.0f}s (ETA {fmt_elapsed(eta)})", flush=True)

    ablation_df = pd.DataFrame(results).sort_values("delta_auroc")
    ablation_df.to_csv(TABLE_DIR / "ablation_results.csv", index=False)

    print(f"\n  Most impactful features (largest AUROC drop when removed):")
    print(ablation_df.head(10).to_string(index=False))

    print(f"\n  Least impactful features:")
    print(ablation_df.tail(5).to_string(index=False))

    return ablation_df


# ── Sensitivity: negative sample ratio ────────────────────────────────────

def sensitivity_neg_ratio(df, feature_cols):
    """Test model performance at different negative:positive ratios."""
    print(f"\n── Sensitivity: negative sample ratio ──")
    t = time.time()

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    train_pos = train[train["label"] == 1]
    train_neg = train[train["label"] == 0]
    n_pos = len(train_pos)

    y_test = test["label"].values
    X_test = test[feature_cols].values.astype(np.float32)

    ratios = [1, 2, 5, 10, 20]
    results = []

    for ratio in ratios:
        n_neg_sample = min(n_pos * ratio, len(train_neg))
        if n_neg_sample == 0:
            continue

        neg_sample = train_neg.sample(n=n_neg_sample, random_state=RANDOM_STATE)
        train_subset = pd.concat([train_pos, neg_sample])

        X_train = train_subset[feature_cols].values.astype(np.float32)
        y_train = train_subset["label"].values

        model = xgb.XGBClassifier(
            scale_pos_weight=n_neg_sample / n_pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "neg_pos_ratio": ratio,
            "n_train": len(train_subset),
            "auroc": roc_auc_score(y_test, y_prob),
            "auprc": average_precision_score(y_test, y_prob),
        })

        print(f"  Ratio {ratio}:1 — AUROC={results[-1]['auroc']:.4f}, "
              f"AUPRC={results[-1]['auprc']:.4f}")

    print(f"  Sensitivity analysis done ({fmt_elapsed(time.time() - t)})")
    return results


# ── Sensitivity: temporal split boundaries ───────────────────────────────

def sensitivity_temporal_split(feature_cols, best_params):
    """
    Re-label and retrain with shifted temporal boundaries.

    Default:  train ≤2019, cal 2020–2022, test 2023–2025
    Shift −1: train ≤2018, cal 2019–2021, test 2022–2025
    Shift +1: train ≤2020, cal 2021–2023, test 2024–2025

    Reloads raw features + evidence years and re-splits to avoid
    data leakage from the original split assignment.
    """
    print("\n── Sensitivity: temporal split boundaries ──")
    t = time.time()

    # Load the full features and evidence year data
    df_feat = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")

    # We need the evidence year to re-split. It's stored in the labelled file
    # if it was kept, or we reconstruct from the evidence_years file.
    if "first_evidence_year" not in df_feat.columns:
        # Try to load evidence years separately
        ey_path = PROCESSED_DIR / "drug_disease_evidence_years.parquet"
        if ey_path.exists():
            ey = pd.read_parquet(ey_path)
            df_feat = df_feat.merge(ey, on=["drug_id", "disease_id"], how="left")
        else:
            print("  WARNING: first_evidence_year not available — skipping temporal sensitivity")
            return []

    # Define shift configurations
    shifts = [
        {"name": "−1 year", "train_end": 2018, "cal_start": 2019, "cal_end": 2021, "test_start": 2022},
        {"name": "default", "train_end": 2019, "cal_start": 2020, "cal_end": 2022, "test_start": 2023},
        {"name": "+1 year", "train_end": 2020, "cal_start": 2021, "cal_end": 2023, "test_start": 2024},
    ]

    rng = np.random.RandomState(RANDOM_STATE)
    results = []

    for shift in shifts:
        t_iter = time.time()
        name = shift["name"]
        te, cs, ce, ts = shift["train_end"], shift["cal_start"], shift["cal_end"], shift["test_start"]

        # Re-assign splits for positives based on evidence year
        df_s = df_feat.copy()
        pos_mask = df_s["label"] == 1
        neg_mask = df_s["label"] == 0

        # Assign positive splits
        df_s["_split"] = np.nan
        ey = df_s.loc[pos_mask, "first_evidence_year"]
        df_s.loc[pos_mask & (ey <= te), "_split"] = "train"
        df_s.loc[pos_mask & (ey >= cs) & (ey <= ce), "_split"] = "calibration"
        df_s.loc[pos_mask & (ey >= ts), "_split"] = "test"

        # Drop positives without a valid split
        valid_pos = df_s["_split"].notna() | neg_mask
        df_s = df_s[valid_pos].copy()
        pos_mask = df_s["label"] == 1
        neg_mask = df_s["label"] == 0

        n_train_pos = ((df_s["_split"] == "train") & pos_mask).sum()
        n_cal_pos = ((df_s["_split"] == "calibration") & pos_mask).sum()
        n_test_pos = ((df_s["_split"] == "test") & pos_mask).sum()

        if n_test_pos < 10 or n_cal_pos < 10:
            print(f"  {name}: skipped (test={n_test_pos}, cal={n_cal_pos} positives — too few)")
            continue

        # Assign negatives proportionally (5:1 ratio for cal/test)
        neg_indices = df_s.index[neg_mask].values
        rng.shuffle(neg_indices)
        n_cal_neg = min(n_cal_pos * 5, len(neg_indices) // 4)
        n_test_neg = min(n_test_pos * 5, len(neg_indices) // 4)
        df_s.loc[neg_indices[:n_cal_neg], "_split"] = "calibration"
        df_s.loc[neg_indices[n_cal_neg:n_cal_neg + n_test_neg], "_split"] = "test"
        df_s.loc[neg_mask & df_s["_split"].isna(), "_split"] = "train"

        # Train and evaluate
        train_s = df_s[df_s["_split"] == "train"]
        test_s = df_s[df_s["_split"] == "test"]

        X_train = train_s[feature_cols].values.astype(np.float32)
        y_train = train_s["label"].values
        X_test = test_s[feature_cols].values.astype(np.float32)
        y_test = test_s["label"].values

        n_neg_tr = (y_train == 0).sum()
        n_pos_tr = (y_train == 1).sum()
        spw = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0

        mdl = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            early_stopping_rounds=10,
        )
        mdl.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = mdl.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        sec = time.time() - t_iter

        results.append({
            "shift": name,
            "train_end": te,
            "cal_range": f"{cs}–{ce}",
            "test_start": ts,
            "n_train_pos": int(n_train_pos),
            "n_cal_pos": int(n_cal_pos),
            "n_test_pos": int(n_test_pos),
            "auroc": auroc,
            "auprc": auprc,
        })

        print(f"  {name} (train≤{te}, cal {cs}–{ce}, test≥{ts}): "
              f"AUROC={auroc:.4f}, AUPRC={auprc:.4f} "
              f"(test: {n_test_pos} pos, {len(test_s) - n_test_pos} neg) — {fmt_elapsed(sec)}")

    print(f"  Temporal sensitivity done ({fmt_elapsed(time.time() - t)})")
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main():
    t_total = time.time()
    print("Script 08: SHAP analysis and ablation studies")
    print("=" * 70)

    print("\n[1/5] Loading model and data...")
    model, df, feature_cols, datasources, best_params = load_model_and_data()
    print(f"  {len(feature_cols)} features, {len(df):,} samples")
    print(f"  Tuned params: {best_params}")

    # SHAP
    print("\n[2/5] Feature importance analysis...")
    shap_df = shap_analysis(model, df, feature_cols)

    # Ablation
    print("\n[3/5] Single-feature ablation...")
    ablation_df = source_ablation(df, feature_cols, datasources, best_params)

    # Sensitivity: negative ratio
    print("\n[4/5] Negative ratio sensitivity...")
    neg_ratio_results = sensitivity_neg_ratio(df, feature_cols)

    # Sensitivity: temporal splits
    print("\n[5/5] Temporal split sensitivity...")
    temporal_results = sensitivity_temporal_split(feature_cols, best_params)

    # Save combined sensitivity results
    sensitivity = {
        "neg_ratio": neg_ratio_results,
        "temporal_split": temporal_results,
    }
    with open(TABLE_DIR / "sensitivity_results.json", "w") as f:
        json.dump(sensitivity, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Script 08 complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
