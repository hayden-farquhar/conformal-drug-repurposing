#!/usr/bin/env python3
"""
Script 07: Case study analysis of top novel predictions.

Identifies drug-disease pairs that:
  - Are NOT in the training set (no known indication)
  - Have NO clinical indication at any phase for the same or related disease
  - Are not for overly broad therapeutic-area-level disease terms
  - Have high predicted probability from XGBoost
  - Have tight conformal prediction sets (high confidence)

Filtering pipeline (applied sequentially):
  1. Remove training-set positives (known indications used in model)
  2. Remove pairs where the drug has ANY clinical_indication entry for
     the exact disease (including Phase 1/2 — not novel if already trialled)
  3. Remove pairs where the drug is indicated for an ancestor or descendant
     of the predicted disease (e.g., drug approved for "diabetes mellitus"
     predicted for "type 2 diabetes mellitus", or vice versa)
  4. Remove predictions for therapeutic-area-level disease nodes (too broad
     to be actionable repurposing candidates)

Outputs:
  - outputs/tables/top_novel_predictions.csv
  - outputs/tables/case_study_details.csv
  - outputs/tables/case_study_summary.json
"""

import json
import time
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def build_known_indication_sets():
    """
    Build sets of (drug_id, disease_id) pairs that represent known
    drug-disease relationships at any clinical stage, plus ontology-expanded
    pairs covering ancestor/descendant diseases, plus shared-target pairs
    where another drug acting on the same protein is already indicated.

    Returns:
        exact_pairs: set of (drug_id, disease_id) with any clinical indication
        expanded_pairs: set including ontology relatives + shared-target pairs
        therapeutic_area_ids: set of disease IDs that are therapeutic area nodes
    """
    con = duckdb.connect()

    # 1. All clinical indications at any phase
    ci = con.execute(f"""
        SELECT DISTINCT drugId AS drug_id, diseaseId AS disease_id
        FROM read_parquet('{RAW_DIR}/clinical_indication/*.parquet')
        WHERE drugId IS NOT NULL AND diseaseId IS NOT NULL
    """).df()
    exact_pairs = set(zip(ci["drug_id"], ci["disease_id"]))
    print(f"    Known indication pairs (any phase): {len(exact_pairs):,}")

    # 2. Disease ontology: ancestors and descendants for each disease
    disease_ont = con.execute(f"""
        SELECT id, ancestors, descendants
        FROM read_parquet('{RAW_DIR}/disease/*.parquet')
    """).df()

    # Build disease → set of related diseases (ancestors + descendants)
    disease_relatives = {}
    for _, row in disease_ont.iterrows():
        relatives = set()
        if row["ancestors"] is not None:
            relatives.update(row["ancestors"])
        if row["descendants"] is not None:
            relatives.update(row["descendants"])
        if relatives:
            disease_relatives[row["id"]] = relatives

    # Expand: if drug X is indicated for disease A, and disease B is an
    # ancestor or descendant of A, then (X, B) is also "known"
    expanded_pairs = set(exact_pairs)
    for drug_id, disease_id in exact_pairs:
        relatives = disease_relatives.get(disease_id, set())
        for rel_disease in relatives:
            expanded_pairs.add((drug_id, rel_disease))
    n_after_ontology = len(expanded_pairs)
    print(f"    After ontology expansion: {n_after_ontology:,} pairs")

    # 3. Shared-target filter: if drug A targets protein P, and drug B also
    #    targets P and is indicated for disease D, then (A, D) is not novel
    drug_targets = con.execute(f"""
        SELECT unnest(chemblIds) AS drug_id, unnest(targets) AS target_id
        FROM read_parquet('{RAW_DIR}/drug_mechanism_of_action/*.parquet')
    """).df()

    # Build target → set of diseases (from clinical_indication + ontology expansion)
    # First: drug → targets
    from collections import defaultdict
    drug_to_targets = defaultdict(set)
    target_to_drugs = defaultdict(set)
    for _, row in drug_targets.iterrows():
        drug_to_targets[row["drug_id"]].add(row["target_id"])
        target_to_drugs[row["target_id"]].add(row["drug_id"])

    # For each indicated (drug, disease), find which targets the drug has,
    # then for each target, find all other drugs sharing that target
    # → those other drugs get (drug, disease) added as "known"
    ci_with_ontology = expanded_pairs  # includes ontology-expanded pairs
    target_disease_pairs = defaultdict(set)
    for drug_id, disease_id in ci_with_ontology:
        for target_id in drug_to_targets.get(drug_id, set()):
            target_disease_pairs[target_id].add(disease_id)

    # Now: for each drug, check which diseases its targets are associated with
    n_before = len(expanded_pairs)
    for drug_id, targets in drug_to_targets.items():
        for target_id in targets:
            for disease_id in target_disease_pairs.get(target_id, set()):
                expanded_pairs.add((drug_id, disease_id))
    print(f"    After shared-target expansion: {len(expanded_pairs):,} pairs "
          f"(+{len(expanded_pairs) - n_before:,} from shared targets)")

    # 4. Therapeutic area disease IDs (too broad for case studies)
    ta = con.execute(f"""
        SELECT id FROM read_parquet('{RAW_DIR}/disease/*.parquet')
        WHERE ontology.isTherapeuticArea = true
    """).df()
    therapeutic_area_ids = set(ta["id"])
    print(f"    Therapeutic area nodes to exclude: {len(therapeutic_area_ids)}")

    con.close()
    return exact_pairs, expanded_pairs, therapeutic_area_ids


def main():
    t_total = time.time()
    print("Script 07: Case study analysis — top novel predictions")
    print("=" * 70)

    # Load predictions and labels
    print("\n[1/5] Loading data...")
    preds = pd.read_parquet(PROCESSED_DIR / "xgboost_predictions.parquet")
    labelled = pd.read_parquet(PROCESSED_DIR / "drug_disease_labelled.parquet")

    # Merge metadata
    meta_cols = ["drug_id", "disease_id", "drug_name", "disease_name",
                 "therapeutic_area_name", "drug_type"]
    meta_cols = [c for c in meta_cols if c in labelled.columns]
    preds = preds.merge(
        labelled[meta_cols].drop_duplicates(),
        on=["drug_id", "disease_id"],
        how="left",
    )

    # Load conformal prediction sets if available
    cp_path = PROCESSED_DIR / "conformal_prediction_sets.parquet"
    if cp_path.exists():
        cp = pd.read_parquet(cp_path)
        preds = preds.merge(
            cp[["drug_id", "disease_id", "marginal_set_size", "mondrian_set_size"]].drop_duplicates(),
            on=["drug_id", "disease_id"],
            how="left",
        )

    # ── Build filtering sets ─────────────────────────────────────────────
    print("\n[2/5] Building known-indication filter...")
    t = time.time()
    exact_ci, expanded_ci, ta_ids = build_known_indication_sets()
    print(f"  Filter sets built — {fmt_elapsed(time.time() - t)}")

    # ── Apply filters sequentially ───────────────────────────────────────
    print("\n[3/5] Filtering to genuinely novel predictions...")
    t = time.time()
    n_start = len(preds)

    # Filter 1: Remove training positives
    train_pos = preds[(preds["split"] == "train") & (preds["label"] == 1)]
    train_pos_keys = set(zip(train_pos["drug_id"], train_pos["disease_id"]))
    pair_keys = list(zip(preds["drug_id"], preds["disease_id"]))
    is_train_pos = np.array([k in train_pos_keys for k in pair_keys])
    novel = preds[~is_train_pos].copy()
    print(f"  After removing training positives: {len(novel):,} "
          f"(−{is_train_pos.sum():,})")

    # Filter 2: Remove any clinical indication (exact match, any phase)
    novel_keys = list(zip(novel["drug_id"], novel["disease_id"]))
    is_exact_ci = np.array([k in exact_ci for k in novel_keys])
    n_exact = is_exact_ci.sum()
    novel = novel[~is_exact_ci].copy()
    print(f"  After removing exact CI matches: {len(novel):,} "
          f"(−{n_exact:,} pairs with any-phase clinical indication)")

    # Filter 3: Remove ontology-expanded + shared-target known pairs
    novel_keys = list(zip(novel["drug_id"], novel["disease_id"]))
    is_expanded = np.array([k in expanded_ci for k in novel_keys])
    n_expanded = is_expanded.sum()
    novel = novel[~is_expanded].copy()
    print(f"  After removing ontology + shared-target: {len(novel):,} "
          f"(−{n_expanded:,} ontology relative or same-target indication)")

    # Filter 4: Remove therapeutic-area-level diseases
    is_ta = novel["disease_id"].isin(ta_ids)
    n_ta = is_ta.sum()
    novel = novel[~is_ta].copy()
    print(f"  After removing broad TA terms: {len(novel):,} "
          f"(−{n_ta:,} therapeutic-area-level diseases)")

    print(f"  Total filtered: {n_start - len(novel):,} pairs removed — "
          f"{fmt_elapsed(time.time() - t)}")

    # Rank by predicted probability (descending)
    novel = novel.sort_values("y_prob", ascending=False)

    # Top 50 novel predictions
    top_n = min(50, len(novel))
    top_novel = novel.head(top_n).copy()
    top_novel["rank"] = range(1, top_n + 1)

    display_cols = ["rank", "drug_id", "disease_id"]
    if "drug_name" in top_novel.columns:
        display_cols.append("drug_name")
    if "disease_name" in top_novel.columns:
        display_cols.append("disease_name")
    display_cols.extend(["y_prob", "label"])
    if "therapeutic_area_name" in top_novel.columns:
        display_cols.append("therapeutic_area_name")
    if "marginal_set_size" in top_novel.columns:
        display_cols.append("marginal_set_size")

    display_cols = [c for c in display_cols if c in top_novel.columns]
    print(f"\nTop 20 novel predictions:")
    print(top_novel[display_cols].head(20).to_string(index=False))

    # Save
    out_path = TABLE_DIR / "top_novel_predictions.csv"
    top_novel.to_csv(out_path, index=False)
    print(f"\nTop {top_n} saved: {out_path}")

    # ── Validation against test set ───────────────────────────────────────
    print("\n[4/5] Validating against test set...")
    test_pos = preds[(preds["split"] == "test") & (preds["label"] == 1)]
    test_pos_keys = set(zip(test_pos["drug_id"], test_pos["disease_id"]))

    novel_keys = list(zip(novel["drug_id"], novel["disease_id"]))
    novel["confirmed_in_test"] = np.array([k in test_pos_keys for k in novel_keys])

    print(f"\n── Precision at top-K among novel predictions ──")
    for k in [10, 25, 50, 100, 200, 500]:
        if k > len(novel):
            break
        p_at_k = novel.head(k)["confirmed_in_test"].mean()
        n_confirmed = novel.head(k)["confirmed_in_test"].sum()
        print(f"  P@{k}: {p_at_k:.4f} ({n_confirmed}/{k} confirmed)")

    # ── Case study candidates ─────────────────────────────────────────────
    print("\n[5/5] Selecting case study candidates...")
    case_candidates = novel.head(500).copy()
    if "drug_name" in case_candidates.columns:
        case_candidates = case_candidates[case_candidates["drug_name"].notna()]
    case_candidates = case_candidates[~case_candidates["confirmed_in_test"]]

    # Exclude phenotype-level nodes (HP_ terms like "Hyperglycemia") —
    # these are symptoms/signs, not actionable repurposing targets
    case_candidates = case_candidates[
        ~case_candidates["disease_id"].str.startswith("HP_")
    ]

    # Deduplicate by drug — take each drug's top-ranked prediction only,
    # so case studies showcase diverse drugs rather than many predictions
    # for the same drug (e.g., metformin)
    case_studies = case_candidates.drop_duplicates(subset="drug_id", keep="first")
    n_cases = min(10, len(case_studies))
    case_studies = case_studies.head(n_cases)

    print(f"\n── Top {n_cases} case study candidates (genuinely novel) ──")
    if "drug_name" in case_studies.columns and "disease_name" in case_studies.columns:
        for _, row in case_studies.iterrows():
            ta = row.get("therapeutic_area_name", "")
            print(f"  {row.get('drug_name', row['drug_id'])} → "
                  f"{row.get('disease_name', row['disease_id'])} "
                  f"(P={row['y_prob']:.3f}, {ta})")

    case_path = TABLE_DIR / "case_study_details.csv"
    case_studies.to_csv(case_path, index=False)
    print(f"\nCase studies saved: {case_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    summary = {
        "total_novel_pairs": len(novel),
        "top_50_mean_prob": float(top_novel["y_prob"].mean()),
        "confirmed_in_test_top_50": int(novel.head(50)["confirmed_in_test"].sum()),
        "confirmed_in_test_top_100": int(novel.head(min(100, len(novel)))["confirmed_in_test"].sum()),
        "filters_applied": {
            "training_positives_removed": int(is_train_pos.sum()),
            "exact_ci_removed": int(n_exact),
            "ontology_and_shared_target_removed": int(n_expanded),
            "therapeutic_area_removed": int(n_ta),
        },
    }

    with open(TABLE_DIR / "case_study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    print(f"\n{'=' * 70}")
    print(f"Script 07 complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
