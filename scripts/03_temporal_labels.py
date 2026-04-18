#!/usr/bin/env python3
"""
Script 03: Temporal labelling and train/calibration/test split.

Builds binary labels for drug-disease pairs:
  - Positive: drug has a known clinical indication (APPROVAL or PHASE_3+)
  - Negative: drug-disease pair exists in OT evidence but has no known indication

Temporal signal sources (combined, taking earliest):
  1. Open Targets clinical_precedence timeseries — first year a non-zero
     clinical association score appears for the (target, disease) pair,
     bridged to (drug, disease) via drug_mechanism_of_action
  2. Drugs@FDA first approval year per drug (supplements not yet mapped)

Splits:
  - Train:       evidence year ≤ 2019
  - Calibration: evidence year 2020–2022
  - Test:        evidence year 2023–2025

Negatives are sampled into all three splits proportionally.

Inputs:
  - data/processed/drug_disease_features.parquet (from Script 02)
  - data/processed/drug_disease_evidence_years.parquet (pre-computed)
  - data/raw/clinical_indication/*.parquet (known indications)
  - data/raw/drugsfda/ (FDA approval dates)

Outputs:
  - data/processed/drug_disease_labelled.parquet (features + label + split)
  - data/processed/split_summary.json (counts per split)
"""

import duckdb
import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

con = duckdb.connect()

# ── Temporal split boundaries ─────────────────────────────────────────────
TRAIN_END = 2019
CAL_START, CAL_END = 2020, 2022
TEST_START, TEST_END = 2023, 2025

# Stage string → numeric phase mapping
STAGE_TO_PHASE = {
    "APPROVAL": 4,
    "PREAPPROVAL": 3.5,
    "PHASE_3": 3,
    "PHASE_2_3": 2.5,
    "PHASE_2": 2,
    "PHASE_1_2": 1.5,
    "PHASE_1": 1,
    "EARLY_PHASE_1": 0.5,
    "IND": 0.25,
    "PRECLINICAL": 0,
    "UNKNOWN": -1,
}


# ── Step 1: Extract known indications ─────────────────────────────────────

def extract_indications() -> pd.DataFrame:
    """Extract (drug_id, disease_id, max_phase) from clinical_indication."""
    ci_path = RAW_DIR / "clinical_indication"

    indications = con.execute(f"""
        SELECT
            drugId AS drug_id,
            diseaseId AS disease_id,
            maxClinicalStage AS clinical_stage
        FROM read_parquet('{ci_path}/*.parquet')
        WHERE drugId IS NOT NULL AND diseaseId IS NOT NULL
    """).df()

    indications["max_phase"] = indications["clinical_stage"].map(STAGE_TO_PHASE).fillna(-1)

    print(f"Clinical indication records: {len(indications):,}")
    print(f"Stage distribution:")
    print(indications["clinical_stage"].value_counts().to_string())

    return indications


# ── Step 2: Build evidence-year temporal signal ───────────────────────────

def build_evidence_years() -> pd.DataFrame:
    """
    Load or compute first-evidence-year for (drug, disease) pairs.

    Uses clinical_precedence timeseries from association_by_datasource_direct,
    bridged to drugs via drug_mechanism_of_action target mapping.
    Supplements with Drugs@FDA first approval year where available.
    """
    ev_path = PROCESSED_DIR / "drug_disease_evidence_years.parquet"

    if ev_path.exists():
        ev_years = pd.read_parquet(ev_path)
        print(f"\nLoaded pre-computed evidence years: {len(ev_years):,} pairs")
    else:
        # Compute from scratch (same logic as the diagnostic we just ran)
        print("\nComputing evidence years from clinical_precedence timeseries...")
        raw = RAW_DIR
        dt = pd.read_parquet(PROCESSED_DIR / "drug_target_map.parquet")

        cp_years = con.execute(f"""
            SELECT
                targetId as target_id,
                diseaseId as disease_id,
                list_min([ts.year FOR ts IN timeseries
                          IF ts.year IS NOT NULL AND ts.associationScore > 0])
                    as first_evidence_year
            FROM read_parquet('{raw}/association_by_datasource_direct/*.parquet')
            WHERE aggregationValue = 'clinical_precedence'
        """).df()

        con.register("dt_df", dt)
        con.register("cp_df", cp_years)

        ev_years = con.execute("""
            SELECT
                dt_df.drug_id,
                cp_df.disease_id,
                MIN(cp_df.first_evidence_year) as first_evidence_year
            FROM dt_df
            JOIN cp_df ON dt_df.target_id = cp_df.target_id
            GROUP BY dt_df.drug_id, cp_df.disease_id
        """).df()

        ev_years.to_parquet(ev_path, index=False)
        print(f"  Saved: {ev_path}")

    # Supplement with Drugs@FDA first approval year
    fda_path = RAW_DIR / "drugsfda"
    if fda_path.exists():
        ev_years = supplement_with_fda(ev_years, fda_path)

    # Summary
    yr = ev_years["first_evidence_year"].dropna()
    print(f"\nEvidence year distribution:")
    print(f"  ≤ 2019:     {(yr <= 2019).sum():,}")
    print(f"  2020–2022:  {((yr >= 2020) & (yr <= 2022)).sum():,}")
    print(f"  2023–2025:  {((yr >= 2023) & (yr <= 2025)).sum():,}")
    print(f"  2026+:      {(yr >= 2026).sum():,}")
    print(f"  No year:    {ev_years['first_evidence_year'].isna().sum():,}")

    return ev_years


def supplement_with_fda(ev_years: pd.DataFrame, fda_path: Path) -> pd.DataFrame:
    """
    Add Drugs@FDA first approval year as a secondary temporal signal.

    Maps FDA drug names to ChEMBL IDs via drug_molecule name matching,
    then uses the earliest approval date from Drugs@FDA submissions.
    """
    try:
        products = pd.read_csv(fda_path / "Products.txt", sep="\t", encoding="latin-1")
        submissions = pd.read_csv(fda_path / "Submissions.txt", sep="\t", encoding="latin-1")
    except Exception as e:
        print(f"  Drugs@FDA files not readable: {e}")
        return ev_years

    # Get first approval year per application
    approved = submissions[submissions["SubmissionStatus"] == "AP"].copy()
    approved["year"] = pd.to_datetime(
        approved["SubmissionStatusDate"], errors="coerce"
    ).dt.year
    first_approval = approved.groupby("ApplNo")["year"].min().reset_index()
    first_approval.columns = ["ApplNo", "fda_first_year"]

    # Join to products to get drug names
    products_with_year = products.merge(first_approval, on="ApplNo", how="inner")

    # Build ingredient → earliest FDA year
    fda_drug_years = (
        products_with_year.groupby("ActiveIngredient")["fda_first_year"]
        .min()
        .reset_index()
    )
    fda_drug_years["ingredient_lower"] = fda_drug_years["ActiveIngredient"].str.lower().str.strip()

    # Load drug_molecule names for matching
    mol_path = RAW_DIR / "drug_molecule"
    drugs = con.execute(f"""
        SELECT id AS drug_id, LOWER(TRIM(name)) AS drug_name_lower
        FROM read_parquet('{mol_path}/*.parquet')
        WHERE name IS NOT NULL
    """).df()

    # Simple exact match on lowercased name
    matched = drugs.merge(
        fda_drug_years[["ingredient_lower", "fda_first_year"]],
        left_on="drug_name_lower",
        right_on="ingredient_lower",
        how="inner",
    )[["drug_id", "fda_first_year"]]

    print(f"  FDA name matches: {len(matched):,} drugs")

    if len(matched) > 0:
        # Merge FDA year into evidence years — take earliest of OT and FDA
        ev_years = ev_years.merge(matched, on="drug_id", how="left")
        has_both = ev_years["fda_first_year"].notna() & ev_years["first_evidence_year"].notna()
        has_fda_only = ev_years["fda_first_year"].notna() & ev_years["first_evidence_year"].isna()

        ev_years.loc[has_both, "first_evidence_year"] = np.minimum(
            ev_years.loc[has_both, "first_evidence_year"],
            ev_years.loc[has_both, "fda_first_year"],
        )
        ev_years.loc[has_fda_only, "first_evidence_year"] = ev_years.loc[
            has_fda_only, "fda_first_year"
        ]
        ev_years = ev_years.drop(columns=["fda_first_year"])

        print(f"  Updated with FDA data: {has_both.sum():,} refined, "
              f"{has_fda_only.sum():,} newly dated")

    return ev_years


# ── Step 3: Build labelled dataset ────────────────────────────────────────

def build_labelled_dataset(
    features: pd.DataFrame,
    indications: pd.DataFrame,
    evidence_years: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge features with indication labels and evidence years for temporal splits.

    Label:
      1 = drug-disease pair has a known indication (Phase ≥ 3)
      0 = pair has OT evidence but no known indication

    Split (for positives, based on evidence year):
      'train'       = evidence_year ≤ 2019
      'calibration' = evidence_year 2020–2022
      'test'        = evidence_year 2023–2025
      'exclude'     = no evidence year and Phase < 4; or year > 2025

    Split (for negatives):
      Sampled into train/calibration/test proportionally
    """
    # Positive pairs: Phase ≥ 3 (APPROVAL or PHASE_3)
    pos = indications[indications["max_phase"] >= 3][
        ["drug_id", "disease_id", "max_phase", "clinical_stage"]
    ].copy()
    pos = pos.sort_values("max_phase", ascending=False)
    pos = pos.drop_duplicates(subset=["drug_id", "disease_id"], keep="first")

    # Merge evidence years into positive pairs
    pos = pos.merge(evidence_years, on=["drug_id", "disease_id"], how="left")

    print(f"\nPositive pairs (Phase ≥ 3): {len(pos):,}")
    print(f"  With evidence year: {pos['first_evidence_year'].notna().sum():,}")
    print(f"  Without evidence year: {pos['first_evidence_year'].isna().sum():,}")

    # For positives without evidence year: APPROVAL → train, PHASE_3 → exclude
    no_year = pos["first_evidence_year"].isna()
    is_approval = pos["max_phase"] == 4
    pos.loc[no_year & is_approval, "first_evidence_year"] = TRAIN_END  # conservative
    # PHASE_3 without year stays NaN → will be excluded

    # Merge into features
    labelled = features.merge(
        pos[["drug_id", "disease_id", "max_phase", "clinical_stage", "first_evidence_year"]],
        on=["drug_id", "disease_id"],
        how="left",
    )

    labelled["label"] = labelled["max_phase"].notna().astype(int)
    labelled["max_phase"] = labelled["max_phase"].fillna(0)

    # Assign splits for positives
    def assign_positive_split(year):
        if pd.isna(year):
            return "exclude"
        year = int(year)
        if year <= TRAIN_END:
            return "train"
        elif CAL_START <= year <= CAL_END:
            return "calibration"
        elif TEST_START <= year <= TEST_END:
            return "test"
        else:
            return "exclude"

    pos_mask = labelled["label"] == 1
    labelled.loc[pos_mask, "split"] = labelled.loc[pos_mask, "first_evidence_year"].apply(
        assign_positive_split
    )

    # Count positives per split
    for s in ["train", "calibration", "test", "exclude"]:
        n = ((labelled["split"] == s) & pos_mask).sum()
        print(f"  Positives in {s}: {n:,}")

    # Assign splits for negatives — proportional sampling
    neg_mask = labelled["label"] == 0
    neg_indices = labelled[neg_mask].index.tolist()
    n_neg = len(neg_indices)

    n_cal_pos = ((labelled["split"] == "calibration") & pos_mask).sum()
    n_test_pos = ((labelled["split"] == "test") & pos_mask).sum()
    n_train_pos = ((labelled["split"] == "train") & pos_mask).sum()

    rng = np.random.RandomState(42)

    # Negative:positive ratio of 5:1 for cal and test; rest goes to train
    n_cal_neg = min(n_cal_pos * 5, n_neg // 4) if n_cal_pos > 0 else 0
    n_test_neg = min(n_test_pos * 5, n_neg // 4) if n_test_pos > 0 else 0

    if n_cal_neg + n_test_neg > 0:
        sampled = rng.choice(neg_indices, size=n_cal_neg + n_test_neg, replace=False)
        cal_neg_idx = sampled[:n_cal_neg]
        test_neg_idx = sampled[n_cal_neg:]

        labelled.loc[cal_neg_idx, "split"] = "calibration"
        labelled.loc[test_neg_idx, "split"] = "test"

    # All remaining negatives → train
    labelled.loc[neg_mask & labelled["split"].isna(), "split"] = "train"

    return labelled


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Script 03: Temporal labelling and split assignment")
    print("=" * 70)

    # Load features
    features_path = PROCESSED_DIR / "drug_disease_features.parquet"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run Script 02 first.")
        return
    features = pd.read_parquet(features_path)
    print(f"Loaded features: {features.shape}")

    # Extract indications
    indications = extract_indications()

    # Build temporal evidence years
    evidence_years = build_evidence_years()

    # Build labelled dataset
    labelled = build_labelled_dataset(features, indications, evidence_years)

    # Remove excluded rows
    n_excluded = (labelled["split"] == "exclude").sum()
    labelled = labelled[labelled["split"] != "exclude"].reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"LABELLED DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total pairs (after exclusions): {len(labelled):,}")
    print(f"Excluded: {n_excluded:,}")

    split_summary = {}
    for split in ["train", "calibration", "test"]:
        subset = labelled[labelled["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
        print(f"\n  {split}:")
        print(f"    Positive: {n_pos:,}")
        print(f"    Negative: {n_neg:,}")
        print(f"    Ratio:    {ratio:.1f}:1")
        split_summary[split] = {"positive": int(n_pos), "negative": int(n_neg)}

    if split_summary["train"]["positive"] > 0:
        train_prevalence = split_summary["train"]["positive"] / (
            split_summary["train"]["positive"] + split_summary["train"]["negative"]
        )
        print(f"\nTrain prevalence: {train_prevalence:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "drug_disease_labelled.parquet"
    labelled.to_parquet(out_path, index=False)
    print(f"\nOutput: {out_path}")
    print(f"Shape: {labelled.shape}")
    print(f"Size: {out_path.stat().st_size / 1e6:.1f} MB")

    summary_path = PROCESSED_DIR / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(split_summary, f, indent=2)
    print(f"Split summary: {summary_path}")


if __name__ == "__main__":
    main()
