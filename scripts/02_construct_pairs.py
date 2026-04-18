#!/usr/bin/env python3
"""
Script 02: Construct drug-disease pairs with Open Targets evidence features.

Pipeline:
  1. Discover schemas of all downloaded Parquet datasets
  2. Build drug → target mapping via drug_mechanism_of_action
  3. Pivot association_by_datasource_direct to get per-datasource scores
  4. Aggregate target-level scores to drug-disease level (max across targets)
  5. Enrich with disease metadata (therapeutic area) and drug metadata
  6. Output: feature matrix of (drug_id, disease_id, [20 datasource scores], metadata)

Uses DuckDB for efficient out-of-core Parquet processing.
"""

import duckdb
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()


# ── Step 0: Schema discovery ──────────────────────────────────────────────

def discover_schemas():
    """Print schema of each downloaded dataset for inspection."""
    datasets = [d for d in RAW_DIR.iterdir() if d.is_dir() and list(d.glob("*.parquet"))]
    print("=" * 70)
    print("SCHEMA DISCOVERY")
    print("=" * 70)

    schemas = {}
    for ds_path in sorted(datasets):
        name = ds_path.name
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{ds_path}/*.parquet') LIMIT 0"
            ).df()
            cols = list(df.columns)
            dtypes = {c: str(df[c].dtype) for c in cols}
            schemas[name] = cols
            print(f"\n{name} ({len(cols)} columns):")
            for c in cols:
                print(f"  {c}: {dtypes[c]}")
        except Exception as e:
            print(f"\n{name}: ERROR — {e}")

    return schemas


# ── Step 1: Drug → target mapping ─────────────────────────────────────────

def build_drug_target_map() -> pd.DataFrame:
    """Extract drug-target pairs from drug_mechanism_of_action."""
    moa_path = RAW_DIR / "drug_mechanism_of_action"

    # First inspect what columns are available
    sample = con.execute(
        f"SELECT * FROM read_parquet('{moa_path}/*.parquet') LIMIT 5"
    ).df()
    print(f"\n── drug_mechanism_of_action sample columns: {list(sample.columns)}")

    # The dataset should have chemblIds (drug) and targets (array of target objects)
    # We need to unnest the targets array to get drug-target pairs
    # Try different possible schemas
    cols = list(sample.columns)

    if "chemblIds" in cols and "targets" in cols:
        # chemblIds = array of ChEMBL IDs, targets = array of Ensembl gene IDs (strings)
        # Cross-join unnest to get all (drug, target) pairs
        drug_targets = con.execute(f"""
            SELECT DISTINCT
                drug_id,
                target_id,
                actionType
            FROM (
                SELECT
                    unnest(chemblIds) AS drug_id,
                    targets,
                    actionType
                FROM read_parquet('{moa_path}/*.parquet')
            ), LATERAL unnest(targets) AS t(target_id)
            WHERE target_id IS NOT NULL
        """).df()
    else:
        print("Unexpected schema for drug_mechanism_of_action:")
        print(f"Columns: {cols}")
        print(sample.to_string())
        raise ValueError("Cannot parse drug_mechanism_of_action schema")

    print(f"Drug-target pairs: {len(drug_targets):,}")
    print(f"Unique drugs: {drug_targets['drug_id'].nunique():,}")
    print(f"Unique targets: {drug_targets['target_id'].nunique():,}")
    return drug_targets


# ── Step 2: Association scores pivoted by datasource ──────────────────────

def build_association_features() -> pd.DataFrame:
    """
    Pivot association_by_datasource_direct into wide format:
    (targetId, diseaseId) → score_datasource1, score_datasource2, ...
    """
    assoc_path = RAW_DIR / "association_by_datasource_direct"

    # Discover datasource IDs
    # Schema: aggregationType='datasourceId', aggregationValue=<datasource name>,
    #         associationScore=<score>
    datasources = con.execute(f"""
        SELECT DISTINCT aggregationValue AS datasource
        FROM read_parquet('{assoc_path}/*.parquet')
        WHERE aggregationType = 'datasourceId'
        ORDER BY datasource
    """).df()["datasource"].tolist()

    print(f"\n── {len(datasources)} datasources found:")
    for ds in datasources:
        print(f"  {ds}")

    # Pivot: one row per (target, disease), one column per datasource score
    pivot_cols = ", ".join(
        f"MAX(CASE WHEN aggregationValue = '{ds}' THEN associationScore ELSE 0 END) AS \"{ds}\""
        for ds in datasources
    )

    features = con.execute(f"""
        SELECT
            targetId AS target_id,
            diseaseId AS disease_id,
            {pivot_cols}
        FROM read_parquet('{assoc_path}/*.parquet')
        WHERE aggregationType = 'datasourceId'
        GROUP BY targetId, diseaseId
    """).df()

    print(f"Target-disease feature rows: {len(features):,}")
    return features, datasources


# ── Step 3: Aggregate to drug-disease level ───────────────────────────────

def aggregate_to_drug_disease(
    drug_targets: pd.DataFrame,
    target_features: pd.DataFrame,
    datasources: list,
) -> pd.DataFrame:
    """
    Join drug→target with target→disease features.
    Aggregate to drug-disease level using MAX across a drug's targets.
    """
    # Register DataFrames in DuckDB for efficient join
    con.register("drug_targets", drug_targets)
    con.register("target_features", target_features)

    score_cols = ", ".join(f'MAX(tf."{ds}") AS "{ds}"' for ds in datasources)

    drug_disease = con.execute(f"""
        SELECT
            dt.drug_id,
            tf.disease_id,
            {score_cols},
            COUNT(DISTINCT dt.target_id) AS n_targets
        FROM drug_targets dt
        JOIN target_features tf ON dt.target_id = tf.target_id
        GROUP BY dt.drug_id, tf.disease_id
    """).df()

    print(f"\nDrug-disease pairs: {len(drug_disease):,}")
    print(f"Unique drugs: {drug_disease['drug_id'].nunique():,}")
    print(f"Unique diseases: {drug_disease['disease_id'].nunique():,}")
    return drug_disease


# ── Step 4: Enrich with disease metadata ──────────────────────────────────

def enrich_disease_metadata(drug_disease: pd.DataFrame) -> pd.DataFrame:
    """Add therapeutic area and disease name from disease dataset."""
    disease_path = RAW_DIR / "disease"

    # Inspect schema
    sample = con.execute(
        f"SELECT * FROM read_parquet('{disease_path}/*.parquet') LIMIT 3"
    ).df()
    print(f"\n── disease columns: {list(sample.columns)}")

    # Extract disease id, name, and therapeutic areas
    # therapeuticAreas is typically an array of disease IDs at the top level
    diseases = con.execute(f"""
        SELECT
            id AS disease_id,
            name AS disease_name,
            therapeuticAreas
        FROM read_parquet('{disease_path}/*.parquet')
    """).df()

    # Map therapeutic areas to a primary TA (first in list, or 'other')
    def primary_ta(ta_list):
        if ta_list is None or len(ta_list) == 0:
            return "other"
        return ta_list[0]

    diseases["primary_therapeutic_area"] = diseases["therapeuticAreas"].apply(primary_ta)

    # We also need TA names — look them up from the disease table itself
    ta_ids = set()
    for ta_list in diseases["therapeuticAreas"].dropna():
        ta_ids.update(ta_list)

    ta_names = con.execute(f"""
        SELECT id, name
        FROM read_parquet('{disease_path}/*.parquet')
        WHERE id IN ({','.join(f"'{t}'" for t in ta_ids)})
    """).df().set_index("id")["name"].to_dict() if ta_ids else {}

    diseases["therapeutic_area_name"] = diseases["primary_therapeutic_area"].map(
        lambda x: ta_names.get(x, x)
    )

    # Merge
    drug_disease = drug_disease.merge(
        diseases[["disease_id", "disease_name", "primary_therapeutic_area", "therapeutic_area_name"]],
        on="disease_id",
        how="left",
    )

    print(f"Therapeutic area distribution:")
    print(drug_disease["therapeutic_area_name"].value_counts().head(10).to_string())

    return drug_disease


# ── Step 5: Enrich with drug metadata ─────────────────────────────────────

def enrich_drug_metadata(drug_disease: pd.DataFrame) -> pd.DataFrame:
    """Add drug name, type, max clinical phase from drug_molecule."""
    mol_path = RAW_DIR / "drug_molecule"

    sample = con.execute(
        f"SELECT * FROM read_parquet('{mol_path}/*.parquet') LIMIT 3"
    ).df()
    print(f"\n── drug_molecule columns: {list(sample.columns)}")

    drugs = con.execute(f"""
        SELECT
            id AS drug_id,
            name AS drug_name,
            drugType AS drug_type,
            maximumClinicalStage AS max_clinical_stage
        FROM read_parquet('{mol_path}/*.parquet')
    """).df()

    drug_disease = drug_disease.merge(drugs, on="drug_id", how="left")

    print(f"Drug type distribution:")
    print(drug_disease["drug_type"].value_counts().to_string())

    return drug_disease


# ── Step 6: Add overall association score ─────────────────────────────────

def add_overall_score(drug_disease: pd.DataFrame, drug_targets: pd.DataFrame) -> pd.DataFrame:
    """Add the overall direct association score (aggregated across all sources)."""
    overall_path = RAW_DIR / "association_overall_direct"

    con.register("drug_targets_2", drug_targets)

    overall = con.execute(f"""
        SELECT
            dt.drug_id,
            ov.diseaseId AS disease_id,
            MAX(ov.associationScore) AS overall_score
        FROM read_parquet('{overall_path}/*.parquet') ov
        JOIN drug_targets_2 dt ON ov.targetId = dt.target_id
        GROUP BY dt.drug_id, ov.diseaseId
    """).df()

    drug_disease = drug_disease.merge(overall, on=["drug_id", "disease_id"], how="left")
    drug_disease["overall_score"] = drug_disease["overall_score"].fillna(0)

    return drug_disease


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Script 02: Construct drug-disease pairs with OT evidence features")
    print("=" * 70)

    # Schema discovery (informational)
    schemas = discover_schemas()

    # Build pipeline
    drug_targets = build_drug_target_map()
    target_features, datasources = build_association_features()
    drug_disease = aggregate_to_drug_disease(drug_targets, target_features, datasources)
    drug_disease = enrich_disease_metadata(drug_disease)
    drug_disease = enrich_drug_metadata(drug_disease)
    drug_disease = add_overall_score(drug_disease, drug_targets)

    # ── Save outputs ──────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "drug_disease_features.parquet"
    drug_disease.to_parquet(out_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"Output: {out_path}")
    print(f"Shape: {drug_disease.shape}")
    print(f"Columns: {list(drug_disease.columns)}")
    print(f"Size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Also save datasource list for reference
    ds_path = PROCESSED_DIR / "datasource_list.json"
    with open(ds_path, "w") as f:
        json.dump(datasources, f, indent=2)
    print(f"Datasource list: {ds_path}")

    # Save drug-target mapping for later use
    dt_path = PROCESSED_DIR / "drug_target_map.parquet"
    drug_targets.to_parquet(dt_path, index=False)
    print(f"Drug-target map: {dt_path}")

    # Summary stats
    print(f"\n── Summary ─────────────────────────────────────")
    print(f"Total drug-disease pairs: {len(drug_disease):,}")
    print(f"Unique drugs: {drug_disease['drug_id'].nunique():,}")
    print(f"Unique diseases: {drug_disease['disease_id'].nunique():,}")
    print(f"Feature columns (datasources): {len(datasources)}")
    print(f"Non-zero feature density: {(drug_disease[datasources] > 0).mean().mean():.3f}")


if __name__ == "__main__":
    main()
