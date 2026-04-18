#!/usr/bin/env python3
"""
Script 02b: Enrich drug-disease feature matrix with structural features.

Adds three feature groups to the existing drug_disease_features.parquet:
  1. Disease ontology similarity — Jaccard on EFO ancestor sets
  2. Target network features — STRING interaction degree/proximity
  3. Drug chemical fingerprint similarity — Morgan FP Tanimoto via RDKit

Run AFTER 02_construct_pairs.py, BEFORE 03_temporal_labels.py.
Requires Python 3.12 (RDKit not available on 3.14): python3.12 scripts/02b_enrichment_features.py
"""

import json
import time
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from collections import defaultdict
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

con = duckdb.connect()


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ═══════════════════════════════════════════════════════════════════════════
# 1. DISEASE ONTOLOGY SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def build_disease_sparse_matrix():
    """
    Encode disease ancestor sets as a sparse binary matrix.
    Returns (csr_matrix, disease_id_to_row, row_to_disease_id).
    """
    diseases = con.execute(f"""
        SELECT id, ancestors
        FROM read_parquet('{RAW_DIR}/disease/*.parquet')
    """).df()

    # Collect all unique ontology terms
    term_set = set()
    disease_ancestors = {}
    for _, row in diseases.iterrows():
        anc = row["ancestors"]
        terms = set(anc) | {row["id"]} if anc is not None and len(anc) > 0 else {row["id"]}
        disease_ancestors[row["id"]] = terms
        term_set |= terms

    term_to_col = {t: i for i, t in enumerate(sorted(term_set))}
    disease_ids = sorted(disease_ancestors.keys())
    disease_to_row = {d: i for i, d in enumerate(disease_ids)}

    # Build sparse matrix
    rows, cols = [], []
    for d, terms in disease_ancestors.items():
        r = disease_to_row[d]
        for t in terms:
            rows.append(r)
            cols.append(term_to_col[t])

    mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(disease_ids), len(term_set)),
    )
    return mat, disease_to_row, disease_ids


def compute_disease_similarity_features(drug_disease, drug_targets_df):
    """
    For each (drug, disease) pair, compute ontology similarity between the
    query disease and all other diseases associated with the drug's targets.
    Uses sparse matrix multiplication for vectorised Jaccard.

    Features:
      - disease_ontology_max_sim: max Jaccard to any disease in drug's evidence base
      - disease_ontology_mean_sim: mean Jaccard across drug's evidence diseases
      - disease_ontology_n_related: count of diseases with Jaccard > 0.1
    """
    t = time.time()
    print("  Building sparse ancestor matrix...")
    anc_mat, disease_to_row, disease_id_list = build_disease_sparse_matrix()
    # Pre-compute set sizes per disease (row sums)
    set_sizes = np.asarray(anc_mat.sum(axis=1)).ravel()  # (D,)
    print(f"    {anc_mat.shape[0]:,} diseases × {anc_mat.shape[1]:,} terms, "
          f"{anc_mat.nnz:,} non-zeros")

    # Build drug → set of evidence diseases
    print("  Building drug → disease evidence map...")
    drug_diseases_evidence = con.execute(f"""
        SELECT DISTINCT dt.drug_id, tf.disease_id
        FROM (SELECT * FROM drug_targets_df) dt
        JOIN (
            SELECT DISTINCT targetId AS target_id, diseaseId AS disease_id
            FROM read_parquet('{RAW_DIR}/association_by_datasource_direct/*.parquet')
        ) tf ON dt.target_id = tf.target_id
    """).df()

    drug_evidence_diseases = defaultdict(list)
    for drug_id, disease_id in zip(drug_diseases_evidence["drug_id"].values,
                                    drug_diseases_evidence["disease_id"].values):
        row = disease_to_row.get(disease_id)
        if row is not None:
            drug_evidence_diseases[drug_id].append(row)
    print(f"    {len(drug_evidence_diseases):,} drugs with evidence disease sets")

    # Compute per drug group using sparse matrix Jaccard
    print("  Computing Jaccard similarities (sparse matrix, grouped by drug)...")
    n = len(drug_disease)
    max_sims = np.zeros(n, dtype=np.float32)
    mean_sims = np.zeros(n, dtype=np.float32)
    n_related = np.zeros(n, dtype=np.int32)

    grouped = drug_disease.groupby("drug_id")
    n_drugs = len(grouped)
    done = 0

    for drug_id, group in grouped:
        ev_rows = drug_evidence_diseases.get(drug_id)
        if not ev_rows:
            done += 1
            continue

        ev_rows = np.array(ev_rows, dtype=int)

        # Get query disease rows for this drug's pairs
        indices = group.index.values
        query_disease_ids = group["disease_id"].values
        query_rows = np.array([disease_to_row.get(d, -1) for d in query_disease_ids])
        valid = query_rows >= 0
        if not valid.any():
            done += 1
            continue

        valid_indices = indices[valid]
        valid_query_rows = query_rows[valid]
        valid_disease_ids = query_disease_ids[valid]

        # Extract sub-matrices: Q × terms, E × terms
        Q_mat = anc_mat[valid_query_rows]  # (Q, V) sparse
        E_mat = anc_mat[ev_rows]           # (E, V) sparse

        # Intersection: Q × E via sparse dot product
        intersection = (Q_mat @ E_mat.T).toarray()  # (Q, E) dense

        # Union: |A| + |B| - |A ∩ B|
        Q_sizes = set_sizes[valid_query_rows][:, None]  # (Q, 1)
        E_sizes = set_sizes[ev_rows][None, :]           # (1, E)
        union = Q_sizes + E_sizes - intersection         # (Q, E)

        # Jaccard
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, intersection / union, 0.0).astype(np.float32)

        # Zero out self-matches (where query disease == evidence disease)
        # Build mask: for each query disease, find if it's in the evidence set
        ev_row_set = set(ev_rows.tolist())
        for j, qr in enumerate(valid_query_rows):
            if qr in ev_row_set:
                ev_idx = np.where(ev_rows == qr)[0]
                jaccard[j, ev_idx] = 0.0

        # Aggregate — mean includes all evidence diseases (incl. Jaccard=0),
        # excluding only the self-match (zeroed out above)
        max_sims[valid_indices] = jaccard.max(axis=1)
        # Count valid evidence diseases per query (total E minus self if present)
        n_evidence = np.full(len(valid_query_rows), len(ev_rows), dtype=np.float32)
        for j, qr in enumerate(valid_query_rows):
            if qr in ev_row_set:
                n_evidence[j] -= 1
        n_evidence = np.where(n_evidence > 0, n_evidence, 1.0)
        mean_sims[valid_indices] = jaccard.sum(axis=1) / n_evidence
        n_related[valid_indices] = (jaccard > 0.1).sum(axis=1)

        done += 1
        if done % 500 == 0 or done == n_drugs:
            elapsed = time.time() - t
            eta = (elapsed / done) * (n_drugs - done)
            print(f"    {done:,}/{n_drugs:,} drugs ({done/n_drugs*100:.1f}%) — "
                  f"elapsed {fmt_elapsed(elapsed)}, ETA {fmt_elapsed(eta)}", flush=True)

    drug_disease["disease_ontology_max_sim"] = max_sims
    drug_disease["disease_ontology_mean_sim"] = mean_sims
    drug_disease["disease_ontology_n_related"] = n_related

    nonzero = (max_sims > 0).sum()
    print(f"  Done ({fmt_elapsed(time.time() - t)}): "
          f"{nonzero:,}/{n:,} pairs have non-zero similarity")
    return drug_disease


# ═══════════════════════════════════════════════════════════════════════════
# 2. TARGET NETWORK FEATURES (STRING)
# ═══════════════════════════════════════════════════════════════════════════

def compute_target_network_features(drug_disease, drug_targets_df):
    """
    From the OT interaction dataset (STRING + IntAct), compute:
      - target_degree_max: max degree of drug's targets in PPI network
      - target_degree_mean: mean degree of drug's targets
      - target_mean_interaction_score: mean edge weight of drug's targets
      - target_disease_network_overlap: fraction of drug target neighbors
        that are also disease-associated targets

    Drug-level features are cached (only 5K drugs). Disease-specific overlap
    is computed via vectorised merge.
    """
    t = time.time()
    print("  Loading interaction network...")

    edges = con.execute(f"""
        SELECT targetA, targetB, scoring
        FROM read_parquet('{RAW_DIR}/interaction/*.parquet')
        WHERE scoring >= 0.4
    """).df()
    print(f"    {len(edges):,} edges (score ≥ 0.4)")

    # Build degree, mean score, neighbor sets
    print("  Building network index...")
    degree = defaultdict(int)
    score_sum = defaultdict(float)
    neighbors = defaultdict(set)

    for a, b, s in zip(edges["targetA"].values, edges["targetB"].values, edges["scoring"].values):
        degree[a] += 1
        degree[b] += 1
        score_sum[a] += s
        score_sum[b] += s
        neighbors[a].add(b)
        neighbors[b].add(a)

    mean_score_map = {tgt: score_sum[tgt] / degree[tgt] for tgt in degree}
    print(f"    {len(degree):,} targets in network")

    # Drug → targets
    drug_target_map = defaultdict(set)
    for drug_id, target_id in zip(drug_targets_df["drug_id"].values, drug_targets_df["target_id"].values):
        drug_target_map[drug_id].add(target_id)

    # Pre-compute drug-level features (invariant across diseases)
    print("  Pre-computing drug-level network features...")
    drug_net_cache = {}
    for drug_id, targets in drug_target_map.items():
        degs = [degree.get(tgt, 0) for tgt in targets]
        scores = [mean_score_map.get(tgt, 0.0) for tgt in targets]
        drug_nbrs = set()
        for tgt in targets:
            drug_nbrs |= neighbors.get(tgt, set())
        drug_net_cache[drug_id] = (
            max(degs) if degs else 0,
            float(np.mean(degs)) if degs else 0.0,
            float(np.mean(scores)) if scores else 0.0,
            drug_nbrs,
        )

    # Disease → genetic target set
    print("  Loading disease → target associations (genetic evidence)...")
    disease_targets = con.execute(f"""
        SELECT DISTINCT diseaseId AS disease_id, targetId AS target_id
        FROM read_parquet('{RAW_DIR}/association_by_datasource_direct/*.parquet')
        WHERE aggregationType = 'datasourceId'
          AND aggregationValue IN ('gwas_credible_sets', 'eva', 'gene_burden',
                                    'genomics_england', 'gene2phenotype', 'clingen')
    """).df()

    disease_target_map = defaultdict(set)
    for d, tgt in zip(disease_targets["disease_id"].values, disease_targets["target_id"].values):
        disease_target_map[d].add(tgt)
    print(f"    {len(disease_target_map):,} diseases with genetic target sets")

    # Assign features — drug-level via map, overlap via groupby
    print("  Assigning features...")
    n = len(drug_disease)
    deg_max = np.zeros(n, dtype=np.float32)
    deg_mean = np.zeros(n, dtype=np.float32)
    mean_int = np.zeros(n, dtype=np.float32)
    net_overlap = np.zeros(n, dtype=np.float32)

    drug_ids = drug_disease["drug_id"].values
    disease_ids = drug_disease["disease_id"].values

    for i in range(n):
        cache = drug_net_cache.get(drug_ids[i])
        if cache is None:
            continue
        deg_max[i] = cache[0]
        deg_mean[i] = cache[1]
        mean_int[i] = cache[2]

        dis_targets = disease_target_map.get(disease_ids[i])
        if dis_targets and cache[3]:
            net_overlap[i] = len(cache[3] & dis_targets) / len(cache[3])

    drug_disease["target_degree_max"] = deg_max
    drug_disease["target_degree_mean"] = deg_mean
    drug_disease["target_mean_interaction_score"] = mean_int
    drug_disease["target_disease_network_overlap"] = net_overlap

    nonzero = (deg_max > 0).sum()
    print(f"  Done ({fmt_elapsed(time.time() - t)}): "
          f"{nonzero:,}/{n:,} pairs have network features")
    return drug_disease


# ═══════════════════════════════════════════════════════════════════════════
# 3. DRUG CHEMICAL FINGERPRINT SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_drug_fingerprint_features(drug_disease, drug_targets_df):
    """
    Compute Morgan fingerprints (radius=2, 2048 bits) for each drug from SMILES.
    Pre-compute full drug-drug Tanimoto matrix (~5K × 5K), then for each
    (drug, disease) pair look up max/mean similarity to candidate drugs.

    Features:
      - drug_fp_sim_max: max Tanimoto to drugs targeting disease-associated targets
      - drug_fp_sim_mean: mean Tanimoto to same
      - drug_has_smiles: binary indicator
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.warning")

    t = time.time()
    print("  Computing Morgan fingerprints...")

    drugs = con.execute(f"""
        SELECT id AS drug_id, canonicalSmiles
        FROM read_parquet('{RAW_DIR}/drug_molecule/*.parquet')
        WHERE canonicalSmiles IS NOT NULL
    """).df()

    fp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_map = {}
    n_failed = 0
    for _, row in drugs.iterrows():
        mol = Chem.MolFromSmiles(row["canonicalSmiles"])
        if mol is not None:
            fp = fp_gen.GetFingerprint(mol)
            fp_map[row["drug_id"]] = fp
        else:
            n_failed += 1
    print(f"    {len(fp_map):,} drugs with valid fingerprints ({n_failed} failed)")

    # Only compute similarities for drugs actually in our feature matrix
    drugs_in_matrix = set(drug_disease["drug_id"].unique())
    relevant_drugs = sorted(drugs_in_matrix & set(fp_map.keys()))
    drug_to_idx = {d: i for i, d in enumerate(relevant_drugs)}
    n_rel = len(relevant_drugs)
    print(f"    {n_rel:,} drugs in feature matrix with fingerprints")

    # Pre-compute full drug-drug Tanimoto matrix using BulkTanimotoSimilarity
    print(f"  Pre-computing {n_rel:,} × {n_rel:,} drug-drug Tanimoto matrix...")
    t_matrix = time.time()
    fp_list = [fp_map[d] for d in relevant_drugs]
    sim_matrix = np.zeros((n_rel, n_rel), dtype=np.float32)
    for i in range(n_rel):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list)
        sim_matrix[i, :] = sims
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_matrix
            eta = (elapsed / (i + 1)) * (n_rel - i - 1)
            print(f"    {i+1:,}/{n_rel:,} rows ({(i+1)/n_rel*100:.1f}%) — "
                  f"elapsed {fmt_elapsed(elapsed)}, ETA {fmt_elapsed(eta)}", flush=True)
    np.fill_diagonal(sim_matrix, 0.0)  # exclude self-similarity
    print(f"    Matrix computed in {fmt_elapsed(time.time() - t_matrix)}")

    # Build disease → set of candidate drug indices
    print("  Building disease → drug map via shared targets...")
    target_drug_map = defaultdict(set)
    for drug_id, target_id in zip(drug_targets_df["drug_id"].values, drug_targets_df["target_id"].values):
        target_drug_map[target_id].add(drug_id)

    disease_targets = con.execute(f"""
        SELECT DISTINCT diseaseId AS disease_id, targetId AS target_id
        FROM read_parquet('{RAW_DIR}/association_by_datasource_direct/*.parquet')
        WHERE aggregationType = 'datasourceId'
    """).df()

    disease_cand_indices = {}
    for d, tgt in zip(disease_targets["disease_id"].values, disease_targets["target_id"].values):
        if d not in disease_cand_indices:
            disease_cand_indices[d] = set()
        for drug in target_drug_map.get(tgt, set()):
            idx = drug_to_idx.get(drug)
            if idx is not None:
                disease_cand_indices[d].add(idx)

    # Convert to numpy arrays for fast indexing
    disease_cand_arrays = {d: np.array(sorted(idxs), dtype=np.int32)
                           for d, idxs in disease_cand_indices.items() if idxs}
    print(f"    {len(disease_cand_arrays):,} diseases with candidate drug indices")

    # Assign features via matrix lookup
    print("  Assigning similarity features...")
    n = len(drug_disease)
    sim_max_out = np.zeros(n, dtype=np.float32)
    sim_mean_out = np.zeros(n, dtype=np.float32)
    has_smiles = np.zeros(n, dtype=np.int8)

    drug_ids = drug_disease["drug_id"].values
    disease_ids = drug_disease["disease_id"].values

    for i in range(n):
        drug_idx = drug_to_idx.get(drug_ids[i])
        if drug_idx is None:
            # Check if drug has SMILES but wasn't in relevant set (shouldn't happen)
            if drug_ids[i] in fp_map:
                has_smiles[i] = 1
            continue
        has_smiles[i] = 1

        cand_arr = disease_cand_arrays.get(disease_ids[i])
        if cand_arr is None:
            continue

        # Look up pre-computed similarities (self already zeroed out)
        sims = sim_matrix[drug_idx, cand_arr]
        # Exclude self (already 0 on diagonal, but mask for mean calculation)
        mask = cand_arr != drug_idx
        sims = sims[mask]
        if len(sims) > 0:
            sim_max_out[i] = sims.max()
            sim_mean_out[i] = sims.mean()

    drug_disease["drug_fp_sim_max"] = sim_max_out
    drug_disease["drug_fp_sim_mean"] = sim_mean_out
    drug_disease["drug_has_smiles"] = has_smiles

    nonzero = (sim_max_out > 0).sum()
    print(f"  Done ({fmt_elapsed(time.time() - t)}): "
          f"{nonzero:,}/{n:,} pairs have fingerprint similarity, "
          f"{has_smiles.sum():,} drugs with SMILES")
    return drug_disease


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("Script 02b: Enrich drug-disease features with structural features")
    print("=" * 70)

    # Load existing feature matrix
    print("\n[0/3] Loading existing data...")
    dd_path = PROCESSED_DIR / "drug_disease_features.parquet"
    drug_disease = pd.read_parquet(dd_path)
    print(f"  Loaded {len(drug_disease):,} drug-disease pairs, "
          f"{len(drug_disease.columns)} columns")

    # Drop any prior enrichment columns (allow safe re-runs)
    enrichment_cols = [
        "disease_ontology_max_sim", "disease_ontology_mean_sim", "disease_ontology_n_related",
        "target_degree_max", "target_degree_mean", "target_mean_interaction_score",
        "target_disease_network_overlap",
        "drug_fp_sim_max", "drug_fp_sim_mean", "drug_has_smiles",
    ]
    existing = [c for c in enrichment_cols if c in drug_disease.columns]
    if existing:
        drug_disease = drug_disease.drop(columns=existing)
        print(f"  Dropped {len(existing)} prior enrichment columns (re-run)")

    # Load drug-target map
    dt_path = PROCESSED_DIR / "drug_target_map.parquet"
    drug_targets = pd.read_parquet(dt_path)
    con.register("drug_targets_df", drug_targets)
    print(f"  Loaded {len(drug_targets):,} drug-target pairs")

    # Feature group 1: Disease ontology similarity
    print(f"\n[1/3] Disease ontology similarity features...")
    drug_disease = compute_disease_similarity_features(drug_disease, drug_targets)

    # Feature group 2: Target network features
    print(f"\n[2/3] Target network features...")
    drug_disease = compute_target_network_features(drug_disease, drug_targets)

    # Feature group 3: Drug fingerprint similarity
    print(f"\n[3/3] Drug chemical fingerprint similarity...")
    drug_disease = compute_drug_fingerprint_features(drug_disease, drug_targets)

    # Save enriched feature matrix (overwrite)
    print(f"\n── Saving enriched feature matrix...")
    drug_disease.to_parquet(dd_path, index=False)
    print(f"  Output: {dd_path}")
    print(f"  Shape: {drug_disease.shape}")
    print(f"  Size: {dd_path.stat().st_size / 1e6:.1f} MB")

    # Report new columns
    print(f"\n  New feature columns ({len(enrichment_cols)}):")
    for c in enrichment_cols:
        if c in drug_disease.columns:
            nonzero = (drug_disease[c] > 0).sum()
            print(f"    {c}: non-zero in {nonzero:,}/{len(drug_disease):,} "
                  f"({nonzero/len(drug_disease)*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"Script 02b complete — total time: {fmt_elapsed(time.time() - t_total)}")


if __name__ == "__main__":
    main()
