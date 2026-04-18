# Calibrated Uncertainty for Drug Repurposing via Conformal Prediction on the Open Targets Knowledge Graph

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19639461.svg)](https://doi.org/10.5281/zenodo.19639461)

Replication code and processed outputs for:

> Farquhar H. Calibrated uncertainty for computational drug repurposing via conformal prediction on the Open Targets knowledge graph. 2026.

## Overview

This repository provides the full analysis pipeline for the first application of conformal prediction to knowledge-graph-based drug repurposing. Using 6.4 million drug-disease pairs from the Open Targets Platform (release 26.03), we demonstrate that split conformal prediction achieves valid coverage guarantees on temporally held-out 2023-2025 drug approvals.

**Key results:**
- XGBoost test AUROC: 0.906 (95% CI: 0.874-0.933)
- Marginal CP coverage at 90% nominal: 91.1% (CI: 88.5-93.5%)
- Mondrian CP maintains group-conditional coverage across 7 therapeutic areas (87.8-100%)
- Exchangeability not rejected (permutation p = 0.514)

## Repository Structure

```
scripts/
  01_download_open_targets.py    # Download OT release 26.03 data
  02_construct_pairs.py          # Drug-disease pair feature engineering
  02b_enrichment_features.py     # Disease ontology, PPI network, chemical features
  03_temporal_labels.py          # Temporal labelling and train/cal/test split
  04_train_xgboost.py            # XGBoost training with hyperparameter tuning
  05_train_baseline.py           # Logistic regression baseline
  06_conformal_calibration.py    # Marginal + Mondrian conformal prediction
  07_case_studies.py             # Novel prediction filtering and case studies
  08_ablation_shap.py            # Feature ablation and sensitivity analyses
  10_supplementary_analyses.py   # Calibration, bootstrap CIs, exchangeability, etc.

notebooks/
  09_graphsage_baseline.ipynb    # GraphSAGE GNN baseline (Colab T4 GPU)

outputs/
  figures/                       # Publication-ready figures
  tables/                        # Analysis output tables (CSV/JSON)
  xgboost_metrics.json           # XGBoost performance metrics
  gnn_metrics.json               # GraphSAGE performance metrics
  conformal_results.json         # Conformal prediction results

data/
  processed/
    datasource_list.json         # Open Targets evidence source names
    split_summary.json           # Train/calibration/test split counts
```

## Requirements

- Python 3.10+ (tested on 3.14)
- Key packages: `pandas`, `pyarrow`, `duckdb`, `scikit-learn`, `xgboost`, `matplotlib`
- Optional (Script 02b): `rdkit` (requires Python 3.12)
- Optional (Notebook 09): `torch`, `torch_geometric` (Colab T4 recommended)

Install dependencies:
```bash
pip install pandas pyarrow duckdb scikit-learn xgboost matplotlib
```

## Reproducing the Analysis

Scripts are designed to run sequentially (01-10). Script 01 downloads ~100 GB of Open Targets data.

```bash
# 1. Download data (requires ~100 GB disk)
python scripts/01_download_open_targets.py

# 2. Feature engineering
python scripts/02_construct_pairs.py
python3.12 scripts/02b_enrichment_features.py  # requires RDKit

# 3. Temporal labelling
python scripts/03_temporal_labels.py

# 4. Model training
python scripts/04_train_xgboost.py
python scripts/05_train_baseline.py

# 5. Conformal prediction
python scripts/06_conformal_calibration.py

# 6. Case studies and ablation
python scripts/07_case_studies.py
python scripts/08_ablation_shap.py

# 7. Supplementary analyses
python scripts/10_supplementary_analyses.py

# 8. GraphSAGE (optional, run on Colab)
# Upload notebooks/09_graphsage_baseline.ipynb to Google Colab with T4 GPU
```

## Data Sources

All input data are publicly available:

| Source | URL | Licence |
|--------|-----|---------|
| Open Targets Platform 26.03 | https://platform.opentargets.org | CC BY 4.0 |
| ChEMBL 34 | https://www.ebi.ac.uk/chembl | CC BY-SA 3.0 |
| STRING v12 | https://string-db.org | CC BY 4.0 |
| Drugs@FDA | https://www.accessdata.fda.gov/scripts/cder/daf/ | Public domain |

Raw data files are not included due to size (~100 GB). Run `scripts/01_download_open_targets.py` to download.

## Citation

If you use this code, please cite:

```bibtex
@misc{farquhar2026conformal,
  title={Calibrated uncertainty for computational drug repurposing via conformal prediction on the Open Targets knowledge graph},
  author={Farquhar, Hayden},
  year={2026}
}
```

## Licence

This code is released under the MIT Licence. See [LICENSE](LICENSE) for details.

## Contact

Hayden Farquhar — hayden.farquhar@icloud.com
