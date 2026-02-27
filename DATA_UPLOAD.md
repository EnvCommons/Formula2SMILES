# Data Upload Requirements for formula2smiles

## Overview
The formula2smiles environment requires a prepared task dataset to be uploaded to
OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── tasks.parquet (~70 KB)
```

## File Description

- **tasks.parquet**: Parquet file with 1100 rows and columns:
  - `id` (str): Task identifier, e.g. "pka_0000", "pka_0001", ...
  - `formula` (str): Molecular formula in Hill notation, e.g. "C12H21NO2"
  - `functional_groups` (str): JSON-encoded list, e.g. `'["hydroxyl", "amide"]'` or `'[]'`
  - `task_type` (str): "functional-group" or "formula-only"
  - `reference_smiles` (str): One valid SMILES satisfying constraints (for reference)
  - `prompt` (str): Pre-rendered prompt text
  - `split` (str): "train" (first 1000) or "test" (last 100)

## Data Source

Tasks are generated from the ZINC20 database via HuggingFace (`sagawa/ZINC-canonicalized`).
Molecular formulas computed with RDKit `CalcMolFormula`, functional groups
detected with `exmol.get_functional_groups`.

## Regenerating the Data
```bash
pip install datasets rdkit exmol>=3.3.0 pandas pyarrow
python generate_dataset.py
```

## Upload Instructions
Upload `tasks.parquet` to the root of the OpenReward namespace storage
for EnvCommons/formula2smiles at https://openreward.ai.
