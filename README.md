# Formula2SMILES

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Formula2SMILES)

## Description

**Formula2SMILES** is an environment for evaluating agents on molecular generation tasks. Given a molecular formula in Hill notation and optional functional group constraints, the agent must produce a valid SMILES string that matches the formula and satisfies all constraints. Verification uses [RDKit](https://www.rdkit.org/) for formula matching and [exmol](https://github.com/ur-whitelab/exmol) for functional group detection, following the [ether0](https://github.com/Future-House/ether0) approach. The dataset is derived from [ZINC20](https://zinc20.docking.org/) (via [`sagawa/ZINC-canonicalized`](https://huggingface.co/datasets/sagawa/ZINC-canonicalized) on HuggingFace).

## Capabilities

- Generating valid SMILES strings from molecular formulas
- Satisfying functional group constraints during molecular generation
- Understanding molecular structure and Hill notation conventions
- Reasoning about chemical validity (parsing, sanitization, fragment checks)

## Compute Requirements

Formula2SMILES does not require a sandbox. It has minimal compute requirements.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) (following the ZINC-canonicalized dataset license).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks), totaling 1,100 tasks. Each task provides a molecular formula in Hill notation and optionally a set of required functional groups. Approximately 60% of tasks include functional group constraints and 40% are formula-only. The dataset covers 856 unique molecular formulas.

## Reward Structure

This is a sparse, verifiable reward environment with binary scoring. The agent calls `submit_answer` once with a SMILES string. The molecule is validated through a 5-step pipeline:

1. SMILES parsing with RDKit
2. Molecule sanitization
3. Reasonableness checks (single fragment, ring size <= 12)
4. Formula match via `CalcMolFormula` (Hill notation exact match)
5. Functional group check via `exmol` (if constraints specified)

- **Correct** (all checks pass): Reward **1.0**.
- **Incorrect** (any check fails): Reward **0.0**.

We do not use LLM graders for this task.

## Data

Tasks are derived from [ZINC20](https://zinc20.docking.org/) (via `sagawa/ZINC-canonicalized` on HuggingFace), stored as a parquet file. Data files are stored on the OpenReward platform.

## Tools

Agents are given a single tool:

- `submit_answer`: Submit a SMILES string as the answer. The molecule is validated against the required molecular formula and any functional group constraints. Returns whether the answer is correct with a diagnostic message. This tool can only be called once per task.

## Time Horizon

Formula2SMILES is a single-turn environment. The agent receives a molecular formula (with optional constraints) and submits one SMILES string. Each task requires exactly one tool call.

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; Formula2SMILES works out of the box with the OpenReward endpoint without any secrets.

## Safety

Agents in Formula2SMILES are asked to generate molecular representations as SMILES strings. The environment does not present direct safety risks, as agents only provide text answers validated by RDKit with no access to external systems.

However, this is a dual-use domain. Models trained for molecular generation capabilities could potentially be misused for designing harmful compounds in other contexts.

## Citations

```bibtex
@dataset{GRFormula2SMILES,
  author    = {General Reasoning Inc. Team},
  title     = {Formula2SMILES},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/Formula2SMILES}
}
```

```bibtex
@article{irwin2020zinc20,
  title={ZINC20 -- A Free Ultralarge-Scale Chemical Database for Ligand Discovery},
  author={Irwin, John J and Tang, Khanh G and Young, Jennifer and Dandarchuluun, Chinzorig and Wong, Benjamin R and Khurelbaatar, Munkhzul and Moroz, Yurii S and Mayfield, John and Sayle, Roger A},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={12},
  pages={6065--6073},
  year={2020},
  publisher={ACS Publications}
}
```
