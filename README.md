# formula2smiles - Molecular Formula to SMILES Environment

OpenReward (ORS) environment for molecular generation: given a molecular
formula (Hill notation) and optional functional group constraints, produce
a valid SMILES string.

## Task

The agent receives a molecular formula (e.g., C12H21NO2) and optionally
a set of required functional groups (e.g., hydroxyl, amide). It must submit
a valid SMILES string that:
1. Parses as a valid molecule (RDKit)
2. Has the correct molecular formula
3. Contains all required functional groups (if specified)

## Dataset

- **Source**: ZINC20 (via sagawa/ZINC-canonicalized on HuggingFace)
- **Tasks**: 1100 total (1000 train, 100 test)
- **Task types**: ~60% functional-group constrained, ~40% formula-only
- **Formulas**: 856 unique molecular formulas

## Verification

Following the [ether0](https://github.com/Future-House/ether0) approach:
- SMILES parsed and sanitized with RDKit
- Formula verified via `CalcMolFormula` (Hill notation exact match)
- Functional groups verified via `exmol.get_functional_groups`
- Binary reward: 1.0 if all constraints satisfied, 0.0 otherwise

## Structure

- `server.py` - Environment implementation
- `generate_dataset.py` - Dataset generation script
- `test_agent.py` - OpenAI test agent
- `requirements.txt` - Dependencies
- `Dockerfile` - Docker configuration
- `DATA_UPLOAD.md` - Data upload instructions

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data (requires: pip install datasets)
python generate_dataset.py

# Run server
python server.py

# Test with agent
export OPENAI_API_KEY=your_key
python test_agent.py
```

## Docker

```bash
docker build -t formula2smiles:test .
docker run -v $(pwd):/orwd_data -p 8080:8080 formula2smiles:test
```

## Deployment

Deployed at EnvCommons/formula2smiles on OpenReward.
