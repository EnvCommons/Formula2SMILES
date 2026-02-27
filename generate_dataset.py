"""
generate_dataset.py - Generate molecular formula -> SMILES tasks from ZINC.

Streams molecules from sagawa/ZINC-canonicalized on HuggingFace, computes
Hill formulas and functional groups using RDKit + exmol, and produces
tasks.parquet.

Dependencies: datasets, rdkit, exmol>=3.3.0, pandas, pyarrow
"""
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from exmol import get_functional_groups

OUTPUT_PATH = Path(__file__).parent / "tasks.parquet"

TARGET_TOTAL = 1100
TARGET_FG = 660  # ~60% functional-group tasks
TARGET_FORMULA_ONLY = 440  # ~40% formula-only tasks
MAX_FORMULA_REPEATS = 3

FORMULA_ONLY_PROMPTS = [
    "Generate a valid SMILES string for a molecule with the molecular formula {formula}.",
    "What is a valid SMILES representation of a compound with formula {formula}?",
    "Propose a SMILES string for a molecule that has the molecular formula {formula}.",
]

FUNCTIONAL_GROUP_PROMPTS = [
    "Generate a SMILES representation for a molecule containing groups: {groups}. It should also have formula {formula}.",
    "Propose a SMILES structure for a molecule with molecular formula {formula} that contains these functional groups: {groups}.",
    "What is a valid SMILES for a compound with formula {formula} containing the following functional groups: {groups}?",
]


def is_suitable_molecule(smiles: str) -> bool:
    if "." in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    mw = Descriptors.ExactMolWt(mol)
    if mw < 100 or mw > 600:
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 5 or n_heavy > 50:
        return False
    # Skip charged species
    total_charge = Chem.GetFormalCharge(mol)
    if total_charge != 0:
        return False
    return True


def format_groups_list(groups: list[str]) -> str:
    if len(groups) == 1:
        return groups[0]
    elif len(groups) == 2:
        return f"{groups[0]} and {groups[1]}"
    else:
        return ", ".join(groups[:-1]) + f", and {groups[-1]}"


def main():
    random.seed(42)
    print("Streaming from sagawa/ZINC-canonicalized...")
    ds = load_dataset("sagawa/ZINC-canonicalized", split="train", streaming=True)

    fg_candidates = []  # molecules with 2+ functional groups
    formula_candidates = []  # all suitable molecules
    formula_counts: dict[str, int] = {}
    seen_smiles: set[str] = set()

    processed = 0
    for example in ds:
        smiles = example.get("smiles", "")
        if not smiles or not is_suitable_molecule(smiles):
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed}, collected {len(fg_candidates)} FG + {len(formula_candidates)} formula-only candidates")
            continue

        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen_smiles:
            processed += 1
            continue
        seen_smiles.add(canonical)

        formula = rdMolDescriptors.CalcMolFormula(mol)

        # Enforce formula diversity
        if formula_counts.get(formula, 0) >= MAX_FORMULA_REPEATS:
            processed += 1
            continue
        formula_counts[formula] = formula_counts.get(formula, 0) + 1

        groups = get_functional_groups(mol, return_all=True)
        groups_list = sorted(groups) if groups else []

        entry = {
            "smiles": canonical,
            "formula": formula,
            "groups": groups_list,
        }

        if len(groups_list) >= 2 and len(fg_candidates) < TARGET_FG * 2:
            fg_candidates.append(entry)
        if len(formula_candidates) < TARGET_FORMULA_ONLY * 2:
            formula_candidates.append(entry)

        processed += 1
        if processed % 10000 == 0:
            print(f"  Processed {processed}, collected {len(fg_candidates)} FG + {len(formula_candidates)} formula-only candidates")

        if len(fg_candidates) >= TARGET_FG * 2 and len(formula_candidates) >= TARGET_FORMULA_ONLY * 2:
            break

    print(f"\nTotal processed: {processed}")
    print(f"FG candidates: {len(fg_candidates)}")
    print(f"Formula-only candidates: {len(formula_candidates)}")

    # Sample final sets
    random.shuffle(fg_candidates)
    random.shuffle(formula_candidates)
    fg_selected = fg_candidates[:TARGET_FG]
    formula_selected = formula_candidates[:TARGET_FORMULA_ONLY]

    # Build tasks
    tasks = []
    task_idx = 0

    for entry in fg_selected:
        groups_list = entry["groups"]
        # Select 1-3 groups as constraints
        n_groups = min(len(groups_list), random.randint(1, 3))
        selected_groups = sorted(random.sample(groups_list, n_groups))
        groups_str = format_groups_list(selected_groups)
        prompt = random.choice(FUNCTIONAL_GROUP_PROMPTS).format(
            formula=entry["formula"], groups=groups_str
        )
        tasks.append({
            "id": f"f2s_{task_idx:04d}",
            "formula": entry["formula"],
            "functional_groups": json.dumps(selected_groups),
            "task_type": "functional-group",
            "reference_smiles": entry["smiles"],
            "prompt": prompt,
        })
        task_idx += 1

    for entry in formula_selected:
        prompt = random.choice(FORMULA_ONLY_PROMPTS).format(formula=entry["formula"])
        tasks.append({
            "id": f"f2s_{task_idx:04d}",
            "formula": entry["formula"],
            "functional_groups": json.dumps([]),
            "task_type": "formula-only",
            "reference_smiles": entry["smiles"],
            "prompt": prompt,
        })
        task_idx += 1

    # Shuffle and assign splits
    random.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["id"] = f"f2s_{i:04d}"
        t["split"] = "train" if i < 1000 else "test"

    df = pd.DataFrame(tasks)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} tasks to {OUTPUT_PATH}")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Test:  {len(df[df['split'] == 'test'])}")
    print(f"  Functional-group tasks: {len(df[df['task_type'] == 'functional-group'])}")
    print(f"  Formula-only tasks: {len(df[df['task_type'] == 'formula-only'])}")
    print(f"  Unique formulas: {df['formula'].nunique()}")

    # Print a few examples
    print("\n--- Sample tasks ---")
    for _, row in df.head(5).iterrows():
        print(f"  [{row['task_type']}] {row['prompt'][:100]}...")


if __name__ == "__main__":
    main()
