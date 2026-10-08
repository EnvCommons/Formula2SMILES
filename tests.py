"""Formula2SMILES tests. The task data is not in the repo, so a small fixture
table stands in for tasks.parquet."""
import asyncio
import json
from unittest.mock import patch

import pandas as pd
import pytest

FIXTURE = pd.DataFrame([
    {"id": "t0", "formula": "C17H24N2O3", "functional_groups": json.dumps(["alkane", "ether"]),
     "task_type": "functional-group", "reference_smiles": "Cc1ccccc1NC(=O)CN1CCCC[C@@H]1C1OCCO1",
     "prompt": "Propose a SMILES with formula C17H24N2O3 containing: alkane and ether.", "split": "train"},
    {"id": "t1", "formula": "C20H30N2O3", "functional_groups": json.dumps(["ether O", "tertiary carbon"]),
     "task_type": "functional-group", "reference_smiles": "CC(=O)c1ccc(NC(=O)[C@@H](C(C)C)N2C[C@@H](C)O[C@H](C)C2)cc1C",
     "prompt": "Propose a SMILES with formula C20H30N2O3 containing: ether O and tertiary carbon.", "split": "train"},
    {"id": "t2", "formula": "C32H31N3O2S", "functional_groups": json.dumps(["hetero N basic no H"]),
     "task_type": "functional-group", "reference_smiles": "Cc1ccc(NC(=O)CSc2nc3c(c(=O)n2-c2ccccc2)C2(CCCCC2)Cc2ccccc2-3)cc1",
     "prompt": "Propose a SMILES with formula C32H31N3O2S containing: hetero N basic no H.", "split": "train"},
    {"id": "t3", "formula": "C6H6", "functional_groups": json.dumps([]), "task_type": "formula-only",
     "reference_smiles": "c1ccccc1", "prompt": "Propose a SMILES with formula C6H6.", "split": "test"},
])

with patch("pandas.read_parquet", return_value=FIXTURE.copy()):
    import server

TASKS = {t["id"]: t for t in server.all_tasks}

# Answers with the right formula that use the textbook reading of the group
# name rather than exmol's pattern: an aryl methyl ether for "ether", a
# tert-butyl ether for "tertiary carbon", pyridine nitrogens for
# "hetero N basic no H".
TEXTBOOK_READINGS = {
    "t0": "COc1ccc(cc1)OC1=CCCCC1C(=O)NCCNC",
    "t1": "CC(C)(C)Oc1ccccc1C(=O)N(C1CC=CC(O)C1)CCN(C)",
    "t2": "S(=O)(=O)(c1c(N(C)C)c(c2cccnc2)c(C=C)cc1)c3c(c4cccnc4)cccc3C7=CCCCC7",
}


def submit(task_id, smiles):
    env = server.Formula2SMILES(TASKS[task_id])
    return asyncio.run(env.submit_answer(server.SubmitSmilesInput(smiles=smiles)))


@pytest.mark.parametrize("task_id", ["t0", "t1", "t2", "t3"])
def test_reference_scores_one(task_id):
    r = submit(task_id, TASKS[task_id]["reference_smiles"])
    assert r.reward == 1.0 and r.finished


@pytest.mark.parametrize("task_id", sorted(TEXTBOOK_READINGS))
def test_textbook_reading_of_the_label_scores_zero(task_id):
    r = submit(task_id, TEXTBOOK_READINGS[task_id])
    assert r.reward == 0.0 and r.finished and r.metadata["reason"] == "functional_group_mismatch"


def test_prompt_states_the_pattern_of_each_required_group():
    text = server.Formula2SMILES(TASKS["t2"]).get_prompt()[0].text
    assert text.startswith(TASKS["t2"]["prompt"])
    assert "- hetero N basic no H: [nX3H0+0]" in text
    text = server.Formula2SMILES(TASKS["t0"]).get_prompt()[0].text
    assert "- ether: COC" in text and "- alkane: [CX4][CX4]" in text
    assert "tertiary carbon: [CX4H1]([#6])([#6])[#6]" in server.Formula2SMILES(TASKS["t1"]).get_prompt()[0].text


def test_prompt_never_contains_the_reference():
    for t in TASKS.values():
        assert t["reference_smiles"] not in server.Formula2SMILES(t).get_prompt()[0].text


def test_formula_only_prompt_is_unchanged():
    assert server.Formula2SMILES(TASKS["t3"]).get_prompt()[0].text == TASKS["t3"]["prompt"]


def test_every_pattern_matches_what_exmol_reports():
    from exmol import get_functional_groups
    from rdkit import Chem

    for t in TASKS.values():
        mol = Chem.MolFromSmiles(t["reference_smiles"])
        reported = {g.lower() for g in get_functional_groups(mol, return_all=True)}
        for label, smarts in server.GROUP_SMARTS.items():
            assert mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)) == (label in reported), (label, smarts)


def test_invalid_smiles_is_not_graded():
    r = submit("t0", "not a smiles")
    assert r.reward == 0.0 and not r.finished
