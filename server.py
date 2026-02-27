"""
Formula2SMILES Environment - Molecular Formula (Hill notation) to SMILES.

Agent receives a molecular formula and optional functional group constraints,
and must submit a valid SMILES string. Verification uses RDKit for formula
matching and exmol for functional group detection, following the ether0 approach.

Binary reward: 1.0 if valid SMILES + correct formula + all FG constraints met, else 0.0.
"""
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool


# ---- Data loading ----
if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent

tasks_df = pd.read_parquet(DATA_PATH / "tasks.parquet")
if "__index_level_0__" in tasks_df.columns:
    tasks_df = tasks_df.drop("__index_level_0__", axis=1)
tasks_df = tasks_df.fillna("")
all_tasks = tasks_df.to_dict(orient="records")

for t in all_tasks:
    t["id"] = str(t["id"])

train_tasks = [t for t in all_tasks if t["split"] == "train"]
test_tasks = [t for t in all_tasks if t["split"] == "test"]


# ---- Pydantic models ----
class TaskSpec(BaseModel):
    id: str
    formula: str
    functional_groups: str
    task_type: str
    reference_smiles: str
    prompt: str
    split: str


class SubmitSmilesInput(BaseModel):
    smiles: str = Field(..., description="A valid SMILES string representing the molecule")


# ---- Environment class ----
class Formula2SMILES(Environment):
    """
    Molecular Formula to SMILES environment.
    Given a molecular formula (Hill notation) and optional functional group
    constraints, produce a valid SMILES string that satisfies all constraints.
    """

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.formula = self.config.formula
        self.functional_groups: list[str] = (
            json.loads(self.config.functional_groups)
            if self.config.functional_groups
            else []
        )

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return train_tasks
        elif split == "test":
            return test_tasks
        raise ValueError(f"Unknown split: {split}")

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=self.config.prompt)]

    @tool
    async def submit_answer(self, params: SubmitSmilesInput) -> ToolOutput:
        """
        Submit a SMILES string as your answer. The molecule will be validated
        against the required molecular formula and any functional group
        constraints. This finishes the episode.
        """
        smiles = params.smiles.strip()
        result = self._verify_smiles(smiles)

        reward = 1.0 if result["correct"] else 0.0

        return ToolOutput(
            blocks=[TextBlock(type="text", text=result["message"])],
            metadata={
                "task_id": self.config.id,
                "submitted_smiles": smiles,
                "expected_formula": self.formula,
                "expected_groups": self.functional_groups,
                "task_type": self.config.task_type,
                **result,
            },
            reward=reward,
            finished=True,
        )

    def _verify_smiles(self, smiles: str) -> dict[str, Any]:
        """
        Verification logic following ether0's approach:
        1. Parse SMILES with RDKit
        2. Sanitize molecule
        3. Reasonableness checks (single fragment, ring size)
        4. Formula match via CalcMolFormula (Hill notation)
        5. Functional group check via exmol (if constraints exist)
        """
        # Step 1: Parse
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "correct": False,
                "message": "Invalid SMILES: could not parse.",
                "reason": "parse_failure",
            }

        # Step 2: Sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return {
                "correct": False,
                "message": f"Invalid molecule: sanitization failed ({e}).",
                "reason": "sanitization_failure",
            }

        # Step 3: Reasonableness
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if "." in canonical:
            return {
                "correct": False,
                "message": "Molecule contains multiple disconnected fragments.",
                "reason": "multiple_fragments",
            }

        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) > 12:
                return {
                    "correct": False,
                    "message": f"Molecule contains unreasonably large ring ({len(ring)} atoms).",
                    "reason": "large_ring",
                }

        # Step 4: Formula match
        computed_formula = rdMolDescriptors.CalcMolFormula(mol)
        if computed_formula != self.formula:
            return {
                "correct": False,
                "message": f"Formula mismatch. Expected: {self.formula}, Got: {computed_formula}.",
                "reason": "formula_mismatch",
                "computed_formula": computed_formula,
            }

        # Step 5: Functional group check
        if self.functional_groups:
            from exmol import get_functional_groups

            detected = get_functional_groups(mol, return_all=True)
            detected_lower = {g.lower() for g in detected}
            required_lower = {g.lower() for g in self.functional_groups}
            missing = required_lower - detected_lower

            if missing:
                return {
                    "correct": False,
                    "message": (
                        f"Missing functional groups: {', '.join(sorted(missing))}. "
                        f"Detected groups: {', '.join(sorted(detected_lower))}."
                    ),
                    "reason": "functional_group_mismatch",
                    "detected_groups": sorted(detected_lower),
                    "missing_groups": sorted(missing),
                }

        return {
            "correct": True,
            "message": "Correct! Valid molecule with matching formula"
            + (" and functional groups." if self.functional_groups else "."),
            "reason": "success",
        }


if __name__ == "__main__":
    server = Server([Formula2SMILES])
    server.run()
