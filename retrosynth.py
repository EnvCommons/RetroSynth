"""
RetroSynth - Single-Step Retrosynthesis Environment

Single-turn environment where agents propose reactants that could produce
a target molecule in a single synthetic step.

Data source: USPTO-50k reaction dataset (filtered for commercially plausible reagents).
Continuous reward based on RDKit fingerprint similarity to ground truth reactants,
with exact match yielding 1.0.
"""

import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent


def load_all_tasks() -> dict[str, list[dict]]:
    data_dir = ENV_PATH / "data"
    all_tasks = {}
    for split in ["train", "test"]:
        json_file = data_dir / f"{split}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                all_tasks[split] = json.load(f)
        else:
            print(f"Warning: {json_file} not found")
            all_tasks[split] = []
    return all_tasks


ALL_TASKS = load_all_tasks()

ANSWERS = {
    task["task_id"]: {"reactant_smiles": task["reactant_smiles"]}
    for split_tasks in ALL_TASKS.values()
    for task in split_tasks
}

print(f"Loaded {len(ANSWERS)} RetroSynth tasks")

_UNCHARGER = rdMolStandardize.Uncharger()


def _identity_key(mol) -> str:
    """Canonical SMILES ignoring stereochemistry and charge state."""
    return Chem.MolToSmiles(_UNCHARGER.uncharge(Chem.Mol(mol)), isomericSmiles=False)


class RetroSynthTaskSpec(BaseModel):
    task_id: str
    target_smiles: str
    num_heavy_atoms: int
    num_reactants: int
    question: str


class SubmitReactantsInput(BaseModel, extra="forbid"):
    reactants: str = Field(
        ...,
        description=(
            "Proposed reactant SMILES separated by dots (e.g., 'CCO.CC(=O)Cl'). "
            "These should be the starting materials that would produce the target "
            "molecule in a single synthetic step."
        ),
    )


class RetroSynth(Environment):
    """
    Single-step retrosynthesis environment.

    Agents propose reactants for a target molecule.
    Reward is continuous [0, 1] based on fingerprint similarity
    to ground truth reactants, with exact match yielding 1.0.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = RetroSynthTaskSpec.model_validate(task_spec)

        if self.validated.task_id not in ANSWERS:
            raise ValueError(f"Task {self.validated.task_id} not found in ANSWERS")

        self.answer = ANSWERS[self.validated.task_id]

        # Graded submissions made this session. Only the first is rewarded.
        # Malformed submissions (unparseable SMILES) are NOT counted and do not
        # end the episode: they never reach grading and leak no signal, so a
        # typo should not burn the attempt.
        self.submitted = 0

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            return []
        return [
            {k: v for k, v in task.items() if k != "reactant_smiles"}
            for task in ALL_TASKS[split]
        ]

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.question)]

    @tool
    async def submit_reactants(self, params: SubmitReactantsInput) -> ToolOutput:
        """Submit proposed reactants for the retrosynthesis task."""
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="Reactants have already been submitted for this task. "
                                       "This episode is over: the submission is not re-graded, and repeat submissions are penalised (reward -0.1).")],
                metadata={
                    "task_id": self.validated.task_id,
                    "already_submitted": True,
                    "submission_count": self.submitted,
                },
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        gt_reactants_str = self.answer["reactant_smiles"]

        # Parse and validate submitted reactants
        submitted_parts = [s.strip() for s in params.reactants.split(".") if s.strip()]

        if not submitted_parts:
            return self._failure_output("No reactants provided.", params.reactants)

        submitted_mols = []
        submitted_canonical = []
        for smi in submitted_parts:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return self._failure_output(
                    f"Invalid SMILES: '{smi}'.", params.reactants
                )
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                return self._failure_output(
                    f"SMILES sanitization failed for '{smi}': {e}", params.reactants
                )
            submitted_mols.append(mol)
            submitted_canonical.append(Chem.MolToSmiles(mol, canonical=True))

        # Parse ground truth reactants
        gt_parts = [s.strip() for s in gt_reactants_str.split(".") if s.strip()]
        gt_mols = [Chem.MolFromSmiles(s) for s in gt_parts]
        gt_canonical = sorted([Chem.MolToSmiles(m, canonical=True) for m in gt_mols])

        # Check exact match
        submitted_sorted = sorted(submitted_canonical)
        if submitted_sorted == gt_canonical:
            feedback = (
                f"Exact match with ground truth reactants.\n\n"
                f"Your reactants: {params.reactants}\n\n"
                f"Reward: 1.0"
            )
            self.submitted += 1
            return ToolOutput(
                blocks=[TextBlock(text=feedback)],
                metadata={
                    "task_id": self.validated.task_id,
                    "target_smiles": self.validated.target_smiles,
                    "submitted": params.reactants,
                    "exact_match": True,
                },
                reward=1.0,
                finished=True,
            )

        # The product shares most of its fingerprint bits with its own reactants,
        # so submitting the target itself (or a stereo/charge variant of it) as a
        # "reactant" would score well on similarity without proposing any reaction.
        target_key = _identity_key(Chem.MolFromSmiles(self.validated.target_smiles))
        if any(_identity_key(m) == target_key for m in submitted_mols):
            feedback = (
                f"Submission received.\n\n"
                f"Your reactants (canonical): {'.'.join(submitted_sorted)}\n\n"
                f"One of the submitted reactants is the target molecule itself, which "
                f"is not a synthetic step.\n\n"
                f"Reward: 0.0"
            )
            self.submitted += 1
            return ToolOutput(
                blocks=[TextBlock(text=feedback)],
                metadata={
                    "task_id": self.validated.task_id,
                    "target_smiles": self.validated.target_smiles,
                    "submitted": params.reactants,
                    "submitted_canonical": submitted_sorted,
                    "exact_match": False,
                    "contains_target": True,
                },
                reward=0.0,
                finished=True,
            )

        # Partial credit via fingerprint similarity
        reward = self._compute_fingerprint_reward(submitted_mols, gt_mols)

        feedback = (
            f"Submission received.\n\n"
            f"Your reactants (canonical): {'.'.join(submitted_sorted)}\n\n"
            f"Fingerprint similarity reward: {reward:.4f}\n\n"
            f"Reward: {reward:.4f}"
        )
        self.submitted += 1
        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "target_smiles": self.validated.target_smiles,
                "submitted": params.reactants,
                "submitted_canonical": submitted_sorted,
                "exact_match": False,
                "tanimoto_similarity": reward,
            },
            reward=reward,
            finished=True,
        )

    def _compute_fingerprint_reward(
        self,
        submitted_mols: list,
        gt_mols: list,
    ) -> float:
        """
        Compute reward based on Morgan fingerprint Tanimoto similarity.

        Combines all reactant fingerprints via bitwise OR, then computes
        Tanimoto between the combined submitted and combined ground truth.
        Capped at 0.95 to reserve 1.0 for exact match.
        """
        radius = 2
        n_bits = 2048

        def combined_fingerprint(mols):
            fps = [
                AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
                for m in mols
            ]
            combined = fps[0]
            for fp in fps[1:]:
                combined = combined | fp
            return combined

        submitted_fp = combined_fingerprint(submitted_mols)
        gt_fp = combined_fingerprint(gt_mols)

        tanimoto = DataStructs.TanimotoSimilarity(submitted_fp, gt_fp)
        reward = min(tanimoto, 0.95)

        return round(reward, 4)

    def _failure_output(self, reason: str, submitted: str) -> ToolOutput:
        feedback = (
            f"Invalid submission.\n\n"
            f"Reason: {reason}\n\n"
            f"Your submission: {submitted}\n\n"
            f"Reward: 0.0. This submission was not graded; submit corrected reactants."
        )
        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "target_smiles": self.validated.target_smiles,
                "submitted": submitted,
                "valid": False,
                "reason": reason,
            },
            reward=0.0,
            finished=False,
        )
