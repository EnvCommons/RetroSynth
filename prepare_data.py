"""
Download and prepare USPTO-50k retrosynthesis tasks.

Downloads USPTO-50k via TDC (Therapeutics Data Commons),
filters for single-step commercially plausible reactions,
canonicalizes SMILES, and creates 1000 train + 100 test tasks.

Run once locally: python prepare_data.py
Requires: pip install rdkit-pypi PyTDC
"""

import json
import random
from pathlib import Path

from rdkit import Chem
from tdc.generation import RetroSyn

TRAIN_SIZE = 1000
TEST_SIZE = 100
TOTAL = TRAIN_SIZE + TEST_SIZE
RANDOM_SEED = 42

# Element whitelist for "commercially plausible" filtering
# C, N, O, F, P, S, Cl, Br, I, B, Si, H
ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

MAX_REACTANT_HEAVY_ATOMS = 30
MAX_REACTANT_RINGS = 3
MIN_PRODUCT_HEAVY_ATOMS = 5
MAX_PRODUCT_HEAVY_ATOMS = 100


def canonicalize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True), mol
    except Exception:
        return None, None


def is_commercially_plausible(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() > MAX_REACTANT_HEAVY_ATOMS:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ELEMENTS:
            return False
    ring_info = mol.GetRingInfo()
    if len(ring_info.AtomRings()) > MAX_REACTANT_RINGS:
        return False
    return True


def make_question(target_smiles: str, num_reactants: int) -> str:
    return (
        f"You are an expert synthetic chemist. Your task is to propose a set of "
        f"reactants that would produce the following target molecule in a single "
        f"synthetic step.\n\n"
        f"**Target molecule (SMILES):** `{target_smiles}`\n\n"
        f"**Number of expected reactants:** {num_reactants}\n\n"
        f"**Instructions:**\n"
        f"1. Analyze the target molecule's structure\n"
        f"2. Identify plausible bond disconnections\n"
        f"3. Propose commercially available starting materials\n"
        f"4. Submit your answer as dot-separated SMILES using the submit_reactants tool\n\n"
        f"**Example format:** `CCO.CC(=O)Cl` (two reactants separated by a dot)\n\n"
        f"Submit your proposed reactants using the `submit_reactants` tool."
    )


def process_dataframe(df) -> list[dict]:
    """Process a TDC dataframe (columns: input=product, output=reactants)."""
    tasks = []
    for _, row in df.iterrows():
        product_smiles = row["input"]
        reactants_smiles = row["output"]

        # Validate product
        canon_product, prod_mol = canonicalize(product_smiles)
        if canon_product is None:
            continue

        n_heavy = prod_mol.GetNumHeavyAtoms()
        if n_heavy < MIN_PRODUCT_HEAVY_ATOMS or n_heavy > MAX_PRODUCT_HEAVY_ATOMS:
            continue

        # Parse and validate each reactant
        reactant_parts = reactants_smiles.split(".")
        if len(reactant_parts) < 1 or len(reactant_parts) > 5:
            continue

        canon_reactants = []
        all_plausible = True
        for r in reactant_parts:
            cr, _ = canonicalize(r)
            if cr is None:
                all_plausible = False
                break
            if not is_commercially_plausible(cr):
                all_plausible = False
                break
            canon_reactants.append(cr)

        if not all_plausible:
            continue

        tasks.append({
            "target_smiles": canon_product,
            "reactant_smiles": ".".join(sorted(canon_reactants)),
            "num_reactants": len(canon_reactants),
            "num_heavy_atoms": n_heavy,
        })

    return tasks


def main():
    random.seed(RANDOM_SEED)

    print("Downloading USPTO-50k via TDC...")
    data = RetroSyn(name="USPTO-50K")
    split = data.get_split()

    all_candidates = []
    for split_name in ["train", "valid", "test"]:
        df = split[split_name]
        print(f"  {split_name}: {len(df)} reactions")
        candidates = process_dataframe(df)
        print(f"  -> {len(candidates)} passed filters")
        all_candidates.extend(candidates)

    # Deduplicate by target_smiles
    seen = set()
    unique = []
    for c in all_candidates:
        if c["target_smiles"] not in seen:
            seen.add(c["target_smiles"])
            unique.append(c)

    print(f"\nTotal unique candidates: {len(unique)}")

    if len(unique) < TOTAL:
        print(f"WARNING: only {len(unique)} candidates, need {TOTAL}. Using all available.")

    random.shuffle(unique)
    selected = unique[:TOTAL]

    # First TEST_SIZE go to test, rest to train
    test_tasks = []
    train_tasks = []
    for idx, item in enumerate(selected):
        if idx < TEST_SIZE:
            split_name = "test"
            split_idx = idx
        else:
            split_name = "train"
            split_idx = idx - TEST_SIZE

        task = {
            "task_id": f"retro_{split_name}_{split_idx:04d}",
            "target_smiles": item["target_smiles"],
            "num_heavy_atoms": item["num_heavy_atoms"],
            "num_reactants": item["num_reactants"],
            "reactant_smiles": item["reactant_smiles"],
            "question": make_question(item["target_smiles"], item["num_reactants"]),
        }

        if split_name == "test":
            test_tasks.append(task)
        else:
            train_tasks.append(task)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    with open(data_dir / "train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)

    print(f"\nSaved {len(train_tasks)} train tasks to data/train.json")
    print(f"Saved {len(test_tasks)} test tasks to data/test.json")

    # Print a sample
    sample = train_tasks[0]
    print(f"\nSample task:")
    print(f"  ID: {sample['task_id']}")
    print(f"  Target: {sample['target_smiles']}")
    print(f"  Reactants: {sample['reactant_smiles']}")
    print(f"  Heavy atoms: {sample['num_heavy_atoms']}")
    print(f"  Num reactants: {sample['num_reactants']}")


if __name__ == "__main__":
    main()
