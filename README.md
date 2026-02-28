# RetroSynth

Single-step retrosynthesis benchmark for evaluating AI agents' ability to propose reactants that produce a target molecule.

## Overview

Given a target molecule's SMILES notation, agents must identify and propose the reactants that would synthesize it in a single reaction step. This task evaluates chemistry reasoning and knowledge of organic synthesis.

**Data Source:** USPTO-50k reaction dataset (via Therapeutics Data Commons)

## Task Format

- **Input:** Target molecule SMILES + number of expected reactants
- **Output:** Proposed reactants as dot-separated SMILES (e.g., `CCO.CC(=O)Cl`)
- **Tool:** `submit_reactants`

### Example Task

```json
{
  "task_id": "retro_test_0000",
  "target_smiles": "Cc1ccc2cc(CN3CCN(c4ccccn4)CC3)ccc2c1",
  "num_heavy_atoms": 24,
  "num_reactants": 2
}
```

**Ground truth reactants:** `Cc1ccc2cc(CCl)ccc2c1.c1ccc(N2CCNCC2)nc1`

## Dataset Statistics

| Split | Tasks |
|-------|-------|
| Train | 1,000 |
| Test  | 100   |
| **Total** | **1,100** |

### Reactant Distribution

| Reactants | Tasks | Percentage |
|-----------|-------|------------|
| 1 | 274 | 25% |
| 2 | 823 | 75% |
| 3 | 3 | 0.3% |

### Molecule Complexity

- **Heavy atoms:** 8-49 (avg: 23)
- **Product size:** 5-100 heavy atoms

### Filtering Criteria (Commercially Plausible)

- **Allowed elements:** C, N, O, F, P, S, Cl, Br, I, B, Si, H
- **Reactant constraints:** ≤30 heavy atoms, ≤3 rings per reactant
- **Product constraints:** 5-100 heavy atoms

## Reward Model

| Result | Reward |
|--------|--------|
| Exact match | 1.0 |
| Partial match | Tanimoto similarity (capped at 0.95) |
| Invalid SMILES | 0.0 |

**Similarity computation:** Morgan fingerprints (radius=2, 2048 bits) combined via bitwise OR, then Tanimoto similarity.

## Installation

### Requirements

```
openreward
pydantic>=2.0
rdkit-pypi
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation (Optional)

To regenerate the dataset from USPTO-50k:
```bash
pip install PyTDC
python prepare_data.py
```

## Usage

### Run Server

```bash
python server.py
```

### Docker

```bash
docker build -t retrosynth .
docker run -p 8080:8080 retrosynth
```

### Test Agent

```bash
export OPENAI_API_KEY=your_api_key
python test_agent.py
```

## File Structure

```
retrosynth/
├── retrosynth.py      # Main environment class
├── server.py          # Server wrapper (7 lines)
├── test_agent.py      # Agent testing script
├── prepare_data.py    # Dataset preparation
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container config
└── data/
    ├── train.json     # 1,000 training tasks
    └── test.json      # 100 test tasks
```

## API

### Environment Methods

- `list_splits()` - Returns `[("train", "train"), ("test", "test")]`
- `list_tasks(split)` - Returns task specifications (without ground truth)
- `get_prompt()` - Returns chemistry question as `List[TextBlock]`

### Tool: `submit_reactants`

**Input:**
```json
{
  "reactants": "CCO.CC(=O)Cl"
}
```

**Output:**
```json
{
  "reward": 1.0,
  "finished": true,
  "metadata": {
    "task_id": "retro_test_0000",
    "exact_match": true
  }
}
```
