# RetroSynth

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/RetroSynth)

## Description

**RetroSynth** is an environment for evaluating agents on single-step retrosynthesis tasks. Given a target molecule's SMILES notation, agents must propose the reactants that would produce it in a single synthetic step. The dataset is derived from the [USPTO-50k](https://tdcommons.ai/generation_tasks/retrosyn/) reaction dataset, filtered for commercially plausible reagents.

## Capabilities

- Proposing reactant sets for single-step retrosynthesis
- Reasoning about organic synthesis reactions and reaction mechanisms
- Understanding structure-reactivity relationships in organic chemistry
- Working with SMILES notation for molecular representation

## Compute Requirements

RetroSynth does not require a sandbox. It has minimal compute requirements.

## License

[CC0](https://creativecommons.org/public-domain/cc0/).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks), totaling 1,100 tasks. Each task presents a target molecule SMILES string and asks the agent to propose the reactants as dot-separated SMILES (e.g., `CCO.CC(=O)Cl`).

Tasks are filtered from [USPTO-50k](https://tdcommons.ai/generation_tasks/retrosyn/) for commercial plausibility:
- Allowed elements: C, N, O, F, P, S, Cl, Br, I, B, Si, H
- Reactant constraints: <= 30 heavy atoms, <= 3 rings per reactant
- Product constraints: 5-100 heavy atoms
- Reactant distribution: ~75% two-reactant, ~25% single-reactant reactions

## Reward Structure

This is a sparse, verifiable reward environment. The agent calls `submit_reactants` once with proposed reactants, and the reward is computed as follows:

- **Exact match**: Reward **1.0** if the canonical SMILES of submitted reactants exactly match the ground truth.
- **Partial match**: Reward is the Tanimoto similarity (Morgan fingerprints, radius 2, 2048 bits) between the combined fingerprints of submitted and ground truth reactants, capped at **0.95**.
- **Invalid SMILES**: Reward **0.0**.

We do not use LLM graders for this task.

## Data

Tasks are derived from the [USPTO-50k](https://tdcommons.ai/generation_tasks/retrosyn/) reaction dataset (via [Therapeutics Data Commons](https://tdcommons.ai/)), filtered for commercially plausible reagents. Data files are stored on the OpenReward platform.

## Tools

Agents are given a single tool:

- `submit_reactants`: Submit proposed reactants as dot-separated SMILES (e.g., `CCO.CC(=O)Cl`). Returns the reward based on fingerprint similarity to ground truth. This tool can only be called once per task.

## Time Horizon

RetroSynth is a single-turn environment. The agent receives a target molecule and submits one set of proposed reactants. Each task requires exactly one tool call.

## Environment Difficulty

Task difficulty varies based on molecular complexity. Products range from 8 to 49 heavy atoms, with larger molecules typically requiring more sophisticated retrosynthetic reasoning. Tasks include both single-reactant transformations (functional group interconversions, ~25%) and multi-reactant reactions (coupling, amide formation, etc., ~75%).

## Other Environment Requirements

There are no further environment requirements; RetroSynth works out of the box with the OpenReward endpoint.

## Safety

Agents in RetroSynth are asked to propose chemical reactants for synthesis tasks. The environment does not present direct safety risks, as agents only provide SMILES strings evaluated computationally, with no access to external systems or real chemical processes.

However, this is a dual-use domain as capabilities learnt in this environment could be used for malicious purposes when combined with other agentic workflows.

## Citations

```bibtex
@dataset{GRRetroSynth,
  author    = {General Reasoning Inc. Team},
  title     = {RetroSynth},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/RetroSynth}
}
```

```bibtex
@article{lowe2012extraction,
  title={Extraction of chemical structures and reactions from the literature},
  author={Lowe, Daniel Mark},
  year={2012},
  publisher={University of Cambridge}
}
```
