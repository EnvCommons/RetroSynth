import pytest

import retrosynth
from retrosynth import RetroSynth, SubmitReactantsInput

TARGET = "OCCOc1cccc(C2=N[C@@H](c3ccccc3)CO2)c1"
GOLD = "N#Cc1cccc(OCCO)c1.N[C@@H](CO)c1ccccc1"
TASK = {
    "task_id": "unit_0000",
    "target_smiles": TARGET,
    "num_heavy_atoms": 21,
    "num_reactants": 2,
    "question": "Propose reactants.",
}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setitem(retrosynth.ANSWERS, TASK["task_id"], {"reactant_smiles": GOLD})
    return RetroSynth(task_spec=TASK)


async def submit(env, reactants):
    return await env.submit_reactants(SubmitReactantsInput(reactants=reactants))


@pytest.mark.asyncio
async def test_gold_scores_one(env):
    result = await submit(env, "N[C@@H](CO)c1ccccc1.N#Cc1cccc(OCCO)c1")
    assert result.reward == 1.0
    assert result.finished


@pytest.mark.asyncio
async def test_honest_wrong_answer_gets_partial_credit(env):
    result = await submit(env, "OCCOc1cccc(C(=O)O)c1.N[C@@H](CO)c1ccccc1")
    assert 0.0 < result.reward < 0.95
    assert result.finished


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reactants",
    [
        TARGET,
        "OCCOc1cccc(C2=NC(c3ccccc3)CO2)c1",
        "OCCOc1cccc(C2=N[C@H](c3ccccc3)CO2)c1",
        "OCCOc1cccc(C2=[NH+][C@@H](c3ccccc3)CO2)c1.[Cl-]",
        f"{TARGET}.O",
        f"N#Cc1cccc(OCCO)c1.{TARGET}",
    ],
)
async def test_target_as_reactant_scores_zero(env, reactants):
    result = await submit(env, reactants)
    assert result.reward == 0.0
    assert result.finished
    assert GOLD not in result.blocks[0].text
    assert "N#Cc1cccc(OCCO)c1.N[C@@H](CO)c1ccccc1" not in str(result.metadata)


@pytest.mark.asyncio
async def test_gold_containing_target_still_scores_one(monkeypatch):
    gold = f"{TARGET}.O"
    monkeypatch.setitem(retrosynth.ANSWERS, TASK["task_id"], {"reactant_smiles": gold})
    env = RetroSynth(task_spec=TASK)
    result = await submit(env, gold)
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_invalid_smiles_is_not_graded(env):
    result = await submit(env, "not_a_smiles")
    assert result.reward == 0.0
    assert not result.finished
    result = await submit(env, TARGET)
    assert result.reward == 0.0
    assert result.finished
