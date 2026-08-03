"""Focused guards for the opt-in plastic episodic source memory."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.plastic_source_memory import PlasticSourceConfig, PlasticSourceMemory


def _small_config():
    return PlasticSourceConfig(
        n_banks=4,
        proposition_neurons_per_bank=2048,
        proposition_pattern_size=16,
        source_neurons_per_bank=8,
        training_cycles=3,
        training_steps=20,
        rest_steps=10,
        read_steps=80,
        support_threshold=0.25,
    )


def test_source_association_is_learned_and_learning_gate_is_causal():
    memory = PlasticSourceMemory(seed=42, config=_small_config())
    initial = memory.weight_summary()

    no_learn = memory.observe(
        kind="what_does",
        cue=("dog", "go"),
        candidate="north",
        learning_enabled=False,
        measure_weights=True,
    )
    frozen = memory.weight_summary()
    memory.observe(
        kind="what_does",
        cue=("dog", "go"),
        candidate="north",
        learning_enabled=True,
    )
    learned = memory.weight_summary()

    seen = memory.support(kind="what_does", cue=("dog", "go"), candidate="north")
    wrong = memory.support(kind="what_does", cue=("dog", "go"), candidate="south")
    after_retrieval = memory.weight_summary()

    assert initial["l1"] == 0.0
    assert no_learn["weight_l1_delta"] == 0.0
    assert frozen["l1"] == 0.0
    assert learned["l1"] > 0.0
    assert after_retrieval == learned
    assert seen["source_consistent"] is True
    assert wrong["source_consistent"] is False
    assert seen["support"] > wrong["support"] + 0.10
    assert not hasattr(memory, "facts")
    assert not hasattr(memory, "expected_answers")


def test_source_path_lesion_and_permuted_experience_follow_synapses():
    memory = PlasticSourceMemory(seed=43, config=_small_config())
    memory.observe(kind="what_does", cue=("dog", "go"), candidate="south")

    taught = memory.support(kind="what_does", cue=("dog", "go"), candidate="south")
    untaught = memory.support(kind="what_does", cue=("dog", "go"), candidate="north")
    lesioned = memory.support(
        kind="what_does",
        cue=("dog", "go"),
        candidate="south",
        lesion=True,
    )

    assert taught["source_consistent"] is True
    assert untaught["source_consistent"] is False
    assert taught["support"] > untaught["support"] + 0.10
    assert lesioned["source_consistent"] is False
    assert lesioned["support"] < 0.05
