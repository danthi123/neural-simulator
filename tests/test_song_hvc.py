import numpy as np
from sim.song_hvc import SongHVC

def test_chain_advances_one_state_per_step_and_is_deterministic():
    c = SongHVC(n_states=6, n_concepts=8, seed=42)
    c.reset(intention=0)
    states = [c.step()["state"] for _ in range(6)]
    assert states == [0, 1, 2, 3, 4, 5]          # synfire-like chain
    # past chain end -> terminal sentinel, not crash
    assert c.step()["state"] == -1
    c2 = SongHVC(n_states=6, n_concepts=8, seed=42)
    c2.reset(intention=0)
    states2 = [c2.step()["state"] for _ in range(6)]
    assert states2 == states                      # deterministic


def test_rollout_returns_ordered_concept_sequence_of_length_k():
    c = SongHVC(n_states=8, n_concepts=10, seed=1)
    seq = c.rollout(intention=2, length=3)
    assert isinstance(seq, list) and len(seq) == 3
    assert all(0 <= k < 10 for k in seq)
    # deterministic for same (intention, length, weights)
    assert c.rollout(intention=2, length=3) == seq

def test_intention_biases_first_states_so_two_intentions_can_differ():
    c = SongHVC(n_states=8, n_concepts=10, seed=1)
    # inject distinct intention bias, then rollouts may differ
    c.set_intention_bias(intention=0, concept_seq=[1, 2, 3])
    c.set_intention_bias(intention=1, concept_seq=[4, 5, 6])
    assert c.rollout(0, 3) == [1, 2, 3]
    assert c.rollout(1, 3) == [4, 5, 6]
