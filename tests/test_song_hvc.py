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


def test_babble_perturbs_one_slot_deterministically_by_rng():
    c = SongHVC(n_states=8, n_concepts=10, seed=1)
    base = [1, 2, 3]
    rng = np.random.default_rng(7)
    cand = c.babble(base, rng, temperature=1.0)
    assert len(cand) == len(base)
    # exactly the babble policy: at most one slot changed, in-range
    assert sum(a != b for a, b in zip(base, cand)) <= 1
    assert all(0 <= k < 10 for k in cand)
    # deterministic given rng state
    rng2 = np.random.default_rng(7)
    assert c.babble(base, rng2, temperature=1.0) == cand

def test_babble_temperature_zero_is_noop():
    c = SongHVC(n_states=8, n_concepts=10, seed=1)
    assert c.babble([1, 2, 3], np.random.default_rng(0),
                    temperature=0.0) == [1, 2, 3]


def test_reinforce_strengthens_rewarded_mapping_only():
    c = SongHVC(n_states=8, n_concepts=10, seed=1)
    seq = [3, 5, 7]
    w_before = c.W.copy()
    c.reinforce(intention=0, concept_seq=seq, reward=1.0, lr=0.5)
    # rewarded (state t -> concept seq[t]) weights increased
    for t, k in enumerate(seq):
        assert c.W[t, k] > w_before[t, k]
    # zero reward -> no change
    w_mid = c.W.copy()
    c.reinforce(0, seq, reward=0.0, lr=0.5)
    assert np.allclose(c.W, w_mid)
    # after enough positive reinforcement the chain emits seq
    for _ in range(50):
        c.reinforce(0, seq, reward=1.0, lr=0.5)
    assert [int(np.argmax(c.W[t])) for t in range(3)] == seq
