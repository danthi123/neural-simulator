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
