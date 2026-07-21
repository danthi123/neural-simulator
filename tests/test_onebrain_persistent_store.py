"""CI guard: OneBrainComposer(persistent_store=True) -- the fact-store lives in DEVICE synapses (cp_rf_store_*) and
persists across binds -- gives IDENTICAL recall to the staged (host store_conns) path, moat intact (2026-07-20,
fact-store on the substrate Phase 2). GPU-path; skips on numpy."""
import pytest

from sim.backend import is_gpu_backend

pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="OneBrainComposer RF path is GPU")

VOCAB = ["dog", "go", "cat", "look", "south", "apple", "stop", "north", "owl", "eat",
         "mouse", "fish", "chase", "wolf", "hunt", "deer", "big", "hot"]
FACTS = [("dog", "chase", "cat"), ("owl", "eat", "mouse"), ("wolf", "hunt", "deer")]


def _build(persistent_store):
    from research.runners.one_brain_composer import OneBrainComposer
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, persistent_store=persistent_store)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for a, v, p in FACTS:
        c.store(a, v, p)
    return c


def test_persistent_store_recall_parity_and_moat():
    staged = _build(False)
    persist = _build(True)
    sa = [staged.query_patient(a, v) for a, v, _ in FACTS]
    pa = [persist.query_patient(a, v) for a, v, _ in FACTS]
    assert sa == ["cat", "mouse", "deer"], f"staged recall wrong: {sa}"
    assert pa == sa, f"persistent-store recall differs from staged: {pa} vs {sa}"
    # the no-confab moat holds with the device-synapse store
    assert persist.query_patient("apple", "stop") is None, "moat breach under persistent_store"


def test_persistent_store_uses_device_synapses():
    # persistent_store installs cp_rf_store_* on the bridge (the store lives in device synapses, not just host)
    persist = _build(True)
    persist.query_patient("dog", "chase")               # a read syncs the persistent store
    assert getattr(persist.b, "cp_rf_store_re", None) is not None, "persistent store not installed on the bridge"
