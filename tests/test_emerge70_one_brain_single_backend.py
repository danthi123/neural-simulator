"""CI for EMERGE-70 -- the ONE-BRAIN single-backend PROBE: can the EMERGE-52/54 reasoner + the EMERGE-67/68 A->W
read-out CO-EXECUTE fully-spiking in ONE cupy process (resolving the backend-split constraint EMERGE-69 named)?

CPU-safe where possible: the from_host shim STRUCTURE (it patches the 2 committed helpers' host->device writes to route
through sim.backend.from_host -- byte-identical on numpy), the documented residual write-sites, and the gate-first-moat
turn logic (an ABSTAIN never invokes the A->W spell) are validated with a token-spell stand-in (no GPU). The actual
on-spikes co-execution (two cupy bridges in one process) is GPU-only -> a skip-if-no-cupy smoke (mirror EMERGE-67/68/69
process-sticky skip-guard). The from_host shim is byte-identical on numpy, so the committed runners' numpy CI is unaffected.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import research.runners._emerge70_one_brain_single_backend_probe_derisk as m70  # noqa: E402


def _has_gpu():
    """Non-destructive GPU probe: cupy + a device WITHOUT flipping the active (numpy) backend."""
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_SIM_BACKEND = os.environ.get("SIM_BACKEND", "numpy")
_GPU_REQUESTED = (_SIM_BACKEND == "cupy")
_CACHE_A = os.path.join(_REPO, "bridges", "emerge67_aw", "aw_content.simstate.h5")
_CACHE_F = os.path.join(_REPO, "bridges", "emerge68_aw", "aw_func.simstate.h5")
_CACHES_EXIST = os.path.exists(_CACHE_A) and os.path.exists(_CACHE_F)


class _CountingSpell:
    """A token-spell stand-in with a call counter + the content/func vocab attrs the turn logic checks (no GPU)."""
    def __init__(self):
        self.spell_calls = 0
        self.content_words = {"penguin", "owl", "fly", "walk", "walks", "breathe", "swim"}
        self.func_words = {"the", "a", "can", "does", "not"}
        self._backend_gpu = False

    def spell(self, word):
        self.spell_calls += 1
        return str(word)


# ---------------------------------------------------------------------------------------------------------------------
# CPU-safe: the residual documentation + the from_host shim structure + the gate-first-moat turn logic.
# ---------------------------------------------------------------------------------------------------------------------
def test_residual_write_sites_documented():
    """The probe documents the PRECISE residual EMERGE-69 named: the reasoner's host->device write LINES. There are a
    small, enumerable number of them (in 2 committed helpers) -- the exact scope of the EMERGE-71 follow-on."""
    assert isinstance(m70.RESIDUAL_SITES, list)
    assert 2 <= len(m70.RESIDUAL_SITES) <= 6              # a SMALL, enumerable residual (not a wide rewrite)
    joined = " ".join(m70.RESIDUAL_SITES)
    assert "apply_kernel_update" in joined               # the teaching write (_emerge14)
    assert "_prime_from_winners" in joined               # the inference priming writes (_emerge12)
    assert "cp_connections.data" in joined and "cp_prev_firing_states" in joined


def test_from_host_shim_installs_and_uses_backend_marshal():
    """install_from_host_shim() patches the 2 committed helpers so their host->device writes route through
    sim.backend.from_host (byte-identical on numpy). After install, the module-level symbols are the patched functions,
    and the console modules' imported symbols are repointed. This is the exact fix EMERGE-69 used in _emerge61."""
    import research.runners._emerge14_stageC_onbridge_learning_derisk as m14
    import research.runners._emerge12_stageB2_bridge_tm_derisk as m12
    import research.runners._emerge52_multilevel_conversational_console as m52
    aku, pfw = m70.install_from_host_shim()
    # the patched functions are installed on the source modules AND the console module (repointed)
    assert m14.apply_kernel_update is aku
    assert m12._prime_from_winners is pfw
    assert m52._prime_from_winners is pfw
    assert m52.apply_kernel_update is aku
    # the patched functions reference from_host (the backend H->D marshal) in their source
    import inspect
    assert "from_host" in inspect.getsource(aku)
    assert "from_host" in inspect.getsource(pfw)


def test_from_host_shim_byte_identical_on_numpy():
    """On the numpy backend, from_host(arr) is a passthrough (arr unchanged), so the shim's writes are byte-identical to
    the committed helpers' numpy path -- the committed runners' numpy CI is unaffected."""
    from sim.backend import from_host, get_backend
    _xp, name = get_backend()
    assert name == "numpy"                                # this CPU-safe test runs on numpy
    a = np.arange(7, dtype=np.float32)
    out = from_host(a)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, a)                         # passthrough on numpy -> byte-identical


def test_turn_logic_gate_first_moat_cpu():
    """The flagship turn is gate-first: the REASONER decides, and an ABSTAIN NEVER invokes the A->W spell (moat by
    construction). Validated with a token-spell stand-in + a stub reasoner (no GPU)."""
    class _StubReasoner:
        def __init__(self):
            self.ovr_prop = {"penguin": "walks"}
        def ask_can(self, member, prop):
            if member == "zzz":
                return "I don't know what a zzz is."      # ABSTAIN surface (gate-first)
            if member == "penguin" and prop == "fly":
                return "No, a penguin walks."             # exception -> intransitive frame
            return "Yes, a %s can %s." % (member, prop)   # inherited -> modal frame

    reasoner = _StubReasoner()
    sp = _CountingSpell()
    # ABSTAIN: the render is NEVER reached -> 0 spell calls
    c0 = sp.spell_calls
    gate, surface = m70._emerge_turn(reasoner, sp, "zzz", "fly")
    assert gate == "ABSTAIN"
    assert surface is None
    assert sp.spell_calls - c0 == 0                       # gate-first moat: A->W never invoked on abstain
    # ANSWER (exception): renders "the penguin walks" and DOES invoke the spell
    gate2, surface2 = m70._emerge_turn(reasoner, sp, "penguin", "fly")
    assert gate2 == "ANSWER"
    assert surface2 == "the penguin walks"
    assert sp.spell_calls - c0 > 0
    # ANSWER (inherited): renders a modal frame "the penguin can breathe"
    gate3, surface3 = m70._emerge_turn(reasoner, sp, "penguin", "breathe")
    assert gate3 == "ANSWER"
    assert surface3 == "the penguin can breathe"


def test_numpy_reference_answers_are_the_emerge54_ground_truth():
    """The numpy reference the cupy reasoner must match is the EMERGE-54 design ground truth (per-dimension cancellation
    + inheritance + sibling-discrimination + moat)."""
    ref = m70._numpy_reference_answers(42)
    assert ref["penguin_fly"] == "No, a penguin walks."          # LOCOMOTION overridden
    assert ref["penguin_breathe"] == "Yes, a penguin can breathe."  # RESPIRATION inherited (the EMERGE-54 fix)
    assert ref["owl_fly"].startswith("Yes,")                     # non-override inherits
    assert ref["owl_swim"].startswith("I don't know")            # sibling-discrimination
    assert ref["zzz_breathe"] == "I don't know what a zzz is."   # moat


# ---------------------------------------------------------------------------------------------------------------------
# GPU smoke (skip unless the PROCESS is SIM_BACKEND=cupy AND the caches exist): the reasoner + A->W co-execute in one
# cupy process; the reasoner's answers == the numpy reference; the moat holds.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (_GPU_REQUESTED and _CACHES_EXIST and _has_gpu()),
                    reason="one-process co-execution needs SIM_BACKEND=cupy + the EMERGE-67/68 caches (process-sticky)")
def test_gpu_one_process_coexecution_and_moat():
    """GPU smoke: build the reasoner (from_host shim, cupy) + the A->W read-out (cupy) in ONE process; a full flagship
    turn co-executes (reason -> spiking render); the reasoner's answers == the numpy reference; the gate-first moat
    holds (0 A->W spell calls on abstain)."""
    d = m70._probe_one(42)
    assert not d.get("skip"), d.get("skip")
    assert d["gpu"] and d["aw_gpu"]
    assert d["reasoner_matches_numpy_ref"]                # cupy reasoner == numpy reference
    assert d["coexecute_ok"]                              # a full turn co-executed in one process
    assert d["moat_holds"] and d["spell_calls_on_abstain"] == 0   # gate-first moat intact
    assert d["penguin_fly_surface"] == "the penguin walks"
    assert d["penguin_breathe_surface"] == "the penguin can breathe"
