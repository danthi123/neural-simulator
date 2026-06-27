"""CI guard for Burndown conversion C1: ArgStructureComposer's verb-frame word-ordering on the VALIDATED
SPIKING competitive-queuing read-out (NeuralSerialOrderRenderer) instead of the numpy FrameCQ.

Pins the de-risk's load-bearing results (research/findings/2026-06-27-burndown-C1-framecq-spiking-GO.md):
  * PARITY -- the spiking emit-order == the numpy FrameCQ.emit-order on every verb frame + every realized-slot
    subset (the spiking substrate computes the SAME ordering function);
  * the wired render -- ArgStructureComposer(use_spiking_cq=True).render produces the SAME prose as the numpy
    default for the headline frames (so the conversion is transparent at the surface);
  * EQUAL-DRIVE anti-cheat FAILS -- a flat primacy gradient does NOT reproduce the order (the neurons serialize,
    not pool bias);
  * MOAT 0-FA + AGRAMMATISM preserved on the spiking path.

The spiking renderer builds a SimulationBridge + runs the Izhikevich step loop -> GPU (CuPy) only; this guard
skips off-GPU (numpy is the CPU-portable oracle path, covered by test_argstructure_composer.py). On GPU it does
NOT skip. Reuse-by-import; NO sim/ edit.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_gpu_backend(),
    reason="C1's spiking competitive-queuing renderer builds a SimulationBridge + runs the Izhikevich step loop; "
           "GPU (CuPy) is the acceptance environment. The numpy oracle path is guarded by "
           "test_argstructure_composer.py.")

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, FrameCQ as NumpyFrameCQ, SpikingFrameCQ, FRAME_LEXICON, frame_id,
    content_slot_count, frame_for, FUNCTION_WORDS, reparse_to_fact)
from research.runners.neural_serial_order_renderer import NeuralSerialOrderRenderer  # noqa: E402
from research.runners._phaseB_serial_order_spiking_derisk import (  # noqa: E402
    pool_rates, EQUAL_pA)


def _all_realized_subsets(verb):
    """Every realized-slot subset render() can produce (agent+action mandatory; obliques present-only), each kept
    in canonical-frame (ascending) order -- exactly how render builds `realized_idx`."""
    units = frame_for(verb)
    optional = [i for i, u in enumerate(units) if u[1] not in ("agent", "action")]
    mandatory = [i for i, u in enumerate(units) if u[1] in ("agent", "action")]
    out = []
    for mask in range(1 << len(optional)):
        chosen = [optional[j] for j in range(len(optional)) if (mask >> j) & 1]
        out.append(sorted(mandatory + chosen))
    return out


@pytest.mark.parametrize("seed", [42, 43])
def test_spiking_emit_order_parity_with_numpy(seed):
    """The spiking competitive-queuing read-out reproduces the numpy FrameCQ ordering EXACTLY on every frame +
    every realized subset (the substrate changed; the function did not)."""
    numpy_cq = NumpyFrameCQ(seed=seed)
    spk_cq = SpikingFrameCQ(seed=seed)
    for verb in FRAME_LEXICON:
        fid = frame_id(verb)
        for idx in _all_realized_subsets(verb):
            assert spk_cq.emit_order(fid, idx) == numpy_cq.emit_order(fid, idx), (verb, idx)


def test_wired_render_matches_numpy_oracle():
    """ArgStructureComposer(use_spiking_cq=True).render produces the SAME prose as the numpy-default composer for
    the headline frames -- the conversion is transparent at the user surface; the moat gates both."""
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase",
             "park", "ball", "bone", "table", "river"]
    facts = [
        {"agent": "boy", "action": "go", "GOAL": "park"},
        {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
        {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
        {"agent": "cat", "action": "chase", "patient": "river"},
    ]
    numpy_c = ArgStructureComposer(seed=42, D=64, vocab=vocab)               # default: numpy FrameCQ oracle
    spk_c = ArgStructureComposer(seed=42, D=64, vocab=vocab, use_spiking_cq=True)  # the spiking renderer
    for f in facts:
        numpy_c.store_fact(f)
        spk_c.store_fact(f)
    for f in facts:
        np_out = numpy_c.render(f, numpy_c._composite_for(f))
        sp_out = spk_c.render(f, spk_c._composite_for(f))
        assert sp_out == np_out, (f, np_out, sp_out)
    # the headline render specifically:
    boy = {"agent": "boy", "action": "go", "GOAL": "park"}
    assert spk_c.render(boy, spk_c._composite_for(boy)) == "the boy goes to the park"


@pytest.mark.parametrize("use_spiking_cq", [False, True])
def test_per_call_override_matches(use_spiking_cq):
    """The per-call `use_spiking_cq` override picks the engine without changing the answer (parity), and a
    numpy-default instance can opt INTO spiking per call (and vice-versa)."""
    vocab = ["boy", "go", "park", "girl", "give", "ball", "dog"]
    c = ArgStructureComposer(seed=42, D=64, vocab=vocab, use_spiking_cq=not use_spiking_cq)
    f = {"agent": "boy", "action": "go", "GOAL": "park"}
    c.store_fact(f)
    assert c.render(f, c._composite_for(f), use_spiking_cq=use_spiking_cq) == "the boy goes to the park"


def test_equal_drive_control_fails():
    """The EQUAL-DRIVE anti-cheat: a flat primacy gradient (equal current to every pool) does NOT reliably
    reproduce the canonical order -- the spiking neurons do the serialization, not a fixed pool bias. We assert
    the equal-drive order is NOT the canonical order on the larger frames (>=3 slots)."""
    r = NeuralSerialOrderRenderer(seed=42)
    mismatches = 0
    total = 0
    for verb in FRAME_LEXICON:
        n = content_slot_count(verb)
        if n < 3:
            continue
        idx = list(range(n))
        eq_drive = {int(c): EQUAL_pA for c in idx}
        eq_rate = pool_rates(r.bridge, r.pool_idx, eq_drive)
        eq_order = [int(c) for c in sorted(idx, key=lambda c: -eq_rate[int(c)])]
        total += 1
        if eq_order != idx:
            mismatches += 1
    # equal drive must break the canonical order on the majority of multi-slot frames (it has no primacy gradient).
    assert total > 0
    assert mismatches >= total, f"equal-drive reproduced the canonical order on {total - mismatches}/{total} frames"


def test_spiking_path_moat_and_agrammatism():
    """The no-confab moat (unstored cue -> None) + the agrammatism ablation are substrate-agnostic and intact on
    the spiking-ordering path."""
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase",
             "park", "ball", "bone", "table", "river"]
    c = ArgStructureComposer(seed=42, D=64, vocab=vocab, use_spiking_cq=True)
    c.store_fact({"agent": "boy", "action": "go", "GOAL": "park"})
    # moat:
    assert c.query_role("GOAL", agent="boy", action="eat") is None
    assert c.query_role("GOAL", agent="cat", action="go") is None
    # agrammatism:
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    full = c.render(fact, c._composite_for(fact))
    tele = c.render(fact, c._composite_for(fact), ablate_closed_class=True)
    assert tele == "boy go park"
    assert tele != full
    assert all(w not in FUNCTION_WORDS for w in tele.split())
    assert reparse_to_fact(full, fact)
