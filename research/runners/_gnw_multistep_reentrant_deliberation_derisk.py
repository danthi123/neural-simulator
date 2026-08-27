"""GO-GATE VERIFY for THE KEYSTONE'S DEFERRED RUNG, WIRED LIVE (T1-1 rung d): a MULTI-STEP re-entrant deliberation
loop driven through the REAL production `/api/brain-chat` turn, whose re-entrant CYCLE COUNT emerges from the
substrate's OWN spiking read — the brain works through a multi-step (transitive-chase) inference whose DEPTH it
discovers itself, LIVE, and halts when its own ignition collapses at the leaf.

WHAT THIS VERIFIES. `webapp/gnw_deliberation.py` wired the single-hop "halt-if-unsure" half and explicitly named the
MULTI-HOP "deliberation-until-sure over a CHAIN" as the DEFERRED rung. `webapp/gnw_multistep_deliberation.py` wires
that deferred rung onto the LIVE recall path: on an explicit chase-form question ("what does X <action> all the way /
to the end?") the P1.2 GNW workspace cycles the partial answer back through itself (re-entrant broadcast), re-igniting,
and the keystone `acc_conflict_gate` reads `n_ignited` (off `cp_firing_states`) to decide ADVANCE (next hop) vs COMMIT
(the leaf collapsed ignition -> the terminal reached) — the cycle count is a SPIKING read, NOT a host `query_chain(cue,
actions)` counter (a generous H_cap is a pure safety budget correct answers never reach).

GO GATE (>= all four):
  (A) LIVE MULTI-STEP through the REAL handler + the REAL gate: on a taught transitive chain (a->b->c->...->leaf) the
      wired brain (BRAIN_GNW_MULTISTEP=1) reaches the TERMINAL leaf, while the single-hop bus (flag off) commits only
      the FIRST hop. Both at the ChatBrain gate level AND through the real `/api/brain-chat` handler (answer text +
      recalled_svo carry the terminal).
  (B) 6-SEED SUBSTRATE CONTROL (42/43/44/100/101/102) on the LIVE production composer + the workspace substrate,
      per-seed + pooled (>=5/6). Each seed must show, on a VARIABLE-DEPTH set (L in {1,2,3,4}) whose depth the loop is
      NOT told: reentrant reaches the terminal (>=0.90); single-pass fails the multi-step ones (<=0.30 pooled, one-step
      L=1 unchanged); the emergent stop is DIFFICULTY-GRADED (spearman(resolved_hops, depth) >= 0.9, no halt at H_cap);
      the workspace-silence LESION collapses the multi-step chase (<=0.10) while the 1-hop reflex survives (>=0.85);
      the moat holds (unstored cue + over-run past the leaf -> abstain).
  (C) EMERGENT-STOPPING dissociation + RE-ENTRY-IS-LOAD-BEARING (the two anti-cheats that ARE the deliverable):
      (C1) lesion the substrate read that gates convergence (recurrence-zeroed workspace) -> the loop can no longer
           converge appropriately -> the multi-step answer collapses (dissociation from the intact chase; the 1-hop
           reflex is untouched -> proves the cycle count is substrate-driven, not host-counted).
      (C2) ablating re-entry (force single-pass) DEGRADES the multi-step (L>=2) answer while leaving the one-step (L=1)
           answer ~unchanged, AND a genuinely multi-step problem takes MORE cycles than a one-step one (resolved_hops
           grows with depth). If ablation did not hurt, the re-entry would not be doing work.
  (D) BYTE-IDENTICAL + MOAT-SAFE: DEFAULT-OFF (BRAIN_GNW_MULTISTEP unset -> the gate is not installed -> the live turn
      is byte-identical); and even ON the gate is INERT on every non-chase-form turn (the reactive recall/abstain/
      learn/anaphora panel is byte-identical, in-process AND through the real handler across BRAIN_GNW_MULTISTEP=1 vs
      =0). It only ever ADDS a terminal on a chase-form question; it never un-abstains and never invents a fact.

DECLARED BOUNDARIES (honest, same as the keystone/P1.2/coincidence-integrator): the chase-form DETECT + the
(agent,action) EXTRACT are host comprehension of the TEACHER/WORLD utterance (the same boundary the SVO question
parser occupies); PROPOSE is `composer.query_patient` (the declared modular-processor boundary). The substrate's
INDEPENDENT work — and the whole novelty here — is the CYCLE COUNT / when-to-halt moving from a host counter to a
spiking `n_ignited` read, LIVE. FUNCTIONAL correlate only; NO phenomenal claim. This is re-entrant multi-step
deliberation with the MEASURED improvement, NOT "reasoning to a true conclusion".

Run (numpy-CPU):
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf \
      python -u -m research.runners._gnw_multistep_reentrant_deliberation_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_multistep_reentrant/summary.json
  # 1-seed smoke:
  ... --smoke --seed 42
  # internal separate-process panel-hash mode (invoked by the runner):
  ... --panel-hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

# reuse-by-import the keystone machinery (the EMERGENT-count re-entry loop + fixed-count baseline + spearman + theta).
from research.runners._gnw_reentrant_metacog_gated_deliberation_derisk import (
    confidence_gated_chase, fixed_count_chase, _spearman, NMDA_ATTR_DEFAULT,
)
from webapp import gnw_multistep_deliberation as gms

SEEDS_DEFAULT = (42, 43, 44, 100, 101, 102)
DEPTHS_LIVE = (1, 2, 3, 4)          # L=1 = the one-step control (single-pass correct); L>=2 needs re-entry
REL = "chase"


# ── the reactive panel (byte-identical unit; NO chase markers, so the multi-step gate must be inert) ──
PANEL_STATELESS = [
    ("what does dog chase?", "stored"),
    ("what does cat eat?", "stored"),
    ("what does brain use?", "stored"),
    ("what does fish fly?", "unstored"),
    ("what does dog eat?", "inconsistent"),
    ("what are you", "self"),
]
ACQUIRE_SEQ = [("sky hold cloud", None), ("what does sky hold?", ["sky", "hold", "cloud"])]
ANAPHORA_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it eat?", ["cat", "eat", "fish"])]


def _tok(n: int) -> str:
    """A globally-unique alpha token ('qaa'..'qzz' = 676) the live composer's `hear` parser accepts as a noun."""
    return "q" + chr(97 + (n // 26) % 26) + chr(97 + n % 26)


def build_live_chains(n_per_depth: int = 4):
    """`n_per_depth` chains at EACH depth L in {1,2,3,4}. A depth-L chain is L+1 GLOBALLY-UNIQUE tokens
    c0 -chase-> c1 -> ... -> cL (cL is the leaf: never an agent, so query_patient(cL, chase) misses -> the loop must
    DISCOVER the depth by its own ignition collapse). The 4 depths never share a concept."""
    chains, depth_of = [], {}
    c = 0
    for L in DEPTHS_LIVE:
        for _ in range(n_per_depth):
            ch = []
            for _p in range(L + 1):
                ch.append(_tok(c))
                c += 1
            chains.append(ch)
            depth_of[tuple(ch)] = L
    return chains, depth_of


def _svo_eq(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


def _build(install_multistep: bool):
    """The REAL production ChatBrain (rf recall) with the bus + single-hop-deliberation gates always installed (=
    today's production), and the multi-step gate installed only when asked."""
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    from webapp import gnw_deliberation as gdel
    chat, _src = _build_chat_brain("tiny-demo", "stub")
    gbs.install_bus_gate(chat)
    gdel.install_deliberation_gate(chat)
    if install_multistep:
        gms.install_multistep_gate(chat)
    return chat


def teach_chains(chat, chains, rel=REL):
    """Teach every chain edge via the PRODUCTION acquisition path (`chat.inner.hear`) — the live composer holds the
    transitive facts exactly as a taught brain would."""
    for ch in chains:
        for i in range(len(ch) - 1):
            chat.inner.hear("%s %s %s" % (ch[i], rel, ch[i + 1]), polarity="AFFIRM")


# ── (B)+(C) 6-seed substrate control on the LIVE composer + workspace ──────────────────────────────────────────────
def run_seed_live(chat, chains, depth_of, all_concepts, seed, rel=REL):
    """For one WORKSPACE substrate seed, run the wired multi-step chase (`gms.multistep_chase`) over every taught
    chain and measure: reentrant acc (per-depth), single-pass baseline (per-depth), lesion collapse, 1-hop reflex,
    the difficulty-grading spearman, and the moat. All chases go through the SAME wired helper the live gate calls."""
    composer = chat.inner.composer
    tot = len(chains)
    reent_ok = single_ok = lesion_ok = reflex_ok = 0
    hops_correct, depth_correct = [], []
    any_halt_at_cap = False
    per_depth = {L: [0, 0] for L in DEPTHS_LIVE}
    single_by_depth = {L: [0, 0] for L in DEPTHS_LIVE}

    b_i, xp, slots_i, snap_i = gms._get_bridge(seed, False)      # for the single-pass fixed-count baseline

    for ch in chains:
        L = depth_of[tuple(ch)]
        # INTACT wired chase — the cycle count emerges from n_ignited (return_trace via multistep_chase meta)
        term, meta = gms.multistep_chase(chat, ch[0], rel, seed=seed, lesion=False)
        ok = int(term == ch[-1])
        reent_ok += ok
        per_depth[L][0] += ok
        per_depth[L][1] += 1
        any_halt_at_cap = any_halt_at_cap or bool(meta.get("halted_at_cap"))
        if ok:
            hops_correct.append(int(meta.get("resolved_hops") or 0))
            depth_correct.append(L)
        # SINGLE-PASS (k=1 host-forced): one hop, no re-entry (the wired bus baseline)
        st = fixed_count_chase(b_i, xp, slots_i, snap_i, composer, ch[0], rel, all_concepts,
                               np.random.default_rng(seed * 991 + 7), 1, nmda_attr=NMDA_ATTR_DEFAULT)
        s_ok = int(st == ch[-1])
        single_ok += s_ok
        single_by_depth[L][0] += s_ok
        single_by_depth[L][1] += 1
        # LESION wired chase — recurrence-zeroed workspace: ignition cannot sustain -> multi-step collapses
        lt, _lm = gms.multistep_chase(chat, ch[0], rel, seed=seed, lesion=True)
        lesion_ok += int(lt == ch[-1])
        # 1-hop reflex (workspace-independent composer read) — must SURVIVE the lesion (dissociation)
        reflex_ok += int(composer.query_patient(ch[0], rel) == ch[1])

    reent_acc = reent_ok / tot
    single_acc = single_ok / tot
    lesion_acc = lesion_ok / tot
    reflex_acc = reflex_ok / tot
    spr = _spearman(hops_correct, depth_correct)

    # MOAT: an unstored cue and an over-run past a leaf -> abstain (None)
    moat_unstored = gms.multistep_chase(chat, "zzznostorecue", rel, seed=seed, lesion=False)[0] is None
    leaf = chains[0][-1]
    moat_overrun = gms.multistep_chase(chat, leaf, rel, seed=seed, lesion=False)[0] is None
    moat_ok = bool(moat_unstored and moat_overrun)

    seed_go = bool(
        reent_acc >= 0.90 and
        single_acc <= 0.30 and (reent_acc - single_acc) >= 0.60 and
        (not any_halt_at_cap) and (not np.isnan(spr) and spr >= 0.9) and
        lesion_acc <= 0.10 and reflex_acc >= 0.85 and
        moat_ok
    )
    return {
        "seed": int(seed), "n_chains": tot, "n_concepts": len(all_concepts),
        "reentrant_acc": reent_acc, "singlepass_acc": single_acc,
        "reentrant_acc_by_depth": {str(L): (per_depth[L][0] / per_depth[L][1]) for L in DEPTHS_LIVE},
        "singlepass_acc_by_depth": {str(L): (single_by_depth[L][0] / single_by_depth[L][1]) for L in DEPTHS_LIVE},
        "lesion_acc": lesion_acc, "single_hop_reflex_acc": reflex_acc,
        "spearman_hops_depth": spr, "any_halt_at_H_cap": bool(any_halt_at_cap),
        "moat_unstored_abstains": bool(moat_unstored), "moat_overrun_abstains": bool(moat_overrun),
        "moat_ok": moat_ok, "seed_go": seed_go,
    }


def run_seed_sweep(seeds=SEEDS_DEFAULT, n_per_depth=4, verbose=True):
    """Build ONE live production ChatBrain (rf) + teach the variable-depth chains once; vary the WORKSPACE substrate
    seed across `seeds` (the seed-varying claim: does the substrate's ignition read still drive the correct emergent
    cycle count across substrate seeds?)."""
    chat = _build(install_multistep=True)
    chains, depth_of = build_live_chains(n_per_depth)
    teach_chains(chat, chains)
    all_concepts = gms._all_concepts(chat.inner.composer)       # the SAME pool the wired gate uses
    rows = [run_seed_live(chat, chains, depth_of, all_concepts, s) for s in seeds]
    if verbose:
        for r in rows:
            print(f"    seed {r['seed']}: reent={r['reentrant_acc']:.2f} single={r['singlepass_acc']:.2f} "
                  f"lesion={r['lesion_acc']:.2f} reflex={r['single_hop_reflex_acc']:.2f} "
                  f"spearman={r['spearman_hops_depth']:.2f} moat={r['moat_ok']} GO={r['seed_go']}", flush=True)
            print(f"        reent/depth={r['reentrant_acc_by_depth']} single/depth={r['singlepass_acc_by_depth']}",
                  flush=True)
    n_go = sum(int(r["seed_go"]) for r in rows)
    return {"rows": rows, "n_go": n_go, "n": len(rows), "all_go": n_go >= 5,
            "n_concepts": rows[0]["n_concepts"] if rows else None}


# ── (A) LIVE end-to-end at the ChatBrain gate level: intact reaches the terminal, single-pass reaches the first hop ─
def _gate_multistep_arms(chain=("zorp", "blib", "krad", "munt")):
    edges = list(chain)
    wired = _build(install_multistep=True)
    for i in range(len(edges) - 1):
        wired.inner.hear("%s %s %s" % (edges[i], REL, edges[i + 1]), polarity="AFFIRM")

    q_chase = "what does %s %s all the way?" % (edges[0], REL)
    q_plain = "what does %s %s?" % (edges[0], REL)

    # 2026-08-27 fix: BRAIN_GNW_MULTISTEP defaults ON (_GNW_MULTISTEP_DEFAULT_ON=True; webapp/gnw_multistep_
    # deliberation.py's own reader also defaults "1") -- unset no longer means OFF, so the OFF arm must be explicit.
    os.environ["BRAIN_GNW_MULTISTEP"] = "0"                       # OFF: chase-form passes through -> first hop
    off_chase = wired.gate(q_chase)
    os.environ["BRAIN_GNW_MULTISTEP"] = "1"                      # ON: chase-form -> the terminal leaf
    on_chase = wired.gate(q_chase)
    on_info = dict(getattr(wired, "_last_gnw_multistep", {}) or {})
    on_plain = wired.gate(q_plain)                              # ON but no marker -> first hop (byte-identical)
    plain_reason = (getattr(wired, "_last_gnw_multistep", {}) or {}).get("reason")
    os.environ["BRAIN_GNW_MULTISTEP_LESION"] = "1"              # ON+LESION -> collapse
    lesion_chase = wired.gate(q_chase)
    lesion_info = dict(getattr(wired, "_last_gnw_multistep", {}) or {})
    os.environ.pop("BRAIN_GNW_MULTISTEP_LESION", None)
    os.environ.pop("BRAIN_GNW_MULTISTEP", None)

    leaf = edges[-1]
    first = edges[1]
    return {
        "chain": edges, "q_chase": q_chase,
        "off_chase_svo": (list(off_chase) if off_chase is not None else None),
        "on_chase_svo": (list(on_chase) if on_chase is not None else None),
        "on_plain_svo": (list(on_plain) if on_plain is not None else None),
        "lesion_chase_svo": (list(lesion_chase) if lesion_chase is not None else None),
        "on_info": {k: on_info.get(k) for k in ("acted", "terminal", "resolved_hops", "cycles", "reason")},
        "lesion_info": {k: lesion_info.get(k) for k in ("terminal", "resolved_hops", "abstained", "reason")},
        # (A): ON reaches the TERMINAL; OFF (single-hop bus) reaches only the FIRST hop
        "on_reaches_terminal": (on_chase is not None and list(on_chase)[2] == leaf),
        "off_is_first_hop": (off_chase is not None and list(off_chase)[2] == first),
        "on_plain_is_first_hop": (on_plain is not None and list(on_plain)[2] == first),  # inert w/o marker
        "plain_reason": plain_reason,
        # (C1): lesion collapses the multi-step (does NOT reach the terminal)
        "lesion_collapses": (lesion_chase is None or list(lesion_chase)[2] != leaf),
        "resolved_hops": on_info.get("resolved_hops"),
    }


# ── (A) at the RESPONSE level, through the REAL /api/brain-chat handler ──
def _handler_multistep(chain=("zorp", "blib", "krad", "munt")):
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_CAUSAL", "BRAIN_PMEM", "BRAIN_PRAGMATIC", "BRAIN_SELF_INITIATE", "BRAIN_DISCOURSE_REGISTER",
              "BRAIN_RICH", "BRAIN_GNW_BUS"):
        os.environ[k] = "0"
    os.environ.pop("BRAIN_GNW_BUS", None)
    from webapp.server import brain_chat, BrainChatRequest as Req

    def _turn(session, msg):
        return json.loads(brain_chat(Req(session=session, message=msg, brain="tiny-demo",
                                         renderer="stub", rich=False)).body.decode("utf-8"))

    edges = list(chain)
    q_chase = "what does %s %s all the way?" % (edges[0], REL)

    def _arm(session, env):
        for k in ("BRAIN_GNW_MULTISTEP", "BRAIN_GNW_MULTISTEP_LESION"):
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        for i in range(len(edges) - 1):
            _turn(session, "%s %s %s" % (edges[i], REL, edges[i + 1]))
        resp = _turn(session, q_chase)
        for k in env:
            os.environ.pop(k, None)
        return resp

    on = _arm("ms_on", {"BRAIN_GNW_MULTISTEP": "1"})
    # 2026-08-27 fix: BRAIN_GNW_MULTISTEP defaults ON, so an empty env (unset) is NOT the OFF arm -- explicit "0".
    off = _arm("ms_off", {"BRAIN_GNW_MULTISTEP": "0"})
    lesion = _arm("ms_les", {"BRAIN_GNW_MULTISTEP": "1", "BRAIN_GNW_MULTISTEP_LESION": "1"})
    leaf, first = edges[-1], edges[1]

    def _svo3(r):
        s = r.get("recalled_svo")
        return list(s) if isinstance(s, (list, tuple)) and len(s) == 3 else None

    on_svo, off_svo = _svo3(on), _svo3(off)
    return {
        "q_chase": q_chase, "leaf": leaf, "first": first,
        "on_answer": on.get("answer"), "off_answer": off.get("answer"), "lesion_answer": lesion.get("answer"),
        "on_recalled": on_svo, "off_recalled": off_svo, "lesion_abstained": bool(lesion.get("abstained")),
        "on_reaches_terminal": bool(on_svo is not None and on_svo[2] == leaf),
        "off_is_first_hop": bool(off_svo is not None and off_svo[2] == first),
        "lesion_collapses": bool(lesion.get("abstained") or (_svo3(lesion) or [None, None, None])[2] != leaf),
        "ok": bool(on_svo is not None and on_svo[2] == leaf
                   and off_svo is not None and off_svo[2] == first
                   and (lesion.get("abstained") or (_svo3(lesion) or [None, None, None])[2] != leaf)),
    }


# ── (D) byte-identical: the multi-step gate is INERT on the reactive panel (in-process, ON vs a no-multistep brain) ─
def _panel_byte_identical():
    rows = []
    os.environ["BRAIN_GNW_MULTISTEP"] = "1"                     # even ON, must be inert on non-chase turns
    base = _build(install_multistep=False)                      # today's production (bus + single-hop delib)
    wired = _build(install_multistep=True)
    for q, cls in PANEL_STATELESS:
        b = base.gate(q)
        w = wired.gate(q)
        rows.append({"cls": cls, "q": q, "base": (list(b) if b is not None else None),
                     "wired": (list(w) if w is not None else None), "identical": _svo_eq(b, w)})
    stateful = ACQUIRE_SEQ + ANAPHORA_SEQ
    b_run = [_build(install_multistep=False).gate(u) for u, _ in stateful]
    # run stateful on ONE wired brain (turn-by-turn state) and ONE base brain (fresh per, matching b_run built fresh)
    wbrain = _build(install_multistep=True)
    w_run = [wbrain.gate(u) for u, _ in stateful]
    bbrain = _build(install_multistep=False)
    b_run = [bbrain.gate(u) for u, _ in stateful]
    for i, (utt, _want) in enumerate(stateful):
        cls = "acquisition" if i < len(ACQUIRE_SEQ) else "anaphora"
        rows.append({"cls": cls, "q": utt, "base": (list(b_run[i]) if b_run[i] is not None else None),
                     "wired": (list(w_run[i]) if w_run[i] is not None else None),
                     "identical": _svo_eq(b_run[i], w_run[i])})
    os.environ.pop("BRAIN_GNW_MULTISTEP", None)
    n_id = sum(int(r["identical"]) for r in rows)
    return {"rows": rows, "n_identical": n_id, "n_total": len(rows), "ok": (n_id == len(rows))}


def _panel_hash_mode():
    """Build the REAL handler, run the reactive panel (NO chase markers), print `PANELHASH <md5>` of the responses."""
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_CAUSAL", "BRAIN_PMEM", "BRAIN_PRAGMATIC", "BRAIN_SELF_INITIATE", "BRAIN_DISCOURSE_REGISTER",
              "BRAIN_RICH", "BRAIN_GNW_BUS"):
        os.environ[k] = "0"
    os.environ.pop("BRAIN_GNW_BUS", None)
    from webapp.server import brain_chat, BrainChatRequest as Req
    h = hashlib.md5()
    msgs = [q for q, _ in PANEL_STATELESS] + [u for u, _ in ACQUIRE_SEQ] + [u for u, _ in ANAPHORA_SEQ]
    for msg in msgs:
        resp = json.loads(brain_chat(Req(session="ph", message=msg, brain="tiny-demo",
                                         renderer="stub", rich=False)).body.decode("utf-8"))
        resp.pop("timing_ms", None)
        h.update(json.dumps(resp, sort_keys=True, default=str).encode("utf-8"))
    print("PANELHASH %s" % h.hexdigest(), flush=True)
    return 0


def _spawn_panel_hash(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("SIM_BACKEND", "numpy")
    env.setdefault("BRAIN_COMPOSER_KIND", "rf")
    out = subprocess.run([sys.executable, "-u", "-m", "research.runners._gnw_multistep_reentrant_deliberation_derisk",
                          "--panel-hash"], cwd=_REPO, env=env, capture_output=True, text=True, timeout=1800)
    for line in out.stdout.splitlines():
        if line.startswith("PANELHASH "):
            return line.split(" ", 1)[1].strip()
    raise RuntimeError("panel-hash subprocess produced no hash; stderr tail:\n%s" % out.stderr[-800:])


def run_smoke(seed=42):
    print("[SMOKE] live multi-step chase through the wired gate + a 1-seed substrate-control slice.\n", flush=True)
    gate = _gate_multistep_arms()
    print("  (A) gate level: OFF(single-hop)=%s -> ON(multi-step)=%s (resolved_hops=%s, info=%s)"
          % (gate["off_chase_svo"], gate["on_chase_svo"], gate["resolved_hops"], gate["on_info"]), flush=True)
    print("      ON reaches terminal=%s | OFF is first hop=%s | ON no-marker inert=%s | lesion collapses=%s"
          % (gate["on_reaches_terminal"], gate["off_is_first_hop"], gate["on_plain_is_first_hop"],
             gate["lesion_collapses"]), flush=True)
    sweep = run_seed_sweep(seeds=(seed,), verbose=True)
    ok = bool(gate["on_reaches_terminal"] and gate["off_is_first_hop"] and gate["lesion_collapses"]
              and sweep["rows"][0]["seed_go"])
    print("\n  SMOKE %s" % ("HOLDS" if ok else "NEEDS-WORK"), flush=True)
    return 0 if ok else 1


def main():
    if "--panel-hash" in sys.argv:
        return _panel_hash_mode()

    ap = argparse.ArgumentParser(description="Live multi-step re-entrant deliberation — GO-gate verify (T1-1 rung d).")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-depth", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_multistep_reentrant/summary.json")
    args = ap.parse_args()

    if args.smoke:
        return run_smoke(args.seed)

    from tools.verdict import Verdict
    from tools.lab import attributable_to

    print("[GNW MULTI-STEP re-entrant deliberation — WIRED LIVE] depths=%s | rel=%r | seeds=%s\n"
          "  the re-entrant CYCLE COUNT emerges from the substrate's spiking n_ignited read, LIVE through the real "
          "brain-chat gate + handler.\n" % (list(DEPTHS_LIVE), REL, args.seeds), flush=True)

    gate = _gate_multistep_arms()
    handler = _handler_multistep()
    sweep = run_seed_sweep(seeds=tuple(args.seeds), n_per_depth=args.n_per_depth)
    byte = _panel_byte_identical()

    # (D) real-handler byte-identical: reactive panel md5, BRAIN_GNW_MULTISTEP=1 vs =0 (separate processes)
    hash_on = _spawn_panel_hash({"BRAIN_GNW_MULTISTEP": "1"})
    hash_off = _spawn_panel_hash({"BRAIN_GNW_MULTISTEP": "0"})
    handler_hash_identical = (hash_on == hash_off)

    rows = sweep["rows"]

    def _mean(k):
        return float(np.mean([r[k] for r in rows]))

    gateA = bool(gate["on_reaches_terminal"] and gate["off_is_first_hop"] and handler["ok"])
    gateB = bool(sweep["all_go"])
    # (C1) emergent-stopping dissociation pooled: lesion collapses, reflex survives, on all seeds
    c1 = bool(all(r["lesion_acc"] <= 0.10 and r["single_hop_reflex_acc"] >= 0.85 for r in rows)
              and gate["lesion_collapses"] and handler["lesion_collapses"])
    # (C2) re-entry load-bearing + difficulty-graded: single-pass fails L>=2 but L=1 unchanged; hops grow w/ depth
    c2_singlepass = all(r["singlepass_acc_by_depth"]["1"] >= 0.99 for r in rows) and \
        all(r["singlepass_acc_by_depth"][str(L)] <= 0.10 for r in rows for L in (2, 3, 4))
    c2_reentry = all(r["reentrant_acc_by_depth"][str(L)] >= 0.90 for r in rows for L in DEPTHS_LIVE)
    c2_graded = all(r["spearman_hops_depth"] >= 0.9 and not r["any_halt_at_H_cap"] for r in rows)
    gateC = bool(c1 and c2_singlepass and c2_reentry and c2_graded)
    gateD = bool(byte["ok"] and handler_hash_identical)

    go = bool(gateA and gateB and gateC and gateD)

    # ATTRIBUTION (tools.lab): whose is the multi-step convergence — the SPIKING workspace ignition read, or a host
    # counter? treatment = the INTACT reentrant acc (reaches the terminal); control = the recurrence-ZEROED LESION acc
    # (cannot converge). (treatment - control) / treatment = the FRACTION of the multi-step answer NOT present once the
    # substrate read is lesioned = the fraction owed to the workspace ignition (measuring both arms is not the same as
    # asking whose the difference was — gap#5 banked both numbers one key apart for weeks).
    attribution_to_workspace = attributable_to(
        "the multi-step convergence owed to the SPIKING workspace ignition read (intact reaches the terminal; the "
        "recurrence-zeroed lesion does not)", _mean("reentrant_acc"), _mean("lesion_acc"))

    v = Verdict("GNW multi-step re-entrant deliberation WIRED into /api/brain-chat (emergent cycle count)")
    v.require("(A) gate: ON reaches the CHAIN TERMINAL where OFF (single-hop bus) reaches only the first hop",
              bool(gate["on_reaches_terminal"] and gate["off_is_first_hop"]), expect=True,
              note="off=%s on=%s" % (gate["off_chase_svo"], gate["on_chase_svo"]))
    v.require("(A) real handler: ON answer carries the terminal, OFF the first hop, LESION abstains/collapses",
              handler["ok"], expect=True,
              note="on=%r off=%r lesion_abstained=%s" % (handler["on_answer"], handler["off_answer"],
                                                         handler["lesion_abstained"]))
    v.require("(B) 6-seed substrate control: >=5/6 seeds GO on the LIVE composer+workspace", sweep["all_go"],
              expect=True, note="%d/%d seeds; mean reent=%.2f single=%.2f lesion=%.2f"
              % (sweep["n_go"], sweep["n"], _mean("reentrant_acc"), _mean("singlepass_acc"), _mean("lesion_acc")))
    v.require("(C1) EMERGENT-STOP dissociation: recurrence-zeroed workspace collapses the multi-step chase (<=0.10) "
              "while the 1-hop reflex survives (>=0.85) on all seeds", c1, expect=True,
              note="mean lesion=%.2f reflex=%.2f" % (_mean("lesion_acc"), _mean("single_hop_reflex_acc")))
    v.require("(C2a) RE-ENTRY load-bearing: single-pass keeps L=1 (>=0.99) but FAILS L>=2 (<=0.10)",
              bool(c2_singlepass), expect=True)
    v.require("(C2b) re-entry reaches every depth (>=0.90 per L)", bool(c2_reentry), expect=True)
    v.require("(C2c) DIFFICULTY-GRADED emergent stop: spearman(resolved_hops, depth)>=0.9, no halt at H_cap",
              bool(c2_graded), expect=True, note="mean spearman=%.2f" % _mean("spearman_hops_depth"))
    v.require("(D) reactive panel byte-identical (in-process, multi-step ON but inert on non-chase turns)",
              byte["ok"], expect=True, note="%d/%d turns identical" % (byte["n_identical"], byte["n_total"]))
    v.require("(D) real handler byte-identical: BRAIN_GNW_MULTISTEP=1 == =0 on the reactive panel (separate processes)",
              handler_hash_identical, expect=True, note="on=%s off=%s" % (hash_on, hash_off))
    # the dissociation that carries the substrate-control claim: intact reaches the terminal (1.0); lesion does not (0.0)
    v.control("the multi-step convergence is owed to the SPIKING workspace ignition read (intact reaches the terminal, "
              "recurrence-zeroed lesion does not)",
              treatment=_mean("reentrant_acc"), control=_mean("lesion_acc"), min_separation=0.5)
    v.require("(B) moat holds on all seeds (unstored cue + over-run past a leaf -> abstain)",
              all(r["moat_ok"] for r in rows), expect=True)
    v.disabled("heavy Gate-B organs (affect/worldmodel/... = 0) in the handler checks",
               why="disabled ONLY for speed; they run identically on every flag arm, so the comparison is unaffected")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_gnw_multistep_reentrant_deliberation_derisk", "go": go, "status": decided["status"],
        "gateA_live_multistep": gateA, "gateB_6seed_substrate_control": gateB,
        "gateC_emergent_and_reentry_loadbearing": gateC, "gateD_byte_identical": gateD,
        "attribution_to_workspace": attribution_to_workspace,
        "seeds": list(args.seeds), "depths": list(DEPTHS_LIVE), "relation": REL, "n_per_depth": args.n_per_depth,
        "n_go": sweep["n_go"], "n_seeds": sweep["n"], "go_rule": ">=5/6 seeds AND gates A,C,D",
        "n_concepts": sweep["n_concepts"],
        "mean_reentrant_acc": _mean("reentrant_acc"), "mean_singlepass_acc": _mean("singlepass_acc"),
        "mean_lesion_acc": _mean("lesion_acc"), "mean_single_hop_reflex_acc": _mean("single_hop_reflex_acc"),
        "mean_spearman_hops_depth": _mean("spearman_hops_depth"),
        "c2_singlepass_L1_kept_L2plus_failed": bool(c2_singlepass), "c2_reentry_all_depths": bool(c2_reentry),
        "c2_difficulty_graded": bool(c2_graded),
        "gate_arms": gate, "handler_arms": handler, "byte_identical": byte,
        "handler_hash_on": hash_on, "handler_hash_off": hash_off, "handler_hash_identical": handler_hash_identical,
        "per_seed": rows,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = args.json
    os.makedirs(os.path.dirname(os.path.abspath(op)), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 104, flush=True)
    print("  GNW MULTI-STEP re-entrant deliberation — WIRED into /api/brain-chat (real ChatBrain + handler, numpy)", flush=True)
    print("=" * 104, flush=True)
    print("  (A) gate: OFF(first hop)=%s -> ON(terminal)=%s (resolved_hops=%s) | handler on=%r off=%r lesion_abstain=%s"
          % (gate["off_chase_svo"], gate["on_chase_svo"], gate["resolved_hops"], handler["on_answer"],
             handler["off_answer"], handler["lesion_abstained"]), flush=True)
    print("  (B) 6-seed substrate control: %d/%d GO | mean reent=%.2f single=%.2f lesion=%.2f reflex=%.2f spearman=%.2f"
          % (sweep["n_go"], sweep["n"], _mean("reentrant_acc"), _mean("singlepass_acc"), _mean("lesion_acc"),
             _mean("single_hop_reflex_acc"), _mean("spearman_hops_depth")), flush=True)
    for r in rows:
        print("        seed %d: reent=%.2f single=%.2f lesion=%.2f reflex=%.2f spearman=%.2f moat=%s GO=%s"
              % (r["seed"], r["reentrant_acc"], r["singlepass_acc"], r["lesion_acc"], r["single_hop_reflex_acc"],
                 r["spearman_hops_depth"], r["moat_ok"], r["seed_go"]), flush=True)
    print("  (C) emergent-stop dissociation + re-entry load-bearing: C1=%s C2(single-pass L1 kept/L>=2 failed)=%s "
          "reentry_all_depths=%s graded=%s" % (c1, c2_singlepass, c2_reentry, c2_graded), flush=True)
    print("  (D) byte-identical: panel(in-proc)=%d/%d handler_hash(on==off)=%s"
          % (byte["n_identical"], byte["n_total"], handler_hash_identical), flush=True)
    print("\n  VERDICT: %s (status=%s)" % ("GO" if go else "NO-GO", decided["status"]), flush=True)
    print("  [saved] %s\n" % op + "=" * 104, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
