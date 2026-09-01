"""VERIFY (board #94, the precise NEXT RUNG named by 2026-08-28-ltm-shard-elaboration-cupy-6seed-GO): with
`BRAIN_ELABORATE_FROM_LTM_SHARD=1` AND `BRAIN_CONFIDENCE_FORTHCOMING=1` together, through the REAL
`webapp.server.brain_chat` handler, on the TRUE un-overridden production floor (max_sentences=4,
max_elaborations=2 -- NO `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR` override), does a confident vs a genuinely-
uncertain turn now produce a DIFFERENT reply forthcomingness (sentence count), where the OLD (elaboration
buffer-tier-only) shape produced NONE?

WHY A CUSTOM FIXTURE, NOT THE SHIPPED 15k WIKIDATA CORE. The shipped core's machine-generated entity/relation
vocabulary (`bill_clinten`, `country_of_citizenship`, ...) does not route through the live NL question parser's
supported surface shapes (`_extract_route`/`_relation_fronted_route`/`_definitional_copula_route` -- checked
empirically, every natural phrasing tried abstained). This fixture reuses the EXACT construction the already-
merged, already-6-seed-cupy-GO `research/findings/raw/_ltm_shard_elab/verify_ltm_shard_elab.py` used (a small
conversational BUFFER + a `ShardedPhasorStore` LTM wrapped in the SAME production `TieredFactStore`), extended
here to go through the REAL end-to-end handler (metacog + apply_cap + hedge), with a REAL (not hardcoded)
confidence read, across 6 seeds.

COMPOSER KIND: `onebrain` (the TRUE `_COMPOSER_KIND_DEFAULT` production substrate in webapp/server.py), not the
`rf` toy composer the isolated LTM-shard verify used for speed. Empirically checked first (this dir's sibling
scratchpad probes): the `rf` composer's cleanup-cleanup "margin" field is a RAW cosine DIFFERENCE
(`top_raw - runner_raw`), structurally capped around ~0.35-0.48 for this fixture even at zero added noise and
D up to 512 -- it never crosses `ROLE_CONF_HI=0.50` (that band was calibrated in issue #181 against
OneBrainComposer's RATIO margin `(peak-runner_up)/peak`, which centers near 0.6 on a genuine clean recall, a
DIFFERENT scale). So a `confident=True` arm is structurally unreachable on `rf` regardless of fixture design --
not a fixture problem, a composer-margin-scale mismatch. Switching to `onebrain` reproduces the real production
scale (clean mrc=0.626 on this exact fixture, matching the 0.608 measured on the shipped tiny-demo in
2026-08-27-confidence-forthcomingness-retest-PARTIAL.md) and is also simply the correct choice: it is the
composer `webapp/server.py` actually ships by default.

FIXTURE (sized empirically, see `size_probe2.py` in this dir's sibling scratchpad -- deterministic across all
6 seeds, not stochastic):
  BUFFER: (brain, use, spikes), (spikes, carry, information)  -- 2 facts, forming a real 2-hop chain.
  LTM:    (spikes, travel, axon), (spikes, generate, current), (spikes, trigger, synapse),
          (spikes, require, threshold)  -- 4 facts, all AGENT-role under 'spikes' (the chain's end concept,
          which is also the elaboration TOPIC -- topic = the direct answer's patient when it is itself an
          agent of a further fact, i.e. exactly 'spikes' here).
  Q = "what does the brain use" -> direct fact (brain, use, spikes); chain extends one more hop via the
  buffer's own (spikes, carry, information); topic becomes 'spikes'.

Gathered-fact-count by condition (production floor = max_sentences 4 / max_elaborations 2; confidence-
forthcoming's reach = floor+1 = 5/3; see webapp/confidence_forthcoming_chat.py):
  flag OFF,      floor OR reach -> n=2 (chain only; elaboration never reaches the LTM -- THE OLD HOLLOW SHAPE)
  flag ON,       floor          -> n=4 (2 chain + 2 LTM-fed elaboration facts, exactly at the floor)
  flag ON,       reach          -> n=5 (2 chain + 3 LTM-fed elaboration facts -- EXCEEDS the floor by exactly
                                    EXTRA_SENTENCES=1, giving apply_cap real content to trim for the first time)

So: with the LTM-shard elaboration flag ON, a CONFIDENT turn (reach kept, n=5) and an UNCERTAIN turn (reach
truncated back to the floor, n=4) now differ -- the exact "hollow flip" shape from
2026-08-27-confidence-forthcomingness-retest-PARTIAL.md's residual 1 is gone. With the flag OFF (or with only
one of the two flags on), no difference is possible (n stays at the floor-or-below regardless of confidence),
reproducing that residual EXACTLY -- proving these two flags TOGETHER are what closes it, not either alone.

CONFIDENCE: a REAL read off the co-resident metacog organ (`_metacog_qualify` in webapp/server.py, reused
unchanged), not a hardcoded confident=True/False. CLEAN (undegraded) turn -> high confidence (unambiguous
single-candidate role decode). UNCERTAIN turn -> the SAME established synaptic-noise degradation
(`research.runners._emergent_graceful_degradation_derisk._noise`, the identical model
2026-08-27/2026-08-28's own confidence-discrimination findings used) perturbing `comp.buffer.store_conns`, a
per-seed bounded sigma-scan (0.3/0.6/0.9/1.2/1.5/2.0) for the FIRST sigma that (a) still answers (not abstain),
(b) still recovers the SAME direct fact (not a misrecall -- the moat), and (c) reads confidence BELOW the
metacog HIGH band -- exactly `_confidence_read_discrimination_derisk.py`'s own bounded-scan discipline (no
runaway sweep).

LESION (`BRAIN_METACOG_LESION=1`, the metacog organ's OWN load-bearing lesion, reused unchanged): repeats the
SAME clean + uncertain pair; if the 5-vs-4 difference collapses to 4-vs-4 (both read `confident=False`
unconditionally), the coupling rides the SPIKING confidence margin, not a host heuristic.

BYTE-IDENTICAL harness (checked per seed): (1) both flags OFF -> n=2, no `confidence_forthcoming` key, IDENTICAL
to a chat built with NO LTM tier at all (with_ltm=False) -- the strongest available proof the flag's absence is
a true no-op. (2) `BRAIN_ELABORATE_FROM_LTM_SHARD=1` alone (confidence-forthcoming OFF) -> n=4 always (floor),
no `confidence_forthcoming` key -- the elaboration fix alone, unconditioned by confidence. (3)
`BRAIN_CONFIDENCE_FORTHCOMING=1` alone (elaboration-from-LTM OFF) -> n=2 regardless of confidence -- reproduces
the ORIGINAL 2026-08-27 hollow finding exactly, proving BOTH flags are required together.

Moat: every gathered fact in every condition is checked against the known buffer+LTM fact set (no confabulation).
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
# isolate the coupling under test from every other faculty that could also change max_sentences/max_elaborations
# or the render path this turn (mirrors research/runners/_confidence_read_discrimination_derisk.py exactly).
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)          # metacog stays default-ON (the confidence read under test)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)   # NEVER override the floor -- the whole point is TRUE floor

import numpy as np                                                            # noqa: E402
import webapp.server as S                                                     # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore, build_ltm_from_facts   # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._emergent_graceful_degradation_derisk import _noise     # noqa: E402
from research.runners.metacog_production_organ import ROLE_CONF_HI            # noqa: E402

Q = "what does the brain use"
BUFFER_FACTS = [("brain", "use", "spikes"), ("spikes", "carry", "information")]
LTM_FACTS = [
    {"agent": "spikes", "action": "travel", "patient": "axon"},
    {"agent": "spikes", "action": "generate", "patient": "current"},
    {"agent": "spikes", "action": "trigger", "patient": "synapse"},
    {"agent": "spikes", "action": "require", "patient": "threshold"},
]
KNOWN_KEYS = {tuple(f) for f in BUFFER_FACTS} | {(f["agent"], f["action"], f["patient"]) for f in LTM_FACTS}
SIGMAS = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]
SEEDS = [42, 43, 44, 100, 101, 102]


def build_chat(seed, with_ltm=True):
    # a MINIMAL buffer-only vocab (LTM words are NOT merged into the buffer composer's cleanup codebook -- a
    # smaller competitive candidate pool gives a cleaner natural margin; the LTM store builds its OWN codebook
    # via build_ltm_from_facts's own vocab default, exactly per TieredFactStore's "tiers need not share a
    # codebook" contract).
    concepts = {w: None for w in sorted({w for f in BUFFER_FACTS for w in f})}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="onebrain",
                                     enable_neural_render=False)
    for a, v, p in BUFFER_FACTS:
        agent.hear(f"{a} {v} {p}", polarity="AFFIRM")
    if with_ltm:
        ltm = build_ltm_from_facts(list(LTM_FACTS), seed=seed, D=128)
        agent.composer = TieredFactStore(agent.composer, ltm)
    return ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())


_sid = [0]


def ask(chat, noised_conns=None, session_prefix="s"):
    """One real /api/brain-chat turn (in-process, the actual `S.brain_chat` handler), optionally with the
    composer's buffer store_conns swapped for a noised copy for the duration of the call (restored after,
    `finally`) -- the SAME degradation model + restore discipline as the established discrimination derisk."""
    comp = chat.inner.composer
    store_holder = comp.buffer if hasattr(comp, "buffer") else comp   # TieredFactStore wraps; a bare composer IS it
    base_conns = list(store_holder.store_conns)
    if noised_conns is not None:
        store_holder.store_conns = noised_conns
    _sid[0] += 1
    ck = (f"{session_prefix}{_sid[0]:04d}", "tiny-demo", "stub")
    S._BRAIN_CHATS[ck] = chat
    try:
        r = S.brain_chat(S.BrainChatRequest(session=f"{session_prefix}{_sid[0]:04d}", message=Q, brain="tiny-demo",
                                            reset=False, rich=True, renderer="stub"))
        return json.loads(bytes(r.body))
    finally:
        store_holder.store_conns = list(base_conns)


def moat_ok(d):
    facts = d.get("supporting_facts") or d.get("facts") or []
    return all(tuple(f) in KNOWN_KEYS for f in facts)


def run_seed(seed):
    out = {"seed": seed}

    # ---------------- (0) BYTE-IDENTICAL harness: 3 flag combinations + the no-LTM-tier reference ----------
    # Only TWO chats are built per seed (the composer's own state -- kb/store_conns -- never changes between
    # asks; only env-var flags + a temporarily-swapped store_conns copy vary), keeping the onebrain build cost
    # (the true production composer, ~90s/build) to 2x/seed instead of 6x/seed.
    chat_noltm = build_chat(seed, with_ltm=False)
    chat_shared = build_chat(seed, with_ltm=True)

    # 2026-09-01 (margin-scale recalibration session): BOTH flags flipped default-ON
    # (research/runners/rich_answer_composer.py _ELABORATE_FROM_LTM_DEFAULT_ON,
    # webapp/confidence_forthcoming_chat.py _CONFIDENCE_FORTHCOMING_DEFAULT_ON) once real-out-of-the-box-traffic
    # GO was reached on the shipped wikidata_core_15k. `os.environ.pop(...)` (UNSET) no longer means "off" under
    # this convention -- it now means the NEW default (ON). The "off" arms below MUST set the EXPLICIT `=0`
    # escape (documented byte-identical-to-pre-flip by both flags' own docstrings) instead of relying on unset,
    # or this harness tests the wrong condition. See
    # research/findings/2026-09-01-confidence-forthcomingness-margin-scale-recalibration.md.
    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "0"
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
    d_noltm = ask(chat_noltm, session_prefix=f"z{seed}n")

    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "0"
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
    d_off = ask(chat_shared, session_prefix=f"z{seed}o")

    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
    d_elab_only = ask(chat_shared, session_prefix=f"z{seed}e")

    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "0"
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
    d_cf_only = ask(chat_shared, session_prefix=f"z{seed}c")

    out["byte_identical"] = {
        "off_n_sentences": d_off.get("n_sentences"), "off_has_cf_key": ("confidence_forthcoming" in d_off),
        "noltm_n_sentences": d_noltm.get("n_sentences"),
        "off_matches_no_ltm_tier": (d_off.get("n_sentences") == d_noltm.get("n_sentences") == 2
                                    and d_off.get("answer") == d_noltm.get("answer")),
        "elab_only_n_sentences": d_elab_only.get("n_sentences"), "elab_only_has_cf_key": ("confidence_forthcoming" in d_elab_only),
        "elab_only_is_floor": (d_elab_only.get("n_sentences") == 4),
        "cf_only_n_sentences": d_cf_only.get("n_sentences"),
        "cf_only_reproduces_old_hollow": (d_cf_only.get("n_sentences") == 2),
    }
    out["byte_identical"]["ok"] = bool(
        out["byte_identical"]["off_n_sentences"] == 2 and not out["byte_identical"]["off_has_cf_key"]
        and out["byte_identical"]["off_matches_no_ltm_tier"]
        and out["byte_identical"]["elab_only_is_floor"] and not out["byte_identical"]["elab_only_has_cf_key"]
        and out["byte_identical"]["cf_only_reproduces_old_hollow"]
    )

    # ---------------- (1) LOAD-BEARING: both flags ON, no lesion -------------------------------------------
    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
    os.environ.pop("BRAIN_METACOG_LESION", None)

    chat_on = chat_shared      # same built chat; composer state (kb) is read-only across these asks
    comp = chat_on.inner.composer
    base_conns = list(comp.buffer.store_conns)

    d_clean = ask(chat_on, session_prefix=f"a{seed}c")
    clean = {
        "n_sentences": d_clean.get("n_sentences"), "abstained": d_clean.get("abstained"),
        "recalled_svo": d_clean.get("recalled_svo"),
        "mean_role_conf": ((d_clean.get("confidence_forthcoming") or {}).get("confident")),
        "cf_reason": (d_clean.get("confidence_forthcoming") or {}).get("reason"),
        "cf": d_clean.get("confidence_forthcoming"), "moat_ok": moat_ok(d_clean),
    }

    rng = np.random.default_rng(1000 + seed)
    uncertain = None
    sigma_used = None
    sigma_scan = []
    for sigma in SIGMAS:
        noised = _noise(base_conns, sigma, rng)
        d = ask(chat_on, noised_conns=noised, session_prefix=f"a{seed}u")
        cf = d.get("confidence_forthcoming") or {}
        row = {"sigma": sigma, "n_sentences": d.get("n_sentences"), "abstained": d.get("abstained"),
               "recalled_svo": d.get("recalled_svo"), "confident": cf.get("confident"), "reason": cf.get("reason")}
        sigma_scan.append(row)
        if (not d.get("abstained")) and d.get("recalled_svo") == ["brain", "use", "spikes"] and cf.get("confident") is False:
            uncertain = {"n_sentences": d.get("n_sentences"), "abstained": d.get("abstained"),
                        "recalled_svo": d.get("recalled_svo"), "cf_reason": cf.get("reason"), "cf": cf,
                        "moat_ok": moat_ok(d)}
            sigma_used = sigma
            break

    out["clean"] = clean
    out["uncertain"] = uncertain
    out["sigma_used"] = sigma_used
    out["sigma_scan"] = sigma_scan
    load_bearing = bool(
        uncertain is not None
        and clean.get("n_sentences") == 5 and clean.get("cf_reason") == "high_confidence"
        and uncertain.get("n_sentences") == 4 and uncertain.get("cf_reason") == "low_confidence_capped"
        and clean.get("n_sentences") > uncertain.get("n_sentences")
        and clean.get("moat_ok") and uncertain.get("moat_ok")
    )
    out["load_bearing_ok"] = load_bearing

    # ---------------- (2) LESION: same pair, BRAIN_METACOG_LESION=1 ----------------------------------------
    lesion_result = {"ok": False, "reason": "sigma_used is None -- no uncertain arm to lesion-compare"}
    if sigma_used is not None:
        os.environ["BRAIN_METACOG_LESION"] = "1"      # a global env read at call time (BRAIN_METACOG_LESION is
        chat_les = chat_shared                          # read fresh per turn by the process-shared metacog organ)
        comp_les = chat_les.inner.composer               # -- no separate chat/composer build needed for the lesion.
        base_conns_les = list(comp_les.buffer.store_conns)
        d_lclean = ask(chat_les, session_prefix=f"l{seed}c")
        rng2 = np.random.default_rng(1000 + seed)
        # advance the rng identically to reach the SAME noise draw as sigma_used (the sigma loop draws once per
        # sigma, in order) so the lesioned uncertain arm is noised IDENTICALLY to the intact one.
        noised_les = None
        for sigma in SIGMAS:
            noised_les = _noise(base_conns_les, sigma, rng2)
            if sigma == sigma_used:
                break
        d_lunc = ask(chat_les, noised_conns=noised_les, session_prefix=f"l{seed}u")
        os.environ.pop("BRAIN_METACOG_LESION", None)
        cf_lc = d_lclean.get("confidence_forthcoming") or {}
        cf_lu = d_lunc.get("confidence_forthcoming") or {}
        lesion_result = {
            "ok": bool(cf_lc.get("confident") is False and cf_lu.get("confident") is False
                      and d_lclean.get("n_sentences") == d_lunc.get("n_sentences")
                      and d_lclean.get("n_sentences") == 4),
            "lesioned_clean_n": d_lclean.get("n_sentences"), "lesioned_uncertain_n": d_lunc.get("n_sentences"),
            "lesioned_clean_confident": cf_lc.get("confident"), "lesioned_uncertain_confident": cf_lu.get("confident"),
            "lesioned_clean_reason": cf_lc.get("reason"), "lesioned_uncertain_reason": cf_lu.get("reason"),
        }
    out["lesion"] = lesion_result

    checks = {
        "byte_identical_off": out["byte_identical"]["ok"],
        "load_bearing": out["load_bearing_ok"],
        "lesion_reverts": out["lesion"]["ok"],
        "moat": bool(clean.get("moat_ok") and (uncertain is None or uncertain.get("moat_ok"))),
    }
    out["checks"] = checks
    out["GO"] = bool(all(checks.values()))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    dt = time.time() - t0

    check_names = list(per_seed[0]["checks"].keys())
    per_check_all_seeds = {c: all(r["checks"].get(c) for r in per_seed) for c in check_names}
    all_go = all(r["GO"] for r in per_seed)

    from tools.verdict import Verdict
    v = Verdict("confidence-forthcomingness is load-bearing on the TRUE production floor once elaboration "
               "reaches the routed LTM shard (BRAIN_ELABORATE_FROM_LTM_SHARD + BRAIN_CONFIDENCE_FORTHCOMING)")
    for c in check_names:
        v.require(c, per_check_all_seeds[c], expect=True, note=f"holds on all {len(SEEDS)} seeds")
    decided = v.decide(go=bool(all_go), verbose=False)

    out = {
        "probe": "confidence_forthcomingness_ltm_loadbearing",
        "flags": "BRAIN_ELABORATE_FROM_LTM_SHARD=1 + BRAIN_CONFIDENCE_FORTHCOMING=1, TRUE floor "
                "(no BRAIN_CONFIDENCE_FORTHCOMING_FLOOR override), through webapp.server.brain_chat in-process",
        "backend": os.environ.get("SIM_BACKEND"), "seeds": SEEDS, "n_seeds": len(SEEDS),
        "elapsed_s": dt, "GO": bool(all_go), "status": decided["status"],
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
        "per_check_all_seeds": per_check_all_seeds, "per_seed": per_seed,
    }
    out_path = os.path.join(_HERE, "verify_confidence_ltm_loadbearing.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("=" * 100)
    for c in check_names:
        print(f"  [{'PASS' if per_check_all_seeds[c] else 'FAIL'}] {c}  (all {len(SEEDS)} seeds)")
    print("=" * 100)
    for r in per_seed:
        print(f"  seed {r['seed']:>4}: GO={r['GO']}  clean_n={r['clean']['n_sentences']} "
              f"uncertain_n={(r['uncertain'] or {}).get('n_sentences')} sigma={r['sigma_used']} "
              f"lesion_ok={r['lesion']['ok']}")
    print("=" * 100)
    print(f"  VERDICT: {'GO' if all_go else 'NO-GO/UNDEFINED'}  (elapsed {dt:.1f}s, wrote "
          f"{os.path.relpath(out_path, _REPO)})")
    return 0 if all_go else 1


if __name__ == "__main__":
    sys.exit(main())
