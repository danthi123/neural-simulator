"""developed_brain_io — SAVE / LOAD a FULLY-developed conversational brain (the EXACT brain + all its knowledge).

The longitudinal develop loop (`_longitudinal_develop_loop{,_gpu}.py`) persists the developing brain's
`DevelopState` (facts / vocab / tier / day / per-day metrics) through `BridgeLineage` as a JSON payload, and the
self-knowledge demo (`_self_knowledge_demo.py`) separately saves the stream-LEARNED grounded codes to a
`_self_knowledge_grounded_codes.json` blob. But the brain's CONVERSATIONAL state is split across those two
artifacts, and the composer's stored facts (`composer.kb`) are NOT persisted as composer state -- they are
reconstructed by RE-TEACHING the curriculum. So nothing bundles {grounded codes + facts + vocab + seed} into one
artifact a downstream consumer (the chat TUI) can `--load` to reconstruct the EXACT developed brain with all its
knowledge, WITHOUT also needing the original curriculum JSON.

This module closes that gap. `save_developed_brain(agent, path)` writes a self-contained bundle:

    <path>/                         (a "developed brain" directory)
      lineage/                      a BridgeLineage (DevelopState payload, metadata, growth log) -- atomic, the
                                    project's standard persistent-state machinery (reused, not re-invented)
      grounded_codes.npz            the {word: phases[D]} the brain LEARNED FROM LISTENING (the composer's
                                    grounded concept codes) -- the exact codes the brain converses on
      facts.json                    the composer's stored SVO facts (agent/action/patient[/attribute/polarity]) --
                                    the brain's accumulated KNOWLEDGE, so a reload re-stores the SAME facts
      brain.json                    the manifest: seed, D, composer_kind, vocab, self_aliases, n_facts, provenance

`load_developed_brain(path)` reconstructs the EXACT brain: it rebuilds a `BrainConversationalAgent` (or a
`MultiTurnAgent` wrapper) over the saved vocab with the saved grounded codes (so every concept code is the one the
brain learned), then RE-STORES every saved fact (so `composer.kb` matches the developed state) -- byte-for-byte
the same composer codes + the same facts. The no-confab moat is intact by construction (only the saved facts are
stored; everything else abstains).

REUSE-BY-IMPORT, NO `sim/` edit. The codes round-trip is exact (the saved phases are loaded verbatim); the facts
round-trip is exact (re-stored through the same `composer.store`). The composer's per-seed RANDOM codes for any
word NOT in `grounded_codes` are reproduced deterministically from `seed` (so an ungrounded vocab word is
identical across save/load too).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sim.lineage import BridgeLineage


SCHEMA_VERSION = 1


# ============================================================================================================
# Extracting the developed brain's state from a live conversational agent.
# ============================================================================================================

def _inner_agent(agent):
    """Return the underlying BrainConversationalAgent whether `agent` is one directly OR a MultiTurnAgent wrapper
    (which holds it at `.agent`)."""
    return getattr(agent, "agent", agent)


def extract_grounded_codes(agent) -> dict[str, list]:
    """The composer's concept codes (the words the brain knows), as a {word: phases-list} dict. These ARE the
    grounded codes the brain converses on -- the codes it learned from listening (or the per-seed random codes for
    any ungrounded word; both round-trip exactly). Polarity tags (AFFIRM/NEGATE) and role codes are NOT concept
    codes and are reproduced from the seed on reload, so they are excluded."""
    comp = _inner_agent(agent).composer
    pol = set(getattr(comp, "pol_words", []))
    codes = {}
    for w, ph in comp.concepts.items():
        if w in pol:
            continue
        codes[w] = np.asarray(ph, dtype=float).tolist()
    return codes


def extract_facts(agent) -> list[dict]:
    """The composer's stored SVO facts as JSON-able dicts (agent/action/patient + optional attribute/attribute2/
    polarity). A clause-patient (a Clause / nested triple) is serialized structurally so it re-stores faithfully."""
    comp = _inner_agent(agent).composer
    out = []
    for fact, _handle in comp.kb:
        rec = {}
        for role in ("agent", "action", "patient", "attribute", "attribute2", "polarity"):
            if role in fact:
                v = fact[role]
                # a clause patient (Clause is a namedtuple (agent, action, patient)) -> a tagged dict
                if role == "patient" and not isinstance(v, str) and hasattr(v, "_fields"):
                    rec[role] = {"__clause__": True, "agent": v[0], "action": v[1], "patient": v[2]}
                else:
                    rec[role] = v
        out.append(rec)
    return out


def extract_vocab(agent) -> list[str]:
    """The full composer vocabulary (sorted), so a reload constructs the composer over the SAME word set (random
    codes for ungrounded words reproduce from the seed)."""
    comp = _inner_agent(agent).composer
    pol = set(getattr(comp, "pol_words", []))
    return sorted(w for w in comp.concepts.keys() if w not in pol)


def extract_speak_value_Q(agent) -> dict[str, float]:
    """The LEARNED per-topic TALKATIVENESS Q ({topic: float}) -- the communicable-brain's persistable
    speak-value (Stage B). Empty {} when the agent has no communicable orchestrator (communicable_mode OFF, or it
    was never built), so a non-communicable bundle is unchanged. The Q IS the talkativeness the brain LEARNED FROM
    INTERACTION (the develop-loop tie-in): persisting it carries the learned talkativeness across sessions."""
    fn = getattr(agent, "speak_value_Q", None)
    if callable(fn):
        try:
            return {str(t): float(q) for t, q in (fn() or {}).items()}
        except Exception:
            return {}
    return {}


def extract_kb_composites(agent) -> dict[str, np.ndarray]:
    """BRAIN-LOAD SPEEDUP (option 1): each stored fact's BOUND COMPOSITE PHASOR -- the `[D]` numpy array `composer.kb`
    already cached from `store()`'s `_encode`, keyed by the fact's index (string, aligned to `extract_facts` order).

    The composite IS the deterministic resonate output of `store()`, so persisting it lets a reload SKIP the ~832-step
    per-fact RF resonate (`_restore_facts` sets `composer.kb` directly instead of re-`store()`-ing). Only the rf/rate
    composer holds a numpy composite in kb; the onebrain composer holds the bound vector ON-SUBSTRATE (kb is
    `(fact, None)`), so a None handle is SKIPPED here -> that fact re-stores on load (the onebrain path is unchanged).
    Indices with a non-array handle are simply absent from the dict (a partial map round-trips: present -> loaded,
    absent -> re-stored)."""
    comp = _inner_agent(agent).composer
    out: dict[str, np.ndarray] = {}
    for i, (_fact, handle) in enumerate(comp.kb):
        # a numpy composite (rf/rate fast path: enable_substrate_store=False). A None (onebrain) or a substrate-store
        # bridge handle is NOT serializable here -> omit it (the fact will re-store on load). The composite is saved in
        # its NATIVE dtype (float64 on the rf path) so the load round-trip is BIT-EXACT -- a float32 down-cast would
        # perturb the phases by ~3e-8 (harmless for the cleanup argmax, but the anti-cheat demands a byte-identical
        # array, so we keep full precision; npz compresses the redundancy away).
        if isinstance(handle, np.ndarray):
            out[str(i)] = np.ascontiguousarray(handle)
    return out


def extract_onebrain_substrate(agent) -> dict[str, np.ndarray]:
    """BRAIN-LOAD SPEEDUP (option 1, ONEBRAIN): the onebrain analog of `extract_kb_composites` -- closes the gap that
    function's own docstring names (a None `kb` handle is skipped there -> every onebrain fact re-`store()`s, i.e.
    re-resonates, on every load; ~416 RF-resonate steps/fact via `_compose_phases`, board 2026-09-08 "onebrain
    re-resonate" blocker, `research/findings/2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-
    GO.md`).

    `OneBrainComposer` never caches the bound composite in `.kb` (kb is `(fact, None)`, the vector lives
    ON-SUBSTRATE) -- but the composite is NOT lost: `_write_block(i, zc)` (one_brain_composer.py:757-777) turns the
    `[D]` complex resonate output `zc` into `D` host-side connection tuples `(post, pre, g*zc[k])` appended
    block-major into `self.store_conns` (block i occupies `store_conns[i*D:(i+1)*D]`; `post`/`pre` are deterministic
    functions of `(i, D, store_base, block)` -- fixed once the composer is constructed with the same layout params,
    which `load_developed_brain` already reproduces for a reload). So the WEIGHT column of that slice (`w[k] = g *
    zc[k]`) is exactly the same deterministic-resonate-output payload `extract_kb_composites` persists for rf/rate,
    just addressed through `store_conns` instead of `kb[i][1]`.

    Returns {fact_index(str): w[D] complex128}, keyed like `extract_kb_composites` (aligned to `extract_facts`
    order). A no-op ({}) unless `comp` structurally looks like `OneBrainComposer` (has `store_conns` + `D`) -- an
    rf/rate bundle's facts stay on `extract_kb_composites` only, so the two never both fire for one bundle (kb
    holds numpy arrays there, never None). Indices whose `store_conns` slice is short (should not happen for a
    fact actually in `kb`, but guarded) are simply absent -- a partial map round-trips exactly like
    `extract_kb_composites`'s (present -> direct-set, absent -> re-`store()`)."""
    comp = _inner_agent(agent).composer
    D = getattr(comp, "D", None)
    store_conns = getattr(comp, "store_conns", None)
    if D is None or store_conns is None:
        return {}                                      # not onebrain-shaped (rf/rate has no store_conns) -> no-op
    out: dict[str, np.ndarray] = {}
    for i, (_fact, handle) in enumerate(comp.kb):
        if handle is not None:
            continue                                   # onebrain's own kb shape is always (fact, None); a numpy
                                                        # handle here means this is NOT the onebrain on-substrate
                                                        # case (already covered by extract_kb_composites) -> skip
        lo, hi = i * D, (i + 1) * D
        if hi <= len(store_conns):
            # store_conns is block-major: slice [lo:hi] is block i's D readout connections, each a
            # (post, pre, weight) tuple in k-order 0..D-1 (see _write_block) -> the weight column IS w[k].
            w = np.array([store_conns[lo + k][2] for k in range(D)], dtype=np.complex128)
            out[str(i)] = w
    return out


# ============================================================================================================
# SAVE.
# ============================================================================================================

def save_developed_brain(agent, path, *, seed=42, D=None, composer_kind="rf",
                         self_aliases=None, develop_state=None, lineage_name="developed_brain",
                         extra_metadata=None) -> dict:
    """Persist the EXACT developed brain (codes + facts + vocab + manifest + a BridgeLineage) to `path`.

    Args:
        agent: a `BrainConversationalAgent` or a `MultiTurnAgent` wrapper (the developed brain).
        path: the developed-brain DIRECTORY to write (created if absent).
        seed: the agent's seed (needed to reproduce ungrounded random codes on reload).
        D: the composer phasor dimension (read from the composer if None).
        composer_kind: 'rf' (default) / 'rate' / 'onebrain' -- how to rebuild on reload.
        self_aliases: optional set of self-reference words ('you'/'your'/'i'/'me'/'it' -> the brain) the TUI maps.
        develop_state: optional `DevelopState` (its payload is written into the lineage so the develop loop can
            RESUME from this bundle); None -> a minimal payload from the extracted facts/vocab.
        lineage_name: the BridgeLineage name under <path>/lineage.
        extra_metadata: optional dict merged into brain.json (provenance, curriculum name, develop config, ...).

    Returns the manifest dict that was written to <path>/brain.json.
    """
    comp = _inner_agent(agent).composer
    D = int(D if D is not None else getattr(comp, "D", len(next(iter(comp.concepts.values())))))
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    codes = extract_grounded_codes(agent)
    facts = extract_facts(agent)
    vocab = extract_vocab(agent)
    speak_value_Q = extract_speak_value_Q(agent)        # (Stage B) the learned talkativeness Q (empty if not communicable)

    # --- the learned talkativeness Q -> speak_value_Q.json (the communicable-brain's persistable speak-value).
    #     Written ONLY when non-empty, so a non-communicable bundle is byte-unchanged (the file is simply absent).
    #     The Q is a flat {topic: float}; a reload seeds the rebuilt agent's CommunicableTurn so the talkativeness
    #     learned from interaction carries across sessions (the develop-loop tie-in). ---
    if speak_value_Q:
        with open(root / "speak_value_Q.json", "w", encoding="utf-8") as fh:
            json.dump({"schema_version": SCHEMA_VERSION, "speak_value_Q": speak_value_Q}, fh, indent=2,
                      ensure_ascii=False)

    # --- grounded codes -> a compact .npz (word -> phases[D]) ---
    #     Keys are PREFIXED "g:" so a concept word can never collide with a numpy reserved kwarg of
    #     np.savez_compressed(file, *args, allow_pickle=..., **kwds) -- a concept literally named
    #     "file"/"allow_pickle" would otherwise raise "multiple values for argument 'file'" under
    #     numpy>=2. Concepts are [a-z]+ (the corpus tokenizer), so ":" never appears in a word.
    #     _load_codes_npz strips the prefix (and reads old, unprefixed bundles unchanged).
    np.savez_compressed(str(root / "grounded_codes.npz"),
                        **{f"g:{w}": np.asarray(ph, dtype=np.float32) for w, ph in codes.items()})

    # --- facts -> facts.json (the brain's accumulated knowledge) ---
    with open(root / "facts.json", "w", encoding="utf-8") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "facts": facts}, fh, indent=2, ensure_ascii=False)

    # --- (BRAIN-LOAD SPEEDUP, option 1) the bound composites -> kb_composites.npz ({fact_index -> comp[D]}) so a
    #     reload SKIPS the per-fact RF resonate (the composite IS the deterministic resonate output). Aligned to the
    #     facts.json order. Empty on the onebrain path (composites are on-substrate) -> the file is just absent. ---
    kb_composites = extract_kb_composites(agent)
    if kb_composites:
        np.savez_compressed(str(root / "kb_composites.npz"), **kb_composites)

    # --- (BRAIN-LOAD SPEEDUP, option 1, ONEBRAIN) the on-substrate composite weights -> onebrain_substrate.npz
    #     ({fact_index -> w[D] complex128}) -- the onebrain analog of kb_composites.npz above (see
    #     extract_onebrain_substrate's docstring): closes the gap that composer family left open (kb=(fact,None) ->
    #     nothing to persist there), so an onebrain bundle ALSO skips the per-fact resonate on reload. Empty on the
    #     rf/rate path (no store_conns) -> the file is just absent (byte-identical to before this existed). ---
    onebrain_substrate = extract_onebrain_substrate(agent)
    if onebrain_substrate:
        np.savez_compressed(str(root / "onebrain_substrate.npz"), **onebrain_substrate)

    # --- the BridgeLineage (DevelopState payload + metadata) -- the project's standard persistent-state machinery.
    #     If a DevelopState is supplied, persist its payload so the develop loop can RESUME from this bundle. ---
    lineage = BridgeLineage(lineage_name, root=root / "lineage")
    payload = (develop_state.to_payload() if develop_state is not None
               else {"seed": int(seed), "day": 0,
                     "facts": [[f.get("agent"), f.get("action"), f.get("patient")] for f in facts
                               if isinstance(f.get("patient"), str)],
                     "vocab": list(vocab), "current_tier": 4, "metrics": [], "t": len(facts)})

    def _save_fn(_unused, p):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    lineage.save(None, save_fn=_save_fn, tier=f"{payload.get('current_tier', 4)}-word",
                 arch={"kind": "developed_brain", "composer_kind": composer_kind, "D": D},
                 metadata_updates={"vocab": list(vocab),
                                   "cumulative_training_events": int(payload.get("t", len(facts)))},
                 snapshot=False)

    # --- the manifest ---
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "developed_brain",
        "seed": int(seed),
        "D": D,
        "composer_kind": composer_kind,
        "n_facts": len(facts),
        "n_grounded_codes": len(codes),
        "n_kb_composites": len(kb_composites),     # (option 1) persisted composites -> per-fact resonate skipped on load
        "n_onebrain_substrate": len(onebrain_substrate),  # (option 1, onebrain) ditto, via store_conns not kb
        "n_speak_value_Q": len(speak_value_Q),     # (Stage B) persisted learned-talkativeness Q entries (0 if not communicable)
        "vocab": list(vocab),
        "self_aliases": sorted(self_aliases) if self_aliases else None,
        "lineage_name": lineage_name,
        "files": {"codes": "grounded_codes.npz", "facts": "facts.json", "lineage": "lineage",
                  **({"kb_composites": "kb_composites.npz"} if kb_composites else {}),
                  **({"onebrain_substrate": "onebrain_substrate.npz"} if onebrain_substrate else {}),
                  **({"speak_value_Q": "speak_value_Q.json"} if speak_value_Q else {})},
    }
    if extra_metadata:
        manifest["metadata"] = extra_metadata
    with open(root / "brain.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


# ============================================================================================================
# LOAD.
# ============================================================================================================

def _read_manifest(path) -> dict | None:
    p = Path(path) / "brain.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _load_codes_npz(path) -> dict[str, np.ndarray]:
    p = Path(path) / "grounded_codes.npz"
    if not p.exists():
        return {}
    with np.load(str(p)) as data:
        # keys are "g:"-prefixed (see save_developed_brain); strip it. Old bundles saved raw
        # word keys (no prefix) -> fall back to the key as-is for backward compatibility.
        return {(k[2:] if k.startswith("g:") else k): np.asarray(data[k], dtype=float)
                for k in data.files}


def _load_facts_json(path) -> list[dict]:
    p = Path(path) / "facts.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh).get("facts", [])


def _load_speak_value_Q(path) -> dict[str, float]:
    """(Stage B) Load speak_value_Q.json -> {topic: float}. Absent file (a non-communicable bundle, or pre-Stage-B)
    -> {} (the rebuilt agent's talkativeness Q stays at its baseline)."""
    p = Path(path) / "speak_value_Q.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return {str(t): float(q) for t, q in json.load(fh).get("speak_value_Q", {}).items()}


def _load_kb_composites(path) -> dict[int, np.ndarray]:
    """(BRAIN-LOAD SPEEDUP, option 1) Load kb_composites.npz -> {fact_index(int) -> comp[D]}. Absent file (e.g. the
    onebrain path, or a pre-speedup bundle) -> {} (then _restore_facts re-stores every fact, the original behavior)."""
    p = Path(path) / "kb_composites.npz"
    if not p.exists():
        return {}
    with np.load(str(p)) as data:
        # keep the NATIVE saved dtype (float64 on the rf path) so the composite round-trips BIT-EXACT.
        return {int(k): np.array(data[k]) for k in data.files}


def _load_onebrain_substrate(path) -> dict[int, np.ndarray]:
    """(BRAIN-LOAD SPEEDUP, option 1, ONEBRAIN) Load onebrain_substrate.npz -> {fact_index(int) -> w[D] complex128}.
    Absent file (an rf/rate bundle, or a pre-speedup onebrain bundle) -> {} (then _restore_facts re-`store()`s every
    fact, the original -- and only -- onebrain behavior, byte-unchanged)."""
    p = Path(path) / "onebrain_substrate.npz"
    if not p.exists():
        return {}
    with np.load(str(p)) as data:
        # complex128 native dtype (matches store_conns' weight dtype) -> the direct-set restore is BIT-EXACT.
        return {int(k): np.asarray(data[k], dtype=np.complex128) for k in data.files}


def _store_fact_dict_from_operand(a, v, p, polarity):
    """Build the EXACT fact dict that `RFPhasorComposer.store(a, v, p, polarity)` appends -- WITHOUT the expensive
    `_encode` resonate. Mirrors the composer's `store` dict-build (rf_phasor_composer.store): a Clause patient stays a
    Clause; an `(adjs, noun)` attributed entity splits into patient=noun + attribute[/attribute2]; else patient=p; a
    polarity tag is added when present. Kept in lock-step with that store (a tiny, stable mapping); the round-trip
    validator asserts this dict == a real `store()`-built dict, so a drift is caught immediately. This is the only
    place option-1 reconstructs the dict instead of calling store (which would re-resonate)."""
    fact = {"agent": a, "action": v}
    _is_clause = getattr(p, "_fields", None) == ("agent", "action", "patient")
    if _is_clause:                                   # a recursive clause filler (check BEFORE tuple: a Clause IS a tuple)
        fact["patient"] = p
    elif isinstance(p, tuple):                        # (adj(s), noun) attributed entity
        adjs, noun = p
        adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
        fact["patient"] = noun
        fact["attribute"] = adjs[0]
        if len(adjs) > 1:
            fact["attribute2"] = adjs[1]
    else:
        fact["patient"] = p
    if polarity is not None:
        fact["polarity"] = polarity
    return fact


def _restore_facts(agent, facts, composites=None, onebrain_substrate=None, composer_kind_changed=False):
    """Re-store the saved facts into the agent's composer (so composer.kb matches the developed state). Handles a
    clause patient (the tagged dict) by reconstructing a Clause. Uses the bound polarity tag when present.

    BRAIN-LOAD SPEEDUP (option 1): when `composites` is a {fact_index -> comp[D]} map (from kb_composites.npz) AND the
    composer holds numpy composites in kb (the rf/rate fast path, `enable_substrate_store=False`), the fact's composite
    is set DIRECTLY (the dict via `_store_fact_dict_from_operand` + the persisted composite, appended to `composer.kb`)
    -- SKIPPING the ~832-step per-fact RF resonate `store()` would run. The composite IS the deterministic resonate
    output, so recall is byte-identical. A fact with no persisted composite (absent index, onebrain on-substrate, or a
    substrate-store composer) falls back to `comp.store()` (re-resonate), so the path is always correct -- the speedup
    is applied only where it is provably byte-identical.

    BRAIN-LOAD SPEEDUP (option 1, ONEBRAIN): when `onebrain_substrate` is a {fact_index -> w[D] complex128} map (from
    onebrain_substrate.npz, see `extract_onebrain_substrate`) AND the composer is onebrain-shaped (`store_conns` +
    `D`), the fact's block is set DIRECTLY via `comp._write_block(i, w)` -- SKIPPING `_compose_phases`'s ~416-step
    RF resonate. `_write_block` still runs (so its dirty-flag/cache-invalidation side effects -- `_store_dirty`,
    `_persistent_dirty`, `integrated_loop`'s `_seq_dirty`/`_fused_dirty` -- fire exactly as a real store would; only
    the expensive resonate that PRODUCES `w` is skipped). This is byte-identical ONLY when the reload composer's own
    `encoding_gain_fn is None` (the production default: `complex(1.0) * w[k] == w[k]` bit-for-bit, so `_write_block`
    re-applying a g=1.0 gain to an already-gain-baked `w` is an identity op) -- when a caller has wired a non-None
    `encoding_gain_fn` onto the reload composer (dynamic DA-gated writes), this path is skipped (falls to
    `comp.store()`) rather than risk double-applying / mismatching a historical gain. A fact with no persisted
    weight vector (absent index, or a non-onebrain composer) falls back to `comp.store()` unchanged.

    `composer_kind_changed`: True when the CALLER's resolved `composer_kind` differs from the bundle's OWN saved
    `manifest['composer_kind']` (an explicit cross-family override -- see `load_developed_brain`). ROOT-CAUSE FIX
    (2026-09-05, an adversarial skeptic on the L3 wire-in de-risk found this): a persisted `kb_composites.npz` is
    only a valid direct-set source for the composer FAMILY it was saved under. `hasattr(comp, "kb")` alone is NOT
    sufficient -- `CoreSimComposer` ('rate') ALSO has a `.kb` list, but each entry is a `(fact, (ON_array,
    OFF_array))` 2-tuple, structurally incompatible with `RFPhasorComposer`'s flat `[D]` composite array. Loading
    an 'rf'-saved bundle with `composer_kind` overridden to 'rate' would previously have hit
    `hasattr(comp,'kb')==True` and silently appended an RF composite where `CoreSimComposer` expects an
    `(ON,OFF)` pair -- no crash, just silently wrong drive currents on recall (worse than the SlotBinder
    AttributeError this same check was built to catch, because it fails silently instead of loudly).
    `composer_kind_changed=True` disables the fast path UNCONDITIONALLY (forces a full re-`store()`, always
    correct, only slower) whenever composer_kind was explicitly changed away from the bundle's own -- this is the
    general, family-agnostic guard; `hasattr(comp, "kb")` remains as defense-in-depth for the same-family case."""
    inner = _inner_agent(agent)
    comp = inner.composer
    composites = composites or {}
    onebrain_substrate = onebrain_substrate or {}
    # the fast direct-set path applies only when (a) composer_kind was NOT changed from what this bundle's own
    # composites were saved under (the root-cause guard above), AND (b) the composer caches a NUMPY composite in
    # kb (rf/rate, no substrate store) -- a substrate-store composer's handle is a bridge (not the composite), so
    # a direct set would corrupt it. `hasattr(comp, "kb")`: SlotBinderComposer has NO `.kb` list at all (facts
    # live in `.facts`, taught into per-slot synapses, not cached composites) -- without this, loading a bundle
    # with a persisted kb_composites.npz under a composer lacking `.kb` would hit `comp.kb.append(...)` below and
    # raise AttributeError. `enable_substrate_store` alone does not catch either case.
    can_direct = ((not composer_kind_changed) and hasattr(comp, "kb")
                 and not bool(getattr(comp, "enable_substrate_store", False)))
    # the onebrain direct-set path (see this function's ONEBRAIN docstring paragraph): the composer must be
    # onebrain-shaped (has `store_conns` + `D` + the private `_write_block` this reuses -- SlotBinderComposer/rf/
    # rate composers have none of these, so this is family-safe the same way `hasattr(comp,"kb")` guards the rf
    # path above) AND not a cross-family override AND its `encoding_gain_fn` must be the byte-identical None
    # default (the guard this function's docstring names -- a non-None gain-fn makes `_write_block`'s g-multiply
    # NOT an identity op on an already-gain-baked persisted weight).
    can_direct_onebrain = ((not composer_kind_changed) and hasattr(comp, "store_conns")
                          and hasattr(comp, "_write_block") and getattr(comp, "D", None) is not None
                          and getattr(comp, "encoding_gain_fn", None) is None)
    try:
        from research.runners.core_sim_composition import Clause
    except Exception:
        Clause = None
    for i, f in enumerate(facts):
        a, v = f.get("agent"), f.get("action")
        p = f.get("patient")
        polarity = f.get("polarity")
        # reconstruct a clause patient
        if isinstance(p, dict) and p.get("__clause__") and Clause is not None:
            p = Clause(p["agent"], p["action"], p["patient"])
        # an attributed-entity patient ((adjs, noun))
        attr = f.get("attribute")
        attr2 = f.get("attribute2")
        if isinstance(p, str) and attr is not None:
            adjs = [attr] + ([attr2] if attr2 is not None else [])
            p = (adjs, p)
        comp_arr = composites.get(i)
        ob_w = onebrain_substrate.get(i)
        if can_direct and comp_arr is not None:
            # DIRECT SET (skip the resonate): the persisted composite is store()'s deterministic _encode output, kept
            # in its native dtype so the kb array is BIT-EXACT to the re-resonated one.
            fact_dict = _store_fact_dict_from_operand(a, v, p, polarity)
            comp.kb.append((fact_dict, comp_arr))
        elif can_direct_onebrain and ob_w is not None:
            # DIRECT SET (skip the resonate), onebrain: `_write_block` reuses the composer's OWN dirty-flag/cache-
            # invalidation logic (see this function's docstring) -- `i` MUST be `len(comp.kb)` at call time (the
            # same invariant `_store_composite` upholds) so the block lands at the SAME store_conns slice / trigger
            # index the original store used; the kb append happens AFTER, exactly like store()'s own
            # `_store_fact` (resonate) -> `kb.append` order.
            fact_dict = _store_fact_dict_from_operand(a, v, p, polarity)
            comp._write_block(len(comp.kb), ob_w)
            comp.kb.append((fact_dict, None))
        else:
            comp.store(a, v, p, polarity=polarity)   # re-resonate (no persisted composite, or a substrate-store composer)


# (scaffold-retirement RANK-6/#211, FLIPPED DEFAULT-ON 2026-09-08 -- semantic-recall STORAGE shortcut retired).
# The cortical LTM now holds each fact's bound composite in the SUBSTRATE (per-fact trigger->readout complex
# synaptic weights, read back by FIRING -- RFPhasorComposer._store_substrate/_retrieve_substrate) instead of a
# numpy composite array cached in `shard.kb`, making the knowledge-core recall genuinely synaptic
# (memory-in-weights, the Crawford/Eliasmith weight store) and retiring the numpy-KB list host storage shortcut.
# Flipped after: function GO 6/6 + VRAM GO (~2.5 GiB at 78,857 facts, ~20 GiB free -- 2026-09-08-rank6-substrate-
# store-cupy-vram-measured-GO) + cupy batched-recall latency GO (~505ms < 1189ms numpy-kb baseline, the
# 2de6ab27a batched fix that turned the ~10s/query per-fact loop into O(1)). MUST run WITH the batched scan
# (_ltm_batched_substrate_scan_on below) -- substrate-store WITHOUT batched recall is the ~10s/query per-fact
# loop. Scope: retires the STORAGE shortcut ONLY -- the VSA exact-inverse BINDING is still host arithmetic (the
# synapse only HOLDS the composite), the store-WRITE is one-time host (not brain plasticity), and which facts
# exist is still host curation. Read INSIDE the loader (not an arg) so the exact production call
# load_developed_brain(ltm_bundle=...) picks it up with no signature change. BRAIN_LTM_SUBSTRATE_STORE=0 (or
# off/false/no/none) is the byte-identical rollback to the numpy-KB fast path.
def _ltm_substrate_store_on() -> bool:
    env = os.environ.get("BRAIN_LTM_SUBSTRATE_STORE")
    if env is None:
        return True
    return env.strip().lower() in ("1", "true", "on", "yes")


# (scaffold-retirement RANK-6/#211 companion, FLIPPED DEFAULT-ON 2026-09-08). The substrate-store recall scan is
# routed through the SAME batched block-diagonal resonate the numpy-kb fast path uses (ONE resonate retrieves all
# K composites, `_retrieve_all_substrate`), dropping the per-query resonate count O(K)->O(1) -- WITHOUT this the
# substrate store falls back to the per-fact loop (~10s/query at ~200 facts/shard). Answer-identical to the
# per-fact loop and still substrate-load-bearing (composites reconstructed by FIRING; lesioning the weights
# collapses the answer -- gated by tests/test_rf_phasor_composer.py). Kept as its OWN knob (mirrors the composer's
# own BRAIN_BATCHED_SUBSTRATE_SCAN env) so the two can be toggled independently for an A/B. Passed to the LTM
# composers ONLY when substrate store is on (it is a no-op on the numpy-kb path, `_can_batch_scan`).
# BRAIN_BATCHED_SUBSTRATE_SCAN=0 (or off/false/no/none) forces the per-fact loop back (the reversibility escape).
def _ltm_batched_substrate_scan_on() -> bool:
    env = os.environ.get("BRAIN_BATCHED_SUBSTRATE_SCAN")
    if env is None:
        return True
    return env.strip().lower() in ("1", "true", "on", "yes")


def load_developed_brain(path, *, seed=None, use_multiturn=False, enable_neural_render=False,
                         referent_nouns=None, wm_n=600, wm_pattern_size=40, composer_kind=None,
                         grounded_codes_override=None, defer_parser=True,
                         communicable_mode=False, communicable_draw="spiking",
                         ltm_bundle=None, ltm_n_shards=None, ltm_seed=None, ltm_D=128,
                         ltm_composer_kwargs=None, enable_codebook_cache=False,
                         enable_decode_escalation=False, decode_escalate_margin=None,
                         integrated_loop=False, onebrain_k_max=None):
    """Reconstruct the EXACT developed brain from a `save_developed_brain` bundle at `path`.

    Returns (agent, manifest). `agent` is a `BrainConversationalAgent` (or a `MultiTurnAgent` wrapper if
    `use_multiturn`), built over the saved vocab with the saved grounded codes, with every saved fact re-stored.

    BRAIN-LOAD SPEEDUP (default-ON for the load path -- three options): the per-fact RF resonate is SKIPPED when
    kb_composites.npz is present (option 1 -- the persisted composite is set directly into composer.kb; ONEBRAIN
    bundles get the same skip via a sibling sidecar, onebrain_substrate.npz -- see `extract_onebrain_substrate` /
    `_restore_facts`'s ONEBRAIN paragraph -- since onebrain's composite lives on-substrate in `store_conns`, not in
    `kb`, so it needed its own extractor; both are ABSENT/reload-unchanged on a bundle saved before this existed, or
    on the composer family the sidecar doesn't apply to); the comprehension parser's ~75K-step Hebbian training is
    DEFERRED (option 2 -- `defer_parser=True`: the parser builds lazily on the FIRST runtime teach); and (option 3,
    MultiTurnAgent only) the persistent discourse WORKING-MEMORY loop is DEFERRED (`defer_planner`, tied to
    `defer_parser` here) -- the WM loop's ~2*len(referents) attractor pathways into the merged ~10M-synapse bridge
    are the dominant load cost (~681s on the SK brain; see the load profile), and a pure Q&A / rich-answer session
    never introduces a multi-turn referent, so it never needs the WM loop. All three are byte-identical to the eager
    path for any Q&A. A loaded brain that then TEACHES a new fact pays the one-time parser training on the first
    teach; one that introduces a multi-turn referent pays the one-time WM build on the first referent write -- both
    identical to a never-deferred agent. Pass `defer_parser=False` to force the eager parser AND eager planner (e.g.
    if you want them warm immediately).

    Args:
        path: the developed-brain directory (must contain brain.json + grounded_codes.npz + facts.json; optionally
            kb_composites.npz from option 1).
        seed: override the saved seed (defaults to the manifest's seed -- keep it to reproduce ungrounded codes).
        use_multiturn: wrap in MultiTurnAgent (the persistent discourse-WM loop, for anaphora + multi-hop).
        enable_neural_render: the brain's own spiking serial-order renderer (slow; default OFF).
        referent_nouns: the WM-loop referent set for MultiTurnAgent (defaults to the saved vocab minus actions).
        composer_kind: override the manifest's composer_kind.
        grounded_codes_override: optional {word: phases} to override the saved codes (rare; e.g. a re-developed run).
        defer_parser: defer the comprehension-parser training to the first runtime teach (default True for the load
            path -- a loaded brain Q&As without ever needing the parser).
        integrated_loop (scaffold-retirement backlog rank-2, default False = byte-identical): pass-through to
            MultiTurnAgent/BrainConversationalAgent -> OneBrainComposer's spiking K-way cue-match SEQUENCER
            (replaces the host first-match `_scan` SELECTION with the validated spiking decision, GO 4/4 at the
            production V=320 tier -- 2026-06-21-shortcut3-fold-integrated-loop-BUILD.md). A no-op unless the
            (possibly-overridden) `composer_kind` resolves to 'onebrain' -- every other composer_kind (including
            this bundle's own saved default, 'rf') ignores it exactly like `BrainConversationalAgent.__init__`
            does. This is the seam webapp/server.py's BRAIN_INTEGRATED_LOOP env flag threads through
            (_build_chat_brain -> here -> MultiTurnAgent/BrainConversationalAgent -> OneBrainComposer).
        onebrain_k_max (scaffold-retirement backlog rank-1 load-path thread, 2026-09-08, default None): the number
            of co-resident fact blocks the OneBrainComposer sizes its store for (see
            `BrainConversationalAgent.__init__`'s own `onebrain_k_max` doc). A no-op unless the (possibly-
            overridden) `composer_kind` resolves to 'onebrain' -- every other composer_kind (including this
            bundle's own saved default, 'rf') ignores it, exactly like `BrainConversationalAgent.__init__` does,
            so an 'rf' bundle reload is BYTE-IDENTICAL to before this parameter existed. When `None` AND the
            resolved composer_kind is 'onebrain', it is derived from THIS bundle's own fact count
            (`len(facts) + 16`, matching `_rank1_composer_rebuild_onebrain_verify.py`'s convention) instead of
            falling through to the composer's hardcoded default of 32 -- the fix for the rank-1 GO finding's named
            blocker (research/findings/2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-
            GO.md): an onebrain bundle with >32 facts previously crashed on reload with "OneBrainComposer store
            full: k_max=32 reached" (or, with a persisted kb_composites fast-path, silently direct-set only the
            first 32 blocks) because NEITHER this function NOR `MultiTurnAgent` threaded the caller's sizing
            through at all -- the constructor's own None-default (32) was the only value that ever reached the
            composer. Pass an explicit int to override the auto-sizing (e.g. to pre-provision headroom for
            facts taught after load).
    """
    manifest = _read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"no brain.json manifest at {path!r} -- not a developed-brain bundle")
    seed = int(manifest.get("seed", 42)) if seed is None else int(seed)
    _manifest_composer_kind = manifest.get("composer_kind", "rf")
    # ROOT-CAUSE GUARD (2026-09-05, adversarial skeptic finding on the L3 wire-in de-risk): a persisted
    # kb_composites.npz is only valid for the composer FAMILY it was saved under (see _restore_facts's own
    # docstring -- CoreSimComposer's `.kb` format is structurally incompatible with RFPhasorComposer's, and
    # hasattr(comp,'kb') alone cannot tell them apart). Record whether this call is asking to load under a
    # DIFFERENT composer_kind than the bundle's own, BEFORE resolving the `or` default below, so _restore_facts
    # can unconditionally skip the composite fast-path on any such cross-family override.
    _composer_kind_changed = bool(composer_kind) and composer_kind != _manifest_composer_kind
    composer_kind = composer_kind or _manifest_composer_kind
    vocab = list(manifest.get("vocab") or [])
    codes = dict(_load_codes_npz(path))
    if grounded_codes_override:
        codes.update({w: np.asarray(v, dtype=float) for w, v in grounded_codes_override.items()})
    facts = _load_facts_json(path)
    # L3 wire-in de-risk (2026-09-05, research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-
    # dominates-latency.md): when the
    # (possibly-overridden) composer_kind resolves to 'slotbinder', size + prewire it from THIS bundle's own
    # facts -- the batch-consolidation scenario slotbinder_composer.py's docstring names (an already-known corpus
    # migrating off FHRR), not a per-query lookahead. `BRAIN_SLOTBINDER_FANOUT` (default 32, the de-risked
    # production recommendation -- research/findings/2026-09-04-slotbinder-L2-sparse-fanout-derisk-GO-fits-3090-
    # and-composes.md) lets a caller tune/disable (0 or >=KF -> dense) without a code change. Embedded-clause
    # facts cannot be wiring-time pre-registered (SlotBinderComposer._required_fillers_from_prewire raises on
    # them by design -- a wrong required-filler set would defeat fanout's guarantee), so a bundle containing any
    # falls back to BLIND sparsification (prewire_facts=None) rather than crashing; day_33 (this task's own
    # de-risk target) is 100% flat SVO, so this fallback is untested live but kept for correctness on any other
    # bundle. Every slotbinder_* value is None (or the composer's own default) unless composer_kind=='slotbinder'
    # -- byte-identical to before for every other composer_kind.
    _slotbinder_kwargs = {}
    if composer_kind == "slotbinder":
        _sb_fanout_env = os.environ.get("BRAIN_SLOTBINDER_FANOUT", "32")
        _sb_fanout = None if _sb_fanout_env in ("", "none", "None", "dense") else int(_sb_fanout_env)
        _sb_has_clause = any(isinstance(f.get("patient"), dict) and f["patient"].get("__clause__")
                             for f in facts)
        _slotbinder_kwargs = dict(
            slotbinder_fanout=_sb_fanout,
            slotbinder_max_facts=max(len(facts), 1),
            slotbinder_prewire_facts=(None if _sb_has_clause else list(facts)),
        )
    # rank-1 composer-rebuild load-path thread (2026-09-08): size the OneBrainComposer's k_max from THIS bundle's
    # own fact count when the (possibly-overridden) composer_kind resolves to 'onebrain' and the caller did not
    # pass an explicit override -- otherwise `BrainConversationalAgent`/`MultiTurnAgent`'s own None-default (32,
    # the composer's original hardcoded cap) is the only value that ever reaches the composer, and a >32-fact
    # onebrain bundle crashes on reload ("OneBrainComposer store full: k_max=32 reached") instead of restoring
    # every saved fact. `+16` matches `_rank1_composer_rebuild_onebrain_verify.py`'s own sizing convention (a
    # small headroom margin over the exact fact count). BYTE-IDENTICAL for every other composer_kind (including
    # this bundle's own saved default, 'rf') -- onebrain_k_max stays None and is never read outside the onebrain
    # branch, exactly like every other onebrain-only kwarg here (slotbinder_*, integrated_loop).
    if composer_kind == "onebrain" and onebrain_k_max is None:
        onebrain_k_max = len(facts) + 16
    composites = _load_kb_composites(path)   # (option 1) {fact_index -> comp[D]} -> skip the per-fact resonate
    onebrain_substrate = _load_onebrain_substrate(path)  # (option 1, onebrain) ditto, via store_conns not kb
    speak_value_Q = _load_speak_value_Q(path)   # (Stage B) the persisted learned-talkativeness Q (seeds CommunicableTurn)
    # the vocab must cover every grounded code + every fact word (so the composer can encode them)
    vocab_set = set(vocab) | set(codes.keys())
    for f in facts:
        for role in ("agent", "action", "patient", "attribute", "attribute2"):
            w = f.get(role)
            if isinstance(w, str):
                vocab_set.add(w)
            elif isinstance(w, dict) and w.get("__clause__"):
                vocab_set.update(x for x in (w.get("agent"), w.get("action"), w.get("patient"))
                                 if isinstance(x, str))
    vocab = sorted(vocab_set)

    from research.runners.brain_conversational_agent import BrainConversationalAgent
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        if referent_nouns is None:
            actions = {f.get("action") for f in facts if isinstance(f.get("action"), str)} | {"is"}
            referent_nouns = [w for w in vocab if w not in actions]
        # grow the WM loop so it can hold every referent (the buffer holds wm_n/wm_pattern_size patterns); a large
        # developed vocabulary must not overrun the pattern budget (same rule as the develop loop's build_agent).
        wm_n = max(int(wm_n), 2 * int(wm_pattern_size) * max(1, len(referent_nouns)))
        # --- SELECTIVE ATTENTION / biased-competition wire-in (env-gated, default OFF = byte-identical) ---------
        # This is the WEBAPP production loader (webapp/server.py:_build_chat_brain -> load_developed_brain), so the
        # env flag reaches the live chat brain here. BRAIN_BIASED_COMPETITION unset/"0" -> enabled()==False == today.
        from research.runners.biased_competition_prod import biased_competition_enabled as _bc_enabled
        agent = MultiTurnAgent(referent_concepts=referent_nouns, concepts=concepts,
                               grounded_codes=codes if codes else None, seed=seed,
                               wm_n=wm_n, wm_pattern_size=wm_pattern_size,
                               enable_neural_render=enable_neural_render, composer_kind=composer_kind,
                               integrated_loop=bool(integrated_loop),
                               enable_biased_competition=_bc_enabled(), defer_parser=defer_parser,
                               defer_planner=defer_parser,
                               communicable_mode=communicable_mode, communicable_draw=communicable_draw,
                               speak_value_Q=(speak_value_Q or None), onebrain_k_max=onebrain_k_max,
                               **_slotbinder_kwargs)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                         grounded_codes=codes if codes else None,
                                         composer_kind=composer_kind,
                                         integrated_loop=bool(integrated_loop),
                                         enable_neural_render=enable_neural_render,
                                         defer_parser=defer_parser,
                                         communicable_mode=communicable_mode, communicable_draw=communicable_draw,
                                         speak_value_Q=(speak_value_Q or None), onebrain_k_max=onebrain_k_max,
                                         **_slotbinder_kwargs)
    _restore_facts(agent, facts, composites=composites, onebrain_substrate=onebrain_substrate,
                   composer_kind_changed=_composer_kind_changed)

    # (KNOWLEDGE-SCALE, opt-in, DEFAULT-OFF = byte-identical) install a cortical LONG-TERM store so the brain can
    # hold + query bulk KNOWLEDGE (100k-1M facts) beyond the small conversation working-set (the k_max=32 co-resident
    # cap). `ltm_bundle` is a path to a SEPARATE bundle (or any dir with a facts.json) whose facts become a routed
    # ShardedPhasorStore. The developed brain's own composer stays the recent-conversation BUFFER; a read checks the
    # buffer, then the routed LTM shard (sub-second at any K). No LTM -> the plain flat composer, byte-for-byte. The
    # tiers self-consistently encode by WORD, so a separate LTM codebook is correct (a fact is read from the tier it
    # was stored in). See tiered_fact_store.py + the sharded-fact-store finding (2026-08-20).
    if ltm_bundle is not None:
        from research.runners.tiered_fact_store import (TieredFactStore, build_ltm_from_facts, auto_n_shards)
        from research.runners.sharded_phasor_store import ShardedPhasorStore
        ltm = None
        # FAST PATH: a persisted sharded store (build ONCE offline, reload in seconds — no per-fact resonate).
        if (Path(ltm_bundle) / "manifest.json").exists():
            try:
                mani = json.load(open(Path(ltm_bundle) / "manifest.json"))
            except Exception:
                mani = {}
            if isinstance(mani, dict) and "n_shards" in mani:
                ltm_kwargs = {}
                if _ltm_substrate_store_on():
                    # substrate-store default-ON (see _ltm_substrate_store_on above): ShardedPhasorStore.load
                    # re-imprints each fact via _store_substrate, so the LTM shards hold synaptic handles read by
                    # firing (memory-in-weights) rather than numpy composites cached in kb.
                    ltm_kwargs["enable_substrate_store"] = True
                    if _ltm_batched_substrate_scan_on():
                        # keep recall O(1) (the ~505ms batched path) instead of the ~10s/query per-fact loop.
                        ltm_kwargs["enable_batched_substrate_scan"] = True
                if enable_codebook_cache:
                    ltm_kwargs["enable_codebook_cache"] = True
                if enable_decode_escalation:
                    ltm_kwargs["enable_decode_escalation"] = True
                    # (#108 R1) optional ESCAPE/rollback override of the tightened default (0.008); None keeps the
                    # composer default. Reachable so an A/B or a rollback to the old 0.02 gate is a one-arg change.
                    if decode_escalate_margin is not None:
                        ltm_kwargs["decode_escalate_margin"] = float(decode_escalate_margin)
                ltm = ShardedPhasorStore.load(str(ltm_bundle), extra_kwargs=ltm_kwargs or None)
        # BUILD PATH: a facts bundle -> build (+ resonate) the sharded LTM.
        if ltm is None:
            ltm_facts = _load_facts_json(ltm_bundle)
            if ltm_facts:
                ns = int(ltm_n_shards) if ltm_n_shards is not None else auto_n_shards(len(ltm_facts))
                cb_kwargs = dict(ltm_composer_kwargs or {})
                if _ltm_substrate_store_on():
                    # substrate-store default-ON: build_ltm_from_facts here defaults fast=False, so store() runs
                    # per-fact and _store_substrate holds each composite in synaptic weights.
                    cb_kwargs["enable_substrate_store"] = True
                    if _ltm_batched_substrate_scan_on():
                        cb_kwargs["enable_batched_substrate_scan"] = True
                if enable_codebook_cache:
                    cb_kwargs["enable_codebook_cache"] = True
                if enable_decode_escalation:
                    cb_kwargs["enable_decode_escalation"] = True
                    if decode_escalate_margin is not None:   # (#108 R1) escape override, see fast path above
                        cb_kwargs["decode_escalate_margin"] = float(decode_escalate_margin)
                ltm = build_ltm_from_facts(ltm_facts, n_shards=ns,
                                           seed=int(ltm_seed) if ltm_seed is not None else seed, D=int(ltm_D),
                                           composer_kwargs=cb_kwargs)
        if ltm is not None:
            inner = _inner_agent(agent)
            inner.composer = TieredFactStore(inner.composer, ltm)
    return agent, manifest


def is_developed_brain_bundle(path) -> bool:
    """True if `path` is a save_developed_brain bundle (has a brain.json manifest)."""
    return (Path(path) / "brain.json").exists()
