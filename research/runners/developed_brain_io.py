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

    # --- grounded codes -> a compact .npz (word -> phases[D]) ---
    np.savez_compressed(str(root / "grounded_codes.npz"),
                        **{w: np.asarray(ph, dtype=np.float32) for w, ph in codes.items()})

    # --- facts -> facts.json (the brain's accumulated knowledge) ---
    with open(root / "facts.json", "w", encoding="utf-8") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "facts": facts}, fh, indent=2, ensure_ascii=False)

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
        "vocab": list(vocab),
        "self_aliases": sorted(self_aliases) if self_aliases else None,
        "lineage_name": lineage_name,
        "files": {"codes": "grounded_codes.npz", "facts": "facts.json", "lineage": "lineage"},
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
        return {w: np.asarray(data[w], dtype=float) for w in data.files}


def _load_facts_json(path) -> list[dict]:
    p = Path(path) / "facts.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh).get("facts", [])


def _restore_facts(agent, facts):
    """Re-store the saved facts into the agent's composer (so composer.kb matches the developed state). Handles a
    clause patient (the tagged dict) by reconstructing a Clause. Uses the bound polarity tag when present."""
    inner = _inner_agent(agent)
    comp = inner.composer
    try:
        from research.runners.core_sim_composition import Clause
    except Exception:
        Clause = None
    for f in facts:
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
        comp.store(a, v, p, polarity=polarity)


def load_developed_brain(path, *, seed=None, use_multiturn=False, enable_neural_render=False,
                         referent_nouns=None, wm_n=600, wm_pattern_size=40, composer_kind=None,
                         grounded_codes_override=None):
    """Reconstruct the EXACT developed brain from a `save_developed_brain` bundle at `path`.

    Returns (agent, manifest). `agent` is a `BrainConversationalAgent` (or a `MultiTurnAgent` wrapper if
    `use_multiturn`), built over the saved vocab with the saved grounded codes, with every saved fact re-stored.

    Args:
        path: the developed-brain directory (must contain brain.json + grounded_codes.npz + facts.json).
        seed: override the saved seed (defaults to the manifest's seed -- keep it to reproduce ungrounded codes).
        use_multiturn: wrap in MultiTurnAgent (the persistent discourse-WM loop, for anaphora + multi-hop).
        enable_neural_render: the brain's own spiking serial-order renderer (slow; default OFF).
        referent_nouns: the WM-loop referent set for MultiTurnAgent (defaults to the saved vocab minus actions).
        composer_kind: override the manifest's composer_kind.
        grounded_codes_override: optional {word: phases} to override the saved codes (rare; e.g. a re-developed run).
    """
    manifest = _read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"no brain.json manifest at {path!r} -- not a developed-brain bundle")
    seed = int(manifest.get("seed", 42)) if seed is None else int(seed)
    composer_kind = composer_kind or manifest.get("composer_kind", "rf")
    vocab = list(manifest.get("vocab") or [])
    codes = dict(_load_codes_npz(path))
    if grounded_codes_override:
        codes.update({w: np.asarray(v, dtype=float) for w, v in grounded_codes_override.items()})
    facts = _load_facts_json(path)
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
        agent = MultiTurnAgent(referent_concepts=referent_nouns, concepts=concepts,
                               grounded_codes=codes if codes else None, seed=seed,
                               wm_n=wm_n, wm_pattern_size=wm_pattern_size,
                               enable_neural_render=enable_neural_render, composer_kind=composer_kind,
                               enable_biased_competition=False)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                         grounded_codes=codes if codes else None,
                                         composer_kind=composer_kind,
                                         enable_neural_render=enable_neural_render)
    _restore_facts(agent, facts)
    return agent, manifest


def is_developed_brain_bundle(path) -> bool:
    """True if `path` is a save_developed_brain bundle (has a brain.json manifest)."""
    return (Path(path) / "brain.json").exists()
