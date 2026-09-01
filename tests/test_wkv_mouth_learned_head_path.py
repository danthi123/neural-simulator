"""Regression pin for the WKV mouth crutch-burndown rung-1 path bug (2026-08-28 -> fixed `aa7c3a23c`).

WHAT BROKE, ONCE. `webapp/wkv_mouth_generator.py`'s `_LEARNED_HEAD_PATH_TEMPLATE` default used to point at
`research/findings/raw/_wkv_eprop_learned_head_seed{seed}.npz`, a location that never existed -- so setting
`BRAIN_WKV_MOUTH_LEARNED_HEAD=1` in production silently fell back to the NATIVE head on every call
(`_apply_learned_head`'s fail-safe: `reason="file_missing"`, `applied=False`, no exception raised). The bug
was invisible at the call site because the fallback is deliberately silent-safe -- exactly the shape that
needs a committed regression test, not just a one-time manual fix. Fixed in `aa7c3a23c` (path now points at
`research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.npz`, the 6/6-GO
per-seed-templated persisted heads, `sub_recov_ratio` mean 0.9273 min 0.8906,
`research/findings/2026-08-28-mouth-better-head-persist-6seed-GO-plus-wander-production-partial.md`).

These tests pin THREE properties so this class of regression fails loudly, mechanically, at commit time
instead of silently degrading a production flag to a no-op:
  1. the default learned-head path resolves to an EXISTING file for every one of the 6 non-negotiable seeds
     (42, 43, 44, 100, 101, 102) -- a wrong template shows up here as a missing file, immediately;
  2. the 6 per-seed artifacts are genuinely DISTINCT (not a repeat of the earlier un-templated-save bug where
     one file silently held only the last-processed seed's head under all 6 seeds' names);
  3. end-to-end through the module's own public/semi-public surface (`_get_readout` / `learned_head_status`),
     enabling the flag actually APPLIES the learned head (`applied=True`, `reason=None`) on every seed, and
     leaves the NATIVE head untouched (byte-identical-off) when the flag is off -- i.e. `generate()`'s
     PRODUCTION path genuinely swaps in the learned head end-to-end, not just "the path string looks right".

Full quality A/B (self-NLL vs native, 6 seeds, through `generate()` itself) lives in
`research/runners/_wkv_learned_vs_native_head_ab_6seed.py` / `research/findings/2026-09-01-wkv-mouth-learned-
head-6seed-ab-through-fixed-default-path-GO.md` -- this file is the CHEAP, ALWAYS-RUN regression pin; that
runner is the (slower, still CPU-only, ~15s) evidence-gathering companion.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

SEEDS = (42, 43, 44, 100, 101, 102)   # the repo's 6-seed non-negotiable (CLAUDE.md)

# The exact broken default this rung fixed -- if this string is ever seen again in the resolved path, the
# regression has returned.
_OLD_BROKEN_PATTERN = "_wkv_eprop_learned_head_seed"


@pytest.fixture()
def wkv_mouth_generator(monkeypatch):
    """Import fresh with the path-override env var explicitly UNSET, so the test exercises the module's own
    default template -- not whatever a prior process/test happened to leave in the environment."""
    monkeypatch.delenv("BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH", raising=False)
    import importlib
    from webapp import wkv_mouth_generator as mod
    importlib.reload(mod)          # re-evaluate the import-time-fixed _LEARNED_HEAD_PATH_TEMPLATE constant
    yield mod
    monkeypatch.delenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", raising=False)
    importlib.reload(mod)          # leave the module in its default state for any test that runs after


def _sha1_file(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


class TestDefaultPathResolvesForAllSixSeeds:
    """Property 1: no seed's default learned-head path is a dangling pointer."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_path_exists(self, wkv_mouth_generator, seed):
        path = Path(wkv_mouth_generator._learned_head_path(seed))
        assert path.exists(), (
            f"seed={seed}: default BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH template resolves to a MISSING file "
            f"({path}) -- this is the exact rung-1 regression: a wrong default silently degrades "
            f"BRAIN_WKV_MOUTH_LEARNED_HEAD=1 to a no-op via the fail-safe fallback."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_path_is_not_the_old_broken_location(self, wkv_mouth_generator, seed):
        path = str(wkv_mouth_generator._learned_head_path(seed))
        assert _OLD_BROKEN_PATTERN not in path, (
            f"seed={seed}: default path reverted to the pre-aa7c3a23c broken template ({path})"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_path_is_seed_templated_not_a_single_literal_file(self, wkv_mouth_generator, seed):
        path = str(wkv_mouth_generator._learned_head_path(seed))
        assert f"s{seed}.npz" in path, (
            f"seed={seed}: resolved path {path!r} does not carry this seed's own id -- looks like a "
            f"non-templated literal path (the exact shape of the earlier persist-side overwrite bug)."
        )


class TestSixSeedArtifactsAreGenuinelyDistinct:
    """Property 2: regression pin for the earlier un-templated `--save-w-hat` bug, where six seeds' worth of
    training silently collapsed onto ONE file (only the last-processed seed's head survived, under all six
    seeds' names). A future accidental revert of the persist-side fix would show up here as duplicate SHA1s."""

    def test_all_six_npz_files_are_byte_distinct(self, wkv_mouth_generator):
        hashes = {seed: _sha1_file(Path(wkv_mouth_generator._learned_head_path(seed))) for seed in SEEDS}
        assert len(set(hashes.values())) == len(SEEDS), (
            f"per-seed learned-head artifacts are NOT all distinct -- {hashes} -- this is the earlier "
            f"un-templated-save overwrite bug (one file's content silently reused under multiple seeds' names)."
        )


class TestLearnedHeadAppliesEndToEndThroughTheProductionLoader:
    """Property 3: the flag genuinely swaps the head in via the SAME code path `generate()` uses
    (`_get_readout` -> `_apply_learned_head`), for every one of the 6 seeds, with no silent fallback."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_learned_head_applies_cleanly(self, wkv_mouth_generator, monkeypatch, seed):
        monkeypatch.setenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", "1")
        ro, _vocab, _word_to_id = wkv_mouth_generator._get_readout(seed)
        status = wkv_mouth_generator.learned_head_status(seed)
        assert status is not None, f"seed={seed}: no learned-head status recorded at all"
        assert status["applied"] is True, (
            f"seed={seed}: learned head did NOT apply (reason={status.get('reason')!r}) -- the fail-safe "
            f"fired, meaning generate() would silently fall back to the native head despite the flag being on."
        )
        assert status["reason"] is None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_learned_head_differs_from_native_head(self, wkv_mouth_generator, monkeypatch, seed):
        """The lever must actually move: native and learned `head_w` must not be the same matrix."""
        monkeypatch.setenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", "0")
        ro_native, _, _ = wkv_mouth_generator._get_readout(seed)
        native_w = ro_native.head_w.copy()

        monkeypatch.setenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", "1")
        ro_learned, _, _ = wkv_mouth_generator._get_readout(seed)
        learned_w = ro_learned.head_w

        assert native_w.shape == learned_w.shape
        assert not (native_w == learned_w).all(), (
            f"seed={seed}: native and learned head_w are IDENTICAL -- the swap did not happen "
            f"(a void A/B: both arms would be the same arm)."
        )


class TestNativeHeadUnaffectedWhenFlagIsOff:
    """Byte-identical-off: with the flag unset/0, `generate()`'s read path must never touch the learned-head
    loader at all (no status recorded), preserving the pre-existing default-OFF behavior exactly."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_learned_head_status_when_flag_off(self, wkv_mouth_generator, monkeypatch, seed):
        monkeypatch.delenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", raising=False)
        wkv_mouth_generator._get_readout(seed)
        assert wkv_mouth_generator.learned_head_status(seed) is None, (
            f"seed={seed}: a learned-head status was recorded even though the flag is off -- "
            f"the default-OFF path is no longer byte-identical to before this flag existed."
        )
