# dev_seed7/ is NOT EVIDENCE (2026-09-24 review)

Every JSON under this directory (`uniform_nrel2000/`, `hub_nrel128/`) was produced by dev seed 7 on pool42
**before** `2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md` was written, to check the
design could answer its own questions (see that prereg's "Seen before registration: dev seed 7" section, which
discloses every design choice made after seeing these numbers).

They were then **hand-edited after the run** — the `backend`, `runner` and `provenance_note` fields were added
manually — and they have **no `.prov.json` sidecars**. `research/runners/__init__.py`'s provenance wrapper never
ran on them.

Consequences, binding on every future reader:

- Seed 7 is **not** in `SEEDS = (42, 43, 44, 100, 101, 102)`; the CLI refuses it without `--allow-dev-seed`.
- These files may inform **design choices** (already disclosed in the prereg) but **must never be cited as a
  result, a measurement, or evidence for any gate** (G1-G9) or for the capacity-law fit / GPU extrapolation in
  `aggregate()`. `aggregate()` only ever reads `<arm>_s<seed>.json` for `seed in SEEDS`, so it cannot pick these
  up by construction — but a human or agent writing a finding must not quote numbers from this directory as if
  they were a seed-42..102 measurement.
- The `sparse_dg_recx2` arm (added 2026-09-24 to isolate the recurrent-synapse capacity-law attribution) has
  **no dev-seed-7 run at all**: its predicted effect is a real, untested prediction, not something seen before
  registration.
