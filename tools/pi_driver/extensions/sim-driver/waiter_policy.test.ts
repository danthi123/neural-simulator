// node --test tools/pi_driver/extensions/sim-driver/waiter_policy.test.ts   (Node >= 23 strips the types itself)
import { test } from "node:test";
import assert from "node:assert/strict";
import { type Absence, GPU_TAG, capTimeoutHours, doneLogged, isStale, matchWaiters, nextAbsence, parseHolds, runExclusive, MAX_TIMEOUT_HOURS, STALE_MIN_MS } from "./waiter_policy.ts";

const ws = [
  { id: "mk1-ab12", label: "s2 cells 0.4B", kind: "gpu" as const, target: "tag-cells-0p4b" },
  { id: "mk2-cd34", label: "pool results", kind: "file" as const, target: "research/findings/raw/x.json" },
  { id: "mk3-ef56", label: "S2 cells 1.5B", kind: "gpu" as const, target: "tag-cells-1p5b" },
];

test("timeout is capped at 8 hours and defaults sanely", () => {
  assert.equal(MAX_TIMEOUT_HOURS, 8);
  assert.equal(capTimeoutHours(48), 8);
  assert.equal(capTimeoutHours(2.5), 2.5);
  assert.equal(capTimeoutHours(undefined), 8);
  assert.equal(capTimeoutHours(0), 8);
  assert.equal(capTimeoutHours(-3), 8);
  assert.equal(capTimeoutHours(Number.NaN), 8);
});

test("cancel matching: exact id, then label or target substring", () => {
  assert.deepEqual(matchWaiters(ws, "mk2-cd34").map((w) => w.id), ["mk2-cd34"]);
  assert.deepEqual(matchWaiters(ws, "S2 CELLS").map((w) => w.id), ["mk1-ab12", "mk3-ef56"]);
  assert.deepEqual(matchWaiters(ws, "cells-1p5b").map((w) => w.id), ["mk3-ef56"]);
  assert.deepEqual(matchWaiters(ws, "nothing"), []);
  assert.deepEqual(matchWaiters(ws, "  "), []);
});

test("gpu tag alphabet", () => {
  assert.ok(GPU_TAG.test("s2-cells_0.4b/x=1"));
  assert.ok(!GPU_TAG.test("a;b"));
  assert.ok(!GPU_TAG.test("a b"));
  assert.ok(!GPU_TAG.test(""));
});

test("holds parsing: only a clean exit with a holds line counts", () => {
  assert.equal(parseHolds(0, "holds: ABSENT\n"), "ABSENT");
  assert.equal(parseHolds(0, "noise\nholds: RUNNING\n"), "RUNNING");
  assert.equal(parseHolds(0, "holds: QUEUED"), "QUEUED");
  assert.equal(parseHolds(255, "holds: ABSENT\n"), "UNKNOWN");
  assert.equal(parseHolds(0, "ssh: connection refused"), "UNKNOWN");
});

const MIN = 60_000;
const T0 = Date.parse("2026-11-03T00:00:00Z");

test("quick ABSENT polls are not stale: the job may be mid-dispatch or awaiting the 15-minute result sync", () => {
  let a = nextAbsence(undefined, "ABSENT", false, false, T0);
  a = nextAbsence(a, "ABSENT", false, false, T0 + 2 * MIN);
  a = nextAbsence(a, "ABSENT", false, false, T0 + 15 * MIN); // one full sync interval: still not enough
  assert.equal(a?.polls, 3);
  assert.equal(isStale(a, T0 + 15 * MIN), false);
  a = nextAbsence(a, "ABSENT", false, false, T0 + 20 * MIN);
  assert.equal(STALE_MIN_MS, 20 * MIN);
  assert.equal(isStale(a, T0 + 20 * MIN), true);
});

test("a single ABSENT poll is never stale, however long ago it was", () => {
  const a = nextAbsence(undefined, "ABSENT", false, false, T0);
  assert.equal(isStale(a, T0 + 120 * MIN), false);
});

test("the ABSENT run resets on any other evidence, and its clock restarts", () => {
  const absent = (p: Absence | undefined, t: number) => nextAbsence(p, "ABSENT", false, false, t);
  let a = absent(undefined, T0);
  a = absent(a, T0 + 10 * MIN);
  assert.equal(nextAbsence(a, "QUEUED", false, false, T0 + 12 * MIN), undefined); // popped then seen again
  assert.equal(nextAbsence(a, "RUNNING", false, false, T0 + 12 * MIN), undefined);
  assert.equal(nextAbsence(a, "UNKNOWN", false, false, T0 + 12 * MIN), undefined); // ssh hiccup never closes
  assert.equal(nextAbsence(a, "ABSENT", true, false, T0 + 12 * MIN), undefined); // result file synced
  // after a reset the 20 minutes count from the new first ABSENT
  let b = absent(undefined, T0 + 12 * MIN);
  b = absent(b, T0 + 25 * MIN);
  assert.equal(isStale(b, T0 + 25 * MIN), false);
  b = absent(b, T0 + 32 * MIN);
  assert.equal(isStale(b, T0 + 32 * MIN), true);
});

test("a DONE line for the tag means finished, never stale; an unreadable log means unknown", () => {
  const log =
    "2026-11-03 01:00:00 START: cd /w && python -m x --out raw/tag-a.json\n" +
    "2026-11-03 01:40:00 DONE(rc=0): cd /w && python -m x --out raw/tag-a.json\nnow: 1 x\n";
  assert.equal(doneLogged(0, log, "tag-a"), true);
  assert.equal(doneLogged(0, log, "tag-b"), false);
  assert.equal(doneLogged(0, "2026-11-03 01:40:00 DONE(rc=-9): x tag-k\nnow: 1 x\n", "tag-k"), true); // killed is DONE too
  assert.equal(doneLogged(0, "2026-11-03 01:00:00 START: tag-a\nnow: 1 x\n", "tag-a"), false); // started, not done
  assert.equal(doneLogged(255, log, "tag-a"), undefined);
  assert.equal(doneLogged(0, "ssh: timeout", "tag-a"), undefined); // no `now:` line = truncated output
  // finished, result not yet synced: ABSENT with a DONE line never starts or continues an absence run
  let a = nextAbsence(undefined, "ABSENT", false, false, T0);
  a = nextAbsence(a, "ABSENT", false, true, T0 + 30 * MIN);
  assert.equal(a, undefined);
  assert.equal(nextAbsence(undefined, "ABSENT", false, undefined, T0), undefined);
});

test("polls never overlap: a call while one is in flight is skipped", async () => {
  const gate = { busy: false };
  let running = 0;
  let maxRunning = 0;
  let release: () => void = () => undefined;
  const slow = () =>
    new Promise<void>((r) => {
      running++;
      maxRunning = Math.max(maxRunning, running);
      release = () => {
        running--;
        r();
      };
    });
  const first = runExclusive(gate, slow);
  assert.equal(gate.busy, true);
  assert.equal(await runExclusive(gate, slow), false); // timer fired again mid-poll: skipped
  assert.equal(maxRunning, 1);
  release();
  assert.equal(await first, true);
  assert.equal(gate.busy, false);
  // a failing poll still releases the gate
  await assert.rejects(runExclusive(gate, async () => { throw new Error("boom"); }));
  assert.equal(gate.busy, false);
  assert.equal(await runExclusive(gate, async () => {}), true);
});
