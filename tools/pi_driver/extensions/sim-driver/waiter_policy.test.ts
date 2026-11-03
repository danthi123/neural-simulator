// node --test tools/pi_driver/extensions/sim-driver/waiter_policy.test.ts   (Node >= 23 strips the types itself)
import { test } from "node:test";
import assert from "node:assert/strict";
import { GPU_TAG, capTimeoutHours, isStale, matchWaiters, nextStrikes, parseHolds, MAX_TIMEOUT_HOURS, STALE_STRIKES } from "./waiter_policy.ts";

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

test("staleness needs consecutive ABSENT polls with no result file", () => {
  let s = 0;
  s = nextStrikes(s, "ABSENT", false);
  assert.equal(isStale(s), false);
  s = nextStrikes(s, "QUEUED", false); // popped from the queue, now visible again
  assert.equal(s, 0);
  s = nextStrikes(s, "ABSENT", false);
  s = nextStrikes(s, "UNKNOWN", false); // an ssh hiccup resets, never closes
  assert.equal(s, 0);
  s = nextStrikes(s, "ABSENT", true); // result file exists: not stale
  assert.equal(s, 0);
  for (let i = 0; i < STALE_STRIKES; i++) s = nextStrikes(s, "ABSENT", false);
  assert.equal(isStale(s), true);
});
