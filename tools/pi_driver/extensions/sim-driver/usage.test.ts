// node --test tools/pi_driver/extensions/sim-driver/usage.test.ts   (Node >= 23 strips the types itself)
import { test } from "node:test";
import assert from "node:assert/strict";
import { type Usage, parseRateLimit, usageDecision, weekCeiling } from "./usage.ts";

// A real line from `claude -p ... --output-format stream-json --verbose` (Claude Code 2.1.316, 2026-11-02).
const EVENT = '{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1793667000,"rateLimitType":"five_hour",' +
  '"overageStatus":"rejected","overageDisabledReason":"org_level_disabled","isUsingOverage":false,"unifiedWindows":{"five_hour":' +
  '{"utilization":0.46,"resetsAt":1793667000,"status":"allowed"},"seven_day":{"utilization":0.7,"resetsAt":1793718000,"status":"allowed"}}},' +
  '"uuid":"x","session_id":"y"}';
const LIM = { max_five_hour: 0.75, max_seven_day: 0.85, week_pace_start: 0.3 };
const DAY = 86400_000;
const u = (five: number, week: number, daysLeft: number, now: number, status = "allowed"): Usage => ({
  at: now, five_hour: five, seven_day: week, five_reset: now / 1000 + 3600, seven_reset: (now + daysLeft * DAY) / 1000, status,
});

test("parses the real rate_limit_event among other stream lines", () => {
  const out = ['{"type":"system","subtype":"init"}', '{"type":"assistant"}', EVENT, '{"type":"result","result":"OK"}'].join("\n");
  const r = parseRateLimit(out, 5);
  assert.ok(r);
  assert.equal(r.five_hour, 0.46);
  assert.equal(r.seven_day, 0.7);
  assert.equal(r.seven_reset, 1793718000);
  assert.equal(r.status, "allowed");
  assert.equal(r.at, 5);
});

test("no event, a truncated event, or one without both windows gives null", () => {
  assert.equal(parseRateLimit('{"type":"result"}'), null);
  assert.equal(parseRateLimit(EVENT.slice(0, 120)), null);
  assert.equal(parseRateLimit('{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","unifiedWindows":{"five_hour":{"utilization":0.1}}}}'), null);
});

test("unknown usage refuses: never call Claude blind", () => {
  assert.equal(usageDecision(null, LIM).ok, false);
});

test("the 5-hour window stops Pi at the ceiling", () => {
  const now = Date.UTC(2026, 10, 2);
  assert.equal(usageDecision(u(0.74, 0.1, 3.5, now), LIM, now).ok, true);
  assert.equal(usageDecision(u(0.75, 0.1, 3.5, now), LIM, now).ok, false);
  assert.equal(usageDecision(u(0.1, 0.1, 3.5, now, "rejected"), LIM, now).ok, false);
});

test("the weekly ceiling rises from pace start to max over the week", () => {
  const now = Date.UTC(2026, 10, 2);
  assert.ok(Math.abs(weekCeiling(u(0, 0, 7, now), LIM, now) - 0.3) < 1e-9);
  assert.ok(Math.abs(weekCeiling(u(0, 0, 0, now), LIM, now) - 0.85) < 1e-9);
  assert.ok(Math.abs(weekCeiling(u(0, 0, 3.5, now), LIM, now) - 0.575) < 1e-9);
  // day one of the week: 40% used is already ahead of pace
  assert.equal(usageDecision(u(0.2, 0.4, 6.5, now), LIM, now).ok, false);
  // last day: 70% used is fine (today's real reading)
  assert.equal(usageDecision(u(0.46, 0.7, 0.8, now), LIM, now).ok, true);
  // never above the end-of-week max, even with the window past its reset
  assert.equal(usageDecision(u(0.2, 0.86, -1, now), LIM, now).ok, false);
});
