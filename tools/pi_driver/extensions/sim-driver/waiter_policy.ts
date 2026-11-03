// Pure rules for waiters (compute.ts): the timeout cap, cancel matching and the staleness verdict. No I/O, so
// node --test waiter_policy.test.ts runs it without Pi.
export const MAX_TIMEOUT_HOURS = 8;
export const DEFAULT_TIMEOUT_HOURS = 8;
// A gpu job is stale only when it has been absent from the queue AND the running slot, with no result file and no
// DONE line in gpu_queue.log, on at least STALE_MIN_POLLS polls in a row spanning at least STALE_MIN_MS. Polls alone
// are not enough: a job is legitimately in neither file while the dispatcher moves it from gpu.queue to gpu.running,
// and after it finishes until sim-sync (every 15 minutes) brings its result file here. 20 minutes covers one sync.
export const STALE_MIN_POLLS = 2;
export const STALE_MIN_MS = 20 * 60_000;

// A gpu target is passed to `gpu_remote.sh holds`, which accepts exactly this alphabet.
export const GPU_TAG = /^[A-Za-z0-9._/=:+,@%-]+$/;

export interface WaiterLike {
  id: string;
  label: string;
  kind: "file" | "gpu" | "shell";
  target: string;
  result_file?: string;
}

// Hours -> a usable timeout: missing or non-finite -> the default, otherwise clamped to (0, MAX].
export function capTimeoutHours(h: number | undefined): number {
  if (h === undefined || !Number.isFinite(h) || h <= 0) return DEFAULT_TIMEOUT_HOURS;
  return Math.min(h, MAX_TIMEOUT_HOURS);
}

// "id" matches exactly; otherwise `needle` is a case-insensitive substring of the label or the target.
export function matchWaiters<T extends WaiterLike>(ws: T[], needle: string): T[] {
  const n = needle.trim().toLowerCase();
  if (!n) return [];
  const exact = ws.filter((w) => w.id.toLowerCase() === n);
  if (exact.length) return exact;
  return ws.filter((w) => w.label.toLowerCase().includes(n) || w.target.toLowerCase().includes(n));
}

// What `gpu_remote.sh holds <tag>` printed: QUEUED, RUNNING, ABSENT, or anything else (ssh down, bad tag) = UNKNOWN.
export type Holds = "QUEUED" | "RUNNING" | "ABSENT" | "UNKNOWN";
export function parseHolds(code: number, out: string): Holds {
  if (code !== 0) return "UNKNOWN";
  const m = /^holds: (QUEUED|RUNNING|ABSENT)\s*$/m.exec(out);
  return m ? (m[1] as Holds) : "UNKNOWN";
}

// Absence record for a gpu waiter: when its job was first seen ABSENT in this unbroken run, and how many polls.
export interface Absence {
  since: number; // epoch ms of the first ABSENT poll of the run
  polls: number;
}

// One poll's evidence -> the next absence record (undefined = not absent). The run continues only on ABSENT with no
// result file and a DONE check that succeeded and found nothing. A DONE line, a result file, QUEUED/RUNNING, or any
// UNKNOWN (holds failed, or the DONE check could not be made: doneLogged undefined) resets it, so a flaky link never
// closes a wait; the deadline still does.
export function nextAbsence(prev: Absence | undefined, holds: Holds, resultPresent: boolean, doneLogged: boolean | undefined, now: number): Absence | undefined {
  if (holds !== "ABSENT" || resultPresent || doneLogged !== false) return undefined;
  return prev ? { since: prev.since, polls: prev.polls + 1 } : { since: now, polls: 1 };
}
export const isStale = (a: Absence | undefined, now: number) => !!a && a.polls >= STALE_MIN_POLLS && now - a.since >= STALE_MIN_MS;

// Does the gpu_queue.log START/DONE tail (`gpu_remote.sh events`) hold a DONE line for <tag>? undefined = could not tell.
export function doneLogged(code: number, out: string, tag: string): boolean | undefined {
  if (code !== 0 || !/^now: /m.test(out)) return undefined;
  return out.split("\n").some((l) => /\bDONE\(rc=-?\d+\): /.test(l) && l.includes(tag));
}

// Single-flight: a call made while the previous one is still running is skipped (returns false), so timer-driven
// polls never overlap. The gate object is owned by the caller (compute.ts keeps one at module level).
export interface Gate {
  busy: boolean;
}
export async function runExclusive(gate: Gate, fn: () => Promise<void>): Promise<boolean> {
  if (gate.busy) return false;
  gate.busy = true;
  try {
    await fn();
  } finally {
    gate.busy = false;
  }
  return true;
}
