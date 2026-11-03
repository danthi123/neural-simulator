// Pure rules for waiters (compute.ts): the timeout cap, cancel matching and the staleness verdict. No I/O, so
// node --test waiter_policy.test.ts runs it without Pi.
export const MAX_TIMEOUT_HOURS = 8;
export const DEFAULT_TIMEOUT_HOURS = 8;
// A gpu job absent from the queue and from the running slot for this many polls in a row is gone. Two, not one:
// the dispatcher pops a job from gpu.queue a moment before it writes gpu.running.
export const STALE_STRIKES = 2;

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

// Strike counter for a gpu waiter: ABSENT (and no result file) adds one, anything else resets to zero.
export function nextStrikes(prev: number, holds: Holds, resultPresent: boolean): number {
  return holds === "ABSENT" && !resultPresent ? prev + 1 : 0;
}
export const isStale = (strikes: number) => strikes >= STALE_STRIKES;
