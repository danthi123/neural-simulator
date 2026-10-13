"""Attribute the per-session ~5.3 GB: RSS of loading ONLY the tiny-demo's LTM store (the per-session TieredFactStore
tier built inside _build_chat_brain), with no brain build."""
import gc, os, time, ctypes
os.environ.setdefault("SIM_BACKEND", "numpy")


def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS:"):
            return int(l.split()[1])


from webapp import server as S
r0 = rss()
bundle = S._resolve_ltm_bundle()
print("bundle", bundle, flush=True)
t = time.time()
ltm = S._load_or_build_ltm_store(bundle, seed=7)
r1 = rss()
n_facts = sum(len(sh.kb) for sh in ltm.shards)
handles = [h for sh in ltm.shards for (_f, h) in sh.kb]
kinds = {}
for h in handles:
    kinds[type(h).__name__] = kinds.get(type(h).__name__, 0) + 1
print("load %.0fs; rss_kb before %d after %d delta %d (%.2f GB)" % (time.time() - t, r0, r1, r1 - r0, (r1 - r0) / 1e6))
print("n_shards %d n_facts %d handle types %s; per fact %.1f KiB" % (len(ltm.shards), n_facts, kinds,
                                                                     (r1 - r0) / max(1, n_facts)))
del handles, ltm
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
print("after del+gc+trim rss_kb %d (released %d)" % (rss(), r1 - rss()))
