# Capability: Self and identity
# Record: brain/research/2026-11-27-voice-what-sleep-stores-DESIGN.md
# Expected-fail: none (host-only count)
"""Stage-0 recount (host only, no model). Trunk lines only (no prunedA), kinds sameday/remention,
self-name items (dan, nova) excluded. Run: python3 -I recount.py [lives_dir] [maxday]"""
import json, re, sys, collections
D = sys.argv[1] if len(sys.argv) > 1 else '_scratch/stage2/read'
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SELF = {'dan', 'nova'}
HEDGE = re.compile(r"\b(not sure|don't know|do not know|i think|i bet|maybe|perhaps|probably|do you know)\b", re.I)

def cls(reply):
    t = re.findall(r"[A-Za-z']+", reply)
    f = t[0].lower() if t else ''
    if f in ('you', 'your', "you're", "you've"): return 'You/Your'
    if f in ('my', 'i', "i'm", "i've", "i'd"): return 'My/I'
    return 'terse/other'

def names(reply, keys):
    r = reply.lower()
    return any(re.search(r"\b" + re.escape(k.lower()), r) for k in keys)

out = {}
for s in (319, 320, 321):
    d = json.load(open(f'{D}/lifeKP_s{s}.json'))
    c = collections.Counter(); fc = collections.Counter(); rows = []
    for l in d['lived']:
        if l['line'] != 'trunk': continue
        for x in l['rows']:
            if x['kind'] not in ('sameday', 'remention') or x['day'] > MAXD: continue
            if x['item'] in SELF: c['self_name_excluded'] += 1; continue
            if not x['brain'].strip(): continue
            c['answered'] += 1
            keys = d['ledger'][x['item']]['keys']
            n = names(x['brain'], keys)
            h = bool(HEDGE.search(x['brain']))
            if n:
                c['names_key'] += 1; fc[cls(x['brain'])] += 1
                if h: c['names_key_hedged'] += 1
                else: c['names_key_unhedged'] += 1
                c['pass_frame_blind_filter'] += 1
            rows.append((x['day'], x['item'], n, h, cls(x['brain']), x['brain'][:60]))
    out[s] = (c, fc, rows)

print(f'days 1-{MAXD}, trunk only')
print('seed answered names_key You/Your My/I terse/other hedged_in_names unhedged_names pass_filter self_excl')
tot = collections.Counter()
for s, (c, fc, rows) in out.items():
    print(s, c['answered'], c['names_key'], fc['You/Your'], fc['My/I'], fc['terse/other'], c['names_key_hedged'],
          c['names_key_unhedged'], c['pass_frame_blind_filter'], c['self_name_excluded'])
    tot.update(c); tot.update(fc)
print('sum', tot['answered'], tot['names_key'], tot['You/Your'], tot['My/I'], tot['terse/other'],
      tot['names_key_hedged'], tot['names_key_unhedged'], tot['pass_frame_blind_filter'], tot['self_name_excluded'])
print('Stage-1 floor (>=8 confirmed per seed):', {s: out[s][0]['pass_frame_blind_filter'] >= 8 for s in out})
if '-v' in sys.argv:
    for s, (_, _, rows) in out.items():
        for r in rows: print(s, *r)
