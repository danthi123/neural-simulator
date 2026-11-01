"""C2 step-0 relay: the patient-inhibition pool's spikes per block vs the summed patient drive (read-only over the shards).

Finding research/findings/2026-11-01-c2-patient-inhibition-pool-pinned-at-ceiling.md; RULING c2-ff-grade-1101. Reads every
flag-on block of research/findings/raw/_c2_step0_relay/s*/*.json and writes
research/findings/raw/_c2_pinh_ceiling/relay_inh_vs_drive_s1-7.json. No simulation; CPU only.
Run: python -m research.runners.c2_pinh_ceiling_table
"""


def main():
    import json, glob, collections, os
    rows = []
    srcs = []
    for p in sorted(glob.glob("research/findings/raw/_c2_step0_relay/s*/*.json")):
        if p.endswith(".prov.json"): continue
        d = json.load(open(p)); srcs.append(p)
        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("patient_relay"), dict) and "patient_slice_nz" in o: yield o
                for v in o.values(): yield from walk(v)
            elif isinstance(o, list):
                for v in o: yield from walk(v)
        for b in walk(d):
            nz = b["patient_slice_nz"]
            rows.append(dict(shard=p, block=b.get("block"), summed_patient_drive=sum(nz.values()),
                             inh_spikes=b["patient_relay"].get("inh_spikes"), n_fired=b.get("n_fired"),
                             fires_decoded_patient=b.get("fires_decoded_patient")))
    tot = sorted(r["summed_patient_drive"] for r in rows)
    out = dict(what="C2 step-0 relay (flag on): pinh spikes per 20-step block vs summed patient drive, dev seeds 1-7",
               made_by="Claude, read-only over the shards listed; RULING c2-ff-grade-1101", ceiling_spikes_per_block=300,
               n_blocks=len(rows), inh_spikes_counts=dict(collections.Counter(str(r["inh_spikes"]) for r in rows)),
               summed_drive_min=tot[0], summed_drive_max=tot[-1], drive_ratio=tot[-1] / tot[0],
               n_fired_counts=dict(collections.Counter(str(r["n_fired"]) for r in rows)),
               fires_decoded_patient_true=sum(1 for r in rows if r["fires_decoded_patient"]),
               sources=srcs, blocks=rows)
    os.makedirs("research/findings/raw/_c2_pinh_ceiling", exist_ok=True)
    json.dump(out, open("research/findings/raw/_c2_pinh_ceiling/relay_inh_vs_drive_s1-7.json", "w"), indent=1)
    print(out["n_blocks"], out["inh_spikes_counts"], round(out["drive_ratio"], 2), out["fires_decoded_patient_true"])


if __name__ == "__main__":
    main()
