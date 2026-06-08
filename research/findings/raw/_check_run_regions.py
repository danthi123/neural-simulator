import json
import sys

d = json.load(open("research/findings/raw/_biorda_neural_s42.json"))
print("top-level keys:", sorted(d.keys()))
for k in ("regions", "region_names", "region_map", "brain_regions", "region_indices", "region_slices"):
    if k in d:
        v = d[k]
        n = len(v) if hasattr(v, "__len__") else "?"
        print(f"FOUND {k}: type={type(v).__name__} len={n}")
        if isinstance(v, dict):
            print("  keys:", list(v.keys())[:40])
        elif isinstance(v, list):
            print("  sample:", v[:40])
