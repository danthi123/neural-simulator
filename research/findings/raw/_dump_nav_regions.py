"""Dump the maximal nav region+pathway set (NO GPU — build_bg_brain_regions
returns plain BrainRegion/RegionPathway dataclasses). Compares to the static
brain3d_layout.json so we can see exactly what the 3D viz is missing/stale."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from research.runners.g11_bg_runner import build_bg_brain_regions

# Maximal set: enable every region-producing option so the layout can be a
# comprehensive union (a given run lights up its active subset).
regions, pathways = build_bg_brain_regions(
    enable_cluster_a_closed_loop=True,
    enable_cluster_e_topography=True,
    enable_cluster_d_hippocampus=True,
    enable_cluster_f_cerebellum=True,
    enable_visual_cortex=True,
    enable_learned_perception=True,
    enable_spiking_wta_readout=True,
    enable_commit_burst=True,
    enable_pfc=True,
    pfc_enable_nmda=True,
    enable_hippocampus=True,
    enable_beacon_perception=True,
    enable_striatal_fsis=True,
    enable_bg_lateral_inhibition=True,
    enable_thal_lateral_inhibition=True,
)
names = [r.name for r in regions]
pw = [[p.from_region, p.to_region] for p in pathways]
print("BUILDER regions:", len(names))
print(json.dumps(sorted(names)))
print("BUILDER pathways:", len(pw))

layout = json.load(open("webapp/static/brain3d_layout.json"))
layout_regions = set(layout["regions"].keys())
builder_regions = set(names)
print("\nLAYOUT regions:", len(layout_regions))
print("\n=== in BUILDER but MISSING from layout (viz won't show these) ===")
print(json.dumps(sorted(builder_regions - layout_regions)))
print("\n=== in LAYOUT but builder does NOT create (stale/removed) ===")
print(json.dumps(sorted(layout_regions - builder_regions)))

# also dump full maximal set for layout regeneration
json.dump(
    {"regions": sorted(names), "pathways": pw},
    open("research/findings/raw/_nav_regions_dump.json", "w"),
    indent=1,
)
print("\nwrote research/findings/raw/_nav_regions_dump.json")
