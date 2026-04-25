"""Brain-region preset validation harnesses.

Each validate_*.py module measures cellular- and network-level metrics for
one brain region preset, compares to published targets, and produces a
findings report. The goal: confirm that our HH/Izhikevich parameter
presets actually produce the firing patterns they're labeled as before
we build composite multi-region experiments on top of them.
"""
