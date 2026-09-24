"""LBF ROW PACKAGE (midnight-plan 2026-09-24). Each module here exposes module-level `EXTRA_LESIONS` /
`EXTRA_PROBES` dicts/lists with exactly the same entry shapes as `FACULTY_LESIONS` / `FACULTY_PROBES` in
`research/runners/load_bearing_fraction.py`. The AG-REG lane's import hook (`load_bearing_fraction.py`) merges
every module here into the two registries -- lanes never edit `FACULTY_LESIONS`/`FACULTY_PROBES` directly. This
package does not import its member modules itself (avoids import-order coupling); the hook discovers and imports
them.
"""
