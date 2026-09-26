#!/usr/bin/env bash
# Test-only systemctl stand-in (tests/conftest.py): every unit reads inactive; stop/start/reset-failed are no-ops.
case "$*" in *is-active*) echo inactive; exit 3 ;; *) exit 0 ;; esac
