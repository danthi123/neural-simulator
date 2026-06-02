#!/bin/bash
set -e
for S in 43 44; do
  echo "### hier320 spiking rm-seed $S (cached codes) ###"
  python -m research.findings.raw._insubstrate_hierarchical320_spiking --rm-seed $S 2>&1 | grep -E "between-concept|RESULT|VERDICT"
done
echo "##### HIER320-MS-DONE #####"
