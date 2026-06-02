#!/bin/bash
set -o pipefail
cd /e/Documents/Projects/sim
echo "===== [$(date +%H:%M:%S)] any-bank escalation (all 320 as fillers in any role) ====="
python -m research.findings.raw._insubstrate_flatdist320_anybank_test
echo "===== [$(date +%H:%M:%S)] conversational-KB demo (tangible artifact) ====="
python -m research.runners.compose_flatdist320_conversation_demo
echo "##### ANYBANK-DEMO-DONE [$(date +%H:%M:%S)] #####"
