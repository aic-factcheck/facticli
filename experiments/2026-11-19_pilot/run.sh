#!/bin/bash
# Pilot: search provider and service tier on 6 AVeriTeC dev claims, plus a VeriTaS smoke test.
cd "$(dirname "$0")"
set -a; . ../../.env; set +a
PY=../../.venv/bin/python
COMMON="--strategy pipeline --block-fact-checkers --max-checks 4 --tool-call-budget 8 --parallel-claims 3"
run() { name=$1; shift; $PY -m facticli.averitec_submission --output $name.json --artifacts-dir art_$name $COMMON "$@" > $name.log 2>&1; echo "$name exit=$?" >> status.txt; }
AV="--input ../../data/averitec/dev_20.json --limit 6"
run brave_std  $AV --search-provider brave &
run exa_std    $AV --search-provider exa &
run exa_flex   $AV --search-provider exa --service-tier flex &
run veritas_exa_flex --input veritas_smoke4.json --claim-field text --claim-date-field date --search-provider exa --service-tier flex &
wait
