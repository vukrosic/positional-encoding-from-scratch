#!/bin/sh
# Runs every step in order. Trained networks are kept in nets/, so a second run is fast.
set -e
cd "$(dirname "$0")"
python3 tests.py
for f in step1_blind step2_tags step3_clocks step4_rotate step5_long; do
  echo; echo "== $f"; python3 $f.py
done
echo; echo "The exercise: read exercise.py, guess, then run  python3 exercise.py"
