#!/usr/bin/env bash
set -euo pipefail

SCRIPT="main_3.py"        
NUM_RUNS=20
NOTES_LIMIT=5
LOG_DIR="run_logs"

mkdir -p "$LOG_DIR"

for i in $(seq 6 "$NUM_RUNS"); do
    echo "=== Run $i/$NUM_RUNS (first $NOTES_LIMIT notes) ==="
    RUN_ID="$i" NOTES_LIMIT="$NOTES_LIMIT" \
        python "$SCRIPT" 2>&1 | tee "$LOG_DIR/run_${i}.log"
done

echo "All $NUM_RUNS runs complete."
echo "Outputs: llm_classification_*_run{1..${NUM_RUNS}}.json / .csv"