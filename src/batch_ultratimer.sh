#!/usr/bin/env bash

MAX_JOBS=4
SCRIPT="UltraTimer.py"

for f in ../../../Songs/*/*.txt; do
    audio="${f%.txt}.m4a"
    (
        python3.11 "$SCRIPT" \
            --audio "$audio" \
            --txt "$f" \
            --no-denoise \
            --gap-step-ms 2 \
            --binary-threshold 0.07 \
            --resolution-ms 2 \
            --correlation cosine \
            --threads 12 \
            --weight-early 1
    ) &

    # limit to MAX_JOBS concurrent processes
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

# wait for all remaining background jobs
wait
echo "✅ All UltraTimer runs completed."

