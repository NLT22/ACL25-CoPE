#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run_with_nan_retry.sh CONFIG EXPERIMENT_NAME ALPHA BETA
#
# Optional environment variables:
#   MAX_RETRIES=5       Maximum number of attempts.
#   RETRY_DELAY=10      Seconds to wait before retrying.
#   SEPARATE_ATTEMPTS=0 Reuse and overwrite the same checkpoint directory.
#
# Set SEPARATE_ATTEMPTS=1 to preserve each attempt in a separate directory.

CONFIG=${1:?Config path is required}
NAME=${2:?Experiment name is required}
ALPHA=${3:?Alpha is required}
BETA=${4:?Beta is required}

MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_DELAY=${RETRY_DELAY:-10}
SEPARATE_ATTEMPTS=${SEPARATE_ATTEMPTS:-0}

mkdir -p logs

for attempt in $(seq 1 "$MAX_RETRIES"); do
    if [[ "$SEPARATE_ATTEMPTS" == "1" ]]; then
        OUTPUT="checkpoints/${NAME}_attempt${attempt}"
    else
        OUTPUT="checkpoints/${NAME}"
    fi
    LOG="logs/${NAME}_attempt${attempt}.log"

    echo "Starting attempt ${attempt}/${MAX_RETRIES}"
    echo "Checkpoint directory: ${OUTPUT}"
    echo "Log file: ${LOG}"

    python train.py -c "$CONFIG" \
        training.epochs=100 \
        model.loss_weights.alpha="$ALPHA" \
        model.loss_weights.beta="$BETA" \
        output.checkpoint.save_dir="$OUTPUT" \
        2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ $status -eq 0 ]]; then
        echo "Training completed successfully: ${OUTPUT}"
        exit 0
    fi

    if grep -qiE "NaN loss|NaN loss encountered" "$LOG"; then
        if [[ $attempt -lt $MAX_RETRIES ]]; then
            echo "NaN detected. Retrying from scratch in ${RETRY_DELAY} seconds..."
            sleep "$RETRY_DELAY"
        fi
    else
        echo "Training failed for a reason unrelated to NaN. See: ${LOG}"
        exit "$status"
    fi
done

echo "Training still encountered NaN after ${MAX_RETRIES} attempts."
exit 1
