#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
PROFILE_RUNS="${PROFILE_RUNS:-20}"
GRANULARITY="node_filtered"

for model in $MODELS; do
  mkdir -p "profiles/${model}_${GRANULARITY}"
  echo "Profiling ${GRANULARITY} cloud model=${model} device=${DEVICE}"
  "$PY" ./run_dads_dsl.py profile-cache \
    --role cloud \
    --model "$model" \
    --partition-granularity "$GRANULARITY" \
    --device "$DEVICE" \
    --input-shape 1 3 224 224 \
    --warmup-runs "$WARMUP_RUNS" \
    --profile-runs "$PROFILE_RUNS" \
    --output "profiles/${model}_${GRANULARITY}/cloud_${DEVICE}.json"
done
