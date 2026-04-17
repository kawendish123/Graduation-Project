#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
MODELS="${MODELS:-mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0}"

for model in $MODELS; do
  echo "Running scaled node_filtered estimate model=${model}"
  "$PY" ./run_dads_dsl.py estimate-experiment --config "configs/estimate_${model}_node_filtered_scaled.json"
done
