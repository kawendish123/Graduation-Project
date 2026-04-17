#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

PY="${PY:-python}"
MODELS="${MODELS:-mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0}"
LOADS="${LOADS:-0 10 20 30 40 50 60}"
CPUSET="${CPUSET:-0}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
PROFILE_RUNS="${PROFILE_RUNS:-20}"
GRANULARITY="node_filtered"

run_python() {
  if [ -n "$CPUSET" ] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$CPUSET" "$PY" "$@"
  else
    "$PY" "$@"
  fi
}

for model in $MODELS; do
  mkdir -p "profiles/${model}_${GRANULARITY}"
  for load in $LOADS; do
    echo "Profiling ${GRANULARITY} edge model=${model} load=${load}"
    run_python ./run_dads_dsl.py profile-cache \
      --role edge \
      --model "$model" \
      --partition-granularity "$GRANULARITY" \
      --cpu-load-target "$load" \
      --cpu-load-tolerance 5 \
      --cpu-load-interval 0.5 \
      --cpu-load-ramp-seconds 2 \
      --input-shape 1 3 224 224 \
      --warmup-runs "$WARMUP_RUNS" \
      --profile-runs "$PROFILE_RUNS" \
      --output "profiles/${model}_${GRANULARITY}/edge_load${load}.json"
  done
done
