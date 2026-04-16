@echo off
setlocal enabledelayedexpansion

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TORCH_NUM_THREADS=1

set "PY=D:\Program4Python\anaconda\envs\pytorch\python.exe"

for %%M in (mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0) do (
  for %%L in (0 10 20 30 40 50 60) do (
    echo Profiling node_filtered edge model=%%M load=%%L
    start "" /wait /affinity 1 "%PY%" .\run_dads_dsl.py profile-cache --role edge --model %%M --partition-granularity node_filtered --cpu-load-target %%L --cpu-load-tolerance 5 --cpu-load-interval 0.5 --cpu-load-ramp-seconds 2 --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles\%%M_node_filtered\edge_load%%L.json"
    if errorlevel 1 exit /b 1
  )
)
