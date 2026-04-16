@echo off
setlocal

set "PY=D:\Program4Python\anaconda\envs\pytorch\python.exe"

for %%M in (mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0) do (
  echo Profiling block cloud model=%%M
  "%PY%" .\run_dads_dsl.py profile-cache --role cloud --model %%M --partition-granularity block --device cuda --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles\%%M_block\cloud_cuda.json"
  if errorlevel 1 exit /b 1
)
