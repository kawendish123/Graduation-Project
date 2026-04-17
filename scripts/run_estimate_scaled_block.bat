@echo off
setlocal

set "PY=D:\Program4Python\anaconda\envs\pytorch\python.exe"

for %%M in (mobilenet_v2 mobilenet_v3_large mobilenet_v3_small googlenet resnet50 vgg16 alexnet tiny_yolo shufflenet_v2 efficientnet_b0) do (
  echo Running scaled block estimate model=%%M
  "%PY%" .\run_dads_dsl.py estimate-experiment --config "configs\estimate_%%M_block_scaled.json"
  if errorlevel 1 exit /b 1
)
