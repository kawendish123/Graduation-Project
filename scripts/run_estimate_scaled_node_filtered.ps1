$ErrorActionPreference = "Stop"

$PY = "D:\Program4Python\anaconda\envs\pytorch\python.exe"
$models = @("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small", "googlenet", "resnet50", "vgg16", "alexnet", "tiny_yolo", "shufflenet_v2", "efficientnet_b0")

foreach ($model in $models) {
  Write-Host "Running scaled node_filtered estimate model=$model"
  & $PY .\run_dads_dsl.py estimate-experiment --config "configs\estimate_${model}_node_filtered_scaled.json"
}
