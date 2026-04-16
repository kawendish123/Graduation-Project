$ErrorActionPreference = "Stop"

$PY = "D:\Program4Python\anaconda\envs\pytorch\python.exe"
$models = @("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small", "googlenet", "resnet50", "vgg16", "alexnet", "tiny_yolo", "shufflenet_v2", "efficientnet_b0")

foreach ($model in $models) {
  Write-Host "Profiling node_filtered cloud model=$model"

  & $PY .\run_dads_dsl.py profile-cache --role cloud --model $model --partition-granularity node_filtered --device cuda --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles/${model}_node_filtered/cloud_cuda.json"
}
