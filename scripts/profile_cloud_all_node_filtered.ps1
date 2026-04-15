$ErrorActionPreference = "Stop"

$PY = "D:\Program4Python\anaconda\envs\pytorch\python.exe"
$models = @("mobilenet_v2", "googlenet", "resnet50", "vgg16")

foreach ($model in $models) {
  Write-Host "Profiling node_filtered cloud model=$model"

  & $PY .\run_dads_dsl.py profile-cache --role cloud --model $model --partition-granularity node_filtered --device cuda --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles/${model}_node_filtered/cloud_cuda.json"
}
