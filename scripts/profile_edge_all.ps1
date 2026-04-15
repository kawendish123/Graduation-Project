$ErrorActionPreference = "Stop"

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:TORCH_NUM_THREADS = "1"

$PY = "D:\Program4Python\anaconda\envs\pytorch\python.exe"
$models = @("mobilenet_v2", "googlenet", "resnet50", "vgg16")
$loads = @(0, 10, 20, 30, 40, 50, 60)

foreach ($model in $models) {
  foreach ($load in $loads) {
    Write-Host "Profiling edge model=$model load=$load"

    cmd /c start "" /wait /affinity 1 $PY .\run_dads_dsl.py profile-cache --role edge --model $model --partition-granularity block --cpu-load-target $load --cpu-load-tolerance 5 --cpu-load-interval 0.5 --cpu-load-ramp-seconds 2 --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles/${model}_block/edge_load${load}.json"
  }
}
