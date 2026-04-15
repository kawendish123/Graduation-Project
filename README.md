# DADS DSL Prototype

This project reconstructs the `DSL` algorithm from the paper "Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge".

It provides three entrypoints:

- `profile`: export a `ModelProfile` JSON from a PyTorch model.
- `dsl`: solve the lightly-loaded partitioning problem for one bandwidth.
- `simulate`: sweep multiple bandwidths and inspect how the cut changes.
- `serve`: run the cloud-side gRPC server.
- `client-run`: run one edge/cloud split inference and print measured latency.

## Install

Core dependencies:

```powershell
python -m pip install -e . --no-build-isolation
```

With PyTorch profiling support:

```powershell
python -m pip install -e .[torch] --no-build-isolation
```

If you do not want to install the package first, you can run the workspace-local wrapper directly:

```powershell
python .\run_dads_dsl.py --help
```

## Usage

Export a profile:

```powershell
python .\run_dads_dsl.py profile --model mobilenet_v2 --output profiles/mobilenet_v2.json
```

Solve DSL at one bandwidth:

```powershell
python .\run_dads_dsl.py dsl --profile profiles/mobilenet_v2.json --bandwidth-mbps 5
```

Sweep several bandwidth points:

```powershell
python .\run_dads_dsl.py simulate --profile profiles/mobilenet_v2.json --bandwidths 0.5 1 2 5 10
```

Run the cloud server:

```powershell
python .\run_dads_dsl.py serve --host 0.0.0.0 --port 50051 --device cuda
```

Run one edge/client inference:

```powershell client-run --server localhost:50051 --model mobilenet_v2 --bandwidth-mbps 5 --cpu-load-target 0
```



start

```powershell
提前测量边缘实验
python.exe .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 0 --profile-runs 5 --warmup-runs 3 --output profiles/resnet50_block/edge_load0.json
```
```powershell
提前测量边缘实验
python.exe .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 30 --profile-runs 5 --warmup-runs 3 --output profiles/resnet50_block/edge_load30.json
```
```powershell
提前测量边缘实验
python.exe .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 60 --profile-runs 5 --warmup-runs 3 --output profiles/resnet50_block/edge_load60.json
```

```powershell
提前测量云端时延
python.exe ./run_dads_dsl.py profile-cache --role cloud --model resnet50 --partition-granularity block --device cuda --profile-runs 5 --warmup-runs 3 --output profiles/resnet50_block/cloud_cuda.json
```
```server
python.exe .\run_dads_dsl.py run-config --config configs\serve_cuda.json
```


```powershell
运行时延
python.exe .\run_dads_dsl.py experiment --config configs\experiment_mobilenet_v2.json
```

```linux
python ./run_dads_dsl.py experiment --config configs/experiment_googlenet_block.json
```


边缘端设置线程
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TORCH_NUM_THREADS=1

边缘端测量时延

start "" /wait /affinity 1 python .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 0 --profile-runs 5 --warmup-runs 3 --output profiles\resnet50_block\edge_load0.json

start "" /wait /affinity 1 python .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 30 --profile-runs 5 --warmup-runs 3 --output profiles\resnet50_block\edge_load30.json

start "" /wait /affinity 1 python .\run_dads_dsl.py profile-cache --role edge --model resnet50 --partition-granularity block --cpu-load-target 60 --profile-runs 5 --warmup-runs 3 --output profiles\resnet50_block\edge_load60.json



云端测量时延
python.exe ./run_dads_dsl.py profile-cache --role cloud --model resnet50 --partition-granularity block --device cuda --profile-runs 5 --warmup-runs 3 --output profiles/resnet50_block/cloud_cuda.json


拷贝
服务端实验
python ./run_dads_dsl.py serve --host 0.0.0.0 --port 6006 --device cuda


边缘端进行实验
start "" /wait /affinity 1 python .\run_dads_dsl.py experiment --config configs\experiment_resnet50_block_cached.json



###start
获取云端时延
python ./run_dads_dsl.py profile-cache --role cloud --model vgg16 --partition-granularity block --device cuda --profile-runs 20 --warmup-runs 5 --output profiles/vgg16_block/cloud_cuda.json

python ./run_dads_dsl.py profile-cache --role cloud --model resnet50 --partition-granularity block --device cuda --profile-runs 20 --warmup-runs 5 --output profiles/resnet50_block/cloud_cuda.json

python ./run_dads_dsl.py profile-cache --role cloud --model googlenet --partition-granularity block --device cuda --profile-runs 20 --warmup-runs 5 --output profiles/google_block/cloud_cuda.json

python ./run_dads_dsl.py profile-cache --role cloud --model mobilenet_v2 --partition-granularity block --device cuda --profile-runs 20 --warmup-runs 5 --output profiles/mobilenet_v2_block/cloud_cuda.json


边缘端：


start "" /wait /affinity 1 python .\run_dads_dsl.py profile-cache --role edge --model mobilenet_v2 --partition-granularity block --cpu-load-target 30 --cpu-load-tolerance 5 --cpu-load-interval 0.5 --cpu-load-ramp-seconds 2 --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output "profiles/mobilenet_v2_block/edge_load30.json"


python.exe .\run_dads_dsl.py estimate-experiment --config configs\estimate_vgg16_block.json