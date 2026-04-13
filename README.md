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

```powershell
python .\run_dads_dsl.py client-run --server localhost:50051 --model mobilenet_v2 --bandwidth-mbps 5 --cpu-load-target 0
```
