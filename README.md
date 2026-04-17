# DADS / DSL 边云切分实验系统

本项目复现论文 **Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge** 中的 DSL 切分思想，并扩展为 PyTorch 边云切分实验系统。系统同时支持真实 gRPC 边云协同推理和离线估计实验。

论文主实验建议使用 **离线估计模式**：先分别在边缘端和云端测 profile，再根据带宽和 DSL 最小割算法计算切分策略、纯边缘时延、纯云端时延和 DSL 估计时延。

## 环境

推荐解释器：

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe
```

安装项目：

```powershell
python -m pip install -e . --no-build-isolation
```

如果环境中还没有 PyTorch 和 torchvision：

```powershell
python -m pip install -e .[torch] --no-build-isolation
```

直接查看入口：

```powershell
python .\run_dads_dsl.py --help
```

## 支持模型

所有模型默认 `weights=None`，不会下载预训练权重。

```text
mobilenet_v2
mobilenet_v3_large
mobilenet_v3_small
googlenet
resnet50
vgg16
alexnet
tiny_yolo
shufflenet_v2
efficientnet_b0
```

说明：

- `tiny_yolo` 是项目内置 Tiny-YOLO-like 网络，只用于时延和切分实验，不做检测后处理和精度评估。
- `shufflenet_v2` 对应 torchvision 的 `shufflenet_v2_x1_0(weights=None)`。
- MobileNet 系列覆盖当前 torchvision 可用的 V2、V3 Large、V3 Small。

## 切分粒度

`--partition-granularity` 支持：

```text
node
node_filtered
block
```

建议：

- `node`：原始 FX 节点，适合调试。
- `node_filtered`：论文主实验推荐，过滤 BN、激活、Dropout、reshape 等轻量节点，保留 Conv、Pool、Linear、add、cat、mul 等关键节点。
- `block`：按模型结构封装为大块，profile 更稳定，但多数模型会变成链式图。

如果要体现 DAG、多分支和辅助节点构图，优先使用 `node_filtered`。

## CLI 入口

```text
profile              导出单个模型 profile
profile-cache        生成 edge/cloud profile 缓存
dsl                  对一个 profile 和一个带宽求 DSL 最小割
simulate             对多个带宽做 DSL sweep
serve                启动云端 gRPC 服务
client-run           客户端执行一次边云切分推理
run-config           从 JSON 配置运行命令
experiment           真实 gRPC 边云实验
estimate-experiment  离线估计主实验
```

## 推荐主实验：离线估计

离线估计不启动 gRPC，不真实传输中间张量。它读取提前测好的 profile，并用公式计算传输时延：

```text
transfer_ms(i) = output_bytes(i) * 8 / (bandwidth_mbps * 1e6) * 1000
```

最终对比：

```text
DSL
Pure Edge
Pure Cloud
```

### 1. 云端测 cloud profile

```powershell
python .\run_dads_dsl.py profile-cache --role cloud --model mobilenet_v2 --partition-granularity node_filtered --device cuda --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output profiles\mobilenet_v2_node_filtered\cloud_cuda.json
```

如果没有 CUDA：

```powershell
python .\run_dads_dsl.py profile-cache --role cloud --model mobilenet_v2 --partition-granularity node_filtered --device cpu --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output profiles\mobilenet_v2_node_filtered\cloud_cpu.json
```

### 2. 边缘端测 edge profile

Windows CMD 示例：

```cmd
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TORCH_NUM_THREADS=1

start "" /wait /affinity 1 python .\run_dads_dsl.py profile-cache --role edge --model mobilenet_v2 --partition-granularity node_filtered --cpu-load-target 30 --cpu-load-tolerance 5 --cpu-load-interval 0.5 --cpu-load-ramp-seconds 2 --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output profiles\mobilenet_v2_node_filtered\edge_load30.json
```

Linux 示例：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

taskset -c 0 python ./run_dads_dsl.py profile-cache --role edge --model mobilenet_v2 --partition-granularity node_filtered --cpu-load-target 30 --cpu-load-tolerance 5 --cpu-load-interval 0.5 --cpu-load-ramp-seconds 2 --input-shape 1 3 224 224 --warmup-runs 5 --profile-runs 20 --output profiles/mobilenet_v2_node_filtered/edge_load30.json
```

### 3. 批量 profile

PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\profile_edge_all_node_filtered.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\profile_cloud_all_node_filtered.ps1
```

CMD：

```cmd
scripts\profile_edge_all_node_filtered.bat
scripts\profile_cloud_all_node_filtered.bat
```

block 粒度：

```cmd
scripts\profile_edge_all_block.bat
scripts\profile_cloud_all_block.bat
```

脚本默认覆盖所有模型和负载：

```text
cpu_load_target = 0, 10, 20, 30, 40, 50, 60
```

### 4. 运行离线估计

```powershell
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_mobilenet_v2_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_resnet50_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_googlenet_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_vgg16_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_alexnet_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_tiny_yolo_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_mobilenet_v3_large_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_mobilenet_v3_small_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_shufflenet_v2_node_filtered.json
python .\run_dads_dsl.py estimate-experiment --config configs\estimate_efficientnet_b0_node_filtered.json
```

输出：

```text
results/estimate_<model>_<granularity>.csv
results/estimate_<model>_<granularity>.json
results/debug_estimate_<model>_<granularity>/*.json
results/plots_estimate_<model>_<granularity>/*.png
```

图中 DSL 曲线使用虚线并最后绘制，因此与纯边缘或纯云端重合时也能看出 DSL 的选择。

## 配置文件

典型配置：

```json
{
  "command": "estimate-experiment",
  "model": "mobilenet_v2",
  "partition_granularity": "node_filtered",
  "bandwidths_mbps": [1, 2, 3, 4, 5, 10, 20],
  "cpu_load_targets": [0, 10, 20, 30, 40, 50, 60],
  "input_shape": [1, 3, 224, 224],
  "edge_profiles": {
    "0": "profiles/mobilenet_v2_node_filtered/edge_load0.json",
    "30": "profiles/mobilenet_v2_node_filtered/edge_load30.json"
  },
  "cloud_profile": "profiles/mobilenet_v2_node_filtered/cloud_cuda.json",
  "report_csv": "results/estimate_mobilenet_v2_node_filtered.csv",
  "report_json": "results/estimate_mobilenet_v2_node_filtered.json",
  "debug_dir": "results/debug_estimate_mobilenet_v2_node_filtered",
  "plot_dir": "results/plots_estimate_mobilenet_v2_node_filtered",
  "plot_format": "png"
}
```

常用字段：

- `model`：模型名。
- `partition_granularity`：切分粒度，推荐 `node_filtered`。
- `bandwidths_mbps`：带宽列表，单位 Mbps。
- `cpu_load_targets`：边缘端负载目标。
- `edge_profiles`：不同负载下的边缘端 profile。
- `cloud_profile`：云端 profile。
- `debug_dir`：DSL 图结构和切分结果。
- `plot_dir`：自动生成的时延图。

## Debug JSON

`debug_dir` 中的 JSON 用于分析 DSL 建图和切分，主要包含：

```text
nodes
flow_edges
cut_edges
aux_nodes
partition
summary
```

可以查看每个节点的 `edge_ms / cloud_ms / transmission_ms / output_bytes`，以及 `V_E / V_S / V_C`、辅助节点和最小割 cut edges。

## DAG 情况

`node_filtered` 下通常是 DAG 的模型：

```text
mobilenet_v2
mobilenet_v3_large
mobilenet_v3_small
googlenet
resnet50
shufflenet_v2
efficientnet_b0
```

基本链式的模型：

```text
vgg16
alexnet
tiny_yolo
```

`block` 粒度下，当前所有模型基本都会变成链式，因为复杂结构被封装到了 block 内部。

## gRPC 边云协同模式

启动云端：

```powershell
python .\run_dads_dsl.py serve --host 0.0.0.0 --port 50051 --device cuda
```

或使用配置：

```powershell
python .\run_dads_dsl.py run-config --config configs\serve_cuda.json
```

客户端运行一次：

```powershell
python .\run_dads_dsl.py client-run --server <server_ip>:50051 --model mobilenet_v2 --bandwidth-mbps 10 --cpu-load-target 0 --partition-granularity node_filtered --report-format table
```

输出包括：

```text
split_nodes
edge_actual_ms
transmission_actual_ms
cloud_actual_ms
total_actual_ms
payload_bytes
```

真实 gRPC 模式适合系统可行性验证。论文主图建议优先使用离线估计模式，避免网络波动影响结论。

## DSL 建图逻辑

代价：

```text
t_edge(i) = edge_ms(i)
t_cloud(i) = cloud_ms(i)
t_transfer(i) = output_bytes(i) * 8 / (bandwidth_mbps * 1e6) * 1000 ms
```

图构造：

- 源点 `e` 表示边缘端。
- 汇点 `c` 表示云端。
- `e -> v` 容量为 `cloud_ms(v)`。
- `v -> c` 容量为 `edge_ms(v)`。
- 模型 DAG 边表示传输代价。
- 多后继节点插入辅助节点，保证同一个中间结果只计算一次传输代价。
- 加入反向 `INF` 约束边，禁止 cloud -> edge 回传。
- 加入虚拟输入节点 `__input__`，保证纯云端也要支付输入上传代价。

求解结果：

- `V_E`：边缘端节点。
- `V_S`：上传中间结果的切分节点。
- `V_C`：云端节点。

## 论文实验建议

推荐主实验设置：

```text
partition_granularity = node_filtered
bandwidths_mbps = 1..20
cpu_load_targets = 0,10,20,30,40,50,60
profile_runs = 20
warmup_runs = 5
```

推荐展示：

- DSL、Pure Edge、Pure Cloud 总时延曲线。
- speedup over best baseline。
- 分割点随带宽/负载变化的 heatmap。
- debug JSON 中的 DAG 辅助节点分析。

如果 DSL 与纯边缘或纯云端曲线重合，这是合理结果，说明 DSL 自动选择了当前条件下的最优基线，而不是强行切分。
