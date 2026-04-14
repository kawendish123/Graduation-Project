from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from .types import ModelProfile, ModelProfileNode

AVAILABLE_MODELS = ("mobilenet_v2", "googlenet", "resnet50", "vgg16")
PARTITION_GRANULARITIES = ("node", "block")

PASS_THROUGH_FUNCTIONS = {"getitem", "getattr"}
PASS_THROUGH_METHODS = {
    "clone",
    "contiguous",
    "detach",
    "flatten",
    "permute",
    "reshape",
    "size",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
}


def _require_torch():
    try:
        import torch
        import torch.fx as fx
        import torchvision.models as tv_models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch profiling support is unavailable. Install the optional "
            "'torch' dependencies first."
        ) from exc
    return torch, fx, tv_models


def _tensor_bytes(value: Any) -> int:
    tensors = []

    def collect(item: Any) -> None:
        if item is None:
            return
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            tensors.append(item)
            return
        if isinstance(item, dict):
            for nested in item.values():
                collect(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                collect(nested)

    collect(value)
    return int(sum(t.numel() * t.element_size() for t in tensors))


def _contains_tensor(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_tensor(item) for item in value)
    return False


def _op_type(node, traced_module) -> str:
    if node.op == "call_module":
        module = traced_module.get_submodule(str(node.target))
        return module.__class__.__name__
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    if node.op == "call_method":
        return str(node.target)
    return node.op


def _is_cut_candidate(node, output: Any) -> bool:
    if node.op not in {"call_module", "call_function", "call_method"}:
        return False
    if not _contains_tensor(output):
        return False
    if node.op == "call_function":
        name = getattr(node.target, "__name__", str(node.target))
        if name in PASS_THROUGH_FUNCTIONS:
            return False
    if node.op == "call_method" and str(node.target) in PASS_THROUGH_METHODS:
        return False
    return True


class _ProfilingInterpreter:
    def __init__(self, fx, module, torch_mod, device, capture_outputs: bool):
        self._fx = fx
        self._module = module
        self._torch = torch_mod
        self._device = device
        self._capture_outputs = capture_outputs
        self.timings_ms = defaultdict(list)
        self.outputs = {}

    def run(self, *args):
        interpreter = self

        class Impl(interpreter._fx.Interpreter):
            def run_node(self, node):
                if interpreter._device.type == "cuda":
                    interpreter._torch.cuda.synchronize(interpreter._device)
                start = perf_counter()
                result = super().run_node(node)
                if interpreter._device.type == "cuda":
                    interpreter._torch.cuda.synchronize(interpreter._device)
                elapsed_ms = (perf_counter() - start) * 1000.0
                interpreter.timings_ms[node.name].append(elapsed_ms)
                if interpreter._capture_outputs:
                    interpreter.outputs[node.name] = result
                return result

        with self._torch.no_grad():
            Impl(self._module).run(*args)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def validate_partition_granularity(partition_granularity: str) -> str:
    value = str(partition_granularity)
    if value not in PARTITION_GRANULARITIES:
        raise ValueError("partition_granularity must be 'node' or 'block'.")
    return value


def _resolve_model_builder(model_name: str, torchvision_models) -> Any:
    builders = {
        "mobilenet_v2": lambda: torchvision_models.mobilenet_v2(weights=None),
        "googlenet": lambda: torchvision_models.googlenet(
            weights=None,
            aux_logits=False,
            init_weights=False,
        ),
        "resnet50": lambda: torchvision_models.resnet50(weights=None),
        "vgg16": lambda: torchvision_models.vgg16(weights=None),
    }
    try:
        return builders[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {', '.join(sorted(builders))}."
        ) from exc


def _make_mobilenet_v2_block_model(base_model: Any, torch_mod: Any) -> tuple[Any, set[str]]:
    nn = torch_mod.nn
    functional = torch_mod.nn.functional

    class MobileNetHead(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, x):
            x = functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch_mod.flatten(x, 1)
            return self.classifier(x)

    class MobileNetBlockModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._block_names = []
            for index, block in enumerate(model.features):
                name = f"features_{index}"
                setattr(self, name, block)
                self._block_names.append(name)
            self.classifier = MobileNetHead(model.classifier)

        def forward(self, x):
            for name in self._block_names:
                x = getattr(self, name)(x)
            return self.classifier(x)

    leaf_modules = {f"features_{index}" for index in range(len(base_model.features))}
    leaf_modules.add("classifier")
    return MobileNetBlockModel(base_model), leaf_modules


def _make_googlenet_block_model(base_model: Any, torch_mod: Any) -> tuple[Any, set[str]]:
    nn = torch_mod.nn

    class GoogLeNetStem1(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv1 = model.conv1
            self.maxpool1 = model.maxpool1

        def forward(self, x):
            return self.maxpool1(self.conv1(x))

    class GoogLeNetStem2(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv2 = model.conv2
            self.conv3 = model.conv3
            self.maxpool2 = model.maxpool2

        def forward(self, x):
            x = self.conv2(x)
            x = self.conv3(x)
            return self.maxpool2(x)

    class PoolThenBlock(nn.Module):
        def __init__(self, pool, block):
            super().__init__()
            self.pool = pool
            self.block = block

        def forward(self, x):
            return self.block(self.pool(x))

    class GoogLeNetHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.avgpool = model.avgpool
            self.dropout = model.dropout
            self.fc = model.fc

        def forward(self, x):
            x = self.avgpool(x)
            x = torch_mod.flatten(x, 1)
            x = self.dropout(x)
            return self.fc(x)

    class GoogLeNetBlockModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.stem1 = GoogLeNetStem1(model)
            self.stem2 = GoogLeNetStem2(model)
            self.inception3a = model.inception3a
            self.inception3b = model.inception3b
            self.inception4a = PoolThenBlock(model.maxpool3, model.inception4a)
            self.inception4b = model.inception4b
            self.inception4c = model.inception4c
            self.inception4d = model.inception4d
            self.inception4e = model.inception4e
            self.inception5a = PoolThenBlock(model.maxpool4, model.inception5a)
            self.inception5b = model.inception5b
            self.head = GoogLeNetHead(model)

        def forward(self, x):
            x = self.stem1(x)
            x = self.stem2(x)
            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.inception4a(x)
            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)
            x = self.inception4e(x)
            x = self.inception5a(x)
            x = self.inception5b(x)
            return self.head(x)

    leaf_modules = {
        "stem1",
        "stem2",
        "inception3a",
        "inception3b",
        "inception4a",
        "inception4b",
        "inception4c",
        "inception4d",
        "inception4e",
        "inception5a",
        "inception5b",
        "head",
    }
    return GoogLeNetBlockModel(base_model), leaf_modules


def _make_resnet50_block_model(base_model: Any, torch_mod: Any) -> tuple[Any, set[str]]:
    nn = torch_mod.nn

    class ResNetStem(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return self.maxpool(x)

    class ResNetHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.avgpool = model.avgpool
            self.fc = model.fc

        def forward(self, x):
            x = self.avgpool(x)
            x = torch_mod.flatten(x, 1)
            return self.fc(x)

    class ResNetBlockModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.stem = ResNetStem(model)
            self._block_names = []
            for layer_index in range(1, 5):
                layer = getattr(model, f"layer{layer_index}")
                for block_index, block in enumerate(layer):
                    name = f"layer{layer_index}_{block_index}"
                    setattr(self, name, block)
                    self._block_names.append(name)
            self.head = ResNetHead(model)

        def forward(self, x):
            x = self.stem(x)
            for name in self._block_names:
                x = getattr(self, name)(x)
            return self.head(x)

    wrapper = ResNetBlockModel(base_model)
    leaf_modules = {"stem", "head", *wrapper._block_names}
    return wrapper, leaf_modules


def _make_vgg16_block_model(base_model: Any, torch_mod: Any) -> tuple[Any, set[str]]:
    nn = torch_mod.nn
    feature_layers = list(base_model.features.children())

    class VGGHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.avgpool = model.avgpool
            self.classifier = model.classifier

        def forward(self, x):
            x = self.avgpool(x)
            x = torch_mod.flatten(x, 1)
            return self.classifier(x)

    class VGGBlockModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.features_block1 = nn.Sequential(*feature_layers[0:5])
            self.features_block2 = nn.Sequential(*feature_layers[5:10])
            self.features_block3 = nn.Sequential(*feature_layers[10:17])
            self.features_block4 = nn.Sequential(*feature_layers[17:24])
            self.features_block5 = nn.Sequential(*feature_layers[24:31])
            self.head = VGGHead(model)

        def forward(self, x):
            x = self.features_block1(x)
            x = self.features_block2(x)
            x = self.features_block3(x)
            x = self.features_block4(x)
            x = self.features_block5(x)
            return self.head(x)

    leaf_modules = {
        "features_block1",
        "features_block2",
        "features_block3",
        "features_block4",
        "features_block5",
        "head",
    }
    return VGGBlockModel(base_model), leaf_modules


def _make_block_model(model_name: str, base_model: Any, torch_mod: Any) -> tuple[Any, set[str]]:
    builders = {
        "mobilenet_v2": _make_mobilenet_v2_block_model,
        "googlenet": _make_googlenet_block_model,
        "resnet50": _make_resnet50_block_model,
        "vgg16": _make_vgg16_block_model,
    }
    try:
        return builders[model_name](base_model, torch_mod)
    except KeyError as exc:
        raise ValueError(f"Block partitioning is unsupported for model '{model_name}'.") from exc


def build_partition_model(model_name: str, torchvision_models: Any, torch_mod: Any, partition_granularity: str = "node") -> tuple[Any, set[str]]:
    partition_granularity = validate_partition_granularity(partition_granularity)
    base_model = _resolve_model_builder(model_name, torchvision_models)()
    base_model.eval()
    if partition_granularity == "node":
        return base_model, set()
    block_model, leaf_module_names = _make_block_model(model_name, base_model, torch_mod)
    block_model.eval()
    return block_model, leaf_module_names


def trace_partition_model(model: Any, fx: Any, leaf_module_names: set[str]):
    if not leaf_module_names:
        return fx.symbolic_trace(model)

    class BlockTracer(fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if module_qualified_name in leaf_module_names:
                return True
            return super().is_leaf_module(module, module_qualified_name)

    tracer = BlockTracer()
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)


def _profile_graph_module(traced, sample, device, torch_mod, fx, warmup: int, runs: int, capture_outputs: bool):
    timed_module = copy.deepcopy(traced).to(device)
    timed_module.eval()
    device_sample = sample.to(device)

    for _ in range(max(warmup, 0)):
        runner = _ProfilingInterpreter(fx, timed_module, torch_mod, device, capture_outputs=False)
        runner.run(device_sample)

    profiler = _ProfilingInterpreter(fx, timed_module, torch_mod, device, capture_outputs=capture_outputs)
    for _ in range(max(runs, 1)):
        profiler.run(device_sample)
    return profiler


def _nearest_kept_predecessors(node, keep_names: set[str], cache: dict[str, set[str]]) -> set[str]:
    if node.name in cache:
        return cache[node.name]
    predecessors: set[str] = set()
    for input_node in node.all_input_nodes:
        if input_node.name in keep_names:
            predecessors.add(input_node.name)
        else:
            predecessors.update(_nearest_kept_predecessors(input_node, keep_names, cache))
    cache[node.name] = predecessors
    return predecessors


@dataclass
class ProfileRunConfig:
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    edge_warmup_runs: int = 1
    edge_profile_runs: int = 3
    cloud_warmup_runs: int = 1
    cloud_profile_runs: int = 3
    cloud_device: str = "auto"
    partition_granularity: str = "node"


def profile_model(model_name: str, config: Optional[ProfileRunConfig] = None) -> ModelProfile:
    config = config or ProfileRunConfig()
    partition_granularity = validate_partition_granularity(config.partition_granularity)
    torch, fx, torchvision_models = _require_torch()

    model, leaf_module_names = build_partition_model(model_name, torchvision_models, torch, partition_granularity)
    traced = trace_partition_model(model, fx, leaf_module_names)
    sample = torch.randn(*config.input_shape)

    edge_profiler = _profile_graph_module(
        traced=traced,
        sample=sample,
        device=torch.device("cpu"),
        torch_mod=torch,
        fx=fx,
        warmup=config.edge_warmup_runs,
        runs=config.edge_profile_runs,
        capture_outputs=True,
    )

    if config.cloud_device == "cpu":
        cloud_device = torch.device("cpu")
    elif config.cloud_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cloud_device='cuda' requested but CUDA is not available.")
        cloud_device = torch.device("cuda")
    else:
        cloud_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cloud_device.type == "cpu":
        cloud_timings = {name: list(values) for name, values in edge_profiler.timings_ms.items()}
        cloud_device_label = "cpu"
    else:
        cloud_profiler = _profile_graph_module(
            traced=traced,
            sample=sample,
            device=cloud_device,
            torch_mod=torch,
            fx=fx,
            warmup=config.cloud_warmup_runs,
            runs=config.cloud_profile_runs,
            capture_outputs=False,
        )
        cloud_timings = dict(cloud_profiler.timings_ms)
        cloud_device_label = str(cloud_device)

    keep_names = {
        node.name
        for node in traced.graph.nodes
        if _is_cut_candidate(node, edge_profiler.outputs.get(node.name))
    }

    cache: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {name: set() for name in keep_names}
    for node in traced.graph.nodes:
        if node.name not in keep_names:
            continue
        for predecessor in _nearest_kept_predecessors(node, keep_names, cache):
            successors[predecessor].add(node.name)

    nodes: list[ModelProfileNode] = []
    for node in traced.graph.nodes:
        if node.name not in keep_names:
            continue
        op_type = _op_type(node, traced)
        nodes.append(
            ModelProfileNode(
                id=node.name,
                op_type=op_type,
                succ_ids=sorted(successors[node.name]),
                edge_ms=_mean(edge_profiler.timings_ms[node.name]),
                cloud_ms=_mean(list(cloud_timings[node.name])),
                output_bytes=_tensor_bytes(edge_profiler.outputs[node.name]),
            )
        )

    return ModelProfile(
        model_name=model_name,
        input_shape=list(config.input_shape),
        nodes=nodes,
        metadata={
            "edge_device": "cpu",
            "cloud_device": cloud_device_label,
            "edge_profile_runs": config.edge_profile_runs,
            "cloud_profile_runs": config.cloud_profile_runs if cloud_device.type == "cuda" else config.edge_profile_runs,
            "partition_granularity": partition_granularity,
        },
    )
