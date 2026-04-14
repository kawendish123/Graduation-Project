from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from .dsl import VIRTUAL_INPUT_ID
from .profile import _require_torch, build_partition_model, trace_partition_model, validate_partition_granularity


@dataclass
class RuntimeModel:
    torch: Any
    fx: Any
    traced: Any
    device: Any
    profile_node_ids: set[str]
    partition_granularity: str


def build_runtime_model(
    model_name: str,
    device_name: str = "cpu",
    profile_node_ids: Optional[set[str]] = None,
    partition_granularity: str = "node",
) -> RuntimeModel:
    partition_granularity = validate_partition_granularity(partition_granularity)
    torch, fx, torchvision_models = _require_torch()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)
    model, leaf_module_names = build_partition_model(model_name, torchvision_models, torch, partition_granularity)
    traced = trace_partition_model(model, fx, leaf_module_names).to(device)
    traced.eval()
    return RuntimeModel(
        torch=torch,
        fx=fx,
        traced=traced,
        device=device,
        profile_node_ids=profile_node_ids or set(),
        partition_granularity=partition_granularity,
    )


def make_random_input(torch_mod: Any, input_shape: list[int], device_name: str = "cpu") -> Any:
    return torch_mod.randn(*tuple(input_shape), device=torch_mod.device(device_name))


def tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _load_arg(fx, node_arg: Any, env: dict[str, Any]) -> Any:
    return fx.node.map_arg(node_arg, lambda item: env[item.name])


def _args_available(fx, node: Any, env: dict[str, Any]) -> bool:
    try:
        _load_arg(fx, node.args, env)
        _load_arg(fx, node.kwargs, env)
        return True
    except KeyError:
        return False


def _execute_node(runtime: RuntimeModel, node: Any, env: dict[str, Any]) -> Any:
    args = _load_arg(runtime.fx, node.args, env)
    kwargs = _load_arg(runtime.fx, node.kwargs, env)
    if node.op == "call_module":
        return runtime.traced.get_submodule(str(node.target))(*args, **kwargs)
    if node.op == "call_function":
        return node.target(*args, **kwargs)
    if node.op == "call_method":
        self_obj, *tail = args
        return getattr(self_obj, str(node.target))(*tail, **kwargs)
    if node.op == "get_attr":
        return getattr(runtime.traced, str(node.target))
    raise RuntimeError(f"Unsupported FX node op: {node.op}")


def execute_edge_partition(
    runtime: RuntimeModel,
    input_tensor: Any,
    edge_nodes: list[str],
    cloud_nodes: list[str],
    transmission_nodes: list[str],
) -> tuple[dict[str, Any], float, Optional[Any]]:
    torch = runtime.torch
    edge_set = set(edge_nodes)
    cloud_set = set(cloud_nodes)
    transmission_set = set(transmission_nodes)
    captured: dict[str, Any] = {}
    env: dict[str, Any] = {}
    output_value = None

    input_tensor = input_tensor.to(runtime.device)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    start = perf_counter()
    with torch.no_grad():
        for node in runtime.traced.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = input_tensor
                if VIRTUAL_INPUT_ID in transmission_set:
                    captured[VIRTUAL_INPUT_ID] = input_tensor.detach().cpu()
                continue
            if node.op == "output":
                if _args_available(runtime.fx, node, env):
                    output_value = _load_arg(runtime.fx, node.args[0], env)
                continue
            is_profile_node = node.name in runtime.profile_node_ids
            should_execute = node.name in edge_set or (not is_profile_node and _args_available(runtime.fx, node, env))
            if node.name in cloud_set:
                should_execute = False
            if not should_execute:
                continue
            if not _args_available(runtime.fx, node, env):
                continue
            env[node.name] = _execute_node(runtime, node, env)
            if node.name in transmission_set:
                captured[node.name] = env[node.name].detach().cpu()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    edge_ms = (perf_counter() - start) * 1000.0
    return captured, edge_ms, output_value


def execute_cloud_partition(
    runtime: RuntimeModel,
    tensors: dict[str, Any],
    cloud_nodes: list[str],
) -> tuple[float, Optional[Any]]:
    torch = runtime.torch
    cloud_set = set(cloud_nodes)
    env: dict[str, Any] = {node_id: tensor.to(runtime.device) for node_id, tensor in tensors.items() if node_id != VIRTUAL_INPUT_ID}
    input_tensor = tensors.get(VIRTUAL_INPUT_ID)
    if input_tensor is not None:
        input_tensor = input_tensor.to(runtime.device)
    output_value = None

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    start = perf_counter()
    with torch.no_grad():
        for node in runtime.traced.graph.nodes:
            if node.op == "placeholder":
                if input_tensor is not None:
                    env[node.name] = input_tensor
                continue
            if node.op == "output":
                if _args_available(runtime.fx, node, env):
                    output_value = _load_arg(runtime.fx, node.args[0], env)
                continue
            if node.name in env:
                continue
            is_profile_node = node.name in runtime.profile_node_ids
            should_execute = node.name in cloud_set or (not is_profile_node and _args_available(runtime.fx, node, env))
            if not should_execute or not _args_available(runtime.fx, node, env):
                continue
            env[node.name] = _execute_node(runtime, node, env)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    cloud_ms = (perf_counter() - start) * 1000.0
    return cloud_ms, output_value
