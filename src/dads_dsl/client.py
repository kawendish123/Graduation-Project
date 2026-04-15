from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Optional

from .cpu_load import CpuLoadController
from .dsl import VIRTUAL_INPUT_ID, build_debug_payload, solve_dsl, with_virtual_input_node
from .profile import ProfileRunConfig, profile_model
from .report import write_report
from .rpc import DadsCloudStub, make_channel
from .runtime import build_runtime_model, execute_edge_partition, make_random_input, tensor_nbytes
from .tensor_codec import payload_nbytes, tensor_to_payload
from .types import ModelProfile, ModelProfileNode


def _merge_profiles(edge_profile: ModelProfile, cloud_profile: ModelProfile) -> ModelProfile:
    if edge_profile.model_name != cloud_profile.model_name:
        raise ValueError(f"Profile model mismatch. edge={edge_profile.model_name}, cloud={cloud_profile.model_name}")
    if list(edge_profile.input_shape) != list(cloud_profile.input_shape):
        raise ValueError(f"Profile input_shape mismatch. edge={edge_profile.input_shape}, cloud={cloud_profile.input_shape}")

    edge_granularity = (edge_profile.metadata or {}).get("partition_granularity", "node")
    cloud_granularity = (cloud_profile.metadata or {}).get("partition_granularity", "node")
    if edge_granularity != cloud_granularity:
        raise ValueError(f"Profile partition granularity mismatch. edge={edge_granularity}, cloud={cloud_granularity}")

    edge_map = edge_profile.node_map()
    cloud_map = cloud_profile.node_map()
    if set(edge_map) != set(cloud_map):
        missing_edge = sorted(set(cloud_map) - set(edge_map))
        missing_cloud = sorted(set(edge_map) - set(cloud_map))
        raise ValueError(f"Profile node mismatch. missing_edge={missing_edge}, missing_cloud={missing_cloud}")

    nodes: list[ModelProfileNode] = []
    for edge_node in edge_profile.nodes:
        cloud_node = cloud_map[edge_node.id]
        if edge_node.succ_ids != cloud_node.succ_ids or edge_node.op_type != cloud_node.op_type:
            raise ValueError(f"Profile topology mismatch at node {edge_node.id}.")
        nodes.append(
            ModelProfileNode(
                id=edge_node.id,
                op_type=edge_node.op_type,
                succ_ids=list(edge_node.succ_ids),
                edge_ms=edge_node.edge_ms,
                cloud_ms=cloud_node.cloud_ms,
                output_bytes=edge_node.output_bytes,
            )
        )
    return ModelProfile(
        model_name=edge_profile.model_name,
        input_shape=list(edge_profile.input_shape),
        nodes=nodes,
        metadata={"edge": edge_profile.metadata or {}, "cloud": cloud_profile.metadata or {}},
    )


def _request_cloud_profile(
    stub: DadsCloudStub,
    model_name: str,
    input_shape: list[int],
    warmup_runs: int,
    profile_runs: int,
    partition_granularity: str = "node",
) -> ModelProfile:
    response = stub.get_cloud_profile(
        {
            "model_name": model_name,
            "input_shape": input_shape,
            "warmup_runs": warmup_runs,
            "profile_runs": profile_runs,
            "partition_granularity": partition_granularity,
        }
    )
    if response.get("status") != "ok":
        raise RuntimeError(response.get("error_message", "GetCloudProfile failed."))
    return ModelProfile.from_dict(response["profile"])


def _shape_of(value: Any) -> list[int]:
    return list(value.shape) if hasattr(value, "shape") else []


def run_client_once(
    server: str,
    model_name: str,
    bandwidth_mbps: float,
    cpu_load_target: float,
    cpu_load_tolerance: float,
    cpu_load_interval: float,
    cpu_load_ramp_seconds: float,
    report_format: str,
    report_output: Optional[str],
    input_shape: list[int],
    profile_runs: int,
    profile_warmup_runs: int,
    debug_output: Optional[str] = None,
    partition_granularity: str = "node",
) -> dict[str, Any]:
    channel = make_channel(server)
    stub = DadsCloudStub(channel)

    with CpuLoadController(cpu_load_target, cpu_load_tolerance, cpu_load_interval) as load_controller:
        load_controller.ramp(cpu_load_ramp_seconds)
        edge_profile = profile_model(
            model_name,
            ProfileRunConfig(
                input_shape=tuple(input_shape),
                edge_warmup_runs=profile_warmup_runs,
                edge_profile_runs=profile_runs,
                cloud_warmup_runs=0,
                cloud_profile_runs=1,
                cloud_device="cpu",
                partition_granularity=partition_granularity,
            ),
        )
        cloud_profile = _request_cloud_profile(stub, model_name, input_shape, profile_warmup_runs, profile_runs, partition_granularity)
        merged = _merge_profiles(edge_profile, cloud_profile)

        edge_runtime = build_runtime_model(model_name, "cpu", set(node.id for node in merged.nodes), partition_granularity)
        input_tensor = make_random_input(edge_runtime.torch, input_shape, "cpu")
        profile_with_input = with_virtual_input_node(merged, tensor_nbytes(input_tensor))
        solution = solve_dsl(profile_with_input, bandwidth_mbps)

        runtime_profile_node_ids = set(node.id for node in merged.nodes)
        captured, edge_actual_ms, local_output = execute_edge_partition(
            edge_runtime,
            input_tensor,
            [node_id for node_id in solution.edge_nodes if node_id != VIRTUAL_INPUT_ID],
            solution.cloud_nodes,
            solution.transmission_nodes,
        )

        payloads = [tensor_to_payload(node_id, tensor) for node_id, tensor in captured.items()]
        payload_bytes = sum(payload_nbytes(payload) for payload in payloads)
        transmission_sleep_seconds = payload_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0) if payload_bytes else 0.0

        cloud_actual_ms = 0.0
        output_shape = _shape_of(local_output)
        transmission_start = perf_counter()
        if payloads:
            if transmission_sleep_seconds > 0:
                sleep(transmission_sleep_seconds)
            response = stub.run_partition(
                {
                    "model_name": model_name,
                    "edge_nodes": solution.edge_nodes,
                    "transmission_nodes": solution.transmission_nodes,
                    "cloud_nodes": solution.cloud_nodes,
                    "profile_node_ids": list(runtime_profile_node_ids),
                    "partition_granularity": partition_granularity,
                    "tensors": payloads,
                }
            )
            if response.get("status") != "ok":
                raise RuntimeError(response.get("error_message", "RunPartition failed."))
            cloud_actual_ms = float(response["cloud_actual_ms"])
            output_shape = response.get("output_shape", [])
        transmission_and_cloud_ms = (perf_counter() - transmission_start) * 1000.0
        transmission_actual_ms = max(0.0, transmission_and_cloud_ms - cloud_actual_ms) if payloads else 0.0
        total_actual_ms = edge_actual_ms + transmission_actual_ms + cloud_actual_ms
        cpu_stats = load_controller.stats()

    all_edge_estimated_ms = sum(node.edge_ms for node in merged.nodes)
    row = {
        "model": model_name,
        "bandwidth_mbps": bandwidth_mbps,
        "cpu_load_target": cpu_stats["cpu_load_target"],
        "cpu_load_avg": cpu_stats["cpu_load_avg"],
        "split_nodes": solution.transmission_nodes,
        "edge_node_count": len(solution.edge_nodes),
        "cloud_node_count": len(solution.cloud_nodes),
        "dsl_estimated_edge_ms": solution.edge_stage_ms,
        "dsl_estimated_transfer_ms": solution.transmission_stage_ms,
        "dsl_estimated_cloud_ms": solution.cloud_stage_ms,
        "dsl_estimated_total_ms": solution.total_inference_ms,
        "all_edge_estimated_ms": all_edge_estimated_ms,
        "edge_actual_ms": edge_actual_ms,
        "transmission_actual_ms": transmission_actual_ms,
        "cloud_actual_ms": cloud_actual_ms,
        "total_actual_ms": total_actual_ms,
        "payload_bytes": payload_bytes,
    }
    full_payload = {
        "summary": row,
        "cpu": cpu_stats,
        "output_shape": output_shape,
        "partition": solution.to_dict(),
    }
    if debug_output:
        debug_payload = build_debug_payload(profile_with_input, bandwidth_mbps, solution)
        debug_path = Path(debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    rendered = write_report(row, full_payload, report_format, report_output)
    print(rendered)
    return full_payload
