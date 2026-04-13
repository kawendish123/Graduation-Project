from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter, sleep
from typing import Any, Callable, Optional

from .client import _merge_profiles, _request_cloud_profile
from .cpu_load import CpuLoadController
from .dsl import VIRTUAL_INPUT_ID, build_debug_payload, compute_transmission_profile, solve_dsl, with_virtual_input_node
from .profile import ProfileRunConfig, profile_model
from .rpc import DadsCloudStub, make_channel
from .runtime import build_runtime_model, execute_edge_partition, make_random_input, tensor_nbytes
from .tensor_codec import payload_nbytes, tensor_to_payload
from .types import DSLSolution, ModelProfile


EXPERIMENT_CSV_COLUMNS = [
    "model",
    "bandwidth_mbps",
    "cpu_load_target",
    "strategy",
    "repeats",
    "actual_warmup_runs",
    "split_nodes",
    "edge_node_count",
    "cloud_node_count",
    "estimated_total_ms",
    "actual_total_mean_ms",
    "actual_total_std_ms",
    "edge_actual_mean_ms",
    "edge_actual_std_ms",
    "transmission_actual_mean_ms",
    "transmission_actual_std_ms",
    "cloud_actual_mean_ms",
    "cloud_actual_std_ms",
    "payload_bytes_mean",
    "payload_bytes_std",
    "cpu_load_avg",
    "cpu_load_min",
    "cpu_load_max",
]


@dataclass
class StrategyPlan:
    name: str
    edge_nodes: list[str]
    transmission_nodes: list[str]
    cloud_nodes: list[str]
    estimated_edge_ms: float
    estimated_transfer_ms: float
    estimated_cloud_ms: float
    estimated_total_ms: float


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "actual_total_mean_ms": _mean([float(item["total_actual_ms"]) for item in measurements]),
        "actual_total_std_ms": _std([float(item["total_actual_ms"]) for item in measurements]),
        "edge_actual_mean_ms": _mean([float(item["edge_actual_ms"]) for item in measurements]),
        "edge_actual_std_ms": _std([float(item["edge_actual_ms"]) for item in measurements]),
        "transmission_actual_mean_ms": _mean([float(item["transmission_actual_ms"]) for item in measurements]),
        "transmission_actual_std_ms": _std([float(item["transmission_actual_ms"]) for item in measurements]),
        "cloud_actual_mean_ms": _mean([float(item["cloud_actual_ms"]) for item in measurements]),
        "cloud_actual_std_ms": _std([float(item["cloud_actual_ms"]) for item in measurements]),
        "payload_bytes_mean": _mean([float(item["payload_bytes"]) for item in measurements]),
        "payload_bytes_std": _std([float(item["payload_bytes"]) for item in measurements]),
    }


def build_strategy_plans(profile: ModelProfile, bandwidth_mbps: float, dsl_solution: DSLSolution) -> dict[str, StrategyPlan]:
    node_map = profile.node_map()
    transmission_ms = compute_transmission_profile(profile, bandwidth_mbps)
    real_node_ids = [node.id for node in profile.nodes if node.id != VIRTUAL_INPUT_ID]
    input_node = node_map[VIRTUAL_INPUT_ID]

    dsl_plan = StrategyPlan(
        name="dsl",
        edge_nodes=list(dsl_solution.edge_nodes),
        transmission_nodes=list(dsl_solution.transmission_nodes),
        cloud_nodes=list(dsl_solution.cloud_nodes),
        estimated_edge_ms=dsl_solution.edge_stage_ms,
        estimated_transfer_ms=dsl_solution.transmission_stage_ms,
        estimated_cloud_ms=dsl_solution.cloud_stage_ms,
        estimated_total_ms=dsl_solution.total_inference_ms,
    )
    pure_edge_plan = StrategyPlan(
        name="pure_edge",
        edge_nodes=[VIRTUAL_INPUT_ID] + real_node_ids,
        transmission_nodes=[],
        cloud_nodes=[],
        estimated_edge_ms=sum(node.edge_ms for node in profile.nodes),
        estimated_transfer_ms=0.0,
        estimated_cloud_ms=0.0,
        estimated_total_ms=sum(node.edge_ms for node in profile.nodes),
    )
    pure_cloud_transfer_ms = transmission_ms[input_node.id]
    pure_cloud_ms = sum(node_map[node_id].cloud_ms for node_id in real_node_ids)
    pure_cloud_plan = StrategyPlan(
        name="pure_cloud",
        edge_nodes=[VIRTUAL_INPUT_ID],
        transmission_nodes=[VIRTUAL_INPUT_ID],
        cloud_nodes=real_node_ids,
        estimated_edge_ms=0.0,
        estimated_transfer_ms=pure_cloud_transfer_ms,
        estimated_cloud_ms=pure_cloud_ms,
        estimated_total_ms=pure_cloud_transfer_ms + pure_cloud_ms,
    )
    return {
        "dsl": dsl_plan,
        "pure_edge": pure_edge_plan,
        "pure_cloud": pure_cloud_plan,
    }


def _execute_strategy_once(
    edge_runtime: Any,
    stub: DadsCloudStub,
    model_name: str,
    input_shape: list[int],
    bandwidth_mbps: float,
    runtime_profile_node_ids: set[str],
    plan: StrategyPlan,
) -> dict[str, Any]:
    input_tensor = make_random_input(edge_runtime.torch, input_shape, "cpu")
    captured, edge_actual_ms, local_output = execute_edge_partition(
        edge_runtime,
        input_tensor,
        [node_id for node_id in plan.edge_nodes if node_id != VIRTUAL_INPUT_ID],
        plan.cloud_nodes,
        plan.transmission_nodes,
    )

    cloud_actual_ms = 0.0
    output_shape = list(local_output.shape) if hasattr(local_output, "shape") else []
    transmission_start = perf_counter()
    payloads = [tensor_to_payload(node_id, tensor) for node_id, tensor in captured.items()]
    payload_bytes = sum(payload_nbytes(payload) for payload in payloads)
    transmission_sleep_seconds = payload_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0) if payload_bytes else 0.0
    if payloads:
        if transmission_sleep_seconds > 0:
            sleep(transmission_sleep_seconds)
        response = stub.run_partition(
            {
                "model_name": model_name,
                "edge_nodes": plan.edge_nodes,
                "transmission_nodes": plan.transmission_nodes,
                "cloud_nodes": plan.cloud_nodes,
                "profile_node_ids": list(runtime_profile_node_ids),
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
    return {
        "edge_actual_ms": edge_actual_ms,
        "transmission_actual_ms": transmission_actual_ms,
        "cloud_actual_ms": cloud_actual_ms,
        "total_actual_ms": total_actual_ms,
        "payload_bytes": payload_bytes,
        "output_shape": output_shape,
    }


def _run_strategy_repeats(execute_once: Callable[[], dict[str, Any]], repeats: int, actual_warmup_runs: int) -> list[dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if actual_warmup_runs < 0:
        raise ValueError("actual_warmup_runs must be non-negative.")
    for _ in range(actual_warmup_runs):
        execute_once()

    measurements = []
    for repeat_index in range(1, repeats + 1):
        measurement = dict(execute_once())
        measurement["repeat_index"] = repeat_index
        measurements.append(measurement)
    return measurements


def _float_list(config: dict[str, Any], key: str, default: list[float]) -> list[float]:
    return [float(item) for item in config.get(key, default)]


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            copied["split_nodes"] = ";".join(copied.get("split_nodes", []))
            writer.writerow({column: copied.get(column, "") for column in EXPERIMENT_CSV_COLUMNS})


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    server = str(config.get("server", "localhost:50051"))
    model_name = str(config.get("model", "mobilenet_v2"))
    input_shape = [int(item) for item in config.get("input_shape", [1, 3, 224, 224])]
    bandwidths_mbps = _float_list(config, "bandwidths_mbps", [20, 50, 100, 200, 500, 1000])
    cpu_load_targets = _float_list(config, "cpu_load_targets", [0, 30, 60])
    repeats = int(config.get("repeats", 10))
    actual_warmup_runs = int(config.get("actual_warmup_runs", 2))
    profile_runs = int(config.get("profile_runs", 5))
    profile_warmup_runs = int(config.get("profile_warmup_runs", 3))
    cpu_load_tolerance = float(config.get("cpu_load_tolerance", 5.0))
    cpu_load_interval = float(config.get("cpu_load_interval", 0.5))
    cpu_load_ramp_seconds = float(config.get("cpu_load_ramp_seconds", 2.0))
    report_csv = str(config.get("report_csv", "results/experiment.csv"))
    report_json = str(config.get("report_json", "results/experiment.json"))
    debug_dir = config.get("debug_dir")

    channel = make_channel(server)
    stub = DadsCloudStub(channel)
    csv_rows: list[dict[str, Any]] = []
    result_items: list[dict[str, Any]] = []

    for cpu_load_target in cpu_load_targets:
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
                ),
            )
            cloud_profile = _request_cloud_profile(stub, model_name, input_shape, profile_warmup_runs, profile_runs)
            merged = _merge_profiles(edge_profile, cloud_profile)
            edge_runtime = build_runtime_model(model_name, "cpu", set(node.id for node in merged.nodes))
            sample_input = make_random_input(edge_runtime.torch, input_shape, "cpu")
            profile_with_input = with_virtual_input_node(merged, tensor_nbytes(sample_input))
            runtime_profile_node_ids = set(node.id for node in merged.nodes)

            for bandwidth_mbps in bandwidths_mbps:
                dsl_solution = solve_dsl(profile_with_input, bandwidth_mbps)
                debug_output: Optional[str] = None
                if debug_dir:
                    debug_output = str(Path(str(debug_dir)) / f"{model_name}_load{int(cpu_load_target)}_bw{int(bandwidth_mbps)}.json")
                    _write_json(debug_output, build_debug_payload(profile_with_input, bandwidth_mbps, dsl_solution))

                plans = build_strategy_plans(profile_with_input, bandwidth_mbps, dsl_solution)
                condition_repeats: dict[str, list[dict[str, Any]]] = {}
                strategy_summaries: dict[str, dict[str, Any]] = {}

                for strategy_name in ["dsl", "pure_edge", "pure_cloud"]:
                    plan = plans[strategy_name]
                    measurements = _run_strategy_repeats(
                        lambda: _execute_strategy_once(
                            edge_runtime=edge_runtime,
                            stub=stub,
                            model_name=model_name,
                            input_shape=input_shape,
                            bandwidth_mbps=bandwidth_mbps,
                            runtime_profile_node_ids=runtime_profile_node_ids,
                            plan=plan,
                        ),
                        repeats=repeats,
                        actual_warmup_runs=actual_warmup_runs,
                    )
                    stats = summarize_measurements(measurements)
                    cpu_stats = load_controller.stats()
                    summary = {
                        "model": model_name,
                        "bandwidth_mbps": bandwidth_mbps,
                        "cpu_load_target": cpu_load_target,
                        "strategy": strategy_name,
                        "repeats": repeats,
                        "actual_warmup_runs": actual_warmup_runs,
                        "split_nodes": plan.transmission_nodes,
                        "edge_node_count": len(plan.edge_nodes),
                        "cloud_node_count": len(plan.cloud_nodes),
                        "estimated_total_ms": plan.estimated_total_ms,
                        **stats,
                        "cpu_load_avg": cpu_stats["cpu_load_avg"],
                        "cpu_load_min": cpu_stats["cpu_load_min"],
                        "cpu_load_max": cpu_stats["cpu_load_max"],
                    }
                    csv_rows.append(summary)
                    condition_repeats[strategy_name] = measurements
                    strategy_summaries[strategy_name] = {
                        **summary,
                        "estimated_edge_ms": plan.estimated_edge_ms,
                        "estimated_transfer_ms": plan.estimated_transfer_ms,
                        "estimated_cloud_ms": plan.estimated_cloud_ms,
                    }

                result_items.append(
                    {
                        "condition": {
                            "model": model_name,
                            "bandwidth_mbps": bandwidth_mbps,
                            "cpu_load_target": cpu_load_target,
                        },
                        "strategy_summaries": strategy_summaries,
                        "repeats": condition_repeats,
                        "dsl_partition": dsl_solution.to_dict(),
                        "debug_output": debug_output,
                    }
                )

    output_payload = {"config": config, "results": result_items}
    _write_csv(report_csv, csv_rows)
    _write_json(report_json, output_payload)
    print(f"Experiment complete. CSV: {report_csv} JSON: {report_json}")
    return output_payload
