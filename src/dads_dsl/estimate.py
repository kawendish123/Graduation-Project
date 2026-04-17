from __future__ import annotations

import csv
import json
from math import prod
from pathlib import Path
from typing import Any, Optional

from .client import _merge_profiles
from .dsl import build_debug_payload, solve_dsl, with_virtual_input_node
from .experiment import (
    StrategyPlan,
    _float_list,
    _format_csv_value,
    _resolve_edge_profile_path,
    build_strategy_plans,
)
from .profile import validate_partition_granularity
from .types import ModelProfile


ESTIMATE_CSV_COLUMNS = [
    "model",
    "bandwidth_mbps",
    "cpu_load_target",
    "strategy",
    "split_nodes",
    "edge_node_count",
    "cloud_node_count",
    "estimated_edge_ms",
    "estimated_transfer_ms",
    "estimated_cloud_ms",
    "estimated_total_ms",
    "payload_bytes",
    "profile_edge_path",
    "profile_cloud_path",
]


def input_bytes_from_shape(input_shape: list[int], element_size_bytes: int = 4) -> int:
    if element_size_bytes <= 0:
        raise ValueError("element_size_bytes must be positive.")
    if not input_shape:
        raise ValueError("input_shape must not be empty.")
    cleaned_shape = [int(dim) for dim in input_shape]
    if any(dim <= 0 for dim in cleaned_shape):
        raise ValueError("input_shape dimensions must be positive.")
    return int(prod(cleaned_shape) * element_size_bytes)


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ESTIMATE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column, "")) for column in ESTIMATE_CSV_COLUMNS})


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _payload_bytes_for_plan(profile: ModelProfile, plan: StrategyPlan) -> int:
    node_map = profile.node_map()
    return int(sum(node_map[node_id].output_bytes for node_id in plan.transmission_nodes))


def _validate_profile_matches_config(profile: ModelProfile, model_name: str, input_shape: list[int], partition_granularity: str) -> None:
    if profile.model_name != model_name:
        raise ValueError(f"Profile model mismatch. config={model_name}, profile={profile.model_name}")
    if list(profile.input_shape) != list(input_shape):
        raise ValueError(f"Profile input_shape mismatch. config={input_shape}, profile={profile.input_shape}")
    profile_granularity = (profile.metadata or {}).get("partition_granularity", "node")
    if profile_granularity != partition_granularity:
        raise ValueError(f"Profile partition granularity mismatch. config={partition_granularity}, profile={profile_granularity}")


def _strategy_summary(
    model_name: str,
    bandwidth_mbps: float,
    cpu_load_target: float,
    strategy_name: str,
    plan: StrategyPlan,
    profile_with_input: ModelProfile,
    edge_profile_path: str,
    cloud_profile_path: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "bandwidth_mbps": float(bandwidth_mbps),
        "cpu_load_target": float(cpu_load_target),
        "strategy": strategy_name,
        "split_nodes": plan.transmission_nodes,
        "edge_node_count": len(plan.edge_nodes),
        "cloud_node_count": len(plan.cloud_nodes),
        "estimated_edge_ms": float(plan.estimated_edge_ms),
        "estimated_transfer_ms": float(plan.estimated_transfer_ms),
        "estimated_cloud_ms": float(plan.estimated_cloud_ms),
        "estimated_total_ms": float(plan.estimated_total_ms),
        "payload_bytes": _payload_bytes_for_plan(profile_with_input, plan),
        "profile_edge_path": edge_profile_path,
        "profile_cloud_path": cloud_profile_path,
    }


def run_estimate_experiment(config: dict[str, Any]) -> dict[str, Any]:
    model_name = str(config.get("model", "mobilenet_v2"))
    partition_granularity = validate_partition_granularity(str(config.get("partition_granularity", "node")))
    input_shape = [int(item) for item in config.get("input_shape", [1, 3, 224, 224])]
    bandwidths_mbps = _float_list(config, "bandwidths_mbps", [1, 2, 5, 10, 20, 50, 100])
    cpu_load_targets = _float_list(config, "cpu_load_targets", [0, 10, 20, 30])
    edge_profile_paths = config.get("edge_profiles")
    cloud_profile_path = config.get("cloud_profile")
    if not cloud_profile_path:
        raise ValueError("estimate-experiment config field 'cloud_profile' is required.")

    report_csv = str(config.get("report_csv", "results/estimate_experiment.csv"))
    report_json = str(config.get("report_json", "results/estimate_experiment.json"))
    debug_dir = config.get("debug_dir")
    plot_dir = config.get("plot_dir", "results/plots_estimate")
    plot_format = str(config.get("plot_format", "png"))

    for cpu_load_target in cpu_load_targets:
        _resolve_edge_profile_path(edge_profile_paths, cpu_load_target)

    cloud_profile = ModelProfile.load(str(cloud_profile_path))
    _validate_profile_matches_config(cloud_profile, model_name, input_shape, partition_granularity)

    input_bytes = input_bytes_from_shape(input_shape)
    csv_rows: list[dict[str, Any]] = []
    result_items: list[dict[str, Any]] = []

    for cpu_load_target in cpu_load_targets:
        edge_profile_path = _resolve_edge_profile_path(edge_profile_paths, cpu_load_target)
        edge_profile = ModelProfile.load(edge_profile_path)
        _validate_profile_matches_config(edge_profile, model_name, input_shape, partition_granularity)

        merged = _merge_profiles(edge_profile, cloud_profile)
        profile_with_input = with_virtual_input_node(merged, input_bytes)

        for bandwidth_mbps in bandwidths_mbps:
            dsl_solution = solve_dsl(profile_with_input, bandwidth_mbps)
            debug_output: Optional[str] = None
            if debug_dir:
                debug_output = str(
                    Path(str(debug_dir))
                    / f"{model_name}_load{_number_token(cpu_load_target)}_bw{_number_token(bandwidth_mbps)}.json"
                )
                _write_json(debug_output, build_debug_payload(profile_with_input, bandwidth_mbps, dsl_solution))

            plans = build_strategy_plans(profile_with_input, bandwidth_mbps, dsl_solution)
            strategy_summaries: dict[str, dict[str, Any]] = {}
            for strategy_name in ["dsl", "pure_edge", "pure_cloud"]:
                summary = _strategy_summary(
                    model_name=model_name,
                    bandwidth_mbps=bandwidth_mbps,
                    cpu_load_target=cpu_load_target,
                    strategy_name=strategy_name,
                    plan=plans[strategy_name],
                    profile_with_input=profile_with_input,
                    edge_profile_path=edge_profile_path,
                    cloud_profile_path=str(cloud_profile_path),
                )
                csv_rows.append(summary)
                strategy_summaries[strategy_name] = summary

            result_items.append(
                {
                    "condition": {
                        "model": model_name,
                        "bandwidth_mbps": bandwidth_mbps,
                        "cpu_load_target": cpu_load_target,
                    },
                    "strategy_summaries": strategy_summaries,
                    "dsl_partition": dsl_solution.to_dict(),
                    "debug_output": debug_output,
                }
            )

    output_payload = {"config": config, "results": result_items}
    _write_csv(report_csv, csv_rows)
    _write_json(report_json, output_payload)

    plot_outputs = []
    if plot_dir:
        from .plotting import (
            plot_estimate_latency,
            plot_estimate_latency_by_bandwidth,
            plot_estimate_speedup_heatmap,
            plot_estimate_stage_breakdown,
        )

        plot_outputs = plot_estimate_latency(report_csv, str(plot_dir), plot_format=plot_format)
        plot_outputs.extend(plot_estimate_latency_by_bandwidth(report_csv, str(plot_dir), plot_format=plot_format))
        if bool(config.get("plot_heatmap", True)):
            plot_outputs.extend(plot_estimate_speedup_heatmap(report_csv, str(plot_dir), plot_format=plot_format))
        if bool(config.get("plot_stage_breakdown", True)):
            plot_outputs.extend(plot_estimate_stage_breakdown(report_csv, str(plot_dir), plot_format=plot_format))
    print(f"Estimate experiment complete. CSV: {report_csv} JSON: {report_json}")
    if plot_outputs:
        print("Plots: " + ", ".join(plot_outputs))
    return output_payload
