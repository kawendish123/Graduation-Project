from __future__ import annotations

import csv
from dataclasses import replace
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
    "edge_latency_mode",
    "edge_condition",
    "edge_slowdown_factor",
    "edge_condition_order",
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


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


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


def scale_edge_profile(profile: ModelProfile, edge_slowdown_factor: float, edge_condition: str) -> ModelProfile:
    factor = float(edge_slowdown_factor)
    if factor <= 0:
        raise ValueError("edge_slowdown_factor must be positive.")
    metadata = dict(profile.metadata or {})
    metadata.update(
        {
            "edge_latency_mode": "scaled",
            "edge_condition": str(edge_condition),
            "edge_slowdown_factor": factor,
        }
    )
    return ModelProfile(
        model_name=profile.model_name,
        input_shape=list(profile.input_shape),
        nodes=[replace(node, edge_ms=float(node.edge_ms) * factor) for node in profile.nodes],
        metadata=metadata,
    )


def _edge_latency_mode(config: dict[str, Any]) -> str:
    value = str(config.get("edge_latency_mode", "cached")).lower()
    if value not in {"cached", "scaled"}:
        raise ValueError("edge_latency_mode must be 'cached' or 'scaled'.")
    return value


def _scaled_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    factors = config.get("edge_slowdown_factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError("scaled edge latency mode requires 'edge_slowdown_factors' to be a non-empty object.")
    conditions: list[dict[str, Any]] = []
    for order, (condition, factor) in enumerate(factors.items()):
        factor_value = float(factor)
        if factor_value <= 0:
            raise ValueError(f"edge_slowdown_factors['{condition}'] must be positive.")
        conditions.append(
            {
                "cpu_load_target": "",
                "edge_condition": str(condition),
                "edge_slowdown_factor": factor_value,
                "edge_condition_order": order,
            }
        )
    return conditions


def _strategy_summary(
    model_name: str,
    bandwidth_mbps: float,
    cpu_load_target: Any,
    edge_latency_mode: str,
    edge_condition: str,
    edge_slowdown_factor: Any,
    edge_condition_order: int,
    strategy_name: str,
    plan: StrategyPlan,
    profile_with_input: ModelProfile,
    edge_profile_path: str,
    cloud_profile_path: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "bandwidth_mbps": float(bandwidth_mbps),
        "cpu_load_target": "" if cpu_load_target == "" else float(cpu_load_target),
        "edge_latency_mode": edge_latency_mode,
        "edge_condition": edge_condition,
        "edge_slowdown_factor": edge_slowdown_factor,
        "edge_condition_order": edge_condition_order,
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
    edge_latency_mode = _edge_latency_mode(config)
    cloud_profile_path = config.get("cloud_profile")
    if not cloud_profile_path:
        raise ValueError("estimate-experiment config field 'cloud_profile' is required.")

    report_csv = str(config.get("report_csv", "results/estimate_experiment.csv"))
    report_json = str(config.get("report_json", "results/estimate_experiment.json"))
    debug_dir = config.get("debug_dir")
    plot_dir = config.get("plot_dir", "results/plots_estimate")
    plot_format = str(config.get("plot_format", "png"))

    cloud_profile = ModelProfile.load(str(cloud_profile_path))
    _validate_profile_matches_config(cloud_profile, model_name, input_shape, partition_granularity)

    input_bytes = input_bytes_from_shape(input_shape)
    csv_rows: list[dict[str, Any]] = []
    result_items: list[dict[str, Any]] = []

    if edge_latency_mode == "cached":
        cpu_load_targets = _float_list(config, "cpu_load_targets", [0, 10, 20, 30])
        edge_profile_paths = config.get("edge_profiles")
        for cpu_load_target in cpu_load_targets:
            _resolve_edge_profile_path(edge_profile_paths, cpu_load_target)
        condition_items = [
            {
                "cpu_load_target": cpu_load_target,
                "edge_condition": "",
                "edge_slowdown_factor": "",
                "edge_condition_order": order,
                "edge_profile_path": _resolve_edge_profile_path(edge_profile_paths, cpu_load_target),
                "edge_profile": None,
                "base_edge_profile": None,
            }
            for order, cpu_load_target in enumerate(cpu_load_targets)
        ]
    else:
        base_edge_profile = config.get("base_edge_profile")
        if not base_edge_profile:
            raise ValueError("scaled edge latency mode requires 'base_edge_profile'.")
        base_edge = ModelProfile.load(str(base_edge_profile))
        _validate_profile_matches_config(base_edge, model_name, input_shape, partition_granularity)
        condition_items = []
        for item in _scaled_conditions(config):
            condition_items.append(
                {
                    **item,
                    "edge_profile_path": str(base_edge_profile),
                    "edge_profile": scale_edge_profile(base_edge, item["edge_slowdown_factor"], item["edge_condition"]),
                    "base_edge_profile": str(base_edge_profile),
                }
            )

    for condition in condition_items:
        edge_profile_path = str(condition["edge_profile_path"])
        edge_profile = condition["edge_profile"]
        if edge_profile is None:
            edge_profile = ModelProfile.load(edge_profile_path)
        _validate_profile_matches_config(edge_profile, model_name, input_shape, partition_granularity)

        merged = _merge_profiles(edge_profile, cloud_profile)
        profile_with_input = with_virtual_input_node(merged, input_bytes)

        for bandwidth_mbps in bandwidths_mbps:
            dsl_solution = solve_dsl(profile_with_input, bandwidth_mbps)
            debug_output: Optional[str] = None
            if debug_dir:
                if edge_latency_mode == "scaled":
                    condition_token = _safe_token(str(condition["edge_condition"]))
                    debug_name = f"{model_name}_{condition_token}_bw{_number_token(bandwidth_mbps)}.json"
                else:
                    debug_name = f"{model_name}_load{_number_token(float(condition['cpu_load_target']))}_bw{_number_token(bandwidth_mbps)}.json"
                debug_output = str(Path(str(debug_dir)) / debug_name)
                _write_json(debug_output, build_debug_payload(profile_with_input, bandwidth_mbps, dsl_solution))

            plans = build_strategy_plans(profile_with_input, bandwidth_mbps, dsl_solution)
            strategy_summaries: dict[str, dict[str, Any]] = {}
            for strategy_name in ["dsl", "pure_edge", "pure_cloud"]:
                summary = _strategy_summary(
                    model_name=model_name,
                    bandwidth_mbps=bandwidth_mbps,
                    cpu_load_target=condition["cpu_load_target"],
                    edge_latency_mode=edge_latency_mode,
                    edge_condition=str(condition["edge_condition"]),
                    edge_slowdown_factor=condition["edge_slowdown_factor"],
                    edge_condition_order=int(condition["edge_condition_order"]),
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
                        "cpu_load_target": condition["cpu_load_target"],
                        "edge_latency_mode": edge_latency_mode,
                        "edge_condition": condition["edge_condition"],
                        "edge_slowdown_factor": condition["edge_slowdown_factor"],
                        "edge_condition_order": condition["edge_condition_order"],
                        "base_edge_profile": condition.get("base_edge_profile"),
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
