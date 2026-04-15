from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .dsl import solve_bandwidth_sweep, solve_dsl
from .profile import AVAILABLE_MODELS, PARTITION_GRANULARITIES, ProfileRunConfig, profile_model
from .types import ModelProfile


def _parse_bandwidths(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def _load_config(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return payload


def _get_input_shape(config: dict[str, Any]) -> list[int]:
    value = config.get("input_shape", [1, 3, 224, 224])
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("client-run config field 'input_shape' must be a list of four integers.")
    return [int(item) for item in value]


def _get_partition_granularity(config: dict[str, Any]) -> str:
    value = str(config.get("partition_granularity", "node"))
    if value not in PARTITION_GRANULARITIES:
        raise ValueError("config field 'partition_granularity' must be 'node' or 'block'.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DADS DSL prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Export a model profile JSON")
    profile_parser.add_argument("--model", required=True, choices=list(AVAILABLE_MODELS))
    profile_parser.add_argument("--output", required=True)
    profile_parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 224, 224])
    profile_parser.add_argument("--edge-warmup-runs", type=int, default=1)
    profile_parser.add_argument("--edge-profile-runs", type=int, default=3)
    profile_parser.add_argument("--cloud-warmup-runs", type=int, default=1)
    profile_parser.add_argument("--cloud-profile-runs", type=int, default=3)
    profile_parser.add_argument("--cloud-device", choices=["auto", "cpu", "cuda"], default="auto")
    profile_parser.add_argument("--partition-granularity", choices=list(PARTITION_GRANULARITIES), default="node")

    profile_cache_parser = subparsers.add_parser("profile-cache", help="Generate an edge or cloud profile cache JSON")
    profile_cache_parser.add_argument("--role", required=True, choices=["edge", "cloud"])
    profile_cache_parser.add_argument("--model", required=True, choices=list(AVAILABLE_MODELS))
    profile_cache_parser.add_argument("--output", required=True)
    profile_cache_parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 224, 224])
    profile_cache_parser.add_argument("--warmup-runs", type=int, default=1)
    profile_cache_parser.add_argument("--profile-runs", type=int, default=3)
    profile_cache_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    profile_cache_parser.add_argument("--cpu-load-target", type=float, default=0.0)
    profile_cache_parser.add_argument("--cpu-load-tolerance", type=float, default=5.0)
    profile_cache_parser.add_argument("--cpu-load-interval", type=float, default=0.5)
    profile_cache_parser.add_argument("--cpu-load-ramp-seconds", type=float, default=2.0)
    profile_cache_parser.add_argument("--partition-granularity", choices=list(PARTITION_GRANULARITIES), default="node")

    dsl_parser = subparsers.add_parser("dsl", help="Solve DSL for one bandwidth")
    dsl_parser.add_argument("--profile", required=True)
    dsl_parser.add_argument("--bandwidth-mbps", type=float, required=True)
    dsl_parser.add_argument("--output")

    simulate_parser = subparsers.add_parser("simulate", help="Sweep bandwidth values")
    simulate_parser.add_argument("--profile", required=True)
    simulate_parser.add_argument("--bandwidths", nargs="+", required=True)
    simulate_parser.add_argument("--output")

    serve_parser = subparsers.add_parser("serve", help="Run the DADS cloud gRPC server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=50051)
    serve_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    serve_parser.add_argument("--max-workers", type=int, default=4)

    client_parser = subparsers.add_parser("client-run", help="Run one edge/cloud split inference")
    client_parser.add_argument("--server", default="localhost:50051")
    client_parser.add_argument("--model", required=True, choices=list(AVAILABLE_MODELS))
    client_parser.add_argument("--bandwidth-mbps", type=float, required=True)
    client_parser.add_argument("--cpu-load-target", type=float, default=0.0)
    client_parser.add_argument("--cpu-load-tolerance", type=float, default=5.0)
    client_parser.add_argument("--cpu-load-interval", type=float, default=0.5)
    client_parser.add_argument("--cpu-load-ramp-seconds", type=float, default=2.0)
    client_parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 224, 224])
    client_parser.add_argument("--profile-runs", type=int, default=1)
    client_parser.add_argument("--profile-warmup-runs", type=int, default=0)
    client_parser.add_argument("--report-format", choices=["table", "json", "csv"], default="table")
    client_parser.add_argument("--report-output")
    client_parser.add_argument("--debug-output")
    client_parser.add_argument("--partition-granularity", choices=list(PARTITION_GRANULARITIES), default="node")

    config_parser = subparsers.add_parser("run-config", help="Run serve/client-run/experiment from a JSON config file")
    config_parser.add_argument("--config", required=True)

    experiment_parser = subparsers.add_parser("experiment", help="Run a DSL latency experiment sweep")
    experiment_parser.add_argument("--config", required=True)

    return parser


def _emit(payload: dict, output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


def _build_profile_cache(args: argparse.Namespace) -> ModelProfile:
    if args.role == "edge":
        config = ProfileRunConfig(
            input_shape=tuple(args.input_shape),
            edge_warmup_runs=args.warmup_runs,
            edge_profile_runs=args.profile_runs,
            cloud_warmup_runs=0,
            cloud_profile_runs=1,
            cloud_device="cpu",
            partition_granularity=args.partition_granularity,
        )
        from .cpu_load import CpuLoadController

        with CpuLoadController(args.cpu_load_target, args.cpu_load_tolerance, args.cpu_load_interval) as load_controller:
            load_controller.ramp(args.cpu_load_ramp_seconds)
            profile = profile_model(args.model, config)
            cpu_stats = load_controller.stats()
    else:
        edge_warmup_runs = args.warmup_runs if args.device in {"auto", "cpu"} else 0
        edge_profile_runs = args.profile_runs if args.device in {"auto", "cpu"} else 1
        config = ProfileRunConfig(
            input_shape=tuple(args.input_shape),
            edge_warmup_runs=edge_warmup_runs,
            edge_profile_runs=edge_profile_runs,
            cloud_warmup_runs=args.warmup_runs,
            cloud_profile_runs=args.profile_runs,
            cloud_device=args.device,
            partition_granularity=args.partition_granularity,
        )
        profile = profile_model(args.model, config)
        cpu_stats = None

    metadata = dict(profile.metadata or {})
    metadata.update(
        {
            "profile_role": args.role,
            "cpu_load_target": args.cpu_load_target if args.role == "edge" else None,
            "profile_cache_created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if cpu_stats is not None:
        metadata["cpu_load_stats"] = cpu_stats
    profile.metadata = metadata
    return profile


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        profile = profile_model(
            model_name=args.model,
            config=ProfileRunConfig(
                input_shape=tuple(args.input_shape),
                edge_warmup_runs=args.edge_warmup_runs,
                edge_profile_runs=args.edge_profile_runs,
                cloud_warmup_runs=args.cloud_warmup_runs,
                cloud_profile_runs=args.cloud_profile_runs,
                cloud_device=args.cloud_device,
                partition_granularity=args.partition_granularity,
            ),
        )
        profile.save(args.output)
        _emit(profile.to_dict(), None)
        return 0

    if args.command == "profile-cache":
        profile = _build_profile_cache(args)
        profile.save(args.output)
        _emit(profile.to_dict(), None)
        return 0

    if args.command == "serve":
        from .server import serve

        serve(args.host, args.port, args.device, args.max_workers)
        return 0

    if args.command == "client-run":
        from .client import run_client_once

        run_client_once(
            server=args.server,
            model_name=args.model,
            bandwidth_mbps=args.bandwidth_mbps,
            cpu_load_target=args.cpu_load_target,
            cpu_load_tolerance=args.cpu_load_tolerance,
            cpu_load_interval=args.cpu_load_interval,
            cpu_load_ramp_seconds=args.cpu_load_ramp_seconds,
            report_format=args.report_format,
            report_output=args.report_output,
            input_shape=list(args.input_shape),
            profile_runs=args.profile_runs,
            profile_warmup_runs=args.profile_warmup_runs,
            debug_output=args.debug_output,
            partition_granularity=args.partition_granularity,
        )
        return 0

    if args.command == "run-config":
        try:
            config = _load_config(args.config)
            config_command = str(config.get("command", ""))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))

        if config_command == "serve":
            device = str(config.get("device", "cuda"))
            if device not in {"cpu", "cuda"}:
                parser.error("serve config field 'device' must be 'cpu' or 'cuda'.")
            from .server import serve

            serve(
                host=str(config.get("host", "0.0.0.0")),
                port=int(config.get("port", 50051)),
                device=device,
                max_workers=int(config.get("max_workers", 4)),
            )
            return 0

        if config_command == "client-run":
            if "model" not in config:
                parser.error("client-run config field 'model' is required.")
            model_name = str(config["model"])
            if model_name not in AVAILABLE_MODELS:
                parser.error(f"client-run config field 'model' must be one of: {', '.join(AVAILABLE_MODELS)}.")
            if "bandwidth_mbps" not in config:
                parser.error("client-run config field 'bandwidth_mbps' is required.")
            report_format = str(config.get("report_format", "table"))
            if report_format not in {"table", "json", "csv"}:
                parser.error("client-run config field 'report_format' must be 'table', 'json', or 'csv'.")
            try:
                input_shape = _get_input_shape(config)
                partition_granularity = _get_partition_granularity(config)
            except ValueError as exc:
                parser.error(str(exc))

            from .client import run_client_once

            run_client_once(
                server=str(config.get("server", "localhost:50051")),
                model_name=model_name,
                bandwidth_mbps=float(config["bandwidth_mbps"]),
                cpu_load_target=float(config.get("cpu_load_target", 0.0)),
                cpu_load_tolerance=float(config.get("cpu_load_tolerance", 5.0)),
                cpu_load_interval=float(config.get("cpu_load_interval", 0.5)),
                cpu_load_ramp_seconds=float(config.get("cpu_load_ramp_seconds", 2.0)),
                report_format=report_format,
                report_output=config.get("report_output"),
                input_shape=input_shape,
                profile_runs=int(config.get("profile_runs", 1)),
                profile_warmup_runs=int(config.get("profile_warmup_runs", 0)),
                debug_output=config.get("debug_output"),
                partition_granularity=partition_granularity,
            )
            return 0

        if config_command == "experiment":
            from .experiment import run_experiment

            run_experiment(config)
            return 0

        parser.error("Config field 'command' must be 'serve', 'client-run', or 'experiment'.")

    if args.command == "experiment":
        try:
            config = _load_config(args.config)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
        if str(config.get("command", "experiment")) != "experiment":
            parser.error("experiment config field 'command' must be 'experiment'.")
        from .experiment import run_experiment

        run_experiment(config)
        return 0

    if args.command == "dsl":
        profile = ModelProfile.load(args.profile)
        solution = solve_dsl(profile, args.bandwidth_mbps)
        _emit(solution.to_dict(), args.output)
        return 0

    if args.command == "simulate":
        profile = ModelProfile.load(args.profile)
        bandwidths = _parse_bandwidths(args.bandwidths)
        solutions = solve_bandwidth_sweep(profile, bandwidths)
        payload = {
            "model_name": profile.model_name,
            "results": [solution.to_dict() for solution in solutions],
        }
        _emit(payload, args.output)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
