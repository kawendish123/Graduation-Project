from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .dsl import solve_bandwidth_sweep, solve_dsl
from .profile import ProfileRunConfig, profile_model
from .types import ModelProfile


def _parse_bandwidths(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DADS DSL prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Export a model profile JSON")
    profile_parser.add_argument("--model", required=True, choices=["mobilenet_v2", "googlenet"])
    profile_parser.add_argument("--output", required=True)
    profile_parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 224, 224])
    profile_parser.add_argument("--edge-warmup-runs", type=int, default=1)
    profile_parser.add_argument("--edge-profile-runs", type=int, default=3)
    profile_parser.add_argument("--cloud-warmup-runs", type=int, default=1)
    profile_parser.add_argument("--cloud-profile-runs", type=int, default=3)
    profile_parser.add_argument("--cloud-device", choices=["auto", "cpu", "cuda"], default="auto")

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
    client_parser.add_argument("--model", required=True, choices=["mobilenet_v2", "googlenet"])
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

    return parser


def _emit(payload: dict, output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


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
            ),
        )
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
        )
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
