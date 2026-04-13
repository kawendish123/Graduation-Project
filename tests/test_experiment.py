import json

from dads_dsl.cli import _load_config, build_parser
from dads_dsl.dsl import VIRTUAL_INPUT_ID, solve_dsl, with_virtual_input_node
from dads_dsl.experiment import _run_strategy_repeats, build_strategy_plans, summarize_measurements
from dads_dsl.types import ModelProfile, ModelProfileNode


def _chain_profile() -> ModelProfile:
    return ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b"], 10.0, 1.0, 100),
            ModelProfileNode("b", "Conv2d", [], 10.0, 1.0, 50),
        ],
    )


def test_summarize_measurements_reports_mean_and_std():
    summary = summarize_measurements(
        [
            {"total_actual_ms": 10.0, "edge_actual_ms": 4.0, "transmission_actual_ms": 3.0, "cloud_actual_ms": 3.0, "payload_bytes": 100},
            {"total_actual_ms": 14.0, "edge_actual_ms": 6.0, "transmission_actual_ms": 4.0, "cloud_actual_ms": 4.0, "payload_bytes": 300},
        ]
    )

    assert summary["actual_total_mean_ms"] == 12.0
    assert round(summary["actual_total_std_ms"], 6) == 2.828427
    assert summary["edge_actual_mean_ms"] == 5.0
    assert round(summary["edge_actual_std_ms"], 6) == 1.414214
    assert round(summary["transmission_actual_std_ms"], 6) == 0.707107
    assert round(summary["cloud_actual_std_ms"], 6) == 0.707107
    assert summary["payload_bytes_mean"] == 200.0
    assert round(summary["payload_bytes_std"], 6) == 141.421356


def test_build_strategy_plans_constructs_dsl_and_baselines():
    profile = with_virtual_input_node(_chain_profile(), input_bytes=192)
    solution = solve_dsl(profile, bandwidth_mbps=1000.0)
    plans = build_strategy_plans(profile, bandwidth_mbps=1000.0, dsl_solution=solution)

    assert plans["dsl"].transmission_nodes == solution.transmission_nodes
    assert plans["pure_edge"].edge_nodes == [VIRTUAL_INPUT_ID, "a", "b"]
    assert plans["pure_edge"].transmission_nodes == []
    assert plans["pure_edge"].cloud_nodes == []
    assert plans["pure_edge"].estimated_total_ms == 20.0

    assert plans["pure_cloud"].edge_nodes == [VIRTUAL_INPUT_ID]
    assert plans["pure_cloud"].transmission_nodes == [VIRTUAL_INPUT_ID]
    assert plans["pure_cloud"].cloud_nodes == ["a", "b"]
    assert plans["pure_cloud"].estimated_transfer_ms == 0.001536
    assert plans["pure_cloud"].estimated_total_ms == 2.001536


def test_cli_accepts_experiment_command_and_config(tmp_path):
    args = build_parser().parse_args(["experiment", "--config", "configs/experiment_mobilenet_v2.json"])
    assert args.command == "experiment"
    assert args.config == "configs/experiment_mobilenet_v2.json"

    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps({"command": "experiment", "bandwidths_mbps": [20], "cpu_load_targets": [0], "repeats": 1, "actual_warmup_runs": 2}),
        encoding="utf-8",
    )
    config = _load_config(str(config_path))
    assert config["command"] == "experiment"
    assert config["bandwidths_mbps"] == [20]
    assert config["actual_warmup_runs"] == 2


def test_run_strategy_repeats_excludes_actual_warmup_measurements():
    calls = []

    def execute_once():
        calls.append(len(calls))
        return {"total_actual_ms": float(len(calls)), "edge_actual_ms": 1.0, "transmission_actual_ms": 2.0, "cloud_actual_ms": 3.0, "payload_bytes": 4}

    measurements = _run_strategy_repeats(execute_once, repeats=3, actual_warmup_runs=2)

    assert len(calls) == 5
    assert [item["repeat_index"] for item in measurements] == [1, 2, 3]
    assert [item["total_actual_ms"] for item in measurements] == [3.0, 4.0, 5.0]
