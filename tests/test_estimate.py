import csv
import json

import pytest

from dads_dsl.cli import _load_config, build_parser
from dads_dsl.estimate import input_bytes_from_shape, run_estimate_experiment
from dads_dsl.types import ModelProfile, ModelProfileNode


def _profile(edge_scale: float = 1.0, cloud_scale: float = 1.0) -> ModelProfile:
    return ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Block", ["b"], 10.0 * edge_scale, 1.0 * cloud_scale, 100),
            ModelProfileNode("b", "Block", [], 10.0 * edge_scale, 1.0 * cloud_scale, 50),
        ],
        metadata={"partition_granularity": "block"},
    )


def test_input_bytes_from_shape_uses_float32_size():
    assert input_bytes_from_shape([1, 3, 224, 224]) == 602112


def test_cli_accepts_estimate_experiment_command_and_config(tmp_path):
    args = build_parser().parse_args(["estimate-experiment", "--config", "configs/estimate_resnet50_block.json"])
    assert args.command == "estimate-experiment"
    assert args.config == "configs/estimate_resnet50_block.json"

    config_path = tmp_path / "estimate.json"
    config_path.write_text(json.dumps({"command": "estimate-experiment", "bandwidths_mbps": [20]}), encoding="utf-8")
    config = _load_config(str(config_path))
    assert config["command"] == "estimate-experiment"
    assert config["bandwidths_mbps"] == [20]


def test_run_estimate_experiment_writes_summaries_and_debug(tmp_path):
    edge_profile_path = tmp_path / "edge.json"
    cloud_profile_path = tmp_path / "cloud.json"
    _profile(edge_scale=1.0, cloud_scale=1.0).save(edge_profile_path)
    _profile(edge_scale=1.0, cloud_scale=1.0).save(cloud_profile_path)

    config = {
        "command": "estimate-experiment",
        "model": "toy",
        "partition_granularity": "block",
        "bandwidths_mbps": [1000],
        "cpu_load_targets": [0],
        "input_shape": [1, 3, 4, 4],
        "edge_profiles": {"0": str(edge_profile_path)},
        "cloud_profile": str(cloud_profile_path),
        "report_csv": str(tmp_path / "estimate.csv"),
        "report_json": str(tmp_path / "estimate.json"),
        "debug_dir": str(tmp_path / "debug"),
        "plot_dir": None,
    }

    payload = run_estimate_experiment(config)

    pure_cloud = payload["results"][0]["strategy_summaries"]["pure_cloud"]
    assert pure_cloud["estimated_transfer_ms"] == pytest.approx(0.001536)
    assert pure_cloud["estimated_total_ms"] == pytest.approx(2.001536)
    assert pure_cloud["payload_bytes"] == 192

    with (tmp_path / "estimate.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strategy"] for row in rows] == ["dsl", "pure_edge", "pure_cloud"]
    assert rows[0]["estimated_total_ms"]
    assert (tmp_path / "debug" / "toy_load0_bw1000.json").exists()


def test_run_estimate_experiment_reports_missing_edge_profile(tmp_path):
    cloud_profile_path = tmp_path / "cloud.json"
    _profile().save(cloud_profile_path)

    config = {
        "command": "estimate-experiment",
        "model": "toy",
        "partition_granularity": "block",
        "bandwidths_mbps": [1000],
        "cpu_load_targets": [30],
        "input_shape": [1, 3, 4, 4],
        "edge_profiles": {"0": str(tmp_path / "edge.json")},
        "cloud_profile": str(cloud_profile_path),
        "report_csv": str(tmp_path / "estimate.csv"),
        "report_json": str(tmp_path / "estimate.json"),
        "plot_dir": None,
    }

    with pytest.raises(ValueError, match="Missing edge profile for cpu_load_target=30"):
        run_estimate_experiment(config)
