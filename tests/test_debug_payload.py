import json

from dads_dsl.cli import _load_config, build_parser
from dads_dsl.dsl import build_debug_payload, solve_dsl, with_virtual_input_node
from dads_dsl.types import ModelProfile, ModelProfileNode


def _branch_profile() -> ModelProfile:
    return ModelProfile(
        model_name="branch",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b", "c"], 1.0, 100.0, 1_000),
            ModelProfileNode("b", "Conv2d", ["d"], 100.0, 1.0, 500),
            ModelProfileNode("c", "Conv2d", ["d"], 100.0, 1.0, 500),
            ModelProfileNode("d", "Conv2d", [], 100.0, 1.0, 250),
        ],
    )


def test_build_debug_payload_contains_node_latencies_and_partition():
    profile = with_virtual_input_node(_branch_profile(), input_bytes=192)
    solution = solve_dsl(profile, bandwidth_mbps=1.0)

    payload = build_debug_payload(profile, bandwidth_mbps=1.0, solution=solution)
    first_node = payload["nodes"][0]

    assert payload["model_name"] == "branch"
    assert payload["summary"]["dsl_estimated_total_ms"] == solution.total_inference_ms
    assert {"id", "edge_ms", "cloud_ms", "transmission_ms", "output_bytes", "partition"}.issubset(first_node)
    assert first_node["id"] == "__input__"
    assert first_node["transmission_ms"] == 1.536


def test_build_debug_payload_records_aux_nodes_and_flow_edges():
    profile = with_virtual_input_node(_branch_profile(), input_bytes=192)
    solution = solve_dsl(profile, bandwidth_mbps=1.0)
    payload = build_debug_payload(profile, bandwidth_mbps=1.0, solution=solution)

    assert payload["aux_nodes"]["a"] == "a__aux"
    assert any(edge["src"] == "a" and edge["dst"] == "a__aux" for edge in payload["flow_edges"])
    assert any(edge["src"] == "b" and edge["dst"] == "a" and edge["role"] == "precedence_constraint" for edge in payload["flow_edges"])
    assert all("capacity_ms" in edge and "role" in edge and "is_cut_edge" in edge for edge in payload["flow_edges"])


def test_cli_debug_output_argument_and_config_field(tmp_path):
    args = build_parser().parse_args(
        [
            "client-run",
            "--model",
            "mobilenet_v2",
            "--bandwidth-mbps",
            "5",
            "--debug-output",
            "results/debug.json",
        ]
    )
    assert args.debug_output == "results/debug.json"

    config_path = tmp_path / "client.json"
    config_path.write_text(json.dumps({"command": "client-run", "model": "mobilenet_v2", "bandwidth_mbps": 5, "debug_output": "debug.json"}), encoding="utf-8")
    assert _load_config(str(config_path))["debug_output"] == "debug.json"
