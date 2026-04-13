import json

import pytest

from dads_dsl.dsl import solve_dsl
from dads_dsl.profile import ProfileRunConfig, profile_model
from dads_dsl.types import ModelProfile


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.mark.slow
def test_mobilenet_profile_exports_and_solves(tmp_path):
    profile = profile_model(
        "mobilenet_v2",
        ProfileRunConfig(
            input_shape=(1, 3, 224, 224),
            edge_warmup_runs=0,
            edge_profile_runs=1,
            cloud_warmup_runs=0,
            cloud_profile_runs=1,
            cloud_device="cpu",
        ),
    )
    assert profile.nodes

    path = tmp_path / "mobilenet.json"
    profile.save(path)
    reloaded = ModelProfile.load(path)
    assert reloaded.model_name == "mobilenet_v2"

    low = solve_dsl(reloaded, bandwidth_mbps=0.5)
    high = solve_dsl(reloaded, bandwidth_mbps=20.0)
    assert low.edge_nodes
    assert high.cloud_nodes or high.edge_nodes
    if low.edge_nodes != high.edge_nodes:
        assert len(high.cloud_nodes) >= len(low.cloud_nodes)


@pytest.mark.slow
def test_googlenet_profile_has_branch_and_solves(tmp_path):
    profile = profile_model(
        "googlenet",
        ProfileRunConfig(
            input_shape=(1, 3, 224, 224),
            edge_warmup_runs=0,
            edge_profile_runs=1,
            cloud_warmup_runs=0,
            cloud_profile_runs=1,
            cloud_device="cpu",
        ),
    )
    assert any(len(node.succ_ids) > 1 for node in profile.nodes)

    low = solve_dsl(profile, bandwidth_mbps=0.5)
    payload = low.to_dict()
    assert payload["transmission_stage_ms"] >= 0
    serialized = json.dumps(payload)
    assert "transmission_nodes" in serialized
