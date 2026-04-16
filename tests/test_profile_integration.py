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


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_name", "expected_ops"),
    [
        ("alexnet", {"Conv2d", "MaxPool2d", "Linear"}),
        ("tiny_yolo", {"Conv2d", "MaxPool2d"}),
        ("shufflenet_v2", {"Conv2d", "MaxPool2d", "Linear"}),
        ("efficientnet_b0", {"Conv2d", "AdaptiveAvgPool2d", "Linear"}),
        ("mobilenet_v3_large", {"Conv2d", "AdaptiveAvgPool2d", "Linear"}),
        ("mobilenet_v3_small", {"Conv2d", "AdaptiveAvgPool2d", "Linear"}),
    ],
)
def test_new_models_node_filtered_profile_exports(model_name, expected_ops):
    profile = profile_model(
        model_name,
        ProfileRunConfig(
            input_shape=(1, 3, 64, 64),
            edge_warmup_runs=0,
            edge_profile_runs=1,
            cloud_warmup_runs=0,
            cloud_profile_runs=1,
            cloud_device="cpu",
            partition_granularity="node_filtered",
        ),
    )

    op_types = {node.op_type for node in profile.nodes}
    assert expected_ops.issubset(op_types)
    assert "BatchNorm2d" not in op_types
    assert "LeakyReLU" not in op_types
    assert "ReLU" not in op_types
    assert "StochasticDepth" not in op_types
    assert "stochastic_depth" not in op_types


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_name", "expected_ids"),
    [
        ("alexnet", {"features_block1", "features_block2", "features_block3", "head"}),
        ("tiny_yolo", {"stage1", "stage2", "stage3", "stage4", "stage5", "neck", "head"}),
        ("mobilenet_v3_large", {"features_0", "features_1", "features_16", "classifier"}),
        ("mobilenet_v3_small", {"features_0", "features_1", "features_12", "classifier"}),
        ("shufflenet_v2", {"stem", "stage2", "stage3", "stage4", "conv5", "head"}),
        (
            "efficientnet_b0",
            {
                "features_0",
                "features_1",
                "features_2",
                "features_3",
                "features_4",
                "features_5",
                "features_6",
                "features_7",
                "features_8",
                "head",
            },
        ),
    ],
)
def test_new_models_block_profile_exports_expected_blocks(model_name, expected_ids):
    profile = profile_model(
        model_name,
        ProfileRunConfig(
            input_shape=(1, 3, 64, 64),
            edge_warmup_runs=0,
            edge_profile_runs=1,
            cloud_warmup_runs=0,
            cloud_profile_runs=1,
            cloud_device="cpu",
            partition_granularity="block",
        ),
    )

    assert expected_ids.issubset({node.id for node in profile.nodes})
