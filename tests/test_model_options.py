import pytest

from dads_dsl.cli import _get_partition_granularity, build_parser
from dads_dsl.client import _merge_profiles
from dads_dsl.types import ModelProfile, ModelProfileNode


def _profile(granularity: str) -> ModelProfile:
    return ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 4, 4],
        nodes=[ModelProfileNode("a", "Block", [], 1.0, 2.0, 16)],
        metadata={"partition_granularity": granularity},
    )


def test_cli_accepts_new_models_and_granularities():
    parser = build_parser()
    for model in ["mobilenet_v2", "googlenet", "resnet50", "vgg16"]:
        for granularity in ["node", "node_filtered", "block"]:
            args = parser.parse_args(
                [
                    "profile",
                    "--model",
                    model,
                    "--output",
                    "profile.json",
                    "--partition-granularity",
                    granularity,
                ]
            )
            assert args.model == model
            assert args.partition_granularity == granularity


def test_cli_accepts_profile_cache_command():
    args = build_parser().parse_args(
        [
            "profile-cache",
            "--role",
            "edge",
            "--model",
            "resnet50",
            "--partition-granularity",
            "block",
            "--cpu-load-target",
            "30",
            "--output",
            "profiles/resnet50_block/edge_load30.json",
        ]
    )

    assert args.command == "profile-cache"
    assert args.role == "edge"
    assert args.model == "resnet50"
    assert args.cpu_load_target == 30.0


def test_partition_granularity_config_validation():
    assert _get_partition_granularity({}) == "node"
    assert _get_partition_granularity({"partition_granularity": "node_filtered"}) == "node_filtered"
    assert _get_partition_granularity({"partition_granularity": "block"}) == "block"
    with pytest.raises(ValueError):
        _get_partition_granularity({"partition_granularity": "bad"})


def test_merge_profiles_rejects_granularity_mismatch():
    with pytest.raises(ValueError, match="granularity mismatch"):
        _merge_profiles(_profile("node_filtered"), _profile("block"))


def test_merge_profiles_rejects_folded_output_alias_mismatch():
    edge = _profile("node_filtered")
    cloud = _profile("node_filtered")
    edge.metadata["folded_output_nodes"] = {"a": "relu"}
    cloud.metadata["folded_output_nodes"] = {"a": "bn"}

    with pytest.raises(ValueError, match="folded output alias mismatch"):
        _merge_profiles(edge, cloud)


def test_merge_profiles_rejects_model_and_input_shape_mismatch():
    edge = _profile("block")
    cloud = _profile("block")
    cloud.model_name = "other"
    with pytest.raises(ValueError, match="model mismatch"):
        _merge_profiles(edge, cloud)

    cloud = _profile("block")
    cloud.input_shape = [1, 3, 8, 8]
    with pytest.raises(ValueError, match="input_shape mismatch"):
        _merge_profiles(edge, cloud)
