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


def test_cli_accepts_new_models_and_block_granularity():
    parser = build_parser()
    for model in ["mobilenet_v2", "googlenet", "resnet50", "vgg16"]:
        args = parser.parse_args(
            [
                "profile",
                "--model",
                model,
                "--output",
                "profile.json",
                "--partition-granularity",
                "block",
            ]
        )
        assert args.model == model
        assert args.partition_granularity == "block"


def test_partition_granularity_config_validation():
    assert _get_partition_granularity({}) == "node"
    assert _get_partition_granularity({"partition_granularity": "block"}) == "block"
    with pytest.raises(ValueError):
        _get_partition_granularity({"partition_granularity": "bad"})


def test_merge_profiles_rejects_granularity_mismatch():
    with pytest.raises(ValueError, match="granularity mismatch"):
        _merge_profiles(_profile("node"), _profile("block"))
