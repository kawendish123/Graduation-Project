from dads_dsl.dsl import VIRTUAL_INPUT_ID, with_virtual_input_node
from dads_dsl.types import ModelProfile, ModelProfileNode


def test_with_virtual_input_node_links_to_roots():
    profile = ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b"], 1.0, 2.0, 12),
            ModelProfileNode("b", "Conv2d", [], 1.0, 2.0, 12),
        ],
    )
    updated = with_virtual_input_node(profile, input_bytes=192)
    assert updated.nodes[0].id == VIRTUAL_INPUT_ID
    assert updated.nodes[0].succ_ids == ["a"]
    assert updated.nodes[0].output_bytes == 192


def test_with_virtual_input_node_is_idempotent():
    profile = ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 4, 4],
        nodes=[ModelProfileNode(VIRTUAL_INPUT_ID, "Input", ["a"], 0.0, 0.0, 192)],
    )
    assert with_virtual_input_node(profile, input_bytes=192) is profile
