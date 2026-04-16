from dads_dsl.profile import (
    _folded_nodes_by_predecessor,
    _initial_keep_names,
    _op_type,
    _promote_ambiguous_filtered_nodes,
    folded_output_aliases,
)


class FakeTensor:
    def numel(self):
        return 1

    def element_size(self):
        return 4


class Conv2d:
    pass


class BatchNorm2d:
    pass


class ReLU:
    pass


class LeakyReLU:
    pass


class FakeNode:
    def __init__(self, name, op, target, inputs=None):
        self.name = name
        self.op = op
        self.target = target
        self.all_input_nodes = inputs or []


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class FakeTraced:
    def __init__(self, nodes, modules):
        self.graph = FakeGraph(nodes)
        self._modules = modules

    def get_submodule(self, target):
        return self._modules[target]


def cat(items, dim=1):
    return items


def test_node_filtered_folds_bn_and_relu_into_upstream_conv():
    conv1 = FakeNode("conv1", "call_module", "conv1")
    bn = FakeNode("bn", "call_module", "bn", [conv1])
    relu = FakeNode("relu", "call_module", "relu", [bn])
    conv2 = FakeNode("conv2", "call_module", "conv2", [relu])
    traced = FakeTraced(
        [conv1, bn, relu, conv2],
        {"conv1": Conv2d(), "bn": BatchNorm2d(), "relu": ReLU(), "conv2": Conv2d()},
    )
    outputs = {node.name: FakeTensor() for node in traced.graph.nodes}
    keep_names = _initial_keep_names(traced, outputs, "node_filtered")
    keep_names = _promote_ambiguous_filtered_nodes(traced, keep_names, outputs)

    kept_types = {_op_type(node, traced) for node in traced.graph.nodes if node.name in keep_names}
    assert kept_types == {"Conv2d"}
    assert "bn" not in keep_names
    assert "relu" not in keep_names

    folded = _folded_nodes_by_predecessor(traced, keep_names, outputs)
    assert folded["conv1"] == ["bn", "relu"]
    assert folded_output_aliases(traced, keep_names, outputs)["conv1"] == "relu"


def test_node_filtered_keeps_cat_as_dag_merge_node():
    conv1 = FakeNode("conv1", "call_module", "conv1")
    conv2 = FakeNode("conv2", "call_module", "conv2", [conv1])
    conv3 = FakeNode("conv3", "call_module", "conv3", [conv1])
    cat_node = FakeNode("cat", "call_function", cat, [conv2, conv3])
    traced = FakeTraced(
        [conv1, conv2, conv3, cat_node],
        {"conv1": Conv2d(), "conv2": Conv2d(), "conv3": Conv2d()},
    )
    outputs = {node.name: FakeTensor() for node in traced.graph.nodes}
    keep_names = _initial_keep_names(traced, outputs, "node_filtered")
    keep_names = _promote_ambiguous_filtered_nodes(traced, keep_names, outputs)

    kept_types = {_op_type(node, traced) for node in traced.graph.nodes if node.name in keep_names}
    assert "Conv2d" in kept_types
    assert "cat" in kept_types
    assert any(node.name in keep_names and _op_type(node, traced) == "cat" for node in traced.graph.nodes)


def test_node_filtered_folds_leaky_relu_into_upstream_conv():
    conv = FakeNode("conv", "call_module", "conv")
    leaky_relu = FakeNode("leaky_relu", "call_module", "leaky_relu", [conv])
    head = FakeNode("head", "call_module", "head", [leaky_relu])
    traced = FakeTraced(
        [conv, leaky_relu, head],
        {"conv": Conv2d(), "leaky_relu": LeakyReLU(), "head": Conv2d()},
    )
    outputs = {node.name: FakeTensor() for node in traced.graph.nodes}
    keep_names = _initial_keep_names(traced, outputs, "node_filtered")
    keep_names = _promote_ambiguous_filtered_nodes(traced, keep_names, outputs)

    assert "conv" in keep_names
    assert "head" in keep_names
    assert "leaky_relu" not in keep_names
    assert _folded_nodes_by_predecessor(traced, keep_names, outputs)["conv"] == ["leaky_relu"]
