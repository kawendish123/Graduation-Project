import importlib.util

from dads_dsl.plotting import (
    STRATEGY_ORDER,
    STRATEGY_STYLES,
    plot_estimate_speedup_heatmap,
    plot_estimate_stage_breakdown,
    plot_estimate_latency,
    plot_estimate_latency_by_bandwidth,
    plot_experiment_latency,
    plot_granularity_comparison,
)


def test_dsl_style_is_dashed_and_drawn_last():
    assert STRATEGY_ORDER[-1] == "dsl"
    assert STRATEGY_STYLES["dsl"]["linestyle"] == "--"


def test_plot_experiment_latency_writes_one_file_per_load(tmp_path):
    csv_path = tmp_path / "experiment.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,strategy,actual_total_mean_ms,actual_total_std_ms",
                "toy,10.0000,0.0000,dsl,10.0000,1.0000",
                "toy,10.0000,0.0000,pure_edge,20.0000,2.0000",
                "toy,10.0000,0.0000,pure_cloud,30.0000,3.0000",
                "toy,100.0000,0.0000,dsl,5.0000,0.5000",
                "toy,100.0000,0.0000,pure_edge,15.0000,1.5000",
                "toy,100.0000,0.0000,pure_cloud,8.0000,0.8000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = plot_experiment_latency(csv_path, tmp_path / "plots")

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_load0_latency.png").exists()


def test_plot_estimate_latency_writes_one_file_per_load(tmp_path):
    csv_path = tmp_path / "estimate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,strategy,estimated_total_ms",
                "toy,10.0000,0.0000,dsl,10.0000",
                "toy,10.0000,0.0000,pure_edge,20.0000",
                "toy,10.0000,0.0000,pure_cloud,30.0000",
                "toy,100.0000,0.0000,dsl,5.0000",
                "toy,100.0000,0.0000,pure_edge,15.0000",
                "toy,100.0000,0.0000,pure_cloud,8.0000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = plot_estimate_latency(csv_path, tmp_path / "plots")

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_load0_estimated_latency.png").exists()


def test_plot_estimate_latency_by_bandwidth_writes_one_file_per_bandwidth(tmp_path):
    csv_path = tmp_path / "estimate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,strategy,estimated_total_ms",
                "toy,10.0000,0.0000,dsl,10.0000",
                "toy,10.0000,0.0000,pure_edge,20.0000",
                "toy,10.0000,0.0000,pure_cloud,30.0000",
                "toy,10.0000,30.0000,dsl,15.0000",
                "toy,10.0000,30.0000,pure_edge,25.0000",
                "toy,10.0000,30.0000,pure_cloud,30.0000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = plot_estimate_latency_by_bandwidth(csv_path, tmp_path / "plots")

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_bw10_load_sweep_estimated_latency.png").exists()


def test_plot_estimate_speedup_heatmap_writes_one_file(tmp_path):
    csv_path = tmp_path / "estimate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,strategy,estimated_total_ms",
                "toy,10.0000,0.0000,dsl,8.0000",
                "toy,10.0000,0.0000,pure_edge,20.0000",
                "toy,10.0000,0.0000,pure_cloud,10.0000",
                "toy,20.0000,0.0000,dsl,9.0000",
                "toy,20.0000,0.0000,pure_edge,20.0000",
                "toy,20.0000,0.0000,pure_cloud,12.0000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = plot_estimate_speedup_heatmap(csv_path, tmp_path / "plots")

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_dsl_speedup_heatmap.png").exists()


def test_plot_estimate_stage_breakdown_writes_one_file_per_load(tmp_path):
    csv_path = tmp_path / "estimate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,strategy,estimated_edge_ms,estimated_transfer_ms,estimated_cloud_ms",
                "toy,10.0000,0.0000,dsl,1.0000,2.0000,3.0000",
                "toy,20.0000,0.0000,dsl,1.5000,1.0000,2.5000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = plot_estimate_stage_breakdown(csv_path, tmp_path / "plots")

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_load0_dsl_stage_breakdown.png").exists()


def test_plot_granularity_comparison_writes_one_file_per_load(tmp_path):
    block_csv = tmp_path / "block.csv"
    filtered_csv = tmp_path / "node_filtered.csv"
    content = "\n".join(
        [
            "model,bandwidth_mbps,cpu_load_target,strategy,estimated_total_ms",
            "toy,10.0000,0.0000,dsl,10.0000",
            "toy,20.0000,0.0000,dsl,8.0000",
        ]
    )
    block_csv.write_text(content, encoding="utf-8")
    filtered_csv.write_text(content.replace("10.0000", "9.0000", 1), encoding="utf-8")

    outputs = plot_granularity_comparison(
        {"block": block_csv, "node_filtered": filtered_csv},
        tmp_path / "plots",
    )

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert len(outputs) == 1
        assert (tmp_path / "plots" / "toy_load0_dsl_granularity_comparison.png").exists()


def test_plot_estimate_functions_support_scaled_edge_conditions(tmp_path):
    csv_path = tmp_path / "estimate_scaled.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,bandwidth_mbps,cpu_load_target,edge_latency_mode,edge_condition,edge_slowdown_factor,edge_condition_order,strategy,estimated_edge_ms,estimated_transfer_ms,estimated_cloud_ms,estimated_total_ms",
                "toy,10.0000,,scaled,Edge-High,1.0000,0,dsl,1.0000,2.0000,3.0000,6.0000",
                "toy,10.0000,,scaled,Edge-High,1.0000,0,pure_edge,10.0000,0.0000,0.0000,10.0000",
                "toy,10.0000,,scaled,Edge-High,1.0000,0,pure_cloud,0.0000,7.0000,1.0000,8.0000",
                "toy,10.0000,,scaled,Edge-Medium,4.0000,1,dsl,4.0000,2.0000,3.0000,9.0000",
                "toy,10.0000,,scaled,Edge-Medium,4.0000,1,pure_edge,40.0000,0.0000,0.0000,40.0000",
                "toy,10.0000,,scaled,Edge-Medium,4.0000,1,pure_cloud,0.0000,7.0000,1.0000,8.0000",
            ]
        ),
        encoding="utf-8",
    )

    outputs = []
    outputs.extend(plot_estimate_latency(csv_path, tmp_path / "plots"))
    outputs.extend(plot_estimate_latency_by_bandwidth(csv_path, tmp_path / "plots"))
    outputs.extend(plot_estimate_speedup_heatmap(csv_path, tmp_path / "plots"))
    outputs.extend(plot_estimate_stage_breakdown(csv_path, tmp_path / "plots"))

    if importlib.util.find_spec("matplotlib") is None:
        assert outputs == []
    else:
        assert (tmp_path / "plots" / "toy_Edge-High_estimated_latency.png").exists()
        assert (tmp_path / "plots" / "toy_Edge-Medium_estimated_latency.png").exists()
        assert (tmp_path / "plots" / "toy_bw10_condition_sweep_estimated_latency.png").exists()
        assert (tmp_path / "plots" / "toy_dsl_speedup_heatmap.png").exists()
        assert (tmp_path / "plots" / "toy_Edge-High_dsl_stage_breakdown.png").exists()
