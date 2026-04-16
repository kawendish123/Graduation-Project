import importlib.util

from dads_dsl.plotting import (
    STRATEGY_ORDER,
    STRATEGY_STYLES,
    plot_estimate_latency,
    plot_estimate_latency_by_bandwidth,
    plot_experiment_latency,
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
