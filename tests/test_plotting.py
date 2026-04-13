import importlib.util

from dads_dsl.plotting import plot_experiment_latency


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
