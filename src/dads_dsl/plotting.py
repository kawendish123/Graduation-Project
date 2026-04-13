from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Optional


STRATEGY_ORDER = ["dsl", "pure_edge", "pure_cloud"]
STRATEGY_LABELS = {
    "dsl": "DSL",
    "pure_edge": "Pure Edge",
    "pure_cloud": "Pure Cloud",
}


def _load_pyplot() -> Optional[Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("matplotlib is not installed; skipping experiment plot generation.")
        return None


def _safe_token(value: str) -> str:
    token = value.strip()
    try:
        token = f"{float(token):g}"
    except ValueError:
        pass
    token = token.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", token)


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_experiment_latency(csv_path: str | Path, output_dir: str | Path, plot_format: str = "png") -> list[str]:
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_rows(csv_path)
    if not rows:
        return []

    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["cpu_load_target"]][row["strategy"]].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")
    outputs = []

    for load_target, by_strategy in sorted(grouped.items(), key=lambda item: float(item[0])):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in STRATEGY_ORDER:
            strategy_rows = sorted(by_strategy.get(strategy, []), key=lambda item: float(item["bandwidth_mbps"]))
            if not strategy_rows:
                continue
            bandwidths = [float(item["bandwidth_mbps"]) for item in strategy_rows]
            means = [float(item["actual_total_mean_ms"]) for item in strategy_rows]
            stds = [float(item["actual_total_std_ms"]) for item in strategy_rows]
            ax.errorbar(
                bandwidths,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=STRATEGY_LABELS.get(strategy, strategy),
            )

        ax.set_xscale("log", base=10)
        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Actual Total Latency (ms)")
        ax.set_title(f"{model_name} latency under CPU load {float(load_target):g}%")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()

        filename = f"{_safe_token(model_name)}_load{_safe_token(load_target)}_latency.{plot_format}"
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs
