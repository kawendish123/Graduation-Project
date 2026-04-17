from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional


STRATEGY_ORDER = ["pure_edge", "pure_cloud", "dsl"]
STRATEGY_LABELS = {
    "dsl": "DSL",
    "pure_edge": "Pure Edge",
    "pure_cloud": "Pure Cloud",
}
STRATEGY_STYLES = {
    "dsl": {"linestyle": "--", "marker": "o", "linewidth": 2.4, "zorder": 5},
    "pure_edge": {"linestyle": "-", "marker": "s", "linewidth": 1.6, "alpha": 0.75, "zorder": 2},
    "pure_cloud": {"linestyle": "-", "marker": "^", "linewidth": 1.6, "alpha": 0.75, "zorder": 2},
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


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"CSV row is missing required field '{key}'.")
    return float(value)


def _uses_edge_condition(rows: list[dict[str, str]]) -> bool:
    return any(row.get("edge_condition") for row in rows)


def _condition_key(row: dict[str, str]) -> str:
    return row.get("edge_condition") or row.get("cpu_load_target", "")


def _condition_order(row: dict[str, str]) -> float:
    if row.get("edge_condition") and row.get("edge_condition_order") not in {None, ""}:
        return float(row["edge_condition_order"])
    return float(row.get("cpu_load_target", "0"))


def _condition_keys(rows: list[dict[str, str]]) -> list[str]:
    orders: dict[str, float] = {}
    for row in rows:
        key = _condition_key(row)
        orders[key] = min(orders.get(key, _condition_order(row)), _condition_order(row))
    return sorted(orders, key=lambda key: orders[key])


def _condition_title(label: str, uses_edge_condition: bool) -> str:
    if uses_edge_condition:
        return label
    return f"CPU load {float(label):g}%"


def _condition_axis_label(uses_edge_condition: bool) -> str:
    return "Edge Condition" if uses_edge_condition else "CPU Load Target (%)"


def _condition_filename_part(label: str, uses_edge_condition: bool) -> str:
    if uses_edge_condition:
        return _safe_token(label)
    return f"load{_safe_token(label)}"


def plot_experiment_latency(csv_path: str | Path, output_dir: str | Path, plot_format: str = "png") -> list[str]:
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_rows(csv_path)
    if not rows:
        return []

    uses_edge_condition = _uses_edge_condition(rows)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[_condition_key(row)][row["strategy"]].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")
    outputs = []

    condition_order = {label: index for index, label in enumerate(_condition_keys(rows))}
    for condition_label, by_strategy in sorted(grouped.items(), key=lambda item: condition_order[item[0]]):
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
                capsize=3,
                label=STRATEGY_LABELS.get(strategy, strategy),
                **STRATEGY_STYLES.get(strategy, {}),
            )

        ax.set_xscale("log", base=10)
        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Actual Total Latency (ms)")
        ax.set_title(f"{model_name} latency under {_condition_title(condition_label, uses_edge_condition)}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()

        filename = f"{_safe_token(model_name)}_{_condition_filename_part(condition_label, uses_edge_condition)}_latency.{plot_format}"
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs


def plot_estimate_speedup_heatmap(csv_path: str | Path, output_dir: str | Path, plot_format: str = "png") -> list[str]:
    """Plot DSL improvement over the best pure baseline for each load/bandwidth cell."""
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_rows(csv_path)
    if not rows:
        return []

    uses_edge_condition = _uses_edge_condition(rows)
    by_condition: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_condition[(_condition_key(row), row["bandwidth_mbps"])][row["strategy"]] = row

    condition_labels = _condition_keys(rows)
    bandwidths = sorted({key[1] for key in by_condition}, key=float)
    if not condition_labels or not bandwidths:
        return []

    matrix: list[list[float]] = []
    finite_values: list[float] = []
    for condition_label in condition_labels:
        line: list[float] = []
        for bandwidth in bandwidths:
            strategies = by_condition.get((condition_label, bandwidth), {})
            if not {"dsl", "pure_edge", "pure_cloud"}.issubset(strategies):
                value = math.nan
            else:
                dsl_total = _float_value(strategies["dsl"], "estimated_total_ms")
                edge_total = _float_value(strategies["pure_edge"], "estimated_total_ms")
                cloud_total = _float_value(strategies["pure_cloud"], "estimated_total_ms")
                best_baseline = min(edge_total, cloud_total)
                value = 0.0 if best_baseline <= 0 else (best_baseline - dsl_total) / best_baseline * 100.0
            line.append(value)
            if not math.isnan(value):
                finite_values.append(value)
        matrix.append(line)

    if not finite_values:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")

    fig, ax = plt.subplots(figsize=(max(8, len(bandwidths) * 0.65), max(4.5, len(condition_labels) * 0.45)))
    try:
        from matplotlib.colors import TwoSlopeNorm

        low = min(min(finite_values), 0.0)
        high = max(max(finite_values), 0.0)
        if low == high:
            high = low + 1.0
        if low < 0 < high:
            norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
            image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm)
        else:
            image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=low, vmax=high)
    except Exception:
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(bandwidths)))
    ax.set_xticklabels([f"{float(item):g}" for item in bandwidths], rotation=45, ha="right")
    ax.set_yticks(range(len(condition_labels)))
    ax.set_yticklabels([str(item) if uses_edge_condition else f"{float(item):g}" for item in condition_labels])
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel(_condition_axis_label(uses_edge_condition))
    ax.set_title(f"{model_name} DSL speedup over best pure baseline")

    for row_index, line in enumerate(matrix):
        for col_index, value in enumerate(line):
            if not math.isnan(value):
                ax.text(col_index, row_index, f"{value:.1f}%", ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Improvement (%)")
    fig.tight_layout()

    destination = output_path / f"{_safe_token(model_name)}_dsl_speedup_heatmap.{plot_format}"
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return [str(destination)]


def plot_estimate_stage_breakdown(
    csv_path: str | Path,
    output_dir: str | Path,
    plot_format: str = "png",
    strategy: str = "dsl",
) -> list[str]:
    """Plot edge/transfer/cloud estimated latency components for a strategy."""
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = [row for row in _read_rows(csv_path) if row.get("strategy") == strategy]
    if not rows:
        return []

    uses_edge_condition = _uses_edge_condition(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_condition_key(row)].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")
    outputs = []

    condition_order = {label: index for index, label in enumerate(_condition_keys(rows))}
    for condition_label, load_rows in sorted(grouped.items(), key=lambda item: condition_order[item[0]]):
        sorted_rows = sorted(load_rows, key=lambda item: float(item["bandwidth_mbps"]))
        labels = [f"{_float_value(row, 'bandwidth_mbps'):g}" for row in sorted_rows]
        edge_values = [_float_value(row, "estimated_edge_ms") for row in sorted_rows]
        transfer_values = [_float_value(row, "estimated_transfer_ms") for row in sorted_rows]
        cloud_values = [_float_value(row, "estimated_cloud_ms") for row in sorted_rows]
        positions = list(range(len(sorted_rows)))

        fig, ax = plt.subplots(figsize=(max(8, len(sorted_rows) * 0.5), 5))
        ax.bar(positions, edge_values, label="Edge")
        transfer_bottom = edge_values
        ax.bar(positions, transfer_values, bottom=transfer_bottom, label="Transfer")
        cloud_bottom = [edge + transfer for edge, transfer in zip(edge_values, transfer_values)]
        ax.bar(positions, cloud_values, bottom=cloud_bottom, label="Cloud")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Estimated Latency (ms)")
        ax.set_title(
            f"{model_name} {STRATEGY_LABELS.get(strategy, strategy)} stage breakdown under "
            f"{_condition_title(condition_label, uses_edge_condition)}"
        )
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()

        filename = (
            f"{_safe_token(model_name)}_{_condition_filename_part(condition_label, uses_edge_condition)}_"
            f"{_safe_token(strategy)}_stage_breakdown.{plot_format}"
        )
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs


def plot_granularity_comparison(
    csv_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    plot_format: str = "png",
    strategy: str = "dsl",
) -> list[str]:
    """Compare estimated latency curves from several partition granularities."""
    plt = _load_pyplot()
    if plt is None:
        return []
    if not csv_paths:
        return []

    rows_by_granularity: dict[str, list[dict[str, str]]] = {}
    for granularity, csv_path in csv_paths.items():
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Granularity CSV for '{granularity}' does not exist: {csv_path}")
        rows = [row for row in _read_rows(csv_path) if row.get("strategy") == strategy]
        if rows:
            rows_by_granularity[str(granularity)] = rows

    if not rows_by_granularity:
        return []

    all_rows = [row for rows in rows_by_granularity.values() for row in rows]
    uses_edge_condition = _uses_edge_condition(all_rows)
    condition_labels = _condition_keys(all_rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    first_rows = next(iter(rows_by_granularity.values()))
    model_name = first_rows[0].get("model", "model")
    outputs = []
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    for condition_label in condition_labels:
        fig, ax = plt.subplots(figsize=(8, 5))
        for index, (granularity, rows) in enumerate(rows_by_granularity.items()):
            selected_rows = [
                row for row in rows if _condition_key(row) == condition_label
            ]
            if not selected_rows:
                continue
            selected_rows = sorted(selected_rows, key=lambda item: float(item["bandwidth_mbps"]))
            bandwidths = [_float_value(row, "bandwidth_mbps") for row in selected_rows]
            totals = [_float_value(row, "estimated_total_ms") for row in selected_rows]
            ax.plot(
                bandwidths,
                totals,
                label=granularity,
                marker=markers[index % len(markers)],
                linestyle=linestyles[index % len(linestyles)],
                linewidth=2.0,
            )

        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Estimated Total Latency (ms)")
        ax.set_title(
            f"{model_name} {STRATEGY_LABELS.get(strategy, strategy)} latency by granularity under "
            f"{_condition_title(condition_label, uses_edge_condition)}"
        )
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(title="Granularity")
        fig.tight_layout()

        filename = (
            f"{_safe_token(model_name)}_{_condition_filename_part(condition_label, uses_edge_condition)}_"
            f"{_safe_token(strategy)}_granularity_comparison.{plot_format}"
        )
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs


def run_plot_estimate_config(config: dict[str, Any]) -> list[str]:
    csv_path = config.get("csv") or config.get("report_csv")
    if not csv_path:
        raise ValueError("plot-estimate config field 'csv' is required.")
    plot_dir = config.get("plot_dir", "results/plots_estimate_extra")
    if not plot_dir:
        return []
    plot_format = str(config.get("plot_format", "png"))
    plots = set(config.get("plots", ["latency", "load_sweep", "speedup_heatmap", "stage_breakdown"]))
    outputs: list[str] = []
    if "latency" in plots:
        outputs.extend(plot_estimate_latency(csv_path, plot_dir, plot_format))
    if "load_sweep" in plots:
        outputs.extend(plot_estimate_latency_by_bandwidth(csv_path, plot_dir, plot_format))
    if "speedup_heatmap" in plots:
        outputs.extend(plot_estimate_speedup_heatmap(csv_path, plot_dir, plot_format))
    if "stage_breakdown" in plots:
        outputs.extend(plot_estimate_stage_breakdown(csv_path, plot_dir, plot_format, str(config.get("strategy", "dsl"))))
    print("Plots: " + ", ".join(outputs) if outputs else "No plots generated.")
    return outputs


def run_plot_granularity_config(config: dict[str, Any]) -> list[str]:
    csv_paths = config.get("csvs") or config.get("granularity_csvs")
    if not isinstance(csv_paths, dict) or not csv_paths:
        raise ValueError("plot-granularity config field 'csvs' must be a non-empty object.")
    plot_dir = config.get("plot_dir", "results/plots_granularity")
    if not plot_dir:
        return []
    outputs = plot_granularity_comparison(
        csv_paths={str(key): str(value) for key, value in csv_paths.items()},
        output_dir=str(plot_dir),
        plot_format=str(config.get("plot_format", "png")),
        strategy=str(config.get("strategy", "dsl")),
    )
    print("Plots: " + ", ".join(outputs) if outputs else "No plots generated.")
    return outputs


def plot_estimate_latency(csv_path: str | Path, output_dir: str | Path, plot_format: str = "png") -> list[str]:
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_rows(csv_path)
    if not rows:
        return []

    uses_edge_condition = _uses_edge_condition(rows)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[_condition_key(row)][row["strategy"]].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")
    outputs = []

    condition_order = {label: index for index, label in enumerate(_condition_keys(rows))}
    for condition_label, by_strategy in sorted(grouped.items(), key=lambda item: condition_order[item[0]]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in STRATEGY_ORDER:
            strategy_rows = sorted(by_strategy.get(strategy, []), key=lambda item: float(item["bandwidth_mbps"]))
            if not strategy_rows:
                continue
            bandwidths = [float(item["bandwidth_mbps"]) for item in strategy_rows]
            totals = [float(item["estimated_total_ms"]) for item in strategy_rows]
            ax.plot(
                bandwidths,
                totals,
                label=STRATEGY_LABELS.get(strategy, strategy),
                **STRATEGY_STYLES.get(strategy, {}),
            )

        # ax.set_xscale("log", base=10)
        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Estimated Total Latency (ms)")
        ax.set_title(f"{model_name} estimated latency under {_condition_title(condition_label, uses_edge_condition)}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()

        filename = f"{_safe_token(model_name)}_{_condition_filename_part(condition_label, uses_edge_condition)}_estimated_latency.{plot_format}"
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs


def plot_estimate_latency_by_bandwidth(csv_path: str | Path, output_dir: str | Path, plot_format: str = "png") -> list[str]:
    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_rows(csv_path)
    if not rows:
        return []

    uses_edge_condition = _uses_edge_condition(rows)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["bandwidth_mbps"]][row["strategy"]].append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = rows[0].get("model", "model")
    outputs = []

    for bandwidth_mbps, by_strategy in sorted(grouped.items(), key=lambda item: float(item[0])):
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels: list[str] = []
        for strategy in STRATEGY_ORDER:
            strategy_rows = sorted(
                by_strategy.get(strategy, []),
                key=lambda item: _condition_order(item),
            )
            if not strategy_rows:
                continue
            if uses_edge_condition:
                loads = list(range(len(strategy_rows)))
                if not x_labels:
                    x_labels = [_condition_key(item) for item in strategy_rows]
            else:
                loads = [float(item["cpu_load_target"]) for item in strategy_rows]
            totals = [float(item["estimated_total_ms"]) for item in strategy_rows]
            ax.plot(
                loads,
                totals,
                label=STRATEGY_LABELS.get(strategy, strategy),
                **STRATEGY_STYLES.get(strategy, {}),
            )

        if uses_edge_condition:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
        ax.set_xlabel(_condition_axis_label(uses_edge_condition))
        ax.set_ylabel("Estimated Total Latency (ms)")
        ax.set_title(f"{model_name} estimated latency at {float(bandwidth_mbps):g} Mbps")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()

        sweep_name = "condition_sweep" if uses_edge_condition else "load_sweep"
        filename = f"{_safe_token(model_name)}_bw{_safe_token(bandwidth_mbps)}_{sweep_name}_estimated_latency.{plot_format}"
        destination = output_path / filename
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        outputs.append(str(destination))

    return outputs
