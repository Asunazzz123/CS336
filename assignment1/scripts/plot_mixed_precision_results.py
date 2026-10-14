"""Reproduce the BF16 sweep figures from archived metrics without smoothing.

Run from assignment1 with:
    conda run -n agent python scripts/plot_mixed_precision_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#D97700", "#009E73", "#111827"]


def load_runs(results_root: Path) -> tuple[list[dict], dict]:
    with (results_root / "summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    best = json.loads((results_root / "best_run.json").read_text())
    runs = []
    for row in rows:
        if row["status"] != "complete" or row["precision"] != "bf16":
            raise ValueError(f"Expected a completed BF16 run: {row['name']}")
        run = dict(row)
        for key in ("lr", "warmup_ratio", "final_val_loss"):
            run[key] = float(row[key])
        for key in ("batch_size", "max_iters", "training_tokens", "attempt"):
            run[key] = int(row[key])
        attempt_dir = results_root / row["name"] / f"attempt_{run['attempt']:03d}"
        run["config"] = json.loads((attempt_dir / "config.json").read_text())
        run["metrics"] = [
            json.loads(line)
            for line in (attempt_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        runs.append(run)
    if len(runs) != 64 or len({run["name"] for run in runs}) != 64:
        raise ValueError("Expected all 64 distinct runs from the archived sweep")
    baseline = next(run for run in runs if run["name"] == best["name"])
    return runs, baseline


def conditional_slices(runs: list[dict], baseline: dict) -> list[tuple[str, list[dict]]]:
    keys = ("lr", "batch_size", "warmup_ratio")
    slices = []
    for variable in keys:
        selected = sorted(
            (
                run
                for run in runs
                if all(run[key] == baseline[key] for key in keys if key != variable)
            ),
            key=lambda run: run[variable],
        )
        if len(selected) != 4:
            raise ValueError(f"Expected four values for {variable}")
        slices.append((variable, selected))
    return slices


def value_label(variable: str, value: float) -> str:
    if variable == "lr":
        return f"{value:.0e}"
    if variable == "warmup_ratio":
        return f"{value:.0%}"
    return str(value)


def baseline_label(baseline: dict) -> str:
    return (
        f"Baseline: lr={baseline['lr']:.0e}, batch={baseline['batch_size']}, "
        f"warmup={baseline['warmup_ratio']:.0%}"
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#262626")
    ax.grid(True, color="#e6e6e6", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)


def add_heading(fig: plt.Figure, title: str, subtitle: str, footer: str) -> None:
    fig.suptitle(title, y=0.985, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.915, subtitle, ha="center", color="#52627b", fontsize=10.5)
    fig.text(0.5, 0.015, footer, ha="center", color="#52627b", fontsize=9)


def plot_learning_curves(
    slices: list[tuple[str, list[dict]]], baseline: dict, output_dir: Path, event: str
) -> None:
    validation = event == "validation"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.74, wspace=0.13)
    lr = f"{baseline['lr']:.0e}"
    batch = baseline["batch_size"]
    warmup = f"{baseline['warmup_ratio']:.0%}"
    panel_titles = (
        f"Vary Learning rate\nFixed: batch_size={batch}, warmup_ratio={warmup}",
        f"Vary Batch size\nFixed: lr={lr}, warmup_ratio={warmup}",
        f"Vary Warmup\nFixed: lr={lr}, batch_size={batch}",
    )
    for ax, (variable, selected), title in zip(axes, slices, panel_titles):
        for index, run in enumerate(selected):
            records = [record for record in run["metrics"] if record["event"] == event]
            if validation and len(records) != 10:
                raise ValueError(f"Expected ten validation checkpoints: {run['name']}")
            tokens_per_step = run["batch_size"] * run["config"]["context_length"]
            x = [record["iteration"] * tokens_per_step / 1e6 for record in records]
            loss_key = "val_loss" if validation else "loss"
            y = [record[loss_key] for record in records]
            if validation and abs(y[-1] - run["final_val_loss"]) > 1e-10:
                raise ValueError(f"Final validation value mismatch: {run['name']}")
            is_baseline = run["name"] == baseline["name"]
            label = value_label(variable, run[variable])
            if is_baseline:
                label += " (baseline)"
            ax.plot(
                x,
                y,
                label=label,
                color=COLORS[index],
                linewidth=2.3 if is_baseline else 1.7,
                marker="o" if validation else None,
                markersize=3.6,
                alpha=1 if validation else 0.82,
            )
        style_axes(ax)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlim(0, 41.5)
        ax.set_xlabel("Training tokens (millions)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.7, frameon=False)
    axes[0].set_ylabel("Cross-entropy loss (nats/token)", fontsize=10)
    training_millions = baseline["training_tokens"] / 1e6
    if validation:
        config = baseline["config"]
        validation_tokens = config["eval_batches"] * config["eval_batch_size"] * config["context_length"]
        title = "BF16 validation loss over training tokens"
        footer = (
            f"Same {validation_tokens:,} validation tokens at every checkpoint; "
            "unsmoothed observed losses."
        )
        filename = "bf16_validation_learning_curves.png"
    else:
        title = "BF16 training batch loss over training tokens"
        footer = "Single training-batch losses; batch sizes differ. No smoothing applied."
        filename = "bf16_training_learning_curves.png"
    add_heading(
        fig,
        title,
        f"{baseline_label(baseline)} | {training_millions:.2f}M training tokens per run",
        footer,
    )
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_sensitivity(slices: list[tuple[str, list[dict]]], baseline: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.78, wspace=0.30)
    x_labels = ("Learning rate", "Batch size", "Warmup (target fraction of steps)")
    for ax, (variable, selected), x_label in zip(axes, slices, x_labels):
        x = [run[variable] for run in selected]
        y = [run["final_val_loss"] for run in selected]
        ax.plot(x, y, marker="o", color=COLORS[0], linewidth=2, markersize=6)
        ax.scatter(
            [baseline[variable]],
            [baseline["final_val_loss"]],
            color=COLORS[-1],
            s=180,
            marker="*",
            label="Best baseline",
            zorder=3,
        )
        if variable in ("lr", "batch_size"):
            ax.set_xscale("log", base=10 if variable == "lr" else 2)
        ax.set_xticks(x, [value_label(variable, value) for value in x])
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        span = max(y) - min(y)
        ax.set_ylim(min(y) - span * 0.16, max(y) + span * 0.28)
        for position, loss in zip(x, y):
            ax.annotate(
                f"{loss:.4f}",
                (position, loss),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Final validation loss", fontsize=10)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        style_axes(ax)
    actual_warmup = baseline["config"]["warmup_iters"] / baseline["max_iters"]
    footer = (
        "Panels use separate y-axis scales. Warmup steps are rounded up; "
        f"{baseline['warmup_ratio']:.0%} at batch {baseline['batch_size']} "
        f"is actually {actual_warmup:.2%}."
    )
    add_heading(
        fig,
        "BF16 final validation loss: one variable at a time",
        f"{baseline_label(baseline)} | All other variables held at baseline",
        footer,
    )
    fig.savefig(output_dir / "bf16_final_validation_sensitivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path,
        default=ASSIGNMENT_ROOT / "result" / "sweep_tinystories_10k_bf16",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=ASSIGNMENT_ROOT / "result" / "asset" / "Mixed-Precision")
    args = parser.parse_args()
    runs, baseline = load_runs(args.results_root)
    slices = conditional_slices(runs, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelcolor": "#111111"})
    plot_learning_curves(slices, baseline, args.output_dir, "validation")
    plot_sensitivity(slices, baseline, args.output_dir)
    plot_learning_curves(slices, baseline, args.output_dir, "train")
    for variable, selected in slices:
        print(f"{variable}: " + ", ".join(
            f"{value_label(variable, run[variable])}={run['final_val_loss']:.10f}"
            for run in selected
        ))
    print(f"Saved three BF16 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
