from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")


def _coerce_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def save_log_history(log_history: list[dict], output_dir: Path) -> tuple[Path, Path, Path]:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = metrics_dir / "metrics_history.jsonl"
    csv_path = metrics_dir / "metrics_history.csv"
    summary_path = metrics_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in log_history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(log_history)
    if not frame.empty:
        numeric_columns = [
            "step",
            "epoch",
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "entropy",
            "eval_entropy",
            "mean_token_accuracy",
            "eval_mean_token_accuracy",
            "num_tokens",
            "eval_num_tokens",
            "train_runtime",
            "eval_runtime",
        ]
        frame = _coerce_numeric_frame(frame, numeric_columns)
        frame.to_csv(csv_path, index=False, encoding="utf-8")

        summary = {
            "num_log_entries": int(len(frame)),
            "last_step": int(frame["step"].dropna().max()) if "step" in frame and frame["step"].notna().any() else None,
            "best_eval_loss": (
                float(frame["eval_loss"].dropna().min())
                if "eval_loss" in frame and frame["eval_loss"].notna().any()
                else None
            ),
            "final_train_loss": (
                float(frame["loss"].dropna().iloc[-1])
                if "loss" in frame and frame["loss"].notna().any()
                else None
            ),
            "final_eval_loss": (
                float(frame["eval_loss"].dropna().iloc[-1])
                if "eval_loss" in frame and frame["eval_loss"].notna().any()
                else None
            ),
        }
    else:
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        summary = {
            "num_log_entries": 0,
            "last_step": None,
            "best_eval_loss": None,
            "final_train_loss": None,
            "final_eval_loss": None,
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path, csv_path, summary_path


def _plot_series(ax, frame: pd.DataFrame, x_col: str, y_col: str, title: str, color: str) -> None:
    subset = frame[[x_col, y_col]].dropna()
    if subset.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(subset[x_col], subset[y_col], color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)


def plot_training_dashboard(log_history: list[dict], output_dir: Path) -> Path:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = metrics_dir / "training_dashboard.png"

    frame = pd.DataFrame(log_history)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No log history available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(dashboard_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return dashboard_path

    numeric_columns = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "mean_token_accuracy",
        "eval_mean_token_accuracy",
        "entropy",
        "eval_entropy",
    ]
    frame = _coerce_numeric_frame(frame, numeric_columns)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    _plot_series(axes[0], frame, "step", "loss", "Train Loss", "#1f77b4")
    _plot_series(axes[1], frame, "step", "eval_loss", "Eval Loss", "#d62728")
    _plot_series(axes[2], frame, "step", "learning_rate", "Learning Rate", "#2ca02c")
    _plot_series(axes[3], frame, "step", "grad_norm", "Grad Norm", "#ff7f0e")

    accuracy_ax = axes[4]
    train_acc = frame[["step", "mean_token_accuracy"]].dropna()
    eval_acc = frame[["step", "eval_mean_token_accuracy"]].dropna()
    if train_acc.empty and eval_acc.empty:
        accuracy_ax.set_title("Mean Token Accuracy")
        accuracy_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=accuracy_ax.transAxes)
        accuracy_ax.set_axis_off()
    else:
        if not train_acc.empty:
            accuracy_ax.plot(train_acc["step"], train_acc["mean_token_accuracy"], label="train", color="#9467bd", linewidth=2)
        if not eval_acc.empty:
            accuracy_ax.plot(eval_acc["step"], eval_acc["eval_mean_token_accuracy"], label="eval", color="#8c564b", linewidth=2)
        accuracy_ax.set_title("Mean Token Accuracy")
        accuracy_ax.set_xlabel("Step")
        accuracy_ax.grid(alpha=0.25)
        accuracy_ax.legend()

    entropy_ax = axes[5]
    train_entropy = frame[["step", "entropy"]].dropna()
    eval_entropy = frame[["step", "eval_entropy"]].dropna()
    if train_entropy.empty and eval_entropy.empty:
        entropy_ax.set_title("Entropy")
        entropy_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=entropy_ax.transAxes)
        entropy_ax.set_axis_off()
    else:
        if not train_entropy.empty:
            entropy_ax.plot(train_entropy["step"], train_entropy["entropy"], label="train", color="#17becf", linewidth=2)
        if not eval_entropy.empty:
            entropy_ax.plot(eval_entropy["step"], eval_entropy["eval_entropy"], label="eval", color="#bcbd22", linewidth=2)
        entropy_ax.set_title("Entropy")
        entropy_ax.set_xlabel("Step")
        entropy_ax.grid(alpha=0.25)
        entropy_ax.legend()

    fig.suptitle("Training Dashboard", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return dashboard_path


def render_metrics_artifacts(log_history: list[dict], output_dir: Path) -> dict[str, Path]:
    jsonl_path, csv_path, summary_path = save_log_history(log_history, output_dir)
    dashboard_path = plot_training_dashboard(log_history, output_dir)
    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "summary": summary_path,
        "dashboard": dashboard_path,
    }


def render_metrics_from_trainer_state(trainer_state_path: Path) -> dict[str, Path]:
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = payload.get("log_history", [])
    return render_metrics_artifacts(log_history, trainer_state_path.parent)
