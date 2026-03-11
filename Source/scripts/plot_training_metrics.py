from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_pipeline.visualization import render_metrics_artifacts, render_metrics_from_trainer_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render training charts from a run output directory or trainer_state.json.")
    parser.add_argument("--output_dir", default=None, help="Run output directory containing trainer_state.json or metrics history.")
    parser.add_argument("--trainer_state", default=None, help="Direct path to trainer_state.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trainer_state:
        trainer_state_path = Path(args.trainer_state)
        artifacts = render_metrics_from_trainer_state(trainer_state_path)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        trainer_state_path = output_dir / "trainer_state.json"
        metrics_jsonl = output_dir / "metrics" / "metrics_history.jsonl"
        if trainer_state_path.exists():
            artifacts = render_metrics_from_trainer_state(trainer_state_path)
        elif metrics_jsonl.exists():
            log_history = [json.loads(line) for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            artifacts = render_metrics_artifacts(log_history, output_dir)
        else:
            raise FileNotFoundError(
                f"Could not find trainer_state.json or metrics_history.jsonl in {output_dir}"
            )
    else:
        raise SystemExit("Provide either --output_dir or --trainer_state.")

    print(f"Metrics JSONL:      {artifacts['jsonl']}")
    print(f"Metrics CSV:        {artifacts['csv']}")
    print(f"Metrics summary:    {artifacts['summary']}")
    print(f"Metrics dashboard:  {artifacts['dashboard']}")


if __name__ == "__main__":
    main()
