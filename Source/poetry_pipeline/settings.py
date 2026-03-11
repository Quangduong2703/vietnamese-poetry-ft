from __future__ import annotations

from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
CONFIGS_DIR = SOURCE_DIR / "configs"
SCRIPTS_DIR = SOURCE_DIR / "scripts"


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
