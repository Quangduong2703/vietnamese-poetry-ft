# Source Layout

## Overview

This folder is organized as a small, reusable project instead of a flat script dump.

```text
Source/
  configs/                 Runtime presets
  poetry_pipeline/         Reusable Python modules
  scripts/                 Thin executable entrypoints
  requirements_qwen25_qlora.txt
```

## Structure

- `poetry_pipeline/data_prep.py`: dataset download, normalization, JSONL export, train/valid split generation.
- `poetry_pipeline/qlora_train.py`: QLoRA training pipeline for `Qwen2.5-1.5B-Instruct`.
- `poetry_pipeline/settings.py`: project-level paths (`Data`, `Outputs`, `Source`).
- `configs/qwen25_15b_qlora.json`: default training preset.
- `scripts/*.py`: stable CLI entrypoints that call into the package.

## Recommended Commands

Prepare data:

```powershell
python D:\NLP\Source\scripts\prepare_dataset.py
```

Run QLoRA training:

```powershell
powershell -ExecutionPolicy Bypass -File D:\NLP\Source\scripts\run_qwen25_15b_qlora.ps1
```

Run on Google Colab:

```text
1. Upload train_creative_train.jsonl and train_creative_valid.jsonl to Google Drive
2. Open Source/notebooks/qwen25_15b_qlora_colab.ipynb in Colab
3. Run the cells in order
```

Colab config preset:

```text
Source/configs/qwen25_15b_qlora_colab.json
```

After training, metrics are written under:

```text
D:\NLP\Outputs\qwen25_15b_poetry_qlora\metrics\
  metrics_history.jsonl
  metrics_history.csv
  summary.json
  training_dashboard.png
```

TensorBoard logs are written under:

```text
D:\NLP\Outputs\qwen25_15b_poetry_qlora\runs\
```

Open TensorBoard:

```powershell
tensorboard --logdir D:\NLP\Outputs\qwen25_15b_poetry_qlora\runs
```

Smoke-test training on a tiny subset:

```powershell
python D:\NLP\Source\scripts\train_qwen25_15b_qlora.py `
  --config D:\NLP\Source\configs\qwen25_15b_qlora.json `
  --max_train_samples 8 `
  --max_eval_samples 4 `
  --output_dir D:\NLP\Outputs\qwen25_15b_poetry_qlora_smoketest
```

Rebuild charts from an existing run:

```powershell
python D:\NLP\Source\scripts\plot_training_metrics.py `
  --output_dir D:\NLP\Outputs\qwen25_15b_poetry_qlora
```
