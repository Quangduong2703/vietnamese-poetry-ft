from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from poetry_pipeline.settings import CONFIGS_DIR, DATA_DIR, OUTPUTS_DIR, ensure_runtime_dirs
from poetry_pipeline.visualization import render_metrics_artifacts


DEFAULT_CONFIG = {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "train_file": str(DATA_DIR / "train_creative_train.jsonl"),
    "valid_file": str(DATA_DIR / "train_creative_valid.jsonl"),
    "output_dir": str(OUTPUTS_DIR / "qwen25_15b_poetry_qlora"),
    "max_length": 512,
    "num_train_epochs": 1.0,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "eval_steps": 250,
    "save_steps": 250,
    "logging_steps": 10,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "max_grad_norm": 0.3,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj",
    "packing": False,
    "dataloader_num_workers": 0,
    "fp16": False,
    "gradient_checkpointing": True,
    "seed": 42,
    "max_train_samples": None,
    "max_eval_samples": None,
    "resume_from_checkpoint": None,
    "plot_metrics": True,
}
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "qwen25_15b_qlora.json"


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def load_config(config_path: str | Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}")
        user_config = json.loads(path.read_text(encoding="utf-8"))
        config.update(user_config)
    return config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-1.5B-Instruct on Vietnamese poetry."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model_id", default=defaults["model_id"])
    parser.add_argument("--train_file", default=defaults["train_file"])
    parser.add_argument("--valid_file", default=defaults["valid_file"])
    parser.add_argument("--output_dir", default=defaults["output_dir"])
    parser.add_argument("--max_length", type=int, default=defaults["max_length"])
    parser.add_argument("--num_train_epochs", type=float, default=defaults["num_train_epochs"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=defaults["per_device_train_batch_size"],
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=defaults["per_device_eval_batch_size"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=defaults["gradient_accumulation_steps"],
    )
    parser.add_argument("--eval_steps", type=int, default=defaults["eval_steps"])
    parser.add_argument("--save_steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--logging_steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--warmup_ratio", type=float, default=defaults["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--max_grad_norm", type=float, default=defaults["max_grad_norm"])
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument("--target_modules", default=defaults["target_modules"])
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=defaults["packing"],
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=defaults["dataloader_num_workers"],
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=defaults["fp16"],
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults["gradient_checkpointing"],
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max_train_samples", type=int, default=defaults["max_train_samples"])
    parser.add_argument("--max_eval_samples", type=int, default=defaults["max_eval_samples"])
    parser.add_argument("--resume_from_checkpoint", default=defaults["resume_from_checkpoint"])
    parser.add_argument(
        "--plot_metrics",
        action=argparse.BooleanOptionalAction,
        default=defaults["plot_metrics"],
    )
    args = parser.parse_args()
    args.train_file = Path(args.train_file)
    args.valid_file = Path(args.valid_file)
    args.output_dir = Path(args.output_dir)
    return args


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def print_gpu_summary() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script expects an NVIDIA GPU.")

    local_rank = get_local_rank()
    props = torch.cuda.get_device_properties(local_rank)
    total_gb = props.total_memory / 1024**3
    if is_main_process():
        print(f"World size: {get_world_size()}")
        print(f"GPU: {props.name}")
        print(f"VRAM per GPU: {total_gb:.2f} GB")
        if total_gb <= 4.1:
            print("Note: 4 GB VRAM is tight. Keep max_length=512 and close other GPU apps.")


def load_jsonl_dataset(
    train_file: Path,
    valid_file: Path,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    data_files = {"train": str(train_file), "validation": str(valid_file)}
    dataset = load_dataset("json", data_files=data_files)
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )

    original_columns = dataset["train"].column_names

    def to_prompt_completion(example: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": example["instruction"]}],
            "completion": [{"role": "assistant", "content": example["poem"]}],
        }

    dataset = dataset.map(to_prompt_completion, remove_columns=original_columns)
    return dataset


def build_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(
    model_id: str,
    target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
):
    local_rank = get_local_rank()
    device_map = {"": local_rank} if get_world_size() > 1 else None
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if is_main_process():
        model.print_trainable_parameters()
    return model


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def save_run_config(args: argparse.Namespace) -> None:
    if not is_main_process():
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()
    require_file(args.train_file)
    require_file(args.valid_file)
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    print_gpu_summary()
    set_seed(args.seed)
    save_run_config(args)
    if is_main_process():
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(args.output_dir / "runs")

    if is_main_process():
        print(f"Train rows on disk: {count_jsonl_rows(args.train_file)}")
        print(f"Valid rows on disk: {count_jsonl_rows(args.valid_file)}")

    dataset = load_jsonl_dataset(
        args.train_file,
        args.valid_file,
        args.max_train_samples,
        args.max_eval_samples,
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    if is_main_process():
        print(f"Train rows loaded: {len(train_dataset)}")
        print(f"Valid rows loaded: {len(eval_dataset)}")

    tokenizer = build_tokenizer(args.model_id)
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    model = build_model(
        model_id=args.model_id,
        target_modules=target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        packing=args.packing,
        completion_only_loss=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        fp16=args.fp16,
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        optim="paged_adamw_8bit",
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        dataset_num_proc=1,
        ddp_find_unused_parameters=False if get_world_size() > 1 else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        trainer.state.save_to_json(str(args.output_dir / "trainer_state.json"))
    if args.plot_metrics and trainer.is_world_process_zero():
        artifacts = render_metrics_artifacts(trainer.state.log_history, args.output_dir)
        print(f"Metrics JSONL:      {artifacts['jsonl']}")
        print(f"Metrics CSV:        {artifacts['csv']}")
        print(f"Metrics summary:    {artifacts['summary']}")
        print(f"Metrics dashboard:  {artifacts['dashboard']}")
        print(f"Training complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
