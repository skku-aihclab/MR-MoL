#!/usr/bin/env python3
"""Stage 1 Training: Graph-Text Alignment."""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.molecular_llm import MolecularLLM
from src.datamodule.dataset import Stage1Dataset
from src.datamodule.collator import Stage1Collator
from src.training.stage1_trainer import Stage1Trainer
from src.utils.helpers import set_seed, load_config, print_model_info


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Training (MolCA Q-Former)")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    config = load_config(str(config_path))

    seed = config.get("hardware", {}).get("seed", 42)
    set_seed(seed)

    device = config.get("hardware", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("\n" + "=" * 60)
    print("STAGE 1 (Q-Former): GRAPH-TEXT ALIGNMENT")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Config and imports validated successfully!")
        return

    model_config = config["model"]
    training_config = config["training"]
    data_config = config.get("data", {})
    checkpoint_config = config.get("checkpointing", {})

    print("\nInitializing model...")
    model = MolecularLLM(
        llm_name=model_config.get("llm_name", "meta-llama/Llama-3.1-8B-Instruct"),
        molca_ckpt_path=model_config["molca_ckpt_path"],
        graph_dim=model_config.get("graph_dim", 300),
        llm_dim=model_config.get("llm_dim", 4096),
        num_query_tokens=model_config.get("num_query_tokens", 8),
        dropout=model_config.get("dropout", 0.1),
        use_lora=False,
        freeze_graph_encoder=model_config.get("freeze_graph_encoder", False),
        freeze_projector=model_config.get("freeze_projector", False),
        device=device,
    )

    print_model_info(model, "Stage 1 Model (Q-Former)")

    print("\nLoading dataset...")
    train_path = PROJECT_ROOT / data_config.get("train_path", "data_prep/stage1/train.json")
    system_prompts_path = PROJECT_ROOT / data_config.get(
        "system_prompts_path", "data_prep/stage1/system_prompts.json"
    )

    train_dataset = Stage1Dataset(
        data_path=str(train_path),
        tokenizer=model.tokenizer,
        system_prompts_path=str(system_prompts_path),
        max_length=training_config.get("max_length", 512),
        mol_token=model.mol_token,
        num_mol_tokens=model.num_mol_tokens,
    )

    train_collator = Stage1Collator(tokenizer=model.tokenizer, include_graphs=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 4),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        collate_fn=train_collator,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Training batches: {len(train_dataloader)}")

    trainer = Stage1Trainer(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=training_config["learning_rate"],
        qformer_lr=training_config["qformer_lr"],
        graph_encoder_lr=training_config["graph_encoder_lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        num_epochs=training_config.get("epochs", 5),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        save_dir=str(PROJECT_ROOT / checkpoint_config.get("save_dir", "checkpoints/trained/stage1")),
        device=device,
    )


    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    print("\nStarting training...")
    trainer.train(start_epoch=start_epoch)

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
