#!/usr/bin/env python3
"""Stage 2 Training: Multi-task Instruction Tuning."""

import os
import sys
import random
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.molecular_llm import MolecularLLM
from src.datamodule.dataset import Stage2Dataset
from src.datamodule.collator import Stage2Collator
from src.training.stage2_trainer import Stage2Trainer
from src.utils.helpers import set_seed, load_config, print_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 2 Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Stage 2 checkpoint to resume from",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=None,
        help="Specific task(s) to train on (e.g., bbbp tox21). Matches against filenames.",
    )
    parser.add_argument(
        "--task-tag",
        type=str,
        default=None,
        help="Override the tag used in the no-validation-fallback checkpoint "
             "filename (stage2_<tag>_epoch<N>.pt). Default: tasks joined by '+', "
             "or 'all' if --task is not set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test imports and config loading without training",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}

    # Set seed
    seed = config.get("hardware", {}).get("seed", 42)
    set_seed(seed)

    device = config.get("hardware", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("\n" + "="*60)
    print("STAGE 2: MULTI-TASK INSTRUCTION TUNING")
    print("="*60)

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] Config and imports validated successfully!")
        return

    # Extract configs
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    checkpoint_config = config.get("checkpointing", {})

    # Create model with LoRA
    print("\nInitializing model with LoRA...")
    model = MolecularLLM(
        llm_name=model_config.get("llm_name", "meta-llama/Llama-3.1-8B-Instruct"),
        graph_dim=model_config.get("graph_dim", 300),
        llm_dim=model_config.get("llm_dim", 4096),
        num_query_tokens=model_config.get("num_query_tokens", 8),
        dropout=model_config.get("dropout", 0.1),
        use_lora=model_config.get("use_lora", True),
        lora_r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        lora_target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        freeze_graph_encoder=True,
        freeze_projector=model_config.get("freeze_projector", False),
        device=device,
    )

    use_graph = model_config.get("use_graph", True)
    if use_graph:
        stage1_ckpt = args.stage1_checkpoint or checkpoint_config.get("stage1_checkpoint")
        if not stage1_ckpt:
            raise ValueError("Stage 2 requires `checkpointing.stage1_checkpoint`.")
        stage1_path = PROJECT_ROOT / stage1_ckpt
        if not stage1_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage1_path}")
        print(f"\nLoading Stage 1 checkpoint from {stage1_path}")
        ckpt = torch.load(stage1_path, map_location=device)
        proj_missing, proj_unexpected = model.projector.load_state_dict(
            ckpt["projector"], strict=False
        )
        print(
            f"  projector: missing={len(proj_missing)} unexpected={len(proj_unexpected)}"
        )
        if ckpt.get("graph_encoder") is not None:
            model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
            print("  graph_encoder + projector loaded")
        else:
            print("  projector loaded (graph_encoder not in Stage 1 ckpt)")
    else:
        print("\nuse_graph=False -> skipping Stage 1 (graph-alignment) checkpoint; vanilla LLM + LoRA")

    print_model_info(model, "Stage 2 Model")

    # Resolve task filter
    task_tag = "all"
    if args.task:
        task_tag = args.task_tag if args.task_tag else "+".join(args.task)
        print(f"\nTask filter: {args.task}")

    # Create datasets
    print("\nLoading datasets...")
    train_dir = PROJECT_ROOT / data_config.get("train_path", "data_prep/stage2/train")
    valid_dir = PROJECT_ROOT / data_config.get("valid_path", "data_prep/stage2/valid")
    system_prompts_path = PROJECT_ROOT / data_config.get("system_prompts_path", "data_prep/stage2/instruction_templates/system_prompts.json")

    # Filter task files if --task is specified
    if args.task and train_dir.is_dir():
        import tempfile, json, shutil
        train_path = Path(tempfile.mkdtemp())
        valid_path = Path(tempfile.mkdtemp()) if valid_dir.is_dir() else valid_dir
        for json_file in sorted(train_dir.glob("*.json")):
            if any(t.lower() in json_file.stem.lower() for t in args.task):
                shutil.copy(json_file, train_path / json_file.name)
        if valid_dir.is_dir():
            for json_file in sorted(valid_dir.glob("*.json")):
                if any(t.lower() in json_file.stem.lower() for t in args.task):
                    shutil.copy(json_file, valid_path / json_file.name)
        matched = list(train_path.glob("*.json"))
        if not matched:
            available = [f.stem for f in sorted(train_dir.glob("*.json"))]
            print(f"No tasks matched '{args.task}'. Available: {available}")
            return
        print(f"Matched task files: {[f.name for f in matched]}")
    else:
        train_path = train_dir
        valid_path = valid_dir

    train_dataset = Stage2Dataset(
        data_path=str(train_path),
        tokenizer=model.tokenizer,
        system_prompts_path=str(system_prompts_path),
        max_length=training_config.get("max_length", 1024),
        mol_token=model.mol_token,
        num_mol_tokens=model.num_mol_tokens,
        use_graph=model_config.get("use_graph", True),
    )

    valid_dataset = None
    if valid_path.exists():
        valid_dataset = Stage2Dataset(
            data_path=str(valid_path),
            tokenizer=model.tokenizer,
            system_prompts_path=str(system_prompts_path),
            max_length=training_config.get("max_length", 1024),
            mol_token=model.mol_token,
            num_mol_tokens=model.num_mol_tokens,
            use_graph=model_config.get("use_graph", True),
        )

    # Create dataloaders
    train_collator = Stage2Collator(
        tokenizer=model.tokenizer,
        include_graphs=use_graph,
    )

    num_workers = data_config.get("num_workers", 4)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    valid_dataloader = None
    if valid_dataset:
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=training_config.get("batch_size", 8),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Training batches: {len(train_dataloader)}")
    if valid_dataset:
        print(f"Validation samples: {len(valid_dataset)}")

    # LRs required by active param groups — no code-side defaults.
    # projector_lr required when freeze_projector=False; lora_lr when use_lora=True.
    projector_lr = training_config["projector_lr"] if not model.freeze_projector else None
    lora_lr = training_config["lora_lr"] if model.use_lora else None

    trainer = Stage2Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        projector_lr=projector_lr,
        lora_lr=lora_lr,
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        num_epochs=training_config.get("epochs", 3),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        save_dir=str(PROJECT_ROOT / checkpoint_config.get("save_dir", "checkpoints/trained")),
        task_tag=task_tag,
        device=device,
    )


    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(start_epoch=start_epoch)

    print("\n" + "="*60)
    print("STAGE 2 TRAINING COMPLETE")
    print("="*60)
    print(f"\nCheckpoints saved to: {checkpoint_config.get('save_dir', 'checkpoints/trained')}")
    print("\nNext step: Run evaluation")
    print("  python scripts/run_stage2.py --config <config> --eval-only")


if __name__ == "__main__":
    main()
