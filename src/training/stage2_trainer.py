"""Stage 2 Trainer: Multi-task Instruction Tuning."""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Optional, List, Any
from tqdm import tqdm
import numpy as np

from .losses import Stage2Loss, compute_per_sample_nll


class Stage2Trainer:
    """
    Trainer for Stage 2: Multi-task Instruction Tuning.

    Trains:
    - Query-based Fusion Projector
    - LoRA adapters on LLM

    Uses CrossEntropy loss for both training and
    validation/checkpoint selection.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
        projector_lr: Optional[float] = None,
        lora_lr: Optional[float] = None,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints/trained",
        task_tag: str = "all",
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.task_tag = task_tag
        self.device = device

        os.makedirs(save_dir, exist_ok=True)
        self.loss_fn = Stage2Loss()

        # Build param groups (projector / lora) with per-group LRs. A group is
        # active only if its module is unfrozen / enabled; its LR is required
        # only when active.
        param_groups = self._build_param_groups(
            projector_lr=projector_lr,
            lora_lr=lora_lr,
        )
        if not param_groups:
            raise ValueError(
                "No trainable parameters found. Check freeze settings in config — "
                "at least one module must be unfrozen."
            )
        self.group_names = [g["name"] for g in param_groups]
        self.group_base_lrs = [g["lr"] for g in param_groups]
        self.trainable_params = [p for g in param_groups for p in g["params"]]

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        min_lr_ratio = 0.1

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return 0.1 + 0.9 * (current_step / max(1, warmup_steps))
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=[lr_lambda] * len(param_groups),
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_valid_loss = float('inf')

    def _build_param_groups(
        self,
        projector_lr: Optional[float],
        lora_lr: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Build AdamW param groups based on model freeze config.

        Stage 2 design:
          - graph_encoder: frozen — it is trained in Stage 1 and never updated here.
          - projector (Qformer + query_tokens + ln_graph + llm_proj): pretrained
            from Stage 1, small LR.
          - LoRA: random-init at Stage 2 start, higher LR.
        """
        groups: List[Dict[str, Any]] = []

        if not self.model.freeze_projector:
            params = [p for p in self.model.projector.parameters() if p.requires_grad]
            if params:
                if projector_lr is None:
                    raise TypeError(
                        "projector_lr is required when freeze_projector=False."
                    )
                groups.append({"name": "projector", "params": params, "lr": projector_lr})

        if self.model.use_lora:
            params = [
                p for n, p in self.model.llm.named_parameters()
                if "lora" in n.lower() and p.requires_grad
            ]
            if params:
                if lora_lr is None:
                    raise TypeError("lora_lr is required when use_lora=True.")
                groups.append({"name": "lora", "params": params, "lr": lora_lr})

        return groups

    def train(self, start_epoch: int = 0):
        """Run full training loop."""
        print(f"Starting Stage 2 training for {self.num_epochs} epochs...")
        print(f"Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*50}")

            self.current_epoch = epoch
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch+1} summary — "
                  f"Loss: {train_metrics['total']:.4f} | "
                  f"Time: {epoch_time/60:.1f}m")

            if self.valid_dataloader:
                valid_loss = self.validate()
                print(f"\nValidation Loss (sample-balanced): {valid_loss:.4f}")

                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_checkpoint("final.pt")
                    print("  → New best model saved!")
            else:
                # No validation — save every epoch
                self.save_checkpoint(f"stage2_{self.task_tag}_epoch{epoch+1}.pt")

        print("\nStage 2 training complete!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        task_losses = {}

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            graphs = batch.get("graphs")
            task_types = batch.get("task_types", ["unknown"] * input_ids.shape[0])

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graphs=graphs,
            )

            # Compute loss
            loss_dict = self.loss_fn(
                lm_logits=outputs["logits"],
                labels=labels,
            )

            loss = loss_dict["total"] / self.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Tracking
            total_loss += loss_dict["total"].item()
            num_batches += 1

            # Track per-task losses
            for task in set(task_types):
                if task not in task_losses:
                    task_losses[task] = []
                task_losses[task].append(loss_dict["total"].item())

            # Update progress bar (show primary group's LR)
            progress_bar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        avg_task_losses = {task: np.mean(losses) for task, losses in task_losses.items()}
        return {"total": total_loss / num_batches, **avg_task_losses}

    @torch.no_grad()
    def validate(self) -> float:
        """
        Sample-balanced validation NTP loss: the mean of the per-sample mean NLLs.
        """
        self.model.eval()

        per_sample_losses: List[float] = []

        for batch in tqdm(self.valid_dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            graphs = batch.get("graphs")

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graphs=graphs,
            )

            sample_nll = compute_per_sample_nll(outputs["logits"], labels)  # [B]
            per_sample_losses.extend(sample_nll.cpu().tolist())

        if not per_sample_losses:
            raise RuntimeError("validation dataloader produced no batches")

        return float(np.mean(per_sample_losses))

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.save_dir, filename)
        checkpoint = {
            "projector": self.model.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_valid_loss": self.best_valid_loss,
        }

        # Save LoRA weights separately
        if self.model.use_lora:
            lora_state = {}
            for name, param in self.model.llm.named_parameters():
                if "lora" in name.lower():
                    lora_state[name] = param.data
            checkpoint["lora"] = lora_state

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns the next epoch to start from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.projector.load_state_dict(ckpt["projector"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.best_valid_loss = ckpt["best_valid_loss"]
        self.current_epoch = ckpt.get("current_epoch", 0)

        if "lora" in ckpt and self.model.use_lora:
            for name, param in self.model.llm.named_parameters():
                if name in ckpt["lora"]:
                    param.data = ckpt["lora"][name]

        start_epoch = self.current_epoch + 1
        print(f"Checkpoint loaded: {path} (resuming from epoch {start_epoch})")
        return start_epoch
