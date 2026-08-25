"""Stage 1 Trainer: Graph-Text Alignment."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict
from tqdm import tqdm

from .losses import Stage1Loss


class Stage1Trainer:
    """
    Trainer for Stage 1: Graph-Language Alignment.

    Trains:
    - Query Projector (learnable query tokens + cross-attention)
    - Graph Encoder (optionally)

    With loss:
    - Next-token-prediction loss on the target description
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        *,
        learning_rate: float,
        qformer_lr: float,
        graph_encoder_lr: float,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints/trained",
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.device = device

        os.makedirs(save_dir, exist_ok=True)

        self.loss_fn = Stage1Loss()

        param_groups, flat_params = self._build_param_groups(
            llm_proj_lr=learning_rate,
            qformer_lr=qformer_lr,
            graph_encoder_lr=graph_encoder_lr,
            weight_decay=weight_decay,
        )
        self.trainable_params = flat_params

        if len(self.trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found. Check freeze settings in config."
            )

        self.optimizer = AdamW(param_groups)

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

    def _build_param_groups(
        self,
        *,
        llm_proj_lr: float,
        qformer_lr: float,
        graph_encoder_lr: float,
        weight_decay: float,
    ):
        """Split trainable parameters into 3 groups with different LRs.

        Group 1: projector.llm_proj (random-init, highest LR)
        Group 2: projector.Qformer + projector.query_tokens + projector.ln_graph (pretrained, mid LR)
        Group 3: graph_encoder (MolCA-pretrained GIN, lowest LR)
        """
        llm_proj_params = []
        qformer_params = []
        graph_params = []
        flat = []

        if not self.model.freeze_projector:
            for name, p in self.model.projector.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("llm_proj."):
                    llm_proj_params.append(p)
                else:
                    qformer_params.append(p)
                flat.append(p)

        if not self.model.freeze_graph_encoder:
            for p in self.model.graph_encoder.parameters():
                if p.requires_grad:
                    graph_params.append(p)
                    flat.append(p)

        groups = []
        if llm_proj_params:
            groups.append({"params": llm_proj_params, "lr": llm_proj_lr, "weight_decay": weight_decay, "name": "llm_proj"})
        if qformer_params:
            groups.append({"params": qformer_params, "lr": qformer_lr, "weight_decay": weight_decay, "name": "qformer"})
        if graph_params:
            groups.append({"params": graph_params, "lr": graph_encoder_lr, "weight_decay": weight_decay, "name": "graph_encoder"})

        for g in groups:
            print(
                f"[Stage1Trainer] param group '{g['name']}': "
                f"{sum(p.numel() for p in g['params']):,} params, lr={g['lr']}"
            )
        return groups, flat

    def train(self, start_epoch: int = 0):
        """Run full training loop."""
        print(f"Starting Stage 1 training for {self.num_epochs} epochs...")
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

            self.save_checkpoint(f"stage1_epoch{epoch+1}.pt")
            print(f"  -> Checkpoint saved: stage1_epoch{epoch+1}.pt")

        # Save final model
        self.save_checkpoint("stage1_final.pt")
        print("\nStage 1 training complete!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if not self.model.freeze_projector:
            self.model.projector.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
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

            loss_dict = self.loss_fn(
                lm_logits=outputs["logits"],
                labels=labels,
            )

            loss = loss_dict["total"] / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss_dict["total"].item()
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return {"total": total_loss / num_batches}

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "projector": self.model.projector.state_dict(),
            "graph_encoder": self.model.graph_encoder.state_dict() if not self.model.freeze_graph_encoder else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns the next epoch to start from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.projector.load_state_dict(ckpt["projector"])
        if ckpt.get("graph_encoder") and not self.model.freeze_graph_encoder:
            self.model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.current_epoch = ckpt.get("current_epoch", 0)
        start_epoch = self.current_epoch + 1
        print(f"Checkpoint loaded: {path} (resuming from epoch {start_epoch})")
        return start_epoch
