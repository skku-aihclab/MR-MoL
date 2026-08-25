"""Per-(task, seed) Mole-BERT GIN finetuning.

Trains a Mole-BERT GIN per task and selects the
checkpoint by best validation metric (avg ROC-AUC for classification, avg RMSE
for regression). Molecules and their train/valid/test membership come from this
project's per-task ``raw.csv`` / ``split.csv`` under DATA_DIR.
The saved checkpoint is the validation-best one (seed_{s}.pth).
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch_geometric.data import Data, DataLoader

from config import (
    DATA_DIR, DROP_RATIO, EMB_DIM, FINETUNE_DEFAULTS, GNN_TYPE, GRAPH_POOLING,
    JK, MOLEBERT_DIR, NUM_LAYER, PRETRAIN_GIN_CKPT, RESULT_DIR, TASK_CONFIG,
)

# Mole-BERT internals
sys.path.insert(0, MOLEBERT_DIR)
from loader import mol_to_graph_data_obj_simple  # noqa: E402
from model import GNN_graphpred                  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: DATA_DIR/{task}/raw.csv + split.csv -> Mole-BERT graphs
# ---------------------------------------------------------------------------

SPLIT_NAMES = ("train", "valid", "test")


def _load_splits(task: str) -> dict[str, list[Data]]:
    """Featurize a task's molecules and group them by their split.csv membership.

    `raw.csv` holds canonical SMILES and the label columns; `split.csv` assigns
    each of those SMILES to train/valid/test. Labels follow the Mole-BERT
    convention for classification -- +1 positive, -1 negative, 0 for a missing
    label, which is what the loss mask and the ROC-AUC eval below expect.
    """
    cfg = TASK_CONFIG[task]
    task_dir = os.path.join(DATA_DIR, task)
    raw_df = pd.read_csv(os.path.join(task_dir, "raw.csv"))
    split_df = pd.read_csv(os.path.join(task_dir, "split.csv"))

    unknown = set(split_df["split"]) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"{task_dir}/split.csv: unexpected split labels {sorted(unknown)}")
    split_of = dict(zip(split_df["smiles"], split_df["split"]))

    labels = raw_df[cfg["label_cols"]]
    if cfg["task_type"] == "classification":
        labels = labels.replace(0, -1).fillna(0)
    y = torch.tensor(labels.values, dtype=torch.float32)

    splits: dict[str, list[Data]] = {name: [] for name in SPLIT_NAMES}
    n_skipped = 0
    for i, smi in enumerate(raw_df["smiles"]):
        split = split_of.get(smi)
        if split is None:
            n_skipped += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        data = mol_to_graph_data_obj_simple(mol) if mol is not None else None
        if data is None:
            n_skipped += 1
            continue
        data.y = y[i]
        splits[split].append(data)

    if n_skipped:
        logger.warning("[%s] skipped %d/%d molecules (unsplit or unfeaturizable)",
                       task, n_skipped, len(raw_df))
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise ValueError(f"{task_dir}: split {name!r} is empty")
    logger.info("[%s] train/valid/test = %d/%d/%d", task,
                *(len(splits[name]) for name in SPLIT_NAMES))
    return splits


# ---------------------------------------------------------------------------
# Eval (multi-target ROC-AUC for cls; RMSE for reg)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, loader, device, is_cls: bool) -> tuple[float, list[float]]:
    """Returns (avg_metric, per_target_metrics)."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.view(pred.shape).cpu())
        y_pred.append(pred.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    per_target = []
    if is_cls:
        for t in range(y_true.shape[1]):
            valid = (y_true[:, t] ** 2) > 0
            if valid.sum() == 0:
                per_target.append(float("nan"))
                continue
            yt = (y_true[valid, t] + 1) / 2  # {-1,1} → {0,1}
            yp = y_pred[valid, t]
            if len(set(yt)) < 2:
                per_target.append(float("nan"))
                continue
            per_target.append(float(roc_auc_score(yt, yp)))
        valids = [v for v in per_target if not np.isnan(v)]
        avg = float(np.mean(valids)) if valids else float("nan")
    else:
        for t in range(y_true.shape[1]):
            yt = y_true[:, t]
            yp = y_pred[:, t]
            per_target.append(float(np.sqrt(mean_squared_error(yt, yp))))
        avg = float(np.mean(per_target))
    model.train()
    return avg, per_target


# ---------------------------------------------------------------------------
# Train one (task, seed)
# ---------------------------------------------------------------------------

def train_one(
    task: str,
    seed: int,
    device: str = "cuda:0",
    epochs: int | None = None,
    batch_size: int | None = None,
    pretrain_ckpt: str | None = None,
) -> dict:
    cfg = TASK_CONFIG[task]
    is_cls = cfg["task_type"] == "classification"
    fd = FINETUNE_DEFAULTS
    epochs = epochs or fd["epochs"]
    batch_size = batch_size or fd["batch_size"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits = _load_splits(task)
    train_ds, valid_ds, test_ds = (splits[name] for name in SPLIT_NAMES)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = GNN_graphpred(
        num_layer=NUM_LAYER, emb_dim=EMB_DIM, num_tasks=cfg["num_tasks"],
        JK=JK, drop_ratio=fd["drop_ratio"],
        graph_pooling=GRAPH_POOLING, gnn_type=GNN_TYPE,
    )
    pretrain = pretrain_ckpt or PRETRAIN_GIN_CKPT
    if pretrain and Path(pretrain).exists():
        model.from_pretrained(pretrain)
    else:
        logger.warning("Pretrain ckpt not found at %s — training from scratch", pretrain)
    model.to(device)

    param_groups = [
        {"params": model.gnn.parameters()},
        {"params": model.graph_pred_linear.parameters(), "lr": fd["lr"] * fd["lr_scale"]},
    ]
    optimizer = optim.Adam(param_groups, lr=fd["lr"], weight_decay=fd["weight_decay"])

    cls_criterion = nn.BCEWithLogitsLoss(reduction="none")
    reg_criterion = nn.MSELoss(reduction="none")

    ckpt_dir = Path(RESULT_DIR) / task
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_best_path = ckpt_dir / f"seed_{seed}.pth"

    best_val = -np.inf if is_cls else np.inf
    best_val_epoch = -1
    best_val_test = float("nan")
    best_val_test_per_target: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            optimizer.zero_grad()
            if is_cls:
                is_valid = (y ** 2) > 0
                loss_mat = cls_criterion(pred.double(), (y + 1) / 2)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
                loss = torch.sum(loss_mat) / torch.sum(is_valid).clamp(min=1)
            else:
                loss = reg_criterion(pred.double(), y).mean()
            loss.backward()
            optimizer.step()

        val_avg, _ = _evaluate(model, valid_loader, device, is_cls)
        test_avg, test_per_t = _evaluate(model, test_loader, device, is_cls)
        logger.info(
            "[%s seed=%d] epoch %d/%d  val=%.4f  test=%.4f",
            task, seed, epoch, epochs, val_avg, test_avg,
        )

        is_better_val = (val_avg > best_val) if is_cls else (val_avg < best_val)
        if is_better_val:
            best_val = val_avg
            best_val_epoch = epoch
            best_val_test = test_avg
            best_val_test_per_target = list(test_per_t)
            torch.save(model.state_dict(), str(val_best_path))

    logger.info(
        "[%s seed=%d] done — val_best epoch=%d val=%.4f test=%.4f",
        task, seed, best_val_epoch, best_val, best_val_test,
    )

    return {
        "task": task, "task_type": cfg["task_type"], "seed": seed,
        "n_train": len(train_ds), "n_valid": len(valid_ds), "n_test": len(test_ds),
        "best_by_val": {
            "epoch": int(best_val_epoch),
            "val_avg": float(best_val), "test_avg": float(best_val_test),
            "test_per_target": best_val_test_per_target,
            "ckpt": str(val_best_path),
        },
    }
