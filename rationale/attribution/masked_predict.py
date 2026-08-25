"""Build prediction jobs from decompositions and run batch masked inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from config import SEEDS, TASK_CONFIG
from utils import smiles_to_pyg_data, load_model, predict_with_mask, prepare_batch
from decompose import MolDecomposition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction job
# ---------------------------------------------------------------------------

@dataclass
class PredictionJob:
    smiles: str
    mask_atoms: list[int]   # atoms to EXCLUDE from pooling
    num_atoms: int
    sub_type: str           # "mol" | "fg" | "murcko" | "brics"
    sub_name: str           # e.g. "full_molecule", "Hydroxyl", "scaffold"
    sub_idx: int            # index within sub_type for this molecule


def build_prediction_jobs(decompositions: list[MolDecomposition]) -> list[PredictionJob]:
    """Create prediction jobs: 1 full-molecule + 1 per substructure (masked out).

    For the full-molecule job, mask_atoms is empty (nothing masked).
    For substructure jobs, mask_atoms contains the substructure's atom indices.
    """
    jobs = []
    for dec in decompositions:
        if dec.error:
            continue

        # Full molecule (no masking)
        jobs.append(PredictionJob(
            smiles=dec.smiles,
            mask_atoms=[],
            num_atoms=dec.num_atoms,
            sub_type="mol",
            sub_name="full_molecule",
            sub_idx=0,
        ))

        # Functional groups
        for i, fg in enumerate(dec.fg):
            jobs.append(PredictionJob(
                smiles=dec.smiles,
                mask_atoms=fg["atoms"],
                num_atoms=dec.num_atoms,
                sub_type="fg",
                sub_name=fg["name"],
                sub_idx=i,
            ))

        # Murcko
        for i, mu in enumerate(dec.murcko):
            jobs.append(PredictionJob(
                smiles=dec.smiles,
                mask_atoms=mu["atoms"],
                num_atoms=dec.num_atoms,
                sub_type="murcko",
                sub_name=mu["name"],
                sub_idx=i,
            ))

        # BRICS
        for i, br in enumerate(dec.brics):
            jobs.append(PredictionJob(
                smiles=dec.smiles,
                mask_atoms=br["atoms"],
                num_atoms=dec.num_atoms,
                sub_type="brics",
                sub_name=br["name"],
                sub_idx=i,
            ))

    return jobs


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

def _job_to_data_and_mask(job: PredictionJob):
    """Convert a PredictionJob to (PyG Data, mask tensor)."""
    data = smiles_to_pyg_data(job.smiles)
    if data is None:
        return None, None
    mask = torch.ones(data.x.size(0), dtype=torch.float)
    for idx in job.mask_atoms:
        if idx < mask.size(0):
            mask[idx] = 0.0
    return data, mask


def run_seed(
    jobs: list[PredictionJob],
    task: str,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
    ckpt_mode: str = "val_best",
) -> np.ndarray:
    """Run all prediction jobs for a single model seed.

    Returns:
        np.ndarray of shape [len(jobs), num_tasks] with raw predictions.
    """
    model = load_model(task, seed, device, mode=ckpt_mode)
    num_tasks = TASK_CONFIG[task]["num_tasks"]
    all_preds = np.full((len(jobs), num_tasks), np.nan, dtype=np.float32)

    # Pre-convert all jobs to (data, mask) pairs
    pairs = [_job_to_data_and_mask(j) for j in jobs]

    for start in range(0, len(jobs), batch_size):
        end = min(start + batch_size, len(jobs))
        batch_data = []
        batch_masks = []
        valid_indices = []

        for i in range(start, end):
            data, mask = pairs[i]
            if data is not None:
                batch_data.append(data)
                batch_masks.append(mask)
                valid_indices.append(i)

        if not batch_data:
            continue

        batch = prepare_batch(batch_data, batch_masks)
        preds = predict_with_mask(model, batch, device)
        for local_i, global_i in enumerate(valid_indices):
            all_preds[global_i] = preds[local_i]

    del model
    torch.cuda.empty_cache()
    return all_preds


def run_all_seeds(
    jobs: list[PredictionJob],
    task: str,
    device: torch.device,
    batch_size: int = 256,
    seeds: list[int] = None,
    ckpt_mode: str = "val_best",
) -> pd.DataFrame:
    """Run predictions across all seeds and return a DataFrame.

    Columns: smiles, sub_type, sub_name, sub_idx, seed, pred_0, pred_1, ...
    """
    num_tasks = TASK_CONFIG[task]["num_tasks"]
    pred_cols = [f"pred_{t}" for t in range(num_tasks)]
    if seeds is None:
        seeds = SEEDS

    rows = []
    for seed in seeds:
        logger.info("Running seed %d for task %s (ckpt_mode=%s)", seed, task, ckpt_mode)
        preds = run_seed(jobs, task, seed, device, batch_size, ckpt_mode=ckpt_mode)
        for i, job in enumerate(jobs):
            row = {
                "smiles": job.smiles,
                "sub_type": job.sub_type,
                "sub_name": job.sub_name,
                "sub_idx": job.sub_idx,
                "seed": seed,
            }
            for t in range(num_tasks):
                row[pred_cols[t]] = preds[i, t]
            rows.append(row)

    return pd.DataFrame(rows)
