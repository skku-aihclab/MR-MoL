#!/usr/bin/env python3
"""End-to-end SME attribution pipeline.

For each task: load molecules, decompose into functional groups / Murcko
scaffold parts / BRICS fragments, run masked inference with the val-best
Mole-BERT checkpoints, and save per-substructure attributions. Rationale text
is then assembled by ``rationale/format/generate_rationale.py``.

Usage:
    python run_pipeline.py --task bbbp --device 0
    python run_pipeline.py --task all --device 0
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time

import pandas as pd
import torch
from rdkit import Chem

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_DIR, OUTPUT_DIR, SEEDS, TASK_CONFIG  # noqa: E402
from decompose import decompose_molecule  # noqa: E402
from masked_predict import build_prediction_jobs, run_all_seeds  # noqa: E402
from attribution import compute_attributions  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_smiles(task: str) -> list[str]:
    """Load all canonical SMILES for a task from raw.csv."""
    raw_path = os.path.join(DATA_DIR, task, "raw.csv")
    df = pd.read_csv(raw_path)
    smiles_list = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))
    logger.info("Loaded %d valid molecules for task %s", len(smiles_list), task)
    return smiles_list


def run_task(task: str, device: torch.device, batch_size: int,
             ckpt_mode: str = "val_best", out_root: str = None, seeds: list = None):
    """Full attribution pipeline for a single task."""
    t0 = time.time()
    if out_root is None:
        out_root = OUTPUT_DIR
    task_out = os.path.join(out_root, task)
    os.makedirs(task_out, exist_ok=True)

    # --- Step 1: Load molecules ---
    smiles_list = load_smiles(task)

    # --- Step 2: Decompose ---
    logger.info("Decomposing %d molecules...", len(smiles_list))
    decompositions = [decompose_molecule(smi) for smi in smiles_list]
    n_ok = sum(1 for d in decompositions if d.error is None)
    n_fg = sum(len(d.fg) for d in decompositions)
    n_mu = sum(len(d.murcko) for d in decompositions)
    n_br = sum(len(d.brics) for d in decompositions)
    logger.info(
        "Decomposition done: %d/%d ok, %d FGs, %d Murcko parts, %d BRICS fragments",
        n_ok, len(decompositions), n_fg, n_mu, n_br,
    )

    # Save decompositions
    with open(os.path.join(task_out, "decompositions.pkl"), "wb") as f:
        pickle.dump(decompositions, f)

    # --- Step 3: Build prediction jobs ---
    jobs = build_prediction_jobs(decompositions)
    logger.info("Built %d prediction jobs", len(jobs))

    # --- Step 4: Run masked predictions across all seeds ---
    logger.info("Running masked predictions (%d seeds, ckpt_mode=%s)...",
                len(seeds) if seeds else len(SEEDS), ckpt_mode)
    predictions_df = run_all_seeds(jobs, task, device, batch_size,
                                   seeds=seeds, ckpt_mode=ckpt_mode)
    predictions_df.to_parquet(os.path.join(task_out, "predictions.parquet"), index=False)
    logger.info("Predictions saved: %d rows", len(predictions_df))

    # --- Step 5: Compute attributions ---
    logger.info("Computing attributions...")
    # Use canonical SMILES consistently
    attributions_df = compute_attributions(predictions_df, task)
    attributions_df.to_parquet(os.path.join(task_out, "attributions.parquet"), index=False)
    logger.info("Attributions saved: %d rows", len(attributions_df))

    elapsed = time.time() - t0
    logger.info("Task %s completed in %.1f seconds", task, elapsed)


def main():
    parser = argparse.ArgumentParser(description="SME Attribution Pipeline")
    parser.add_argument("--task", required=True, help="Task name or 'all'")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ckpt-mode", default="val_best", choices=["val_best"],
                        help="Checkpoint selection: val-best Mole-BERT (seed_{s}.pth).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"Subset of seeds (default: {SEEDS}).")
    parser.add_argument("--output-dir", default=None,
                        help=f"Override output root (default: {OUTPUT_DIR}).")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    out_root = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    logger.info("Output root: %s  (ckpt_mode=%s)", out_root, args.ckpt_mode)

    tasks = list(TASK_CONFIG.keys()) if args.task == "all" else [args.task]
    for task in tasks:
        if task not in TASK_CONFIG:
            logger.error("Unknown task: %s", task)
            continue
        run_task(task, device, args.batch_size,
                 ckpt_mode=args.ckpt_mode, out_root=out_root, seeds=args.seeds)


if __name__ == "__main__":
    main()
