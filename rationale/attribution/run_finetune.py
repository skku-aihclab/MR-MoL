#!/usr/bin/env python3
"""Orchestrate Mole-BERT finetuning across (task, seed).

Usage:
    python run_finetune.py --task all --device 0
    python run_finetune.py --task bbbp --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import RESULT_DIR, SEEDS, TASK_CONFIG  # noqa: E402
from finetune import train_one  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mole-BERT finetune orchestrator")
    parser.add_argument("--task", required=True, help="Task name or 'all'")
    parser.add_argument("--device", type=int, default=0, help="CUDA device idx (-1 = CPU)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (task, seed) pairs whose val_best ckpt already exists")
    args = parser.parse_args()

    import torch
    if args.device >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    tasks = list(TASK_CONFIG.keys()) if args.task == "all" else [args.task]
    for t in tasks:
        if t not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {t}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for task in tasks:
        for seed in args.seeds:
            ckpt = Path(RESULT_DIR) / task / f"seed_{seed}.pth"
            tag = f"{task}/seed_{seed}"
            if args.skip_existing and ckpt.exists():
                logger.info("[skip] %s (ckpt exists)", tag)
                continue
            logger.info("[train] %s", tag)
            t1 = time.time()
            try:
                res = train_one(
                    task, seed, device=device,
                    epochs=args.epochs, batch_size=args.batch_size,
                )
                wall_seconds = time.time() - t1
                n_done += 1
                bv = res["best_by_val"]
                logger.info(
                    "[done] %s  val_best ep%d (val=%.4f, test=%.4f)  (%.0fs)",
                    tag, bv["epoch"], bv["val_avg"], bv["test_avg"], wall_seconds,
                )
            except Exception as e:
                logger.exception("[FAIL] %s: %s", tag, e)

    total_elapsed = time.time() - t0
    logger.info("This run: %d entries, %.0fs total", n_done, total_elapsed)


if __name__ == "__main__":
    main()
