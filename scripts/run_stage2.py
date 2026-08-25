#!/usr/bin/env python3
"""Stage 2 multi-task runner (config-driven train + per-checkpoint eval).

The run is selected entirely by its YAML config. The config sets
`data.template` (rationale), `model.use_graph`, and
`checkpointing.save_dir`; this script wires the rationale-augmented data for the
chosen template into the save dir, trains via train_stage2.py, then evaluates
the best-validation checkpoint on each task's own test split.

Rationale-augmented Stage 2 data is read from:
    rationale/format/generated/{task}/{template}/{split}/property_prediction_{task}.json
for split in (train, valid, test).

Checkpoint selection happens inside train_stage2.py and is by validation loss:
each experiment saves a single `final.pt`.

Usage:
    python scripts/run_stage2.py --config configs/stage2.yaml
    python scripts/run_stage2.py --config configs/stage2.yaml --train-only
    python scripts/run_stage2.py --config configs/stage2.yaml --eval-only
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CLASSIFICATION_TASKS = {"bace", "bbbp", "clintox", "hiv", "sider", "tox21"}
REGRESSION_TASKS     = {"esol", "lipo"}

PAPER_TASKS = ["bace", "bbbp", "clintox", "hiv", "sider", "tox21", "esol", "lipo"]

DATA_ROOT = PROJECT_ROOT / "rationale" / "format" / "generated"
EXP_ROOT  = PROJECT_ROOT / "experiments"
DATE_TAG  = datetime.now().strftime("%Y%m%d")


def get_task_type(task):
    if task in CLASSIFICATION_TASKS:
        return "classification"
    if task in REGRESSION_TASKS:
        return "regression"
    raise ValueError(f"Unknown task: {task}")


def link_data_into(save_dir, tasks, template):
    """Symlink each task's train/valid jsons for `template` into
    save_dir/multitask_data/{split}/. Fails loudly if a source is missing."""
    multitask_root = save_dir / "multitask_data"
    train_path = multitask_root / "train"
    valid_path = multitask_root / "valid"
    for split, split_dir in (("train", train_path), ("valid", valid_path)):
        split_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            src = DATA_ROOT / task / template / split / f"property_prediction_{task}.json"
            if not src.exists():
                raise FileNotFoundError(f"Source data missing: {src}")
            dst = split_dir / f"property_prediction_{task}.json"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
    return train_path, valid_path


def setup_experiment(config_path, tasks):
    """Load the ablation config, link data into its save_dir, and write the
    resolved config to save_dir/config.yaml. Returns (save_dir, resolved_config_path)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    template = config["data"]["template"]
    save_dir = (PROJECT_ROOT / config["checkpointing"]["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_path, valid_path = link_data_into(save_dir, tasks, template)

    config["data"]["train_path"] = str(train_path)
    config["data"]["valid_path"] = str(valid_path)
    config["checkpointing"]["save_dir"] = str(save_dir)

    resolved_config_path = save_dir / "config.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return save_dir, resolved_config_path


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(config_path, save_dir):
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_stage2.py"),
        "--config", str(config_path),
    ]

    print(f"\n[TRAIN] {save_dir.name}")
    print(f"  Config: {config_path}")

    return subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_subprocess(save_dir, ckpt_variant=None):
    """Spawn an isolated eval process for one run. evaluate_single writes
    results_final.json into the save dir."""
    ckpt_name = ckpt_variant or "final.pt"
    if not (save_dir / ckpt_name).exists():
        print(f"  No checkpoint {ckpt_name} in {save_dir} — skip eval")
        return False

    print(f"\n[EVAL] {save_dir.name}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_stage2.py"),
        "--eval-single", save_dir.name,
    ]
    if ckpt_variant:
        cmd += ["--ckpt-variant", ckpt_variant]
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    rc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        print(f"  Evaluation FAILED (exit code {rc})")
    return rc == 0


def _best_checkpoint(save_dir, ckpt_variant):
    """Path to the checkpoint to evaluate: final.pt by default, or an
    explicit override filename via ckpt_variant."""
    path = save_dir / (ckpt_variant or "final.pt")
    if not path.exists():
        print(f"Checkpoint not found: {path}"); sys.exit(1)
    return path


def evaluate_single(save_dir_name, tasks, ckpt_variant=None):
    """Load the model, evaluate the best-validation checkpoint on every task's
    test set, and write results_final.json into the save dir."""
    import gc
    import torch
    from src.models.molecular_llm import MolecularLLM
    from src.evaluation.evaluator import Evaluator
    from src.utils.helpers import set_seed, load_config

    save_dir    = EXP_ROOT / save_dir_name
    config_path = save_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}"); sys.exit(1)

    config   = load_config(str(config_path))
    template = config["data"]["template"]
    use_graph = config["model"].get("use_graph", True)

    ckpt_path = _best_checkpoint(save_dir, ckpt_variant)

    print(f"Experiment : {save_dir_name} | template {template} | use_graph {use_graph}")
    print(f"Checkpoint : {ckpt_path.name}")

    set_seed(config.get("hardware", {}).get("seed", 42))

    mc, lc = config["model"], config["lora"]
    model = MolecularLLM(
        llm_name=mc["llm_name"],
        graph_dim=mc["graph_dim"],
        llm_dim=mc["llm_dim"],
        num_query_tokens=mc["num_query_tokens"],
        dropout=mc["dropout"],
        use_lora=True,
        lora_r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        lora_target_modules=lc["target_modules"],
        freeze_graph_encoder=True,
        freeze_projector=mc.get("freeze_projector", False),
        device="cuda",
    )

    if use_graph:
        s1 = torch.load(str(PROJECT_ROOT / config["checkpointing"]["stage1_checkpoint"]), map_location="cuda")
        model.projector.load_state_dict(s1["projector"], strict=False)
        if s1.get("graph_encoder") is not None:
            model.graph_encoder.load_state_dict(s1["graph_encoder"])
        del s1; gc.collect(); torch.cuda.empty_cache()

    system_prompts = PROJECT_ROOT / config["data"]["system_prompts_path"]
    batch_size     = config["training"]["batch_size"]
    device         = config["hardware"]["device"]

    # The graph encoder comes from the Stage 1 checkpoint above; Stage 2 freezes it
    # and never saves it.
    ckpt = torch.load(str(ckpt_path), map_location="cuda")
    model.projector.load_state_dict(ckpt["projector"], strict=False)
    if "lora" in ckpt:
        for name, param in model.llm.named_parameters():
            if name in ckpt["lora"]:
                param.data.copy_(ckpt["lora"][name])
    del ckpt; gc.collect(); torch.cuda.empty_cache()
    model.eval()

    res = {}
    for task in tasks:
        test_dir = DATA_ROOT / task / template / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"test dir missing: {test_dir}")
        evaluator = Evaluator(
            model=model,
            tokenizer=model.tokenizer,
            eval_data_dir=str(test_dir),
            system_prompts_path=str(system_prompts),
            batch_size=batch_size,
            device=device,
            output_dir=str(save_dir),
            use_graph=use_graph,
        )
        res[task] = (evaluator.evaluate_classification(task)
                     if get_task_type(task) == "classification"
                     else evaluator.evaluate_regression(task))

    out = save_dir / "results_final.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  -> {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 multi-task runner (config-driven)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a Stage 2 config (e.g. configs/stage2.yaml). "
                             "Required except with --eval-single, which reads the "
                             "resolved config from the run's save_dir.")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--eval-single", type=str, default=None,
                        help="Evaluate an existing run by its save_dir name (under experiments/).")
    parser.add_argument("--ckpt-variant", type=str, default=None,
                        help="If set, eval this checkpoint filename instead of "
                             "the default final.pt.")
    args = parser.parse_args()

    tasks = PAPER_TASKS

    if args.eval_single:
        evaluate_single(args.eval_single, tasks, args.ckpt_variant)
        return

    if not args.config:
        parser.error("--config is required unless --eval-single is given")

    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    for t in tasks:
        if not (DATA_ROOT / t).exists():
            print(f"Data dir missing: {DATA_ROOT / t}")
            sys.exit(1)

    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    template = base_config["data"]["template"]
    use_graph = base_config["model"].get("use_graph", True)
    save_dir = (PROJECT_ROOT / base_config["checkpointing"]["save_dir"]).resolve()

    mode = "train-only" if args.train_only else "eval-only" if args.eval_only else "train+eval"

    print(f"{'='*60}")
    print(f"Stage 2 Runner")
    print(f"Config     : {config_path}")
    print(f"Save dir   : {save_dir}")
    print(f"Tasks      : {tasks}")
    print(f"Template   : {template}")
    print(f"use_graph  : {use_graph}")
    print(f"Date       : {DATE_TAG}")
    print(f"Mode       : {mode}")
    print(f"{'='*60}")

    train_failed = False

    if not args.eval_only:
        save_dir, resolved_config_path = setup_experiment(config_path, tasks)
        rc = train(resolved_config_path, save_dir)
        if rc != 0:
            print(f"  [FAIL] training (exit code {rc})")
            train_failed = True

    if args.train_only:
        print("\nTraining complete. Run with --eval-only to evaluate.")
        return

    if train_failed:
        print("  skip eval: training failed")
        return

    # --eval-only against a checkpoint that was downloaded rather than trained
    # here: the run directory has no resolved config yet, so write one from the
    # config we were given.
    if args.eval_only and not (save_dir / "config.yaml").exists():
        if not save_dir.exists():
            print(f"Nothing to evaluate: {save_dir} does not exist")
            sys.exit(1)
        resolved = dict(base_config)
        resolved["checkpointing"] = {**base_config["checkpointing"],
                                     "save_dir": str(save_dir)}
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(resolved, f, default_flow_style=False)
        print(f"  wrote {save_dir.name}/config.yaml from {config_path.name}")

    if not evaluate_subprocess(save_dir, args.ckpt_variant):
        print("\nEvaluation did not complete; no results were written.")
        sys.exit(1)
    print(f"\nEval done. Results: {save_dir.name}/results_final.json")


if __name__ == "__main__":
    main()
