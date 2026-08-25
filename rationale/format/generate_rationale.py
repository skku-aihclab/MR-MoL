"""Generate stage2 JSON files with rationale from templates + SME attributions.

CLI:
    python generate_rationale.py --task bbbp --template rationale
    python generate_rationale.py --task bbbp --all-templates
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------------
# Paths (package-relative; override with environment variables)
# ---------------------------------------------------------------------------

RATIONALE_DIR = Path(__file__).resolve().parent          # rationale/format/
TEMPLATES_DIR = RATIONALE_DIR / "templates"
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", RATIONALE_DIR / "generated"))
# Base property-prediction instruction data (rationale is injected into these).
STAGE2_DIR = Path(os.environ.get("STAGE2_DIR", RATIONALE_DIR.parents[1] / "data_prep" / "stage2"))
# SME attribution outputs (one subdir per task); also overridable via --attribution-dir.
ATTRIBUTION_OUTPUT_DIR = Path(os.environ.get("ATTRIBUTION_OUTPUT_DIR", RATIONALE_DIR.parent / "attribution" / "output"))
# SME module dir, needed to unpickle decompositions.pkl (decompose.MolDecomposition).
ATTRIBUTION_MODULE_DIR = Path(os.environ.get("ATTRIBUTION_MODULE_DIR", RATIONALE_DIR.parent / "attribution"))


def _stage2_filename(task: str) -> str:
    return f"property_prediction_{task}.json"


def _attribution_dir_for_task(task: str) -> Path:
    return ATTRIBUTION_OUTPUT_DIR / task

# ---------------------------------------------------------------------------
# Import format_registry (same directory) + SME module (for unpickling)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(RATIONALE_DIR))
sys.path.insert(0, str(ATTRIBUTION_MODULE_DIR))
from format_registry import format_rationale_from_template  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task config (mirrored from SME pipeline — only what we need here)
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "bace":    {"num_tasks": 1,  "task_type": "classification", "label_cols": ["Class"]},
    "bbbp":    {"num_tasks": 1,  "task_type": "classification", "label_cols": ["p_np"]},
    "clintox": {"num_tasks": 2,  "task_type": "classification", "label_cols": ["FDA_APPROVED", "CT_TOX"]},
    "hiv":     {"num_tasks": 1,  "task_type": "classification", "label_cols": ["HIV_active"]},
    "sider":   {"num_tasks": 27, "task_type": "classification", "label_cols": [
        "Hepatobiliary disorders", "Metabolism and nutrition disorders",
        "Product issues", "Eye disorders", "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders", "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders", "Surgical and medical procedures",
        "Vascular disorders", "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders", "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders", "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]},
    "tox21":   {"num_tasks": 12, "task_type": "classification", "label_cols": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]},
    "esol":    {"num_tasks": 1,  "task_type": "regression", "label_cols": ["measured log solubility in mols per litre"]},
    "lipo":    {"num_tasks": 1,  "task_type": "regression", "label_cols": ["exp"]},
}

SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _build_instruction_index() -> dict[str, dict[str, int]]:
    """Build {task: {normalized_instruction: task_idx}} from property_prediction.json."""
    path = STAGE2_DIR / "instruction_templates" / "property_prediction.json"
    if not path.exists():
        return {}
    with open(path) as f:
        templates = json.load(f)
    index: dict[str, dict[str, int]] = {}
    for task, label_map in templates.items():
        if not isinstance(label_map, dict):
            continue
        task_index: dict[str, int] = {}
        for task_idx, (label, items) in enumerate(label_map.items()):
            for item in items:
                raw = item.get("input", "")
                normalized = raw.replace("<INPUT>", "").replace("\n", " ").strip()
                task_index[normalized] = task_idx
        index[task] = task_index
    return index


_INSTRUCTION_INDEX: dict[str, dict[str, int]] = _build_instruction_index()


def _match_subtask(instruction: str, task: str, cfg: dict) -> int | None:
    if cfg["num_tasks"] == 1:
        return 0
    instruction = instruction.strip()
    instruction_lower = instruction.lower()
    task_index = _INSTRUCTION_INDEX.get(task, {})
    for key, idx in task_index.items():
        if instruction_lower.startswith(key.lower()):
            return idx
    return None


def build_decomposition_map(decomposition) -> dict[str, dict[int, dict]]:
    """Build a lookup from a MolDecomposition for format_rationale."""
    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for i, fg in enumerate(decomposition.fg):
        result["fg"][i] = fg
    for i, mu in enumerate(decomposition.murcko):
        result["murcko"][i] = mu
    for i, br in enumerate(decomposition.brics):
        result["brics"][i] = br
    return result


def load_template(name: str) -> dict:
    path = TEMPLATES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path) as f:
        return json.load(f)


def list_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_for_template(
    task: str,
    template_name: str,
    template: dict,
    attributions_df: pd.DataFrame,
    decompositions: list,
    output_dir: Path,
) -> dict[str, int]:
    """Generate stage2 JSON with rationale for all splits.

    Returns:
        {split: matched_count}
    """
    cfg = TASK_CONFIG[task]

    # Build canonical SMILES -> decomposition lookup
    dec_by_smi = {}
    for dec in decompositions:
        can = _canonical(dec.smiles)
        if can:
            dec_by_smi[can] = dec

    results = {}

    filename = _stage2_filename(task)
    for split in SPLITS:
        json_path = STAGE2_DIR / split / filename
        if not json_path.exists():
            logger.warning("Stage2 JSON not found: %s", json_path)
            continue

        with open(json_path) as f:
            entries = json.load(f)

        # Rationale cache: (canonical_smiles, task_idx) -> rationale text
        rationale_cache: dict[tuple, str] = {}
        matched = 0

        for entry in entries:
            input_smi = entry.get("INPUT", "")
            can_smi = _canonical(input_smi)
            if can_smi is None:
                entry["rationale"] = ""
                continue

            task_idx = _match_subtask(entry.get("instruction", ""), task, cfg)
            if task_idx is None:
                entry["rationale"] = ""
                continue

            cache_key = (can_smi, task_idx)
            if cache_key not in rationale_cache:
                mask = (
                    (attributions_df["smiles"] == can_smi)
                    & (attributions_df["task_idx"] == task_idx)
                )
                attrs_subset = attributions_df[mask]

                if attrs_subset.empty:
                    rationale_cache[cache_key] = ""
                else:
                    dec = dec_by_smi.get(can_smi)
                    dec_map = build_decomposition_map(dec) if dec else {}
                    rationale_cache[cache_key] = format_rationale_from_template(
                        attrs_subset, dec_map, template, task,
                    )

            entry["rationale"] = rationale_cache[cache_key]
            if entry["rationale"]:
                matched += 1

        # Save
        out_dir = output_dir / task / template_name / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] %s/%s: %d/%d entries with rationale → %s",
            template_name, task, split, matched, len(entries), out_path,
        )
        results[split] = matched

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rationale-augmented stage2 JSON")
    parser.add_argument("--task", required=True, help="Task name (bbbp, clintox, ...) or 'all'")
    parser.add_argument("--template", default=None, help="Template name (rationale)")
    parser.add_argument("--all-templates", action="store_true", help="Run all templates")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output directory (default: {GENERATED_DIR})")
    parser.add_argument("--attribution-dir", default=None,
                        help="Attribution output root; each task is a subdirectory under it "
                             f"(default: {ATTRIBUTION_OUTPUT_DIR}).")
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else GENERATED_DIR
    attribution_dir_override = Path(args.attribution_dir) if args.attribution_dir else None

    # Determine tasks
    if args.task == "all":
        tasks = list(TASK_CONFIG.keys())
    else:
        tasks = [args.task]

    # Determine templates
    if args.all_templates:
        template_names = list_templates()
    elif args.template:
        template_names = [args.template]
    else:
        parser.error("Specify --template or --all-templates")

    logger.info("Tasks: %s", tasks)
    logger.info("Templates: %s", template_names)

    for task in tasks:
        # Load SME artifacts (--attribution-dir overrides for property tasks)
        if attribution_dir_override is not None:
            attribution_dir = attribution_dir_override / task
        else:
            attribution_dir = _attribution_dir_for_task(task)
        attr_path = attribution_dir / "attributions.parquet"
        dec_path = attribution_dir / "decompositions.pkl"

        if not attr_path.exists() or not dec_path.exists():
            logger.warning("SME output not found for task '%s' at %s — skipping", task, attribution_dir)
            continue

        logger.info("Loading SME artifacts for %s...", task)
        attributions_df = pd.read_parquet(attr_path)
        with open(dec_path, "rb") as f:
            decompositions = pickle.load(f)

        for tname in template_names:
            template = load_template(tname)
            generate_for_template(
                task, tname, template, attributions_df, decompositions,
                output_dir=output_dir,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
