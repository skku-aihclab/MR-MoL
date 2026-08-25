"""Parametric rationale formatter — ranked top-K items.

Murcko items include both the scaffold row and side-chain rows.

Output format per item:

    1. type: Murcko
    scaffold SMILES: O=c1cc[nH]c2ccccc12
    effect: toward higher predicted probability of blood-brain barrier permeability

    2. type: functional group
    name: carboxylic acids
    effect: toward lower predicted probability of blood-brain barrier permeability

    3. type: BRICS
    fragment SMILES: O=CO
    raw BRICS: [6*]C(=O)O
    attachments: [6]
    effect: toward lower predicted probability of blood-brain barrier permeability

"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DROP_CUTOFF = 1e-6   # |attribution| must exceed this to be kept (filters exact zeros)
TOP_K = 5


# ---------------------------------------------------------------------------
# Task-specific text (positive-class phrasing for classification)
# ---------------------------------------------------------------------------

TASK_TARGETS: dict[str, list[str]] = {
    "bace":       ["beta-secretase 1 (BACE-1) inhibition"],
    "bbbp":       ["blood-brain barrier permeability"],
    "clintox":    ["FDA approval", "clinical toxicity"],
    "esol":       ["log aqueous solubility"],
    "hiv":        ["HIV replication inhibition"],
    "lipo":       ["logD at pH 7.4"],
    "tox21": [
        "NR-AR activity", "NR-AR-LBD activity", "NR-AhR activity",
        "NR-Aromatase activity", "NR-ER activity", "NR-ER-LBD activity",
        "NR-PPAR-gamma activity", "SR-ARE activity", "SR-ATAD5 activity",
        "SR-HSE activity", "SR-MMP activity", "SR-p53 activity",
    ],
    "sider": [
        "hepatobiliary disorders", "metabolism and nutrition disorders",
        "product issues", "eye disorders", "abnormal investigations",
        "musculoskeletal and connective tissue disorders",
        "gastrointestinal disorders", "social circumstance issues",
        "immune system disorders", "reproductive system and breast disorders",
        "neoplasms", "general disorders and administration site conditions",
        "endocrine disorders", "surgical and medical procedures",
        "vascular disorders", "blood and lymphatic system disorders",
        "skin and subcutaneous tissue disorders",
        "congenital, familial, and genetic disorders",
        "infections and infestations",
        "respiratory, thoracic, and mediastinal disorders",
        "psychiatric disorders", "renal and urinary disorders",
        "pregnancy, puerperium, and perinatal conditions",
        "ear and labyrinth disorders", "cardiac disorders",
        "nervous system disorders",
        "injury, poisoning, and procedural complications",
    ],
}

TASK_TYPES: dict[str, str] = {
    "bace":    "classification",
    "bbbp":    "classification",
    "clintox": "classification",
    "hiv":     "classification",
    "tox21":   "classification",
    "sider":   "classification",
    "esol":    "regression",
    "lipo":    "regression",
}


def _effect_phrase(row: pd.Series, task: str) -> str:
    target = TASK_TARGETS[task][int(row["task_idx"])]
    direction = "higher" if row["attribution"] > 0 else "lower"
    if TASK_TYPES[task] == "classification":
        return f"toward {direction} predicted probability of {target}"
    return f"toward {direction} predicted {target}"


# ---------------------------------------------------------------------------
# Per-molecule selection (filter → rank)
# ---------------------------------------------------------------------------

# Template granularity key → attribution sub_type.
# "murcko" selects all Murcko items (scaffold row + side-chain rows).
_GRAN_TO_SUBTYPE = {"murcko": "murcko", "fg": "fg", "brics": "brics"}


def _select_top_k(attrs: pd.DataFrame, granularity: list[str]) -> pd.DataFrame:
    """Filter by granularity + cutoff, sort by |attribution| desc, take top-K."""
    if not granularity:
        return attrs.iloc[0:0]
    allowed = {_GRAN_TO_SUBTYPE[g] for g in granularity if g in _GRAN_TO_SUBTYPE}
    sub = attrs[attrs["sub_type"].isin(allowed)].copy()
    if "brics" in allowed:
        sub = sub[sub["sub_name"] != "whole_molecule"]
    sub = sub[sub["attribution"].abs() > DROP_CUTOFF]
    if sub.empty:
        return sub
    sub = sub.assign(_abs=sub["attribution"].abs())
    sub = sub.sort_values("_abs", ascending=False).head(TOP_K).reset_index(drop=True)
    return sub.drop(columns="_abs")


# ---------------------------------------------------------------------------
# Per-item rendering
# ---------------------------------------------------------------------------

def _render_item(row: pd.Series, decomp_map: dict, rank: int, task: str) -> list[str]:
    sub_type = row["sub_type"]
    sub_name = row["sub_name"]
    sub_idx = int(row["sub_idx"])
    effect = _effect_phrase(row, task)

    if sub_type == "murcko":
        info = decomp_map.get("murcko", {}).get(sub_idx, {})
        smi = info.get("smiles", "")
        key = "scaffold SMILES" if sub_name == "scaffold" else "side chain SMILES"
        return [
            f"{rank}. type: Murcko",
            f"{key}: {smi}",
            f"effect: {effect}",
        ]

    if sub_type == "fg":
        return [
            f"{rank}. type: functional group",
            f"name: {sub_name}",
            f"effect: {effect}",
        ]

    if sub_type == "brics":
        info = decomp_map.get("brics", {}).get(sub_idx, {})
        extra = info.get("extra", {})
        smi = info.get("smiles", "")
        raw = extra.get("raw_brics", smi)
        attachments = extra.get("attachments", [])
        attach_str = ",".join(str(a) for a in attachments) if attachments else "none"
        return [
            f"{rank}. type: BRICS",
            f"fragment SMILES: {smi}",
            f"raw BRICS: {raw}",
            f"attachments: [{attach_str}]",
            f"effect: {effect}",
        ]

    raise ValueError(f"Unknown sub_type: {sub_type!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_rationale_from_template(
    attrs: pd.DataFrame,
    decomp_map: dict,
    template: dict,
    task: str,
) -> str:
    """Build a rationale block for one (molecule, task_idx)."""
    if task not in TASK_TARGETS:
        raise KeyError(f"No TASK_TARGETS entry for task={task!r}")
    if task not in TASK_TYPES:
        raise KeyError(f"No TASK_TYPES entry for task={task!r}")

    granularity = template.get("granularity", [])
    if not granularity:
        return ""

    selected = _select_top_k(attrs, granularity)
    if selected.empty:
        return ""

    parts = ["Ranked structural rationale items for this molecule."]
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        parts.append("\n".join(_render_item(row, decomp_map, rank, task)))
    return "\n\n".join(parts)
