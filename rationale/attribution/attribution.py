"""Compute substructure-masking attributions from masked predictions.

Follows the attribution of SME (Wu et al., Nat. Commun. 14, 2585, 2023;
https://github.com/wzxxxx/Substructure-Mask-Explanation), written with reference
to their implementation. The code here is restructured for multi-task labels and
seed averaging. See NOTICE.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TASK_CONFIG


def compute_attributions(predictions_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compute attributions from the raw predictions DataFrame.

    Args:
        predictions_df: output of ``run_all_seeds`` with columns
            smiles, sub_type, sub_name, sub_idx, seed, pred_0, ...
        task: task name (e.g. "bbbp")

    Returns:
        DataFrame with columns:
            smiles, sub_type, sub_name, sub_idx, task_idx, label_col,
            mol_pred_mean, masked_pred_mean, attribution
    """
    cfg = TASK_CONFIG[task]
    num_tasks = cfg["num_tasks"]
    is_cls = cfg["task_type"] == "classification"
    pred_cols = [f"pred_{t}" for t in range(num_tasks)]

    # Apply sigmoid for classification
    df = predictions_df.copy()
    if is_cls:
        for col in pred_cols:
            df[col] = 1.0 / (1.0 + np.exp(-df[col].values))

    # Split into mol-level and substructure-level
    mol_df = df[df["sub_type"] == "mol"].copy()
    sub_df = df[df["sub_type"] != "mol"].copy()

    # Average across seeds
    group_keys = ["smiles"]
    mol_mean = mol_df.groupby(group_keys)[pred_cols].mean().reset_index()
    mol_mean.columns = ["smiles"] + [f"mol_{c}" for c in pred_cols]

    sub_group_keys = ["smiles", "sub_type", "sub_name", "sub_idx"]
    sub_mean = sub_df.groupby(sub_group_keys)[pred_cols].mean().reset_index()
    sub_std = sub_df.groupby(sub_group_keys)[pred_cols].std().reset_index()
    sub_std.columns = sub_group_keys + [f"std_{c}" for c in pred_cols]

    # Merge
    merged = sub_mean.merge(mol_mean, on="smiles", how="left")
    merged = merged.merge(sub_std, on=sub_group_keys, how="left")

    # Compute attributions per task_idx
    rows = []
    for t in range(num_tasks):
        pc = f"pred_{t}"
        mol_col = f"mol_{pc}"
        std_col = f"std_{pc}"

        attr_raw = merged[mol_col].values - merged[pc].values
        attr_std = merged[std_col].values

        for i in range(len(merged)):
            rows.append({
                "smiles": merged.iloc[i]["smiles"],
                "sub_type": merged.iloc[i]["sub_type"],
                "sub_name": merged.iloc[i]["sub_name"],
                "sub_idx": int(merged.iloc[i]["sub_idx"]),
                "task_idx": t,
                "label_col": cfg["label_cols"][t],
                "mol_pred_mean": merged.iloc[i][mol_col],
                "masked_pred_mean": merged.iloc[i][pc],
                "attribution": attr_raw[i],
                "attribution_std": attr_std[i] if not np.isnan(attr_std[i]) else 0.0,
            })

    return pd.DataFrame(rows)
