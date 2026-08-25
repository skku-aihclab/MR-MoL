"""Data collators for batching molecular data."""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass

from .preprocessing import smiles_to_graph_batch


@dataclass
class MolecularCollator:
    """
    Collator for molecular data that handles both text and graph data.

    Batches:
    - SMILES strings (as list)
    - Tokenized text (padded tensors)
    - Graph data (batched PyG format)
    """

    tokenizer: Any
    include_graphs: bool = True
    padding_side: str = "right"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples with dynamic padding.

        Pads to the longest sequence in the batch.
        padding_side="right": [real tokens | PAD PAD] — use for forward() (classification)
        padding_side="left":  [PAD PAD | real tokens] — use for generate() (regression)
        """
        smiles_list = [item["smiles"] for item in batch]

        # Dynamic padding: pad to max length in batch
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = item["input_ids"].size(0)
            if self.padding_side == "left":
                input_ids[i, max_len - seq_len:] = item["input_ids"]
                attention_mask[i, max_len - seq_len:] = item["attention_mask"]
            else:
                input_ids[i, :seq_len] = item["input_ids"]
                attention_mask[i, :seq_len] = item["attention_mask"]

        result = {
            "smiles": smiles_list,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "labels" in batch[0]:
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, item in enumerate(batch):
                seq_len = item["labels"].size(0)
                if self.padding_side == "left":
                    labels[i, max_len - seq_len:] = item["labels"]
                else:
                    labels[i, :seq_len] = item["labels"]
            result["labels"] = labels

        if "task_type" in batch[0]:
            result["task_types"] = [item["task_type"] for item in batch]

        if "output_text" in batch[0]:
            result["output_texts"] = [item["output_text"] for item in batch]

        if "ground_truth" in batch[0]:
            result["ground_truths"] = [item["ground_truth"] for item in batch]

        if "raw_item" in batch[0]:
            result["raw_items"] = [item["raw_item"] for item in batch]

        # Process graph data
        if self.include_graphs:
            if any(not s for s in smiles_list):
                raise ValueError(
                    f"Empty SMILES in batch (positions "
                    f"{[i for i, s in enumerate(smiles_list) if not s]})."
                )
            result["graphs"] = smiles_to_graph_batch(smiles_list)

        return result


@dataclass
class Stage1Collator(MolecularCollator):
    """Collator for Stage 1 training."""
    include_graphs: bool = True


@dataclass
class Stage2Collator(MolecularCollator):
    """Collator for Stage 2 training."""
    include_graphs: bool = True


@dataclass
class EvaluationCollator(MolecularCollator):
    """Collator for evaluation. Preserves metadata for metric computation."""
    include_graphs: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super().__call__(batch)
        if "ground_truth" in batch[0]:
            result["ground_truths"] = [item["ground_truth"] for item in batch]
        return result
