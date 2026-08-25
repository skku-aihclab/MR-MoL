"""Dataset classes for molecular instruction tuning."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset

from configs.model_config import MOL_TOKEN_START, MOL_TOKEN_EMBED, MOL_TOKEN_END



def create_molecule_placeholder(mol_token: str, num_mol_tokens: int) -> str:
    """
    Create molecule placeholder string with start/end markers.

    Format: <|start_mol_id|><|mol_embed|>...<|mol_embed|><|end_mol_id|>
    """
    embed_tokens = mol_token * num_mol_tokens
    return f"{MOL_TOKEN_START}{embed_tokens}{MOL_TOKEN_END}"


def format_prompt_for_llm(
    instruction: str,
    output: str,
    system_prompt: str = None,
) -> str:
    """
    Format instruction and output into the Llama-3 chat template.

    Args:
        instruction: The instruction/input text (includes mol placeholder + raw SMILES)
        output: The expected output/response
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string
    """
    if system_prompt:
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )
    else:
        prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )

    return prompt


_SENTINEL_TEXT = "<|reserved_special_token_42|>"


def encode_preserving_suffix(
    tokenizer,
    processed_input: str,
    max_length: int,
    system_prompt: Optional[str],
    output_text: Optional[str] = None,
) -> tuple:
    """
    Tokenize a chat prompt while guaranteeing the suffix (chat scaffolding +
    answer/cue) is never truncated. If the full prompt would exceed `max_length`,
    only the `processed_input` region (instruction + mol + SMILES + rationale)
    is truncated from its end.

    Args:
        tokenizer: HF tokenizer (Llama-3.1 here).
        processed_input: User-content text to be embedded inside the chat template.
        max_length: Hard cap on total token length.
        system_prompt: Optional system message.
        output_text: Train target. None → inference prompt.

    Returns:
        (input_ids: List[int], prefix_len: int)
        prefix_len is the position where the answer starts (== full length when
        output_text is None).

    Raises:
        ValueError: if max_length is too small to fit the fixed scaffold + 1 input token.
    """
    is_train = output_text is not None

    if is_train:
        scaffold = format_prompt_for_llm(_SENTINEL_TEXT, output_text, system_prompt)
    else:
        scaffold = format_inference_prompt(_SENTINEL_TEXT, system_prompt)

    scaffold_ids = tokenizer(scaffold)["input_ids"]
    sentinel_id = tokenizer.convert_tokens_to_ids(_SENTINEL_TEXT)
    if sentinel_id is None or sentinel_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Sentinel '{_SENTINEL_TEXT}' did not map to a single token. "
            f"This tokenizer is not supported."
        )
    sentinel_idx = scaffold_ids.index(sentinel_id)

    pre_overhead = sentinel_idx
    post_overhead = len(scaffold_ids) - sentinel_idx - 1

    input_ids_only = tokenizer(processed_input, add_special_tokens=False)["input_ids"]
    budget = max_length - pre_overhead - post_overhead
    if budget < 1:
        raise ValueError(
            f"max_length={max_length} too small for fixed scaffold overhead="
            f"{pre_overhead + post_overhead}."
        )

    if len(input_ids_only) > budget:
        input_ids_only = input_ids_only[:budget]

    pre = scaffold_ids[:sentinel_idx]
    post = scaffold_ids[sentinel_idx + 1:]
    all_ids = pre + input_ids_only + post

    if is_train:
        inf_scaffold = format_inference_prompt(_SENTINEL_TEXT, system_prompt)
        inf_scaffold_ids = tokenizer(inf_scaffold)["input_ids"]
        inf_sentinel_idx = inf_scaffold_ids.index(sentinel_id)
        inf_post_overhead = len(inf_scaffold_ids) - inf_sentinel_idx - 1
        prefix_len = pre_overhead + len(input_ids_only) + inf_post_overhead
    else:
        prefix_len = len(all_ids)

    return all_ids, prefix_len


def format_inference_prompt(
    instruction: str,
    system_prompt: str = None,
) -> str:
    """
    Format instruction for inference (without output).

    Args:
        instruction: The instruction/input text
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string for generation
    """
    if system_prompt:
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    return prompt


class Stage1Dataset(Dataset):
    """
    Dataset for Stage 1: Graph-Text Alignment.

    Each sample contains:
    - instruction: instruction text
    - INPUT: SMILES string
    - output: description

    Prompt format: [Instruction]\n[Projected graph tokens]\nSMILES: [Raw SMILES]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        system_prompts_path: str,
        max_length: int = 512,
        mol_token: str = MOL_TOKEN_EMBED,
        num_mol_tokens: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mol_token = mol_token
        self.num_mol_tokens = num_mol_tokens

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        with open(system_prompts_path, 'r') as f:
            self.system_prompts = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        instruction = item["instruction"]
        smiles = item["INPUT"]
        output_text = item["output"]

        # Create molecule placeholder
        mol_placeholder = create_molecule_placeholder(self.mol_token, self.num_mol_tokens)

        # Format: [Instruction]\nMolecule: [Projected graph tokens]\nSMILES: [Raw SMILES]
        processed_input = f"{instruction}\nMolecule: {mol_placeholder}\nSMILES: {smiles}"

        system_prompt = self.system_prompts["description"]

        prompt = format_prompt_for_llm(
            processed_input, output_text, system_prompt
        )
        prefix_prompt = format_inference_prompt(
            processed_input, system_prompt
        )

        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prefix_len = len(self.tokenizer(prefix_prompt)["input_ids"])

        labels = encoded["input_ids"].clone()
        input_ids = encoded["input_ids"][0]

        mask_to = min(prefix_len, labels.shape[1])
        labels[0, :mask_to] = -100

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if input_ids[-1].item() == eot_id:
            labels[0, -1] = -100

        return {
            "smiles": smiles if smiles else "",
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2: Multi-task Instruction Tuning.

    Supports multiple task types:
    - Property prediction (classification/regression)

    Prompt format: [Instruction]\n[Projected graph tokens]\nSMILES: [Raw SMILES]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        system_prompts_path: str,
        max_length: int = 512,
        mol_token: str = MOL_TOKEN_EMBED,
        num_mol_tokens: int = 4,
        use_graph: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mol_token = mol_token
        self.num_mol_tokens = num_mol_tokens
        self.use_graph = use_graph

        with open(system_prompts_path, 'r') as f:
            self.system_prompts = json.load(f)

        self.data = []
        self.task_counts = {}
        data_path = Path(data_path)

        if data_path.is_dir():
            for json_file in sorted(data_path.glob("*.json")):
                task_name = json_file.stem
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                for item in task_data:
                    item["task"] = task_name
                    self.data.append(item)
                self.task_counts[task_name] = len(task_data)
                print(f"  Loaded {len(task_data)} samples from {json_file.name}")
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            for item in self.data:
                task = item["task"]
                self.task_counts[task] = self.task_counts.get(task, 0) + 1

        print(f"Total: {len(self.data)} samples")
        print("Task distribution:")
        for task, count in sorted(self.task_counts.items()):
            print(f"  {task}: {count}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        task_type = item["task"]

        instruction = item["instruction"]
        smiles = item["INPUT"]
        output_text = item["output"]

        if self.use_graph:
            mol_placeholder = create_molecule_placeholder(self.mol_token, self.num_mol_tokens)
            processed_input = f"{instruction}\nMolecule: {mol_placeholder}\nSMILES: {smiles}"
        else:
            processed_input = f"{instruction}\nSMILES: {smiles}"
        rationale = item.get("rationale", "")
        if rationale:
            processed_input = f"{processed_input}\n{rationale}"
        system_prompt = self.system_prompts["default"]

        all_ids, prefix_len = encode_preserving_suffix(
            self.tokenizer,
            processed_input,
            self.max_length,
            system_prompt,
            output_text=output_text,
        )

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if input_ids[-1].item() == eot_id:
            labels[-1] = -100

        return {
            "smiles": smiles if smiles else "",
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_type": task_type,
            "output_text": output_text,
        }


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation.

    Handles different task types with appropriate formatting.
    """

    def __init__(
        self,
        data_path: str,
        task_type: str,
        tokenizer,
        system_prompts_path: str,
        max_length: int = 512,
        mol_token: str = MOL_TOKEN_EMBED,
        num_mol_tokens: int = 4,
        use_graph: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mol_token = mol_token
        self.num_mol_tokens = num_mol_tokens
        self.task_type = task_type
        self.use_graph = use_graph

        with open(system_prompts_path, 'r') as f:
            self.system_prompts = json.load(f)

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples for evaluation ({task_type})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        instruction = item["instruction"]
        smiles = item["INPUT"]
        ground_truth = item["output"]

        if self.use_graph:
            mol_placeholder = create_molecule_placeholder(self.mol_token, self.num_mol_tokens)
            processed_input = f"{instruction}\nMolecule: {mol_placeholder}\nSMILES: {smiles}"
        else:
            processed_input = f"{instruction}\nSMILES: {smiles}"
        rationale = item.get("rationale", "")
        if rationale:
            processed_input = f"{processed_input}\n{rationale}"
        system_prompt = self.system_prompts["default"]

        all_ids, _ = encode_preserving_suffix(
            self.tokenizer,
            processed_input,
            self.max_length,
            system_prompt,
            output_text=None,
        )
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "smiles": smiles if smiles else "",
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth": ground_truth,
            "task_type": self.task_type,
            "raw_item": item,
        }
