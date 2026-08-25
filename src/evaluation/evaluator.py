"""Main evaluation class for all molecular tasks."""

import os
import json
import re
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from .metrics import (
    compute_classification_metrics,
    compute_multilabel_classification_metrics,
    compute_regression_metrics,
)
from ..datamodule.dataset import EvaluationDataset
from ..datamodule.collator import EvaluationCollator


# Robust regression parser: first signed float anywhere in the generated text
# (recovers e.g. '0.3616"...' that a naive split()[0] would drop).
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_regression_float(text: str):
    """First signed float anywhere in `text`; None if none parseable."""
    m = _FLOAT_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group())
    except (ValueError, OverflowError):
        return None


MULTILABEL_TASKS = {"tox21", "sider", "clintox"}
INSTRUCTION_SUFFIX = " Answer with only Yes or No."

# esol/lipo stage2 targets are z-score normalized (DeepChem NormalizationTransformer,
# fit on the train split); (mean, std) below inverts that so regression metrics come
# out in the original physical units (log solubility, logD) instead of z-units.
REGRESSION_DENORM_STATS = {
    "esol": (-2.8668604017333720, 2.0667167197226846),
    "lipo": (2.1629047619047745, 1.2109786829093336),
}


class Evaluator:
    """
    Unified evaluator for all molecular tasks.

    Supports:
    - Classification: BBBP, ClinTox, HIV, SIDER, Tox21
    - Regression: ESOL, Lipo
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_data_dir: str = "./data/stage2/test",
        system_prompts_path: str = "./data/stage2/instruction_templates/system_prompts.json",
        batch_size: int = 8,
        device: str = "cuda",
        output_dir: str = "./results",
        use_graph: bool = True,
    ):
        """
        Args:
            model: MolecularLLM model
            tokenizer: LLM tokenizer
            eval_data_dir: Directory containing evaluation data
            system_prompts_path: Path to system_prompts.json
            batch_size: Batch size for evaluation
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_data_dir = eval_data_dir
        self.system_prompts_path = system_prompts_path
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.use_graph = use_graph

        os.makedirs(output_dir, exist_ok=True)

        # Task configurations - all classification tasks together
        self.classification_tasks = ["bace", "bbbp", "clintox", "hiv", "sider", "tox21"]
        self.regression_tasks = ["esol", "lipo"]

        templates_dir = os.path.dirname(system_prompts_path)
        self.property_prediction_templates_path = os.path.join(
            templates_dir, "property_prediction.json"
        )

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Run evaluation on all tasks.

        Returns:
            Dictionary with all task metrics
        """
        results = {}

        print("\n" + "="*60)
        print("RUNNING FULL EVALUATION")
        print("="*60)

        # Classification tasks (including tox21, sider, clintox)
        for task in self.classification_tasks:
            print(f"\n--- Evaluating {task.upper()} ---")
            try:
                metrics = self.evaluate_classification(task)
                results[task] = metrics
                print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, MCC: {metrics['mcc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            except Exception as e:
                print(f"Error evaluating {task}: {e}")
                results[task] = {"error": str(e)}

        # Regression tasks
        for task in self.regression_tasks:
            print(f"\n--- Evaluating {task.upper()} ---")
            try:
                metrics = self.evaluate_regression(task)
                results[task] = metrics
                print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            except Exception as e:
                print(f"Error evaluating {task}: {e}")
                results[task] = {"error": str(e)}

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def evaluate_classification(self, task: str) -> Dict[str, float]:
        """Evaluate a classification task (binary Yes/No).

        For multi-label datasets (tox21, sider, clintox) the data is flattened as
        (molecule, sub_task) pairs. ROC-AUC and other metrics are computed per
        sub-task and macro-averaged (MoleculeNet convention).
        """
        data_path = os.path.join(self.eval_data_dir, f"property_prediction_{task}.json")
        task_type = f"property_prediction_{task}"

        dataset = EvaluationDataset(
            data_path=data_path,
            task_type=task_type,
            tokenizer=self.tokenizer,
            system_prompts_path=self.system_prompts_path,
            max_length=1024,
            use_graph=self.use_graph,
        )

        collator = EvaluationCollator(tokenizer=self.tokenizer, include_graphs=self.use_graph)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        is_multilabel = task in MULTILABEL_TASKS
        instr_to_subtask = self._build_subtask_lookup(task) if is_multilabel else None

        self.model.eval()
        y_true, y_prob, sub_task_ids = [], [], []

        yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                graphs = batch.get("graphs")
                ground_truths = batch["ground_truths"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graphs=graphs,
                )
                last_token_idx = attention_mask.sum(dim=-1) - 1
                logits = outputs["logits"][torch.arange(outputs["logits"].size(0), device=outputs["logits"].device), last_token_idx, :]

                yes_logits = logits[:, yes_id]
                no_logits = logits[:, no_id]
                probs = torch.softmax(torch.stack([yes_logits, no_logits], dim=1), dim=1)

                y_prob.extend(probs[:, 0].cpu().tolist())
                y_true.extend([1 if gt.lower() == "yes" else 0 for gt in ground_truths])

                if is_multilabel:
                    sub_task_ids.extend(
                        self._lookup_subtask(item["instruction"], instr_to_subtask, task)
                        for item in batch["raw_items"]
                    )

        if is_multilabel:
            return compute_multilabel_classification_metrics(y_true, y_prob, sub_task_ids)
        return compute_classification_metrics(y_true, y_prob)

    def _build_subtask_lookup(self, task: str) -> Dict[str, str]:
        """Map cleaned instruction text → sub-task id from the template JSON."""
        with open(self.property_prediction_templates_path) as f:
            templates = json.load(f)
        if task not in templates:
            raise KeyError(
                f"task '{task}' missing from {self.property_prediction_templates_path}"
            )
        lookup: Dict[str, str] = {}
        for sub_task, paraphrases in templates[task].items():
            for entry in paraphrases:
                instr = entry["input"].replace("\n<INPUT>", "").replace("<INPUT>", "").strip()
                lookup[instr] = sub_task
        return lookup

    @staticmethod
    def _lookup_subtask(instruction: str, lookup: Dict[str, str], task: str) -> str:
        cleaned = instruction
        if cleaned.endswith(INSTRUCTION_SUFFIX):
            cleaned = cleaned[: -len(INSTRUCTION_SUFFIX)]
        cleaned = cleaned.strip()
        if cleaned not in lookup:
            raise KeyError(
                f"could not map instruction to sub-task for '{task}':\n  {instruction!r}"
            )
        return lookup[cleaned]

    def evaluate_regression(self, task: str) -> Dict[str, float]:
        """Evaluate a regression task."""
        data_path = os.path.join(self.eval_data_dir, f"property_prediction_{task}.json")
        task_type = f"property_prediction_{task}"

        dataset = EvaluationDataset(
            data_path=data_path,
            task_type=task_type,
            tokenizer=self.tokenizer,
            system_prompts_path=self.system_prompts_path,
            max_length=1024,
            use_graph=self.use_graph,
        )

        collator = EvaluationCollator(tokenizer=self.tokenizer, padding_side="left", include_graphs=self.use_graph)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
                ground_truths = batch["ground_truths"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                graphs = batch.get("graphs")

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graphs=graphs,
                    max_new_tokens=32,
                    do_sample=False,
                )

                for pred, gt in zip(outputs, ground_truths):
                    if gt is None or (isinstance(gt, str) and gt.strip() == ""):
                        raise ValueError(f"empty/None regression ground-truth: {gt!r}")
                    y_pred.append(parse_regression_float(pred))
                    y_true.append(float(gt))

        if task not in REGRESSION_DENORM_STATS:
            raise KeyError(f"No de-normalization stats for task={task!r}")
        mean, std = REGRESSION_DENORM_STATS[task]
        y_true = [v * std + mean for v in y_true]
        y_pred = [None if v is None else v * std + mean for v in y_pred]

        return compute_regression_metrics(y_true, y_pred)

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        output_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Classification
        print("\n--- Classification ---")
        print(f"{'Task':<12} {'Accuracy':>10} {'F1':>10} {'MCC':>10} {'ROC-AUC':>10}")
        print("-" * 52)
        for task in self.classification_tasks:
            if task in results and "accuracy" in results[task]:
                m = results[task]
                print(f"{task.upper():<12} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['mcc']:>10.4f} {m['roc_auc']:>10.4f}")

        # Regression
        print("\n--- Regression ---")
        print(f"{'Task':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print("-" * 42)
        for task in self.regression_tasks:
            if task in results and "rmse" in results[task]:
                m = results[task]
                print(f"{task.upper():<12} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>10.4f}")

        print("="*60)
