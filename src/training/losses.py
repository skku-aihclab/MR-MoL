"""Loss functions for Stage 1 and Stage 2 training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def compute_per_sample_nll(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Per-sample mean next-token NLL over answer tokens.

    Same shift logic as nn.CrossEntropyLoss-on-shifted-logits, but instead of
    pooling all tokens into a single mean we return one value per sample —
    the mean NLL of that sample's answer tokens. This makes each sample
    contribute equal weight regardless of answer length.

    Args:
        lm_logits: [B, T, V]
        labels:    [B, T] with -100 on prefix / pad positions
        ignore_index: label value to ignore (default -100)

    Returns:
        Tensor of shape [B], per-sample mean NLL.

    Raises:
        RuntimeError: if any sample has zero non-ignored answer tokens.
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    B, T_minus_1, V = shift_logits.shape

    flat_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=ignore_index,
    ).view(B, T_minus_1)

    answer_mask = (shift_labels != ignore_index).to(flat_loss.dtype)
    sum_per_sample = (flat_loss * answer_mask).sum(dim=1)
    n_per_sample = answer_mask.sum(dim=1)

    if (n_per_sample == 0).any():
        zero_idx = (n_per_sample == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            f"sample(s) with zero non-ignored answer tokens at batch indices "
            f"{zero_idx} — likely truncation cut the entire answer"
        )

    return sum_per_sample / n_per_sample


class Stage1Loss(nn.Module):
    """
    Loss for Stage 1: Graph-Language Alignment.

    Cross-entropy next-token prediction on the target description.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute next-token-prediction loss.

        Args:
            lm_logits: LLM output logits [batch, seq_len, vocab_size]
            labels: Target token ids [batch, seq_len]

        Returns:
            Dictionary with total loss
        """
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.lm_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return {'total': loss}


class Stage2Loss(nn.Module):
    """
    Loss for Stage 2: Multi-task Instruction Tuning.

    All tasks use CrossEntropy loss for token generation.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.task_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task loss.

        Args:
            lm_logits: LLM output logits [batch, seq_len, vocab_size]
            labels: Target token ids [batch, seq_len]

        Returns:
            Dictionary with total loss
        """
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.task_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return {'total': loss}
