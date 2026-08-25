"""SMILES-to-graph, model loading, masked pooling, and inference utilities.

Builds on the Mole-BERT modules that download_molebert.sh fetches into
`rationale/molebert/` for the GIN architecture and the graph featurization.
See NOTICE.
"""

import sys
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch

from config import (
    MOLEBERT_DIR, RESULT_DIR,
    NUM_LAYER, EMB_DIM, JK, DROP_RATIO, GRAPH_POOLING, GNN_TYPE, TASK_CONFIG,
)

# Import Mole-BERT modules
sys.path.insert(0, MOLEBERT_DIR)
from model import GNN_graphpred  # noqa: E402
from loader import mol_to_graph_data_obj_simple  # noqa: E402


CKPT_MODES = ("val_best",)


def finetune_ckpt_path(task: str, seed: int, mode: str = "val_best") -> str:
    """Resolve the on-disk path for the val-best Mole-BERT finetuned ckpt:
    RESULT_DIR/{task}/seed_{s}.pth
    """
    if mode not in CKPT_MODES:
        raise ValueError(f"Unknown ckpt mode: {mode!r}; expected one of {CKPT_MODES}")
    return f"{RESULT_DIR}/{task}/seed_{seed}.pth"


# ---------------------------------------------------------------------------
# SMILES -> PyG Data
# ---------------------------------------------------------------------------

def smiles_to_pyg_data(smiles: str) -> Data | None:
    """Convert a SMILES string to a PyG Data object.

    Returns None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_graph_data_obj_simple(mol)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(task: str, seed: int, device: torch.device, mode: str = "val_best") -> GNN_graphpred:
    """Load the val-best finetuned Mole-BERT checkpoint and set to eval mode."""
    cfg = TASK_CONFIG[task]
    model = GNN_graphpred(
        num_layer=NUM_LAYER,
        emb_dim=EMB_DIM,
        num_tasks=cfg["num_tasks"],
        JK=JK,
        drop_ratio=DROP_RATIO,
        graph_pooling=GRAPH_POOLING,
        gnn_type=GNN_TYPE,
    )
    ckpt_path = finetune_ckpt_path(task, seed, mode)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Masked mean pooling
# ---------------------------------------------------------------------------

def masked_mean_pool(x: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool node features, excluding masked-out nodes.

    Args:
        x:     [N, D] node representations
        batch: [N]    graph assignment per node
        mask:  [N]    1.0 = keep, 0.0 = exclude
    Returns:
        [B, D] graph-level representations
    """
    from torch_geometric.utils import scatter
    x_masked = x * mask.unsqueeze(-1)
    out_sum = scatter(x_masked, batch, dim=0, reduce="sum")
    count = scatter(mask, batch, dim=0, reduce="sum").clamp(min=1)
    return out_sum / count.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Masked prediction (decomposed forward pass)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_mask(model: GNN_graphpred, batch: Batch, device: torch.device) -> np.ndarray:
    """Run model inference with per-node masking at the pooling step.

    Expects ``batch.mask`` to be a float tensor of shape [N] with 1=keep / 0=exclude.
    GIN message passing runs on the full graph; only pooling is masked.

    Returns:
        np.ndarray of shape [B, num_tasks] -- raw logits (classification) or values (regression).
    """
    batch = batch.to(device)
    node_repr = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
    pooled = masked_mean_pool(node_repr, batch.batch, batch.mask)
    pred = model.graph_pred_linear(pooled)
    return pred.cpu().numpy()


@torch.no_grad()
def predict_standard(model: GNN_graphpred, batch: Batch, device: torch.device) -> np.ndarray:
    """Standard forward pass (no masking) for sanity-check comparison."""
    batch = batch.to(device)
    pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def prepare_batch(data_list: list[Data], mask_list: list[torch.Tensor]) -> Batch:
    """Attach mask tensors to Data objects and create a Batch.

    Args:
        data_list:  list of PyG Data objects
        mask_list:  list of float tensors, each of shape [num_atoms_i]
    """
    for data, mask in zip(data_list, mask_list):
        data.mask = mask
    return Batch.from_data_list(data_list, follow_batch=[], exclude_keys=[])
