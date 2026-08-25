"""Graph encoder (GNN) wrapper for MolCA weights.

Our code; the weights it loads come from MolCA (Liu et al., EMNLP 2023).

The MolCA checkpoint is a unified `stage1.ckpt` that also holds the
Q-Former weights, so it is read once inside `MolecularLLM._load_molca` and the
graph encoder slice is passed in via `load_molca_graph_state(...)`.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MolCAGraphEncoder(nn.Module):
    """MolCA fine-tuned GIN wrapper.

    Architecture: 5-layer GIN, 300D hidden, JK=last, mean pool — same as
    MoleculeSTM, but with weights from MolCA's `stage1.ckpt`.
    """

    def __init__(self, freeze: bool = False, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.freeze = freeze

        from torch_geometric.nn import global_mean_pool
        from .molecule_gnn_model import GNN, GNN_graphpred

        molecule_node_model = GNN(
            num_layer=5,
            emb_dim=300,
            JK="last",
            drop_ratio=0.0,
            gnn_type="gin",
        )
        self.encoder = GNN_graphpred(
            num_layer=5,
            emb_dim=300,
            JK="last",
            graph_pooling="mean",
            num_tasks=1,
            molecule_node_model=molecule_node_model,
        )
        self.pool = global_mean_pool
        self.output_dim = 300

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.encoder.to(self.device)

    def load_molca_graph_state(self, stripped_state_dict: Dict[str, torch.Tensor]) -> None:
        """Load the graph-encoder slice of a MolCA checkpoint.

        MolCA stores GIN weights under `graph_encoder.*`. Our `GNN_graphpred`
        wraps the GIN under `molecule_node_model.*`, so we remap:

            graph_encoder.foo  ->  molecule_node_model.foo
        """
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in stripped_state_dict.items():
            if k.startswith("graph_encoder."):
                new_key = "molecule_node_model." + k[len("graph_encoder."):]
                remapped[new_key] = v

        if not remapped:
            raise RuntimeError(
                "load_molca_graph_state: no keys matched `graph_encoder.*`. "
                "Check the MolCA checkpoint prefix detection."
            )

        model_state = self.encoder.state_dict()
        kept: Dict[str, torch.Tensor] = {}
        shape_skipped = []
        for k, v in remapped.items():
            if k in model_state and v.shape == model_state[k].shape:
                kept[k] = v
            else:
                shape_skipped.append(k)

        missing, unexpected = self.encoder.load_state_dict(kept, strict=False)
        print(
            f"[MolCAGraphEncoder] loaded {len(kept)}/{len(remapped)} MolCA graph keys "
            f"(shape_skipped={len(shape_skipped)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)})"
        )
        for k in shape_skipped[:10]:
            print(f"  [shape_skipped] {k}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_output:   [num_nodes, 300]
            pooled_output: [batch_size, 300]
        """
        node_output = self.encoder.molecule_node_model(x, edge_index, edge_attr)
        if batch is not None:
            pooled_output = self.pool(node_output, batch)
        else:
            pooled_output = node_output.mean(dim=0, keepdim=True)
        return node_output, pooled_output
