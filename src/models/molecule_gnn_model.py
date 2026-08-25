"""MolCA-compatible GIN model.

Our code. It reproduces the "pretrain-gnns" architecture (Hu et al., ICLR 2020,
https://github.com/snap-stanford/pretrain-gnns, MIT) that MolCA builds on, so
MolCA's released weights load into it key-for-key:
    - Atom features: (N, 2) = [atom_type (120), chirality_tag (3)]
    - Edge features: (E, 2) = [bond_type (6), bond_direction (3)]
    - 5 GINConv layers with BatchNorm + ReLU + Dropout between them
    - JK='last' -> returns final layer representation
    - Graph pooling: mean

MolCA checkpoint keys this must match:
    x_embedding1.weight (120, 300)
    x_embedding2.weight (3, 300)
    gnns.{i}.mlp.{0,2}.{weight,bias}
    gnns.{i}.edge_embedding1.weight (6, 300)
    gnns.{i}.edge_embedding2.weight (3, 300)
    batch_norms.{i}.{weight,bias,running_mean,running_var,num_batches_tracked}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool

num_atom_type = 120
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(nn.Module):
    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        drop_ratio: float = 0.0,
        gnn_type: str = "gin",
    ):
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be >= 2.")
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        if gnn_type != "gin":
            raise ValueError(f"Unsupported gnn_type={gnn_type} for MolCA variant (gin only).")

        self.gnns = nn.ModuleList([GINConv(emb_dim, aggr="add") for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            return sum(h_list[1:])
        if self.JK == "max":
            return torch.max(torch.stack(h_list[1:], dim=0), dim=0)[0]
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        raise ValueError(f"Unknown JK={self.JK}")


class GNN_graphpred(nn.Module):
    """Thin wrapper matching the MoleculeSTM interface."""

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        graph_pooling: str = "mean",
        num_tasks: int = 1,
        molecule_node_model: nn.Module = None,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.num_tasks = num_tasks
        self.molecule_node_model = molecule_node_model if molecule_node_model is not None else GNN(
            num_layer=num_layer, emb_dim=emb_dim, JK=JK
        )
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown graph_pooling={graph_pooling}")

        pool_dim = emb_dim * num_layer if JK == "concat" else emb_dim
        self.graph_pred_linear = nn.Linear(pool_dim, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.molecule_node_model(x, edge_index, edge_attr)
        pooled = self.pool(h, batch)
        return self.graph_pred_linear(pooled)
