"""MolCA-compatible preprocessing.

Produces graph features that match the canonical Hu et al. pretrain-gnns scheme
used by MolCA:
    - Atom features (N, 2): [atom_type_index, chirality_index]
    - Edge features (E, 2): [bond_type_index, bond_direction_index]
"""

from typing import Dict, List

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType, BondDir

ALLOWED_CHIRALITY = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]

ALLOWED_BOND_TYPE = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]

ALLOWED_BOND_DIR = [
    BondDir.NONE,
    BondDir.ENDUPRIGHT,
    BondDir.ENDDOWNRIGHT,
]


def _atom_to_feature(atom) -> list:
    atomic_num = atom.GetAtomicNum()
    chirality = atom.GetChiralTag()
    chirality_idx = ALLOWED_CHIRALITY.index(chirality) if chirality in ALLOWED_CHIRALITY else 0
    return [atomic_num, chirality_idx]


def _bond_to_feature(bond) -> list:
    bt = bond.GetBondType()
    bd = bond.GetBondDir()
    bt_idx = ALLOWED_BOND_TYPE.index(bt) if bt in ALLOWED_BOND_TYPE else 0
    bd_idx = ALLOWED_BOND_DIR.index(bd) if bd in ALLOWED_BOND_DIR else 0
    return [bt_idx, bd_idx]


def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
    """Convert SMILES to MolCA-format graph. Raises on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit failed to parse SMILES or empty molecule: {smiles!r}")

    atom_feats = [_atom_to_feature(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.long)

    edge_src = []
    edge_dst = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_to_feature(bond)
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attr.extend([feat, feat])

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, 2), dtype=torch.long)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr_t}


def smiles_to_graph_batch(smiles_list: List[str]) -> Dict[str, torch.Tensor]:
    """Batched MolCA-format graph construction. Raises if any SMILES is unparseable."""
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_batch = []
    node_offset = 0

    for batch_idx, smiles in enumerate(smiles_list):
        graph = smiles_to_graph(smiles)

        num_nodes = graph["x"].shape[0]
        all_x.append(graph["x"])
        if graph["edge_index"].shape[1] > 0:
            all_edge_index.append(graph["edge_index"] + node_offset)
            all_edge_attr.append(graph["edge_attr"])
        all_batch.append(torch.full((num_nodes,), batch_idx, dtype=torch.long))
        node_offset += num_nodes

    x = torch.cat(all_x, dim=0)
    batch = torch.cat(all_batch, dim=0)
    if all_edge_index:
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "batch": batch}
