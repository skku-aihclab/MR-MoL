"""Substructure decomposition: functional groups, Murcko, BRICS fragments.

The functional-group SMARTS set and two of its helpers are ported from SME
(Wu et al., Nat. Commun. 14, 2585, 2023;
https://github.com/wzxxxx/Substructure-Mask-Explanation); the ported blocks are
marked inline. The Murcko and BRICS views are our own. See NOTICE.

Each decomposition function takes an RDKit Mol and returns a list of
``SubstructureHit`` dicts describing each identified substructure.

SubstructureHit schema::

    {
        "sub_type": "fg" | "murcko" | "brics",
        "name":     str,           # human-readable name
        "atoms":    list[int],     # atom indices in the original mol
        "smiles":   str | None,    # representative SMILES of the substructure
        "extra":    dict,          # type-specific extra fields
    }
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import BRICS, FragmentCatalog, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Functional-group SMARTS: same 39 groups as SME's build_data.py, ported
# unmodified from https://github.com/wzxxxx/Substructure-Mask-Explanation
# ---------------------------------------------------------------------------

_FG_SETUP_CACHE: dict | None = None


def _setup_fg():
    """Lazy-init functional group SMARTS from RDKit's FunctionalGroups.txt."""
    global _FG_SETUP_CACHE
    if _FG_SETUP_CACHE is not None:
        return _FG_SETUP_CACHE

    fName = os.path.join(RDConfig.RDDataDir, "FunctionalGroups.txt")
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)

    # SMARTS *without* the connecting-carbon environment (core FG atoms only)
    fg_without_ca_smart = [
        "[N;D2]-[C;D3](=O)-[C;D1;H3]",
        "C(=O)[O;D1]",
        "C(=O)[O;D2]-[C;D1;H3]",
        "C(=O)-[H]",
        "C(=O)-[N;D1]",
        "C(=O)-[C;D1;H3]",
        "[N;D2]=[C;D2]=[O;D1]",
        "[N;D2]=[C;D2]=[S;D1]",
        "[N;D3](=[O;D1])[O;D1]",
        "[N;R0]=[O;D1]",
        "[N;R0]-[O;D1]",
        "[N;R0]-[C;D1;H3]",
        "[N;R0]=[C;D1;H2]",
        "[N;D2]=[N;D2]-[C;D1;H3]",
        "[N;D2]=[N;D1]",
        "[N;D2]#[N;D1]",
        "[C;D2]#[N;D1]",
        "[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
        "[S;D4](=[O;D1])(=[O;D1])-[O;D1]",
        "[S;D4](=[O;D1])(=[O;D1])-[C;D1]",
        "[S;D2](=O)-[C;D1]",
        "[P;D4](=[O;D1])(-[O;D2])(-[O;D2])-[O;D2]",
        "[P;D4](=[O;D1])(-[O;D1])(-[O;D1])-[O;D1]",
        "[C;D2]=[C;D1;H2]",
        "[C;D3]([C;D1])=[C;D1]",
        "[C;D4]([C;D1])([C;D1])-[C;D1]",
        "[C;D4](F)(F)F",
        "[C;D2]#[C;D1;H]",
        "[C;D3]1-[C;D2]-[C;D2]1",
        "[O;D2]-[C;D2]-[C;D1;H3]",
        "[O;D2]-[C;D1;H3]",
        "[O;D1]",
        "[O;D1]",
        "[N;D1]",
        "[N;D1]",
        "[N;D1]",
        "[S;D1]",
        "[S;D1]",
        "[Cl]",
    ]
    fg_without_ca_list = [Chem.MolFromSmarts(s) for s in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    _label_to_notes = {}
    with open(fName) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("//"):
                continue
            _parts = [p.strip() for p in _line.split("\t") if p.strip() and not p.strip().startswith("*")]
            if len(_parts) >= 2:
                _label_to_notes[_parts[0]] = _parts[1]
    fg_name_list = [
        (v if v and v != "???" else None)
        for fg in fg_with_ca_list
        for v in [_label_to_notes.get(fg.GetProp("_Name"))]
    ]

    _FG_SETUP_CACHE = {
        "names": fg_name_list,
        "with_ca": fg_with_ca_list,
        "without_ca": fg_without_ca_list,
        "_fparams": fparams,  # prevent GC — GetFuncGroup results are invalidated otherwise
    }
    return _FG_SETUP_CACHE


# ---------------------------------------------------------------------------
# Helpers ported from SME's build_data.py
# (https://github.com/wzxxxx/Substructure-Mask-Explanation)
# ---------------------------------------------------------------------------

def _fg_wash(fg_with_c_matches, fg_without_c_matches):
    """Remove redundant carbon atoms from FG matches."""
    result = []
    for fg_with_c in fg_with_c_matches:
        for fg_without_c in fg_without_c_matches:
            if set(fg_without_c).issubset(set(fg_with_c)):
                result.append(list(fg_without_c))
    return result


def _bfs_fragment(mol, start_atoms: list[int], forbidden: set[int]) -> list[int]:
    """BFS from start_atoms, not crossing into forbidden atom set."""
    visited = set(start_atoms)
    queue = list(start_atoms)
    while queue:
        idx = queue.pop(0)
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in visited and nidx not in forbidden:
                visited.add(nidx)
                queue.append(nidx)
    return sorted(visited)


# ---------------------------------------------------------------------------
# Decomposition: Functional Groups
# ---------------------------------------------------------------------------

def decompose_fg(mol: Chem.Mol) -> list[dict]:
    """Identify functional groups in *mol* using RDKit's 39 standard FG SMARTS.

    Returns one entry per matched FG *instance* (a single FG type can match
    multiple times in the same molecule).
    """
    setup = _setup_fg()
    names = setup["names"]
    with_ca = setup["with_ca"]
    without_ca = setup["without_ca"]

    # Step 1: find all matches
    hit_at = []        # list of list-of-list  (per FG type, per instance)
    hit_names = []
    all_hit = []       # flat list of all atom-index lists

    for i in range(len(with_ca)):
        if names[i] is None:
            continue
        matches_with = mol.GetSubstructMatches(with_ca[i])
        matches_without = mol.GetSubstructMatches(without_ca[i])
        washed = _fg_wash(matches_with, matches_without)
        if washed:
            hit_at.append(washed)
            hit_names.append(names[i])
            all_hit.extend(washed)

    # Step 2: de-duplicate — larger groups first, remove subsets
    sorted_hits = sorted(all_hit, key=len, reverse=True)
    remaining = []
    for fg in sorted_hits:
        if fg in remaining:
            continue
        if not remaining:
            remaining.append(fg)
        else:
            is_subset = any(set(fg).issubset(set(r)) for r in remaining)
            if not is_subset:
                remaining.append(fg)

    # Step 3: filter original hits against remaining
    results = []
    for j, instances in enumerate(hit_at):
        for inst in instances:
            if inst in remaining:
                results.append({
                    "sub_type": "fg",
                    "name": hit_names[j],
                    "atoms": sorted(inst),
                    "smiles": None,  # filled below
                    "extra": {},
                })

    # Generate SMILES for each FG instance
    for r in results:
        try:
            r["smiles"] = Chem.MolFragmentToSmiles(mol, r["atoms"])
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Decomposition: Murcko
# ---------------------------------------------------------------------------

def decompose_murcko(mol: Chem.Mol) -> list[dict]:
    """Decompose *mol* into Murcko scaffold + side-chain fragments.

    Returns one entry for the scaffold and one per side-chain group.
    """
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return []

    if core.GetNumAtoms() == 0:
        return []

    match = mol.GetSubstructMatch(core)
    scaffold_atoms = set(match)
    scaffold_smi = Chem.MolToSmiles(core)

    results = [{
        "sub_type": "murcko",
        "name": "scaffold",
        "atoms": sorted(scaffold_atoms),
        "smiles": scaffold_smi,
        "extra": {"role": "scaffold"},
    }]

    # Find link bonds (scaffold<->sidechain) and side-chain groups via BFS
    link_bond_atoms = []  # (scaffold_atom, sidechain_atom) pairs
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (u in scaffold_atoms) != (v in scaffold_atoms):
            sc_atom = u if u not in scaffold_atoms else v
            link_bond_atoms.append(sc_atom)

    # BFS from each sidechain seed, avoiding scaffold atoms
    visited_sc = set()
    sc_idx = 0
    for seed in link_bond_atoms:
        if seed in visited_sc:
            continue
        group = _bfs_fragment(mol, [seed], scaffold_atoms)
        visited_sc.update(group)
        try:
            sc_smi = Chem.MolFragmentToSmiles(mol, group)
        except Exception:
            sc_smi = None
        results.append({
            "sub_type": "murcko",
            "name": f"sidechain_{sc_idx}",
            "atoms": group,
            "smiles": sc_smi,
            "extra": {"role": "sidechain"},
        })
        sc_idx += 1

    return results


# ---------------------------------------------------------------------------
# Decomposition: BRICS
# ---------------------------------------------------------------------------

def decompose_brics(mol: Chem.Mol) -> list[dict]:
    """Break all BRICS bonds at once and return leaf fragments.

    Uses ``BRICS.BreakBRICSBonds`` (single linear-time decomposition) instead of
    ``BRICS.BRICSDecompose`` (combinatorial enumeration that explodes on
    peptide-like molecules). The fragment atom sets and raw_brics SMILES with
    [n*] dummy labels are obtained directly from the broken-bond molecule.
    """
    res = list(BRICS.FindBRICSBonds(mol))
    if not res:
        # No BRICS bonds — entire molecule is one fragment
        all_atoms = list(range(mol.GetNumAtoms()))
        return [{
            "sub_type": "brics",
            "name": "whole_molecule",
            "atoms": all_atoms,
            "smiles": Chem.MolToSmiles(mol),
            "extra": {"raw_brics": Chem.MolToSmiles(mol), "attachments": []},
        }]

    brics_bond_types = {r[0]: r[1] for r in res}  # (a,b) -> ('type_a', 'type_b')

    broken = BRICS.BreakBRICSBonds(mol)
    n_orig = mol.GetNumAtoms()
    frag_groups = Chem.GetMolFrags(broken)
    frag_mols = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)

    results = []
    for i, (atoms_tuple, fmol) in enumerate(zip(frag_groups, frag_mols)):
        orig_atoms = sorted(a for a in atoms_tuple if a < n_orig)
        if not orig_atoms:
            continue
        try:
            frag_smi = Chem.MolFragmentToSmiles(mol, orig_atoms)
        except Exception:
            frag_smi = None
        try:
            raw_brics = Chem.MolToSmiles(fmol)
        except Exception:
            raw_brics = frag_smi

        # Which BRICS bond type labels attach to this fragment
        atom_set = set(orig_atoms)
        attach_labels = []
        for (a, b), bond_types in brics_bond_types.items():
            if a in atom_set and b not in atom_set:
                attach_labels.append(bond_types[0])
            elif b in atom_set and a not in atom_set:
                attach_labels.append(bond_types[1])

        results.append({
            "sub_type": "brics",
            "name": f"brics_{i}",
            "atoms": orig_atoms,
            "smiles": frag_smi,
            "extra": {
                "raw_brics": raw_brics or frag_smi,
                "attachments": sorted(set(attach_labels)),
            },
        })

    return results


# ---------------------------------------------------------------------------
# Master decomposition
# ---------------------------------------------------------------------------

@dataclass
class MolDecomposition:
    smiles: str
    num_atoms: int
    fg: list[dict] = field(default_factory=list)
    murcko: list[dict] = field(default_factory=list)
    brics: list[dict] = field(default_factory=list)
    error: Optional[str] = None


def decompose_molecule(smiles: str) -> MolDecomposition:
    """Decompose a molecule into functional groups, Murcko, and BRICS fragments."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MolDecomposition(smiles=smiles, num_atoms=0, error="parse_failed")

    result = MolDecomposition(smiles=smiles, num_atoms=mol.GetNumAtoms())

    try:
        result.fg = decompose_fg(mol)
    except Exception as e:
        logger.warning("FG decomposition failed for %s: %s", smiles, e)

    try:
        result.murcko = decompose_murcko(mol)
    except Exception as e:
        logger.warning("Murcko decomposition failed for %s: %s", smiles, e)

    try:
        result.brics = decompose_brics(mol)
    except Exception as e:
        logger.warning("BRICS decomposition failed for %s: %s", smiles, e)

    return result
