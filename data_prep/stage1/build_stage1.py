"""Build the Stage 1 molecule-text alignment corpus (89,919 pairs).

Merges four public molecule-description corpora in one pass, ChEBI-20 first so
its curated descriptions win over PubChem datasets. Deduplicates on canonical
SMILES and on description text, and drops every molecule that also appears in a
Stage 2 property-prediction task. ChEBI-20's test split is held out.

Raw sources are downloaded separately into `raw/`:

  raw/ChEBI-20/{train,validation,test}.txt   ChEBI-20_data/ from the MolT5 repo
                                             (github.com/blender-nlp/MolT5);
                                             `CID<TAB>SMILES<TAB>description`
  raw/PubChem324kV2/                         HF dataset `acharkq/PubChem324kV2`,
                                             PubChem324kV2.zip -> the inner dir
                                             with train/valid/test/pretrain .pt
  raw/DrugBank/                              HF dataset `chao1224/MoleculeSTM`,
                                             DrugBank_data/raw/SMILES_description_full.txt
                                             and SMILES_pharmacodynamics_full.txt
                                             (the copies under raw/, not index/)
  raw/molecular_description_generation.json  HF dataset `zjunlp/Mol-Instructions`,
                                             data/Molecule-oriented_Instructions.zip

Build the Stage 2 data first — the leak filter reads `../stage2/{train,valid,test}/`.

Run:
  uv run python ../stage2/prepare_stage2.py
  uv run python build_stage1.py --out train.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import selfies as sf
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import InMemoryDataset

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MIN_LEN = 30
INSTRUCTIONS = [
    "Could you give me a brief overview of this molecule?",
    "What can you tell me about this molecule?",
    "Please give me some details about this molecule.",
    "Provide a description of this molecule.",
    "Describe this molecule.",
    "Provide a brief overview of this molecule.",
    "Could you provide a description of this molecule?",
]


class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.slices = torch.load(path, weights_only=False)

    def __getitem__(self, idx):
        return self.get(idx)


def canon(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?]\s+[A-Z]|[.!?]$", text))


def iter_chebi(chebi_dir: Path, splits):
    """Yield (smiles, description) from ChEBI-20 .txt files (CID\\tSMILES\\tdescription)."""
    for split in splits:
        path = chebi_dir / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            next(f)  # header: CID  SMILES  description
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                yield parts[1], parts[2]


def load_pubchem_pt(path: Path):
    ds = PubChemDataset(str(path))
    for i in range(len(ds)):
        d = ds[i]
        yield str(d.smiles), str(d.text)


def load_drugbank(path: Path):
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            smi, text = line.split("\t", 1)
            yield smi.strip(), text.strip()


def load_mol_instructions(path: Path, split="train"):
    """Mol-Instructions description task: SELFIES -> SMILES, keep one split."""
    for item in json.load(path.open()):
        if item.get("metadata", {}).get("split") != split:
            continue
        smiles = sf.decoder(item["input"])
        if not smiles:
            continue
        yield smiles, item["output"]


def load_property_leak_smiles(stage2_dir: Path) -> tuple[set[str], list[str]]:
    """Canonical SMILES of every Stage 2 property-prediction task (all splits)."""
    smiles = set()
    tasks = set()
    for split in ("train", "valid", "test"):
        for path in sorted((stage2_dir / split).glob("property_prediction_*.json")):
            tasks.add(path.stem.replace("property_prediction_", ""))
            for e in json.load(path.open()):
                c = canon(e["INPUT"])
                if c:
                    smiles.add(c)
    if not tasks:
        raise FileNotFoundError(f"No property_prediction_*.json under {stage2_dir}")
    return smiles, sorted(tasks)


def build(args):
    pubchem_dir = Path(args.pubchem_dir)
    drugbank_dir = Path(args.drugbank_dir)
    mol_instructions = Path(args.mol_instructions)
    chebi_dir = Path(args.chebi_dir)
    stage2_dir = Path(args.stage2_dir)
    out_json = Path(args.out)

    # ChEBI-20 test is held out (never enters Stage 1).
    chebi_test = {c for c in (canon(s) for s, _ in iter_chebi(chebi_dir, ("test",))) if c}
    print(f"ChEBI-20 test SMILES held out: {len(chebi_test)}")

    # ---- (1) single-pass refinement, ChEBI first -------------------------
    plan = [
        ("chebi", lambda: iter_chebi(chebi_dir, ("train", "validation"))),
        ("curated_train", lambda: load_pubchem_pt(pubchem_dir / "train.pt")),
        ("curated_valid", lambda: load_pubchem_pt(pubchem_dir / "valid.pt")),
        ("curated_test", lambda: load_pubchem_pt(pubchem_dir / "test.pt")),
        ("drugbank_description", lambda: load_drugbank(drugbank_dir / "SMILES_description_full.txt")),
        ("drugbank_pharmacodynamics", lambda: load_drugbank(drugbank_dir / "SMILES_pharmacodynamics_full.txt")),
        ("pubchem_pretrain", lambda: load_pubchem_pt(pubchem_dir / "pretrain.pt")),
        ("mol_instructions", lambda: load_mol_instructions(mol_instructions)),
    ]

    merged: dict[str, tuple[str, str, str]] = {}  # canonical_smi -> (source, smiles, text)
    seen_texts: set[str] = set()

    for source_name, make_iter in plan:
        n_raw = n_invalid = n_short = n_heldout = n_smi_dup = n_text_dup = n_added = 0
        for smi, text in make_iter():
            n_raw += 1
            text = text.strip()
            if not text:
                continue
            if len(text) < MIN_LEN:
                n_short += 1
                continue
            c = canon(smi)
            if c is None:
                n_invalid += 1
                continue
            if c in chebi_test:
                n_heldout += 1
                continue
            if c in merged:
                n_smi_dup += 1
                continue
            key = text.lower()
            if key in seen_texts:
                n_text_dup += 1
                continue
            merged[c] = (source_name, smi, text)
            seen_texts.add(key)
            n_added += 1

        print(f"  {source_name}: raw={n_raw} added={n_added} short={n_short} "
              f"heldout={n_heldout} smi_dup={n_smi_dup} text_dup={n_text_dup}")

    n_after_refine = len(merged)

    # ---- (2) remove Stage 2 property-prediction leak ---------------------
    leak_smiles, leak_tasks = load_property_leak_smiles(stage2_dir)
    leaked = [c for c in merged if c in leak_smiles]
    for c in leaked:
        del merged[c]
    n_final = len(merged)
    print(f"  property leak removed={len(leaked)} (tasks: {', '.join(leak_tasks)})")

    # ---- emit ------------------------------------------------------------
    entries = [
        {"instruction": INSTRUCTIONS[i % len(INSTRUCTIONS)], "INPUT": smi, "output": text}
        for i, (_, smi, text) in enumerate(merged.values())
    ]
    lengths = [len(e["output"]) for e in entries]
    multi = sum(1 for e in entries if count_sentences(e["output"]) >= 2)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(entries, out_json.open("w"), indent=2, ensure_ascii=False)

    print()
    print(f"refine={n_after_refine} -> -leak={n_final}")
    print(f"length: min={min(lengths)} median={sorted(lengths)[len(lengths) // 2]} "
          f"mean={round(sum(lengths) / len(lengths), 1)} max={max(lengths)}")
    print(f"multi-sentence: {multi} ({round(100 * multi / len(entries), 1)}%)")
    print(f"source distribution: {dict(Counter(v[0] for v in merged.values()))}")
    print(f"Wrote {len(entries)} entries to {out_json}")

    if args.expect is not None and len(entries) != args.expect:
        raise SystemExit(f"Expected {args.expect} entries but built {len(entries)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pubchem-dir", default=str(HERE / "raw" / "PubChem324kV2"),
                   help="PubChem324kV2 dir with train/valid/test/pretrain .pt")
    p.add_argument("--drugbank-dir", default=str(HERE / "raw" / "DrugBank"),
                   help="dir with SMILES_description_full.txt and SMILES_pharmacodynamics_full.txt")
    p.add_argument("--mol-instructions", default=str(HERE / "raw" / "molecular_description_generation.json"),
                   help="Mol-Instructions molecular_description_generation.json (SELFIES, decoded here)")
    p.add_argument("--chebi-dir", default=str(HERE / "raw" / "ChEBI-20"),
                   help="ChEBI-20 dir with train.txt / validation.txt / test.txt")
    p.add_argument("--stage2-dir", default=str(HERE.parent / "stage2"),
                   help="Stage 2 dir with {train,valid,test}/property_prediction_*.json (leak filter)")
    p.add_argument("--out", default=str(HERE / "train.json"))
    p.add_argument("--expect", type=int, default=None, help="fail if the final count differs")
    build(p.parse_args())


if __name__ == "__main__":
    main()
