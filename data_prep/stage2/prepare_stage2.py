"""
Prepare Stage 2 data for the 8 MoleculeNet tasks.

Loads each task from DeepChem with a scaffold split, drops molecules whose
labels conflict across duplicate rows, and renders each record with a randomly
chosen instruction template.

Data sources:
- Property Prediction: MoleculeNet (BACE, BBBP, ClinTox, HIV, SIDER, ESOL, Lipo,
  Tox21) via DeepChem

The instruction templates in instruction_templates/ were written with
reference to SMolInstruct (osunlp/SMolInstruct, CC BY 4.0). See NOTICE.

Outputs:
- <output-dir>/train/{task}.json
- <output-dir>/valid/{task}.json
- <output-dir>/test/{task}.json
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

try:
    import deepchem as dc
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdAll')
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit(1)

random.seed(42)

CLASSIFICATION_SUFFIX = " Answer with only Yes or No."
REGRESSION_SUFFIX = " Return only the numerical value."

# Module-level paths, set from CLI args in main().
OUTPUT_DIR = None
TEMPLATE_DIR = None

# ============= SMILES PROCESSING =============

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form. Returns None if RDKit can't process it."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                _ = atom.GetAtomicNum()
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None

# ============= LOAD TEMPLATES =============

_TEMPLATE_CACHE = {}

def load_template(template_name):
    """Load template from instruction_templates folder."""
    if template_name not in _TEMPLATE_CACHE:
        filepath = TEMPLATE_DIR / f"{template_name}.json"
        with open(filepath, 'r') as f:
            _TEMPLATE_CACHE[template_name] = json.load(f)
    return _TEMPLATE_CACHE[template_name]

def get_property_templates(task_name, endpoint=None):
    """Get templates for property prediction tasks."""
    templates = load_template("property_prediction")

    if task_name in ["bbbp", "hiv", "esol", "lipo", "bace"]:
        return templates[task_name]
    elif task_name in ["clintox", "sider", "tox21"]:
        if endpoint is None:
            raise ValueError(f"endpoint required for multi-task dataset: {task_name}")
        return templates[task_name].get(endpoint, [])
    else:
        raise ValueError(f"Unknown task: {task_name}")

# ============= LABEL CONFLICT FILTER =============

def filter_label_conflicts_classification(samples, tasks, dataset_name):
    """Remove samples with conflicting labels and deduplicate for classification tasks."""
    from collections import defaultdict

    smiles_task_labels = defaultdict(list)
    for sample in samples:
        smiles = sample["smiles"]
        task_name = sample.get("task_name", tasks[0] if len(tasks) == 1 else None)
        label = sample["label"]
        smiles_task_labels[(smiles, task_name)].append(label)

    conflicting_keys = set()
    for key, labels in smiles_task_labels.items():
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            conflicting_keys.add(key)

    if conflicting_keys:
        print(f"    Removing {len(conflicting_keys)} label conflicts")

    seen_keys = set()
    filtered = []
    for s in samples:
        key = (s["smiles"], s.get("task_name", tasks[0] if len(tasks) == 1 else None))
        if key in conflicting_keys:
            continue
        dedup_key = (s["smiles"], s.get("task_name", tasks[0] if len(tasks) == 1 else None), s["label"])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        filtered.append(s)

    return filtered

def filter_label_conflicts_regression(samples):
    """Remove samples with conflicting values and deduplicate for regression tasks."""
    from collections import defaultdict

    smiles_values = defaultdict(set)
    for sample in samples:
        smiles = sample["smiles"]
        value = sample["value"]
        smiles_values[smiles].add(value)

    conflicting_smiles = set()
    for smiles, values in smiles_values.items():
        if len(values) > 1:
            conflicting_smiles.add(smiles)

    if conflicting_smiles:
        print(f"    Removing {len(conflicting_smiles)} value conflicts")

    seen_keys = set()
    filtered = []
    for s in samples:
        if s["smiles"] in conflicting_smiles:
            continue
        dedup_key = (s["smiles"], s["value"])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        filtered.append(s)

    return filtered

# ============= MOLECULENET LOADING =============

MOLECULENET_DATASETS = {
    "bace": {"loader": dc.molnet.load_bace_classification, "task_type": "classification"},
    "bbbp": {"loader": dc.molnet.load_bbbp, "task_type": "classification"},
    "clintox": {"loader": dc.molnet.load_clintox, "task_type": "classification"},
    "hiv": {"loader": dc.molnet.load_hiv, "task_type": "classification"},
    "sider": {"loader": dc.molnet.load_sider, "task_type": "classification"},
    "esol": {"loader": dc.molnet.load_delaney, "task_type": "regression"},
    "lipo": {"loader": dc.molnet.load_lipo, "task_type": "regression"},
    "tox21": {"loader": dc.molnet.load_tox21, "task_type": "classification"},
}

def load_moleculenet_dataset(name):
    """Load MoleculeNet dataset with scaffold split."""
    config = MOLECULENET_DATASETS[name]
    print(f"  Loading {name} from DeepChem...")

    tasks, datasets, _ = config["loader"](
        splitter='scaffold',
        featurizer=dc.feat.DummyFeaturizer(),
        reload=True
    )
    train_ds, valid_ds, test_ds = datasets

    print(f"    Tasks: {tasks}")
    print(f"    Raw counts - Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    return tasks, train_ds, valid_ds, test_ds, config

def process_moleculenet_classification(dataset, tasks, split_name, dataset_name):
    """Process classification dataset, filtering invalid SMILES."""
    samples = []
    skipped = 0

    for i in range(len(dataset)):
        smiles = dataset.ids[i]
        labels = dataset.y[i]
        weights = dataset.w[i] if hasattr(dataset, 'w') and dataset.w is not None else [1] * len(tasks)

        canonical = canonicalize_smiles(smiles)
        if not canonical:
            skipped += 1
            continue

        if len(tasks) == 1:
            label = labels[0] if hasattr(labels, '__len__') else labels
            weight = weights[0] if hasattr(weights, '__len__') else weights

            if weight == 0 or (hasattr(label, '__float__') and str(label) == 'nan'):
                continue

            samples.append({
                "smiles": canonical,
                "label": "Yes" if int(label) == 1 else "No",
                "split": split_name
            })
        else:
            for j, task in enumerate(tasks):
                label = labels[j]
                weight = weights[j]

                if weight == 0 or (hasattr(label, '__float__') and str(label) == 'nan'):
                    continue

                samples.append({
                    "smiles": canonical,
                    "task_name": task,
                    "label": "Yes" if int(label) == 1 else "No",
                    "split": split_name
                })

    if skipped > 0:
        print(f"    {split_name}: Skipped {skipped} invalid SMILES")

    return samples

def process_moleculenet_regression(dataset, tasks, split_name, dataset_name):
    """Process regression dataset, filtering invalid SMILES."""
    samples = []
    skipped = 0

    for i in range(len(dataset)):
        smiles = dataset.ids[i]
        values = dataset.y[i]

        canonical = canonicalize_smiles(smiles)
        if not canonical:
            skipped += 1
            continue

        value = values[0] if hasattr(values, '__len__') else values
        if str(value) == 'nan':
            continue

        samples.append({
            "smiles": canonical,
            "value": round(float(value), 3),
            "split": split_name
        })

    if skipped > 0:
        print(f"    {split_name}: Skipped {skipped} invalid SMILES")

    return samples

# ============= FORMAT WITH TEMPLATES =============

def format_simple_property(samples, task_name):
    """Format single-task property prediction (BBBP, HIV)."""
    templates = get_property_templates(task_name)
    formatted = []

    for sample in samples:
        template = random.choice(templates)
        # Remove <INPUT> placeholder from instruction, append task-type suffix
        instruction = template["input"].replace("\n<INPUT>", "") + CLASSIFICATION_SUFFIX
        output_text = template["output"].replace("<OUTPUT>", sample['label'])
        formatted.append({
            "instruction": instruction,
            "INPUT": sample['smiles'],
            "output": output_text,
        })
    return formatted

def format_clintox(samples):
    """Format ClinTox with endpoint-specific templates."""
    formatted = []

    for sample in samples:
        endpoint = sample["task_name"]
        templates = get_property_templates("clintox", endpoint)

        if not templates:
            continue

        template = random.choice(templates)
        instruction = template["input"].replace("\n<INPUT>", "") + CLASSIFICATION_SUFFIX
        output_text = template["output"].replace("<OUTPUT>", sample['label'])
        formatted.append({
            "instruction": instruction,
            "INPUT": sample['smiles'],
            "output": output_text,
        })
    return formatted

def format_sider(samples):
    """Format SIDER with target-specific templates."""
    formatted = []

    for sample in samples:
        target = sample["task_name"]
        templates = get_property_templates("sider", target)

        if not templates:
            continue

        template = random.choice(templates)
        instruction = template["input"].replace("\n<INPUT>", "") + CLASSIFICATION_SUFFIX
        output_text = template["output"].replace("<OUTPUT>", sample['label'])
        formatted.append({
            "instruction": instruction,
            "INPUT": sample['smiles'],
            "output": output_text,
        })
    return formatted

def format_tox21(samples):
    """Format Tox21 with assay-specific templates."""
    formatted = []

    for sample in samples:
        assay = sample["task_name"]
        templates = get_property_templates("tox21", assay)

        if not templates:
            continue

        template = random.choice(templates)
        instruction = template["input"].replace("\n<INPUT>", "") + CLASSIFICATION_SUFFIX
        output_text = template["output"].replace("<OUTPUT>", sample['label'])
        formatted.append({
            "instruction": instruction,
            "INPUT": sample['smiles'],
            "output": output_text,
        })
    return formatted

def format_regression(samples, task_name):
    """Format regression tasks (ESOL, Lipo)."""
    templates = get_property_templates(task_name)
    formatted = []

    for sample in samples:
        template = random.choice(templates)
        instruction = template["input"].replace("\n<INPUT>", "") + REGRESSION_SUFFIX
        output_text = template["output"].replace("<OUTPUT>", str(sample['value']))
        formatted.append({
            "instruction": instruction,
            "INPUT": sample['smiles'],
            "output": output_text,
        })
    return formatted

# ============= SAVE BY TASK =============

def save_task_data(task_data, split_name):
    """Save task data to split folder."""
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for task_name, samples in task_data.items():
        if not samples:
            continue

        # Convert task name to filename: "property_prediction-bbbp" -> "property_prediction_bbbp.json"
        filename = task_name.replace("-", "_") + ".json"
        filepath = split_dir / filename

        random.shuffle(samples)
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"    {split_name}/{filename}: {len(samples)} samples")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Stage 2 output directory (default: this script's directory).",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Instruction templates directory (default: <output-dir>/instruction_templates).",
    )
    args = parser.parse_args()

    global OUTPUT_DIR, TEMPLATE_DIR
    OUTPUT_DIR = args.output_dir
    TEMPLATE_DIR = args.template_dir or (OUTPUT_DIR / "instruction_templates")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create split directories
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "valid").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test").mkdir(parents=True, exist_ok=True)

    # Track data by task and split
    train_data = defaultdict(list)
    valid_data = defaultdict(list)
    test_data = defaultdict(list)

    print("Loading templates from instruction_templates/...")

    # ============= MoleculeNet =============
    print("\n" + "="*60)
    print("MOLECULENET DATASETS")
    print("="*60)

    for name in ["bace", "bbbp", "clintox", "hiv", "sider", "esol", "lipo", "tox21"]:
        print(f"\n--- {name.upper()} ---")
        tasks, train_ds, valid_ds, test_ds, config = load_moleculenet_dataset(name)

        # Process data
        if config["task_type"] == "classification":
            train_samples = process_moleculenet_classification(train_ds, tasks, "train", name)
            valid_samples = process_moleculenet_classification(valid_ds, tasks, "valid", name)
            test_samples = process_moleculenet_classification(test_ds, tasks, "test", name)

            all_samples = train_samples + valid_samples + test_samples
            all_filtered = filter_label_conflicts_classification(all_samples, tasks, name)
            train_samples = [s for s in all_filtered if s["split"] == "train"]
            valid_samples = [s for s in all_filtered if s["split"] == "valid"]
            test_samples = [s for s in all_filtered if s["split"] == "test"]
        else:
            train_samples = process_moleculenet_regression(train_ds, tasks, "train", name)
            valid_samples = process_moleculenet_regression(valid_ds, tasks, "valid", name)
            test_samples = process_moleculenet_regression(test_ds, tasks, "test", name)

            all_samples = train_samples + valid_samples + test_samples
            all_filtered = filter_label_conflicts_regression(all_samples)
            train_samples = [s for s in all_filtered if s["split"] == "train"]
            valid_samples = [s for s in all_filtered if s["split"] == "valid"]
            test_samples = [s for s in all_filtered if s["split"] == "test"]

        print(f"    After conflict filter - Train: {len(train_samples)}, Valid: {len(valid_samples)}, Test: {len(test_samples)}")

        # Format with templates
        task_name = f"property_prediction_{name}"

        if name == "bbbp":
            formatted_train = format_simple_property(train_samples, "bbbp")
            formatted_valid = format_simple_property(valid_samples, "bbbp")
            formatted_test = format_simple_property(test_samples, "bbbp")
        elif name == "clintox":
            formatted_train = format_clintox(train_samples)
            formatted_valid = format_clintox(valid_samples)
            formatted_test = format_clintox(test_samples)
        elif name == "hiv":
            formatted_train = format_simple_property(train_samples, "hiv")
            formatted_valid = format_simple_property(valid_samples, "hiv")
            formatted_test = format_simple_property(test_samples, "hiv")
        elif name == "sider":
            formatted_train = format_sider(train_samples)
            formatted_valid = format_sider(valid_samples)
            formatted_test = format_sider(test_samples)
        elif name == "esol":
            formatted_train = format_regression(train_samples, "esol")
            formatted_valid = format_regression(valid_samples, "esol")
            formatted_test = format_regression(test_samples, "esol")
        elif name == "lipo":
            formatted_train = format_regression(train_samples, "lipo")
            formatted_valid = format_regression(valid_samples, "lipo")
            formatted_test = format_regression(test_samples, "lipo")
        elif name == "tox21":
            formatted_train = format_tox21(train_samples)
            formatted_valid = format_tox21(valid_samples)
            formatted_test = format_tox21(test_samples)
        elif name == "bace":
            formatted_train = format_simple_property(train_samples, "bace")
            formatted_valid = format_simple_property(valid_samples, "bace")
            formatted_test = format_simple_property(test_samples, "bace")

        train_data[task_name].extend(formatted_train)
        valid_data[task_name].extend(formatted_valid)
        test_data[task_name].extend(formatted_test)

    # ============= Save =============
    print("\n" + "="*60)
    print("SAVING")
    print("="*60)

    print("\n  Train:")
    save_task_data(train_data, "train")

    print("\n  Valid:")
    save_task_data(valid_data, "valid")

    print("\n  Test:")
    save_task_data(test_data, "test")

    total_train = sum(len(samples) for samples in train_data.values())
    total_valid = sum(len(samples) for samples in valid_data.values())
    total_test = sum(len(samples) for samples in test_data.values())

    print("\n=== SUMMARY ===")
    print(f"Total train: {total_train}")
    print(f"Total valid: {total_valid}")
    print(f"Total test: {total_test}")

if __name__ == "__main__":
    main()
