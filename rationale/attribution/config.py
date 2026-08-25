"""Paths, per-task metadata, and constants for the SME attribution pipeline.

All paths default to the package layout and can be overridden with environment
variables, so the pipeline can run against externally stored datasets and
checkpoints without editing code.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # rationale/attribution
_RATIONALE = os.path.dirname(_HERE)                   # rationale/

# Mole-BERT model code, pretrained GIN init, and processed datasets.
MOLEBERT_DIR = os.environ.get("MOLEBERT_DIR", os.path.join(_RATIONALE, "molebert"))
# Per-task GNN inputs (raw.csv / split.csv), produced by data_prep.
DATA_DIR = os.environ.get("ATTRIBUTION_DATA_DIR", os.path.join(_RATIONALE, "gnn_data"))
# Finetuned Mole-BERT checkpoints: val-best seed_{s}.pth.
RESULT_DIR = os.environ.get("ATTRIBUTION_RESULT_DIR", os.path.join(_HERE, "result"))
# Attribution outputs (decompositions / predictions / attributions parquet).
OUTPUT_DIR = os.environ.get("ATTRIBUTION_OUTPUT_DIR", os.path.join(_HERE, "output"))
# Pretrained Mole-BERT GIN weights used to initialize finetuning.
PRETRAIN_GIN_CKPT = os.environ.get(
    "MOLEBERT_PRETRAIN_CKPT", os.path.join(MOLEBERT_DIR, "model_gin", "Mole-BERT.pth")
)

SEEDS = [0, 1, 2]

# Default finetune hyperparameters.
FINETUNE_DEFAULTS = dict(
    epochs=100,
    batch_size=32,
    lr=1.0e-3,
    lr_scale=1.0,
    weight_decay=0.0,
    drop_ratio=0.5,
)

# Model hyperparameters (same for all checkpoints).
NUM_LAYER = 5
EMB_DIM = 300
JK = "last"
GRAPH_POOLING = "mean"
GNN_TYPE = "gin"
DROP_RATIO = 0.5

TASK_CONFIG = {
    "bace": {
        "num_tasks": 1,
        "task_type": "classification",
        "label_cols": ["Class"],
    },
    "bbbp": {
        "num_tasks": 1,
        "task_type": "classification",
        "label_cols": ["p_np"],
    },
    "clintox": {
        "num_tasks": 2,
        "task_type": "classification",
        "label_cols": ["FDA_APPROVED", "CT_TOX"],
    },
    "hiv": {
        "num_tasks": 1,
        "task_type": "classification",
        "label_cols": ["HIV_active"],
    },
    "sider": {
        "num_tasks": 27,
        "task_type": "classification",
        "label_cols": [
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications",
        ],
    },
    "tox21": {
        "num_tasks": 12,
        "task_type": "classification",
        "label_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
    },
    "esol": {
        "num_tasks": 1,
        "task_type": "regression",
        "label_cols": ["measured log solubility in mols per litre"],
    },
    "lipo": {
        "num_tasks": 1,
        "task_type": "regression",
        "label_cols": ["exp"],
    },
}
