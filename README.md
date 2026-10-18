# TGDE

## Contents

```
TGDE/
├── code/                        # Complete code for Scheme A
│   ├── patent_concept/      # Main package (GNN training/evaluation entry: model/gnn/train_pyg.py)
│   ├── BM/pipeline/             # BM
│   │   ├── run_experiment.py / evaluate_gnn.py
│   │   └── experiments/fusion_c23_out128.yaml        # llm
│   │   └── experiments/fusion_c23_rake_out128.yaml   # rake
│   ├── IT/pipeline/             # IT
│   ├── requirements*.txt, pyproject.toml
├── dataset/                     # Not included in this repository (hosted on Google Drive)
│   ├── BM_llm/  BM_rake/        # BM × llm / rake
│   ├── IT_llm/  IT_rake/        # IT × llm / rake
│   │   └── Each directory has 4 files:
│   │       edges.pkl                # Graph edge set (v1, v2, day_offset) np.array
│   │       data.pkl                 # Classification samples: train/val/test node pairs + labels
│   │       concept_embs_train.pkl.gz  # 768D concept embeddings for nodes in the training period {"v_features": array}
│   │       concept_embs_test.pkl.gz   # 768D concept embeddings for nodes in the test period
├── link_datasets.sh             # Script linking data → code directory (idempotent, no code changes)
└── run_a.sh                     # One-click run for Scheme A
```

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
# Note: The torch_geometric extensions (pyg_lib, torch_scatter, torch_sparse) must exactly match the torch+CUDA version
# On Windows, use code/requirements_windows.txt
```

## Dataset Download

The dataset is relatively large, so it is hosted on Google Drive instead of being included in this repository.

Please download the dataset from the Google Drive link [], extract it, and place the four dataset directories (`BM_llm/`, `BM_rake/`, `IT_llm/`, `IT_rake/`) under `TGDE/dataset/`.

After that, run `link_datasets.sh` to link the downloaded data into the code directory. The script is idempotent and requires no code changes.

Example:

```bash
cd TGDE
# After downloading and extracting the dataset, the directory structure should be:
# TGDE/dataset/BM_llm
# TGDE/dataset/BM_rake
# TGDE/dataset/IT_llm
# TGDE/dataset/IT_rake
bash link_datasets.sh
```

## Run Scheme A

```bash
cd TGDE
bash link_datasets.sh         # Run this first after downloading the dataset
bash run_a.sh BM_llm 0        # The number after the dataset name is the GPU index; the same applies to the other three datasets:
# bash run_a.sh BM_rake 0     # BM·RAKE
# bash run_a.sh IT_llm 0      # IT·LLM
# bash run_a.sh IT_rake 0     # IT·RAKE
```

