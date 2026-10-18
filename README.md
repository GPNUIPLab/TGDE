# TGDE

## 目录

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
├── dataset/                     # representation
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

## 环境准备

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
# Note: The torch_geometric extensions (pyg_lib, torch_scatter, torch_sparse) must exactly match the torch+CUDA version
# On Windows, use code/requirements_windows.txt
```

## 运行方案 A

```bash
cd TGDE
bash run_a.sh BM_llm 0        # The number after the dataset name is the GPU index; the same applies to the other three datasets:
# bash run_a.sh BM_rake 0     # BM·RAKE
# bash run_a.sh IT_llm 0      # IT·LLM
# bash run_a.sh IT_rake 0     # IT·RAKE
```

