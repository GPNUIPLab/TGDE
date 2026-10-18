import gzip
import logging
import math
import os
import pickle
import platform
import random
import shlex
import sys
from importlib import reload
from typing import Any

# Windows CUDA + pin_memory + num_workers>0 causes "resource already mapped" crashes
_DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else 8

import fire
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from patent_concept.model.graph import Graph
from patent_concept.model.metrics import print_metrics, test


def _default_wandb_run_name(
    *,
    features_path: str,
    year_start_train: int,
    seed: int,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    concept_embs_as_features: bool = False,
) -> str:
    # Keep names compact but informative; full details live in wandb.config.
    if concept_embs_as_features:
        type_ = "emb"
    elif "binary" in features_path:
        type_ = "base"
    else:
        type_ = "emb"
    bs = int(train_cfg.get("batch_size", 0) or 0)
    lr = float(train_cfg.get("lr", 0.0) or 0.0)
    wd = float(train_cfg.get("weight_decay", 0.0) or 0.0)
    ep = int(train_cfg.get("num_epochs", 0) or 0)
    h = int(model_cfg.get("hidden_dim", 0) or 0)
    o = int(model_cfg.get("out_dim", 0) or 0)
    dec = str(model_cfg.get("decoder", ""))
    f1 = int(sampling_cfg.get("fanout1", 0) or 0)
    f2 = int(sampling_cfg.get("fanout2", 0) or 0)
    return f"{type_}_{year_start_train}_bs{bs}_lr{lr:g}_wd{wd:g}_ep{ep}_h{h}_o{o}_f{f1}-{f2}_{dec}_s{seed}"


def _maybe_init_wandb(
    wandb_cfg: dict[str, Any],
    config: dict[str, Any],
    logger: logging.Logger,
):
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "W&B logging requested but 'wandb' is not installed.\n"
            "Install with: python -m pip install wandb\n"
            "Or disable with: --wandb=enabled=false"
        ) from e

    tags_raw = str(wandb_cfg.get("tags", "")).strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None
    mode = str(wandb_cfg.get("mode", "online")).strip() or "online"
    fail_fast = bool(wandb_cfg.get("fail_fast", False))

    init_kwargs: dict[str, Any] = {
        "project": str(wandb_cfg.get("project", "mat-concepts")),
        "entity": str(wandb_cfg.get("entity", "")) or None,
        "name": str(wandb_cfg.get("name", "")) or None,
        "group": str(wandb_cfg.get("group", "")) or None,
        "job_type": str(wandb_cfg.get("job_type", "train")) or None,
        "tags": tags,
        "mode": mode,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        # Common on clusters: no internet / blocked outbound, or permission issues.
        # Keep training running unless explicitly asked to fail-fast.
        logger.error(
            "W&B init failed (project=%s, entity=%s, mode=%s): %s: %s",
            init_kwargs.get("project"),
            init_kwargs.get("entity"),
            init_kwargs.get("mode"),
            type(e).__name__,
            e,
        )
        if fail_fast:
            raise

        if str(init_kwargs.get("mode", "")) != "offline":
            try:
                logger.info("Retrying W&B init in offline mode")
                init_kwargs["mode"] = "offline"
                run = wandb.init(**init_kwargs)
            except Exception as e2:
                logger.error(
                    "W&B offline init also failed: %s: %s", type(e2).__name__, e2
                )
                return None
        else:
            return None

    wandb.config.update(config, allow_val_change=True)
    logger.info(
        "W&B enabled: project=%s, entity=%s, name=%s, mode=%s",
        init_kwargs.get("project"),
        init_kwargs.get("entity"),
        init_kwargs.get("name"),
        init_kwargs.get("mode"),
    )

    return run


def _require_pyg():
    try:
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.loader import LinkNeighborLoader  # noqa: F401
        from torch_geometric.nn import SAGEConv  # noqa: F401
        # LinkNeighborLoader -> NeighborSampler requires either pyg-lib or torch-sparse.
        # (Otherwise it will crash inside DataLoader workers.)
        try:
            import pyg_lib  # type: ignore  # noqa: F401

            return True
        except Exception:
            try:
                import torch_sparse  # type: ignore  # noqa: F401

                return True
            except Exception:
                import torch

                torch_ver = torch.__version__
                # torch.__version__ is like: 2.0.0+cu117
                wheel_tag = torch_ver
                raise RuntimeError(
                    "PyTorch Geometric neighbor sampling backend missing.\n"
                    "LinkNeighborLoader requires either 'pyg-lib' or 'torch-sparse'.\n\n"
                    "Fix (recommended): install the official PyG wheels that match your torch build:\n"
                        f"  python -m pip install pyg-lib torch-sparse torch-scatter \\\n    -f https://data.pyg.org/whl/torch-{wheel_tag}.html\n\n"
                    "If you're on a CPU-only machine, use the '+cpu' wheel tag instead.\n"
                    f"Detected torch: {torch_ver}"
                )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch Geometric is required for train_pyg.py.\n"
            "Install it matching your torch/cuda build (see https://pytorch-geometric.readthedocs.io).\n"
            f"Original import error: {type(e).__name__}: {e}"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(file, level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_compressed(path: str | None):
    if not path:
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _to_numpy_2d_float32(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _try_build_matrix_from_id_dict(obj: dict[Any, Any]) -> np.ndarray | None:
    if not obj:
        return None

    keys_int: list[int] = []
    values: list[np.ndarray] = []
    for k, v in obj.items():
        try:
            k_int = int(k)
        except Exception:
            return None
        keys_int.append(k_int)
        values.append(_to_numpy_2d_float32(v).reshape(-1))

    if not values:
        return None

    d = int(values[0].shape[0])
    if any(int(v.shape[0]) != d for v in values):
        return None

    n = int(max(keys_int)) + 1
    mat = np.zeros((n, d), dtype=np.float32)
    for k_int, v in zip(keys_int, values, strict=False):
        if k_int < 0:
            continue
        mat[k_int] = v
    return mat


def load_node_feature_matrix(path: str, *, name: str, logger: logging.Logger) -> np.ndarray:
    obj = load_compressed(path)
    if obj is None:
        raise ValueError(f"{name}: failed to load features from {path} (got None)")

    if isinstance(obj, dict):
        if "v_features" in obj:
            logger.info("%s: loaded from key 'v_features' (%s)", name, path)
            return _to_numpy_2d_float32(obj["v_features"])

        for k in ("embeddings", "embs", "x", "features"):
            if k in obj:
                logger.info("%s: loaded from key '%s' (%s)", name, k, path)
                return _to_numpy_2d_float32(obj[k])

        mat = _try_build_matrix_from_id_dict(obj)
        if mat is not None:
            logger.info(
                "%s: loaded from id->vector dict (%s) | shape=%s",
                name,
                path,
                tuple(mat.shape),
            )
            return mat

        raise ValueError(
            f"{name}: unsupported feature file format at {path}. "
            f"Expected key 'v_features' (or one of embeddings/embs/x/features) or an id->vector dict. "
            f"Keys found: {sorted(map(str, obj.keys()))[:50]}"
        )

    logger.info("%s: loaded from raw array (%s)", name, path)
    return _to_numpy_2d_float32(obj)


def _parse_config_str(value: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    if not value:
        return out

    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid config item '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in out:
            raise ValueError(f"Unknown config key '{k}'. Allowed: {sorted(out.keys())}")

        default_val = out[k]
        if isinstance(default_val, bool):
            out[k] = v.lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            out[k] = int(float(v))
        elif isinstance(default_val, float):
            out[k] = float(v)
        else:
            out[k] = v

    return out


def blend_grid_best(
    y_true: np.ndarray,
    preds_graph: np.ndarray,
    preds_text: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Paper-protocol mixture: 11-point weight grid on probabilities.

    Mirrors patent_concept/model/mixture/blend.py: for w in linspace(0,1,11)
    evaluate AUC of w*preds_graph + (1-w)*preds_text and return the best
    (auc, w, blended_predictions).
    """
    best_auc = float("-inf")
    best_w = 0.5
    best_preds = preds_graph
    for w1 in np.linspace(0.0, 1.0, 11):
        blended = preds_graph * float(w1) + preds_text * (1.0 - float(w1))
        auc, _, _ = test(
            torch.tensor(y_true, dtype=torch.float32), blended, threshold=0.5
        )
        if float(auc) > best_auc:
            best_auc = float(auc)
            best_w = float(w1)
            best_preds = blended
    return best_auc, best_w, best_preds


def build_edge_index_for_year(graph: Graph, year: int, num_nodes: int) -> torch.Tensor:
    # edges: (m, 3) with columns [u, v, day_offset]
    edges = graph.get_until_year(year)
    uv = edges[:, :2].astype(np.int64, copy=False)
    # undirected message passing
    uv_rev = uv[:, [1, 0]]
    uv2 = np.vstack([uv, uv_rev])
    edge_index = torch.from_numpy(uv2.T).contiguous()

    # Defensive checks
    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError(
                f"edge_index has nodes outside [0,{num_nodes-1}]: "
                f"min={int(edge_index.min())}, max={int(edge_index.max())}"
            )

    return edge_index


class SAGEEncoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        num_layers: int = 2,
        use_norm: bool = False,
        use_residual: bool = False,
        conv: str = "sage",
        activation: str = "relu",
    ):
        super().__init__()
        _require_pyg()
        from torch_geometric.nn import SAGEConv

        self.num_layers = num_layers
        self.use_norm = use_norm
        self.use_residual = use_residual
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        if conv == "gat":
            from torch_geometric.nn import GATConv

            self.convs = nn.ModuleList(
                [GATConv(dims[i], dims[i + 1]) for i in range(num_layers)]
            )
        else:
            self.convs = nn.ModuleList(
                [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
            )
        if use_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(dims[i + 1]) for i in range(num_layers - 1)]
            )
        self.act = nn.GELU() if str(activation).lower() == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        if use_residual:
            self.residual_projs = nn.ModuleList()
            for i in range(num_layers - 1):
                if dims[i] != dims[i + 1]:
                    self.residual_projs.append(nn.Linear(dims[i], dims[i + 1]))
                else:
                    self.residual_projs.append(nn.Identity())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                if self.use_norm:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
                if self.use_residual:
                    x = x + self.residual_projs[i](residual)
        return x


class EdgeMLPDecoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        _x_u: torch.Tensor | None = None,
        _x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([z_u, z_v, torch.abs(z_u - z_v), z_u * z_v], dim=1)
        return self.mlp(x).squeeze(1)


class MLPNoInteractionDecoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        _x_u: torch.Tensor | None = None,
        _x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([z_u, z_v], dim=1)
        return self.mlp(x).squeeze(1)


class MLPDiffDecoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        _x_u: torch.Tensor | None = None,
        _x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.abs(z_u - z_v)
        return self.mlp(x).squeeze(1)


class MLPProductDecoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        _x_u: torch.Tensor | None = None,
        _x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = z_u * z_v
        return self.mlp(x).squeeze(1)


class EdgeMLPCatRawDecoder(nn.Module):
    """Decoder that concatenates graph-encoded embeddings with raw node features.

    Input per edge: [z_u, z_v, x_u, x_v] where z = encoder output and
    x = raw (normalized) node features fed to the encoder.
    """

    def __init__(self, emb_dim: int, raw_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + raw_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        x_u: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_u is None or x_v is None:
            raise ValueError(
                "EdgeMLPCatRawDecoder requires raw node features (x_u, x_v)"
            )
        x = torch.cat([z_u, z_v, x_u, x_v], dim=1)
        return self.mlp(x).squeeze(1)


class EdgeTwoTowerDecoder(nn.Module):
    """Two-tower decoder: each view gets its own nonlinear tower before fusion.

    Text tower: MLP over [x_u, x_v]; graph tower: MLP over [z_u, z_v];
    final logit = Linear([t, g]).

    With aux=True, each tower also gets its own linear head so it can be
    supervised directly by an auxiliary BCE loss. With aux_learnable=True the
    aux loss uses a learnable convex pair w_graph=sigmoid(aux_logit),
    w_text=1-w_graph (init 0.5/0.5); otherwise fixed weights are used by the
    caller (model_cfg aux_w_graph / aux_w_text).
    """

    def __init__(  # noqa: PLR0913
        self,
        emb_dim: int,
        raw_dim: int,
        hidden_dim: int,
        dropout: float,
        aux: bool = False,
        aux_learnable: bool = False,
        drop_path: float = 0.0,
        activation: str = "relu",
        fusion_head: str = "linear",
    ):
        super().__init__()

        def _act() -> nn.Module:
            return nn.GELU() if str(activation).lower() == "gelu" else nn.ReLU()

        self.text_tower = nn.Sequential(
            nn.Linear(raw_dim * 2, hidden_dim),
            _act(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_tower = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            _act(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if str(fusion_head).lower() == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                _act(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(hidden_dim * 2, 1)
        self.aux = bool(aux)
        self.drop_path = float(drop_path)
        self.aux_logit: nn.Parameter | None = None
        if self.aux:
            self.graph_head = nn.Linear(hidden_dim, 1)
            self.text_head = nn.Linear(hidden_dim, 1)
            if aux_learnable:
                self.aux_logit = nn.Parameter(torch.zeros(1))

    def _drop_path(self, h: torch.Tensor) -> torch.Tensor:
        if not (self.drop_path > 0.0 and self.training):
            return h
        keep = torch.empty(h.shape[0], 1, device=h.device).bernoulli_(
            1.0 - self.drop_path
        )
        return h * keep / (1.0 - self.drop_path)

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        x_u: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_u is None or x_v is None:
            raise ValueError(
                "EdgeTwoTowerDecoder requires raw node features (x_u, x_v)"
            )
        if return_aux and not self.aux:
            raise ValueError("EdgeTwoTowerDecoder return_aux=True requires aux=True")
        t = self._drop_path(self.text_tower(torch.cat([x_u, x_v], dim=1)))
        g = self._drop_path(self.graph_tower(torch.cat([z_u, z_v], dim=1)))
        joint = self.head(torch.cat([t, g], dim=1)).squeeze(1)
        if return_aux:
            return (
                joint,
                self.graph_head(g).squeeze(1),
                self.text_head(t).squeeze(1),
            )
        return joint


class EdgeSumFusionDecoder(nn.Module):
    """Projected-sum fusion: text features aligned into the embedding space.

    f = z + Linear(x); logit = MLP([f_u, f_v]). Low-capacity alternative to
    the raw concat decoder: the text contribution shares the graph weights.
    """

    def __init__(self, emb_dim: int, raw_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(raw_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        x_u: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_u is None or x_v is None:
            raise ValueError(
                "EdgeSumFusionDecoder requires raw node features (x_u, x_v)"
            )
        f_u = z_u + self.proj(x_u)
        f_v = z_v + self.proj(x_v)
        return self.mlp(torch.cat([f_u, f_v], dim=1)).squeeze(1)


class DotDecoder(nn.Module):
    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        _x_u: torch.Tensor | None = None,
        _x_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (z_u * z_v).sum(dim=1)


class GradProbe:
    """Breakpoint monitor: encoder grads + graph representation z per optimizer step.

    Enabled by env ``GRAD_PROBE_INTERVAL`` (0 = off). Every step it verifies
    (1) all encoder grads are finite and nonzero, (2) z of a fixed probe
    subgraph differs from the previous step. A ``PROBE`` log line is emitted
    every ``interval`` steps; counters accumulate across the whole run.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device,
        interval: int,
        n_probe: int = 600,
        max_edges: int = 5000,
    ) -> None:
        n = min(int(n_probe), int(x.size(0)))
        mask = (edge_index[0] < n) & (edge_index[1] < n)
        probe_ei = edge_index[:, mask]
        if probe_ei.size(1) > max_edges:
            probe_ei = probe_ei[:, :max_edges]
        self.x = x[:n].detach().to(device)
        self.ei = probe_ei.detach().to(device).contiguous()
        self.interval = max(int(interval), 1)
        self.step = 0
        self.grad_checks = 0
        self.grad_ok = 0
        self.dz_checks = 0
        self.dz_ok = 0
        self.z_prev: torch.Tensor | None = None

    def after_step(self, encoder: nn.Module, logger: logging.Logger) -> None:
        self.step += 1
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        if grads:
            norms = torch.stack([g.norm() for g in grads])
            bad = int(((norms <= 0) | ~torch.isfinite(norms)).sum().item())
            self.grad_checks += 1
            if bad == 0:
                self.grad_ok += 1
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            z_now = encoder(self.x, self.ei)
        if was_training:
            encoder.train()
        dz = 0.0
        if self.z_prev is None:
            self.z_prev = z_now
        else:
            dz = float((z_now - self.z_prev).abs().sum().item())
            self.dz_checks += 1
            if dz > 0.0:
                self.dz_ok += 1
            if self.step % self.interval == 0:
                rel = dz / (float(self.z_prev.norm().item()) + 1e-12)
                logger.info(
                    "PROBE step=%d grad_ok=%d/%d dz_changed=%d/%d"
                    " last_dz=%.6f rel=%.6f",
                    self.step,
                    self.grad_ok,
                    self.grad_checks,
                    self.dz_ok,
                    self.dz_checks,
                    dz,
                    rel,
                )
            self.z_prev = z_now


def main(
    graph_path="data-v2/graph/edges.M.pkl",
    data_path="data-v2/model/data.M.pkl",
    v_features_path="data-v2/model/baseline/features.2016.binary.M.pkl.gz",
    year_start_train=2016,
    train=None,
    model=None,
    sampling=None,
    features=None,
    wandb=None,
    ood_data_path=None,
    ood_year_start=None,
    ood_eval_interval=None,
    ood_eval_batch_size=None,
    ood_features_path=None,
    concept_embs_as_features=False,
    seed=42,
    log_file="logs-v2/gnn/gnn_train_pyg.log",
    save_model_path=None,
    wandb_save_path=False,
):
    """Fast GNN trainer using PyTorch Geometric neighbor sampling.

    This is intended to be a drop-in faster alternative to train.py:
    - uses LinkNeighborLoader for neighbor sampling in C++/CUDA friendly pipeline
    - uses SAGEConv layers
    - supports MLP decoder for link prediction
    """

    _require_pyg()
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    reload(logging)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_cfg = _parse_config_str(
        train,
        defaults={
            "batch_size": 4096,
            "pos_ratio": 0.3,
            "num_epochs": 10,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "log_interval": 1,
            "eval_batch_size": 16384,
            "ood_eval_interval": 0,
            "num_workers": _DEFAULT_NUM_WORKERS,
            "amp": True,
            "grad_clip_norm": 1.0,
            "scheduler": "none",
            "scheduler_patience": 10,
            "scheduler_T_max": 0,
            "scheduler_warmup": 5,
            "scheduler_T0": 30,
            "early_stopping_patience": 0,
            "optimizer": "adam",
            "drop_edge": 0.0,
        },
    )
    model_cfg = _parse_config_str(
        model,
        defaults={
            "hidden_dim": 128,
            "out_dim": 128,
            "dropout": 0.1,
            "decoder": "mlp",
            "decoder_hidden_dim": 256,
            "decoder_dropout": 0.1,
            "num_layers": 2,
            "use_norm": False,
            "use_residual": False,
            "decoder_aux": False,
            "decoder_raw_input": False,
            "aux_w_graph": 0.9,
            "aux_w_text": 0.1,
            "aux_lambda": 1.0,
            "aux_learnable": False,
            "aux_joint_loss": True,
            "infer_aux_mix": False,
            "infer_aux_grid": False,
            "conv": "sage",
            "decoder_drop_path": 0.0,
            "activation": "relu",
            "decoder_fusion_head": "linear",
        },
    )
    sampling_cfg = _parse_config_str(
        sampling,
        defaults={
            "fanout1": 15,
            "fanout2": 10,
            "fanout3": 5,
            "fanout4": 10,
            "fanout5": 5,
            "fanout": "",
        },
    )
    raw_fanout = str(sampling_cfg.pop("fanout", "")).strip()
    if raw_fanout:
        fanout_list = [int(x.strip()) for x in raw_fanout.split(",") if x.strip()]
    else:
        num_layers = int(model_cfg.get("num_layers", 2))
        fanout_list = [int(sampling_cfg[f"fanout{i+1}"]) for i in range(num_layers)]
    features_cfg = _parse_config_str(
        features,
        defaults={
            "log1p": True,
            "zscore": True,
            "eps": 1e-6,
        },
    )

    wandb_cfg = _parse_config_str(
        wandb,
        defaults={
            "enabled": False,
            "project": "patent_concept",
            "entity": "",
            "name": "",
            "group": "",
            "job_type": "train_pyg",
            "tags": "",
            "mode": "online",  # online | offline | disabled
            "watch": False,
            "log_model": False,
            "fail_fast": False,
        },
    )

    run_name = _default_wandb_run_name(
        features_path=v_features_path,
        year_start_train=int(year_start_train),
        seed=int(seed),
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        sampling_cfg=sampling_cfg,
        concept_embs_as_features=concept_embs_as_features,
    )

    if bool(wandb_save_path):
        if not save_model_path:
            raise ValueError("wandb_save_path=true requires save_model_path")
        save_model_path = os.path.join(str(save_model_path), run_name)

    if bool(wandb_cfg.get("enabled", False)) and not str(wandb_cfg.get("name", "")).strip():
        wandb_cfg["name"] = run_name

    logger.info(f"device: {device}")
    logger.info(f"seed: {seed}")
    logger.info(f"year_start_train: {year_start_train}")
    logger.info(f"fanout: {fanout_list}")
    if save_model_path:
        logger.info("save_model_path: %s", save_model_path)

    wandb_run = _maybe_init_wandb(
        wandb_cfg,
        config={
            "graph_path": graph_path,
            "data_path": data_path,
            "v_features_path": v_features_path,
            "year_start_train": int(year_start_train),
            "seed": int(seed),
            "train": dict(train_cfg),
            "model": dict(model_cfg),
            "sampling": dict(sampling_cfg),
            "features": dict(features_cfg),
            "concept_embs_as_features": concept_embs_as_features,
            "args_raw": {
                "train": train or "",
                "model": model or "",
                "sampling": sampling or "",
                "features": features or "",
                "wandb": wandb or "",
                "ood_data_path": ood_data_path or "",
                "ood_year_start": "" if ood_year_start is None else str(ood_year_start),
                "ood_eval_interval": ""
                if ood_eval_interval is None
                else str(ood_eval_interval),
                "ood_eval_batch_size": ""
                if ood_eval_batch_size is None
                else str(ood_eval_batch_size),
                "ood_features_path": ood_features_path or "",
                "save_model_path": save_model_path or "",
                "wandb_save_path": str(bool(wandb_save_path)),
                "log_file": log_file or "",
            },
            "argv": list(sys.argv),
            "argv_str": " ".join(shlex.quote(a) for a in sys.argv),
        },
        logger=logger,
    )

    logger.info("Loading dataset")
    data_dict = load_pickle(data_path)
    x_train = np.asarray(data_dict["X_train"], dtype=np.int64)
    y_train = np.asarray(data_dict["y_train"], dtype=np.float32)
    x_val = np.asarray(data_dict.get("X_val", data_dict.get("X_test")), dtype=np.int64)
    y_val = np.asarray(data_dict.get("y_val", data_dict.get("y_test")), dtype=np.float32)

    logger.info("Loading node features")
    v_features = load_node_feature_matrix(
        v_features_path, name="v_features", logger=logger
    )

    use_raw_x = bool(model_cfg.get("decoder_raw_input", False))
    # Kept as the pre-normalization copy; transforms below rebind v_features.
    v_features_raw = v_features if use_raw_x else None

    if bool(features_cfg.get("log1p", True)):
        if float(np.min(v_features)) < 0.0:
            logger.warning(
                "v_features contains negative values; skipping log1p transform"
            )
        else:
            v_features = np.log1p(v_features)
    feature_mean = None
    feature_std = None
    if bool(features_cfg.get("zscore", True)):
        feature_mean = v_features.mean(axis=0, keepdims=True)
        feature_std = v_features.std(axis=0, keepdims=True)
        eps = float(features_cfg.get("eps", 1e-6))
        v_features = (v_features - feature_mean) / (feature_std + eps)

    num_nodes = int(v_features.shape[0])
    in_dim = int(v_features.shape[1])

    logger.info("Building past-graph edge_index")
    graph = Graph(graph_path)
    edge_index = build_edge_index_for_year(graph, year_start_train, num_nodes=num_nodes)

    pyg_data = Data(
        x=torch.from_numpy(v_features),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    if use_raw_x and v_features_raw is not None:
        pyg_data.x_raw = torch.from_numpy(v_features_raw)

    # edge labels: use provided pairs (already non-edges in past graph)
    train_edge_label_index = torch.from_numpy(x_train.T).contiguous()
    train_edge_label = torch.from_numpy(y_train).to(torch.float32)

    val_edge_label_index = torch.from_numpy(x_val.T).contiguous()
    val_edge_label = torch.from_numpy(y_val).to(torch.float32)

    # Optional OOD evaluation dataset (e.g., generated at year_start_train + 3)
    ood_loader = None
    ood_y = None
    if ood_data_path:
        ood = load_pickle(ood_data_path)
        if "X_test" not in ood or "y_test" not in ood:
            raise ValueError(
                f"OOD data file must contain keys X_test and y_test: {ood_data_path}"
            )
        ood_x = np.asarray(ood["X_test"], dtype=np.int64)
        ood_y = np.asarray(ood["y_test"], dtype=np.float32)

        v_features_ood_raw = None
        if ood_features_path:
            logger.info("Loading OOD node features")
            v_features_ood = load_node_feature_matrix(
                ood_features_path, name="v_features_ood", logger=logger
            )
            if use_raw_x:
                v_features_ood_raw = v_features_ood

            if bool(features_cfg.get("log1p", True)):
                if float(np.min(v_features_ood)) < 0.0:
                    logger.warning(
                        "v_features_ood contains negative values; skipping log1p transform"
                    )
                else:
                    v_features_ood = np.log1p(v_features_ood)
            if bool(features_cfg.get("zscore", True)):
                if feature_mean is not None and feature_std is not None:
                    eps = float(features_cfg.get("eps", 1e-6))
                    v_features_ood = (v_features_ood - feature_mean) / (feature_std + eps)
                else:
                    mean_ood = v_features_ood.mean(axis=0, keepdims=True)
                    std_ood = v_features_ood.std(axis=0, keepdims=True)
                    eps = float(features_cfg.get("eps", 1e-6))
                    v_features_ood = (v_features_ood - mean_ood) / (std_ood + eps)
        else:
            v_features_ood = v_features
            v_features_ood_raw = v_features_raw

        ood_year = (
            int(ood_year_start)
            if ood_year_start is not None
            else int(year_start_train) + 3
        )
        logger.info("Building OOD past-graph edge_index at year_start=%d", ood_year)
        edge_index_ood = build_edge_index_for_year(
            graph, ood_year, num_nodes=num_nodes
        )

        pyg_data_ood = Data(
            x=torch.from_numpy(v_features_ood),
            edge_index=edge_index_ood,
            num_nodes=num_nodes,
        )
        if use_raw_x and v_features_ood_raw is not None:
            pyg_data_ood.x_raw = torch.from_numpy(v_features_ood_raw)

        ood_edge_label_index = torch.from_numpy(ood_x.T).contiguous()
        ood_edge_label = torch.from_numpy(ood_y).to(torch.float32)

        if ood_eval_batch_size is None:
            ood_eval_batch_size = int(train_cfg["eval_batch_size"])

        ood_loader = LinkNeighborLoader(
            pyg_data_ood,
            edge_label_index=ood_edge_label_index,
            edge_label=ood_edge_label,
            num_neighbors=fanout_list,
            batch_size=int(ood_eval_batch_size),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=int(train_cfg["num_workers"]) > 0,
            persistent_workers=int(train_cfg["num_workers"]) > 0,
        )

        logger.info(
            "OOD pairs: %d | pos_rate: %.4f | year_start=%d",
            int(ood_x.shape[0]),
            float(ood_y.mean()) if ood_y.size else float("nan"),
            int(ood_year),
        )

    # NOTE: LinkNeighborLoader does not support "pos_ratio" directly when you provide labels.
    # If you want a controlled pos ratio, you should subsample x_train/y_train offline.
    loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label,
        num_neighbors=fanout_list,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=int(train_cfg["num_workers"]) > 0,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )

    val_loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=val_edge_label_index,
        edge_label=val_edge_label,
        num_neighbors=fanout_list,
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=int(train_cfg["num_workers"]) > 0,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )

    encoder = SAGEEncoder(
        in_dim=in_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        dropout=float(model_cfg["dropout"]),
        num_layers=int(model_cfg["num_layers"]),
        use_norm=bool(model_cfg["use_norm"]),
        use_residual=bool(model_cfg["use_residual"]),
        conv=str(model_cfg.get("conv", "sage")),
        activation=str(model_cfg.get("activation", "relu")),
    ).to(device)

    decoder_kind = str(model_cfg.get("decoder", "mlp")).lower()
    valid_decoders = {
        "mlp",
        "dot",
        "mlp_no_interaction",
        "mlp_only_diff",
        "mlp_only_product",
        "mlp_cat_raw",
        "mlp_two_tower",
        "mlp_sum_fusion",
    }
    if decoder_kind not in valid_decoders:
        raise ValueError(f"model.decoder must be one of: {', '.join(sorted(valid_decoders))}")
    if bool(model_cfg.get("decoder_aux", False)) and decoder_kind != "mlp_two_tower":
        raise ValueError("model.decoder_aux=true requires decoder=mlp_two_tower")
    if bool(model_cfg.get("aux_learnable", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.aux_learnable=true requires decoder_aux=true")
    if not bool(model_cfg.get("aux_joint_loss", True)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.aux_joint_loss=false requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_mix", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.infer_aux_mix=true requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_grid", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.infer_aux_grid=true requires decoder_aux=true")

    decoder: nn.Module
    if decoder_kind == "mlp":
        decoder = EdgeMLPDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = f"EdgeMLPDecoder(hidden={model_cfg['decoder_hidden_dim']})"
    elif decoder_kind == "mlp_no_interaction":
        decoder = MLPNoInteractionDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = f"MLPNoInteractionDecoder(hidden={model_cfg['decoder_hidden_dim']})"
    elif decoder_kind == "mlp_only_diff":
        decoder = MLPDiffDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = f"MLPDiffDecoder(hidden={model_cfg['decoder_hidden_dim']})"
    elif decoder_kind == "mlp_only_product":
        decoder = MLPProductDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = f"MLPProductDecoder(hidden={model_cfg['decoder_hidden_dim']})"
    elif decoder_kind == "mlp_cat_raw":
        decoder = EdgeMLPCatRawDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = (
            f"EdgeMLPCatRawDecoder(hidden={model_cfg['decoder_hidden_dim']},"
            f" raw_dim={in_dim})"
        )
    elif decoder_kind == "mlp_two_tower":
        decoder = EdgeTwoTowerDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
            aux=bool(model_cfg.get("decoder_aux", False)),
            aux_learnable=bool(model_cfg.get("aux_learnable", False)),
            drop_path=float(model_cfg.get("decoder_drop_path", 0.0)),
            activation=str(model_cfg.get("activation", "relu")),
            fusion_head=str(model_cfg.get("decoder_fusion_head", "linear")),
        ).to(device)
        decoder_params = list(decoder.parameters())
        if bool(model_cfg.get("decoder_aux", False)):
            if bool(model_cfg.get("aux_learnable", False)):
                decoder_desc = (
                    f"EdgeTwoTowerDecoder(hidden={model_cfg['decoder_hidden_dim']},"
                    f" raw_dim={in_dim}, aux(learnable init 0.5,"
                    f" lambda={model_cfg['aux_lambda']}))"
                )
            else:
                decoder_desc = (
                    f"EdgeTwoTowerDecoder(hidden={model_cfg['decoder_hidden_dim']},"
                    f" raw_dim={in_dim}, aux(w_g={model_cfg['aux_w_graph']},"
                    f" w_t={model_cfg['aux_w_text']},"
                    f" lambda={model_cfg['aux_lambda']}))"
                )
        else:
            decoder_desc = (
                f"EdgeTwoTowerDecoder(hidden={model_cfg['decoder_hidden_dim']},"
                f" raw_dim={in_dim})"
            )
    elif decoder_kind == "mlp_sum_fusion":
        decoder = EdgeSumFusionDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = (
            f"EdgeSumFusionDecoder(hidden={model_cfg['decoder_hidden_dim']},"
            f" raw_dim={in_dim})"
        )
    else:
        decoder = DotDecoder().to(device)
        decoder_params = []
        decoder_desc = "DotDecoder"

    aux_decoder: EdgeTwoTowerDecoder | None = None
    if bool(model_cfg.get("decoder_aux", False)):
        if not isinstance(decoder, EdgeTwoTowerDecoder):
            raise ValueError("model.decoder_aux=true requires decoder=mlp_two_tower")
        aux_decoder = decoder
    aux_learnable = bool(model_cfg.get("aux_learnable", False))
    # Inference-time fixed mixture of the two tower heads (used by aux_joint_loss=false
    # "independent towers" training; also usable with joint-loss training).
    infer_aux_mix = bool(model_cfg.get("infer_aux_mix", False)) and aux_decoder is not None
    infer_w_graph = float(model_cfg.get("aux_w_graph", 0.9))
    infer_w_text = float(model_cfg.get("aux_w_text", 0.1))
    # Paper-protocol mixture: keep both tower probabilities and pick the best
    # 11-point grid weight w*pg + (1-w)*pt (blend.py semantics).
    infer_aux_grid = bool(model_cfg.get("infer_aux_grid", False)) and aux_decoder is not None

    optimizer_kind = str(train_cfg.get("optimizer", "adam")).lower()
    optimizer: torch.optim.Optimizer
    if optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + decoder_params,
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        logger.info("Using AdamW optimizer (decoupled weight decay)")
    else:
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + decoder_params,
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        logger.info("Using Adam optimizer")

    scheduler_kind = str(train_cfg.get("scheduler", "none")).lower()
    num_epochs = int(train_cfg["num_epochs"])
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    )
    if scheduler_kind == "cosine":
        T_max = int(train_cfg.get("scheduler_T_max", 0))
        if T_max <= 0:
            T_max = num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        logger.info(f"Using CosineAnnealingLR scheduler (T_max={T_max})")
    elif scheduler_kind == "warmup_cosine":
        warmup_epochs = int(train_cfg.get("scheduler_warmup", 5))

        def _warmup_cosine_lambda(epoch_idx: int) -> float:
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(max(1, warmup_epochs))
            progress = (epoch_idx - warmup_epochs) / float(
                max(1, num_epochs - warmup_epochs)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _warmup_cosine_lambda
        )
        logger.info(
            f"Using warmup({warmup_epochs}ep)+cosine LR schedule"
        )
    elif scheduler_kind == "cosine_restart":
        restart_t0 = int(train_cfg.get("scheduler_T0", 30))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_t0, T_mult=2
        )
        logger.info(
            f"Using CosineAnnealingWarmRestarts (T_0={restart_t0}, T_mult=2)"
        )
    elif scheduler_kind == "plateau":
        sched_patience = int(train_cfg.get("scheduler_patience", 10))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=sched_patience, factor=0.5
        )
        logger.info(f"Using ReduceLROnPlateau scheduler (patience={sched_patience})")
    elif scheduler_kind == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
        logger.info("Using StepLR scheduler")
    else:
        scheduler = None
        logger.info("No LR scheduler")

    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    best_val_auc = float("-inf")
    patience_counter = 0
    if early_stopping_patience > 0:
        logger.info(f"Early stopping enabled (patience={early_stopping_patience})")

    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)))

    logger.info(
        "Model: SAGEConv%dLayer(in_dim=%d, hidden_dim=%s, out_dim=%s) + %s",
        int(model_cfg["num_layers"]),
        in_dim,
        model_cfg["hidden_dim"],
        model_cfg["out_dim"],
        decoder_desc,
    )
    logger.info(f"Train pairs: {len(x_train)} | Val pairs: {len(x_val)}")

    if wandb_run is not None and bool(wandb_cfg.get("watch", False)):
        # Optional and can be expensive.
        try:
            import wandb  # type: ignore

            wandb.watch([encoder, decoder], log="all", log_freq=200)
        except Exception:
            pass

    if wandb_run is not None:
        try:
            n_params = int(
                sum(p.numel() for p in encoder.parameters())
                + sum(p.numel() for p in decoder.parameters())
            )
            wandb_run.log(
                {
                    "model/params": n_params,
                    "data/train_pairs": int(len(x_train)),
                    "data/val_pairs": int(len(x_val)),
                },
                step=0,
            )
        except Exception:
            pass

    eval_dir = None
    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        eval_dir = os.path.join(save_model_path, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        logger.info("eval_dir: %s", eval_dir)

    def _predict(loader, *, desc: str):
        encoder.eval()
        decoder.eval()
        scores: list[float] = []
        graph_scores: list[float] = []
        text_scores: list[float] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(device, non_blocking=True)
                z = encoder(batch.x, batch.edge_index)
                u = batch.edge_label_index[0]
                v = batch.edge_label_index[1]
                xu = batch.x_raw[u] if use_raw_x else batch.x[u]
                xv = batch.x_raw[v] if use_raw_x else batch.x[v]
                if infer_aux_mix or infer_aux_grid:
                    _, g_logits, t_logits = decoder(
                        z[u], z[v], xu, xv, return_aux=True
                    )
                    p_g = torch.sigmoid(g_logits).detach().cpu().numpy()
                    p_t = torch.sigmoid(t_logits).detach().cpu().numpy()
                    graph_scores.extend(p_g.tolist())
                    text_scores.extend(p_t.tolist())
                    if not infer_aux_grid:
                        probs = infer_w_graph * p_g + infer_w_text * p_t
                        scores.extend(probs.tolist())
                else:
                    logits = decoder(z[u], z[v], xu, xv)
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    scores.extend(probs.tolist())
        if infer_aux_grid:
            return np.asarray(graph_scores), np.asarray(text_scores)
        return np.asarray(scores)

    def _maybe_save_predictions(predictions: np.ndarray, *, filename: str) -> None:
        if not eval_dir:
            return
        save_compressed(predictions, os.path.join(eval_dir, filename))

    def evaluate() -> tuple[float, tuple[int, int, int, int], np.ndarray]:
        if infer_aux_grid:
            p_g, p_t = _predict(val_loader, desc="Eval")
            auc_grid, best_w, predictions = blend_grid_best(
                np.asarray(y_val, dtype=np.float32), p_g, p_t
            )
            logger.info(
                "Val mix grid (11-pt, blend.py protocol): best w_graph=%.2f AUC=%.4f",
                best_w,
                auc_grid,
            )
            _maybe_save_predictions(predictions, filename="val_grid_bestw.pkl.gz")
            _maybe_save_predictions(p_g, filename="val_grid_graph.pkl.gz")
            _maybe_save_predictions(p_t, filename="val_grid_text.pkl.gz")
        else:
            predictions = _predict(val_loader, desc="Eval")
        auc, _, cm = test(
            torch.tensor(y_val, dtype=torch.float32), predictions, threshold=0.5
        )
        return float(auc), (int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3])), predictions

    def evaluate_ood() -> tuple[float, tuple[int, int, int, int], np.ndarray]:
        if ood_loader is None or ood_y is None:
            raise RuntimeError("OOD loader not initialized")
        if infer_aux_grid:
            p_g, p_t = _predict(ood_loader, desc="OOD Eval")
            auc_grid, best_w, predictions = blend_grid_best(
                np.asarray(ood_y, dtype=np.float32), p_g, p_t
            )
            logger.info(
                "OOD mix grid (11-pt): best w_graph=%.2f AUC=%.4f", best_w, auc_grid
            )
            _maybe_save_predictions(predictions, filename="ood_grid_bestw.pkl.gz")
        else:
            predictions = _predict(ood_loader, desc="OOD Eval")
        auc, _, cm = test(
            torch.tensor(ood_y, dtype=torch.float32), predictions, threshold=0.5
        )
        return float(auc), (int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3])), predictions

    def _save_best_model(epoch_idx: int) -> None:
        if not save_model_path:
            return
        best_path = os.path.join(save_model_path, "best_model.pt")
        logger.info("Saving best model (epoch %d) to %s", epoch_idx, best_path)
        model_save_cfg = dict(model_cfg)
        model_save_cfg["num_layers"] = int(model_cfg.get("num_layers", 2))
        sampling_save_cfg = dict(sampling_cfg)
        sampling_save_cfg["fanout_list"] = fanout_list
        ckpt_dict = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "year_start_train": int(year_start_train),
            "epoch": int(epoch_idx),
            "best_val_auc": float(best_val_auc),
            "model": model_save_cfg,
            "sampling": sampling_save_cfg,
            "features": dict(features_cfg),
            "concept_embs_as_features": concept_embs_as_features,
            "v_features_path": v_features_path,
        }
        if feature_mean is not None:
            ckpt_dict["feature_mean"] = feature_mean
        if feature_std is not None:
            ckpt_dict["feature_std"] = feature_std
        torch.save(ckpt_dict, best_path)

    def _save_final_model() -> str | None:
        if not save_model_path:
            return None
        final_path = os.path.join(save_model_path, "model.pt")
        logger.info("Saving final model to %s", final_path)
        model_save_cfg = dict(model_cfg)
        model_save_cfg["num_layers"] = int(model_cfg.get("num_layers", 2))
        sampling_save_cfg = dict(sampling_cfg)
        sampling_save_cfg["fanout_list"] = fanout_list
        ckpt_dict = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "year_start_train": int(year_start_train),
            "epoch": int(train_cfg["num_epochs"]),
            "model": model_save_cfg,
            "sampling": sampling_save_cfg,
            "features": dict(features_cfg),
            "concept_embs_as_features": concept_embs_as_features,
            "v_features_path": v_features_path,
        }
        if feature_mean is not None:
            ckpt_dict["feature_mean"] = feature_mean
        if feature_std is not None:
            ckpt_dict["feature_std"] = feature_std
        torch.save(ckpt_dict, final_path)
        return final_path

    drop_edge_p = float(train_cfg.get("drop_edge", 0.0))
    if drop_edge_p > 0.0:
        logger.info(f"DropEdge enabled (p={drop_edge_p}, train batches only)")

    grad_probe: GradProbe | None = None
    probe_interval = int(os.environ.get("GRAD_PROBE_INTERVAL", "0") or 0)
    if probe_interval > 0 and pyg_data.x is not None:
        grad_probe = GradProbe(
            pyg_data.x, pyg_data.edge_index, device, probe_interval
        )
        logger.info(
            "GRAD_PROBE enabled: per-step grad+z checks, log every %d steps",
            probe_interval,
        )

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        encoder.train()
        decoder.train()
        losses: list[float] = []

        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("amp", True))):
                edge_index = batch.edge_index
                if drop_edge_p > 0.0 and edge_index.size(1) > 0:
                    keep_mask = (
                        torch.rand(edge_index.size(1), device=edge_index.device)
                        >= drop_edge_p
                    )
                    if bool(keep_mask.any()):
                        edge_index = edge_index[:, keep_mask]
                z = encoder(batch.x, edge_index)
                u = batch.edge_label_index[0]
                v = batch.edge_label_index[1]
                xu = batch.x_raw[u] if use_raw_x else batch.x[u]
                xv = batch.x_raw[v] if use_raw_x else batch.x[v]
                if aux_decoder is not None:
                    logits, g_logits, t_logits = aux_decoder(
                        z[u], z[v], xu, xv, return_aux=True
                    )
                    if aux_learnable and aux_decoder.aux_logit is not None:
                        w_g: torch.Tensor | float = torch.sigmoid(aux_decoder.aux_logit)
                        w_t: torch.Tensor | float = 1.0 - w_g
                    else:
                        w_g = float(model_cfg["aux_w_graph"])
                        w_t = float(model_cfg["aux_w_text"])
                    if bool(model_cfg.get("aux_joint_loss", True)):
                        loss = criterion(logits, batch.edge_label) + float(
                            model_cfg["aux_lambda"]
                        ) * (
                            w_g * criterion(g_logits, batch.edge_label)
                            + w_t * criterion(t_logits, batch.edge_label)
                        )
                    else:
                        # Independent-tower training: no joint-head loss at all.
                        loss = w_g * criterion(g_logits, batch.edge_label) + w_t * criterion(
                            t_logits, batch.edge_label
                        )
                else:
                    logits = decoder(z[u], z[v], xu, xv)
                    loss = criterion(logits, batch.edge_label)

            scaler.scale(loss).backward()

            grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + decoder_params,
                    max_norm=grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            if grad_probe is not None:
                grad_probe.after_step(encoder, logger)

            losses.append(float(loss.detach().cpu().item()))

        if epoch % int(train_cfg["log_interval"]) == 0:
            auc, (tn, fp, fn, tp), predictions = evaluate()
            logger.info(
                "Epoch: %d, Loss: %.4f, Val AUC: %.4f, TP: %d, FP: %d, FN: %d, TN: %d",
                epoch,
                float(np.mean(losses)) if losses else float("nan"),
                auc,
                tp,
                fp,
                fn,
                tn,
            )

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "epoch": int(epoch),
                            "loss": float(np.mean(losses))
                            if losses
                            else float("nan"),
                            "val_auc": float(auc),
                        },
                        step=int(epoch),
                    )
                except Exception:
                    pass

            if early_stopping_patience > 0:
                if auc > best_val_auc:
                    best_val_auc = auc
                    patience_counter = 0
                    _save_best_model(epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d: no improvement for %d epochs (best AUC=%.4f)",
                            epoch, early_stopping_patience, best_val_auc,
                        )
                        break
            elif auc > best_val_auc:
                best_val_auc = auc
                _save_best_model(epoch)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(auc if epoch % int(train_cfg["log_interval"]) == 0 else best_val_auc)
            else:
                scheduler.step()

        # OOD eval can be expensive; default is off unless configured.
        interval = int(ood_eval_interval) if ood_eval_interval is not None else int(
            train_cfg.get("ood_eval_interval", 0)
        )
        if ood_loader is not None and interval and interval > 0 and epoch % interval == 0:
            ood_auc, (ood_tn, ood_fp, ood_fn, ood_tp), ood_predictions = evaluate_ood()
            _maybe_save_predictions(
                ood_predictions,
                filename=f"predictions.ood_eval_epoch_{epoch}.pkl.gz",
            )
            logger.info(
                "OOD | Epoch: %d, AUC: %.4f, TP: %d, FP: %d, FN: %d, TN: %d",
                epoch,
                float(ood_auc),
                int(ood_tp),
                int(ood_fp),
                int(ood_fn),
                int(ood_tn),
            )

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "epoch": int(epoch),
                            "val_auc": float(ood_auc),
                        },
                        step=int(epoch),
                    )
                except Exception:
                    pass

    if grad_probe is not None:
        logger.info(
            "GRAD_PROBE FINAL: steps=%d, encoder grad finite&nonzero %d/%d,"
            " z changed %d/%d",
            grad_probe.step,
            grad_probe.grad_ok,
            grad_probe.grad_checks,
            grad_probe.dz_ok,
            grad_probe.dz_checks,
        )

    final_model_path = _save_final_model()

    if final_model_path and wandb_run is not None and bool(wandb_cfg.get("log_model", False)):
        try:
            import wandb  # type: ignore

            artifact = wandb.Artifact(
                name=str(wandb_cfg.get("artifact_name", "train_pyg_model")),
                type="model",
            )
            artifact.add_file(final_model_path)
            wandb_run.log_artifact(artifact)
        except Exception:
            pass

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


def eval_predictions(
    graph_path="data-v2/graph/edges.M.pkl",
    data_path="data-v2/model/data.M.pkl",
    v_features_path="data-v2/model/baseline/features.2019.binary.M.pkl.gz",
    model_path=None,
    pred_path="data-v2/model/gnn/predictions.pkl.gz",
    split="test",
    year_start=2019,
    sampling=None,
    features=None,
    model=None,
    eval_batch_size=16384,
    num_workers=_DEFAULT_NUM_WORKERS,
    log_file="logs-v2/gnn/gnn_eval_mixture_pyg.log",
):
    """Evaluate a saved checkpoint/model and write predictions to a pkl(.gz)."""

    _require_pyg()
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    if not model_path:
        raise ValueError("model_path is required")

    reload(logging)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    ckpt = torch.load(model_path, map_location=device)
    ckpt_model_cfg = ckpt.get("model") or {}
    ckpt_sampling_cfg = ckpt.get("sampling") or {}
    ckpt_features_cfg = ckpt.get("features") or {}

    model_cfg = _parse_config_str(
        model,
        defaults={
            "hidden_dim": int(ckpt_model_cfg.get("hidden_dim", 128)),
            "out_dim": int(ckpt_model_cfg.get("out_dim", 128)),
            "dropout": float(ckpt_model_cfg.get("dropout", 0.1)),
            "decoder": str(ckpt_model_cfg.get("decoder", "mlp")),
            "decoder_hidden_dim": int(ckpt_model_cfg.get("decoder_hidden_dim", 256)),
            "decoder_dropout": float(ckpt_model_cfg.get("decoder_dropout", 0.1)),
            "num_layers": int(ckpt_model_cfg.get("num_layers", 2)),
            "use_norm": bool(ckpt_model_cfg.get("use_norm", False)),
            "use_residual": bool(ckpt_model_cfg.get("use_residual", False)),
            "decoder_aux": bool(ckpt_model_cfg.get("decoder_aux", False)),
            "decoder_raw_input": bool(ckpt_model_cfg.get("decoder_raw_input", False)),
            "aux_w_graph": float(ckpt_model_cfg.get("aux_w_graph", 0.9)),
            "aux_w_text": float(ckpt_model_cfg.get("aux_w_text", 0.1)),
            "aux_lambda": float(ckpt_model_cfg.get("aux_lambda", 1.0)),
            "aux_learnable": bool(ckpt_model_cfg.get("aux_learnable", False)),
            "aux_joint_loss": bool(ckpt_model_cfg.get("aux_joint_loss", True)),
            "infer_aux_mix": bool(ckpt_model_cfg.get("infer_aux_mix", False)),
            "infer_aux_grid": bool(ckpt_model_cfg.get("infer_aux_grid", False)),
            "conv": str(ckpt_model_cfg.get("conv", "sage")),
            "decoder_drop_path": float(
                ckpt_model_cfg.get("decoder_drop_path", 0.0)
            ),
            "activation": str(ckpt_model_cfg.get("activation", "relu")),
            "decoder_fusion_head": str(
                ckpt_model_cfg.get("decoder_fusion_head", "linear")
            ),
        },
    )
    sampling_cfg = _parse_config_str(
        sampling,
        defaults={
            "fanout1": int(ckpt_sampling_cfg.get("fanout1", 20)),
            "fanout2": int(ckpt_sampling_cfg.get("fanout2", 15)),
            "fanout3": int(ckpt_sampling_cfg.get("fanout3", 5)),
            "fanout4": int(ckpt_sampling_cfg.get("fanout4", 10)),
            "fanout5": int(ckpt_sampling_cfg.get("fanout5", 5)),
        },
    )
    ckpt_fanout_list = ckpt_sampling_cfg.get("fanout_list", None)
    if ckpt_fanout_list is not None:
        fanout_list_eval = [int(x) for x in ckpt_fanout_list]
    else:
        num_layers = int(ckpt_model_cfg.get("num_layers", 2))
        fanout_list_eval = [int(sampling_cfg[f"fanout{i+1}"]) for i in range(num_layers)]
    features_cfg = _parse_config_str(
        features,
        defaults={
            "log1p": bool(ckpt_features_cfg.get("log1p", True)),
            "zscore": bool(ckpt_features_cfg.get("zscore", True)),
            "eps": float(ckpt_features_cfg.get("eps", 1e-6)),
        },
    )

    if year_start is None:
        year_start = ckpt.get("year_start_train")
    if year_start is None:
        raise ValueError("year_start is required (or present in checkpoint)")

    logger.info("Loading dataset")
    data_dict = load_pickle(data_path)
    split = str(split).lower()
    if split == "train":
        x = np.asarray(data_dict["X_train"], dtype=np.int64)
        y = np.asarray(data_dict["y_train"], dtype=np.float32)
    elif split == "val":
        x = np.asarray(data_dict.get("X_val", data_dict.get("X_test")), dtype=np.int64)
        y = np.asarray(data_dict.get("y_val", data_dict.get("y_test")), dtype=np.float32)
    elif split == "test":
        x = np.asarray(data_dict.get("X_test", data_dict.get("X_val")), dtype=np.int64)
        y = np.asarray(data_dict.get("y_test", data_dict.get("y_val")), dtype=np.float32)
    else:
        raise ValueError("split must be one of: train, val, test")

    logger.info("Loading node features")
    v_features = load_node_feature_matrix(
        v_features_path, name="v_features", logger=logger
    )

    use_raw_x = bool(model_cfg.get("decoder_raw_input", False))
    # Kept as the pre-normalization copy; transforms below rebind v_features.
    v_features_raw = v_features if use_raw_x else None

    if bool(features_cfg.get("log1p", True)):
        if float(np.min(v_features)) < 0.0:
            logger.warning("v_features contains negative values; skipping log1p transform")
        else:
            v_features = np.log1p(v_features)
    if bool(features_cfg.get("zscore", True)):
        ckpt_feature_mean = ckpt.get("feature_mean")
        ckpt_feature_std = ckpt.get("feature_std")
        eps = float(features_cfg.get("eps", 1e-6))
        if ckpt_feature_mean is not None and ckpt_feature_std is not None:
            v_features = (v_features - ckpt_feature_mean) / (ckpt_feature_std + eps)
        else:
            mean = v_features.mean(axis=0, keepdims=True)
            std = v_features.std(axis=0, keepdims=True)
            v_features = (v_features - mean) / (std + eps)

    num_nodes = int(v_features.shape[0])
    in_dim = int(v_features.shape[1])

    logger.info("Building past-graph edge_index")
    graph = Graph(graph_path)
    edge_index = build_edge_index_for_year(graph, int(year_start), num_nodes=num_nodes)

    pyg_data = Data(
        x=torch.from_numpy(v_features),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    if use_raw_x and v_features_raw is not None:
        pyg_data.x_raw = torch.from_numpy(v_features_raw)

    edge_label_index = torch.from_numpy(x.T).contiguous()
    edge_label = torch.from_numpy(y).to(torch.float32)

    loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        num_neighbors=fanout_list_eval,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=int(num_workers) > 0,
        persistent_workers=int(num_workers) > 0,
    )

    encoder = SAGEEncoder(
        in_dim=in_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        dropout=float(model_cfg["dropout"]),
        num_layers=int(model_cfg.get("num_layers", 2)),
        use_norm=bool(model_cfg.get("use_norm", False)),
        use_residual=bool(model_cfg.get("use_residual", False)),
        conv=str(model_cfg.get("conv", "sage")),
        activation=str(model_cfg.get("activation", "relu")),
    ).to(device)

    decoder_kind = str(model_cfg.get("decoder", "mlp")).lower()
    valid_decoders = {
        "mlp",
        "dot",
        "mlp_no_interaction",
        "mlp_only_diff",
        "mlp_only_product",
        "mlp_cat_raw",
        "mlp_two_tower",
        "mlp_sum_fusion",
    }
    if decoder_kind not in valid_decoders:
        raise ValueError(f"model.decoder must be one of: {', '.join(sorted(valid_decoders))}")
    if bool(model_cfg.get("decoder_aux", False)) and decoder_kind != "mlp_two_tower":
        raise ValueError("model.decoder_aux=true requires decoder=mlp_two_tower")
    if bool(model_cfg.get("aux_learnable", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.aux_learnable=true requires decoder_aux=true")
    if not bool(model_cfg.get("aux_joint_loss", True)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.aux_joint_loss=false requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_mix", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.infer_aux_mix=true requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_grid", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("model.infer_aux_grid=true requires decoder_aux=true")

    decoder: nn.Module
    if decoder_kind == "mlp":
        decoder = EdgeMLPDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "mlp_no_interaction":
        decoder = MLPNoInteractionDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "mlp_only_diff":
        decoder = MLPDiffDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "mlp_only_product":
        decoder = MLPProductDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "mlp_cat_raw":
        decoder = EdgeMLPCatRawDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "mlp_two_tower":
        decoder = EdgeTwoTowerDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
            aux=bool(model_cfg.get("decoder_aux", False)),
            aux_learnable=bool(model_cfg.get("aux_learnable", False)),
            drop_path=float(model_cfg.get("decoder_drop_path", 0.0)),
            activation=str(model_cfg.get("activation", "relu")),
            fusion_head=str(model_cfg.get("decoder_fusion_head", "linear")),
        ).to(device)
    elif decoder_kind == "mlp_sum_fusion":
        decoder = EdgeSumFusionDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            raw_dim=in_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    else:
        decoder = DotDecoder().to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()
    scores: list[float] = []
    graph_scores: list[float] = []
    text_scores: list[float] = []
    infer_aux_mix = bool(model_cfg.get("infer_aux_mix", False)) and bool(
        model_cfg.get("decoder_aux", False)
    )
    infer_aux_grid = bool(model_cfg.get("infer_aux_grid", False)) and bool(
        model_cfg.get("decoder_aux", False)
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval({split})", leave=False):
            batch = batch.to(device, non_blocking=True)
            z = encoder(batch.x, batch.edge_index)
            u = batch.edge_label_index[0]
            v = batch.edge_label_index[1]
            xu = batch.x_raw[u] if use_raw_x else batch.x[u]
            xv = batch.x_raw[v] if use_raw_x else batch.x[v]
            if infer_aux_mix or infer_aux_grid:
                _, g_logits, t_logits = decoder(
                    z[u], z[v], xu, xv, return_aux=True
                )
                p_g = torch.sigmoid(g_logits).detach().cpu().numpy()
                p_t = torch.sigmoid(t_logits).detach().cpu().numpy()
                graph_scores.extend(p_g.tolist())
                text_scores.extend(p_t.tolist())
                if not infer_aux_grid:
                    probs = (
                        float(model_cfg.get("aux_w_graph", 0.9)) * p_g
                        + float(model_cfg.get("aux_w_text", 0.1)) * p_t
                    )
                    scores.extend(probs.tolist())
            else:
                logits = decoder(z[u], z[v], xu, xv)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                scores.extend(probs.tolist())

    if infer_aux_grid:
        auc_grid, best_w, predictions = blend_grid_best(
            np.asarray(y, dtype=np.float32),
            np.asarray(graph_scores),
            np.asarray(text_scores),
        )
        logger.info(
            "Eval mix grid (11-pt, blend.py protocol): best w_graph=%.2f AUC=%.4f",
            best_w,
            float(auc_grid),
        )
    else:
        predictions = np.asarray(scores)
    auc, _, cm = test(
        torch.tensor(y, dtype=torch.float32), predictions, threshold=0.5
    )
    logger.info(
        "Eval split=%s | AUC=%.4f | TP=%d FP=%d FN=%d TN=%d",
        split,
        float(auc),
        int(cm[3]),
        int(cm[1]),
        int(cm[2]),
        int(cm[0]),
    )

    save_compressed(predictions, pred_path)
    logger.info("Saved predictions to %s", pred_path)

    print_metrics(torch.tensor(y, dtype=torch.float32), predictions, threshold=0.5)


if __name__ == "__main__":
    fire.Fire({"train": main, "eval": eval_predictions})
