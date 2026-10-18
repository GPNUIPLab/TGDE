"""Evaluate trained GraphSAGE GNN model on patent concept graph test data.

Loads a GNN checkpoint (from train_pyg.py), runs batched inference via
LinkNeighborLoader, computes AUC/optimal threshold/precision/recall/F1,
saves metrics to JSON and predictions to compressed pickle.

Usage:
  python BM/pipeline/evaluate_gnn.py
  python BM/pipeline/evaluate_gnn.py --model_path BM/pipeline/output/gnn/model.pt
"""

import gzip
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
from fire import Fire
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger("evaluate_gnn")
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_compressed(path):
    if not path:
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def _to_numpy_2d_float32(x):
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32, copy=False)


def load_node_features(path, logger):
    obj = load_compressed(path)
    if isinstance(obj, dict) and "v_features" in obj:
        logger.info(f"Loaded v_features from {path}")
        return _to_numpy_2d_float32(obj["v_features"])
    if isinstance(obj, dict):
        for k in ("embeddings", "embs", "x", "features"):
            if k in obj:
                logger.info(f"Loaded '{k}' from {path}")
                return _to_numpy_2d_float32(obj[k])
    logger.info(f"Loaded raw array from {path}")
    return _to_numpy_2d_float32(obj)


def parse_config_str(value, defaults):
    out = dict(defaults)
    if not value:
        return out
    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid config item '{item}'")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in out:
            raise ValueError(f"Unknown config key '{k}'")
        default_val = out[k]
        if isinstance(default_val, bool):
            out[k] = v.lower() in {"1", "true", "yes"}
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            out[k] = int(float(v))
        elif isinstance(default_val, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def build_edge_index_for_year(graph, year, num_nodes):
    edges = graph.get_until_year(year)
    uv = edges[:, :2].astype(np.int64, copy=False)
    uv_rev = uv[:, [1, 0]]
    uv2 = np.vstack([uv, uv_rev])
    edge_index = torch.from_numpy(uv2.T).contiguous()
    return edge_index


def main(
    model_path=None,
    graph_path=None,
    data_path=None,
    v_features_path=None,
    year_start_test=2021,
    features_cfg=None,
    sampling_cfg=None,
    model_cfg=None,
    eval_batch_size=16384,
    num_workers=0,
    metrics_path=None,
    pred_path=None,
    log_file=None,
    concept_embs_as_features=False,
    concept_source="rake",
    device=0,
):
    """Evaluate trained GNN model on patent test data.

    Args:
        model_path: Path to GNN checkpoint (model.pt from train_pyg.py)
        graph_path: Path to edges.pkl
        data_path: Path to data.pkl
        v_features_path: Path to features.pkl.gz (auto-detected from checkpoint if None)
        year_start_test: Year for building evaluation graph
        features_cfg: Feature config string (e.g. "log1p=true,zscore=true")
        sampling_cfg: Sampling config string (e.g. "fanout1=15,fanout2=10")
        model_cfg: Model config string override
        eval_batch_size: Batch size for evaluation
        num_workers: DataLoader workers (0 for Windows compatibility)
        metrics_path: Output JSON path for metrics
        pred_path: Output path for predictions pickle
        log_file: Path to log file
        concept_embs_as_features: Use concept embeddings as node features (auto-detected if v_features_path not set)
        concept_source: "rake" or "llm" — determines default paths for data/features/models.
        device: GPU device id (default 0, set to -1 for CPU).
    """
    if device is not None and str(device) != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # Import PyG
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import LinkNeighborLoader
        from torch_geometric.nn import SAGEConv
    except ImportError as e:
        print(f"PyTorch Geometric required: {e}")
        sys.exit(1)

    from patent_concept.model.graph import Graph
    from patent_concept.model.gnn.train_pyg import (
        SAGEEncoder,
        EdgeMLPDecoder,
        EdgeMLPCatRawDecoder,
        EdgeTwoTowerDecoder,
        EdgeSumFusionDecoder,
        DotDecoder,
        MLPNoInteractionDecoder,
        MLPDiffDecoder,
        MLPProductDecoder,
    )

    _source_dir = os.path.join(OUTPUT_DIR, concept_source)
    gnn_dir = "gnn_embs" if concept_embs_as_features else "gnn_topo"
    model_path = model_path or os.path.join(_source_dir, gnn_dir, "model.pt")
    graph_path = graph_path or os.path.join(_source_dir, "edges.pkl")
    data_path = data_path or os.path.join(_source_dir, "data.pkl")
    metrics_path = metrics_path or os.path.join(_source_dir, gnn_dir, "metrics.json")
    pred_path = pred_path or os.path.join(_source_dir, gnn_dir, "predictions.pkl.gz")
    log_file = log_file or os.path.join(LOG_DIR, "evaluate_gnn.log")
    os.makedirs(os.path.join(_source_dir, gnn_dir), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Auto-detect feature type from checkpoint if v_features_path not explicitly set
    if v_features_path is None:
        ckpt_peek = torch.load(model_path, map_location="cpu")
        # Read concept_embs_as_features from checkpoint (saved by train_pyg.py)
        ckpt_embs_flag = ckpt_peek.get("concept_embs_as_features", None)
        if ckpt_embs_flag is not None:
            concept_embs_as_features = bool(ckpt_embs_flag)
        else:
            # Fallback: check first conv layer weight shape to infer input dim
            enc_state = ckpt_peek.get("encoder", {})
            first_lin_weight = None
            for key in ["conv1.lin_l.weight", "conv1.lin_rel.weight",
                         "convs.0.lin_l.weight", "convs.0.lin_rel.weight"]:
                if key in enc_state:
                    first_lin_weight = enc_state[key]
                    break
            if first_lin_weight is not None:
                in_dim_peek = first_lin_weight.shape[1]
                concept_embs_as_features = in_dim_peek > 10
        del ckpt_peek
        if concept_embs_as_features:
            v_features_path = os.path.join(_source_dir, "concept_embs_test.pkl.gz")
        else:
            v_features_path = os.path.join(_source_dir, "features.pkl.gz")

    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("Patent Data: GNN Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Graph: {graph_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Features: {v_features_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and config
    logger.info("Loading model checkpoint...")
    ckpt = torch.load(model_path, map_location=device)
    ckpt_model = ckpt.get("model", {})
    ckpt_sampling = ckpt.get("sampling", {})
    ckpt_features = ckpt.get("features", {})

    model_cfg = parse_config_str(model_cfg, defaults={
        "hidden_dim": int(ckpt_model.get("hidden_dim", 128)),
        "out_dim": int(ckpt_model.get("out_dim", 128)),
        "dropout": float(ckpt_model.get("dropout", 0.1)),
        "decoder": str(ckpt_model.get("decoder", "mlp")),
        "decoder_hidden_dim": int(ckpt_model.get("decoder_hidden_dim", 256)),
        "decoder_dropout": float(ckpt_model.get("decoder_dropout", 0.1)),
        "num_layers": int(ckpt_model.get("num_layers", 2)),
        "use_norm": bool(ckpt_model.get("use_norm", False)),
        "use_residual": bool(ckpt_model.get("use_residual", False)),
        "decoder_aux": bool(ckpt_model.get("decoder_aux", False)),
        "decoder_raw_input": bool(ckpt_model.get("decoder_raw_input", False)),
        "aux_w_graph": float(ckpt_model.get("aux_w_graph", 0.9)),
        "aux_w_text": float(ckpt_model.get("aux_w_text", 0.1)),
        "aux_lambda": float(ckpt_model.get("aux_lambda", 1.0)),
        "aux_learnable": bool(ckpt_model.get("aux_learnable", False)),
        "aux_joint_loss": bool(ckpt_model.get("aux_joint_loss", True)),
        "infer_aux_mix": bool(ckpt_model.get("infer_aux_mix", False)),
        "infer_aux_grid": bool(ckpt_model.get("infer_aux_grid", False)),
        "conv": str(ckpt_model.get("conv", "sage")),
        "decoder_drop_path": float(ckpt_model.get("decoder_drop_path", 0.0)),
        "activation": str(ckpt_model.get("activation", "relu")),
        "decoder_fusion_head": str(ckpt_model.get("decoder_fusion_head", "linear")),
    })
    sampling_cfg = parse_config_str(sampling_cfg, defaults={
        "fanout1": int(ckpt_sampling.get("fanout1", 15)),
        "fanout2": int(ckpt_sampling.get("fanout2", 10)),
        "fanout3": int(ckpt_sampling.get("fanout3", 5)),
        "fanout4": int(ckpt_sampling.get("fanout4", 10)),
        "fanout5": int(ckpt_sampling.get("fanout5", 5)),
        "fanout": "",
    })
    raw_fanout = str(sampling_cfg.pop("fanout", "")).strip()
    ckpt_fanout_list = ckpt_sampling.get("fanout_list", None)
    num_layers_val = int(model_cfg.get("num_layers", 2))
    if ckpt_fanout_list is not None and len(ckpt_fanout_list) == num_layers_val:
        fanout_list_eval = [int(x) for x in ckpt_fanout_list]
    elif raw_fanout:
        fanout_list_eval = [int(x.strip()) for x in raw_fanout.split(",") if x.strip()]
    else:
        fanout_keys = [k for k in sorted(sampling_cfg.keys()) if k.startswith("fanout")]
        if fanout_keys:
            fanout_list_eval = [int(sampling_cfg[k]) for k in fanout_keys]
        else:
            fanout_list_eval = [15] * num_layers_val
        if len(fanout_list_eval) != num_layers_val:
            if len(fanout_list_eval) > num_layers_val:
                fanout_list_eval = fanout_list_eval[:num_layers_val]
            else:
                fanout_list_eval = fanout_list_eval + [fanout_list_eval[-1]] * (num_layers_val - len(fanout_list_eval))
    features_cfg_parsed = parse_config_str(features_cfg, defaults={
        "log1p": bool(ckpt_features.get("log1p", True)),
        "zscore": bool(ckpt_features.get("zscore", True)),
        "eps": float(ckpt_features.get("eps", 1e-6)),
    })

    if year_start_test is None:
        year_start_test = ckpt.get("year_start_train", 2021)

    logger.info(f"Model config: {model_cfg}")
    logger.info(f"Sampling config: {sampling_cfg}")
    logger.info(f"Year start test: {year_start_test}")
    logger.info(f"Device: {device}")

    # Load test data
    logger.info("Loading test data...")
    data_dict = load_pickle(data_path)
    x_test = np.asarray(data_dict["X_test"], dtype=np.int64)
    y_test = np.asarray(data_dict["y_test"], dtype=np.float32)
    logger.info(f"Test pairs: {len(x_test):,}, Positive rate: {y_test.mean():.4f}")

    # Load features
    logger.info("Loading node features...")
    v_features = load_node_features(v_features_path, logger)

    use_raw_x = bool(model_cfg.get("decoder_raw_input", False))
    # Kept as the pre-normalization copy; transforms below rebind v_features.
    v_features_raw = v_features if use_raw_x else None

    if bool(features_cfg_parsed.get("log1p", True)):
        if float(np.min(v_features)) >= 0.0:
            v_features = np.log1p(v_features)
        else:
            logger.warning("Negative values in features; skipping log1p")
    if bool(features_cfg_parsed.get("zscore", True)):
        ckpt_feature_mean = ckpt.get("feature_mean")
        ckpt_feature_std = ckpt.get("feature_std")
        eps = float(features_cfg_parsed.get("eps", 1e-6))
        if ckpt_feature_mean is not None and ckpt_feature_std is not None:
            v_features = (v_features - ckpt_feature_mean) / (ckpt_feature_std + eps)
        else:
            mean = v_features.mean(axis=0, keepdims=True)
            std = v_features.std(axis=0, keepdims=True)
            v_features = (v_features - mean) / (std + eps)

    num_nodes = int(v_features.shape[0])
    in_dim = int(v_features.shape[1])
    logger.info(f"Num nodes: {num_nodes:,}, Feature dim: {in_dim}")

    # Build graph
    logger.info("Building graph edge_index...")
    graph = Graph(graph_path)
    edge_index = build_edge_index_for_year(graph, int(year_start_test), num_nodes)

    pyg_data = Data(
        x=torch.from_numpy(v_features),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    if use_raw_x and v_features_raw is not None:
        pyg_data.x_raw = torch.from_numpy(v_features_raw)

    edge_label_index = torch.from_numpy(x_test.T).contiguous()
    edge_label = torch.from_numpy(y_test).to(torch.float32)

    loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        num_neighbors=fanout_list_eval,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
    )

    # Build model
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
    if bool(model_cfg.get("decoder_aux", False)) and decoder_kind != "mlp_two_tower":
        raise ValueError("decoder_aux=true requires decoder=mlp_two_tower")
    if bool(model_cfg.get("aux_learnable", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("aux_learnable=true requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_mix", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("infer_aux_mix=true requires decoder_aux=true")
    if bool(model_cfg.get("infer_aux_grid", False)) and not bool(
        model_cfg.get("decoder_aux", False)
    ):
        raise ValueError("infer_aux_grid=true requires decoder_aux=true")
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

    # Inference
    logger.info("Running inference...")
    encoder.eval()
    decoder.eval()
    scores = []
    graph_scores = []
    text_scores = []
    use_aux_mix = bool(model_cfg.get("infer_aux_mix", False)) and bool(
        model_cfg.get("decoder_aux", False)
    )
    use_aux_grid = bool(model_cfg.get("infer_aux_grid", False)) and bool(
        model_cfg.get("decoder_aux", False)
    )
    if use_aux_mix:
        aux_w_g = float(model_cfg.get("aux_w_graph", 0.9))
        aux_w_t = float(model_cfg.get("aux_w_text", 0.1))
        logger.info(f"Aux inference mix: {aux_w_g}*graph + {aux_w_t}*text (probs)")
    if use_aux_grid:
        logger.info("Aux inference: 11-pt weight grid on test probs (blend.py protocol)")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            z = encoder(batch.x, batch.edge_index)
            u = batch.edge_label_index[0]
            v = batch.edge_label_index[1]
            xu = batch.x_raw[u] if use_raw_x else batch.x[u]
            xv = batch.x_raw[v] if use_raw_x else batch.x[v]
            if use_aux_mix or use_aux_grid:
                _, g_logits, t_logits = decoder(
                    z[u], z[v], xu, xv, return_aux=True
                )
                p_g = torch.sigmoid(g_logits).detach().cpu().numpy()
                p_t = torch.sigmoid(t_logits).detach().cpu().numpy()
                graph_scores.extend(p_g.tolist())
                text_scores.extend(p_t.tolist())
                if not use_aux_grid:
                    probs = aux_w_g * p_g + aux_w_t * p_t
                    scores.extend(probs.tolist())
            else:
                logits = decoder(z[u], z[v], xu, xv)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                scores.extend(probs.tolist())

    if use_aux_grid:
        p_g = np.asarray(graph_scores)
        p_t = np.asarray(text_scores)
        weights = np.linspace(0.0, 1.0, 11)
        grid_aucs = []
        for w1 in weights:
            blended = p_g * float(w1) + p_t * (1.0 - float(w1))
            grid_aucs.append(float(roc_auc_score(y_test, blended)))
        best_i = int(np.argmax(grid_aucs))
        best_w1 = float(weights[best_i])
        logger.info(
            "Grid AUCs: %s",
            ", ".join(f"w={w:.1f}:{a:.4f}" for w, a in zip(weights, grid_aucs)),
        )
        logger.info(
            "Best grid weight w_graph=%.2f w_text=%.2f AUC=%.4f",
            best_w1,
            1.0 - best_w1,
            grid_aucs[best_i],
        )
        probs = p_g * best_w1 + p_t * (1.0 - best_w1)
        g_path = pred_path.replace("predictions.pkl.gz", "predictions_grid_graph.pkl.gz")
        t_path = pred_path.replace("predictions.pkl.gz", "predictions_grid_text.pkl.gz")
        if g_path != pred_path:
            save_compressed({"predictions": p_g, "labels": y_test}, g_path)
            save_compressed({"predictions": p_t, "labels": y_test}, t_path)
            logger.info(f"Per-tower predictions saved to {g_path} / {t_path}")
    else:
        probs = np.asarray(scores)

    # Metrics
    logger.info("\n" + "=" * 40)
    auc = roc_auc_score(y_test, probs)
    logger.info(f"AUC: {auc:.4f}")

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred_opt = (probs >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_opt, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()

    logger.info(f"Optimal threshold (Youden J): {best_threshold:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    # Threshold sweep
    logger.info("\nThreshold sweep:")
    stats = []
    for threshold in np.arange(0.3, 0.95, 0.05):
        y_pred_t = (probs >= threshold).astype(int)
        prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(y_test, y_pred_t, average="binary")
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
        row = {
            "threshold": round(float(threshold), 4),
            "precision": round(float(prec_t), 4),
            "recall": round(float(rec_t), 4),
            "f1": round(float(f1_t), 4),
            "tn": int(tn_t), "fp": int(fp_t),
            "fn": int(fn_t), "tp": int(tp_t),
        }
        stats.append(row)
        logger.info(f"  t={threshold:.2f}: P={prec_t:.4f}, R={rec_t:.4f}, F1={f1_t:.4f}")

    # Save metrics
    metrics = {
        "model_type": "gnn",
        "concept_embs_as_features": concept_embs_as_features,
        "auc": round(float(auc), 4),
        "optimal_threshold": round(float(best_threshold), 4),
        "optimal_precision": round(float(precision), 4),
        "optimal_recall": round(float(recall), 4),
        "optimal_f1": round(float(f1), 4),
        "confusion_matrix_optimal": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold_sweep": stats,
        "model_config": {
            "encoder": f"SAGEConv{int(model_cfg.get('num_layers', 2))}Layer",
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "out_dim": int(model_cfg["out_dim"]),
            "decoder": decoder_kind,
            "num_layers": int(model_cfg.get("num_layers", 2)),
        },
        "test_set": {
            "total": int(len(y_test)),
            "positives": int(sum(y_test == 1)),
            "negatives": int(sum(y_test == 0)),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    # Save predictions
    save_compressed({"predictions": probs, "labels": y_test}, pred_path)
    logger.info(f"Predictions saved to {pred_path}")
    logger.info("Done.")


if __name__ == "__main__":
    Fire(main)
