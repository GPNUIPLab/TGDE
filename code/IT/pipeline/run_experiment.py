"""Run experiments defined in YAML configuration files.

Each experiment gets its own isolated directory under output/experiments/{group}/{name}/
containing: model.pt, metrics.json, predictions.pkl.gz, train.log, eval.log, config.yaml

Usage:
    # Run all experiments in a config:
    python IT/pipeline/run_experiment.py --config IT/pipeline/experiments/ablation_decoder.yaml

    # Run specific experiments:
    python IT/pipeline/run_experiment.py --config ... --experiments A1_dot,A2_mlp_default

    # Skip already completed experiments:
    python IT/pipeline/run_experiment.py --config ... --skip_existing

    # Only evaluate (skip training):
    python IT/pipeline/run_experiment.py --config ... --skip_train

    # Only train (skip evaluation):
    python IT/pipeline/run_experiment.py --config ... --skip_eval
"""

import gzip
import json
import logging
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from fire import Fire

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, "experiments")
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))


def _setup_logger(log_file, name="experiment"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not logger.handlers:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _render_path(template, context):
    for key, value in context.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _build_mlp_train_cmd(exp_config, exp_dir, shared_data, concept_source, seed=None, device=0):
    model_type = exp_config.get("model_type", "combi")
    source_dir = os.path.join(OUTPUT_DIR, concept_source)
    cmd_parts = ["python", "IT/pipeline/train_model.py"]

    cmd_parts.extend([
        f"--model_type={model_type}",
        f"--concept_source={concept_source}",
        f"--data_path={os.path.join(source_dir, 'data.pkl')}",
        f"--dropout={exp_config.get('dropout', 0.1)}",
        f"--lr={exp_config.get('lr', 0.001)}",
        f"--batch_size={exp_config.get('batch_size', 4096)}",
        f"--num_epochs={exp_config.get('num_epochs', 500)}",
        f"--pos_ratio={exp_config.get('pos_ratio', 0.3)}",
        f"--step_size={exp_config.get('step_size', 40)}",
        f"--gamma={exp_config.get('gamma', 0.8)}",
        f"--sliding_window={exp_config.get('sliding_window', 5)}",
        f"--save_model={os.path.join(exp_dir, 'model.pt')}",
        f"--log_file={os.path.join(exp_dir, 'train.log')}",
        f"--device={device}",
    ])
    if seed is not None:
        cmd_parts.append(f"--seed={seed}")

    emb_comb = exp_config.get("emb_comb_strategy", "concat")
    cmd_parts.append(f"--emb_comb_strategy={emb_comb}")

    loss_fn = exp_config.get("loss_fn", "bce")
    if loss_fn != "bce":
        cmd_parts.append(f"--loss_fn={loss_fn}")

    layers = exp_config.get("layers")
    if layers:
        cmd_parts.append(f'--layers="{layers}"')

    if model_type in ("baseline", "combi"):
        cmd_parts.append(f"--emb_f_train_path={os.path.join(source_dir, 'features.pkl.gz')}")
        cmd_parts.append(f"--emb_f_test_path={os.path.join(source_dir, 'features.pkl.gz')}")
    if model_type in ("pure_embs", "combi"):
        cmd_parts.append(f"--emb_c_train_path={os.path.join(source_dir, 'concept_embs_train.pkl.gz')}")
        cmd_parts.append(f"--emb_c_test_path={os.path.join(source_dir, 'concept_embs_test.pkl.gz')}")

    return " ".join(cmd_parts)


def _build_mlp_eval_cmd(exp_config, exp_dir, shared_data, concept_source, device=0):
    model_type = exp_config.get("model_type", "combi")
    source_dir = os.path.join(OUTPUT_DIR, concept_source)

    model_path = os.path.join(exp_dir, "model.pt.best")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, "model.pt")

    cmd_parts = ["python", "IT/pipeline/evaluate.py"]
    cmd_parts.extend([
        f"--model_type={model_type}",
        f"--concept_source={concept_source}",
        f"--data_path={os.path.join(source_dir, 'data.pkl')}",
        f"--model_path={model_path}",
        f"--metrics_path={os.path.join(exp_dir, 'metrics.json')}",
        f"--pred_path={os.path.join(exp_dir, 'predictions.pkl.gz')}",
        f"--log_file={os.path.join(exp_dir, 'eval.log')}",
        f"--dropout={exp_config.get('dropout', 0.1)}",
        f"--device={device}",
    ])

    loss_fn = exp_config.get("loss_fn", "bce")
    if loss_fn != "bce":
        cmd_parts.append(f"--loss_fn={loss_fn}")

    emb_comb = exp_config.get("emb_comb_strategy", "concat")
    if emb_comb != "concat":
        cmd_parts.append(f"--emb_comb_strategy={emb_comb}")

    layers = exp_config.get("layers")
    if layers:
        cmd_parts.append(f'--layers="{layers}"')

    if model_type == "pure_embs":
        cmd_parts.append(f"--features_path={os.path.join(source_dir, 'concept_embs_test.pkl.gz')}")
    elif model_type in ("baseline", "combi"):
        cmd_parts.append(f"--features_path={os.path.join(source_dir, 'features.pkl.gz')}")
    if model_type == "combi":
        cmd_parts.append(f"--concept_embs_path={os.path.join(source_dir, 'concept_embs_test.pkl.gz')}")

    return " ".join(cmd_parts)


def _build_gnn_train_cmd(exp_config, exp_dir, shared_data, concept_source, seed=None, device=0):
    source_dir = os.path.join(OUTPUT_DIR, concept_source)
    concept_embs_as_features = exp_config.get("concept_embs_as_features", True)
    year_start_train = exp_config.get("year_start_train", 2018)

    gnn_dir = "gnn_embs" if concept_embs_as_features else "gnn_topo"

    train_cfg_str = exp_config.get("train_cfg", "batch_size=4096,lr=3e-4,num_epochs=20")
    if "num_workers" not in train_cfg_str:
        train_cfg_str += ",num_workers=2"
    model_cfg_str = exp_config.get("model_cfg", "hidden_dim=256,out_dim=128,dropout=0.1,decoder=mlp")
    sampling_cfg_raw = exp_config.get("sampling_cfg", "fanout1=15,fanout2=10")

    cmd_parts = [
        "python", "-m", "patent_concept.model.gnn.train_pyg", "train",
        f'--graph_path="{os.path.join(source_dir, "edges.pkl")}"',
        f'--data_path="{os.path.join(source_dir, "data.pkl")}"',
        f"--year_start_train={year_start_train}",
        f'--train="{train_cfg_str}"',
        f'--model="{model_cfg_str}"',
        f'--sampling="{sampling_cfg_raw}"',
        f"--save_model_path={os.path.join(exp_dir, gnn_dir)}",
        f"--log_file={os.path.join(exp_dir, 'train.log')}",
    ]

    if concept_embs_as_features:
        cmd_parts.append(
            f'--v_features_path="{os.path.join(source_dir, "concept_embs_train.pkl.gz")}"'
        )
        cmd_parts.append("--concept_embs_as_features=True")
    else:
        cmd_parts.append(
            f'--v_features_path="{os.path.join(source_dir, "features.pkl.gz")}"'
        )
        cmd_parts.append("--concept_embs_as_features=False")

    if seed is not None:
        cmd_parts.append(f"--seed={seed}")

    return " ".join(cmd_parts)


def _build_gnn_eval_cmd(exp_config, exp_dir, shared_data, concept_source, device=0):
    source_dir = os.path.join(OUTPUT_DIR, concept_source)
    concept_embs_as_features = exp_config.get("concept_embs_as_features", True)
    year_start_test = exp_config.get("year_start_test", 2021)

    gnn_dir = "gnn_embs" if concept_embs_as_features else "gnn_topo"

    model_cfg_str = exp_config.get("model_cfg", "hidden_dim=256,out_dim=128,dropout=0.1,decoder=mlp")
    sampling_cfg_raw = exp_config.get("sampling_cfg", "fanout1=15,fanout2=10")
    features_cfg_str = exp_config.get("features_cfg", "log1p=true,zscore=true")

    embs_flag = f"--concept_embs_as_features={str(concept_embs_as_features)}"

    model_path = os.path.join(exp_dir, gnn_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, gnn_dir, "model.pt")

    cmd_parts = [
        "python", "IT/pipeline/evaluate_gnn.py",
        f"--model_path={model_path}",
        f"--metrics_path={os.path.join(exp_dir, 'metrics.json')}",
        f"--pred_path={os.path.join(exp_dir, 'predictions.pkl.gz')}",
        f"--log_file={os.path.join(exp_dir, 'eval.log')}",
        f"--concept_source={concept_source}",
        f"--year_start_test={year_start_test}",
        f'--model_cfg="{model_cfg_str}"',
        f'--sampling_cfg="{sampling_cfg_raw}"',
        f'--features_cfg="{features_cfg_str}"',
        f"--eval_batch_size={exp_config.get('eval_batch_size', 16384)}",
        embs_flag,
    ]

    if concept_embs_as_features:
        cmd_parts.append(
            f'--v_features_path="{os.path.join(source_dir, "concept_embs_test.pkl.gz")}"'
        )
    else:
        cmd_parts.append(
            f'--v_features_path="{os.path.join(source_dir, "features.pkl.gz")}"'
        )

    cmd_parts.append(f"--device={device}")

    return " ".join(cmd_parts)


def _run_command(cmd, logger):
    logger.info(f"CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        return False
    return True


def _save_config(exp_config, exp_dir, concept_source):
    config_path = os.path.join(exp_dir, "config.yaml")
    full_config = dict(exp_config)
    full_config["concept_source"] = concept_source
    full_config["created_at"] = datetime.now().isoformat()
    with open(config_path, "w") as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)


def _check_completed(exp_dir):
    return os.path.exists(os.path.join(exp_dir, "metrics.json"))


def _check_model_exists(exp_dir, model_type, exp_config):
    """Check if a trained model file already exists (for skipping re-training
    when only evaluation failed previously)."""
    if model_type == "gnn":
        concept_embs_as_features = exp_config.get("concept_embs_as_features", True)
        gnn_dir = "gnn_embs" if concept_embs_as_features else "gnn_topo"
        for name in ("best_model.pt", "model.pt"):
            if os.path.exists(os.path.join(exp_dir, gnn_dir, name)):
                return True
        return False
    for name in ("model.pt.best", "model.pt"):
        if os.path.exists(os.path.join(exp_dir, name)):
            return True
    return False


def _collect_group_summary(group_dir, group_name):
    results = []
    seed_groups = {}

    for exp_name in sorted(os.listdir(group_dir)):
        exp_dir = os.path.join(group_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        metrics_path = os.path.join(exp_dir, "metrics.json")
        config_path = os.path.join(exp_dir, "config.yaml")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        exp_config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                exp_config = yaml.safe_load(f) or {}
        results.append({
            "name": exp_name,
            "auc": metrics.get("auc"),
            "optimal_f1": metrics.get("optimal_f1"),
            "optimal_precision": metrics.get("optimal_precision"),
            "optimal_recall": metrics.get("optimal_recall"),
            "optimal_threshold": metrics.get("optimal_threshold"),
            "config": exp_config,
        })

        suffix_match = re.match(r"^(.+)_s(\d+)$", exp_name)
        if suffix_match:
            stem = suffix_match.group(1)
            seed_groups.setdefault(stem, []).append(metrics)

    results.sort(key=lambda x: x.get("auc") or 0, reverse=True)

    aggregated = []
    for stem, metrics_list in sorted(seed_groups.items()):
        aucs = [m.get("auc") for m in metrics_list if m.get("auc") is not None]
        f1s = [m.get("optimal_f1") for m in metrics_list if m.get("optimal_f1") is not None]
        precs = [m.get("optimal_precision") for m in metrics_list if m.get("optimal_precision") is not None]
        recs = [m.get("optimal_recall") for m in metrics_list if m.get("optimal_recall") is not None]

        def _stats(vals):
            if not vals:
                return None, None
            arr = np.array(vals, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        auc_mean, auc_std = _stats(aucs)
        f1_mean, f1_std = _stats(f1s)
        prec_mean, prec_std = _stats(precs)
        rec_mean, rec_std = _stats(recs)
        aggregated.append({
            "name": stem,
            "num_seeds": len(metrics_list),
            "auc_mean": round(auc_mean, 4) if auc_mean is not None else None,
            "auc_std": round(auc_std, 4) if auc_std is not None else None,
            "f1_mean": round(f1_mean, 4) if f1_mean is not None else None,
            "f1_std": round(f1_std, 4) if f1_std is not None else None,
            "precision_mean": round(prec_mean, 4) if prec_mean is not None else None,
            "precision_std": round(prec_std, 4) if prec_std is not None else None,
            "recall_mean": round(rec_mean, 4) if rec_mean is not None else None,
            "recall_std": round(rec_std, 4) if rec_std is not None else None,
        })

    summary_path = os.path.join(group_dir, "summary.json")
    summary_data = {"group": group_name, "results": results}
    if aggregated:
        summary_data["seed_aggregation"] = aggregated
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return results


def main(
    config,
    experiments=None,
    skip_train=False,
    skip_eval=False,
    skip_existing=True,
    summarize=True,
    concept_source_override=None,
    seed=None,
    device=0,
):
    if device is not None and str(device) != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    group = cfg["group"]
    concept_source = concept_source_override or cfg.get("concept_source", "llm")
    base_dir = cfg.get("base_dir", EXPERIMENTS_DIR)
    base_dir = base_dir.replace("{EXPERIMENT_BASE}", EXPERIMENTS_DIR)

    shared_data = {}
    for key, value in cfg.get("shared_data", {}).items():
        shared_data[key] = _render_path(value, {
            "EXPERIMENT_BASE": EXPERIMENTS_DIR,
            "concept_source": concept_source,
        })

    group_dir = os.path.join(base_dir, group)
    os.makedirs(group_dir, exist_ok=True)

    log_file = os.path.join(group_dir, "run_experiment.log")
    logger = _setup_logger(log_file, name=f"exp_{group}")

    logger.info(f"Experiment group: {group}")
    logger.info(f"Concept source: {concept_source}")
    logger.info(f"Group directory: {group_dir}")
    if seed is not None:
        logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")

    if experiments:
        if isinstance(experiments, (list, tuple)):
            exp_names = [e.strip() for e in experiments]
        else:
            exp_names = [e.strip() for e in str(experiments).split(",")]
    else:
        exp_names = list(cfg.get("experiments", {}).keys())

    exp_configs = cfg.get("experiments", {})
    completed, failed, skipped = 0, 0, 0

    for exp_name in exp_names:
        exp_config = exp_configs.get(exp_name)
        if exp_config is None:
            logger.warning(f"Experiment '{exp_name}' not found in config")
            continue

        exp_suffix = f"{exp_name}_s{seed}" if seed is not None else exp_name
        exp_dir = os.path.join(group_dir, exp_suffix)
        os.makedirs(exp_dir, exist_ok=True)

        if skip_existing and _check_completed(exp_dir):
            logger.info(f"Skipping {exp_suffix}: already completed")
            skipped += 1
            continue

        description = exp_config.get("description", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_suffix} — {description}")
        logger.info(f"Directory: {exp_dir}")

        _save_config(exp_config, exp_dir, concept_source)

        model_type = exp_config.get("model_type", "gnn")

        if not skip_train:
            if _check_model_exists(exp_dir, model_type, exp_config):
                logger.info(f"Model already exists, skipping training for {exp_name}")
            else:
                if model_type == "gnn":
                    cmd = _build_gnn_train_cmd(exp_config, exp_dir, shared_data, concept_source, seed=seed, device=device)
                else:
                    cmd = _build_mlp_train_cmd(exp_config, exp_dir, shared_data, concept_source, seed=seed, device=device)

                success = _run_command(cmd, logger)
                if not success:
                    logger.error(f"Training failed for {exp_name}")
                    failed += 1
                    continue

        if not skip_eval:
            if model_type == "gnn":
                cmd = _build_gnn_eval_cmd(exp_config, exp_dir, shared_data, concept_source, device=device)
            else:
                cmd = _build_mlp_eval_cmd(exp_config, exp_dir, shared_data, concept_source, device=device)

            success = _run_command(cmd, logger)
            if not success:
                logger.error(f"Evaluation failed for {exp_name}")
                failed += 1
                continue

        completed += 1

    if summarize:
        logger.info(f"\n{'='*60}")
        logger.info("Collecting group summary...")
        results = _collect_group_summary(group_dir, group)
        logger.info(f"Summary for '{group}':")
        for r in results:
            auc = r.get("auc", "N/A")
            f1 = r.get("optimal_f1", "N/A")
            logger.info(f"  {r['name']}: AUC={auc}, F1={f1}")
        logger.info(f"Summary saved to {os.path.join(group_dir, 'summary.json')}")

    logger.info(f"\nDone! Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    Fire(main)
