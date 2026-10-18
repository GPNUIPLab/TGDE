"""vLLM-based concept extraction backend for high-throughput inference.

Uses vLLM's continuous batching, PagedAttention, and efficient KV cache
management for 5-10x throughput vs naive HuggingFace inference.

Requires: pip install vllm (Linux + Ampere GPU only)
"""

import gzip
import io
import logging
import os
import pickle
import sys

import pandas as pd
import torch
from tqdm import tqdm

from patent_concept.llm_concepts.annotate import parse_llm_output, validate_concepts
from patent_concept.llm_concepts.config import MODEL_REGISTRY
from patent_concept.llm_concepts.inference import merge_chunks_to_csv, setup_logger
from patent_concept.llm_concepts.prompts.generic import GenericPrompt


def _vllm_generate_quiet(llm, prompts, sampling_params):
    """Call vLLM generate while suppressing its built-in tqdm progress bar."""
    _stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        outputs = llm.generate(prompts, sampling_params)
    finally:
        sys.stderr = _stderr
    return outputs


def extract_concepts_vllm(
    input_path: str,
    output_path: str,
    model_id: str = "llama2-13b",
    adapter_path: str | None = None,
    quantization: str = "8bit",
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    step_size: int = 1000,
    start: int = 0,
    end: int | None = None,
    concepts_col: str = "llama_concepts",
    id_col: str = "id",
    abstract_col: str = "abstract",
    log_file: str | None = None,
    seed: int = 42,
    model_path: str | None = None,
    batch_size: int = 64,
):
    """Run batch concept extraction using vLLM backend.

    vLLM handles continuous batching internally — the batch_size param
    controls how many prompts are submitted per vLLM engine call.

    Args:
        input_path: Path to input CSV with abstracts
        output_path: Path to output CSV with llama_concepts column
        model_id: Key in MODEL_REGISTRY
        adapter_path: Path to LoRA adapter (vLLM supports LoRA natively)
        quantization: "8bit", "4bit", or "none"
        max_new_tokens: Max output tokens per abstract
        temperature: Sampling temperature
        step_size: Abstracts per chunk for resume
        start: Resume start index
        end: Resume end index (None = all)
        concepts_col: Output column name
        id_col: Record ID column name
        abstract_col: Abstract text column name
        log_file: Log file path
        seed: Random seed
        model_path: Override model path
        batch_size: Prompts per vLLM generate call
    """
    from vllm import LLM, SamplingParams

    model_config = MODEL_REGISTRY[model_id]
    model_name_or_path = model_path or model_config.model_name

    log_file = log_file or os.path.join(os.path.dirname(output_path) or ".", "logs", "llm_inference_vllm.log")
    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("LLM Concept Extraction - vLLM Backend")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id} ({model_name_or_path})")
    logger.info(f"Quantization: {quantization}, Adapter: {adapter_path or 'none'}")

    # Load data
    df = pd.read_csv(input_path, dtype={abstract_col: str})
    total = len(df)
    if end is None:
        end = total
    logger.info(f"Total abstracts: {total:,}, Processing: [{start}, {end})")

    # vLLM engine kwargs — lower gpu_memory_utilization to leave room for
    # CUDA Graphs and temporary allocations (default 0.9 is too aggressive
    # for 7B+ models on 24GB GPUs)
    llm_kwargs = {
        "model": model_name_or_path,
        "trust_remote_code": model_config.trust_remote_code,
        "seed": seed,
        "max_model_len": min(model_config.max_length, 4096),
        "gpu_memory_utilization": 0.85,
    }
    if quantization == "4bit":
        llm_kwargs["quantization"] = "awq"
    if adapter_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    logger.info("Loading vLLM engine...")
    llm = LLM(**llm_kwargs)
    logger.info("vLLM engine ready.")

    prompt_template = GenericPrompt()
    use_greedy = temperature < 0.05
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0 if use_greedy else temperature,
        seed=seed,
    )

    # Chunks directory
    chunks_dir = os.path.join(os.path.dirname(output_path) or ".", "llm_concepts")
    os.makedirs(chunks_dir, exist_ok=True)

    for i in range(start, end, step_size):
        chunk_end = min(i + step_size, end)
        save_path = os.path.join(chunks_dir, f"concepts_{i:06d}.pkl.gz")

        if os.path.exists(save_path):
            logger.info(f"Skipping chunk {i}-{chunk_end} (already exists)")
            continue

        partial_df = df.iloc[i:chunk_end]

        # Build prompts
        prompts = []
        record_ids = []
        for _, row in partial_df.iterrows():
            abstract = str(row[abstract_col]) if not pd.isna(row[abstract_col]) else ""
            if not abstract or len(abstract.strip()) < 20:
                continue
            record_ids.append(str(row[id_col]))
            prompt = prompt_template.format_user(abstract[:2000])
            full_prompt = prompt_template.system_prompt + "\n\n" + prompt
            prompts.append(full_prompt)

        # Unified progress bar for this chunk
        pbar = tqdm(total=len(prompts), desc=f"Chunk {i}-{chunk_end}", unit="abs", file=sys.stderr)

        # vLLM continuous batching — submit in sub-batches with OOM retry
        store = {}
        current_bs = batch_size
        j = 0
        while j < len(prompts):
            batch_prompts = prompts[j:j + current_bs]
            batch_ids = record_ids[j:j + current_bs]

            try:
                outputs = _vllm_generate_quiet(llm, batch_prompts, sampling_params)
                for rid, output in zip(batch_ids, outputs):
                    generated_text = output.outputs[0].text
                    concepts = parse_llm_output(generated_text)
                    concepts = validate_concepts(concepts)
                    store[rid] = concepts
                pbar.update(len(batch_prompts))
                j += current_bs
                # If retrying with smaller batch succeeded, don't restore size mid-chunk
                # (next OOM would just retry again anyway)
            except Exception as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower() and current_bs > 1:
                    # OOM: shrink batch and retry this slice (no progress lost)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_bs = max(1, current_bs // 2)
                    logger.warning(f"vLLM OOM (bs={current_bs}), retrying with bs={new_bs}")
                    current_bs = new_bs
                    continue  # retry same slice with smaller batch
                else:
                    logger.warning(f"vLLM batch failed: {e}, skipping {len(batch_prompts)} samples")
                    pbar.update(len(batch_prompts))
                    j += current_bs

        pbar.close()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with gzip.open(save_path, "wb") as f:
            pickle.dump(store, f)
        logger.info(f"Saved chunk {i}-{chunk_end}: {len(store)} patents → {save_path}")

    # Merge all chunks
    logger.info("Merging chunks into final CSV...")
    merge_chunks_to_csv(
        chunks_dir=chunks_dir,
        output_csv_path=output_path,
        input_csv_path=input_path,
        concepts_col=concepts_col,
        id_col=id_col,
    )
    logger.info("vLLM inference complete.")
