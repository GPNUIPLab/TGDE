"""Batch concept extraction inference using fine-tuned LLM.

Processes patent abstracts in chunks with resume support,
following the same chunk/pkl.gz pattern as generate_embeddings.py.

Optimizations over naive single-sample inference:
- Batched generation (configurable batch_size)
- System prompt KV cache reuse across samples
- Flash Attention 2 auto-detection
- bfloat16→float16 dtype fix for 8-bit quantization
- Greedy decoding for near-zero temperatures
- Reduced default max_new_tokens (128 vs 256)
"""

import gzip
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from patent_concept.llm_concepts.annotate import parse_llm_output, validate_concepts
from patent_concept.llm_concepts.config import MODEL_REGISTRY, parse_config_str
from patent_concept.llm_concepts.prompts.base import build_messages
from patent_concept.llm_concepts.prompts.generic import GenericPrompt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "BM", "pipeline", "output")


def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger("llm_inference")
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _flash_attention_available() -> bool:
    """Check if flash-attn is installed and GPU supports it (Ampere+)."""
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 8  # Ampere = 8.x
        return False
    except ImportError:
        return False


def load_model(model_id: str, adapter_path: str | None = None, quantization: str = "8bit", model_path: str | None = None):
    """Load base model + optional LoRA adapter for inference.

    Returns (model, tokenizer).

    Fixes:
    - Auto-detects Flash Attention 2 support
    - Fixes bfloat16→float16 cast overhead with 8-bit quantization
      (bitsandbytes 8-bit only supports float16 compute; bfloat16 models
       cause implicit casts on every forward pass)
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = MODEL_REGISTRY[model_id]
    model_name_or_path = model_path or model_config.model_name
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }

    # Flash Attention 2: auto-detect and enable
    if _flash_attention_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if quantization == "8bit":
        # bitsandbytes 8-bit only supports float16 compute.
        # bfloat16 models trigger implicit casts every forward pass.
        if torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16
            model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif quantization == "4bit":
        # 4-bit quantization supports bfloat16 compute via bnb_4bit_compute_dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_concepts(
    model,
    tokenizer,
    abstract: str,
    prompt_template: GenericPrompt,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> list[str]:
    """Generate concept list for a single abstract."""
    prompt = prompt_template.format_user(abstract)
    full_prompt = prompt_template.system_prompt + "\n\n" + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Greedy decoding for near-zero temperatures (avoids sampling overhead)
    use_greedy = temperature < 0.05

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0 if use_greedy else temperature,
            do_sample=not use_greedy,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    concepts = parse_llm_output(generated_text)
    concepts = validate_concepts(concepts)
    return concepts


def generate_concepts_batch(
    model,
    tokenizer,
    abstracts: list[str],
    prompt_template: GenericPrompt,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> list[list[str]]:
    """Generate concept lists for a batch of abstracts.

    Uses batched tokenization + generation for much higher GPU utilization
    than single-sample inference. For large batches, the speedup can be 3-5x.
    """
    if not abstracts:
        return []

    # Build full prompts
    full_prompts = []
    for abstract in abstracts:
        prompt = prompt_template.format_user(abstract)
        full_prompts.append(prompt_template.system_prompt + "\n\n" + prompt)

    # Batch tokenize with left padding (required for decoder-only models)
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length - max_new_tokens,
        )
    finally:
        tokenizer.padding_side = saved_padding_side
    # input_lengths: actual token count per sample (excluding padding)
    input_lengths = inputs["attention_mask"].sum(dim=1)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    use_greedy = temperature < 0.05

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0 if use_greedy else temperature,
            do_sample=not use_greedy,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each sample's generated portion
    all_concepts = []
    for i, output_ids in enumerate(outputs):
        generated_ids = output_ids[input_lengths[i]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        concepts = parse_llm_output(generated_text)
        concepts = validate_concepts(concepts)
        all_concepts.append(concepts)

    return all_concepts


def _is_quantized(model) -> bool:
    """Check if model uses bitsandbytes quantized linear layers."""
    try:
        import bitsandbytes as bnb
        for module in model.modules():
            if isinstance(module, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)):
                return True
    except ImportError:
        pass
    return False


def _precompute_system_kv_cache(model, tokenizer, system_prompt: str, max_new_tokens: int = 128):
    """Pre-compute KV cache for the system prompt to reuse across samples.

    Returns (past_key_values, system_input_ids) or (None, None) if the model
    doesn't support extractable KV cache (e.g. with quantization — causes
    reshape errors during generate with past_key_values).
    """
    # Quantized models: KV cache reuse causes "cannot reshape tensor of 0 elements" errors
    if _is_quantized(model):
        return None, None

    try:
        system_inputs = tokenizer(
            system_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length - max_new_tokens - 512,  # room for user prompt + generation
        )
        system_inputs = {k: v.to(model.device) for k, v in system_inputs.items()}

        with torch.no_grad():
            outputs = model(**system_inputs, use_cache=True)

        return outputs.past_key_values, system_inputs["input_ids"]
    except Exception:
        return None, None


def generate_concepts_cached(
    model,
    tokenizer,
    abstract: str,
    prompt_template: GenericPrompt,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    system_kv_cache=None,
    system_input_ids=None,
) -> list[str]:
    """Generate concept list for a single abstract, reusing system prompt KV cache."""
    prompt = prompt_template.format_user(abstract)
    user_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, return_length=False)
    user_inputs = {k: v.to(model.device) for k, v in user_inputs.items()}

    use_greedy = temperature < 0.05

    if system_kv_cache is not None and system_input_ids is not None:
        # Concatenate system + user input IDs for generate
        combined_input_ids = torch.cat([system_input_ids, user_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([
            torch.ones(1, system_input_ids.shape[1], device=model.device, dtype=torch.long),
            user_inputs.get("attention_mask", torch.ones_like(user_inputs["input_ids"])),
        ], dim=1)

        generate_inputs = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "past_key_values": system_kv_cache,
        }
        input_len = combined_input_ids.shape[1]
    else:
        # Fallback: no cache, build full prompt
        full_prompt = prompt_template.system_prompt + "\n\n" + prompt
        generate_inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True,
                                    max_length=tokenizer.model_max_length - max_new_tokens)
        generate_inputs = {k: v.to(model.device) for k, v in generate_inputs.items()}
        input_len = generate_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **generate_inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0 if use_greedy else temperature,
            do_sample=not use_greedy,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    concepts = parse_llm_output(generated_text)
    concepts = validate_concepts(concepts)
    return concepts


def merge_chunks_to_csv(
    chunks_dir: str,
    output_csv_path: str,
    input_csv_path: str,
    concepts_col: str = "llama_concepts",
    id_col: str = "id",
):
    """Merge all chunk pkl.gz files into a single CSV with llama_concepts column."""
    logger = logging.getLogger("llm_inference")

    all_concepts = {}
    chunk_files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".pkl.gz"))
    logger.info(f"Merging {len(chunk_files)} chunk files...")

    for chunk_file in tqdm(chunk_files, desc="Merging chunks", file=sys.stderr):
        with gzip.open(os.path.join(chunks_dir, chunk_file), "rb") as f:
            chunk_data = pickle.load(f)
        all_concepts.update(chunk_data)

    logger.info(f"Total records with concepts: {len(all_concepts):,}")

    df = pd.read_csv(input_csv_path, dtype={"abstract": str})
    logger.info(f"Original CSV: {len(df):,} records")

    def format_concepts(record_id):
        concepts = all_concepts.get(record_id)
        if concepts is None:
            return ""
        return str(concepts)

    df[concepts_col] = df[id_col].astype(str).apply(format_concepts)

    non_empty = (df[concepts_col] != "").sum()
    logger.info(f"Records with {concepts_col}: {non_empty:,} / {len(df):,} ({non_empty/len(df)*100:.1f}%)")

    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved to {output_csv_path}")


def _process_chunk_single(
    model, tokenizer, partial_df, prompt_template,
    max_new_tokens, temperature, id_col, abstract_col,
    system_kv_cache=None, system_input_ids=None,
    desc="Chunk",
):
    """Process a chunk one sample at a time (with optional KV cache reuse)."""
    store = {}
    use_cache = system_kv_cache is not None

    for _, row in tqdm(partial_df.iterrows(), total=len(partial_df), desc=desc, file=sys.stderr):
        record_id = str(row[id_col])
        abstract = str(row[abstract_col]) if not pd.isna(row[abstract_col]) else ""

        if not abstract or len(abstract.strip()) < 20:
            continue

        try:
            if use_cache:
                concepts = generate_concepts_cached(
                    model=model,
                    tokenizer=tokenizer,
                    abstract=abstract[:2000],
                    prompt_template=prompt_template,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    system_kv_cache=system_kv_cache,
                    system_input_ids=system_input_ids,
                )
            else:
                concepts = generate_concepts(
                    model=model,
                    tokenizer=tokenizer,
                    abstract=abstract[:2000],
                    prompt_template=prompt_template,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            store[record_id] = concepts
        except Exception as e:
            logging.getLogger("llm_inference").warning(f"Error processing {record_id}: {e}")
            continue

    return store


def _process_chunk_batched(
    model, tokenizer, partial_df, prompt_template,
    max_new_tokens, temperature, id_col, abstract_col, batch_size,
    desc="Chunk",
):
    """Process a chunk using batched generation."""
    logger = logging.getLogger("llm_inference")
    store = {}

    # Pre-collect valid abstracts and their IDs
    records = []
    for _, row in partial_df.iterrows():
        record_id = str(row[id_col])
        abstract = str(row[abstract_col]) if not pd.isna(row[abstract_col]) else ""
        if abstract and len(abstract.strip()) >= 20:
            records.append((record_id, abstract[:2000]))

    # Single progress bar counting abstracts, updating per batch
    pbar = tqdm(total=len(records), desc=desc, unit="abs", file=sys.stderr)

    current_bs = batch_size
    i = 0
    while i < len(records):
        # Adaptive batch size: shrink on OOM, never grow above original
        batch = records[i:i + current_bs]
        batch_ids = [r[0] for r in batch]
        batch_abstracts = [r[1] for r in batch]

        try:
            batch_concepts = generate_concepts_batch(
                model=model,
                tokenizer=tokenizer,
                abstracts=batch_abstracts,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for rid, concepts in zip(batch_ids, batch_concepts):
                store[rid] = concepts
            pbar.update(len(batch))
            i += len(batch)
            # Recover batch size after successful batch (up to original)
            current_bs = min(current_bs + 1, batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_bs > 1:
                logger.warning(f"OOM at batch_size={current_bs}, reducing to {current_bs // 2}")
                torch.cuda.empty_cache()
                current_bs = max(current_bs // 2, 1)
                continue  # retry same records with smaller batch
            elif current_bs == 1:
                # Single sample also OOM — skip this one
                logger.warning(f"OOM on single sample, skipping: {batch_ids[0]}")
                torch.cuda.empty_cache()
                pbar.update(1)
                i += 1
            else:
                raise
        except Exception as e:
            logger.warning(f"Batch failed (size={current_bs}): {e}, falling back to single")
            for rid, abstract in batch:
                try:
                    concepts = generate_concepts(
                        model=model,
                        tokenizer=tokenizer,
                        abstract=abstract,
                        prompt_template=prompt_template,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    store[rid] = concepts
                except Exception as e2:
                    logger.warning(f"Error processing {rid}: {e2}")
            pbar.update(len(batch))
            i += len(batch)

    pbar.close()
    return store


def extract_concepts_batch(
    input_path: str,
    output_path: str,
    model_id: str = "llama2-13b",
    adapter_path: str | None = None,
    quantization: str = "8bit",
    batch_size: int = 1,
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
    use_kv_cache: bool = True,
    backend: str = "vllm",
):
    """Run batch concept extraction on patent abstracts.

    Loads the base model + LoRA adapter, processes abstracts in chunks,
    saves intermediate results for resume support.

    Output: adds `llama_concepts` column to CSV (Python list string format).

    Args:
        input_path: Path to patents.cleaned.csv (or patents.concepts.csv)
        output_path: Path to output CSV with llama_concepts column
        model_id: Key in MODEL_REGISTRY
        adapter_path: Path to fine-tuned LoRA adapter directory
        quantization: "8bit", "4bit", or "none"
        batch_size: Batch size for generation (1 = single-sample, >1 = batched)
        max_new_tokens: Max output tokens per abstract (default 128)
        temperature: Sampling temperature (0.1 = nearly deterministic)
        step_size: Abstracts per chunk for resume
        start: Resume start index
        end: Resume end index (None = all)
        concepts_col: Name for the output column
        id_col: Column name for record ID
        abstract_col: Column name for abstract text
        log_file: Path to log file
        seed: Random seed
        model_path: Override model path (instead of MODEL_REGISTRY)
        use_kv_cache: Pre-compute and reuse system prompt KV cache (single-sample mode)
        backend: "huggingface" (default) or "vllm" (requires vllm package)
    """
    # Delegate to vLLM backend if requested
    if backend == "vllm":
        from patent_concept.llm_concepts.inference_vllm import extract_concepts_vllm
        return extract_concepts_vllm(
            input_path=input_path,
            output_path=output_path,
            model_id=model_id,
            adapter_path=adapter_path,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            step_size=step_size,
            start=start,
            end=end,
            concepts_col=concepts_col,
            id_col=id_col,
            abstract_col=abstract_col,
            log_file=log_file,
            seed=seed,
            model_path=model_path,
            batch_size=batch_size,
        )

    torch.manual_seed(seed)

    log_file = log_file or os.path.join(os.path.dirname(output_path) or ".", "logs", "llm_inference.log")
    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("LLM Concept Extraction - Batch Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}, Quantization: {quantization}")
    logger.info(f"Adapter: {adapter_path or 'none (base model only)'}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Step size: {step_size}, Start: {start}, End: {end or 'all'}")
    logger.info(f"Batch size: {batch_size}, Max new tokens: {max_new_tokens}")
    logger.info(f"KV cache reuse: {use_kv_cache}, Backend: {backend}")
    logger.info(f"Flash Attention: {'available' if _flash_attention_available() else 'not available'}")

    # Load data
    df = pd.read_csv(input_path, dtype={abstract_col: str})
    total = len(df)
    if end is None:
        end = total
    logger.info(f"Total abstracts: {total:,}, Processing: [{start}, {end})")

    # Chunks directory
    chunks_dir = os.path.join(os.path.dirname(output_path) or ".", "llm_concepts")
    os.makedirs(chunks_dir, exist_ok=True)

    # Load model
    logger.info("Loading model (this may take a few minutes)...")
    model, tokenizer = load_model(model_id, adapter_path, quantization, model_path)
    logger.info("Model loaded.")

    prompt_template = GenericPrompt()

    # Pre-compute system prompt KV cache (only useful in single-sample mode)
    # Skip for quantized models: bitsandbytes layers have incompatible KV cache formats
    system_kv_cache = None
    system_input_ids = None
    if use_kv_cache and batch_size <= 1 and quantization == "none":
        logger.info("Pre-computing system prompt KV cache...")
        system_kv_cache, system_input_ids = _precompute_system_kv_cache(
            model, tokenizer, prompt_template.system_prompt, max_new_tokens
        )
        if system_kv_cache is not None:
            logger.info(f"KV cache ready (system prompt: {system_input_ids.shape[1]} tokens)")
        else:
            logger.info("KV cache not supported for this model/quantization, skipping")

    # Process in chunks
    for i in range(start, end, step_size):
        chunk_end = min(i + step_size, end)
        save_path = os.path.join(chunks_dir, f"concepts_{i:06d}.pkl.gz")

        # Skip if already exists (resume support)
        if os.path.exists(save_path):
            logger.info(f"Skipping chunk {i}-{chunk_end} (already exists)")
            continue

        partial_df = df.iloc[i:chunk_end]

        chunk_desc = f"Chunk {i}-{chunk_end}"
        if batch_size > 1:
            store = _process_chunk_batched(
                model, tokenizer, partial_df, prompt_template,
                max_new_tokens, temperature, id_col, abstract_col, batch_size,
                desc=chunk_desc,
            )
        else:
            store = _process_chunk_single(
                model, tokenizer, partial_df, prompt_template,
                max_new_tokens, temperature, id_col, abstract_col,
                system_kv_cache, system_input_ids,
                desc=chunk_desc,
            )

        # Save chunk
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with gzip.open(save_path, "wb") as f:
            pickle.dump(store, f)
        logger.info(f"Saved chunk {i}-{chunk_end}: {len(store)} patents → {save_path}")

    # Merge all chunks into final CSV
    logger.info("Merging chunks into final CSV...")
    merge_chunks_to_csv(
        chunks_dir=chunks_dir,
        output_csv_path=output_path,
        input_csv_path=input_path,
        concepts_col=concepts_col,
        id_col=id_col,
    )
    logger.info("Batch inference complete.")


if __name__ == "__main__":
    from fire import Fire
    Fire(extract_concepts_batch)
