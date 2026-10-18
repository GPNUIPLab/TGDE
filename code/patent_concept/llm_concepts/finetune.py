"""LoRA fine-tuning for concept extraction using PEFT + bitsandbytes."""

import json
import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import random

import numpy as np
import torch
from tqdm import tqdm

from patent_concept.llm_concepts.config import (
    MODEL_REGISTRY,
    LoRAConfig,
    ModelConfig,
    TrainConfig,
    parse_config_str,
)


def prepare_training_data(
    annotations_path: str,
    output_dir: str,
    val_split: float = 0.1,
    prompt_template: str | None = None,
    seed: int = 42,
):
    """Convert annotations JSONL to HuggingFace Dataset for fine-tuning.

    Reads the JSONL from annotate.py and converts each record into
    input/output pairs suitable for causal LM fine-tuning.

    Output format per line (JSONL):
    {"text": "<prompt>\n\nAbstract: ...\n\nConcepts: ['concept1', 'concept2']"}

    The output concept list matches the original llama_concepts format
    (Python list string) so downstream literal_eval() continues to work.
    """
    from datasets import Dataset

    random.seed(seed)

    default_prompt = prompt_template or (
        "Extract technical concepts from the following abstract. "
        "Return a Python list of lowercase strings.\n\n"
    )

    records = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading annotations"):
            if not line.strip():
                continue
            record = json.loads(line)
            concepts = record.get("concepts", [])
            if not concepts:
                continue

            abstract = record.get("abstract", "")
            concept_str = str(concepts)  # e.g. "['phase transformation', 'titanium dioxide']"
            text = f"{default_prompt}Abstract: {abstract}\n\nConcepts: {concept_str}"
            records.append({"text": text})

    random.shuffle(records)
    n_val = max(1, int(len(records) * val_split))
    train_records = records[:-n_val]
    val_records = records[-n_val:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    for path, data in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Training data: {len(train_records)} train, {len(val_records)} val")
    print(f"Saved to {train_path} and {val_path}")

    # Also save as HuggingFace Dataset
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)
    ds_dir = os.path.join(output_dir, "dataset")
    train_ds.save_to_disk(os.path.join(ds_dir, "train"))
    val_ds.save_to_disk(os.path.join(ds_dir, "val"))
    print(f"HuggingFace Dataset saved to {ds_dir}")

    return len(train_records), len(val_records)


def get_quantization_config(quantization: str):
    """Get BitsAndBytesConfig for the specified quantization level."""
    from transformers import BitsAndBytesConfig

    if quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def finetune(
    model_id: str = "llama2-13b",
    data_path: str | None = None,
    output_dir: str | None = None,
    annotations_path: str | None = None,
    lora_cfg: str | None = None,
    train_cfg: str | None = None,
    quantization: str = "8bit",
    val_split: float = 0.1,
    seed: int = 42,
    log_file: str | None = None,
    prepare_data_only: bool = False,
    model_path: str | None = None,
):
    """Fine-tune a causal LM with LoRA + quantization for concept extraction.

    Uses HuggingFace PEFT + bitsandbytes. The LoRA adapter is saved
    separately from the base model, enabling model-agnostic switching.

    Args:
        model_id: Key in MODEL_REGISTRY (e.g. "llama2-13b", "llama3-8b", "qwen2.5-7b")
        data_path: Path to pre-processed training data directory (with dataset/ subdir).
                   If None, will be generated from annotations_path.
        output_dir: Path to save LoRA adapter and training artifacts
        annotations_path: Path to annotations JSONL (used if data_path is None)
        lora_cfg: LoRA config as comma-separated key=value (e.g. "r=16,lora_alpha=32")
        train_cfg: Training config as comma-separated key=value (e.g. "lr=2e-4,num_epochs=3")
        quantization: "8bit", "4bit", or "none"
        val_split: Validation split ratio when preparing data from annotations
        seed: Random seed
        log_file: Path to log file
        prepare_data_only: If True, only prepare training data without fine-tuning
    """
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Resolve model config
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")
    model_config = MODEL_REGISTRY[model_id]
    print(f"Model: {model_id} ({model_config.model_name})")

    # Parse configs
    lora_defaults = {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.01,
        "bias": "none", "task_type": "CAUSAL_LM",
    }
    lora_params = parse_config_str(lora_cfg, lora_defaults)

    train_defaults = {
        "num_epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 16,
        "lr": 2e-4, "weight_decay": 0.01, "warmup_ratio": 0.03,
        "lr_scheduler": "cosine", "max_length": 1024,
    }
    train_params = parse_config_str(train_cfg, train_defaults)

    output_dir = output_dir or os.path.join(os.path.dirname(data_path or annotations_path or "."), "llm_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Prepare training data if needed
    if data_path is None:
        if annotations_path is None:
            raise ValueError("Either data_path or annotations_path must be provided")
        data_path = os.path.join(output_dir, "training_data")
        prepare_training_data(
            annotations_path=annotations_path,
            output_dir=data_path,
            val_split=val_split,
            seed=seed,
        )

    if prepare_data_only:
        print("Data preparation complete (prepare_data_only=True)")
        return

    # Step 2: Load dataset
    ds_dir = os.path.join(data_path, "dataset")
    if os.path.exists(ds_dir):
        train_ds = load_from_disk(os.path.join(ds_dir, "train"))
        val_ds = load_from_disk(os.path.join(ds_dir, "val"))
    else:
        # Load from JSONL
        from datasets import Dataset
        train_records = []
        with open(os.path.join(data_path, "train.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
        val_records = []
        with open(os.path.join(data_path, "val.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
        train_ds = Dataset.from_list(train_records)
        val_ds = Dataset.from_list(val_records)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")

    # Step 3: Load model + tokenizer
    model_name_or_path = model_path or model_config.model_name
    print(f"Loading model: {model_name_or_path}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float16)

    quant_config = get_quantization_config(quantization)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        # {"": 0} avoids dispatch_model .to() which fails on bitsandbytes models
        model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # Step 4: Apply LoRA
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_params.get("target_modules", model_config.target_modules)
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(";")]

    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 5: Tokenize dataset
    max_length = train_params["max_length"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Step 6: Train
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_params["num_epochs"],
        per_device_train_batch_size=train_params["batch_size"],
        per_device_eval_batch_size=train_params["batch_size"],
        gradient_accumulation_steps=train_params["gradient_accumulation_steps"],
        learning_rate=train_params["lr"],
        weight_decay=train_params["weight_decay"],
        warmup_ratio=train_params["warmup_ratio"],
        lr_scheduler_type=train_params["lr_scheduler"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=seed,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Step 7: Save LoRA adapter only
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # Save config for inference
    config_record = {
        "model_id": model_id,
        "base_model": model_config.model_name,
        "quantization": quantization,
        "lora_r": lora_params["r"],
        "lora_alpha": lora_params["lora_alpha"],
        "target_modules": target_modules,
    }
    with open(os.path.join(output_dir, "finetune_config.json"), "w") as f:
        json.dump(config_record, f, indent=2)


if __name__ == "__main__":
    from fire import Fire
    Fire(finetune)
