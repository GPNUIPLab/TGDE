"""Configuration for LLM concept extraction: model registry, LoRA, annotation configs."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture-specific settings for a registered model."""
    model_name: str
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    torch_dtype: str = "float16"
    max_length: int = 4096
    trust_remote_code: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama2-13b": ModelConfig(
        model_name="meta-llama/Llama-2-13b-hf",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        torch_dtype="float16",
        max_length=4096,
    ),
    "llama3-8b": ModelConfig(
        model_name="meta-llama/Meta-Llama-3-8B",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        torch_dtype="bfloat16",
        max_length=8192,
    ),
    "qwen2.5-7b": ModelConfig(
        model_name="Qwen/Qwen2.5-7B",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        torch_dtype="bfloat16",
        max_length=32768,
        trust_remote_code=True,
    ),
    "qwen3-0.6b": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        torch_dtype="bfloat16",
        max_length=32768,
        trust_remote_code=True,
    ),
}


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class AnnotateConfig:
    api_provider: str = "openai"
    model: str = "gpt-4"
    batch_size: int = 10
    num_samples: int = 3
    consistency_threshold: float = 0.6


@dataclass
class TrainConfig:
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_length: int = 2048


def parse_config_str(config_str: str, defaults: dict) -> dict:
    """Parse comma-separated key=value config string into a dict.

    Values are auto-typed: int, float, bool, or str.
    Missing keys fall back to defaults.

    Example: "r=16,lora_alpha=32,lora_dropout=0.05"
    """
    result = dict(defaults)
    if not config_str:
        return result

    for pair in config_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key not in defaults:
            print(f"Warning: unknown config key '{key}', ignoring")
            continue

        default_val = defaults[key]
        if isinstance(default_val, bool):
            result[key] = value.lower() in ("true", "1", "yes")
        elif isinstance(default_val, int):
            result[key] = int(value)
        elif isinstance(default_val, float):
            result[key] = float(value)
        elif isinstance(default_val, list):
            result[key] = [s.strip() for s in value.split(";")]
        else:
            result[key] = value

    return result
