"""Generate concept annotations using GPT-4/Claude API with self-consistency."""

import json
import os
import random
import re
import time
from collections import Counter
from tqdm import tqdm

import pandas as pd

from patent_concept.llm_concepts.config import AnnotateConfig, parse_config_str
from patent_concept.llm_concepts.prompts.base import build_messages
from patent_concept.llm_concepts.prompts.generic import get_prompt


def parse_llm_output(text: str) -> list[str]:
    """Parse LLM output into a concept list with multiple fallback strategies.

    Tries from most specific to least specific:
    1. Direct literal_eval / JSON parse
    2. Find JSON array from END of text (avoids matching thinking fragments)
    3. Extract quoted strings
    4. Fallback: comma or newline separated
    """
    text = text.strip()
    if not text:
        return []

    # 1. Try direct Python literal_eval (matches original llama_concepts format)
    try:
        from ast import literal_eval
        result = literal_eval(text)
        if isinstance(result, list):
            return [str(c).lower().strip() for c in result if str(c).strip()]
    except (ValueError, SyntaxError):
        pass

    # 2. Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(c).lower().strip() for c in result if str(c).strip()]
    except json.JSONDecodeError:
        pass

    # 3. Find JSON array from the END of text (avoids matching thinking fragments)
    # Scan backward for ']' to find the actual output array, not brackets in thinking
    decoder = json.JSONDecoder()
    idx = len(text) - 1
    while idx >= 0:
        if text[idx] == ']':
            # Try to find matching '[' and decode
            depth = 0
            start = idx
            for j in range(idx, -1, -1):
                if text[j] == ']':
                    depth += 1
                elif text[j] == '[':
                    depth -= 1
                    if depth == 0:
                        start = j
                        break
            try:
                result, _ = decoder.raw_decode(text[start:])
                if isinstance(result, list):
                    return [str(c).lower().strip() for c in result if str(c).strip()]
            except json.JSONDecodeError:
                pass
        idx -= 1

    # 4. Extract quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    if quoted:
        return [c.lower().strip() for c in quoted]

    # 5. Fallback: comma or newline separated
    parts = re.split(r'[,\n]', text)
    parts = [p.strip().strip('- ').strip('"\'') for p in parts]
    return [p.lower() for p in parts if len(p) >= 3]


def validate_concepts(concepts: list[str]) -> list[str]:
    """Apply filters matching the llama_concepts filter chain from build.py.

    Extended with GLM-specific noise filters for UUIDs, thinking fragments,
    list markers, and other common artifacts from extended-thinking models.
    """
    # Common thinking-process phrases from GLM/extended-thinking models
    THINKING_PHRASES = {
        "let's", "let\\", "s ", "i ", "we ", "the prompt says",
        "the prompt", "however", "but rather", "is definitely",
        "s include", "s extract", "s evaluate", "s format",
        "s stick", "wait", "what about",
        "this is", "is this", "might be", "s look",
    }

    filtered = []
    for c in concepts:
        # Skip UUIDs (8-4-4-4-12 hex pattern — GLM thinking artifact)
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', c):
            continue
        # Skip if contains newlines (thinking process fragments)
        if '\n' in c:
            continue
        # Skip if starts with '-' (list item fragments from thinking)
        if c.startswith('-') or c == '-':
            continue
        # Skip if starts with comma or "type=" (GLM artifact)
        if c.startswith(',') or c.startswith('type=') or c == 'type=':
            continue
        # Skip if contains "\n" literal (GLM thinking artifact)
        if '\\n' in c:
            continue
        # Skip if ends with backslash (truncated quote from thinking)
        if c.endswith('\\') or c.endswith("'"):
            continue
        # Skip if contains thinking-process phrases
        c_lower = c.lower().strip()
        if any(phrase in c_lower for phrase in THINKING_PHRASES):
            continue
        # Skip if starts with a single char + space (e.g. "s evaluate", "i think")
        if re.match(r'^[a-z] ', c_lower):
            continue
        # Skip fragments like "ll go with", "ve been", "'s include"
        if re.match(r"^(ll |ve |re |s )", c_lower):
            continue
        # Skip if starts with parenthesis (model's inline reasoning like "(biological phenomenon - ...)")
        if c.startswith('(') or c.startswith('['):
            continue
        # Skip if contains "maybe", "or maybe" (model uncertainty phrases)
        if re.search(r'\b(maybe|or maybe|but,?)\b', c_lower):
            continue
        # Skip if starts with ". " (sentence fragment)
        if c_lower.startswith('. ') or c_lower.startswith('.'):
            continue
        # Skip if contains sentence-ending punctuation (not a concept noun phrase)
        if re.search(r'[.!?]', c):
            continue
        # Skip if contains "or " connecting 3+ items (GLM list artifact like "X, Y, or Z")
        if ' or ' in c and len(c.split(',')) >= 3:
            continue
        # NonAsciiFilter: only ASCII characters
        if not all(ord(ch) < 128 for ch in c):
            continue
        # LlamaFilter: no standalone numbers
        if re.findall(r'\b\d+\b', c):
            continue
        # WordFilter: 2-10 words (min=2 matches paper's llama_concepts min_words=2)
        n_words = len(c.split())
        if n_words < 2 or n_words > 10:
            continue
        # LengthFilter: min 4 chars (2-letter words like "formula i" still pass)
        if len(c) < 4:
            continue
        filtered.append(c)
    return filtered


def _call_openai(messages: list[dict], model: str, temperature: float = 0.7) -> str:
    """Call OpenAI-compatible API and return the response text.

    Supports custom base_url via OPENAI_BASE_URL env var (for local models).
    """
    import openai
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    client = openai.OpenAI(base_url=base_url, api_key=api_key) if base_url else openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content


def _call_anthropic(messages: list[dict], model: str, temperature: float = 0.7) -> str:
    """Call Anthropic-compatible API and return the response text.

    Supports custom base_url via ANTHROPIC_BASE_URL env var (for local models like GLM).
    Uses ANTHROPIC_AUTH_TOKEN (preferred) or ANTHROPIC_API_KEY for authentication.
    """
    import anthropic
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(base_url=base_url, api_key=api_key) if base_url else anthropic.Anthropic()
    # Anthropic API separates system prompt from messages
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=model,
        system=system_msg,
        messages=user_messages,
        temperature=temperature,
        max_tokens=512,
    )
    # Handle ThinkingBlock from models with extended thinking (e.g. GLM)
    for block in reversed(response.content):
        if hasattr(block, 'text'):
            return block.text
    return str(response.content[-1])


API_CALLERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def annotate_single(
    abstract: str,
    api_caller,
    prompt_template,
    num_samples: int = 3,
    temperature: float = 0.7,
    consistency_threshold: float = 0.6,
) -> tuple[list[str], list[list[str]]]:
    """Annotate a single abstract with self-consistency.

    Returns (consensus_concepts, raw_outputs).
    """
    messages = build_messages(prompt_template, abstract)
    raw_outputs = []

    for _ in range(num_samples):
        try:
            response = api_caller(messages, temperature=temperature)
            concepts = parse_llm_output(response)
            concepts = validate_concepts(concepts)
            raw_outputs.append(concepts)
        except Exception as e:
            print(f"API call failed: {e}, retrying in 5s...")
            time.sleep(5)
            try:
                response = api_caller(messages, temperature=temperature)
                concepts = parse_llm_output(response)
                concepts = validate_concepts(concepts)
                raw_outputs.append(concepts)
            except Exception as e2:
                print(f"API call failed again: {e2}, skipping")
                raw_outputs.append([])

    if not raw_outputs:
        return [], raw_outputs

    # Self-consistency: keep concepts appearing in >= threshold of samples
    all_concepts = [c for output in raw_outputs for c in output]
    counts = Counter(all_concepts)
    min_count = max(1, int(len(raw_outputs) * consistency_threshold))
    consensus = [c for c, n in counts.items() if n >= min_count]

    return consensus, raw_outputs


def generate_annotations(
    input_path: str,
    output_path: str,
    api_provider: str = "openai",
    api_model: str = "gpt-4",
    abstract_col: str = "abstract",
    id_col: str = "id",
    num_samples: int = 3,
    consistency_threshold: float = 0.6,
    max_abstracts: int = 500,
    batch_size: int = 10,
    temperature: float = 0.7,
    resume: bool = True,
    seed: int = 42,
    n_few_shot: int = 3,
    annotate_cfg: str | None = None,
):
    """Batch-generate concept annotations using GPT-4/Claude API.

    For each abstract:
    1. Query API num_samples times (temperature > 0 for diversity)
    2. Intersect: keep concepts appearing in >= consistency_threshold of samples
    3. Validate: apply llama_concepts filters (no numbers, ASCII, word count)
    4. Write to JSONL: {"id": ..., "concepts": [...], "raw_outputs": [...]}

    Args:
        input_path: Path to cleaned CSV with abstracts
        output_path: Path to output JSONL annotations file
        api_provider: "openai" or "anthropic"
        api_model: Model name (e.g. "gpt-4", "claude-3-5-sonnet-20241022")
        abstract_col: Column name for abstract text
        id_col: Column name for record ID
        num_samples: Self-consistency repetitions per abstract
        consistency_threshold: Fraction of samples a concept must appear in
        max_abstracts: Max number of abstracts to annotate
        batch_size: Progress logging batch size
        temperature: Sampling temperature for diversity
        resume: Skip already-annotated IDs
        seed: Random seed for reproducibility
        n_few_shot: Number of few-shot examples (0 to disable)
        annotate_cfg: Override config as comma-separated key=value string
    """
    # Parse config overrides
    if annotate_cfg:
        cfg = parse_config_str(annotate_cfg, {
            "api_provider": api_provider,
            "model": api_model,
            "num_samples": num_samples,
            "consistency_threshold": consistency_threshold,
            "max_abstracts": max_abstracts,
            "temperature": temperature,
            "batch_size": batch_size,
        })
        api_provider = cfg["api_provider"]
        api_model = cfg["model"]
        num_samples = cfg["num_samples"]
        consistency_threshold = cfg["consistency_threshold"]
        max_abstracts = cfg["max_abstracts"]
        temperature = cfg["temperature"]
        batch_size = cfg["batch_size"]

    random.seed(seed)

    # Load input data
    df = pd.read_csv(input_path, dtype={abstract_col: str})
    print(f"Loaded {len(df):,} records from {input_path}")

    # Resume: load already-annotated IDs
    annotated_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    annotated_ids.add(record[id_col])
        print(f"Resuming: {len(annotated_ids)} already annotated")

    # Filter to unannotated records
    remaining = df[~df[id_col].isin(annotated_ids)]
    if max_abstracts > 0:
        remaining = remaining.head(max_abstracts - len(annotated_ids))
    print(f"Annotating {len(remaining)} abstracts (provider={api_provider}, model={api_model})")

    if len(remaining) == 0:
        print("No new abstracts to annotate")
        return

    # Setup
    prompt_template = get_prompt(n_few_shot=n_few_shot)
    api_caller = API_CALLERS[api_provider]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Process
    mode = "a" if resume and annotated_ids else "w"
    stats = {"total": 0, "concepts_per_abstract": []}

    with open(output_path, mode, encoding="utf-8") as f:
        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Annotating"):
            abstract = str(row[abstract_col])
            record_id = str(row[id_col])

            if not abstract or len(abstract.strip()) < 20:
                continue

            consensus, raw_outputs = annotate_single(
                abstract=abstract,
                api_caller=lambda msgs, temperature=temperature: api_caller(msgs, api_model, temperature),
                prompt_template=prompt_template,
                num_samples=num_samples,
                temperature=temperature,
                consistency_threshold=consistency_threshold,
            )

            record = {
                id_col: record_id,
                abstract_col: abstract[:500],
                "concepts": consensus,
                "raw_outputs": raw_outputs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["total"] += 1
            stats["concepts_per_abstract"].append(len(consensus))

            if stats["total"] % batch_size == 0:
                avg = sum(stats["concepts_per_abstract"]) / len(stats["concepts_per_abstract"])
                print(f"Progress: {stats['total']}/{len(remaining)}, avg concepts/abstract: {avg:.1f}")

    avg = sum(stats["concepts_per_abstract"]) / max(1, len(stats["concepts_per_abstract"]))
    print(f"Done. Annotated {stats['total']} abstracts, avg {avg:.1f} concepts/abstract")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_annotations)
