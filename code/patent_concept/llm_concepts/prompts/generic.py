"""Generic technical concept extraction prompt template for patent abstracts."""

import os
from dataclasses import dataclass, field

import pandas as pd


SYSTEM_PROMPT = """\
You are an expert in technical concept extraction from patent abstracts.
Given a patent abstract, identify ALL technical concepts, methods, \
properties, materials, and phenomena mentioned in the text.

Requirements for each concept:
- A multi-word noun phrase (2-6 words)
- Specific technical terms (not generic terms like "result", "experiment", "study")
- No standalone numbers, abbreviations, or single words
- Lowercase

Return the concepts as a JSON array of strings.
Example: ["liver x receptor", "pharmaceutical compositions", "reactive carboxyl group"]\
"""

USER_TEMPLATE = """\
Abstract: {abstract}

Extract the technical concepts from this abstract. Return only the JSON array.\
"""


@dataclass
class GenericPrompt:
    system_prompt: str = SYSTEM_PROMPT
    user_template: str = USER_TEMPLATE
    few_shot_examples: list[dict] = field(default_factory=list)

    def format_user(self, abstract: str) -> str:
        return self.user_template.format(abstract=abstract)

    def format_few_shot(self) -> list[dict]:
        messages = []
        for ex in self.few_shot_examples:
            messages.append({
                "role": "user",
                "content": self.format_user(ex["abstract"]),
            })
            messages.append({
                "role": "assistant",
                "content": str(ex["concepts"]),
            })
        return messages


def load_few_shot_from_annotations(
    annotations_path: str | None = None,
    n_examples: int = 3,
) -> list[dict]:
    """Load ground-truth annotations as few-shot examples.

    Reads data/concept_benchmark/annotations.csv and groups by Work ID,
    returning up to n_examples complete abstract→concepts pairs.

    Requires access to the original works CSV to get abstracts.
    """
    if annotations_path is None:
        annotations_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "concept_benchmark", "annotations.csv"
        )

    if not os.path.exists(annotations_path):
        print(f"Annotations file not found: {annotations_path}, skipping few-shot examples")
        return []

    df = pd.read_csv(annotations_path)
    work_ids = df["Work"].unique()[:n_examples]

    # Try to load abstracts from the original works CSV
    works_csv = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "table", "materials-science.llama2.works.csv"
    )

    abstracts_map = {}
    if os.path.exists(works_csv):
        works_df = pd.read_csv(works_csv, dtype={"abstract": str})
        for _, row in works_df.iterrows():
            abstracts_map[row["id"]] = row["abstract"]

    examples = []
    for work_id in work_ids:
        abstract = abstracts_map.get(work_id, "")
        if not abstract:
            continue

        concepts = df[df["Work"] == work_id]["Annotator1"].dropna().tolist()
        if not concepts:
            continue

        examples.append({
            "abstract": abstract[:1000],
            "concepts": [c.lower().strip() for c in concepts],
        })

    return examples


def get_prompt(
    annotations_path: str | None = None,
    n_few_shot: int = 3,
) -> GenericPrompt:
    """Create a GenericPrompt with optional few-shot examples."""
    examples = load_few_shot_from_annotations(annotations_path, n_few_shot)
    return GenericPrompt(few_shot_examples=examples)
