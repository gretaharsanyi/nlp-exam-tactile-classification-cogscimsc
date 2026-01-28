from typing import Dict, Any, List
import pandas as pd

from src.config import STYLE_SEEDS, MIN_CHARS, MAX_CHARS

def parse_anchors(anchors_field: str) -> List[str]:
    if anchors_field is None:
        return []
    s = str(anchors_field).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [a.strip() for a in s.split(";") if a.strip()]

def anchors_for_prompt(anchors_field: str) -> str:
    anchors = parse_anchors(anchors_field)
    if not anchors:
        return ""
    return "; ".join(a.lower() for a in anchors)

def prompt_for_row(row: pd.Series, variation_note: str) -> Dict[str, Any]:
    labels = row["labels"]
    literal = int(row["literal"])
    specificity = int(row["specificity"])
    consistent = int(row["consistent"])
    anchor_regime = str(row["anchor_regime"]).lower().strip()

    anchors_raw = row.get("anchors", "")
    anchors_natural = anchors_for_prompt(anchors_raw)

    metaphor_rule = (
        "Use exactly one comparison marker: 'like', 'as if', or 'as though'."
        if literal == 0 else
        "Do NOT use 'like', 'as if', or 'as though'. No comparisons."
    )

    specificity_rule = (
        "Include ONE body location AND one simple intensity or time cue (e.g., 'mild', 'sharp', 'brief', 'for minutes')."
        if specificity == 1 else
        "Stay vague: avoid specific body locations and avoid precise intensity/time details."
    )

    consistency_rule = (
        "Keep it consistent with the modalities."
        if consistent == 1 else
        "Make it clearly inconsistent: Include at least one modality-relevant, explicit contradiction cue."
    )

    if anchor_regime == "strict":
        anchor_rule = (
            "Anchors MUST appear (case-insensitive). Simple grammatical forms are allowed. "
            f"Anchors: {anchors_natural}"
        )
    elif anchor_regime == "paraphrase":
        anchor_rule = (
            "Avoid using the anchor word(s) themselves (even in lowercase); use synonyms/paraphrases instead."
            f"Anchors: {anchors_raw}"
        )
    else:
        anchor_rule = "No anchor constraints."

    system = (
        "Write short first-person bodily sensation descriptions for a synthetic dataset. "
        "Be concrete and descriptive, not literary. Output JSON only."
    )

    user = f"""
Generate ONE single-sentence description of a felt sensation for these modalities: {labels}.

Rules:
- {metaphor_rule}
- {specificity_rule}
- {consistency_rule}
- {anchor_rule}

Style:
- 1 sentence, {MIN_CHARS}–{MAX_CHARS} characters
- first person
- describe a sensation happening to the body
- avoid reusing the same sentence structure as previous outputs for this job.
- simple wording, minimal adjectives, not poetic
- no medical advice/diagnosis, no mention of labels/datasets

{variation_note}

Return JSON only:
{{"items": ["<description>"]}}
""".strip()

    return {"system": system, "user": user}

def choose_variation_note() -> str:
    import random
    style = random.choice(STYLE_SEEDS)
    return f"Write {style}. Keep it short and descriptive."
