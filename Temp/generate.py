# src/generate.py
import os, json, time, hashlib, random, re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from openai import OpenAI
import anthropic

# Import seed set / helpers from your existing file
try:
    from src.anchors import SEEDS  # must exist
except Exception:
    # If your project structure uses relative imports (when running as a script),
    # allow fallback:
    from anchors import SEEDS  # type: ignore

# Optional helpers if you already implemented them in anchors.py
try:
    from src.anchors import expand_anchor_family  # type: ignore
except Exception:
    expand_anchor_family = None  # type: ignore

try:
    from src.anchors import get_family_variants  # type: ignore
except Exception:
    get_family_variants = None  # type: ignore


# ---------------------------
# Config
# ---------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

MIN_CHARS = 40
MAX_CHARS = 110

METAPHOR_MARKERS = [" like ", " as if ", " as though "]

OUT_SAMPLES = "data/generated/raw/samples.jsonl"
OUT_REJECTS = "data/generated/raw/rejects.jsonl"
JOB_LOG = "logs/job_log.csv"

# If True: clear samples/rejects/joblog at the start of each run (OVERWRITE)
RESET_OUTPUTS_EACH_RUN = False

# Less “book prose”, more description-like
STYLE_SEEDS = [
    "plain and direct",
    "concise",
    "matter-of-fact",
    "simple and descriptive",
    "neutral tone",
    "slightly vivid but not poetic",
]


# ---------------------------
# Validation helpers
# ---------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_anchors(anchors_field: str) -> List[str]:
    if anchors_field is None:
        return []
    s = str(anchors_field).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [a.strip() for a in s.split(";") if a.strip()]

def anchors_for_prompt(anchors_field: str) -> str:
    """
    Make anchors look natural in prompts.
    Example: 'PRESS DOWN; TINGLE' -> 'press down; tingle'
    """
    anchors = parse_anchors(anchors_field)
    if not anchors:
        return ""
    return "; ".join(a.lower() for a in anchors)

def has_metaphor_marker(text: str) -> bool:
    t = f" {text.lower()} "
    return any(m in t for m in METAPHOR_MARKERS)

def _seed_terms_upper() -> Set[str]:
    """
    Uppercase canonical vocabulary from SEEDS.
    """
    out: Set[str] = set()
    for _, terms in SEEDS.items():
        for t in terms:
            out.add(str(t).strip().upper())
    return out

_SEED_UPPER = _seed_terms_upper()

def _simple_inflections_for_single_word(word: str) -> Set[str]:
    """
    Very lightweight inflection set (not perfect English, but good enough for your anchor families).
    Returns lowercase variants.
    """
    w = word.lower().strip()
    if not w:
        return set()
    variants = {w}

    # basic suffixes
    variants.add(w + "s")
    variants.add(w + "ed")
    variants.add(w + "ing")

    # a couple of common endings
    if w.endswith("e"):
        variants.add(w[:-1] + "ing")  # ache -> aching
        variants.add(w + "d")         # ache -> ached
    if w.endswith("y") and len(w) > 2:
        variants.add(w[:-1] + "ies")

    return {v for v in variants if v}

def strict_anchor_present(text: str, anchor: str) -> bool:
    """
    STRICT regime:
    Accept if the text contains the anchor OR any family variant derived from SEEDS.
    - For multi-word anchors: accept 'press down', 'pressed down', 'pressing down', etc.
    - For single-word anchors: accept family variants (e.g., ACHING -> ache/aches/ached/aching),
      and also accept direct inclusion of the canonical seed word if it appears in the seed set.
    """
    t = text.lower()
    a = anchor.strip()
    if not a:
        return True

    a_upper = a.upper()
    a_lower = a.lower()

    # Multi-word anchor: match "press down" with simple verb inflection on first token
    if " " in a_lower:
        tokens = a_lower.split()
        first = re.escape(tokens[0])
        rest = r"\s+".join(re.escape(tok) for tok in tokens[1:])
        pattern = rf"\b{first}(?:s|es|ing|ed)?\b\s+{rest}\b"
        return re.search(pattern, t) is not None

    # Single-word anchor: try to use your anchors.py helper if present
    variants: Set[str] = set()
    if expand_anchor_family is not None:
        try:
            variants = {v.lower() for v in expand_anchor_family(a)}  # type: ignore
        except Exception:
            variants = set()

    if not variants and get_family_variants is not None:
        try:
            variants = {v.lower() for v in get_family_variants(a, SEEDS)}  # type: ignore
        except Exception:
            variants = set()

    # If no helper: build variants using SEEDS membership + simple inflections
    if not variants:
        # If the anchor itself is in the seed vocabulary, treat it as canonical
        # and accept its simple inflections.
        if a_upper in _SEED_UPPER:
            variants |= _simple_inflections_for_single_word(a_lower)

        # Also: if the anchor is e.g. ACHING, but SEEDS contains ACHE and ACHING,
        # accept any seed term that “looks related” by substring overlap.
        # This is deliberately permissive.
        for term_upper in _SEED_UPPER:
            term = term_upper.lower()
            if term == a_lower:
                variants |= _simple_inflections_for_single_word(term)
                continue
            if a_lower in term or term in a_lower:
                variants |= _simple_inflections_for_single_word(term)

    # Always include direct match of the anchor word itself
    variants.add(a_lower)

    # Match any variant as a token
    for v in sorted(variants, key=len, reverse=True):
        if re.search(rf"\b{re.escape(v)}\b", t):
            return True

    return False

def paraphrase_anchor_leak(text: str, anchor: str) -> bool:
    """
    PARAPHRASE regime:
    Only flag leaks if the EXACT ALL-CAPS anchor appears in the output.
    This avoids rejecting legitimate natural language like "jolts" when the anchor is "JOLT".

    - Single-word: reject if "JOLT" appears as token.
    - Multi-word: reject if exact phrase "PRESS DOWN" appears (case-sensitive).
    """
    if not anchor:
        return False

    txt = text
    a = anchor.strip()

    if " " in a:
        # phrase leak: require exact phrase match (case-sensitive) with word boundaries
        pattern = rf"(?<!\w){re.escape(a)}(?!\w)"
        return re.search(pattern, txt) is not None

    # token leak: exact all-caps token only
    pattern = rf"(?<!\w){re.escape(a)}(?!\w)"
    return re.search(pattern, txt) is not None

def passes_validation(text: str, row: pd.Series) -> Tuple[bool, str]:
    txt = normalize_ws(text)
    if len(txt) < MIN_CHARS:
        return False, "too_short"
    if len(txt) > MAX_CHARS:
        return False, "too_long"

    literal = int(row["literal"])
    if literal == 0 and not has_metaphor_marker(txt):
        return False, "missing_metaphor_marker"
    if literal == 1 and has_metaphor_marker(txt):
        return False, "metaphor_marker_in_literal"

    anchor_regime = str(row["anchor_regime"]).lower().strip()
    anchors = parse_anchors(row.get("anchors", ""))

    if anchor_regime == "strict":
        missing = [a for a in anchors if not strict_anchor_present(txt, a)]
        if missing:
            return False, f"missing_anchors:{','.join(missing[:3])}"

    elif anchor_regime == "paraphrase":
        violating = [a for a in anchors if paraphrase_anchor_leak(txt, a)]
        if violating:
            return False, f"anchor_leak:{','.join(violating[:3])}"

    elif anchor_regime == "drift":
        pass
    else:
        return False, f"unknown_anchor_regime:{anchor_regime}"

    return True, "ok"


# ---------------------------
# Prompting
# ---------------------------

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

    # You said you *don't* want contradiction checks in general — keep this mild.
    # If consistent==0, we allow “a bit odd” but not force contradictions.
    consistency_rule = (
        "Keep it consistent with the modalities."
        if consistent == 1 else
        "It can feel slightly odd or mixed, but do not force dramatic contradictions."
    )

    if anchor_regime == "strict":
        anchor_rule = (
            "Anchors MUST appear (case-insensitive). Simple grammatical forms are allowed. "
            f"Anchors: {anchors_natural}"
        )
    elif anchor_regime == "paraphrase":
        anchor_rule = (
            "Do NOT output the anchors exactly as written. Use synonyms or paraphrases of the anchor word/words. "
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
- 1 sentence, 40–110 characters
- first person
- describe a sensation happening to the body
- simple wording, minimal adjectives, not poetic
- no medical advice/diagnosis, no mention of labels/datasets

{variation_note}

Return JSON only:
{{"items": ["<description>"]}}
""".strip()

    return {"system": system, "user": user}


# ---------------------------
# API calls
# ---------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_openai(system: str, user: str) -> str:
    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=system,
        input=user,
    )
    return resp.output_text

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_claude(system: str, user: str) -> str:
    msg = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=220,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


# ---------------------------
# Robust JSON parsing
# ---------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None

    raw = s.strip()

    # 1) direct JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) fenced ```json ... ```
    m = _JSON_FENCE_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) try to extract the first {...} block (best-effort)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


# ---------------------------
# IO helpers
# ---------------------------

def file_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def reset_outputs_if_needed():
    if not RESET_OUTPUTS_EACH_RUN:
        return
    for p in [OUT_SAMPLES, OUT_REJECTS, JOB_LOG]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("")

def load_done_job_ids(samples_path: str) -> set:
    if RESET_OUTPUTS_EACH_RUN:
        return set()
    if not os.path.exists(samples_path):
        return set()

    done = set()
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                done.add(o.get("job_id"))
            except Exception:
                continue
    return done


# ---------------------------
# Main loop
# ---------------------------

def run(plan_csv: str, max_jobs: Optional[int] = None):
    reset_outputs_if_needed()

    df = pd.read_csv(plan_csv)
    done_jobs = load_done_job_ids(OUT_SAMPLES)

    job_logs = []
    n_jobs_done = 0

    for _, row in df.iterrows():
        job_id = row["job_id"]
        if job_id in done_jobs:
            continue

        if max_jobs is not None and n_jobs_done >= max_jobs:
            break

        provider = str(row["generator"]).lower().strip()
        n_samples = int(row["n_samples"])

        accepted = 0
        rejected = 0
        attempts = 0
        model = ""

        while accepted < n_samples and attempts < n_samples * 8:
            attempts += 1

            style = random.choice(STYLE_SEEDS)
            variation_note = f"Write {style}. Keep it short and descriptive."

            spec = prompt_for_row(row, variation_note)
            system, user = spec["system"], spec["user"]
            prompt_id = file_hash(system + "\n" + user)

            if provider == "openai":
                raw = call_openai(system, user)
                model = OPENAI_MODEL
            elif provider == "claude":
                raw = call_claude(system, user)
                model = CLAUDE_MODEL
            else:
                raise ValueError(f"Unknown provider: {provider}")

            js = safe_json_load(raw)
            if not js or "items" not in js or not isinstance(js["items"], list) or len(js["items"]) == 0:
                append_jsonl(OUT_REJECTS, {
                    "job_id": job_id, "provider": provider, "model": model,
                    "reason": "bad_json", "raw": raw[:1000], "prompt_id": prompt_id
                })
                rejected += 1
                continue

            text = str(js["items"][0])

            ok, reason = passes_validation(text, row)
            if not ok:
                append_jsonl(OUT_REJECTS, {
                    "job_id": job_id, "provider": provider, "model": model,
                    "reason": reason, "text": normalize_ws(text), "prompt_id": prompt_id
                })
                rejected += 1
                continue

            out = {
                "job_id": job_id,
                "provider": provider,
                "model": model,
                "prompt_id": prompt_id,
                "text": normalize_ws(text),
                "labels": row["labels"],
                "literal": int(row["literal"]),
                "specificity": int(row["specificity"]),
                "consistent": int(row["consistent"]),
                "anchor_regime": row["anchor_regime"],
                "anchors_planned": row.get("anchors", ""),
            }
            append_jsonl(OUT_SAMPLES, out)
            accepted += 1

            time.sleep(0.2)

        job_logs.append({
            "job_id": job_id,
            "provider": provider,
            "model": model,
            "accepted": accepted,
            "rejected": rejected,
            "attempts": attempts,
        })

        n_jobs_done += 1

    os.makedirs(os.path.dirname(JOB_LOG), exist_ok=True)
    log_df = pd.DataFrame(job_logs)
    log_df.to_csv(JOB_LOG, index=False)

    print(f"Done. Jobs completed this run: {n_jobs_done}")


if __name__ == "__main__":
    run(plan_csv="notebooks/generation_plan.csv", max_jobs=None)  # full run
