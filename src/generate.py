# src/generate.py
import os, json, time, hashlib, random, re
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from openai import OpenAI
import anthropic


# ---------------------------
# Config
# ---------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# note: these I can override in .env if needed
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

MIN_CHARS = 40
MAX_CHARS = 160  # shortened

METAPHOR_MARKERS = [" like ", " as if ", " as though "]

OUT_SAMPLES = "data/generated/raw/samples.jsonl"
OUT_REJECTS = "data/generated/raw/rejects.jsonl"
JOB_LOG = "logs/job_log.csv"

# variation seeds (NO body-part/timecourse suggestions)
STYLE_SEEDS = [
    "plain", "slightly poetic", "matter-of-fact", "tight and direct",
    "sensory and vivid", "more emotional"
]


# ---------------------------
# Validation helpers
# ---------------------------

def anchor_violation(text: str, anchor: str) -> bool:
    t = text.lower()
    a = anchor.lower()
    variants = {a, a.rstrip("s"), a + "s", a + "ing"}
    return any(v and v in t for v in variants)

def has_metaphor_marker(text: str) -> bool:
    t = f" {text.lower()} "
    return any(m in t for m in METAPHOR_MARKERS)

def parse_anchors(anchors_field: str) -> List[str]:
    if not anchors_field or str(anchors_field).strip() == "":
        return []
    return [a.strip() for a in str(anchors_field).split(";") if a.strip()]

def _anchor_regex_for_phrase(anchor: str) -> re.Pattern:
    """
    Build a regex that matches reasonable grammatical variants for anchors,
    including multi-word anchors like 'PRESS DOWN'.

    For 'press down' we accept:
    press down / presses down / pressed down / pressing down
    (case-insensitive, allowing whitespace between words)
    """
    a = anchor.strip().lower()
    a = re.sub(r"\s+", " ", a)

    # Special-case verb+particle anchor that we know is problematic
    if a == "press down":
        return re.compile(r"\bpress(?:es|ed|ing)?\s+down\b", flags=re.IGNORECASE)

    # Fallback: exact phrase match (case-insensitive, flexible whitespace)
    escaped = re.escape(a).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)

def strict_anchor_present(text: str, anchor: str) -> bool:
    return _anchor_regex_for_phrase(anchor).search(text) is not None

def passes_validation(text: str, row: pd.Series) -> (bool, str):
    txt = (text or "").strip()
    if len(txt) < MIN_CHARS:
        return False, "too_short"
    if len(txt) > MAX_CHARS:
        return False, "too_long"

    literal = int(row["literal"])
    if literal == 0 and not has_metaphor_marker(txt):
        return False, "missing_metaphor_marker"
    if literal == 1 and has_metaphor_marker(txt):
        return False, "metaphor_marker_in_literal"

    anchor_regime = str(row["anchor_regime"]).lower()
    anchors = parse_anchors(row.get("anchors", ""))

    if anchor_regime == "strict":
        missing = [a for a in anchors if not strict_anchor_present(txt, a)]
        if missing:
            return False, f"missing_anchors:{','.join(missing[:3])}"
    elif anchor_regime == "paraphrase":
        violating = [a for a in anchors if anchor_violation(txt, a)]
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
    anchor_regime = row["anchor_regime"]
    anchors = row.get("anchors", "")

    metaphor_rule = (
        "Use an explicit comparison marker: 'like', 'as if', or 'as though'."
        if literal == 0 else
        "Do NOT use 'like', 'as if', or 'as though'. No comparisons."
    )
    specificity_rule = (
        "Be specific: include at least ONE body location AND ONE intensity/temporal cue."
        if specificity == 1 else
        "Stay vague: avoid concrete body locations and avoid detailed intensity/temporal profiling."
    )
    consistency_rule = (
        "Be internally consistent: all cues should align with the target modalities."
        if consistent == 1 else
        "Be internally inconsistent: include at least one contradictory cue while still mentioning the target modalities."
    )
    if str(anchor_regime).lower() == "strict":
        anchor_rule = f"Anchors MUST appear verbatim (case-insensitive): {anchors}"
    elif str(anchor_regime).lower() == "paraphrase":
        anchor_rule = f"Anchors MUST NOT appear (case-insensitive, including simple variants): {anchors}"
    else:
        anchor_rule = "No anchor constraints."

    system = (
        "You write short first-person tactile sensation descriptions for a synthetic dataset. "
        "Output must be JSON only."
    )

    user = f"""
Generate ONE short tactile description (1 sentence, max 160 characters) that expresses these somatosensory modalities: {labels}.
Constraints:
- {metaphor_rule}
- {specificity_rule}
- {consistency_rule}
- {anchor_rule}
Style:
- first person, bodily sensation focus
- The description should read as a naturally written sentence. Any required anchor word must be integrated smoothly into the grammar of the sentence and should not appear forced or highlighted.
- Frame the description as a passive bodily sensation. Avoid first-person motor actions or intentional touch; the sensation should be experienced, not enacted.
- If an anchor is verb-like, embed it in a passive construction (e.g., "seems to press down") rather than an intentional action.
- avoid reusing the same phrasing across outputs; keep wording varied
- no medical advice, no diagnosis, no mention of datasets or labels

{variation_note}

Return JSON only with this schema:
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
        max_tokens=220,  # shorter outputs
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON even if the model wraps it in ```json fences
    or includes minor formatting noise.
    """
    if not s:
        return None
    t = s.strip()

    # Strip markdown code fences if present: ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t.strip())

    # Try direct load
    try:
        return json.loads(t)
    except Exception:
        pass

    # Extract first JSON object substring
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        candidate = m.group(0).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # Common typo: extra trailing "}"
            if candidate.endswith("}}"):
                try:
                    return json.loads(candidate[:-1])
                except Exception:
                    pass

    return None


# ---------------------------
# Main loop
# ---------------------------

def file_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_job_ids(samples_path: str) -> set:
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

def run(plan_csv: str, max_jobs: Optional[int] = None):
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

        while accepted < n_samples and attempts < n_samples * 8:
            attempts += 1

            # variation per attempt (no semantic slot-filling)
            style = random.choice(STYLE_SEEDS)
            variation_note = (
                f"Variation seed: write in a {style} style. "
                f"Do not reuse phrasing from previous outputs for this same job."
            )

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

            text = str(js["items"][0]).strip()

            ok, reason = passes_validation(text, row)
            if not ok:
                append_jsonl(OUT_REJECTS, {
                    "job_id": job_id, "provider": provider, "model": model,
                    "reason": reason, "text": text, "prompt_id": prompt_id
                })
                rejected += 1
                continue

            out = {
                "job_id": job_id,
                "provider": provider,
                "model": model,
                "prompt_id": prompt_id,
                "text": text,
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
    if os.path.exists(JOB_LOG):
        old = pd.read_csv(JOB_LOG)
        log_df = pd.concat([old, log_df], ignore_index=True)
    log_df.to_csv(JOB_LOG, index=False)

    print(f"Done. Jobs completed this run: {n_jobs_done}")


if __name__ == "__main__":
    run(plan_csv="notebooks/generation_plan.csv", max_jobs=3)  # tiny live test
