import time
from typing import Optional

import pandas as pd

from src.config import (
    OUT_SAMPLES, OUT_REJECTS, JOB_LOG,
    OPENAI_MODEL, CLAUDE_MODEL,
)
from src.io_utils import (
    append_jsonl, reset_outputs_if_needed, load_done_job_ids, file_hash
)
from src.parsing import safe_json_load
from src.validation import passes_validation, normalize_ws
from src.prompting import prompt_for_row, choose_variation_note
from src.providers.openai_client import call_openai
from src.providers.claude_client import call_claude


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

            variation_note = choose_variation_note()
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

    import os
    os.makedirs(os.path.dirname(JOB_LOG), exist_ok=True)
    pd.DataFrame(job_logs).to_csv(JOB_LOG, index=False)

    print(f"Done. Jobs completed this run: {n_jobs_done}")


if __name__ == "__main__":
    run(plan_csv="notebooks/generation_plan.csv", max_jobs=None)
