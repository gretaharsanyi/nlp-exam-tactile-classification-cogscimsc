import os, json, hashlib
from typing import Dict, Any, Set

from src.config import OUT_SAMPLES, OUT_REJECTS, JOB_LOG, RESET_OUTPUTS_EACH_RUN

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

def load_done_job_ids(samples_path: str) -> Set[str]:
    if RESET_OUTPUTS_EACH_RUN:
        return set()

    if not os.path.exists(samples_path):
        return set()

    done: Set[str] = set()
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                job_id = o.get("job_id")
                if job_id:
                    done.add(job_id)
            except Exception:
                continue
    return done
