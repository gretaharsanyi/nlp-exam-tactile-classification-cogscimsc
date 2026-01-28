import json
import re
from typing import Dict, Any, Optional

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
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None
