import re
from typing import List, Tuple, Set

import pandas as pd

from src.config import MIN_CHARS, MAX_CHARS, METAPHOR_MARKERS

# Import seed set / helpers from your existing file
try:
    from src.anchors import SEEDS  # must exist
except Exception:
    from anchors import SEEDS  # type: ignore

try:
    from src.anchors import expand_anchor_family  # type: ignore
except Exception:
    expand_anchor_family = None  # type: ignore

try:
    from src.anchors import get_family_variants  # type: ignore
except Exception:
    get_family_variants = None  # type: ignore


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_anchors(anchors_field: str) -> List[str]:
    if anchors_field is None:
        return []
    s = str(anchors_field).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [a.strip() for a in s.split(";") if a.strip()]

def has_metaphor_marker(text: str) -> bool:
    t = f" {text.lower()} "
    return any(m in t for m in METAPHOR_MARKERS)

def _seed_terms_upper() -> Set[str]:
    out: Set[str] = set()
    for _, terms in SEEDS.items():
        for t in terms:
            out.add(str(t).strip().upper())
    return out

_SEED_UPPER = _seed_terms_upper()

def _simple_inflections_for_single_word(word: str) -> Set[str]:
    w = word.lower().strip()
    if not w:
        return set()
    variants = {w}

    variants.add(w + "s")
    variants.add(w + "ed")
    variants.add(w + "ing")

    if w.endswith("e"):
        variants.add(w[:-1] + "ing")
        variants.add(w + "d")
    if w.endswith("y") and len(w) > 2:
        variants.add(w[:-1] + "ies")

    return {v for v in variants if v}

def strict_anchor_present(text: str, anchor: str) -> bool:
    t = text.lower()
    a = anchor.strip()
    if not a:
        return True

    a_upper = a.upper()
    a_lower = a.lower()

    # Multi-word anchor: match with simple verb inflection on first token
    if " " in a_lower:
        tokens = a_lower.split()
        first = re.escape(tokens[0])
        rest = r"\s+".join(re.escape(tok) for tok in tokens[1:])
        pattern = rf"\b{first}(?:s|es|ing|ed)?\b\s+{rest}\b"
        return re.search(pattern, t) is not None

    # Single-word anchor: try helpers
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

    # No helper: permissive SEEDS-based fallback
    if not variants:
        if a_upper in _SEED_UPPER:
            variants |= _simple_inflections_for_single_word(a_lower)

        for term_upper in _SEED_UPPER:
            term = term_upper.lower()
            if term == a_lower:
                variants |= _simple_inflections_for_single_word(term)
                continue
            if a_lower in term or term in a_lower:
                variants |= _simple_inflections_for_single_word(term)

    variants.add(a_lower)

    for v in sorted(variants, key=len, reverse=True):
        if re.search(rf"\b{re.escape(v)}\b", t):
            return True

    return False

def paraphrase_anchor_leak(text: str, anchor: str) -> bool:
    if not anchor:
        return False

    txt = text
    a = anchor.strip()

    # exact ALL-CAPS only
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
