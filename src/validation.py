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


def _norm(s: str) -> str:
    """
    Normalize for matching:
    - lowercase
    - hyphen -> space
    - collapse whitespace
    """
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _simple_inflections_for_single_word(word: str) -> Set[str]:
    """
    Lightweight morphology. We keep this intentionally simple (not a full lemmatizer).
    """
    w = _norm(word)
    if not w:
        return set()
    variants = {w}

    variants.add(w + "s")
    variants.add(w + "es")
    variants.add(w + "ed")
    variants.add(w + "ing")

    if w.endswith("e"):
        variants.add(w[:-1] + "ing")
        variants.add(w + "d")
    if w.endswith("y") and len(w) > 2:
        variants.add(w[:-1] + "ies")

    return {v for v in variants if v}


# Small, high-impact family mapping for anchors that frequently surface in different forms.
# This is meant to *reduce rejects* from trivial morphological variation.
_MORPH_FAMILY = {
    # vibration family
    "vibration": {"vibrate", "vibrating", "vibrations"},
    "vibrate": {"vibration", "vibrating", "vibrations"},
    "vibrating": {"vibration", "vibrate", "vibrations"},
    # pressure / press family
    "press": {"pressure", "pressing", "pressed"},
    "pressure": {"press", "pressing", "pressed"},
    "pressing": {"press", "pressure", "pressed"},
    # compress family
    "compress": {"compresses", "compressed", "compressing", "compression", "compressible"},
    "compression": {"compress", "compressed", "compressing", "compressible"},
    "compressible": {"compress", "compressed", "compressing", "compression"},
    # pulse family
    "pulse": {"pulsate", "pulsing", "pulses"},
    "pulsate": {"pulse", "pulsing", "pulses"},
    "pulsing": {"pulse", "pulsate", "pulses"},
    # temperature-ish families that show up a lot
    "freeze": {"freezing", "frozen", "frost", "frosty"},
    "frozen": {"freeze", "freezing", "frost", "frosty"},
    "frost": {"frosty", "frostiness", "frozen", "freezing"},
    "frosty": {"frost", "frostiness", "frozen", "freezing"},
    "icy": {"iciness"},
    "iciness": {"icy"},
    "warm": {"warmth", "warmness", "warming"},
    "warmth": {"warm", "warmness", "warming"},
    "heat": {"heating", "hot", "hotness", "warming"},
    "cool": {"cooling", "coolness"},
    "cold": {"coldness", "colder", "coldish", "chilly", "chilled"},
    # nociception-ish families (common)
    "sting": {"stinging"},
    "stinging": {"sting"},
    "ache": {"aching"},
    "aching": {"ache"},
    "sore": {"soreness"},
    "soreness": {"sore"},
    "hurt": {"hurting"},
    "hurting": {"hurt"},
    "tight": {"tightness"},
    "tightness": {"tight"},
    "tense": {"tension"},
    "tension": {"tense"},
    "prick": {"pricking", "pricked", "pinprick"},
    "pricking": {"prick", "pricked", "pinprick"},
    "pricked": {"prick", "pricking", "pinprick"},
}


# Special multiword anchors that need phrase-level flexibility
_SPECIAL_MULTIWORD = {
    "ice cold": {"ice cold", "icecold", "ice-cold"},
    "room temperature": {
        "room temperature",
        "room temp",
        "room-temp",
        "ambient temperature",
        "at room temperature",
        "ambient",
    },
    "electric shock": {"electric shock", "electrical shock", "electroshock"},
    "press down": {"press down", "pressing down", "pressed down"},
}


def _anchor_variants(anchor: str) -> Set[str]:
    """
    Build a permissive (but still anchored) set of acceptable surface forms for a planned anchor.
    """
    a = _norm(anchor)
    if not a:
        return set()

    variants: Set[str] = {a}

    # Phrase-level flexibility for a few anchors that often get paraphrased in a "still essentially same" way
    if a in _SPECIAL_MULTIWORD:
        variants |= {_norm(x) for x in _SPECIAL_MULTIWORD[a]}

    # If multiword, accept hyphen/space and "collapsed" forms
    if " " in a:
        variants.add(a.replace(" ", ""))  # e.g., "ice cold" -> "icecold"
        return variants

    # 1) Try your existing anchor-family helpers if present
    helper_variants: Set[str] = set()
    if expand_anchor_family is not None:
        try:
            helper_variants |= {_norm(v) for v in expand_anchor_family(anchor)}  # type: ignore
        except Exception:
            pass

    if get_family_variants is not None:
        try:
            helper_variants |= {_norm(v) for v in get_family_variants(anchor, SEEDS)}  # type: ignore
        except Exception:
            pass

    variants |= helper_variants

    # 2) Add lightweight inflections
    variants |= _simple_inflections_for_single_word(a)

    # 3) Add small family mapping (high value for your rejects)
    if a in _MORPH_FAMILY:
        variants |= {_norm(v) for v in _MORPH_FAMILY[a]}
        for v in _MORPH_FAMILY[a]:
            variants |= _simple_inflections_for_single_word(v)

    # 4) SEEDS-based fallback: if the anchor is in your seed universe, add inflections for near matches
    a_upper = a.upper()
    if a_upper in _SEED_UPPER:
        variants |= _simple_inflections_for_single_word(a)

    # Also allow inflections of any seed term that is substring-related (kept conservative)
    for term_upper in _SEED_UPPER:
        term = term_upper.lower()
        if term == a:
            continue
        if a in term or term in a:
            variants |= _simple_inflections_for_single_word(term)

    return {v for v in variants if v}


def _contains_any_variant(text: str, anchor: str) -> bool:
    """
    Check if text contains the anchor in any acceptable variant.
    Uses word boundaries; for multiword variants, allows flexible whitespace.
    """
    t = _norm(text)
    variants = _anchor_variants(anchor)
    if not variants:
        return True

    # Prefer longer matches first (slightly reduces accidental boundary matches)
    for v in sorted(variants, key=len, reverse=True):
        # Turn spaces into \s+ for robust matching
        pattern = r"\b" + re.escape(v).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, t):
            return True
    return False


def strict_anchor_present(text: str, anchor: str) -> bool:
    """
    Strict regime: require the planned anchor concept to appear,
    allowing common surface variants (case, hyphenation, inflection, a few family mappings).
    """
    a = (anchor or "").strip()
    if not a:
        return True
    return _contains_any_variant(text, a)


def paraphrase_anchor_leak(text: str, anchor: str) -> bool:
    """
    Paraphrase regime: forbid exact ALL-CAPS anchor leakage only.
    (We keep this strict to preserve the intended "no exact anchor string" constraint.)
    """
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
