# src/anchors.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Your seed set (kept as-is, but completed the pressure list bracket/brace)
SEEDS: Dict[str, List[str]] = {
    "nociception": [
        "PAIN","PAINFUL","PAINFULNESS","BURN","BURNING","SCALDING","BLISTERING",
        "STING","STINGING","PRICKING","PRICK","PRICKED","PINPRICK","ITCH","ITCHING",
        "ITCHINESS","BRUISE","JAB","JABBED","PUNCTURE","PIERCED","INJURE","SHARP",
        "SHARPER","SPIKY","SPIKINESS","THORNY","THORNINESS","JAGGED","JAGGEDNESS",
        "ELECTROSHOCK","ELECTRIC SHOCK","ELECTRICAL SHOCK","JOLT","ACHE","ACHING",
        "SORE","SORENESS","HURT","HURTING"
    ],
    "temperature": [
        "HOT","HOTNESS","HEAT","WARM","WARMTH","WARMNESS","WARMER","LUKEWARM","TEPID",
        "COOL","COOLNESS","COOLER","COLD","COLDNESS","COLDER","COLDISH","FREEZING",
        "FROZEN","FREEZE","FROST","FROSTINESS","ICY","ICINESS","ICE COLD","OVERHEATED",
        "SUPERHEATED","HEATING","COOLING","WARMING","UNHEATED","ROOM TEMPERATURE",
        "CHILLY","CHILLINESS","CHILLED","CHILLING","FRIGID","FROSTY"
    ],
    "vibration": [
        "VIBRATION","VIBRATE","VIBRATING","PULSE","PULSATE",
        "THROBBING","TREMBLING","OSCILLATING"
    ],
    "pressure": [
        "PRESSURE","PRESS","PRESSING","PRESS DOWN","SQUEEZE","SQUEEZING",
        "COMPRESS","COMPRESSIBLE","TIGHT","TIGHTNESS","FORCE",
        "COMPRESSION","TENSION"
    ],
}


def _basic_inflections(token: str) -> Set[str]:
    """
    Very lightweight English-ish inflection expansion for single words.
    (Enough to catch ACHING/ACHED/ACHES, JOLT/JOLTS/JOLTED/JOLTING, etc.)
    """
    t = token.lower()

    # If it's already a multiword phrase, don't inflect here.
    if " " in t:
        return {t}

    forms = {t}

    # Already looks like an inflected form
    # (Still keep it as a member; we don't try to reverse-stem.)
    # Add a few likely alternates.
    if t.endswith("ing"):
        forms.add(t[:-3])  # aching -> ach
    if t.endswith("ed"):
        forms.add(t[:-2])  # jolted -> jolt
    if t.endswith("s"):
        forms.add(t[:-1])  # jolts -> jolt

    # Regular-ish expansions
    if t.endswith("y") and len(t) > 2:
        forms.update({t[:-1] + "ies", t[:-1] + "ied"})
    else:
        forms.add(t + "s")
        forms.add(t + "ed")
        forms.add(t + "ing")

    # Some common doubling pattern (jog->jogging etc) is ignored on purpose.
    return forms


def _phrase_variants(phrase: str) -> Set[str]:
    """
    Expand a phrase like 'PRESS DOWN' into common natural variants:
    'press down', 'pressing down', 'pressed down', 'presses down'
    Also allow optional hyphen 'press-down' in matching via regex later.
    """
    p = phrase.lower().strip()
    if " " not in p:
        return _basic_inflections(p)

    parts = p.split()
    if len(parts) != 2:
        return {p}

    a, b = parts
    variants = {
        f"{a} {b}",
        f"{a}s {b}",
        f"{a}ed {b}",
        f"{a}ing {b}",
    }
    return variants


def build_anchor_families(seeds: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Returns mapping from canonical anchor (UPPER as in seeds) -> set of lowercased variants.
    """
    families: Dict[str, Set[str]] = {}
    for _mod, anchors in seeds.items():
        for anc in anchors:
            canon = anc.strip().upper()
            if not canon:
                continue
            if " " in canon:
                fam = _phrase_variants(canon)
            else:
                fam = _basic_inflections(canon)

            # Always include the lowercase exact form as well.
            fam.add(canon.lower())
            families[canon] = fam
    return families


@dataclass(frozen=True)
class AnchorMatchResult:
    ok: bool
    missing: Tuple[str, ...] = ()
    leaked: Tuple[str, ...] = ()


def _compile_family_regex(family: Iterable[str]) -> re.Pattern:
    """
    Compile a regex that matches any family member as a whole word / phrase.
    For phrases we allow spaces or hyphens between words.
    """
    alts: List[str] = []
    for f in set(family):
        f = f.strip().lower()
        if not f:
            continue
        if " " in f:
            w1, w2 = f.split()
            alts.append(rf"{re.escape(w1)}(?:\s+|-){re.escape(w2)}")
        else:
            alts.append(re.escape(f))
    # word boundaries around the whole alternative
    return re.compile(rf"\b(?:{'|'.join(sorted(alts, key=len, reverse=True))})\b", re.IGNORECASE)


def parse_anchors_planned(anchors_planned: Optional[str]) -> List[str]:
    if not anchors_planned:
        return []
    # Your logs show "JOLT; OSCILLATING; COMPRESS"
    parts = [p.strip() for p in anchors_planned.split(";")]
    return [p for p in parts if p]


def check_anchors(
    text: str,
    anchors_planned: Optional[str],
    families: Dict[str, Set[str]],
    *,
    require_all: bool = True,
    leak_only_if_allcaps: bool = True,
) -> AnchorMatchResult:
    """
    - Missing: anchor family not found in text.
    - Leak: only if the literal anchor appears in ALL CAPS (or exact-cased as provided),
      which is usually the true "control token leaked" case.
    """
    planned = parse_anchors_planned(anchors_planned)
    if not planned:
        return AnchorMatchResult(ok=True)

    t = text.strip()
    t_low = t.lower()

    missing: List[str] = []
    leaked: List[str] = []

    for anc in planned:
        canon = anc.strip().upper()
        fam = families.get(canon)

        # If unknown anchor, fall back to literal matching
        if not fam:
            fam = {canon.lower()}

        rx = _compile_family_regex(fam)
        found = bool(rx.search(t_low))

        if require_all and not found:
            missing.append(canon)

        if leak_only_if_allcaps:
            # Only flag "leak" if the anchor appears as ALL CAPS literally
            # (this catches "PRESS DOWN" pasted in caps, but NOT "jolts")
            if canon in t:
                leaked.append(canon)
        else:
            # If you ever want strict "anchor must not appear at all", set False above and use this.
            if rx.search(t_low):
                leaked.append(canon)

    ok = (len(missing) == 0) and (len(leaked) == 0)
    return AnchorMatchResult(ok=ok, missing=tuple(missing), leaked=tuple(leaked))
