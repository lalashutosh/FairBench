"""Hard or soft: how binding the wording of a rule is.

Decided by a fixed word list in code, never by a model, so the same sentence always gets
the same answer and the matched words can be shown to a reviewer.

    hard   the sentence commits ("must", "shall", "may not", "at least", "no more than")
           and contains no soft wording
    soft   the sentence hedges ("seeks to", "normally", "intends to", "expects to",
           "where practicable", a bare "may"), or contains no commitment wording at all

A sentence with BOTH kinds of wording is soft: "under normal circumstances, at least 80%"
does not automatically become a hard constraint. ``regulatory_basis`` records when the
sentence has the shape of a statutory fund policy (the 80% names rule, the diversified-fund
test, the 25% industry concentration limit). That is information for the reviewer who may
promote the rule; it does not change the classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HARD_TERMS: tuple[tuple[str, str], ...] = (
    ("must", r"\bmust\b"),
    ("shall", r"\bshall\b"),
    ("may not", r"\bmay\s+not\b"),
    ("will not", r"\bwill\s+not\b"),
    ("does not invest", r"\bdo(?:es)?\s+not\s+invest\b"),
    ("cannot", r"\bcan\s*not\b"),
    ("prohibited", r"\bprohibit(?:s|ed)?\b"),
    ("excluded", r"\bexclud(?:e|es|ed|ing)\b"),
    ("at least", r"\bat\s+least\b"),
    ("no more than", r"\b(?:no|not)\s+more\s+than\b"),
    ("not exceed", r"\bnot\s+(?:to\s+)?exceed\b"),
    ("no less than", r"\b(?:no|not)\s+less\s+than\b"),
    ("only", r"\bonly\b"),
)
SOFT_TERMS: tuple[tuple[str, str], ...] = (
    ("seeks to", r"\bseeks?\s+to\b"),
    ("aims", r"\baims?\b"),
    ("normally", r"\bnormally\b"),
    ("under normal circumstances", r"\bunder\s+normal\s+(?:circumstances|market\s+conditions|conditions)\b"),
    ("generally", r"\bgenerally\b"),
    ("typically", r"\btypically\b"),
    ("intends to", r"\bintends?\s+to\b"),
    ("expects to", r"\bexpect(?:s|ed)?\s+to\b"),
    ("where practicable", r"\b(?:where|when|if|to\s+the\s+extent)\s+practicable\b"),
    ("best efforts", r"\bbest\s+efforts\b"),
    ("considers", r"\bconsiders?\b"),
    ("approximately", r"\bapproximately\b"),
    ("may", r"\bmay\b(?!\s+not\b)"),
    ("target", r"\btargets?\b"),
    ("primarily", r"\bprimarily\b"),
)
_REGULATORY: tuple[tuple[str, str], ...] = (
    ("rule_35d-1_names_rule", r"\bat\s+least\s+80\s*%"),
    ("1940_act_diversification", r"\b75\s*%\s+of\s+(?:its|the\s+fund'?s)\s+total\s+assets\b"),
    ("1940_act_concentration", r"\b25\s*%.*\b(?:any\s+one|a\s+single|one)\s+industry\b|\bconcentrat"),
)


@dataclass(frozen=True)
class Modality:
    """constraint_type: "hard" or "soft". hard_terms / soft_terms: the words that were found.
    unmarked: no commitment or hedging wording at all (classified soft for that reason)."""
    constraint_type: str
    hard_terms: tuple[str, ...]
    soft_terms: tuple[str, ...]
    regulatory_basis: str | None
    unmarked: bool

    @property
    def terms(self) -> list[str]:
        return list(self.hard_terms + self.soft_terms)

    @property
    def mixed(self) -> bool:
        return bool(self.hard_terms) and bool(self.soft_terms)


def classify(evidence: str) -> Modality:
    """Classify the wording of one quoted rule."""
    text = " ".join(evidence.split()).casefold().replace("’", "'")
    hard = tuple(name for name, pat in HARD_TERMS if re.search(pat, text))
    soft = tuple(name for name, pat in SOFT_TERMS if re.search(pat, text))
    basis = next((name for name, pat in _REGULATORY if re.search(pat, text)), None)
    kind = "hard" if hard and not soft else "soft"
    return Modality(kind, hard, soft, basis, unmarked=not hard and not soft)
