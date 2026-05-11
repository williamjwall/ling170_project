"""
POS + dependency disambiguation for ambiguous filler lemmas (like, well, so).

Streamlit reads **precomputed** columns from the corpus CSV (see ``scripts/precompute_hybrid_columns.py``).
spaCy runs **offline** in that script only—not in the app.

this is for will, notes for later debugginggggggggg
- Tweak rules only in hybrid_classify_token; keep AMBIGUOUS_LEMMAS + regexes in sync if you add lemmas.
- analyze_transcript: hl += max(0, rl - n_like) preserves regex when tokenizer misses a span.
- Occurrence char offsets use token.idx on the same string passed to nlp() (no HTML escape until render).
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

AMBIGUOUS_LEMMAS = frozenset({"like", "well", "so"})

_RE_LIKE = re.compile(r"\blike\b", re.I)
_RE_WELL = re.compile(r"\bwell\b", re.I)
_RE_SO = re.compile(r"\bso\b", re.I)


@dataclass
class AmbiguousOccurrence:
    start: int
    end: int
    surface: str
    lemma: str
    pos: str
    dep: str
    head_lemma: str
    hybrid_is_filler: bool


# Columns written by scripts/precompute_hybrid_columns.py and read by Streamlit (no runtime spaCy).
HYBRID_NUMERIC_COLS: Tuple[str, ...] = (
    "regex_like",
    "regex_well",
    "regex_so",
    "hybrid_like",
    "hybrid_well",
    "hybrid_so",
    "parsed_like",
    "parsed_well",
    "parsed_so",
    "removed_like",
    "removed_well",
    "removed_so",
    "regex_ambiguous_total",
    "hybrid_ambiguous_total",
    "removed_ambiguous_total",
)
HYBRID_OCCURRENCES_JSON_COL = "hybrid_occurrences_json"


def occurrences_to_json(occurrences: List[AmbiguousOccurrence]) -> str:
    """One row of CSV: JSON list of token spans for highlights / Token detail."""
    payload = [
        {
            "start": o.start,
            "end": o.end,
            "surface": o.surface,
            "lemma": o.lemma,
            "pos": o.pos,
            "dep": o.dep,
            "head_lemma": o.head_lemma,
            "hybrid_is_filler": o.hybrid_is_filler,
        }
        for o in occurrences
    ]
    return json.dumps(payload, ensure_ascii=False)


def occurrences_from_json(raw: str) -> List[AmbiguousOccurrence]:
    if not raw or str(raw).strip() in ("", "nan", "[]"):
        return []
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    out: List[AmbiguousOccurrence] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            out.append(
                AmbiguousOccurrence(
                    start=int(d["start"]),
                    end=int(d["end"]),
                    surface=str(d.get("surface", "")),
                    lemma=str(d.get("lemma", "")),
                    pos=str(d.get("pos", "")),
                    dep=str(d.get("dep", "")),
                    head_lemma=str(d.get("head_lemma", "")),
                    hybrid_is_filler=bool(d.get("hybrid_is_filler", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def attach_hybrid_occurrences_column(df: Any) -> Any:
    """Add ``_hybrid_occurrences`` list column from ``hybrid_occurrences_json`` if present."""
    import pandas as pd

    out = df.copy()
    if HYBRID_OCCURRENCES_JSON_COL not in out.columns:
        out["_hybrid_occurrences"] = [[] for _ in range(len(out))]
        return out
    col = out[HYBRID_OCCURRENCES_JSON_COL].astype(str).replace({"nan": ""})
    out["_hybrid_occurrences"] = [occurrences_from_json(s) for s in col]
    return out


def hybrid_numeric_precomputed(df: Any) -> bool:
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return False
    return "hybrid_like" in df.columns and "regex_like" in df.columns


def _regex_counts(text: str) -> Tuple[int, int, int]:
    t = str(text).lower()
    return len(_RE_LIKE.findall(t)), len(_RE_WELL.findall(t)), len(_RE_SO.findall(t))


def _surface_is_target(token: Any) -> bool:
    w = token.text.lower().strip(".,!?;:\"'")
    return w in AMBIGUOUS_LEMMAS


def hybrid_classify_token(token: Any) -> bool:
    """
    Return True if this token should count toward hybrid 'filler' for like/well/so.
    False = grammatical / meaningful (excluded from hybrid count).

    this is for will, notes for later debugginggggggggg
    - Uses spaCy token.pos_, token.dep_, token.head; model version changes tag distributions.
    - Default branch returns True (lean filler) when tags don't match a "grammatical" branch.
    """
    lem = token.lemma_.lower()
    pos, dep = token.pos_, token.dep_
    head = token.head

    if lem == "well":
        if pos == "INTJ" or dep == "intj":
            return True
        if pos == "ADJ":
            return False
        if pos == "ADV":
            return False
        if pos == "NOUN":
            return False
        return True

    if lem == "so":
        if pos == "INTJ" or dep == "intj":
            return True
        if pos == "CCONJ":
            return False
        if pos == "SCONJ":
            return False
        if pos == "ADV" and dep == "advmod":
            if head.pos_ in ("ADJ", "ADV", "NOUN", "DET", "NUM"):
                return False
            try:
                nbor = token.nbor(1)
            except (IndexError, KeyError):
                nbor = None
            if head.pos_ in ("VERB", "AUX") and nbor is not None and nbor.text.strip() == ",":
                return True
            if head.pos_ in ("VERB", "AUX") and token.i == token.sent[0].i:
                return True
        return True

    if lem == "like":
        if pos == "ADP" and dep == "prep":
            return False
        if pos == "VERB":
            return False
        if pos in ("NOUN", "PROPN", "ADJ", "NUM"):
            return False
        return True

    return True


def analyze_transcript(text: str, nlp: Any) -> Dict[str, Any]:
    """Per transcript: regex vs hybrid counts for like/well/so + occurrence metadata.

    this is for will, notes for later debugginggggggggg
    - Dict keys must stay aligned with pages/Research_Studies_EDA.HYBRID_NUMERIC_COLS for merge.
    """
    empty = {
        "regex_like": 0,
        "regex_well": 0,
        "regex_so": 0,
        "hybrid_like": 0,
        "hybrid_well": 0,
        "hybrid_so": 0,
        "parsed_like": 0,
        "parsed_well": 0,
        "parsed_so": 0,
        "removed_like": 0,
        "removed_well": 0,
        "removed_so": 0,
        "regex_ambiguous_total": 0,
        "hybrid_ambiguous_total": 0,
        "removed_ambiguous_total": 0,
        "occurrences": [],
    }
    if not text or str(text).startswith("[ERROR"):
        return empty

    raw = str(text)
    rl, rw, rs = _regex_counts(raw)

    doc = nlp(raw)
    occ: List[AmbiguousOccurrence] = []
    hl, hw, hs = 0, 0, 0

    for token in doc:
        if token.is_space or not token.text.strip():
            continue
        lem = token.lemma_.lower()
        if lem not in AMBIGUOUS_LEMMAS or not _surface_is_target(token):
            continue
        is_filler = hybrid_classify_token(token)
        head_lm = token.head.lemma_.lower() if token.head is not None else ""
        occ.append(
            AmbiguousOccurrence(
                start=int(token.idx),
                end=int(token.idx) + len(token.text),
                surface=token.text,
                lemma=lem,
                pos=token.pos_,
                dep=token.dep_,
                head_lemma=head_lm,
                hybrid_is_filler=is_filler,
            )
        )
        if is_filler:
            if lem == "like":
                hl += 1
            elif lem == "well":
                hw += 1
            else:
                hs += 1

    n_like = sum(1 for o in occ if o.lemma == "like")
    n_well = sum(1 for o in occ if o.lemma == "well")
    n_so = sum(1 for o in occ if o.lemma == "so")
    # Regex matches spaCy did not align to a token: keep regex (filler) behaviour.
    hl += max(0, rl - n_like)
    hw += max(0, rw - n_well)
    hs += max(0, rs - n_so)

    rtot = rl + rw + rs
    htot = hl + hw + hs
    out = {
        "regex_like": rl,
        "regex_well": rw,
        "regex_so": rs,
        "hybrid_like": hl,
        "hybrid_well": hw,
        "hybrid_so": hs,
        "parsed_like": n_like,
        "parsed_well": n_well,
        "parsed_so": n_so,
        "removed_like": max(0, rl - hl),
        "removed_well": max(0, rw - hw),
        "removed_so": max(0, rs - hs),
        "regex_ambiguous_total": rtot,
        "hybrid_ambiguous_total": htot,
        "removed_ambiguous_total": max(0, rtot - htot),
        "occurrences": occ,
    }
    return out


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    s = sorted(spans)
    merged = [s[0]]
    for a, b in s[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def html_apply_marks(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """
    spans: (start, end, html_open_tag_without_close) — close tag is always </mark> or </span>.
    Non-overlapping spans required.
    """
    if not spans:
        return html.escape(text)
    spans = sorted(spans, key=lambda x: x[0])
    parts: List[str] = []
    cur = 0
    for start, end, open_tag in spans:
        if start < cur:
            continue
        if start > cur:
            parts.append(html.escape(text[cur:start]))
        inner = html.escape(text[start:end])
        if open_tag.startswith("<mark"):
            parts.append(f"{open_tag}{inner}</mark>")
        else:
            parts.append(f"{open_tag}{inner}</span>")
        cur = end
    if cur < len(text):
        parts.append(html.escape(text[cur:]))
    return "".join(parts)


def before_after_html(text: str, occurrences: List[AmbiguousOccurrence]) -> Tuple[str, str]:
    """
    Before: highlight every ambiguous lemma token (regex-aligned via parser).
    After: hybrid filler = green mark; excluded (grammatical) = red span.
    """
    raw = str(text)
    if not occurrences:
        esc = html.escape(raw)
        return esc, esc

    before_spans: List[Tuple[int, int, str]] = []
    for o in occurrences:
        before_spans.append((o.start, o.end, '<mark class="hybrid-before">'))
    before_html = html_apply_marks(raw, before_spans)

    parts_after: List[str] = []
    cur = 0
    for o in sorted(occurrences, key=lambda x: x.start):
        start, end = o.start, o.end
        if start > cur:
            parts_after.append(html.escape(raw[cur:start]))
        inner = html.escape(raw[start:end])
        if o.hybrid_is_filler:
            parts_after.append(f'<mark class="hybrid-after-yes">{inner}</mark>')
        else:
            parts_after.append(f'<span class="hybrid-after-no">{inner}</span>')
        cur = end
    if cur < len(raw):
        parts_after.append(html.escape(raw[cur:]))
    after_html = "".join(parts_after)

    return before_html, after_html


def hybrid_styles_block() -> str:
    return """
<style>
.hybrid-before { background-color: #ffe066; padding: 0 1px; border-radius: 2px; }
.hybrid-after-yes { background-color: #95d5b2; padding: 0 1px; border-radius: 2px; }
.hybrid-after-no {
  background-color: #ffb3b3;
  color: #3d0a0a;
  text-decoration: line-through;
  text-decoration-thickness: 1px;
  text-decoration-color: #7a1e1e;
  padding: 0 1px;
  border-radius: 2px;
}
</style>
"""
