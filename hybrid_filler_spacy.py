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


_COPULA_HEADS = frozenset({"be", "is", "are", "was", "were", "am", "been", "being"})
_INTENSIFIER_SO_HEADS = frozenset({"far", "many", "much", "long", "often", "few", "little"})
_LIKE_QUANTIFIER_NEXT = frozenset({
    "million", "thousand", "hundred", "billion", "dozen", "tons", "lot", "lots",
    "couple", "few", "many", "much", "some", "any",
})
_SO_INTENSIFIER_NEXT = frozenset({
    "upset", "lazy", "busy", "tired", "mean", "cool", "good", "bad", "angry", "happy",
    "sad", "mad", "funny", "weird", "hard", "easy", "fast", "slow", "big", "small",
    "young", "old", "hot", "cold", "loud", "quiet", "scary", "nice", "great", "awful",
    "pretty", "really", "very", "much", "many", "few", "little", "long", "far",
})
_FILLER_LABEL = {"like": "filtered_like", "well": "filtered_well", "so": "filtered_so"}
_RE_AMBIGUOUS_WORD = re.compile(r"\b(like|well|so)\b", re.I)


def _at_discourse_boundary(text: str, start: int) -> bool:
    """True when the token begins a sentence or strong clause boundary."""
    if start <= 0:
        return True
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    return prefix[-1] in ".!?"


def _preceded_by_as(text: str, start: int) -> bool:
    left = text[:start].rstrip().lower()
    return left.endswith(" as") or left.endswith("\tas")


def _next_word_after(text: str, end: int) -> str:
    m = re.match(r"\s+([A-Za-z'+]+)", text[end:])
    return m.group(1).lower() if m else ""


def hybrid_classify_tags(
    lemma: str,
    pos: str,
    dep: str,
    head_lemma: str,
    *,
    text: str = "",
    start: int = 0,
    end: int = 0,
) -> bool:
    """
    Return True if this *like* / *well* / *so* token is a discourse filler.

    False = grammatical / content use (kept as the surface word in simplified exports).
    """
    lem = str(lemma).lower()
    pos = str(pos)
    dep = str(dep)
    head = str(head_lemma).lower()

    if lem == "well":
        if pos == "INTJ" or dep == "intj":
            return True
        if pos in ("ADJ", "NOUN", "PROPN"):
            return False
        if pos == "ADV":
            if _preceded_by_as(text, start):
                return False
            if _at_discourse_boundary(text, start):
                return True
            return False
        if _at_discourse_boundary(text, start):
            return True
        return False

    if lem == "so":
        if pos == "INTJ" or dep == "intj":
            return True
        if pos in ("CCONJ", "SCONJ"):
            return False
        nxt = _next_word_after(text, end)
        if nxt in _INTENSIFIER_SO_HEADS or nxt in _SO_INTENSIFIER_NEXT or head in _SO_INTENSIFIER_NEXT:
            return False
        if pos == "ADV" and dep == "advmod":
            if _at_discourse_boundary(text, start):
                return True
            if end < len(text) and text[end : end + 1] == ",":
                return True
            return True
        if _at_discourse_boundary(text, start):
            return True
        return False

    if lem == "like":
        nxt = _next_word_after(text, end)
        if nxt in _LIKE_QUANTIFIER_NEXT:
            return False
        if pos == "VERB":
            return False
        if pos in ("NOUN", "PROPN", "ADJ", "NUM"):
            return False
        if pos == "ADP" and dep == "prep":
            if head in _COPULA_HEADS:
                return True
            return False
        if pos in ("INTJ", "PART", "SCONJ") or dep in ("intj", "mark", "discourse"):
            return True
        if head in _COPULA_HEADS and dep in ("prep", "mark", "intj"):
            return True
        return True

    return False


def hybrid_classify_token(token: Any) -> bool:
    """
    Return True if this token should count toward hybrid 'filler' for like/well/so.
    False = grammatical / meaningful (excluded from hybrid count).
    """
    head = token.head
    head_lm = head.lemma_.lower() if head is not None else ""
    return hybrid_classify_tags(
        token.lemma_.lower(),
        token.pos_,
        token.dep_,
        head_lm,
        text=token.doc.text,
        start=int(token.idx),
        end=int(token.idx) + len(token.text),
    )


def occurrence_is_filler(occ: AmbiguousOccurrence, *, text: str = "") -> bool:
    """Re-evaluate filler vs grammatical using stored tags (and optional full text)."""
    raw = text if text else ""
    return hybrid_classify_tags(
        occ.lemma,
        occ.pos,
        occ.dep,
        occ.head_lemma,
        text=raw,
        start=occ.start,
        end=occ.end,
    )


def _heuristic_unparsed_filler(lemma: str, text: str, start: int, end: int) -> bool:
    """Regex fallback when spaCy did not align a token."""
    lem = lemma.lower()
    if lem == "well" and _preceded_by_as(text, start):
        return False
    if lem == "so":
        after = text[end : end + 12].lower()
        if re.match(r"\s+far\b", after):
            return False
        if re.match(r"\s+(many|much|long|few|little)\b", after):
            return False
    if lem == "like":
        nxt = _next_word_after(text, end)
        if nxt in _LIKE_QUANTIFIER_NEXT:
            return False
        before = text[max(0, start - 12) : start].lower()
        if re.search(r"\b(was|were|is|are|am|be|been|being)\s+$", before):
            return True
    if _at_discourse_boundary(text, start):
        return True
    return lem == "like"


def _span_overlaps(a: int, b: int, occupied: List[Tuple[int, int]]) -> bool:
    for x, y in occupied:
        if a < y and b > x:
            return True
    return False


def text_with_filtered_fillers(raw: str, occurrences: List[AmbiguousOccurrence]) -> str:
    """
    Keep grammatical *like* / *well* / *so* as-is; replace discourse fillers with
    ``filtered_like``, ``filtered_well``, or ``filtered_so``.
    """
    s = str(raw)
    if not s:
        return s

    occupied: List[Tuple[int, int]] = [(o.start, o.end) for o in occurrences]
    replacements: List[Tuple[int, int, str]] = []

    for o in occurrences:
        if occurrence_is_filler(o, text=s):
            label = _FILLER_LABEL.get(o.lemma.lower())
            if label:
                replacements.append((o.start, o.end, label))

    for m in _RE_AMBIGUOUS_WORD.finditer(s):
        a, b = m.start(), m.end()
        if _span_overlaps(a, b, occupied):
            continue
        lem = m.group(1).lower()
        if _heuristic_unparsed_filler(lem, s, a, b):
            label = _FILLER_LABEL.get(lem)
            if label:
                replacements.append((a, b, label))

    if not replacements:
        return s

    n = len(s)
    clipped: List[Tuple[int, int, str]] = []
    for a, b, label in replacements:
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        if a < b:
            clipped.append((a, b, label))
    clipped.sort(key=lambda x: x[0], reverse=True)

    out = s
    for a, b, label in clipped:
        out = out[:a] + label + out[b:]
    return out


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


_RE_COLLAPSE_WS = re.compile(r"\s+")


def text_without_nonfiller_ambiguous(raw: str, occurrences: List[AmbiguousOccurrence]) -> str:
    """
    Delete *like* / *well* / *so* spans that ``hybrid_classify_token`` marked as not filler.
    Keeps filler uses and all other words; collapses whitespace where gaps were left.
    """
    spans = [(o.start, o.end) for o in occurrences if not o.hybrid_is_filler]
    if not spans:
        return str(raw)
    s = str(raw)
    n = len(s)
    clipped: List[Tuple[int, int]] = []
    for a, b in spans:
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        if a < b:
            clipped.append((a, b))
    if not clipped:
        return s
    merged = _merge_spans(clipped)
    parts: List[str] = []
    cur = 0
    for a, b in merged:
        if a > cur:
            parts.append(s[cur:a])
        cur = max(cur, b)
    if cur < n:
        parts.append(s[cur:])
    return _RE_COLLAPSE_WS.sub(" ", "".join(parts)).strip()


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
