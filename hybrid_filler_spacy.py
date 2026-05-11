"""
POS + dependency disambiguation for ambiguous filler lemmas (like, well, so).

Used by the Research page pilot: compare regex hits vs hybrid-classified counts
and render before/after HTML highlights. Requires spaCy + en_core_web_sm (or lg).

this is for will, notes for later debugginggggggggg
- Tweak rules only in hybrid_classify_token; keep AMBIGUOUS_LEMMAS + regexes in sync if you add lemmas.
- analyze_transcript: hl += max(0, rl - n_like) preserves regex when tokenizer misses a span.
- Occurrence char offsets use token.idx on the same string passed to nlp() (no HTML escape until render).
"""

from __future__ import annotations

import html
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
