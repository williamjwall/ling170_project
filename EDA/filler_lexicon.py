"""
Filler detection: fixed word list, longest match first, no double counting.
Edit FILLER_WORDS or CATEGORY_FOR_FILLER to change behavior.

In this study, uses of *like*, *so*, *actually*, and *basically* were coded as fillers only
when they were not grammatical. On raw transcripts the app skips those four for automatic
counts; use a ``text_filler`` column in the CSV if you export hand-filtered text.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

FILLER_WORDS: list[str] = [
    "i mean",
    "kind of",
    "sort of",
    "you know",
    "actually",
    "basically",
    "er",
    "erm",
    "hmm",
    "like",
    "literally",
    "so",
    "uh",
    "uhm",
    "um",
    "well",
    "err",
    "uhh",
    "uhhh",
    "uhhhh",
    "umm",
    "ummm",
    "ummmm",
    "hm",
    "hmmm",
    "hmmmm",
    "mhm",
    "mhmm",
    "mhmmm",
    "mm",
    "mm-hmm",
    "mmhmm",
    "mm hmm",
    "mh-hmm",
    "mmm",
    "mmmm",
    "uh-huh",
    "uh huh",
    "uhhuh",
    "unh unh",
    "unh-unh",
    "huh",
    "heh",
]

# Not auto-counted on raw transcripts (grammatical vs filler was decided in manual coding).
GRAMMATICAL_AMBIGUOUS: frozenset[str] = frozenset({"like", "so", "actually", "basically"})

# Research buckets (H3 / H4): every token in FILLER_WORDS should appear exactly once.
CATEGORY_LABELS: tuple[str, ...] = ("Placeholders", "Californese", "Feedback")

CATEGORY_FOR_FILLER: dict[str, str] = {
    # hesitation / floor-holding
    "er": "Placeholders",
    "erm": "Placeholders",
    "err": "Placeholders",
    "heh": "Placeholders",
    "hm": "Placeholders",
    "hmm": "Placeholders",
    "hmmm": "Placeholders",
    "hmmmm": "Placeholders",
    "huh": "Placeholders",
    "mhm": "Placeholders",
    "mh-hmm": "Placeholders",
    "mhmm": "Placeholders",
    "mhmmm": "Placeholders",
    "mm": "Placeholders",
    "mm hmm": "Placeholders",
    "mm-hmm": "Placeholders",
    "mmhmm": "Placeholders",
    "mmm": "Placeholders",
    "mmmm": "Placeholders",
    "uh": "Placeholders",
    "uh huh": "Placeholders",
    "uh-huh": "Placeholders",
    "uhh": "Placeholders",
    "uhhhh": "Placeholders",
    "uhhh": "Placeholders",
    "uhhuh": "Placeholders",
    "uhm": "Placeholders",
    "um": "Placeholders",
    "umm": "Placeholders",
    "ummmm": "Placeholders",
    "ummm": "Placeholders",
    "unh unh": "Placeholders",
    "unh-unh": "Placeholders",
    # discourse style (like/so/actually/basically: filler-only in this study; see GRAMMATICAL_AMBIGUOUS)
    "actually": "Californese",
    "basically": "Californese",
    "like": "Californese",
    "literally": "Californese",
    "so": "Californese",
    # oriented toward listener / softening / checking understanding
    "i mean": "Feedback",
    "kind of": "Feedback",
    "sort of": "Feedback",
    "well": "Feedback",
    "you know": "Feedback",
}


def ordered_fillers(skip_grammatical: bool = False) -> tuple[str, ...]:
    uniq: list[str] = []
    seen: set[str] = set()
    for w in FILLER_WORDS:
        k = w.strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    if skip_grammatical:
        uniq = [u for u in uniq if u not in GRAMMATICAL_AMBIGUOUS]
    return tuple(sorted(uniq, key=lambda s: (-len(s), s)))


def category_for_filler(label: str) -> str:
    return CATEGORY_FOR_FILLER.get(label.lower(), "Placeholders")


@lru_cache(maxsize=None)
def _compiled_patterns(skip_grammatical: bool) -> tuple[tuple[re.Pattern[str], str], ...]:
    out: list[tuple[re.Pattern[str], str]] = []
    for phrase in ordered_fillers(skip_grammatical):
        parts = phrase.split()
        if len(parts) == 1:
            pat = r"\b" + re.escape(parts[0]) + r"\b"
        else:
            pat = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
        out.append((re.compile(pat, flags=re.IGNORECASE), phrase))
    return tuple(out)


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text.lower()))


def find_filler_spans(text: str, skip_grammatical: bool = False) -> list[tuple[int, int, str]]:
    if not isinstance(text, str) or not text:
        return []
    lowered = text.lower()
    n = len(text)
    taken = [False] * n
    hits: list[tuple[int, int, str]] = []

    for regex, label in _compiled_patterns(skip_grammatical):
        for m in regex.finditer(lowered):
            s, e = m.span()
            if s >= n or e > n:
                continue
            if any(taken[s:e]):
                continue
            for i in range(s, e):
                taken[i] = True
            hits.append((s, e, label))

    hits.sort(key=lambda h: h[0])
    return hits


def count_by_filler(text: str, skip_grammatical: bool = False) -> dict[str, int]:
    counts = {w: 0 for w in ordered_fillers(skip_grammatical)}
    for _, _, lab in find_filler_spans(text, skip_grammatical=skip_grammatical):
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def count_by_category(text: str, skip_grammatical: bool = False) -> dict[str, int]:
    out = {c: 0 for c in CATEGORY_LABELS}
    for _, _, lab in find_filler_spans(text, skip_grammatical=skip_grammatical):
        cat = category_for_filler(lab)
        out[cat] = out.get(cat, 0) + 1
    return out


def transcript_highlight_html(
    text: str, highlight: set[str] | None = None, skip_grammatical: bool = False
) -> str:
    if not isinstance(text, str):
        text = ""
    spans = find_filler_spans(text, skip_grammatical=skip_grammatical)
    if highlight is not None:
        h = {x.strip().lower() for x in highlight}
        spans = [t for t in spans if t[2] in h]

    parts: list[str] = []
    cur = 0
    for s, e, lab in spans:
        parts.append(html.escape(text[cur:s]))
        cat = category_for_filler(lab)
        cls = {
            "Placeholders": "filler-ph",
            "Californese": "filler-ca",
            "Feedback": "filler-fb",
        }.get(cat, "filler-hit")
        parts.append(f'<mark class="{cls}">')
        parts.append(html.escape(text[s:e]))
        parts.append("</mark>")
        cur = e
    parts.append(html.escape(text[cur:]))
    return "".join(parts)


def enrich_dataframe(df, text_col: str = "text", skip_grammatical: bool = False):
    import pandas as pd

    fillers = ordered_fillers(skip_grammatical)
    wc = df[text_col].map(word_count)
    out = df.copy()
    out["words"] = wc

    rows: list[dict[str, int]] = []
    cat_rows: list[dict[str, int]] = []
    totals: list[int] = []
    for t in df[text_col]:
        c = count_by_filler(str(t), skip_grammatical=skip_grammatical)
        rows.append(c)
        cat = count_by_category(str(t), skip_grammatical=skip_grammatical)
        cat_rows.append(cat)
        totals.append(sum(c.values()))

    count_df = pd.DataFrame(rows, columns=list(fillers))
    cat_df = pd.DataFrame(cat_rows, columns=list(CATEGORY_LABELS))

    for lab in fillers:
        out[f"n {lab}"] = count_df[lab].values

    for cat in CATEGORY_LABELS:
        out[f"n {cat}"] = cat_df[cat].values

    out["n total"] = pd.Series(totals, dtype=int)
    wsafe = out["words"].replace(0, float("nan"))
    out["rate total"] = (out["n total"] / wsafe) * 100.0
    for lab in fillers:
        out[f"rate {lab}"] = (out[f"n {lab}"] / wsafe) * 100.0
    for cat in CATEGORY_LABELS:
        out[f"rate {cat}"] = (out[f"n {cat}"] / wsafe) * 100.0

    return out
