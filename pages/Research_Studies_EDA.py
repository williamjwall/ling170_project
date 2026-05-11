"""
Research snapshots (ages 18–24): emotion filler comparison vs phone F/M fillers,
plus a hybrid POS + dependency pilot (one chart + before/after text).

Run: streamlit run streamlit_app.py → open this page from the sidebar.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app import (
    ALL_FILLER_NAMES,
    DEFAULT_CSV,
    attach_filler_columns,
    load_corpus,
    render_filler_emotion_by_task,
    render_filler_female_male,
    render_filler_insights_emotion,
    render_filler_insights_phone_fm,
)

RESEARCH_EXPORTS = REPO_ROOT / "ucla_box_parsed" / "research_exports"

HYBRID_NUMERIC_COLS = (
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


def _ensure_research_exports_dir() -> Path:
    RESEARCH_EXPORTS.mkdir(parents=True, exist_ok=True)
    return RESEARCH_EXPORTS


def _export_csv_timestamped(df: pd.DataFrame, stem: str) -> Path:
    """Write a new timestamped CSV under ucla_box_parsed/research_exports/ (keeps older runs)."""
    out_dir = _ensure_research_exports_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stem}_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def _filler_per_file_export_columns(df: pd.DataFrame) -> List[str]:
    base = [
        "speaker_id",
        "folder_id",
        "file_id",
        "file_name",
        "extension",
        "session",
        "session_key",
        "task",
        "variant",
        "textgrid_role",
        "_word_count",
        "_filler_total",
        "_filler_per100",
    ]
    pat = [f"_f_{n}" for n in ALL_FILLER_NAMES]
    hybrid_extra = [c for c in HYBRID_NUMERIC_COLS if c in df.columns]
    return [c for c in base + pat + hybrid_extra if c in df.columns]


# this is for will, notes for later debugginggggggggg
# - Hybrid pilot: Research_Studies_EDA.render_hybrid_pilot_full_page + _cached_hybrid_analyses
# - Rules live in hybrid_filler_spacy.hybrid_classify_token / analyze_transcript
# - cache_data key is (tuple(texts), model_name); change either to force re-parse
# - cache_resource loads nlp once per process per model_name


def _hybrid_per_file_export_columns(df: pd.DataFrame) -> List[str]:
    meta = [
        c
        for c in (
            "speaker_id",
            "folder_id",
            "file_id",
            "file_name",
            "session",
            "task",
            "session_key",
            "variant",
            "textgrid_role",
        )
        if c in df.columns
    ]
    nums = [c for c in HYBRID_NUMERIC_COLS if c in df.columns]
    tail = [c for c in ("_word_count", "_filler_total", "_filler_per100", "_filler_total_after_hybrid") if c in df.columns]
    return meta + nums + tail


@st.cache_resource(show_spinner="Loading spaCy model…")
def _research_spacy_nlp(model_name: str) -> Any:
    # this is for will, notes for later debugginggggggggg: OSError here → UI catch; wrong model name → install wheel
    import spacy

    return spacy.load(model_name)


@st.cache_data(show_spinner="Hybrid POS/dep (like · well · so)…")
def _cached_hybrid_analyses(texts: Tuple[str, ...], model_name: str) -> Tuple[Dict[str, Any], ...]:
    from hybrid_filler_spacy import analyze_transcript

    nlp = _research_spacy_nlp(model_name)
    # this is for will, notes for later debugginggggggggg: len(texts) large → long first run; tuple(texts) must be stable order
    return tuple(analyze_transcript(t, nlp) for t in texts)

AGE_MIN = 18
AGE_MAX = 24
EMO_TASKS = ("happy", "annoyed", "neutral")


def render_filler_tab_emotion_tasks(f_base: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    """Defined here so the page does not depend on wrapper exports in streamlit_app."""
    st.markdown("### Fillers · emotion comparison")
    st.caption("Neutral, happy, and annoyed tasks only; same filler patterns as the main app.")
    use_all_lengths = st.checkbox(
        "Include every transcript length (set min words to 0)",
        value=False,
        key=f"{widget_key_prefix}filler_all_lengths",
        help="Unchecked: use min words below. Checked: no length cutoff—full emotion-task cohort.",
    )
    min_words = st.number_input(
        "Min words per file",
        0,
        500,
        20,
        10,
        key=f"{widget_key_prefix}filler_min_words",
        disabled=use_all_lengths,
    )
    eff_min = 0 if use_all_lengths else int(min_words)
    work = f_base[f_base["_word_count"] >= eff_min].copy()
    if len(work) == 0:
        st.warning("No rows left—relax filters or lower min words.")
        return
    with st.spinner("Counting…"):
        fw = attach_filler_columns(work)

    use_hybrid = st.checkbox(
        "Use spaCy-adjusted counts for *like* / *well* / *so* (runs parser on **all** files above)",
        value=False,
        key=f"{widget_key_prefix}use_hybrid_fillers",
        help="Other patterns stay regex-only. First run parses every transcript in this tab’s slice; use the Hybrid tab for highlights.",
    )
    hybrid_model = "en_core_web_sm"
    if use_hybrid:
        hybrid_model = st.selectbox(
            "spaCy pipeline",
            ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"),
            index=0,
            key=f"{widget_key_prefix}hybrid_model_fillers",
        )
    fw_display = fw
    if use_hybrid:
        try:
            with st.spinner("spaCy: adjusting like / well / so on every transcript…"):
                fw_display = apply_hybrid_metrics_to_filler_frame(fw, hybrid_model)
            st.caption(
                "**Data source:** spaCy hybrid for *like*, *well*, and *so*; **regex** for every other pattern. "
                "**Totals** subtract only the three-word overcount on each file."
            )
        except OSError as e:
            st.error(
                f"Could not load `{hybrid_model}` ({e}). Install spaCy + model (see Hybrid pilot → Setup), "
                "or run `bash scripts/install_spacy_pep668.sh`."
            )
            fw_display = fw

    total_words = int(fw_display["_word_count"].sum())
    m1, m2 = st.columns(2)
    m1.metric("Files", f"{len(fw_display):,}")
    m2.metric("Words", f"{total_words:,}")
    render_filler_emotion_by_task(fw_display)
    render_filler_insights_emotion(fw_display)

    with st.expander("Export full cohort to disk", expanded=False):
        st.caption(
            f"Saves **one new timestamped CSV** under `{RESEARCH_EXPORTS.relative_to(REPO_ROOT)}` "
            "(older exports are kept). Omits full transcript text to keep files small. "
            "Reflects **spaCy-adjusted** columns if that option is turned on."
        )
        if st.button("Save per-file filler counts (CSV)", key=f"{widget_key_prefix}export_emotion"):
            cols = _filler_per_file_export_columns(fw_display)
            path = _export_csv_timestamped(fw_display[cols].copy(), "research_fillers_emotion_age18-24")
            st.success(f"Wrote `{path}`")


def render_filler_tab_phone_sex_only(f_base: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    """Defined here so the page does not depend on wrapper exports in streamlit_app."""
    st.markdown("### Fillers · phone transcripts")
    st.caption("Female vs male on phonecall transcripts; same filler patterns as the main app.")
    use_all_lengths = st.checkbox(
        "Include every transcript length (min words = 0)",
        value=False,
        key=f"{widget_key_prefix}filler_all_lengths",
        help="Unchecked: use min words below. Checked: full phone cohort by length.",
    )
    min_words = st.number_input(
        "Min words per file",
        0,
        500,
        20,
        10,
        key=f"{widget_key_prefix}filler_min_words",
        disabled=use_all_lengths,
    )
    eff_min = 0 if use_all_lengths else int(min_words)
    work = f_base[f_base["_word_count"] >= eff_min].copy()
    if len(work) == 0:
        st.warning("No rows left—relax filters or lower min words.")
        return
    with st.spinner("Counting…"):
        fw = attach_filler_columns(work)

    use_hybrid = st.checkbox(
        "Use spaCy-adjusted counts for *like* / *well* / *so* (runs parser on **all** files above)",
        value=False,
        key=f"{widget_key_prefix}use_hybrid_fillers",
        help="Other patterns stay regex-only. Parses every phone transcript in this slice.",
    )
    hybrid_model = "en_core_web_sm"
    if use_hybrid:
        hybrid_model = st.selectbox(
            "spaCy pipeline",
            ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"),
            index=0,
            key=f"{widget_key_prefix}hybrid_model_fillers",
        )
    fw_display = fw
    if use_hybrid:
        try:
            with st.spinner("spaCy: adjusting like / well / so on every transcript…"):
                fw_display = apply_hybrid_metrics_to_filler_frame(fw, hybrid_model)
            st.caption(
                "**Data source:** spaCy hybrid for *like*, *well*, *so*; **regex** for all other patterns."
            )
        except OSError as e:
            st.error(
                f"Could not load `{hybrid_model}` ({e}). Install spaCy + model (Hybrid pilot → Setup), "
                "or run `bash scripts/install_spacy_pep668.sh`."
            )
            fw_display = fw

    total_words = int(fw_display["_word_count"].sum())
    total_hits = int(fw_display["_filler_total"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Files", f"{len(fw_display):,}")
    m2.metric("Words", f"{total_words:,}")
    m3.metric("Total filler hits", f"{total_hits:,}")
    render_filler_female_male(fw_display, compact_caption=True)
    render_filler_insights_phone_fm(fw_display)

    with st.expander("Export full cohort to disk", expanded=False):
        st.caption(
            f"Saves **one new timestamped CSV** under `{RESEARCH_EXPORTS.relative_to(REPO_ROOT)}` "
            "(older exports are kept). Omits full transcript text. "
            "Reflects **spaCy-adjusted** columns if that option is on."
        )
        if st.button("Save per-file filler counts (CSV)", key=f"{widget_key_prefix}export_phone"):
            cols = _filler_per_file_export_columns(fw_display)
            path = _export_csv_timestamped(fw_display[cols].copy(), "research_fillers_phone_age18-24")
            st.success(f"Wrote `{path}`")


def filter_age_18_24(df: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(df["info_age"], errors="coerce")
    m = a.ge(AGE_MIN) & a.le(AGE_MAX)
    return df.loc[m].copy()


def apply_common_filters(df: pd.DataFrame, hide_errors: bool) -> pd.DataFrame:
    out = filter_age_18_24(df)
    if hide_errors:
        out = out[~out["text"].astype(str).str.startswith("[ERROR:", na=False)]
    return out


def sidebar_csv() -> Tuple[Path, bool]:
    st.sidebar.markdown("### Data source")
    csv_path = st.sidebar.text_input("CSV path", value=str(DEFAULT_CSV))
    path = Path(csv_path).expanduser()
    if not path.is_file():
        st.sidebar.error("CSV not found.")
        st.stop()
    hide_errors = st.sidebar.checkbox("Hide parse-error rows", value=True)
    return path, hide_errors


def cohort_header(work: pd.DataFrame) -> None:
    """One row of context."""
    if len(work) == 0:
        st.warning("No rows match age and task filters.")
        return
    w = ", ".join(sorted(work["task"].unique().tolist()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Transcript files", f"{len(work):,}")
    c2.metric("Speakers", f"{work['speaker_id'].nunique():,}")
    c3.metric("Tasks in slice", w)


def render_hybrid_intro_and_flow() -> None:
    """Readable intro for the hybrid pilot; heavy detail lives in one appendix expander."""
    st.markdown(
        "**What this pilot adds.** The main Fillers tab counts **fixed strings** in the transcript. "
        "That is fine for *um*, *you know*, and most items—but **like**, **well**, and **so** are special: "
        "the same spelling can be a **discourse marker** (structuring talk, hedging, pausing) or a **normal piece of syntax** "
        "(preposition, degree adverb, coordinator, etc.)."
    )
    st.markdown(
        "We keep the string-level count as a **baseline** (“before”). Then an **automatic parser** (spaCy) proposes, "
        "for each of those three words, a **category** and **how it hooks into the clause**—the sort of evidence you would "
        "use if you asked “is this integrated in the syntax tree or just floating as discourse?” A short **rule pass** "
        "uses that evidence to **drop** uses we treat as structural; what survives is the **hybrid** (“after”) count. "
        "The chart and the transcript colours show how far **after** moves from **before**."
    )
    st.markdown(
        "**How to read the page.** "
        "**Chart:** taller “before” bar = more raw matches; shorter “after” bar = fewer once structural uses are peeled off. "
        "**Transcript:** yellow highlights every *like* / *well* / *so* the parser aligned to a token; "
        "**green** = still counted in the hybrid total; **red strikethrough** = treated as structural and removed there."
    )
    st.markdown(
        "**Pipeline.** "
        "(1) Choose a slice of files → "
        "(2) Regex counts every *like*, *well*, *so* (same patterns as the main app) → "
        "(3) spaCy builds a parse → "
        "(4) Rules split **discourse-leaning** vs **syntax-integrated** readings → "
        "(5) Compare **before** and **after**."
    )
    st.caption(
        "Worked examples, Universal Dependencies–style tag glosses, removal logic, and programmer notes → "
        "**Appendix** (collapsed by default). On **Emotion** and **Phone** tabs you can turn on the same spaCy "
        "adjustment for *like* / *well* / *so* on the whole cohort there."
    )

    with st.expander("Appendix: examples, parser tags (UD-style), and implementation", expanded=False):
        st.markdown(
            r"""
### Same surface form, different syntactic job

Only **like · well · so** get this second pass; other fillers stay on pure string matching.

**Well**

| Reading | Example | Pilot leans… | Parser cues we use (when spaCy cooperates) |
| --- | --- | --- | --- |
| Opener / interjection | “**Well**, I don’t know.” | Filler | Interjection-style tags (`INTJ`, relation `intj`). |
| Manner adverb | “She did **well**.” | Structural (removed) | Adverb modifying the verb (`ADV`, often `advmod`). |
| Predicative adjective | “I hope you’re **well**.” | Structural (removed) | Adjective (`ADJ`). |
| Noun (rare in convo) | “the village **well**” | Structural (removed) | Noun (`NOUN`). |

**Like**

| Reading | Example | Pilot leans… | Parser cues |
| --- | --- | --- | --- |
| Hedge / discourse | “It was **like**, twelve of them.” | Filler | Anything that is *not* classified as prep / lexical verb / clear content POS below. |
| Preposition (similarity) | “She sounds **like** her mom.” | Structural (removed) | Adposition + prepositional object hook (`ADP`, `prep`). |
| Verb (preference) | “I **like** cats.” | Structural (removed) | Verb (`VERB`). |
| Filled pause | “I was **like** waiting…” | Often filler | Model-dependent; check **Token detail** on a real line. |

**So**

| Reading | Example | Pilot leans… | Parser cues |
| --- | --- | --- | --- |
| Turn / bridge | “**So**, we should go.” | Often filler | Interjection, or special-case `ADV` on the main verb (incl. comma heuristic in code). |
| Intensifier | “**so** cold”, “**so** many” | Structural (removed) | `ADV` modifying adjective / noun / quantity (`advmod` on `ADJ`, `NOUN`, …). |
| Coordinator (result) | “It was raining, **so** I stayed.” | Structural (removed) | Coordinating conjunction (`CCONJ`). |
| Subordinator (cause) | “**So** I left early …” (if tagged that way) | Structural (removed) | Subordinating conjunction (`SCONJ`). |

Automatic tags **mis-fire** on fragments and fast speech. Use the per-line **Token detail** table (POS, dependency, head lemma) to see what the model actually did.

---

### When we remove a hit from the “after” count

**Baseline (“before”).** Count every word-boundary match for *like*, *well*, and *so* in the raw transcript string—the same idea as the main app’s patterns.

**Second pass (“after”).** For each of those words that spaCy tokenizes as its own word, we read its **lemma**, **part-of-speech** (coarse Universal POS–style label), **dependency relation** to its **head** (what it attaches to in the parse), and the head’s category. If the combination matches a **structural** profile in our rules, that token is **not** added to the hybrid total for that lemma. If it does **not** match a structural profile, we **keep** it as discourse-like for counting purposes (**conservative default**: unclear cases stay on the “keep” side so we do not wipe messy speech to zero).

**Mismatch safety.** If the string matcher finds *more* hits than the parser gave us separate tokens for, the extra hits stay in the hybrid count so numbers do not drift for opaque tokenization reasons.

---

### Implementation (for reproducibility and debugging)

- **Rule source:** `hybrid_filler_spacy.py`, function `hybrid_classify_token` (returns keep vs structural); per-file aggregation in `analyze_transcript`.
- **Alignment:** Regex counts are on the raw string; hybrid counts walk spaCy `Token` objects whose surface form is one of the three lemmas (light punctuation strip on the token text).
- **Highlights:** Character spans come from `token.idx` on the same string passed into the parser; HTML escaping happens only at render time.
- **Caching (Streamlit):** one loaded pipeline per model name (`@st.cache_resource`); parsed results memoized on the tuple of transcript texts plus model name (`@st.cache_data`). Clear Streamlit cache after editing rules, or nudge inputs, if numbers look stale.

#### this is for will, notes for later debugginggggggggg

- `_cached_hybrid_analyses` — `st.cache_data`; `texts` must stay a **tuple** for hashing. Stale after code edits → **Clear cache** or change the slice.
- `_research_spacy_nlp` — first successful load per `model_name` sticks for the process; restart after installing a new model.
- `HYBRID_NUMERIC_COLS` — new numeric keys from `analyze_transcript` must be added here or `_merge_hybrid_into_work` drops them.
- `_filler_total_after_hybrid` — only subtracts the like/well/so adjustment row; full regex filler totals still come from `attach_filler_columns` / `streamlit_app._count_fillers_one`.
- **Token detail empty but regex > 0** — tokenizer may not have surfaced a standalone token; compare `parsed_*` vs `regex_*` on that row.
- **Discourse *like*** — often `PART` / `SCONJ` / `INTJ`; many non-prep/verb paths still **count** in hybrid unless you tighten `hybrid_classify_token`.
"""
        )


def _merge_hybrid_into_work(work: pd.DataFrame, analyses: Tuple[Dict[str, Any], ...]) -> pd.DataFrame:
    # this is for will, notes for later debugginggggggggg: analyses order must match work rows after reset_index
    out = work.reset_index(drop=True).copy()
    for c in HYBRID_NUMERIC_COLS:
        out[c] = 0
    occ_lists: list = []
    for i, d in enumerate(analyses):
        occ_lists.append(d.get("occurrences", []))
        for c in HYBRID_NUMERIC_COLS:
            out.at[i, c] = int(d.get(c, 0))
    out["_hybrid_occurrences"] = occ_lists
    return out


def apply_hybrid_metrics_to_filler_frame(fw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Recompute per-file filler columns using spaCy for *like* / *well* / *so* only.

    Replaces ``_f_like``, ``_f_well``, ``_f_so`` with hybrid counts, subtracts
    ``removed_ambiguous_total`` from ``_filler_total``, and refreshes ``_filler_per100``.
    All other patterns stay regex-based. Runs on **every row** of ``fw`` (full cohort for that tab).
    """
    fw2 = fw.reset_index(drop=True).copy()
    analyses = _cached_hybrid_analyses(tuple(fw2["text"].astype(str)), model_name)
    out = _merge_hybrid_into_work(fw2, analyses)
    for n in ("like", "well", "so"):
        out[f"_f_{n}"] = out[f"hybrid_{n}"]
    out["_filler_total"] = (out["_filler_total"] - out["removed_ambiguous_total"]).clip(lower=0)
    w = out["_word_count"].replace({0: pd.NA}).astype(float)
    out["_filler_per100"] = (100.0 * out["_filler_total"] / w).fillna(0.0)
    return out


def render_hybrid_pilot_full_page(df_age: pd.DataFrame) -> None:
    """Hybrid pilot: intro, one chart, transcript before/after."""
    from hybrid_filler_spacy import before_after_html, hybrid_styles_block

    st.markdown("### Hybrid pilot")
    render_hybrid_intro_and_flow()

    with st.expander("Setup: spaCy + English model (click if install failed)", expanded=False):
        st.markdown(
            "Use **`python3`** on Debian/WSL. Easiest: `sudo apt install python3.12-venv` → `python3 -m venv .venv` → "
            "`.venv/bin/pip install -r requirements.txt` → `.venv/bin/python -m spacy download en_core_web_sm`. "
            "Or run `bash scripts/install_spacy_pep668.sh` from the repo root (PEP 668 workaround)."
        )

    st.markdown("#### Choose a slice, then run the parser")
    parse_all_files = st.checkbox(
        "Parse **every** file matching filters (ignore max-files cap; can be slow)",
        value=False,
        key="hybrid_parse_all",
        help="Unchecked: only the first N files (slider), longest transcripts first. Checked: all rows after task + min-words filters.",
    )
    model_name = st.selectbox(
        "spaCy pipeline",
        ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"),
        index=0,
        key="hybrid_model",
    )
    max_files = st.slider(
        "Max transcript files to parse (when not parsing all)",
        10,
        400,
        120,
        10,
        key="hybrid_max",
        disabled=parse_all_files,
    )
    min_words = st.number_input(
        "Min words per file",
        0,
        500,
        20,
        10,
        key="hybrid_minw",
    )
    task_opts = sorted(df_age["task"].dropna().unique().tolist())
    tasks_pick = st.multiselect(
        "Tasks (empty = all)",
        task_opts,
        default=[],
        key="hybrid_tasks",
    )
    work = df_age[df_age["_word_count"] >= min_words].copy()
    if tasks_pick:
        work = work[work["task"].isin(tasks_pick)]
    work = work.sort_values("_word_count", ascending=False)
    if not parse_all_files:
        work = work.head(int(max_files))
    if len(work) == 0:
        st.warning("No rows match filters.")
        return

    try:
        analyses = _cached_hybrid_analyses(tuple(work["text"].astype(str)), model_name)
    except OSError as e:
        st.error(
            f"Could not load spaCy model `{model_name}` ({e}). "
            "Install spaCy with python3 (see the Setup expander above), then install the matching model wheel from "
            "https://github.com/explosion/spacy-models/releases — or run `bash scripts/install_spacy_pep668.sh` from the repo root."
        )
        return

    merged = _merge_hybrid_into_work(work, analyses)
    with st.spinner("Regex filler totals (all patterns)…"):
        merged_fw = attach_filler_columns(merged)
    merged_fw["_filler_total_after_hybrid"] = (
        merged_fw["_filler_total"] - merged_fw["removed_ambiguous_total"]
    ).clip(lower=0)

    rsum = int(merged_fw["regex_ambiguous_total"].sum())
    hsum = int(merged_fw["hybrid_ambiguous_total"].sum())
    rem = int(merged_fw["removed_ambiguous_total"].sum())
    pct = 100.0 * rem / rsum if rsum else 0.0
    tot_r = int(merged_fw["_filler_total"].sum())
    tot_h = int(merged_fw["_filler_total_after_hybrid"].sum())

    st.markdown("#### Results on your slice")
    st.caption(
        f"**{len(merged_fw):,}** transcript files in this run. "
        "**Before** = naive string matches for *like* / *well* / *so*; **after** = same words once structural readings are peeled off. "
        f"Here: **{rem:,}** hits dropped (**{pct:.1f}%** of the naive total for those three words). "
        f"Whole filler inventory (all patterns): **{tot_r:,}** naive hits → **{tot_h:,}** after this adjustment only; "
        "everything except *like* / *well* / *so* is unchanged."
    )

    rl, rw, rs = (
        int(merged_fw["regex_like"].sum()),
        int(merged_fw["regex_well"].sum()),
        int(merged_fw["regex_so"].sum()),
    )
    hl, hw, hs = (
        int(merged_fw["hybrid_like"].sum()),
        int(merged_fw["hybrid_well"].sum()),
        int(merged_fw["hybrid_so"].sum()),
    )
    chart_df = pd.DataFrame(
        {
            "Before (regex)": [rl, rw, rs, rsum],
            "After (hybrid)": [hl, hw, hs, hsum],
        },
        index=["like", "well", "so", "Total (like+well+so)"],
    )
    st.bar_chart(chart_df, height=300)
    st.caption("Bars = raw counts in this slice. **Before** counts every string match; **after** is smaller when the parser read the word as doing real syntax instead of loose discourse.")

    chart_explain = pd.DataFrame(
        {
            "Word": ["like", "well", "so", "Total (sum of the three)"],
            "Before": [rl, rw, rs, rsum],
            "After": [hl, hw, hs, hsum],
            "What this is showing (plain language)": [
                "The naive counter treats every *like* the same. **After** keeps tokens that still look discourse-y to the rules; it strips *like* when it is doing grammar—often preposition-like (*sounds like her*) or verb-like (*I like pizza*).",
                "Same for *well*: the string matcher cannot tell an opener from manner *well* (*did well*) or predicative *well* (*hope you’re well*). **After** tries to keep the former and drop the latter.",
                "*So* can link a new turn, intensify (*so tired*), or mark cause (*rained, so I stayed*). **After** keeps bridge-y cases the model labels that way and pulls out degree and connective *so* so they are not double-counted as “filler words.”",
                "Adds up the three rows. The **before − after** gap is how many hits in this slice moved from “count every spelling” to “count only what we still treat as filler-like for these three items.”",
            ],
        }
    )
    st.markdown("##### Read the chart with a short gloss")
    st.dataframe(
        chart_explain,
        use_container_width=True,
        hide_index=True,
        height=min(380, 120 + 68 * len(chart_explain)),
        column_config={
            "What this is showing (plain language)": st.column_config.TextColumn(
                "What this is showing (plain language)",
                width="large",
            ),
            "Before": st.column_config.NumberColumn("Before", format="%d"),
            "After": st.column_config.NumberColumn("After", format="%d"),
        },
    )

    with st.expander("Export hybrid run to disk", expanded=False):
        st.caption(
            f"Saves **one new timestamped CSV** under `{RESEARCH_EXPORTS.relative_to(REPO_ROOT)}` "
            "(older runs kept). Omits `text` and parsed token lists; join back on `file_id` if needed."
        )
        if st.button("Save per-file hybrid + filler totals (CSV)", key="hybrid_export_btn"):
            cols = _hybrid_per_file_export_columns(merged_fw)
            path = _export_csv_timestamped(merged_fw[cols].copy(), "research_hybrid_like-well-so_age18-24")
            st.success(f"Wrote `{path}`")

    st.markdown("#### See it in a transcript")
    st.caption(
        "**Left:** every *like* / *well* / *so* the parser aligned (yellow). **Right:** green = still counted as filler; "
        "red strikethrough = grammatical, no longer in the hybrid count."
    )
    st.markdown(hybrid_styles_block(), unsafe_allow_html=True)

    pick_labels = (
        merged_fw["speaker_id"].astype(str)
        + " · "
        + merged_fw["session"].astype(str)
        + " · "
        + merged_fw["task"].astype(str)
        + " — "
        + merged_fw["file_name"].astype(str)
        + " (removed "
        + merged_fw["removed_ambiguous_total"].astype(str)
        + ")"
    ).tolist()
    pick_i = st.selectbox("Transcript", range(len(merged_fw)), format_func=lambda i: pick_labels[i], key="hybrid_pick")
    row = merged_fw.iloc[pick_i]
    occ = row["_hybrid_occurrences"]
    before_h, after_h = before_after_html(str(row["text"]), occ)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Before (all ambiguous tokens highlighted)**")
        st.markdown(
            f'<div style="font-size:0.9rem;line-height:1.45;max-height:420px;overflow:auto;">{before_h}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("**After (hybrid filler vs excluded)**")
        st.markdown(
            f'<div style="font-size:0.9rem;line-height:1.45;max-height:420px;overflow:auto;">{after_h}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Token detail (parsed occurrences only)", expanded=False):
        if not occ:
            st.caption("No *like* / *well* / *so* tokens in this transcript (or parser found none).")
        else:
            rows = []
            for o in occ:
                rows.append(
                    {
                        "surface": o.surface,
                        "lemma": o.lemma,
                        "POS": o.pos,
                        "dep": o.dep,
                        "head": o.head_lemma,
                        "hybrid = filler": o.hybrid_is_filler,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_hybrid_parser_cues_deep_dive()


def render_hybrid_parser_cues_deep_dive() -> None:
    """
    Long-form reader at the bottom of the Hybrid pilot: what spaCy gives us and how rules use it.
    Kept separate so the rest of the tab stays scannable.
    """
    st.markdown("---")
    st.markdown("### Deep dive · Parser cues we use (when spaCy cooperates)")
    st.caption(
        "Below is the same logic the pilot implements in code—written for **linguistics readers**, not engineers. "
        "If something here disagrees with a **Token detail** row, trust the table for that sentence: the model wins."
    )

    with st.expander("Read the full walk-through (POS, dependencies, heads, failure modes)", expanded=False):
        st.markdown(
            r"""
#### Why we say “when spaCy cooperates”

spaCy’s English pipelines are trained mostly on **edited written** text plus some speech-like data. Spontaneous
orthographic transcripts have **fragments, overlaps, repairs, and weird punctuation**. The parser still assigns
each token a **lemma** (dictionary form), a coarse **part-of-speech tag** (`POS`, UD-style), a **dependency relation**
(`dep`) to a **head** (parent token), and a **sentence** unit. Those are **hypotheses**, not annotations. When they
line up with your reading, we treat them as **evidence** for “discourse-y” vs “syntax-integrated.” When they don’t,
our rules can misfire—that is what “cooperates” is trying to flag.

---

#### The four fields we actually read on each *like* / *well* / *so* token

| Field | You can think of it as… | Why it matters for fillers |
| --- | --- | --- |
| **Lemma** | Citation form spaCy guessed (*like*, *well*, *so*) | Tells us **which** of the three rule families to apply. |
| **`POS` (UPOS-style)** | Very coarse word class: `INTJ`, `ADV`, `ADP`, `VERB`, `CCONJ`, `SCONJ`, … | Separates **interjection**-flavor from **modifier**-flavor from **function-word** connective flavor. |
| **`dep`** | How this word **attaches** in the tree: `advmod`, `prep`, `intj`, `cc`, … | Tells us **what job** the parser thinks the word is doing for its head. |
| **Head (parent token)** | The word this token **depends on** in the parse | Lets us ask: is *so* modifying an adjective, attaching to the whole clause as a bridge, or linking two clauses? |

The **Token detail** table under a transcript is literally these fields row-by-row for each hit.

---

#### Universal Dependencies, in one paragraph

UD uses a small set of **content** vs **function** intuitions expressed as **head–dependent** pairs: every token
except the root has exactly one head; the **`dep`** label names the relationship (**advmod** = adverb-like modifier,
**prep** = preposition introducing an object, **intj** = interjection-like, **cc** = coordinating marker, …). We do
**not** re-run a full linguistic analysis in this app—we **piggyback** on spaCy’s UD-ish analysis and apply a **tiny**
hand-written policy on top.

---

#### *Well* — what the rules are trying to hear

**Keep as filler (hybrid = yes)** when spaCy says something interjection-like: `POS = INTJ` **or** `dep = intj`. That
matches the intuition of *well* as a **response marker** or floor-holder rather than a predicate modifier.

**Remove (hybrid = no)** when spaCy treats *well* as **lexical content** in the clause:

- **`ADJ`** — predicative or attributive adjective reading (*feel **well***, *a **well** person* in principle).
- **`ADV`** with ordinary adverb distribution — especially manner on a verb (*did **well***).
- **`NOUN`** — the noun reading is rare in these transcripts but is clearly **not** the discourse particle.

**Otherwise** we **default to filler** (conservative for messy tags): if the model is confused and does not assign one
of the clear “content” classes above, we still count the hit rather than silently erase it.

---

#### *Like* — what the rules are trying to hear

**Remove** when the parse looks like **compositional syntax**:

- **`ADP` + `dep = prep`** — canonical “**like** a noun / pronoun” comparison frame (*sounds **like** her*).
- **`VERB`** — the main-verb reading (*I **like** cats*).
- **`NOUN` / `PROPN` / `ADJ` / `NUM`** — non-verb lexical categories we treat as “this is not hedge-*like*” when the
  model assigns them (rare edge shapes, but cheap to list).

**Keep as filler** in all other tag combinations. That is where **discourse *like*** usually lives in messy speech—
`PART`, `SCONJ`, `INTJ`, odd `ADV`, etc.—and also where **parser error** lives. So: hedge *like* is **not** reliably
separated from noise; the pilot is **directional**, not a gold standard for *like*.

---

#### Mini-glossary of `dep` labels you will see in **Token detail**

| `dep` | Everyday gloss | Why we stare at it |
| --- | --- | --- |
| `intj` | Interjection-style hook | Strong signal for *well* / *so* as **floor management**, not argument structure. |
| `advmod` | Adverb-like modifier of the **head** token | For *so*, we look at **what** is being modified (adjective vs verb) to split **degree** from **bridge**. |
| `prep` | Preposition introducing an object | For *like*, **`ADP` + `prep`** is our main “this is grammar-*like*** signature. |
| `cc` / `conj` | Coordination in the broad family | Often co-travel with `CCONJ` POS on *so* when *so* links clauses. |
| `discourse` | Some parsers use this for pragmatic markers | spaCy’s small English models may **not** emit it often; do not expect it everywhere. |

Remember: **`dep` names a relation to the head**, not a category by itself—you almost always read **POS + dep +
head POS** together.

---

#### *So* — what the rules are trying to hear (including punctuation)

**Remove** when the parse is clearly **structural**:

- **`CCONJ`** — coordinating reading (*rained, **so** I stayed*).
- **`SCONJ`** — subordinator-style reading when the model chooses that label.
- **`ADV` + `advmod`** whose **head** is an **`ADJ` / `ADV` / `NOUN` / `DET` / `NUM`** — the classic **degree**
  intensifier (*so* tired, *so* many). Here *so* is doing **scalar modification**, not “discourse *so*.”

**Keep as filler** in a few **deliberately narrow** `ADV` + `advmod` cases that looked like **turn bridges** in pilot
development: e.g. when the head is a **verb/aux** and the **next token is a comma** (rough proxy for “**So**, …”
framing), or when *so* is the **first token of the sentence** and still `advmod` on the main verb—patterns where we
lean **discourse** rather than degree.

**Otherwise** (including odd `ADV` attachments) we **default to filler** again—same conservative stance as *well*.

---

#### When spaCy is *not* cooperating (high-signal cases for spoken data)

- **Repairs and restarts** — the tree may attach *like* to the wrong head; a human would re-parse.
- **Quotative / pragmatic *like*** — tags bounce between `PART`, `SCONJ`, `INTJ`, and `ADV` across model versions.
- **Clause fragments** — without a finite verb, UD heads get weird; `advmod` may land on whatever token the model
  treats as root.
- **Punctuation** — commas and false starts steer the comma-sensitive *so* heuristic; they are not linguistic evidence
  on their own, just a cheap cue.

When you see a **red** strikeout in the transcript that feels wrong, open **Token detail** and ask: *Is the POS+dep
story wrong, or is our rule too blunt?* That is the right order of blame.

---

#### How this connects to the chart

The **before** bar is blind to everything above—it is only string shape. The **after** bar applies these cue-based
policies **per token** that spaCy actually segmented as *like*, *well*, or *so*. The **gloss table** under the chart
summarizes the *linguistic story*; this section is the *mechanistic story* you can line up with `hybrid_classify_token`
in `hybrid_filler_spacy.py` if you want to change the policy later.
"""
        )


def main() -> None:
    st.title("Research · fillers")
    st.caption(f"Ages **{AGE_MIN}–{AGE_MAX}** · orthographic tier only.")

    path, hide_errors = sidebar_csv()

    try:
        df_raw = load_corpus(str(path))
    except Exception as e:
        st.exception(e)
        st.stop()

    df_base = apply_common_filters(df_raw, hide_errors)

    study = st.radio(
        "View",
        [
            "Emotion tasks (neutral · happy · annoyed)",
            "Phone transcripts (female · male)",
            "Hybrid pilot (POS + dep)",
        ],
        horizontal=True,
    )

    if study.startswith("Hybrid"):
        render_hybrid_pilot_full_page(df_base)
        return

    if study.startswith("Emotion"):
        st.markdown(
            "**Goal:** Compare **filler use** across **neutral**, **happy**, and **annoyed** monologues "
            "(same protocol, ages 18–24)."
        )
        work = df_base[df_base["task"].isin(EMO_TASKS)].copy()
        cohort_header(work)
        render_filler_tab_emotion_tasks(work, widget_key_prefix="s1_")
    else:
        st.markdown(
            "**Goal:** Compare **female vs male** speakers on **phonecall** transcripts (ages 18–24)."
        )
        work = df_base[df_base["task"].eq("phonecall")].copy()
        cohort_header(work)
        render_filler_tab_phone_sex_only(work, widget_key_prefix="s2_")


if __name__ == "__main__":
    main()
