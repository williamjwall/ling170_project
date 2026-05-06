"""
UCLA Speaker Variability — interactive text explorer (Streamlit).

Run from repo root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "ucla_box_parsed" / "ucla_text_state_parsed.csv"

# Eligibility + design session (from readme_data.txt). "Typical" — redos may use session D.
TASK_SPECS: Dict[str, Dict[str, str]] = {
    "vowels": {
        "label": "Vowels",
        "typical_sessions": "A, B, C (each session)",
        "summary": "Sustained [a] (as in “spa”) three times with pauses—short, non-conversational.",
        "prompt": (
            'Please say the “aahh” vowel sound (as in the word “spa”) three times, pausing in between, '
            'like this: “aahh” … “aahh” … “aahh”.'
        ),
    },
    "sentences": {
        "label": "Sentence reading",
        "typical_sessions": "A, B, C (each session)",
        "summary": "Read sentences from the screen; five IEEE/Harvard-style sentences, repeated across screens.",
        "prompt": (
            "Each of the next 30 screens shows one sentence. Read it out loud, then click “Next.” "
            "The five recorded sentences include: *The boy was there when the sun rose*; "
            "*Kick the ball straight and follow through*; *Help the woman get back to her feet*; "
            "*A pot of tea helps to pass the evening*; *The soft cushion broke the man's fall*."
        ),
    },
    "instructions": {
        "label": "Instructions / directions",
        "typical_sessions": "A",
        "summary": "Spontaneous ~30s monologue: give the RA directions OR how-to instructions (choose topic).",
        "prompt": (
            "Talk to the RA outside the booth. Give her DIRECTIONS to go somewhere, or INSTRUCTIONS to do something "
            "(your choice). Try to talk for 30 seconds."
        ),
    },
    "neutral": {
        "label": "Neutral (mundane conversation)",
        "typical_sessions": "A",
        "summary": "Retell a boring, unemotional conversation in a “first she said… then I said” style.",
        "prompt": (
            "Tell the RA about a CONVERSATION that wasn’t important—not exciting, not upsetting, just normal. "
            "Repeat it in a “FIRST SHE SAID… THEN I SAID” style. ~30 seconds."
        ),
    },
    "happy": {
        "label": "Happy (exciting conversation)",
        "typical_sessions": "B",
        "summary": "Retell a conversation about something that made the speaker very happy.",
        "prompt": (
            "Talk to the RA. Tell her about a CONVERSATION about something exciting that made you really happy. "
            "Repeat in a “FIRST SHE SAID… THEN I SAID” style. ~30 seconds."
        ),
    },
    "phonecall": {
        "label": "Phone call",
        "typical_sessions": "B",
        "summary": "Real call to a pre-arranged friend/relative; only the speaker’s side is recorded.",
        "prompt": (
            "Use your own phone or ours; call the person you arranged to talk to. "
            "Talk about anything for a couple of minutes. Only your side is recorded."
        ),
    },
    "annoyed": {
        "label": "Annoyed (upsetting conversation)",
        "typical_sessions": "C",
        "summary": "Retell a conversation about something that really annoyed the speaker.",
        "prompt": (
            "Talk to the RA. Tell her about a CONVERSATION about something that really annoyed you. "
            "Repeat in a “FIRST HE SAID… THEN I SAID …” style. ~30 seconds. Don’t embarrass others."
        ),
    },
    "video": {
        "label": "Video (pet-directed speech)",
        "typical_sessions": "C",
        "summary": "Watch ~1 min of kitten or puppy videos; talk out loud to the animals.",
        "prompt": (
            "You’ll watch a 1-minute collection of kitten or puppy videos (your choice). "
            "Talk out loud to the pets. Can you be as cute as they are?"
        ),
    },
}

# Coarse groups for comparing filler rates in a principled way (not “everything not sentences”).
TASK_TO_EDA: Dict[str, str] = {
    "sentences": "read_aloud",
    "instructions": "monologue_to_ra",
    "neutral": "monologue_to_ra",
    "happy": "monologue_to_ra",
    "annoyed": "monologue_to_ra",
    "phonecall": "phone",
    "video": "pet_directed",
    "vowels": "vowel_items",
}

EDA_CATEGORY_LABEL: Dict[str, str] = {
    "read_aloud": "Read-aloud (sentences)",
    "monologue_to_ra": "Monologue to RA (instructions, neutral, happy, annoyed)",
    "phone": "Phone (real interlocutor)",
    "pet_directed": "Pet-directed (video task)",
    "vowel_items": "Vowel task (isolated [a] — not running speech)",
}

EDA_CATEGORY_ORDER = [
    "read_aloud",
    "monologue_to_ra",
    "phone",
    "pet_directed",
    "vowel_items",
]

# --- Filler / discourse marker patterns (word-token orthographic text; case-insensitive) ---
# Multi-word first so substrings don’t skew single-token counts.
PHRASE_PATTERNS: List[Tuple[str, str]] = [
    ("you know", r"\byou know\b"),
    ("i mean", r"\bi mean\b"),
    ("sort of", r"\bsort of\b"),
    ("kind of", r"\bkind of\b"),
]
TOKEN_PATTERNS: List[Tuple[str, str]] = [
    ("um", r"\bum\b"),
    ("uh", r"\buh\b"),
    ("uhm", r"\buhm\b"),
    ("erm", r"\berm\b"),
    ("er", r"\ber\b"),
    ("hmm", r"\bhmm+\b"),
    ("like", r"\blike\b"),
    ("well", r"\bwell\b"),
    ("so", r"\bso\b"),
    ("actually", r"\bactually\b"),
    ("basically", r"\bbasically\b"),
    ("literally", r"\bliterally\b"),
]
ALL_FILLER_NAMES = [n for n, _ in PHRASE_PATTERNS + TOKEN_PATTERNS]


def _count_fillers_one(text: str) -> Tuple[int, Dict[str, int]]:
    """Returns (total hits, per-label counts). Overlapping regex can double-count marginally; ok for EDA."""
    if not text or str(text).startswith("[ERROR"):
        return 0, {n: 0 for n in ALL_FILLER_NAMES}
    t = str(text).lower()
    counts: Dict[str, int] = {n: 0 for n in ALL_FILLER_NAMES}
    total = 0
    for name, pat in PHRASE_PATTERNS + TOKEN_PATTERNS:
        n = len(re.findall(pat, t, flags=re.IGNORECASE))
        counts[name] = n
        total += n
    return total, counts


def _normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace on Excel-merged fields (fixes duplicate sex codes like `M ` vs `M`)."""
    out = df.copy()
    for col in out.columns:
        if col.startswith("info_"):
            out[col] = out[col].astype(str).str.strip().replace({"nan": ""})
    if "info_sex" in out.columns:
        out["info_sex"] = out["info_sex"].replace({"m": "M", "f": "F"})
    return out


@st.cache_data(show_spinner=False)
def load_corpus(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].astype(str).replace({"nan": ""})
    df = _normalize_metadata_columns(df)
    # App scope: orthographic transcript tier only (human-readable words; not ARPAbet alignment tiers).
    df = df[df["textgrid_role"] == "orthographic"].copy()
    df["_word_count"] = df["text"].apply(
        lambda x: len(re.findall(r"\b[\w']+\b", str(x).lower()))
        if x and not str(x).startswith("[ERROR:")
        else 0
    )
    df["_char_count"] = df["text"].str.len()
    df["_is_error"] = df["text"].str.startswith("[ERROR:", na=False)
    df["_eda_category"] = df["task"].map(TASK_TO_EDA).fillna("unknown")
    df["_task_title"] = df["task"].map(
        lambda t: TASK_SPECS.get(t, {}).get("label", str(t).replace("_", " ").title())
    )
    return df


@st.cache_data(show_spinner=False)
def attach_filler_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expensive: run once on filtered slice."""
    out = df.copy()
    totals: List[int] = []
    cols: Dict[str, List[int]] = {n: [] for n in ALL_FILLER_NAMES}
    for txt in out["text"]:
        tot, d = _count_fillers_one(txt)
        totals.append(tot)
        for n in ALL_FILLER_NAMES:
            cols[n].append(d[n])
    out["_filler_total"] = totals
    for n in ALL_FILLER_NAMES:
        out[f"_f_{n}"] = cols[n]
    out["_filler_per100"] = out.apply(
        lambda r: (100.0 * r["_filler_total"] / r["_word_count"])
        if r["_word_count"] > 0
        else 0.0,
        axis=1,
    )
    return out


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = df[df["task"].isin(cfg["tasks"])]
    f = f[f["session"].isin(cfg["sessions"])]
    if cfg.get("sex"):
        f = f[f["info_sex"].isin(cfg["sex"])]
    if cfg.get("l1"):
        f = f[f["info_l1_english"].isin(cfg["l1"])]
    if cfg.get("speakers"):
        f = f[f["speaker_id"].isin(cfg["speakers"])]
    if cfg.get("hide_errors", True):
        f = f[~f["_is_error"]]
    q = cfg.get("search", "").strip()
    if q:
        f = f[f["text"].str.contains(re.escape(q), case=False, na=False)]
    return f


def task_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = (
        df.groupby("task", dropna=False)
        .agg(
            n=("file_name", "count"),
            mean_words=("_word_count", "mean"),
            mean_chars=("_char_count", "mean"),
        )
        .reset_index()
    )
    s["situation"] = s["task"].map(
        lambda t: EDA_CATEGORY_LABEL.get(TASK_TO_EDA.get(t, ""), "")
    )
    s["typical_session"] = s["task"].map(
        lambda t: TASK_SPECS.get(t, {}).get("typical_sessions", "—")
    )
    s["description"] = s["task"].map(
        lambda t: TASK_SPECS.get(t, {}).get("summary", "—")
    )
    _ord = {k: i for i, k in enumerate(EDA_CATEGORY_ORDER)}
    s["_o"] = s["task"].map(lambda t: _ord.get(TASK_TO_EDA.get(t, ""), 99))
    s = s.sort_values(["_o", "task"]).drop(columns=["_o"])
    return s.round(1)


def render_about_tab() -> None:
    st.markdown("### Dataset")
    st.markdown(
        "**UCLA Speaker Variability** — ~202 speakers, visits **A–C** (sometimes **D**), multiple tasks per visit "
        "(reading, monologue, phone, video, etc.). **Fillers** tab: word-level counts in orthographic transcripts. "
        "CSV from `parse_ucla_box_text_state.py` + optional `public_database_speaker_info.xlsx`."
    )

    st.markdown("### Column reference (CSV)")

    field_docs = [
        ("**speaker_id**", "Speaker number (digits only, no leading zeros—matches the spreadsheet `speakerID`)."),
        ("**folder_id** / **file_id**", "Internal Box folder and file identifiers from scraping (for traceability)."),
        ("**file_name**", "Original TextGrid name, e.g. `153C_video.TextGrid` or `153B_sentences_FAVE.TextGrid`."),
        ("**extension**", "Always `textgrid` here."),
        ("**session**", "Recording session letter: **A**, **B**, **C**, or sometimes **D** (extra session in metadata)."),
        ("**session_key**", "Shorthand `speaker_id` + session, e.g. `153C`."),
        ("**task**", (
            "Elicitation condition (`instructions`, `neutral`, `happy`, `phonecall`, `annoyed`, `video`, "
            "`sentences`, `vowels`). See **Speech tasks** below for full prompts."
        )),
        ("**variant**", "Empty for the main file; **FAVE** or **darla** if the file is a forced-alignment output."),
        ("**textgrid_role**", (
            "**orthographic** — human-readable transcript tier (what people said), best for **words & fillers**. "
            "**aligned_fave** / **aligned_darla** — FAVE or DARLA output: **ARPAbet phones**, `sp` for silence, "
            "plus aligned words—not suitable for simple word searches like “um”."
        )),
        ("**text**", "Concatenated `text = \"...\"` lines from the TextGrid."),
        ("**info_sex**", "From public metadata: **M** / **F**."),
        ("**info_age**", "Age at recording."),
        ("**info_l1_english**", "Whether L1 is English (**Y** / **N** / other codes from the spreadsheet)."),
        ("**info_l1_other**", "Other L1 label when applicable."),
        ("**info_l2_english_l1** / **info_l2_english_aoa**", "Second-language / age of acquisition fields when present."),
        ("**info_db_session** / **info_db_clipping**", (
            "Which session this task was logged in for QA (**A–D**), and clipping quality (**OK**, "
            "**pos_min_clip**, etc.)—from the Excel documentation."
        )),
    ]
    for title, body in field_docs:
        st.markdown(f"{title}")
        st.caption(body)

    st.markdown("### Tasks × sessions (design)")
    st.markdown(
        """
| Session | Typical tasks |
|--------|----------------|
| **A** | instructions, neutral, sentences, vowels |
| **B** | happy, phonecall, sentences, vowels |
| **C** | annoyed, video, sentences, vowels |

**Sentences** and **vowels** occur in every session; other tasks are scheduled once per design (**redos** may appear as session **D** in metadata).
        """
    )

    st.markdown("### Speech tasks (what participants heard)")
    st.caption("Quoted prompts are from the corpus readme (on-screen instructions during recording).")

    task_order = [
        "sentences",
        "vowels",
        "instructions",
        "neutral",
        "happy",
        "phonecall",
        "annoyed",
        "video",
    ]
    for key in task_order:
        spec = TASK_SPECS.get(key)
        if not spec:
            continue
        with st.expander(f"**{spec['label']}** — `{key}`"):
            st.markdown(f"**Designed for session(s):** {spec['typical_sessions']}")
            st.markdown(spec["summary"])
            st.markdown(f"*Instructions shown to speakers:* {spec['prompt']}")


def _mean_filler_per_file(g: pd.DataFrame) -> float:
    if len(g) == 0:
        return 0.0
    return float(g["_filler_per100"].mean())


def _filler_histogram_series(fw: pd.DataFrame) -> pd.Series:
    """Bin per-file filler rates for a simple frequency chart."""
    bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 25, 1e9]
    labels = ["0–1", "1–2", "2–3", "3–4", "4–5", "5–7", "7–10", "10–15", "15–25", "25+"]
    s = pd.cut(fw["_filler_per100"], bins=bins, labels=labels, right=False, include_lowest=True)
    return s.astype(str).value_counts().reindex(labels).fillna(0)


def _filler_by_word_count_deciles(fw: pd.DataFrame) -> pd.DataFrame | None:
    """Mean filler rate by transcript length quantile (confounding check)."""
    n = len(fw)
    if n < 15:
        return None
    q = min(10, max(4, n // 30))
    try:
        g = fw.copy()
        g["_bin"] = pd.qcut(g["_word_count"], q=q, duplicates="drop")
    except (ValueError, TypeError):
        return None
    out = (
        g.groupby("_bin", observed=True)
        .agg(
            mean_hits_per100=("_filler_per100", "mean"),
            files=("speaker_id", "count"),
            med_words=("_word_count", "median"),
        )
        .reset_index()
    )
    out = out.sort_values("med_words").reset_index(drop=True)
    out["length_bin"] = [f"Q{i}" for i in range(1, len(out) + 1)]
    return out[["length_bin", "mean_hits_per100", "files"]]


def _filler_breakdown_categorical(fw: pd.DataFrame, col: str) -> pd.DataFrame | None:
    """Mean filler rate per transcript file, grouped by a metadata column (non-empty rows only)."""
    if col not in fw.columns:
        return None
    x = fw.copy()
    x["_m"] = x[col].astype(str).str.strip()
    x = x[x["_m"].ne("") & ~x["_m"].str.lower().eq("nan")]
    if len(x) == 0:
        return None
    rows = []
    for val in sorted(x["_m"].unique(), key=str):
        g = x[x["_m"] == val]
        rows.append(
            {
                "category": val,
                "files": len(g),
                "mean_per_file_per100": _mean_filler_per_file(g),
            }
        )
    out = pd.DataFrame(rows)
    if col == "info_sex":
        _order = {"F": 0, "M": 1}
        out["_o"] = out["category"].map(lambda c: _order.get(str(c).strip(), 99))
        out = out.sort_values("_o").drop(columns=["_o"])
    elif col == "info_db_session":
        out = out.sort_values("category")
    else:
        out = out.sort_values("category")
    return out


def _filler_breakdown_age(fw: pd.DataFrame) -> pd.DataFrame | None:
    """Age bands with enough spread; uses numeric info_age only."""
    if "info_age" not in fw.columns:
        return None
    x = fw.copy()
    x["_age"] = pd.to_numeric(x["info_age"], errors="coerce")
    x = x.loc[x["_age"].notna() & (x["_age"] > 0)]
    if len(x) < 15:
        return None
    q = min(5, max(3, len(x) // 50))
    try:
        x["_band"] = pd.qcut(x["_age"], q=q, duplicates="drop")
    except (ValueError, TypeError):
        try:
            x["_band"] = pd.cut(x["_age"], bins=min(5, len(x["_age"].unique())))
        except ValueError:
            return None
    rows = []
    for band, g in x.groupby("_band", observed=True):
        rows.append(
            (
                float(g["_age"].min()),
                {
                    "age_band": str(band),
                    "files": len(g),
                    "mean_per_file_per100": _mean_filler_per_file(g),
                },
            )
        )
    rows.sort(key=lambda t: t[0])
    return pd.DataFrame([r[1] for r in rows])


def _filler_mean_rate_chart(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """Single series: mean filler matches per 100 words, averaging transcript-level rates within each group."""
    return df.set_index(index_col)[["mean_per_file_per100"]].rename(
        columns={"mean_per_file_per100": "Mean matches / 100 words"}
    )


def render_filler_tab(f_base: pd.DataFrame) -> None:
    st.markdown("### Fillers")
    with st.expander("What the numbers mean", expanded=False):
        st.markdown(
            """
For **each transcript file** we compute:

**rate (file)** = (all filler pattern matches in that transcript) ÷ (word tokens in that transcript) × 100  

So it is **matches per 100 words** for that recording.

**Situation, task, metadata, age** — bar height is the **mean of those file rates** in the group (every transcript counts equally). Long and short files contribute one value each.

**Patterns** — each bar is still “how often this pattern appears per 100 words,” but counted **across all words in your current filter** (so longer transcripts contribute more words). That answers “what dominates the corpus,” not “typical file.”

**Top summary row “Overall”** — mean file rate (same idea as the group charts).

*like* / *well* / *so* use plain word matching and also hit grammatical uses.

**Sidebar filters** apply to everything on this tab.
            """
        )
    st.caption("Open **What the numbers mean** for definitions. Group charts = **mean of per-file rates** (except *Patterns*).")

    c1, c2 = st.columns(2)
    with c1:
        excl_vow = st.checkbox(
            "Drop vowel task from situation chart",
            value=True,
            key="filler_excl_vowels",
            help="Vowel clips are [a] holds—not dialogue.",
        )
    with c2:
        min_words = st.number_input(
            "Min words per file",
            0,
            500,
            20,
            10,
            key="filler_min_words",
        )

    work = f_base[f_base["_word_count"] >= min_words].copy()
    if len(work) == 0:
        st.warning("No rows left—relax filters or lower min words.")
        return

    with st.spinner("Counting…"):
        fw = attach_filler_columns(work)

    fw_sit = fw[fw["task"] != "vowels"] if excl_vow else fw

    total_words = fw["_word_count"].sum()
    total_hits = fw["_filler_total"].sum()
    mean_file_rate = float(fw["_filler_per100"].mean()) if len(fw) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Files", f"{len(fw):,}")
    m2.metric("Words", f"{total_words:,}")
    m3.metric("Matches", f"{total_hits:,}")
    m4.metric("Mean rate (per file)", f"{mean_file_rate:.2f}")

    hits = {n: int(fw[f"_f_{n}"].sum()) for n in ALL_FILLER_NAMES}
    inv_df = pd.DataFrame(
        [
            {
                "Pattern": k,
                "n": v,
                "%": (100.0 * v / total_hits) if total_hits else 0.0,
                "/100w": (100.0 * v / total_words) if total_words else 0.0,
            }
            for k, v in hits.items()
        ]
    ).sort_values("/100w", ascending=False)

    st.subheader("Patterns")
    pc = inv_df.set_index("Pattern")[["/100w"]].rename(columns={"/100w": "per 100 words"})
    st.bar_chart(pc, height=280)
    st.dataframe(inv_df.round(3), use_container_width=True, hide_index=True, height=220)

    st.subheader("Situation")
    sit_rows = []
    for cat in EDA_CATEGORY_ORDER:
        g = fw_sit[fw_sit["_eda_category"] == cat]
        if len(g) == 0:
            continue
        sit_rows.append(
            {
                "situation": EDA_CATEGORY_LABEL.get(cat, cat),
                "files": len(g),
                "mean_per_file_per100": _mean_filler_per_file(g),
            }
        )
    if sit_rows:
        sdf = pd.DataFrame(sit_rows)
        st.bar_chart(_filler_mean_rate_chart(sdf, "situation"), height=280)
        st.dataframe(
            sdf.rename(
                columns={
                    "situation": "Situation",
                    "files": "n",
                    "mean_per_file_per100": "mean / 100 w",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
            height=200,
        )
    else:
        st.warning("Nothing to show for situation (check filters).")

    st.subheader("Task")
    task_weighted = []
    for task, g in fw.groupby("task"):
        task_weighted.append(
            {
                "task": task,
                "situation": EDA_CATEGORY_LABEL.get(TASK_TO_EDA.get(task, ""), ""),
                "files": len(g),
                "mean_per_file_per100": float(g["_filler_per100"].mean()),
            }
        )
    _ord = {k: i for i, k in enumerate(EDA_CATEGORY_ORDER)}
    tw = pd.DataFrame(task_weighted)
    tw["_o"] = tw["task"].map(lambda t: _ord.get(TASK_TO_EDA.get(t, ""), 99))
    tw = tw.sort_values(["_o", "task"]).drop(columns=["_o"])
    st.bar_chart(_filler_mean_rate_chart(tw, "task"), height=300)
    st.dataframe(
        tw.rename(
            columns={
                "files": "n",
                "mean_per_file_per100": "mean / 100 w",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
        height=220,
    )

    c_hist, c_dec = st.columns(2)
    with c_hist:
        st.subheader("Rate distribution")
        hist = _filler_histogram_series(fw)
        st.bar_chart(pd.DataFrame({"files": hist}), height=220)
    with c_dec:
        st.subheader("Rate vs length")
        dec = _filler_by_word_count_deciles(fw)
        if dec is not None and len(dec):
            dc = dec.set_index("length_bin")[["mean_hits_per100"]].rename(
                columns={"mean_hits_per100": "mean /100w"}
            )
            st.bar_chart(dc, height=220)
        else:
            st.caption("—")

    st.subheader("Metadata")

    META_PAIRS = [
        ("info_sex", "Sex"),
        ("info_l1_english", "L1 English"),
        ("info_db_session", "DB session"),
        ("info_db_clipping", "Clip QA"),
    ]
    for row_i in range(0, len(META_PAIRS), 2):
        slice_pairs = META_PAIRS[row_i : row_i + 2]
        cols = st.columns(2)
        for ci in range(2):
            with cols[ci]:
                if ci >= len(slice_pairs):
                    continue
                col, title = slice_pairs[ci]
                st.markdown(f"**{title}**")
                bd = _filler_breakdown_categorical(fw, col)
                if bd is not None and len(bd) >= 1:
                    st.bar_chart(_filler_mean_rate_chart(bd, "category"), height=200)
                    st.dataframe(
                        bd.rename(
                            columns={
                                "category": "group",
                                "files": "n",
                                "mean_per_file_per100": "mean / 100 w",
                            }
                        ).round(2),
                        use_container_width=True,
                        hide_index=True,
                        height=140,
                    )
                else:
                    st.caption("—")

    st.subheader("Age")
    age_df = _filler_breakdown_age(fw)
    if age_df is not None and len(age_df):
        st.bar_chart(_filler_mean_rate_chart(age_df, "age_band"), height=220)
        st.dataframe(
            age_df.rename(
                columns={
                    "age_band": "age",
                    "files": "n",
                    "mean_per_file_per100": "mean / 100 w",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
            height=160,
        )
    else:
        st.caption("—")


def main() -> None:
    st.set_page_config(
        page_title="UCLA Speaker Variability",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
    div[data-testid="stVerticalBlock"] > div:first-child h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
    .muted { color: #666; font-size: 0.9rem; }
    hr { margin: 1rem 0; border: none; border-top: 1px solid #33333322; }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.title("UCLA Speaker Variability")
    st.markdown('<p class="muted">Transcripts · filler-word counts</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### CSV")
        csv_path = st.text_input("CSV path", value=str(DEFAULT_CSV), label_visibility="collapsed")
        path = Path(csv_path).expanduser()
        if not path.is_file():
            st.error("File not found.")
            st.stop()

        try:
            df = load_corpus(str(path))
        except Exception as e:
            st.exception(e)
            st.stop()

        st.caption(f"{len(df):,} rows · {df['speaker_id'].nunique()} speakers")

        st.markdown("---")
        st.markdown("### Filter")
        q = st.text_input("Text contains", placeholder="optional…")

        tasks = sorted(df["task"].unique().tolist())
        task_pick = st.multiselect("Task", tasks, default=tasks)

        sessions = sorted(df["session"].unique().tolist())
        session_pick = st.multiselect("Session", sessions, default=sessions)

        sex_opts_raw = [x for x in df["info_sex"].unique().tolist() if str(x).strip()]
        _sex_order = {"F": 0, "M": 1}
        sex_opts = sorted(sex_opts_raw, key=lambda x: (_sex_order.get(str(x).strip().upper(), 50), x))
        sex_pick = st.multiselect("Sex", sex_opts)

        l1_opts = sorted(x for x in df["info_l1_english"].unique().tolist() if x)
        l1_pick = st.multiselect("L1 English", l1_opts)

        hide_errors = st.checkbox("Hide parse errors", value=True)

        speakers = sorted(df["speaker_id"].unique().tolist(), key=lambda x: int(x))
        speaker_pick = st.multiselect("Speakers", speakers, format_func=lambda x: str(x))

    cfg = {
        "tasks": task_pick,
        "sessions": session_pick,
        "sex": sex_pick,
        "l1": l1_pick,
        "speakers": speaker_pick,
        "hide_errors": hide_errors,
        "search": q,
    }

    f = apply_filters(df, cfg)

    tab_about, tab_overview, tab_browse, tab_summary, tab_filler = st.tabs(
        ["About", "Overview", "Browse", "Tasks", "Fillers"]
    )

    with tab_about:
        render_about_tab()

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files", f"{len(f):,}")
        c2.metric("Speakers", f"{f['speaker_id'].nunique()}")
        c3.metric("Mean words", f"{f['_word_count'].mean():.1f}")
        c4.metric("Mean chars", f"{f['_char_count'].mean():.0f}")

        st.markdown("##### Session × task (file counts)")
        ct = (
            f.groupby(["session", "task"])
            .size()
            .reset_index(name="n")
            .pivot(index="task", columns="session", values="n")
            .fillna(0)
            .astype(int)
        )
        st.dataframe(ct, use_container_width=True)

    with tab_browse:
        sort_by = st.radio(
            "Sort",
            ["speaker · file", "words ↓", "words ↑"],
            horizontal=True,
        )
        browse = f.copy()
        if sort_by == "words ↓":
            browse = browse.sort_values("_word_count", ascending=False)
        elif sort_by == "words ↑":
            browse = browse.sort_values("_word_count", ascending=True)
        else:
            browse = browse.sort_values(["speaker_id", "session", "file_name"])

        show_n = st.slider("Show rows", 5, 80, 15)
        display_cols = ["speaker_id", "session", "task", "file_name", "_word_count"]
        pretty = browse[display_cols].head(show_n).rename(
            columns={
                "speaker_id": "Speaker",
                "session": "Visit",
                "task": "Task",
                "file_name": "File",
                "_word_count": "Words",
            }
        )
        st.dataframe(pretty, use_container_width=True, hide_index=True)

        browse = browse.reset_index(drop=True)
        if len(browse):
            labels = (
                browse["speaker_id"].astype(str)
                + " · "
                + browse["session"].astype(str)
                + " · "
                + browse["task"].astype(str)
                + " — "
                + browse["file_name"].astype(str)
            ).tolist()
            pick_i = st.selectbox(
                "File",
                range(len(browse)),
                format_func=lambda i: labels[i],
            )
            row = browse.iloc[pick_i]
            meta_cols = [
                "speaker_id",
                "session",
                "session_key",
                "task",
                "variant",
                "textgrid_role",
                "file_name",
                "info_sex",
                "info_age",
                "info_l1_english",
                "info_l1_other",
                "info_db_session",
                "info_db_clipping",
            ]
            with st.expander("Metadata", expanded=False):
                st.json({c: row[c] for c in meta_cols if c in row.index and row[c]})

            tk = str(row["task"])
            spec = TASK_SPECS.get(tk)
            if spec:
                with st.expander(f"Task description (`{tk}`)", expanded=False):
                    st.markdown(f"**Typical session(s):** {spec['typical_sessions']}")
                    st.markdown(spec["summary"])
                    st.markdown(f"*On-screen prompt:* {spec['prompt']}")

            st.text_area("Transcript", value=str(row["text"]), height=280)

    with tab_summary:
        ts = task_summary(f).rename(
            columns={
                "n": "files",
                "mean_words": "mean words",
                "mean_chars": "mean chars",
                "typical_session": "usual visit",
            }
        )
        st.dataframe(ts, use_container_width=True, hide_index=True)

    with tab_filler:
        render_filler_tab(f)


if __name__ == "__main__":
    main()
