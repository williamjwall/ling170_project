"""
UCLA Speaker Variability — interactive text explorer (Streamlit).

Run from repo root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
_EDA_DIR = REPO_ROOT / "EDA"
if str(_EDA_DIR) not in sys.path:
    sys.path.insert(0, str(_EDA_DIR))

from simple_stats import compare_metrics_in_dataframe
from stats_display import (
    build_metric_means,
    pvalue_help_expander,
    render_metrics_results_table,
    render_multi_group_block,
    render_two_group_block,
)
from research_findings_page import render_research_findings_from_raw
from discussion import (
    NEXT_EMOTION,
    NEXT_GENDER,
    NEXT_SITUATION,
    PRIOR_EMOTION,
    PRIOR_GENDER,
    PRIOR_SITUATION,
    _means,
    render_plain_english,
)
# Fixed corpus path (orthographic rows + hybrid columns). Regenerate offline only.
CORPUS_CSV = REPO_ROOT / "ucla_box_parsed" / "ucla_text_state_parsed_with_hybrid.csv"
# Back-compat name for scripts / pages that import DEFAULT_CSV.
DEFAULT_CSV = CORPUS_CSV

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

# Coarser buckets for EDA summaries (research page & insights).
FILLER_GROUPS: List[Tuple[str, List[str]]] = [
    ("Hesitation (um, uh, …)", ["um", "uh", "uhm", "erm", "er", "hmm"]),
    ("Phrases (you know, I mean, …)", ["you know", "i mean", "sort of", "kind of"]),
    ("Like · well · so · actually…", ["like", "well", "so", "actually", "basically", "literally"]),
]


def _pooled_pat_per100(df: pd.DataFrame, patterns: List[str]) -> float:
    """Combined rate for named patterns per 100 words in slice df."""
    words = int(df["_word_count"].sum())
    if words <= 0:
        return 0.0
    hits = sum(int(df[f"_f_{p}"].sum()) for p in patterns if f"_f_{p}" in df.columns)
    return 100.0 * hits / words


def render_filler_insights_emotion(
    fw: pd.DataFrame,
    *,
    emotion_order: Tuple[str, ...] = ("neutral", "happy", "annoyed"),
    short_labels: Dict[str, str] | None = None,
) -> None:
    """Patterns, families, and contrasts across emotion tasks (fw must include _f_* columns)."""
    short_labels = short_labels or {"neutral": "Neutral", "happy": "Happy", "annoyed": "Annoyed"}

    tasks_present = [t for t in emotion_order if fw["task"].eq(t).any()]
    if len(tasks_present) < 1:
        return

    with st.expander("Deeper insights · emotion ↔ fillers", expanded=True):
        st.caption(
            "Here **/100 w** values are **pooled** (all hits ÷ all words in that task × 100), not mean-per-file. "
            "**Mix %** = that pattern’s share of *all pattern hits summed* in that task. "
            "*like/well/so* also match grammatical uses unless hybrid counts are on in Research."
        )

        st.markdown("##### Filler families by emotion")
        fam_rows = []
        for title, plist in FILLER_GROUPS:
            row: Dict[str, object] = {"Family": title}
            for t in tasks_present:
                g = fw.loc[fw["task"].eq(t)]
                row[short_labels[t]] = round(_pooled_pat_per100(g, plist), 3)
            fam_rows.append(row)
        fam_df = pd.DataFrame(fam_rows)
        st.bar_chart(fam_df.set_index("Family"), height=min(260, 60 + 40 * len(fam_df)))
        st.dataframe(fam_df, use_container_width=True, hide_index=True)

        st.markdown("##### Top patterns per emotion (pooled /100 w)")
        rk = 5
        top_cols = st.columns(len(tasks_present))
        rates: Dict[str, Dict[str, float]] = {}
        for t in tasks_present:
            g = fw.loc[fw["task"].eq(t)]
            rates[t] = {p: _pooled_pat_per100(g, [p]) for p in ALL_FILLER_NAMES}
        for i, t in enumerate(tasks_present):
            with top_cols[i]:
                ranked = sorted(rates[t].items(), key=lambda x: -x[1])[:rk]
                tbl = pd.DataFrame(
                    [{"#": j + 1, "pattern": p, "/100 w": round(rv, 3)} for j, (p, rv) in enumerate(ranked)]
                )
                st.markdown(f"**{short_labels[t]}**")
                st.dataframe(tbl, use_container_width=True, hide_index=True, height=220)

        st.markdown("##### Strongest emotion contrasts")
        st.caption("**Range** = max − min pooled /100 w across emotions (larger ⇒ more unequal across tasks).")
        contrast = []
        for p in ALL_FILLER_NAMES:
            vals = [rates[t][p] for t in tasks_present]
            if not vals:
                continue
            rmin, rmax = min(vals), max(vals)
            row = {"Pattern": p, "range": round(rmax - rmin, 4)}
            for i, t in enumerate(tasks_present):
                row[short_labels[t]] = round(rates[t][p], 3)
            contrast.append(row)
        con_df = pd.DataFrame(contrast).sort_values("range", ascending=False)
        st.dataframe(con_df.round(4), use_container_width=True, hide_index=True, height=min(360, 52 + 22 * min(14, len(con_df))))

        if "neutral" in tasks_present:
            st.markdown("##### Relative change vs neutral (ratio)")
            st.caption("Ratio = pooled /100 w for that emotion ÷ neutral. Above 1 means more dense than neutral.")
            ratio_rows = []
            for p in ALL_FILLER_NAMES:
                base = rates["neutral"].get(p, 0.0)
                if base < 1e-6:
                    continue
                rrow: Dict[str, object] = {"Pattern": p}
                for t in tasks_present:
                    if t == "neutral":
                        continue
                    rrow[f"{short_labels[t]} / Neutral"] = round(rates[t][p] / base, 3)
                ratio_rows.append(rrow)
            if ratio_rows:
                rdf = pd.DataFrame(ratio_rows)
                st.dataframe(rdf, use_container_width=True, hide_index=True, height=min(380, 48 + 22 * min(14, len(rdf))))

        st.markdown("##### Mix · what fraction of fillers is each pattern?")
        mix_rows = []
        for p in ALL_FILLER_NAMES:
            mr: Dict[str, object] = {"Pattern": p}
            for t in tasks_present:
                g = fw.loc[fw["task"].eq(t)]
                fh = int(g["_filler_total"].sum())
                hp = int(g[f"_f_{p}"].sum()) if f"_f_{p}" in g.columns else 0
                mr[f"mix_{short_labels[t]} %"] = round(100.0 * hp / fh, 1) if fh else 0.0
            mix_rows.append(mr)
        mdf = pd.DataFrame(mix_rows).sort_values(
            f"mix_{short_labels[tasks_present[0]]} %",
            ascending=False,
        )
        st.dataframe(mdf.round(2), use_container_width=True, hide_index=True, height=min(380, 48 + 20 * min(14, len(mdf))))


def render_filler_insights_phone_fm(fw: pd.DataFrame) -> None:
    """F vs M: families, gaps, dominant words, mixture — fw must include filler columns."""
    if "info_sex" not in fw.columns:
        return
    xm = fw.loc[_sex_fm_mask(fw["info_sex"])].copy()
    xm["_sx"] = xm["info_sex"].astype(str).str.strip().str.upper()
    if xm["_sx"].isin({"F", "M"}).sum() == 0:
        return

    with st.expander("Deeper insights · female vs male (phone)", expanded=True):
        st.caption(
            "**Δ (F − M)** = pooled /100 w female minus male. "
            "Positive ⇒ higher female rate by this measure. Per-file p-values are on the **Fillers** tab."
        )

        st.markdown("##### Filler families")
        grp_rows = []
        for title, plist in FILLER_GROUPS:
            row: Dict[str, object] = {"Family": title}
            for code, lbl in (("F", "Female"), ("M", "Male")):
                g = xm.loc[xm["_sx"].eq(code)]
                row[f"{lbl} /100 w"] = round(_pooled_pat_per100(g, plist), 3)
            row["Δ (F − M)"] = round(float(row["Female /100 w"]) - float(row["Male /100 w"]), 4)
            grp_rows.append(row)
        gdf = pd.DataFrame(grp_rows)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(
                gdf.set_index("Family")[["Female /100 w", "Male /100 w"]].rename(
                    columns={"Female /100 w": "Female", "Male /100 w": "Male"}
                ),
                height=220,
            )
        with c2:
            st.dataframe(gdf, use_container_width=True, hide_index=True)

        fm_pat: Dict[str, Tuple[float, float]] = {}
        for p in ALL_FILLER_NAMES:
            rf = rm = 0.0
            gf = xm.loc[xm["_sx"].eq("F")]
            gm = xm.loc[xm["_sx"].eq("M")]
            if len(gf):
                rf = _pooled_pat_per100(gf, [p])
            if len(gm):
                rm = _pooled_pat_per100(gm, [p])
            fm_pat[p] = (rf, rm)

        st.markdown("##### Largest female–male gaps (by pattern)")
        gap_rows = []
        for p, (rf, rm) in fm_pat.items():
            gap_rows.append(
                {
                    "Pattern": p,
                    "Female /100 w": round(rf, 3),
                    "Male /100 w": round(rm, 3),
                    "Δ (F − M)": round(rf - rm, 4),
                }
            )
        gfp = pd.DataFrame(gap_rows)
        gfp["|Δ|"] = gfp["Δ (F − M)"].abs()
        gfp = gfp.sort_values("|Δ|", ascending=False).drop(columns=["|Δ|"])
        st.dataframe(gfp, use_container_width=True, hide_index=True, height=min(400, 48 + 22 * len(gfp)))

        st.markdown("##### Top patterns by sex (pooled /100 w)")
        ph_cols = st.columns(2)
        for ci, (lbl, code) in enumerate((("Female", "F"), ("Male", "M"))):
            gx = xm.loc[xm["_sx"].eq(code)]
            with ph_cols[ci]:
                st.markdown(f"**{lbl}**")
                if len(gx) == 0:
                    st.caption("—")
                    continue
                ranked = sorted(
                    [(p, _pooled_pat_per100(gx, [p])) for p in ALL_FILLER_NAMES],
                    key=lambda x: -x[1],
                )[:6]
                tdf = pd.DataFrame(
                    [{"#": j + 1, "pattern": p, "/100 w": round(rv, 3)} for j, (p, rv) in enumerate(ranked)]
                )
                st.dataframe(tdf, use_container_width=True, hide_index=True, height=240)

        st.markdown("##### Who dominates each sex’s fillers?")
        st.caption("**Share %** = that pattern’s hits ÷ summed filler hits for that sex (same summed-hit definition as totals).")
        share_rows = []
        for code, slab in (("F", "Female share %"), ("M", "Male share %")):
            g = xm.loc[xm["_sx"].eq(code)]
            tot = int(g["_filler_total"].sum())
            for p in ALL_FILLER_NAMES:
                nh = int(g[f"_f_{p}"].sum())
                pct = 100.0 * nh / tot if tot else 0.0
                share_rows.append({"Pattern": p, "_sex_col": slab, "share %": pct})
        sdf = pd.DataFrame(share_rows)
        piv = sdf.pivot(index="Pattern", columns="_sex_col", values="share %").reset_index().fillna(0.0)
        sort_c = "Female share %" if "Female share %" in piv.columns else piv.columns[-1]
        piv = piv.sort_values(sort_c, ascending=False)
        st.dataframe(piv.round(2), use_container_width=True, hide_index=True, height=min(420, 48 + 20 * len(piv)))


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
    # Optional offline spaCy columns (scripts/precompute_hybrid_columns.py) — coerce to int for math in the app.
    from hybrid_filler_spacy import HYBRID_NUMERIC_COLS

    for col in HYBRID_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
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
        "(reading, monologue, phone, video, etc.). The app loads **orthographic** transcript rows only and, by default, "
        "**`ucla_text_state_parsed_with_hybrid.csv`** (regex + offline parser columns for *like* / *well* / *so*). "
        "Rebuild it with `scripts/precompute_hybrid_columns.py` after changing parser rules (path is fixed in the app). "
        "Base export: `parse_ucla_box_text_state.py` + optional `public_database_speaker_info.xlsx`."
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


def _sex_fm_mask(series: pd.Series) -> pd.Series:
    """True where info_sex is F or M (normalized)."""
    sx = series.astype(str).str.strip().str.upper()
    return sx.isin({"F", "M"})


EMO_TASKS = ("neutral", "happy", "annoyed")


def _attach_pattern_rate_columns(fw: pd.DataFrame) -> pd.DataFrame:
    """Per-file rate (hits ÷ words × 100) for total and each filler pattern."""
    work = fw.copy()
    wsafe = work["_word_count"].replace(0, float("nan"))
    for name in ALL_FILLER_NAMES:
        hit = f"_f_{name}"
        if hit in work.columns:
            work[f"_rate_{name}"] = 100.0 * work[hit] / wsafe
    for title, plist in FILLER_GROUPS:
        slug = title.split("(")[0].strip().lower().replace(" ", "_")[:24]
        fam_cols = [f"_f_{p}" for p in plist if f"_f_{p}" in work.columns]
        if fam_cols:
            work[f"_rate_{slug}"] = 100.0 * work[fam_cols].sum(axis=1) / wsafe
    return work


def _filler_rate_metric_cols(fw: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    cols = ["_filler_per100"]
    labels: dict[str, str] = {"_filler_per100": "All fillers per 100 words"}
    for name in ALL_FILLER_NAMES:
        rc = f"_rate_{name}"
        if rc in fw.columns:
            cols.append(rc)
            labels[rc] = name
    for title, _plist in FILLER_GROUPS:
        slug = title.split("(")[0].strip().lower().replace(" ", "_")[:24]
        rc = f"_rate_{slug}"
        if rc in fw.columns:
            cols.append(rc)
            labels[rc] = title
    return cols, labels


def render_filler_pvalues_sex(fw: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    """Mann-Whitney U: female vs male on per-file filler rates."""
    sub = fw.loc[_sex_fm_mask(fw["info_sex"]) & (fw["_word_count"] > 0)].copy()
    sub["_sx"] = sub["info_sex"].astype(str).str.strip().str.upper()
    if len(sub.loc[sub["_sx"] == "F"]) < 2 or len(sub.loc[sub["_sx"] == "M"]) < 2:
        st.caption("Not enough F/M recordings for p-values.")
        return
    work = _attach_pattern_rate_columns(sub)
    metric_cols, metric_labels = _filler_rate_metric_cols(work)
    st.markdown("##### Female vs male · p-values")
    pvalue_help_expander(key=f"{widget_key_prefix}p_sex")
    headline = render_two_group_block(
        work.assign(info_sex=work["_sx"]),
        "info_sex",
        "_filler_per100",
        label_a="Female",
        label_b="Male",
        code_a="F",
        code_b="M",
        metric_name="All fillers per 100 words",
        title="Headline: all fillers",
    )
    table_cols = [c for c in metric_cols if c != "_filler_per100"]
    results = compare_metrics_in_dataframe(
        work,
        "_sx",
        table_cols,
        group_order=["F", "M"],
        metric_labels=metric_labels,
        group_labels={"F": "Female", "M": "Male"},
    )
    sex_labels = {"F": "Female", "M": "Male"}
    render_metrics_results_table(
        results,
        caption="Green rows = p below 0.05 (worth mentioning).",
        headline=headline,
        means=_means(work, "_sx", "_filler_per100", sex_labels),
        metric_means=build_metric_means(
            work,
            "_sx",
            metric_cols,
            group_labels=sex_labels,
            column_labels=metric_labels,
        ),
    )
    all_results = [headline] + list(results)
    render_plain_english(
        hypothesis="Do women and men use fillers differently in the recordings you filtered to?",
        headline=headline,
        all_results=all_results,
        means=_means(work, "_sx", "_filler_per100", sex_labels),
        prior_work=PRIOR_GENDER,
        next_steps=NEXT_GENDER,
        topic="this filtered slice (female vs male)",
        two_group=True,
        key=f"{widget_key_prefix}disc_sex",
    )


def render_filler_pvalues_emotion(fw: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    """Kruskal-Wallis across neutral / happy / annoyed when those tasks are in the slice."""
    present = [t for t in EMO_TASKS if fw["task"].eq(t).any()]
    if len(present) < 2:
        return
    sub = fw.loc[fw["task"].isin(present) & (fw["_word_count"] > 0)].copy()
    work = _attach_pattern_rate_columns(sub)
    metric_cols, metric_labels = _filler_rate_metric_cols(work)
    task_labels = {"neutral": "Neutral", "happy": "Happy", "annoyed": "Annoyed"}
    st.markdown("##### Emotion tasks · p-values")
    pvalue_help_expander(key=f"{widget_key_prefix}p_emo")
    headline = None
    if len(present) >= 3:
        headline = render_multi_group_block(
            work,
            "task",
            "_filler_per100",
            group_order=list(EMO_TASKS),
            group_labels=task_labels,
            metric_name="All fillers per 100 words",
            title="Headline: all fillers across story types",
        )
        table_cols = [c for c in metric_cols if c != "_filler_per100"]
        results = compare_metrics_in_dataframe(
            work,
            "task",
            table_cols,
            group_order=list(EMO_TASKS),
            metric_labels=metric_labels,
            group_labels=task_labels,
        )
    else:
        t0, t1 = present[0], present[1]
        headline = render_two_group_block(
            work,
            "task",
            "_filler_per100",
            label_a=task_labels.get(t0, t0),
            label_b=task_labels.get(t1, t1),
            code_a=t0,
            code_b=t1,
            metric_name="All fillers per 100 words",
        )
        results = compare_metrics_in_dataframe(
            work,
            "task",
            metric_cols,
            group_order=present,
            metric_labels=metric_labels,
            group_labels=task_labels,
        )
    emo_mean_labels = {t: task_labels[t] for t in present}
    render_metrics_results_table(
        results,
        caption="Green rows = p below 0.05.",
        headline=headline,
        means=_means(work, "task", "_filler_per100", emo_mean_labels),
        metric_means=build_metric_means(
            work,
            "task",
            metric_cols,
            group_labels=emo_mean_labels,
            column_labels=metric_labels,
        ),
        pairwise=headline.pairwise if headline else None,
    )
    all_results = ([headline] if headline else []) + list(results)
    render_plain_english(
        hypothesis="Do people fill more or less depending on whether the story is neutral, happy, or annoyed?",
        headline=headline,
        all_results=all_results,
        means=_means(work, "task", "_filler_per100", emo_mean_labels),
        prior_work=PRIOR_EMOTION,
        next_steps=NEXT_EMOTION,
        topic="neutral, happy, and annoyed retellings",
        two_group=False,
        key=f"{widget_key_prefix}disc_emo",
    )


def render_filler_pvalues_situation(fw: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    """Kruskal-Wallis across EDA situation categories (monologue, phone, etc.)."""
    if "_eda_category" not in fw.columns:
        return
    sub = fw.loc[fw["_word_count"] > 0].copy()
    cats = [c for c in EDA_CATEGORY_ORDER if (sub["_eda_category"] == c).any()]
    if len(cats) < 2:
        return
    work = _attach_pattern_rate_columns(sub.loc[sub["_eda_category"].isin(cats)])
    metric_cols, metric_labels = _filler_rate_metric_cols(work)
    sit_labels = {c: EDA_CATEGORY_LABEL.get(c, c) for c in cats}
    st.markdown("##### Situation · p-values")
    pvalue_help_expander(key=f"{widget_key_prefix}p_sit")
    headline = None
    if len(cats) >= 3:
        headline = render_multi_group_block(
            work,
            "_eda_category",
            "_filler_per100",
            group_order=cats,
            group_labels=sit_labels,
            metric_name="All fillers per 100 words",
            title="Headline: all fillers across situations",
        )
    else:
        headline = render_two_group_block(
            work,
            "_eda_category",
            "_filler_per100",
            label_a=sit_labels[cats[0]],
            label_b=sit_labels[cats[1]],
            code_a=cats[0],
            code_b=cats[1],
            metric_name="All fillers per 100 words",
        )
    table_cols = [c for c in metric_cols if c != "_filler_per100"]
    results = compare_metrics_in_dataframe(
        work,
        "_eda_category",
        table_cols,
        group_order=cats,
        metric_labels=metric_labels,
        group_labels=sit_labels,
    )
    sit_mean_labels = {c: sit_labels[c] for c in cats}
    render_metrics_results_table(
        results,
        caption="Green rows = p below 0.05.",
        headline=headline,
        means=_means(work, "_eda_category", "_filler_per100", sit_mean_labels),
        metric_means=build_metric_means(
            work,
            "_eda_category",
            metric_cols,
            group_labels=sit_mean_labels,
            column_labels=metric_labels,
        ),
        pairwise=headline.pairwise if headline else None,
    )
    all_results = ([headline] if headline else []) + list(results)
    render_plain_english(
        hypothesis="Do filler rates change depending on the speech task (reading, talking to the researcher, phone call, etc.)?",
        headline=headline,
        all_results=all_results,
        means=_means(work, "_eda_category", "_filler_per100", sit_mean_labels),
        prior_work=PRIOR_SITUATION,
        next_steps=NEXT_SITUATION,
        topic="speech situations (read-aloud, monologue, phone, etc.)",
        two_group=len(cats) == 2,
        key=f"{widget_key_prefix}disc_sit",
    )


def render_filler_emotion_by_task(fw: pd.DataFrame, *, emotion_order: Tuple[str, ...] = ("neutral", "happy", "annoyed")) -> None:
    """Compare fillers across emotion elicitation tasks (subset of corpus tasks)."""
    st.subheader("By emotion task")
    st.caption(
        "**Mean / file** = average of each transcript’s own (hits ÷ words × 100); **Pooled /100 w** = all hits in that task ÷ all words × 100. "
        "See the Research page expander *What are Pooled /100 w and Mean / file?* for a full explanation. "
        "Tasks are ordered neutral → happy → annoyed when present."
    )
    rows: List[Dict[str, object]] = []
    for t in emotion_order:
        g = fw.loc[fw["task"].eq(t)]
        if len(g) == 0:
            continue
        wsum = int(g["_word_count"].sum())
        hsum = int(g["_filler_total"].sum())
        lbl = TASK_SPECS.get(t, {}).get("label", str(t))
        rows.append(
            {
                "Task": lbl,
                "key": t,
                "Files": len(g),
                "Words": wsum,
                "Total hits": hsum,
                "Pooled /100 w": (100.0 * hsum / wsum) if wsum else 0.0,
                "Mean / file": float(g["_filler_per100"].mean()),
            }
        )
    if not rows:
        st.warning("No rows for the expected emotion tasks.")
        return
    summary = pd.DataFrame(rows)
    st.bar_chart(
        summary.set_index("Task")[["Mean / file"]].rename(columns={"Mean / file": "Mean filler matches / 100 words"}),
        height=260,
    )
    show = summary.drop(columns=["key"], errors="ignore")
    st.dataframe(show.round(3), use_container_width=True, hide_index=True, height=min(220, 40 + 28 * len(show)))

    short_task = {"neutral": "Neutral", "happy": "Happy", "annoyed": "Annoyed"}
    st.markdown("**Per pattern (pooled matches / 100 words within each emotion task)**")
    pat_rows: List[Dict[str, object]] = []
    for name in ALL_FILLER_NAMES:
        r: Dict[str, object] = {"Pattern": name}
        for t in emotion_order:
            g = fw.loc[fw["task"].eq(t)]
            if len(g) == 0:
                continue
            w = int(g["_word_count"].sum())
            hits = int(g[f"_f_{name}"].sum())
            r[short_task.get(t, t)] = (100.0 * hits / w) if w else 0.0
        pat_rows.append(r)
    pat_df = pd.DataFrame(pat_rows)
    drop_pat = {"Pattern"}
    num_cols = [c for c in pat_df.columns if c not in drop_pat]
    if num_cols:
        pat_df = pat_df.sort_values(num_cols[0], ascending=False)
    st.dataframe(pat_df.round(3), use_container_width=True, hide_index=True, height=min(420, 40 + 22 * len(pat_df)))


def render_filler_female_male(fw: pd.DataFrame, *, compact_caption: bool = False) -> None:
    """F vs M: total filler hits, word counts, pooled and per-file rates, per-pattern counts."""
    if "info_sex" not in fw.columns:
        st.caption("— No sex column for F vs M comparison.")
        return

    x = fw.copy()
    n_excl = int((~_sex_fm_mask(x["info_sex"])).sum())
    xm = x.loc[_sex_fm_mask(x["info_sex"])].copy()
    xm["_sx"] = xm["info_sex"].astype(str).str.strip().str.upper()

    if len(xm) == 0:
        st.warning("No rows with sex coded **F** or **M** under current filters.")
        return

    skip_note = (
        f" Not shown: **{n_excl:,}** transcript(s) with no **F** or **M** sex code (we only compare those two)."
        if n_excl
        else ""
    )
    if compact_caption:
        st.caption(
            "**Pooled /100 w:** Imagine one long transcript per sex: add all filler hits, add all words, then "
            "hits ÷ words × 100. **Mean / file:** For each transcript, hits ÷ its own words × 100, then average "
            "those numbers (short and long files each count once)."
            + skip_note
        )
    else:
        st.subheader("Female vs male")
        st.caption(
            "**Filler matches** = sum of hits from the same patterns as elsewhere on this tab. "
            "**Pooled /100 w:** all hits for that sex ÷ all words for that sex × 100 (weights longer transcripts more). "
            "**Mean / file:** each transcript’s hits ÷ words × 100, then averaged (weights each file equally)."
            + skip_note
        )

    rows = []
    for code, label in (("F", "Female"), ("M", "Male")):
        g = xm.loc[xm["_sx"] == code]
        if len(g) == 0:
            continue
        wsum = int(g["_word_count"].sum())
        hsum = int(g["_filler_total"].sum())
        rows.append(
            {
                "Sex": label,
                "Files": len(g),
                "Words": wsum,
                "Filler hits (total)": hsum,
                "Pooled matches / 100 w": (100.0 * hsum / wsum) if wsum else 0.0,
                "Mean rate / file": float(g["_filler_per100"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    c_left, c_right = st.columns((1, 1))
    with c_left:
        st.dataframe(
            summary.round(3),
            use_container_width=True,
            hide_index=True,
            height=min(120 + 28 * len(summary), 220),
        )
    with c_right:
        if len(summary) >= 1:
            ch = summary.set_index("Sex")[["Mean rate / file"]].rename(
                columns={"Mean rate / file": "Mean matches / 100 words (per file)"}
            )
            st.bar_chart(ch, height=220)

    pat_rows = []
    for name in ALL_FILLER_NAMES:
        r: Dict[str, object] = {"Pattern": name}
        for code, label in (("F", "Female"), ("M", "Male")):
            g = xm.loc[xm["_sx"] == code]
            w = int(g["_word_count"].sum())
            hits = int(g[f"_f_{name}"].sum())
            r[f"{label} hits"] = hits
            r[f"{label} /100w"] = (100.0 * hits / w) if w else 0.0
        pat_rows.append(r)
    pat_df = pd.DataFrame(pat_rows)
    sort_col = "Female /100w" if "Female /100w" in pat_df.columns else pat_df.columns[-1]
    pat_df = pat_df.sort_values(sort_col, ascending=False)
    st.markdown("**Per-pattern counts by sex** (pooled words within each sex)")
    st.dataframe(pat_df.round(3), use_container_width=True, hide_index=True, height=min(400, 36 + 24 * len(pat_df)))


def render_filler_tab(f_base: pd.DataFrame, *, widget_key_prefix: str = "") -> None:
    with st.expander("What the numbers mean", expanded=False):
        st.markdown(
            """
For **each transcript file** we compute:

**rate (file)** = (all filler pattern matches in that transcript) ÷ (word tokens in that transcript) × 100  

So it is **matches per 100 words** for that recording.

**Situation, task, metadata, age** — bar height is the **mean of those file rates** in the group (every transcript counts equally). Long and short files contribute one value each.

**Patterns** — each bar is still “how often this pattern appears per 100 words,” but counted **across all words in your current filter** (so longer transcripts contribute more words). That answers “what dominates the corpus,” not “typical file.”

**Top summary row “Overall”** — mean file rate (same idea as the group charts).

**Female vs male** — total **filler hits** and **word counts** by sex, plus pooled and per-file rates and a per-pattern table (F/M codes only; other/missing sex excluded from that block).

**p-values** — Mann-Whitney (F vs M) and Kruskal-Wallis (emotion tasks or situations) on **per-file** rates (fillers per 100 words). **p < 0.05** = worth mentioning; not proof of a real-world difference.

*like* / *well* / *so* use plain word matching and also hit grammatical uses.

**Sidebar filters** apply to everything on this tab.
            """
        )
    st.caption(
        "Open **What the numbers mean** for definitions. "
        "Group charts = **mean of per-file rates** (except *Patterns*)."
    )

    c1, c2 = st.columns(2)
    with c1:
        excl_vow = st.checkbox(
            "Drop vowel task from situation chart",
            value=True,
            key=f"{widget_key_prefix}filler_excl_vowels",
            help="Vowel clips are [a] holds—not dialogue.",
        )
    with c2:
        min_words = st.number_input(
            "Min words per file",
            0,
            500,
            20,
            10,
            key=f"{widget_key_prefix}filler_min_words",
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
    m3.metric("Filler matches", f"{total_hits:,}")
    m4.metric("Mean rate (per file)", f"{mean_file_rate:.2f}")

    render_filler_female_male(fw)

    with st.expander("Statistical checks (p-values)", expanded=True):
        render_filler_pvalues_sex(fw, widget_key_prefix=widget_key_prefix)
        if any(fw["task"].isin(EMO_TASKS)):
            render_filler_emotion_by_task(fw)
            render_filler_pvalues_emotion(fw, widget_key_prefix=widget_key_prefix)
        render_filler_pvalues_situation(fw_sit, widget_key_prefix=widget_key_prefix)

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
    st.markdown(
        '<p class="muted">Orthographic transcripts · filler patterns · fixed hybrid corpus</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Corpus")
        path = CORPUS_CSV
        st.caption(str(path.resolve()))
        if not path.is_file():
            st.error(
                f"Corpus not found at `{path}`. From the repo root run: "
                "`python scripts/precompute_hybrid_columns.py`"
            )
            st.stop()

        try:
            df = load_corpus(str(path))
        except Exception as e:
            st.exception(e)
            st.stop()

        st.caption(f"{len(df):,} orthographic transcripts · {df['speaker_id'].nunique():,} speakers")

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
        l1_pick = st.multiselect("English as L1", l1_opts, help="Values from `info_l1_english` in the spreadsheet (e.g. Y/N).")

        hide_errors = st.checkbox("Hide rows with parse errors", value=True)

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

    tab_about, tab_overview, tab_browse, tab_summary, tab_filler, tab_findings = st.tabs(
        ["About", "Overview", "Browse", "Task stats", "Fillers", "Research findings"]
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
            "Sort files by",
            ["Speaker · visit · file", "Words (most first)", "Words (fewest first)"],
            horizontal=True,
        )
        browse = f.copy()
        if sort_by == "Words (most first)":
            browse = browse.sort_values("_word_count", ascending=False)
        elif sort_by == "Words (fewest first)":
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
                "Transcript",
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

    with tab_findings:
        render_research_findings_from_raw(
            f,
            attach_fn=attach_filler_columns,
            age_min=18,
            age_max=24,
            min_words=20,
            cohort_note="Ages 18–24 · uses sidebar filters above",
        )


if __name__ == "__main__":
    main()
