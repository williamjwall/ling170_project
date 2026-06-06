"""
Emotion × gender chart layouts — sentiment progression and paired F/M comparisons.

Run: streamlit run streamlit_app.py → open this page from the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
_EDA_DIR = REPO_ROOT / "EDA"
if str(_EDA_DIR) not in sys.path:
    sys.path.insert(0, str(_EDA_DIR))

from filler_lexicon import CATEGORY_LABELS, enrich_dataframe, ordered_fillers

DEFAULT_CSV = REPO_ROOT / "ucla_box_parsed" / "emotion_phone_simplified.csv"
FALLBACK_CSV = _EDA_DIR / "emotion_phone_simplified.csv"

# Logical sentiment order (low → high arousal / positivity in the retelling task).
EMO_ORDER = ("annoyed", "neutral", "happy")
EMO_LABELS = {"annoyed": "Annoyed", "neutral": "Neutral", "happy": "Happy"}
COL_EMO = {"annoyed": "#ef4444", "neutral": "#94a3b8", "happy": "#fbbf24"}
COL_F = "#e85d8a"
COL_M = "#5b8def"


def _resolve_csv() -> Path:
    if DEFAULT_CSV.is_file():
        return DEFAULT_CSV
    if FALLBACK_CSV.is_file():
        return FALLBACK_CSV
    st.error(f"CSV not found at `{DEFAULT_CSV}` or `{FALLBACK_CSV}`.")
    st.stop()
    return DEFAULT_CSV


@st.cache_data
def load_emotion_cohort(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["info_sex"] = df["info_sex"].astype(str).str.strip()
    df = df[df["info_sex"].isin(("F", "M"))].copy()
    count_col = "text_filler" if "text_filler" in df.columns else "text"
    df = enrich_dataframe(df, text_col=count_col)
    df = df[df["task"].isin(EMO_ORDER)].copy()
    return df


def _mean_rate(df: pd.DataFrame, col: str) -> float:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(x.mean()) if len(x) else 0.0


def _display_filler(name: str) -> str:
    return "I mean" if name == "i mean" else name


def fig_sentiment_progression(emotion: pd.DataFrame, metric_col: str, metric_label: str) -> go.Figure:
    """Horizontal bars: annoyed → neutral → happy (matches sentiment arc and sample means)."""
    sub = emotion.loc[emotion["words"] > 0]
    rows = []
    for task in EMO_ORDER:
        g = sub.loc[sub["task"] == task]
        if len(g) == 0:
            continue
        rows.append(
            {
                "Story": EMO_LABELS[task],
                "task": task,
                "Rate": _mean_rate(g, metric_col),
                "n": len(g),
            }
        )
    if not rows:
        return go.Figure().update_layout(title="No data for emotion tasks")
    df = pd.DataFrame(rows)
    df["Story"] = pd.Categorical(
        df["Story"], categories=[EMO_LABELS[t] for t in EMO_ORDER], ordered=True
    )
    df = df.sort_values("Story")
    colors = [COL_EMO[t] for t in df["task"]]
    fig = go.Figure(
        go.Bar(
            x=df["Rate"],
            y=df["Story"],
            orientation="h",
            marker_color=colors,
            text=df["Rate"].round(2),
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.2f} per 100 words<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Sentiment progression: {metric_label} (annoyed → neutral → happy)",
        template="plotly_white",
        height=320,
        xaxis_title="Average per 100 words (mean of each recording)",
        yaxis_title="",
        margin=dict(l=10, r=40),
    )
    return fig


def fig_gender_paired_by_emotion(
    emotion: pd.DataFrame, metric_col: str, metric_label: str
) -> go.Figure:
    """
    Horizontal grouped bars: female and male side by side within each story type.
    Rows follow annoyed → neutral → happy so the primary comparison is F vs M per category.
    """
    sub = emotion.loc[emotion["words"] > 0].copy()
    rows = []
    for task in EMO_ORDER:
        for sex, gender in (("F", "Female"), ("M", "Male")):
            g = sub.loc[(sub["task"] == task) & (sub["info_sex"].str.upper() == sex)]
            if len(g) == 0:
                continue
            rows.append(
                {
                    "Story": EMO_LABELS[task],
                    "task": task,
                    "Gender": gender,
                    "Rate": _mean_rate(g, metric_col),
                    "n": len(g),
                }
            )
    if not rows:
        return go.Figure().update_layout(title="No data")
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        y="Story",
        x="Rate",
        color="Gender",
        barmode="group",
        orientation="h",
        color_discrete_map={"Female": COL_F, "Male": COL_M},
        category_orders={"Story": [EMO_LABELS[t] for t in EMO_ORDER]},
        title=f"Female vs male within each story: {metric_label}",
        labels={"Rate": "Average per 100 words", "Story": "Story type"},
        text="Rate",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=380,
        legend_title_text="Narrator",
        margin=dict(r=50),
    )
    return fig


def fig_gender_paired_by_filler(emotion: pd.DataFrame, filler: str) -> go.Figure | None:
    """One filler type: F and M bars adjacent within annoyed, neutral, happy."""
    col = f"rate {filler}"
    if col not in emotion.columns:
        return None
    return fig_gender_paired_by_emotion(emotion, col, _display_filler(filler))


def summary_sentiment_table(emotion: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    sub = emotion.loc[emotion["words"] > 0]
    rows = []
    for task in EMO_ORDER:
        g = sub.loc[sub["task"] == task]
        rows.append(
            {
                "Story": EMO_LABELS[task],
                "Recordings": len(g),
                "Mean / file": round(_mean_rate(g, metric_col), 3),
            }
        )
    return pd.DataFrame(rows)


def summary_gender_table(emotion: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    sub = emotion.loc[emotion["words"] > 0]
    rows = []
    for task in EMO_ORDER:
        for sex, gender in (("F", "Female"), ("M", "Male")):
            g = sub.loc[(sub["task"] == task) & (sub["info_sex"].str.upper() == sex)]
            rows.append(
                {
                    "Story": EMO_LABELS[task],
                    "Narrator": gender,
                    "Recordings": len(g),
                    "Mean / file": round(_mean_rate(g, metric_col), 3),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Emotion layouts", layout="wide")
    st.title("Emotion story layouts")
    st.caption(
        "Two chart organizations for the same emotion-retelling cohort "
        f"(`{DEFAULT_CSV.name}`). Rates = fillers per 100 words, averaged per recording."
    )

    csv_path = _resolve_csv()
    st.sidebar.markdown("### Data")
    st.sidebar.caption(str(csv_path))

    emotion = load_emotion_cohort(str(csv_path))
    min_words = st.sidebar.slider("Min words per recording", 0, 80, 0, 5)
    if min_words > 0:
        emotion = emotion.loc[emotion["words"] >= min_words].copy()

    metric_choice = st.sidebar.radio(
        "Metric",
        ["All fillers", "By filler type", "By category"],
        index=0,
    )

    if metric_choice == "All fillers":
        metric_col = "rate total"
        metric_label = "all fillers"
        filler_pick = None
        category_pick = None
    elif metric_choice == "By filler type":
        fillers = [w for w in ordered_fillers() if f"rate {w}" in emotion.columns]
        filler_pick = st.sidebar.selectbox(
            "Filler",
            fillers,
            format_func=_display_filler,
        )
        metric_col = f"rate {filler_pick}"
        metric_label = _display_filler(filler_pick)
        category_pick = None
    else:
        cat_opts = [c for c in CATEGORY_LABELS if f"rate {c}" in emotion.columns]
        category_pick = st.sidebar.selectbox("Category", cat_opts)
        metric_col = f"rate {category_pick}"
        metric_label = category_pick
        filler_pick = None

    n_emo = len(emotion.loc[emotion["words"] > 0])
    if n_emo == 0:
        st.warning("No emotion recordings match the filters.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Emotion recordings", f"{n_emo:,}")
    c2.metric("Speakers", f"{emotion['speaker_id'].nunique():,}")
    c3.metric("Tasks", "annoyed · neutral · happy")

    st.markdown("---")
    st.header("1 · Sentiment progression")
    st.markdown(
        "Bars run **annoyed → neutral → happy**. That order follows the emotional arc of the "
        "retelling task and matches the sample pattern: overall filler rate **rises** from annoyed "
        "through neutral to happy (more animated retellings tend to carry slightly more fillers)."
    )
    st.plotly_chart(
        fig_sentiment_progression(emotion, metric_col, metric_label),
        use_container_width=True,
    )
    st.dataframe(
        summary_sentiment_table(emotion, metric_col),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.header("2 · Female vs male within each story")
    st.markdown(
        "The primary comparison here is **narrator sex within each response category**. "
        "Female and male bars sit **next to each other** on every row (annoyed, then neutral, then happy). "
        "Horizontal layout keeps story labels readable; scan across each pair rather than jumping between sections."
    )
    st.plotly_chart(
        fig_gender_paired_by_emotion(emotion, metric_col, metric_label),
        use_container_width=True,
    )
    st.dataframe(
        summary_gender_table(emotion, metric_col),
        use_container_width=True,
        hide_index=True,
    )

    if metric_choice == "All fillers":
        with st.expander("Same layout for individual filler types (packs of three)", expanded=False):
            st.caption(
                "Each small chart repeats the F/M pairing within annoyed · neutral · happy for one filler."
            )
            top_fillers = (
                emotion.loc[emotion["words"] > 0, [f"rate {w}" for w in ordered_fillers()]]
                .mean()
                .sort_values(ascending=False)
                .head(6)
                .index.str.replace("rate ", "", regex=False)
                .tolist()
            )
            for filler in top_fillers:
                fig = fig_gender_paired_by_filler(emotion, filler)
                if fig is not None:
                    st.markdown(f"**{_display_filler(filler)}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"pack_{filler}")


if __name__ == "__main__":
    main()
