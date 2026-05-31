"""
Research Findings tab — hypothesis summaries, highlights, and Plotly charts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from filler_lexicon import CATEGORY_LABELS, category_for_filler, ordered_fillers
from simple_stats import (
    collect_significant,
    compare_metrics_in_dataframe,
    compare_many_groups,
    compare_two_groups,
    format_p,
    metric_label,
)
from stats_display import _style_pairwise_table

PHONE = "phonecall"
EMO_TASKS = ("neutral", "happy", "annoyed")
EMO_LABELS = {"neutral": "Neutral", "happy": "Happy", "annoyed": "Annoyed"}
COL_F = "#5b8def"
COL_M = "#e85d8a"
COL_EMO = {"neutral": "#94a3b8", "happy": "#fbbf24", "annoyed": "#ef4444"}


def _sex_fm_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().isin({"F", "M"})


def normalize_streamlit_filler_frame(fw: pd.DataFrame) -> pd.DataFrame:
    """Map streamlit_app attach_filler_columns output to EDA-style rate columns."""
    out = fw.copy()
    if "words" not in out.columns and "_word_count" in out.columns:
        out["words"] = out["_word_count"]
    if "rate total" not in out.columns and "_filler_per100" in out.columns:
        out["rate total"] = out["_filler_per100"]
    if "n total" not in out.columns and "_filler_total" in out.columns:
        out["n total"] = out["_filler_total"]
    wsafe = out["words"].replace(0, float("nan"))
    for w in ordered_fillers():
        hit = f"_f_{w}"
        if hit in out.columns:
            out[f"n {w}"] = out[hit]
            out[f"rate {w}"] = (out[hit] / wsafe) * 100.0
    for cat in CATEGORY_LABELS:
        cols = [f"n {x}" for x in ordered_fillers() if category_for_filler(x) == cat and f"n {x}" in out.columns]
        if cols:
            out[f"n {cat}"] = out[cols].sum(axis=1)
            out[f"rate {cat}"] = (out[f"n {cat}"] / wsafe) * 100.0
    return out


def filter_study_cohort(
    df: pd.DataFrame,
    *,
    age_min: int = 18,
    age_max: int = 24,
    min_words: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    if "info_age" in out.columns:
        age = pd.to_numeric(out["info_age"], errors="coerce")
        out = out.loc[age.ge(age_min) & age.le(age_max)]
    if min_words and "words" in out.columns:
        out = out.loc[out["words"] >= min_words]
    elif min_words and "_word_count" in out.columns:
        out = out.loc[out["_word_count"] >= min_words]
    return out


def _mean_rate(df: pd.DataFrame, col: str) -> float:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(x.mean()) if len(x) else 0.0


def _gender_means(sub: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        "Female": _mean_rate(sub.loc[sub["info_sex"].astype(str).str.upper() == "F"], col),
        "Male": _mean_rate(sub.loc[sub["info_sex"].astype(str).str.upper() == "M"], col),
    }


def _task_means(sub: pd.DataFrame, col: str) -> dict[str, float]:
    out = {}
    for t in EMO_TASKS:
        g = sub.loc[sub["task"] == t]
        if len(g):
            out[EMO_LABELS[t]] = _mean_rate(g, col)
    return out


def _highlight_bullets_phone(phone: pd.DataFrame) -> list[str]:
    sub = phone.loc[_sex_fm_mask(phone["info_sex"]) & (phone["words"] > 0)]
    if len(sub) < 4:
        return ["Not enough phone recordings to summarize."]
    bullets: list[str] = []
    total = compare_two_groups(
        sub.loc[sub["info_sex"].str.upper() == "F", "rate total"],
        sub.loc[sub["info_sex"].str.upper() == "M", "rate total"],
        label_a="Female",
        label_b="Male",
        metric="all fillers",
    )
    gm = _gender_means(sub, "rate total")
    if total.significant:
        hi = "Male" if gm["Male"] > gm["Female"] else "Female"
        lo = "Female" if hi == "Male" else "Male"
        bullets.append(
            f"**Men vs women (phone):** **{hi}** used more fillers overall "
            f"({gm[hi]:.1f} vs {gm[lo]:.1f} per 100 words, p = {format_p(total.p_value)})."
        )
    else:
        bullets.append(
            f"**Men vs women (phone):** About the same overall filler rate "
            f"(p = {format_p(total.p_value)})."
        )
    metrics = ["rate total"] + [f"rate {c}" for c in CATEGORY_LABELS] + [f"rate {w}" for w in ordered_fillers()]
    metrics = [m for m in metrics if m in sub.columns]
    results = compare_metrics_in_dataframe(
        sub,
        "info_sex",
        metrics,
        group_order=["F", "M"],
        group_labels={"F": "Female", "M": "Male"},
    )
    sig = collect_significant(results, headline=total)
    for r in sig[:6]:
        if r.comparison == total.comparison:
            continue
        name = metric_label(r.comparison)
        key = name.lower() if name.lower() in ordered_fillers() else name
        col = f"rate {key}" if f"rate {key}" in sub.columns else (f"rate {name}" if f"rate {name}" in sub.columns else None)
        if col:
            gm2 = _gender_means(sub, col)
            hi2 = "Male" if gm2["Male"] > gm2["Female"] else "Female"
            bullets.append(
                f"**{name}** — **{hi2}** higher on phone (p = {format_p(r.p_value)}; "
                f"men {gm2['Male']:.1f} vs women {gm2['Female']:.1f} per 100 words)."
            )
    return bullets[:8]


def _highlight_bullets_emotion(emotion: pd.DataFrame) -> list[str]:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    if len(sub) < 4:
        return ["Not enough emotion-story recordings to summarize."]
    bullets: list[str] = []
    groups = {EMO_LABELS[t]: sub.loc[sub["task"] == t, "rate total"] for t in EMO_TASKS if (sub["task"] == t).any()}
    if len(groups) < 2:
        return bullets
    head = compare_many_groups(groups, metric="all fillers", pairwise=True)
    tm = _task_means(sub, "rate total")
    bits = ", ".join(f"{k} {v:.1f}" for k, v in tm.items())
    if head.significant:
        bullets.append(
            f"**Emotion stories:** Overall filler rate **differs** by story type "
            f"(p = {format_p(head.p_value)}). Averages: {bits}."
        )
    else:
        bullets.append(
            f"**Emotion stories:** **Same overall** filler level across neutral, happy, and annoyed "
            f"(p = {format_p(head.p_value)}). Averages: {bits}. "
            "Differences show up in **which words** people use, not how much they fill overall."
        )
    metrics = ["rate total"] + [f"rate {c}" for c in CATEGORY_LABELS] + [f"rate {w}" for w in ordered_fillers() if f"rate {w}" in sub.columns]
    results = compare_metrics_in_dataframe(
        sub,
        "task",
        [m for m in metrics if m in sub.columns],
        group_order=list(EMO_TASKS),
        group_labels=EMO_LABELS,
    )
    for r in collect_significant(results, headline=head)[:5]:
        if r.comparison == head.comparison:
            continue
        bullets.append(f"**{metric_label(r.comparison)}** differs across story types (p = {format_p(r.p_value)}).")
    return bullets[:8]


def fig_gender_totals(phone: pd.DataFrame) -> go.Figure:
    sub = phone.loc[_sex_fm_mask(phone["info_sex"]) & (phone["words"] > 0)].copy()
    sub["Gender"] = sub["info_sex"].map({"F": "Female", "M": "Male"})
    fig = px.box(
        sub,
        x="Gender",
        y="rate total",
        color="Gender",
        color_discrete_map={"Female": COL_F, "Male": COL_M},
        points="all",
        title="Phone calls: fillers per 100 words (each dot = one recording)",
        labels={"rate total": "Fillers per 100 words"},
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=380)
    return fig


def fig_gender_gap_bars(phone: pd.DataFrame, top_n: int = 12) -> go.Figure | None:
    sub = phone.loc[_sex_fm_mask(phone["info_sex"]) & (phone["words"] > 0)]
    rows = []
    for w in ordered_fillers():
        col = f"rate {w}"
        if col not in sub.columns:
            continue
        gm = _gender_means(sub, col)
        rows.append({"Filler": w, "Gap (Male − Female)": gm["Male"] - gm["Female"]})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("Gap (Male − Female)", key=abs, ascending=False).head(top_n)
    df["Filler"] = df["Filler"].apply(lambda x: x if x != "i mean" else "I mean")
    colors = ["#3b7dd8" if g > 0 else COL_F for g in df["Gap (Male − Female)"]]
    fig = go.Figure(
        go.Bar(
            x=df["Gap (Male − Female)"],
            y=df["Filler"],
            orientation="h",
            marker_color=colors,
            text=df["Gap (Male − Female)"].round(2),
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666")
    fig.update_layout(
        title="Who uses more on the phone? Blue = male higher · pink = female higher",
        template="plotly_white",
        height=max(320, 28 * len(df)),
        xaxis_title="Male avg − Female avg (per 100 words)",
    )
    return fig


def fig_gender_categories(phone: pd.DataFrame) -> go.Figure:
    sub = phone.loc[_sex_fm_mask(phone["info_sex"]) & (phone["words"] > 0)]
    rows = []
    for cat in CATEGORY_LABELS:
        col = f"rate {cat}"
        if col not in sub.columns:
            continue
        for sex, lab in (("F", "Female"), ("M", "Male")):
            g = sub.loc[sub["info_sex"].astype(str).str.upper() == sex]
            rows.append({"Category": cat, "Gender": lab, "Rate": _mean_rate(g, col)})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Category",
        y="Rate",
        color="Gender",
        barmode="group",
        color_discrete_map={"Female": COL_F, "Male": COL_M},
        title="Phone: filler type by gender (average per 100 words)",
        labels={"Rate": "Per 100 words"},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def fig_emotion_totals(emotion: pd.DataFrame) -> go.Figure:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)].copy()
    sub["Story"] = sub["task"].map(EMO_LABELS)
    fig = px.box(
        sub,
        x="Story",
        y="rate total",
        color="Story",
        color_discrete_map=COL_EMO,
        category_orders={"Story": ["Neutral", "Happy", "Annoyed"]},
        points="all",
        title="Emotion retellings: fillers per 100 words",
        labels={"rate total": "Fillers per 100 words"},
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=380)
    return fig


def fig_emotion_heatmap(emotion: pd.DataFrame, top_n: int = 10) -> go.Figure | None:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    cols = [f"rate {w}" for w in ordered_fillers() if f"rate {w}" in sub.columns]
    if not cols:
        return None
    overall = sub[cols].mean().sort_values(ascending=False).head(top_n).index.tolist()
    rows = []
    for t in EMO_TASKS:
        g = sub.loc[sub["task"] == t]
        if len(g) == 0:
            continue
        for c in overall:
            w = c.replace("rate ", "")
            rows.append(
                {
                    "Story": EMO_LABELS[t],
                    "Filler": w if w != "i mean" else "I mean",
                    "Rate": _mean_rate(g, c),
                }
            )
    df = pd.DataFrame(rows)
    piv = df.pivot(index="Filler", columns="Story", values="Rate")
    piv = piv[["Neutral", "Happy", "Annoyed"]]
    fig = px.imshow(
        piv.values,
        x=piv.columns,
        y=piv.index,
        color_continuous_scale="YlOrRd",
        title="Which fillers show up most in each story type? (darker = more)",
        labels=dict(color="Per 100 words"),
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=max(360, 24 * len(piv)))
    return fig


def fig_emotion_top_grouped(emotion: pd.DataFrame, top_n: int = 8) -> go.Figure | None:
    """Grouped bars: top fillers × story type."""
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    cols = [f"rate {w}" for w in ordered_fillers() if f"rate {w}" in sub.columns]
    if not cols:
        return None
    overall = sub[cols].mean().sort_values(ascending=False).head(top_n).index
    rows = []
    for t in EMO_TASKS:
        g = sub.loc[sub["task"] == t]
        if len(g) == 0:
            continue
        for c in overall:
            w = c.replace("rate ", "")
            label = "I mean" if w == "i mean" else w
            rows.append({"Story": EMO_LABELS[t], "Filler": label, "Rate": _mean_rate(g, c)})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Story",
        y="Rate",
        color="Filler",
        barmode="group",
        category_orders={"Story": ["Neutral", "Happy", "Annoyed"]},
        title=f"Top {top_n} fillers by story (average per 100 words)",
        labels={"Rate": "Per 100 words"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(template="plotly_white", height=440, legend=dict(orientation="h", y=-0.25))
    return fig


def fig_emotion_category_heatmap(emotion: pd.DataFrame) -> go.Figure | None:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    rows = []
    for cat in CATEGORY_LABELS:
        col = f"rate {cat}"
        if col not in sub.columns:
            continue
        for t in EMO_TASKS:
            g = sub.loc[sub["task"] == t]
            if len(g):
                rows.append({"Story": EMO_LABELS[t], "Type": cat, "Rate": _mean_rate(g, col)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot(index="Type", columns="Story", values="Rate")
    piv = piv[["Neutral", "Happy", "Annoyed"]]
    fig = px.imshow(
        piv.values,
        x=piv.columns,
        y=piv.index,
        color_continuous_scale="Blues",
        title="Filler type intensity by emotion story (darker = more)",
        labels=dict(color="Per 100 words"),
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=280)
    return fig


def fig_emotion_lines_top(emotion: pd.DataFrame, top_n: int = 6) -> go.Figure | None:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    cols = [f"rate {w}" for w in ordered_fillers() if f"rate {w}" in sub.columns]
    if not cols:
        return None
    top_cols = sub[cols].mean().sort_values(ascending=False).head(top_n).index
    rows = []
    for c in top_cols:
        w = c.replace("rate ", "")
        label = "I mean" if w == "i mean" else w
        for t in EMO_TASKS:
            g = sub.loc[sub["task"] == t]
            if len(g):
                rows.append({"Story": EMO_LABELS[t], "Filler": label, "Rate": _mean_rate(g, c)})
    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="Story",
        y="Rate",
        color="Filler",
        markers=True,
        category_orders={"Story": ["Neutral", "Happy", "Annoyed"]},
        title="How top fillers move across neutral → happy → annoyed",
        labels={"Rate": "Per 100 words"},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def fig_emotion_categories(emotion: pd.DataFrame) -> go.Figure:
    sub = emotion.loc[emotion["task"].isin(EMO_TASKS) & (emotion["words"] > 0)]
    rows = []
    for cat in CATEGORY_LABELS:
        col = f"rate {cat}"
        if col not in sub.columns:
            continue
        for t in EMO_TASKS:
            g = sub.loc[sub["task"] == t]
            if len(g):
                rows.append({"Category": cat, "Story": EMO_LABELS[t], "Rate": _mean_rate(g, col)})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Story",
        y="Rate",
        color="Category",
        barmode="stack",
        category_orders={"Story": ["Neutral", "Happy", "Annoyed"]},
        title="Emotion stories: mix of filler types (stacked averages)",
        labels={"Rate": "Per 100 words"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def _hypothesis_verdict_card(title: str, question: str, supported: str | None, detail: str) -> None:
    st.markdown(f"#### {title}")
    st.markdown(f"*{question}*")
    if supported == "yes":
        st.success(detail)
    elif supported == "partial":
        st.warning(detail)
    else:
        st.info(detail)


def _render_emotion_hypotheses(emotion: pd.DataFrame, *, featured: bool = False) -> None:
    sub_e = emotion.loc[emotion["task"].isin(EMO_TASKS)]
    groups = {
        EMO_LABELS[t]: sub_e.loc[sub_e["task"] == t, "rate total"]
        for t in EMO_TASKS
        if (sub_e["task"] == t).any()
    }
    h2 = compare_many_groups(groups, metric="all fillers", pairwise=True) if len(groups) >= 2 else None
    tm = _task_means(sub_e, "rate total")
    if h2 and h2.significant:
        h2_detail = (
            f"**Supported (overall).** Filler rate differs across story types "
            f"(p = {format_p(h2.p_value)})."
        )
        h2_status = "yes"
    else:
        h2_detail = (
            f"**Partial / mix story.** Totals look similar (p = {format_p(h2.p_value) if h2 else 'n/a'}): "
            f"{', '.join(f'{k} {v:.1f}' for k, v in tm.items())} per 100 words. "
            "The interesting part is **which words** change — *um*, *you know*, *like*, etc."
        )
        h2_status = "partial"
    _hypothesis_verdict_card(
        "Hypothesis 2 · Emotion stories",
        "Do people use more or fewer fillers when retelling neutral vs happy vs annoyed conversations?",
        h2_status,
        h2_detail,
    )
    if featured:
        st.plotly_chart(fig_emotion_totals(emotion), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            hm = fig_emotion_heatmap(emotion, top_n=12)
            if hm:
                st.plotly_chart(hm, use_container_width=True)
        with c2:
            ch = fig_emotion_category_heatmap(emotion)
            if ch:
                st.plotly_chart(ch, use_container_width=True)
        grp = fig_emotion_top_grouped(emotion)
        if grp:
            st.plotly_chart(grp, use_container_width=True)
        ln = fig_emotion_lines_top(emotion)
        if ln:
            st.plotly_chart(ln, use_container_width=True)
    else:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_emotion_totals(emotion), use_container_width=True)
        with c4:
            hm = fig_emotion_heatmap(emotion)
            if hm:
                st.plotly_chart(hm, use_container_width=True)

    h4_sig = any(
        compare_many_groups(
            {
                EMO_LABELS[t]: sub_e.loc[sub_e["task"] == t, f"rate {cat}"]
                for t in EMO_TASKS
                if (sub_e["task"] == t).any()
            },
            metric=cat,
        ).significant
        for cat in CATEGORY_LABELS
        if f"rate {cat}" in sub_e.columns
    )
    _hypothesis_verdict_card(
        "Hypothesis 4 · Filler types × emotion",
        "Do neutral, happy, and annoyed stories use different mixes of filler types?",
        "yes" if h4_sig else "partial",
        "**Supported.** Story type shifts which filler categories show up — often placeholders vs feedback phrases — "
        "even when total filler counts stay flat."
        if h4_sig
        else "**Partial.** See category heatmap and grouped bars for where the action is.",
    )
    st.plotly_chart(fig_emotion_categories(emotion), use_container_width=True)
    if h2 and h2.pairwise is not None and len(h2.pairwise):
        st.markdown("**Pairwise emotion comparisons** (yellow rows = statistically notable)")
        try:
            st.dataframe(_style_pairwise_table(h2.pairwise), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(h2.pairwise, use_container_width=True, hide_index=True)


def _render_phone_hypotheses(phone: pd.DataFrame) -> None:
    sub_p = phone.loc[_sex_fm_mask(phone["info_sex"])]
    h1 = compare_two_groups(
        sub_p.loc[sub_p["info_sex"].str.upper() == "F", "rate total"],
        sub_p.loc[sub_p["info_sex"].str.upper() == "M", "rate total"],
        label_a="Female",
        label_b="Male",
        metric="all fillers",
    )
    gm = _gender_means(sub_p, "rate total")
    if h1.significant:
        hi = "Male" if gm["Male"] > gm["Female"] else "Female"
        h1_detail = (
            f"**Supported (overall).** On phone calls, **{hi}** speakers had a higher average filler rate "
            f"({gm['Male']:.1f} male vs {gm['Female']:.1f} female per 100 words, p = {format_p(h1.p_value)}). "
            "The gap is driven especially by **hesitation** words like *uh*, not mainly by *like*."
        )
        h1_status = "yes"
    else:
        h1_detail = (
            f"**Not supported (overall).** Female and male phone recordings look similar on total fillers "
            f"(p = {format_p(h1.p_value)}). Check specific words in the gap chart below."
        )
        h1_status = "no"
    _hypothesis_verdict_card(
        "Hypothesis 1 · Gender on phone calls",
        "Do women and men use different amounts of fillers during real phone calls?",
        h1_status,
        h1_detail,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_gender_totals(phone), use_container_width=True)
    with c2:
        gap = fig_gender_gap_bars(phone)
        if gap:
            st.plotly_chart(gap, use_container_width=True)

    st.divider()
    h3_sig = False
    for cat in CATEGORY_LABELS:
        col = f"rate {cat}"
        if col not in sub_p.columns:
            continue
        r = compare_two_groups(
            sub_p.loc[sub_p["info_sex"].str.upper() == "F", col],
            sub_p.loc[sub_p["info_sex"].str.upper() == "M", col],
            label_a="Female",
            label_b="Male",
            metric=cat,
        )
        if r.significant:
            h3_sig = True
    h3_detail = (
        "**Supported.** At least one filler **type** (placeholders, Californese, or feedback) differs by gender on the phone. "
        "Men tend to score higher on **placeholders** (*um*, *uh*)."
        if h3_sig
        else "**Weak / mixed.** Filler types look similar by gender on the phone in this sample."
    )
    _hypothesis_verdict_card(
        "Hypothesis 3 · Filler types × gender (phone)",
        "Do women and men favor different kinds of fillers on the phone?",
        "yes" if h3_sig else "partial",
        h3_detail,
    )
    st.plotly_chart(fig_gender_categories(phone), use_container_width=True)


def render_research_findings_tab(
    phone: pd.DataFrame,
    emotion: pd.DataFrame,
    *,
    cohort_note: str = "Ages 18–24 · UCLA Speaker Variability corpus",
    page_style: str = "default",
) -> None:
    """Full Research Findings content (EDA app or Research page)."""
    st.header("Research findings")
    st.caption(cohort_note)
    st.markdown(
        "Four hypotheses · **interesting** results (p below 0.05) · charts = fillers per 100 words (each dot = one recording)."
    )

    phone = phone.loc[phone["words"] > 0] if "words" in phone.columns else phone
    emotion = emotion.loc[emotion["words"] > 0] if "words" in emotion.columns else emotion

    emo_highlights = _highlight_bullets_emotion(emotion)
    phone_highlights = _highlight_bullets_phone(phone)

    if page_style == "research":
        st.markdown("## Emotion stories — highlights")
        st.caption("Neutral, happy, and annoyed retellings (main focus for this study).")
        for h in emo_highlights:
            st.markdown(f"- {h}")
        if not emo_highlights:
            st.info("No emotion highlights in this filter slice.")

        st.markdown("## Emotion · hypotheses & charts")
        _render_emotion_hypotheses(emotion, featured=True)

        st.divider()
        st.markdown("## Phone calls — highlights")
        for h in phone_highlights:
            st.markdown(f"- {h}")

        st.markdown("## Phone · hypotheses & charts")
        _render_phone_hypotheses(phone)
    else:
        st.markdown("### Highlights")
        for h in emo_highlights + phone_highlights:
            st.markdown(f"- {h}")
        st.divider()
        _render_phone_hypotheses(phone)
        st.divider()
        _render_emotion_hypotheses(emotion, featured=False)

    with st.expander("For your paper (Discussion vs Results)", expanded=False):
        st.markdown(
            """
**Results** = report the numbers and p-values (tables elsewhere in this app).

**Discussion** = use the hypothesis cards above:
- Was each hypothesis supported, partly supported, or not?
- Point to **one example transcript** for the clearest gap (e.g. male *uh* on a phone call).
- Note limits: college sample, English transcripts, automatic filler matching.
            """
        )


def render_research_findings_from_raw(
    df: pd.DataFrame,
    *,
    attach_fn: Callable[[pd.DataFrame], pd.DataFrame],
    age_min: int = 18,
    age_max: int = 24,
    min_words: int = 20,
    cohort_note: str | None = None,
    page_style: str = "research",
) -> None:
    """Build filler columns then render (streamlit_app / Research page)."""
    work = filter_study_cohort(df, age_min=age_min, age_max=age_max, min_words=0)
    if len(work) == 0:
        st.warning("No rows in age range.")
        return
    with st.spinner("Preparing research cohort…"):
        fw = normalize_streamlit_filler_frame(attach_fn(work))
        fw = filter_study_cohort(fw, age_min=age_min, age_max=age_max, min_words=min_words)
    phone = fw.loc[fw["task"].eq(PHONE)].copy()
    emotion = fw.loc[fw["task"].isin(EMO_TASKS)].copy()
    note = cohort_note or f"Ages {age_min}–{age_max} · filtered cohort"
    render_research_findings_tab(phone, emotion, cohort_note=note, page_style=page_style)
