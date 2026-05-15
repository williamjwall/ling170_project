"""
Simple filler exploration for a linguistics project (Streamlit).

Run: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from filler_lexicon import (
    CATEGORY_LABELS,
    enrich_dataframe,
    ordered_fillers,
    transcript_highlight_html,
)

DEFAULT_CSV = Path(__file__).resolve().parent / "emotion_phone_simplified.csv"
PHONE = "phonecall"
EMOTIONS = ("annoyed", "happy", "neutral")


def l1_group(value: object) -> str:
    s = str(value).strip()
    if s == "Y":
        return "L1 English"
    if s == "N":
        return "L1 not English"
    if s == "Y sounds non-native":
        return "English L1 (rated non-native)"
    if s == "N sounds native":
        return "Not English L1 (rated native-like)"
    return "Other"


def counting_text_column(df: pd.DataFrame) -> str:
    if "text_filler" in df.columns:
        return "text_filler"
    return "text"


@st.cache_data
def cached(csv_path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    df["info_sex"] = df["info_sex"].astype(str).str.strip()
    df = df[df["info_sex"].isin(("M", "F"))].copy()
    count_col = counting_text_column(df)
    df = enrich_dataframe(df, text_col=count_col)
    if "info_l1_english" in df.columns:
        df["L1 group"] = df["info_l1_english"].map(l1_group)
    else:
        df["L1 group"] = "Unknown"
    return df, count_col


def y_total(use_rate: bool) -> str:
    return "rate total" if use_rate else "n total"


def y_label(use_rate: bool) -> str:
    return "Fillers per 100 words" if use_rate else "Total filler count"


def category_rate_cols() -> list[str]:
    return [f"rate {c}" for c in CATEGORY_LABELS]


def category_count_cols() -> list[str]:
    return [f"n {c}" for c in CATEGORY_LABELS]


def display_phrase(phrase: str) -> str:
    return phrase if phrase != "i mean" else "I mean"


def filler_metric_cols(data: pd.DataFrame, use_rate: bool) -> list[str]:
    prefix = "rate " if use_rate else "n "
    return [f"{prefix}{w}" for w in ordered_fillers() if f"{prefix}{w}" in data.columns]


def sex_label(code: str) -> str:
    return {"M": "Male", "F": "Female"}.get(str(code), str(code))


def story_label(task: str) -> str:
    return str(task).capitalize()


def transcript_label(
    speaker_id: object, task: object, sex: object, age: object, words: int | None = None
) -> str:
    meta = str(sex).strip()
    if pd.notna(age):
        try:
            meta += f", age {int(age)}"
        except (TypeError, ValueError):
            pass
    if words is not None:
        meta += f", {int(words)} words"
    return f"Speaker {int(speaker_id)} — {task} ({meta})"


def reading_guide() -> None:
    with st.expander("How to read these charts (30 seconds)", expanded=False):
        st.markdown(
            """
- **Each dot** = one person’s recording for that task.
- **Box plots** show the middle of the pack (the box), the typical spread, and dots that sit far from the rest.
- **Bar charts** show the **average** for a group. If two bars are close, the groups are similar; if they are far apart, the difference is easier to see.
- **Fillers per 100 words** = a fair way to compare short vs long recordings (similar to “per minute” instead of raw totals).
            """
        )


def simple_summary(data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    sub = data[data["words"] > 0]
    out = sub.groupby(group_col, observed=False)[value_col].agg(
        people="count",
        average="mean",
        typical="median",
    )
    return out.round(2)


def show_summary_table(tbl: pd.DataFrame, index_names: dict | None = None) -> None:
    if index_names:
        tbl = tbl.rename(index=index_names)
    tbl.columns = ["Number of people", "Average", "Typical person (middle value)"]
    st.dataframe(tbl, use_container_width=True)


def inject_css() -> None:
    st.markdown(
        """
<style>
mark.filler-ph { background: #b8e0ff; padding: 0 0.12em; border-radius: 3px; }
mark.filler-ca { background: #ffd6a8; padding: 0 0.12em; border-radius: 3px; }
mark.filler-fb { background: #caffbf; padding: 0 0.12em; border-radius: 3px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def fig_box_sex_task(df: pd.DataFrame, y: str, title: str):
    d = df[df["words"] > 0].copy()
    d["task"] = pd.Categorical(d["task"], categories=list(EMOTIONS), ordered=True)
    return px.box(
        d,
        x="task",
        y=y,
        color="info_sex",
        points="all",
        title=title,
        labels={"task": "Story type", "info_sex": "Gender", y: y_label("rate" in y)},
    )


def fig_heatmap_emotion_gender(emotion: pd.DataFrame, y: str):
    sub = emotion[emotion["words"] > 0]
    p = sub.pivot_table(index="task", columns="info_sex", values=y, aggfunc="mean")
    p = p.reindex(index=list(EMOTIONS), columns=["F", "M"], fill_value=float("nan"))
    return px.imshow(
        p.values,
        x=["Female", "Male"],
        y=[t.capitalize() for t in p.index],
        color_continuous_scale="Blues",
        title="Average filler use (warmer = more)",
        aspect="auto",
        labels=dict(x="Gender", y="Story type", color="Average"),
    )


def fig_line_emotion_sex(emotion: pd.DataFrame, y: str):
    sub = emotion[emotion["words"] > 0]
    g = sub.groupby(["task", "info_sex"], observed=False)[y].mean().reset_index()
    g["task"] = pd.Categorical(g["task"], categories=list(EMOTIONS), ordered=True)
    return px.line(
        g,
        x="task",
        y=y,
        color="info_sex",
        markers=True,
        title="Average by story type and gender",
        labels={"task": "Story type", "info_sex": "Gender", y: y_label("rate" in y)},
    )


def fig_bar_l1(emotion: pd.DataFrame, y: str):
    sub = emotion[emotion["words"] > 0]
    if "L1 group" not in sub.columns:
        return px.bar(
            pd.DataFrame({"L1 group": ["(not in data file)"], "value": [0.0]}),
            x="L1 group",
            y="value",
            title="This CSV has no first-language column to group on.",
        )
    g = sub.groupby("L1 group", observed=False)[y].mean().reset_index()
    g = g.sort_values(y, ascending=False)
    return px.bar(
        g,
        x="L1 group",
        y=y,
        title="Average filler use by first language (emotion stories only)",
        labels={"L1 group": "Language background", y: y_label("rate" in y)},
    )


def fig_stacked_category_emotion(emotion: pd.DataFrame, use_rate: bool):
    """Mean fillers per 100 words, stacked by category (same scale)."""
    sub = emotion[emotion["words"] > 0].copy()
    cols = category_rate_cols() if use_rate else category_count_cols()
    g = sub.groupby("task", observed=False)[cols].mean().reset_index()
    g["task"] = pd.Categorical(g["task"], categories=list(EMOTIONS), ordered=True)
    g = g.sort_values("task")
    long = g.melt(id_vars=["task"], value_vars=cols, var_name="Category", value_name="value")
    long["Category"] = long["Category"].str.replace("rate ", "").str.replace("n ", "")
    return px.bar(
        long,
        x="task",
        y="value",
        color="Category",
        barmode="stack",
        title="How much of the filler use is each category? (mean per story type)",
        labels={"task": "Story type", "value": y_label(use_rate)},
    )


def fig_grouped_category_phone(phone: pd.DataFrame, use_rate: bool):
    sub = phone[phone["words"] > 0].copy()
    cols = category_rate_cols() if use_rate else category_count_cols()
    g = sub.groupby("info_sex", observed=False)[cols].mean().reset_index()
    long = g.melt(id_vars=["info_sex"], value_vars=cols, var_name="Category", value_name="value")
    long["Category"] = long["Category"].str.replace("rate ", "").str.replace("n ", "")
    return px.bar(
        long,
        x="info_sex",
        y="value",
        color="Category",
        barmode="group",
        title="Phone call: average use by category and gender",
        labels={"info_sex": "Gender", "value": y_label(use_rate)},
    )


def fig_grouped_category_emotion_sex(emotion: pd.DataFrame, use_rate: bool):
    sub = emotion[emotion["words"] > 0].copy()
    cols = category_rate_cols() if use_rate else category_count_cols()
    g = sub.groupby(["task", "info_sex"], observed=False)[cols].mean().reset_index()
    long = g.melt(
        id_vars=["task", "info_sex"],
        value_vars=cols,
        var_name="Category",
        value_name="value",
    )
    long["Category"] = long["Category"].str.replace("rate ", "").str.replace("n ", "")
    long["task"] = pd.Categorical(long["task"], categories=list(EMOTIONS), ordered=True)
    return px.bar(
        long,
        x="task",
        y="value",
        color="Category",
        facet_col="info_sex",
        barmode="group",
        title="Emotion stories: averages by category, story type, and gender",
        labels={"task": "Story type", "value": y_label(use_rate)},
    )


def top_filler_gap_bar(phone: pd.DataFrame, use_rate: bool, top_n: int = 12):
    cols = filler_metric_cols(phone, use_rate)
    sub = phone[phone["words"] > 0]
    m = sub.loc[sub["info_sex"] == "M", cols].mean()
    f = sub.loc[sub["info_sex"] == "F", cols].mean()
    diff_raw = m - f
    diff = diff_raw.loc[diff_raw.abs().sort_values(ascending=False).index].head(top_n)
    frame = pd.DataFrame(
        {"gap": diff.values, "filler": [display_phrase(c.removeprefix("rate ").removeprefix("n ")) for c in diff.index]}
    )
    return px.bar(
        frame,
        x="gap",
        y="filler",
        orientation="h",
        title="Which fillers differ most between male and female on the phone?",
        labels={"gap": "Male average minus female average", "filler": "Filler"},
    )


def summary_table(data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return simple_summary(data, group_col, value_col)


def fig_mean_bars(data: pd.DataFrame, group_col: str, y: str, title: str, order: list | None = None):
    sub = data[data["words"] > 0]
    g = sub.groupby(group_col, observed=False)[y].mean().reset_index()
    if order:
        g[group_col] = pd.Categorical(g[group_col], categories=order, ordered=True)
        g = g.sort_values(group_col)
    return px.bar(
        g,
        x=group_col,
        y=y,
        title=title,
        labels={group_col: group_col.replace("_", " ").title(), y: y_label("rate" in y)},
        text_auto=".2f",
    )


def fig_top_fillers_used(data: pd.DataFrame, use_rate: bool, top_n: int = 15, title: str = ""):
    cols = filler_metric_cols(data, use_rate)
    sub = data[data["words"] > 0]
    if not cols:
        return px.bar(title=title or "Most common fillers on average")
    means = sub[cols].mean().sort_values(ascending=False).head(top_n)
    frame = pd.DataFrame(
        {
            "value": means.values,
            "filler": [display_phrase(c.removeprefix("rate ").removeprefix("n ")) for c in means.index],
        }
    )
    return px.bar(
        frame,
        x="value",
        y="filler",
        orientation="h",
        title=title or "Most common fillers on average",
        labels={"value": y_label(use_rate), "filler": "Filler"},
    )


def fig_filler_heatmap_by_group(data: pd.DataFrame, group_col: str, use_rate: bool, top_n: int = 12):
    cols = filler_metric_cols(data, use_rate)
    sub = data[data["words"] > 0]
    overall = sub[cols].mean().sort_values(ascending=False).head(top_n).index.tolist()
    g = sub.groupby(group_col, observed=False)[overall].mean()
    labels = [display_phrase(c.removeprefix("rate ").removeprefix("n ")) for c in overall]
    return px.imshow(
        g.values,
        x=labels,
        y=[str(x) for x in g.index],
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title=f"Average filler use by {group_col} (top {top_n} fillers)",
        labels=dict(x="Filler", y=group_col.replace("_", " ").title(), color="Average"),
    )


def averages_by_story_table(emotion: pd.DataFrame, y: str) -> pd.DataFrame:
    sub = emotion[emotion["words"] > 0]
    g = sub.groupby("task", observed=False)[y].mean().reindex(EMOTIONS)
    return pd.DataFrame({"Story type": [story_label(t) for t in g.index], "Average": g.values.round(2)})


def fig_heatmap_emotion_l1(emotion: pd.DataFrame, y: str):
    sub = emotion[emotion["words"] > 0]
    if "L1 group" not in sub.columns:
        return None
    p = sub.pivot_table(index="task", columns="L1 group", values=y, aggfunc="mean")
    p = p.reindex(index=list(EMOTIONS))
    return px.imshow(
        p.values,
        x=list(p.columns),
        y=[t.capitalize() for t in p.index],
        color_continuous_scale="Greens",
        aspect="auto",
        title="Average filler use: story type × first language",
        labels=dict(x="Language background", y="Story type", color="Average"),
    )


def category_cols(use_rate: bool) -> list[str]:
    return category_rate_cols() if use_rate else category_count_cols()


def melt_categories(data: pd.DataFrame, use_rate: bool, id_cols: list[str]) -> pd.DataFrame:
    cols = category_cols(use_rate)
    long = data[data["words"] > 0].melt(id_vars=id_cols, value_vars=cols, var_name="Category", value_name="value")
    long["Category"] = long["Category"].str.replace("rate ", "").str.replace("n ", "")
    return long


def add_category_share_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Percent of each person's fillers that fall in each bucket."""
    out = data.copy()
    total = out["n total"].replace(0, float("nan"))
    for cat in CATEGORY_LABELS:
        out[f"share {cat}"] = (out[f"n {cat}"] / total) * 100.0
    return out


def fig_category_gap_phone(phone: pd.DataFrame, use_rate: bool):
    sub = phone[phone["words"] > 0]
    cols = category_cols(use_rate)
    m = sub.loc[sub["info_sex"] == "M", cols].mean()
    f = sub.loc[sub["info_sex"] == "F", cols].mean()
    frame = pd.DataFrame(
        {
            "gap": (m - f).values,
            "Category": [c.removeprefix("rate ").removeprefix("n ") for c in cols],
        }
    )
    return px.bar(
        frame,
        x="gap",
        y="Category",
        orientation="h",
        title="Phone: male average minus female average (by category)",
        labels={"gap": "Difference (male − female)", "Category": "Category"},
    )


def fig_heatmap_category_cross(data: pd.DataFrame, row_col: str, col_col: str, cat_col: str, title: str):
    sub = data[data["words"] > 0]
    p = sub.pivot_table(index=row_col, columns=col_col, values=cat_col, aggfunc="mean")
    if row_col == "task":
        p = p.reindex(EMOTIONS)
    if col_col == "info_sex":
        p = p.reindex(columns=["F", "M"])
    return px.imshow(
        p.values,
        x=["Female" if c == "F" else "Male" if c == "M" else str(c) for c in p.columns],
        y=[str(r).capitalize() for r in p.index],
        color_continuous_scale="Purples",
        aspect="auto",
        title=title,
        labels=dict(color="Average"),
    )


def category_definitions_block() -> None:
    st.markdown(
        """
We group every filler into **one of three types**:

1. **Placeholders** — pauses and sounds such as *um*, *uh*, *hmm*, *mhm*
2. **Californese** — discourse words such as *like*, *literally*, *so*, *basically*, *actually*
3. **Feedback** — phrases that soften or check in with the listener: *you know*, *I mean*, *well*, *kind of*
        """
    )


def render_question3(phone: pd.DataFrame, use_rate: bool) -> None:
    st.header("Question 3: Phone call — three types of fillers")
    st.write("Same question as **Question 1** (male vs female on the phone), but we add up fillers into **three types** instead of listing every word.")
    reading_guide()
    category_definitions_block()

    d = add_category_share_columns(phone[phone["words"] > 0].copy())
    if d.empty:
        st.warning("No phone transcripts to show.")
        return

    st.subheader("How much does each type get used?")
    st.plotly_chart(fig_grouped_category_phone(phone, use_rate), use_container_width=True)
    st.caption("Taller bars = more of that type on average. Compare male and female side by side.")

    long = melt_categories(d, use_rate, ["info_sex"])
    st.plotly_chart(
        px.box(
            long,
            x="Category",
            y="value",
            color="info_sex",
            points="all",
            labels={"value": y_label(use_rate), "info_sex": "Gender", "Category": "Type"},
            title="Spread of each type on the phone (dots = one person)",
        ),
        use_container_width=True,
    )

    st.subheader("Average scores by gender")
    for cat in CATEGORY_LABELS:
        ccol = f"rate {cat}" if use_rate else f"n {cat}"
        st.markdown(f"**{cat}**")
        show_summary_table(simple_summary(phone, "info_sex", ccol), {"M": "Male", "F": "Female"})

    st.subheader("What share of fillers is each type?")
    st.write("Out of all fillers someone said on the phone, what percent were placeholders, Californese, or feedback? (Only people who used at least one filler.)")
    share_cols = [f"share {c}" for c in CATEGORY_LABELS]
    long_share = d.melt(id_vars=["info_sex"], value_vars=share_cols, var_name="Category", value_name="share")
    long_share["Category"] = long_share["Category"].str.replace("share ", "")
    st.plotly_chart(
        px.box(
            long_share,
            x="Category",
            y="share",
            color="info_sex",
            points="all",
            labels={"share": "Percent of their fillers", "info_sex": "Gender"},
            title="Mix of filler types on the phone",
        ),
        use_container_width=True,
    )

    pick = st.selectbox("Look closer at one type", CATEGORY_LABELS, key="q3_cat_pick")
    ccol = f"rate {pick}" if use_rate else f"n {pick}"
    st.plotly_chart(
        fig_mean_bars(d, "info_sex", ccol, f"Average {pick} — male vs female", order=["F", "M"]),
        use_container_width=True,
    )


def render_question4(emotion: pd.DataFrame, use_rate: bool) -> None:
    st.header("Question 4: Emotion stories — three types of fillers")
    st.write("Same question as **Question 2** (annoyed vs happy vs neutral), but using the **three filler types**.")
    reading_guide()
    category_definitions_block()

    d = add_category_share_columns(emotion[emotion["words"] > 0].copy())
    if d.empty:
        st.warning("No emotion transcripts to show.")
        return

    d["task"] = pd.Categorical(d["task"], categories=list(EMOTIONS), ordered=True)

    st.subheader("Which type shows up most in each story?")
    st.plotly_chart(fig_stacked_category_emotion(emotion, use_rate), use_container_width=True)
    st.caption("Each bar is one story type. The colored chunks show how much of the total filler use came from each type.")

    st.plotly_chart(fig_grouped_category_emotion_sex(emotion, use_rate), use_container_width=True)
    st.caption("Compare story types and gender. Look for bars that stand out, not tiny differences.")

    st.subheader("Average scores by story type")
    for cat in CATEGORY_LABELS:
        ccol = f"rate {cat}" if use_rate else f"n {cat}"
        st.markdown(f"**{cat}**")
        show_summary_table(
            simple_summary(emotion, "task", ccol),
            {t: story_label(t) for t in EMOTIONS},
        )

    st.subheader("Story type and gender together")
    st.write("These tables show the **average** for each combination. Higher numbers = more fillers of that type.")
    for cat in CATEGORY_LABELS:
        ccol = f"rate {cat}" if use_rate else f"n {cat}"
        cross = (
            d.pivot_table(index="task", columns="info_sex", values=ccol, aggfunc="mean")
            .reindex(EMOTIONS)
            .round(2)
        )
        cross.columns = [sex_label(c) for c in cross.columns]
        cross.index = [story_label(t) for t in cross.index]
        st.markdown(f"**{cat}**")
        st.dataframe(cross, use_container_width=True)

    pick = st.selectbox("Look closer at one type", CATEGORY_LABELS, key="q4_cat_pick")
    ccol = f"rate {pick}" if use_rate else f"n {pick}"
    st.plotly_chart(
        px.box(
            d,
            x="task",
            y=ccol,
            color="info_sex",
            points="all",
            labels={"task": "Story type", "info_sex": "Gender", ccol: pick},
            title=f"{pick} — annoyed, happy, and neutral (dots = one person)",
        ),
        use_container_width=True,
    )


def render_question1(phone: pd.DataFrame, use_rate: bool, ycol: str) -> None:
    st.header("Question 1: Do males and females use different fillers on the phone?")
    st.write(
        "We only look at the **phone call** task. The score is **fillers per 100 words** "
        "(turn off in the sidebar if you prefer raw counts)."
    )
    reading_guide()

    d = phone[phone["words"] > 0].copy()
    if d.empty:
        st.warning("No phone transcripts to show.")
        return

    st.subheader("Overall filler use: male vs female")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.box(
                d,
                x="info_sex",
                y=ycol,
                points="all",
                color="info_sex",
                labels={"info_sex": "Gender", ycol: y_label(use_rate)},
                title="Each dot is one person",
            ).update_layout(showlegend=False),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            fig_mean_bars(d, "info_sex", ycol, "Average fillers by gender", order=["F", "M"]),
            use_container_width=True,
        )

    st.subheader("Summary")
    show_summary_table(simple_summary(phone, "info_sex", ycol), {"M": "Male", "F": "Female"})

    st.subheader("Which filler words show up most on the phone?")
    st.plotly_chart(
        fig_top_fillers_used(d, use_rate, top_n=12, title="Most common fillers (average across everyone)"),
        use_container_width=True,
    )

    st.subheader("Where do male and female averages differ the most?")
    st.plotly_chart(top_filler_gap_bar(phone, use_rate, top_n=10), use_container_width=True)
    st.caption("Bars to the right = higher average for males; bars to the left = higher for females.")

    pick = st.selectbox(
        "Look at one filler word",
        list(ordered_fillers()),
        format_func=display_phrase,
        key="q1_filler_pick",
    )
    fcol = f"rate {pick}" if use_rate else f"n {pick}"
    st.plotly_chart(
        fig_mean_bars(d, "info_sex", fcol, f"Average “{display_phrase(pick)}” by gender", order=["F", "M"]),
        use_container_width=True,
    )


def render_question2(emotion: pd.DataFrame, use_rate: bool, ycol: str) -> None:
    st.header("Question 2: Do annoyed, happy, and neutral stories differ?")
    st.write(
        "These are the **three emotional story tasks** (not the phone call). "
        "We ask whether people use more or fewer fillers when they retell an annoyed, happy, or neutral story."
    )
    reading_guide()

    d = emotion[emotion["words"] > 0].copy()
    if d.empty:
        st.warning("No emotion transcripts to show.")
        return

    d["task"] = pd.Categorical(d["task"], categories=list(EMOTIONS), ordered=True)

    st.subheader("Overall filler use by story type")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.box(
                d,
                x="task",
                y=ycol,
                points="all",
                color="task",
                labels={"task": "Story type", ycol: y_label(use_rate)},
                title="Each dot is one person",
            ).update_layout(showlegend=False),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            fig_mean_bars(d, "task", ycol, "Average fillers by story type", order=list(EMOTIONS)),
            use_container_width=True,
        )

    st.subheader("Summary")
    show_summary_table(
        simple_summary(emotion, "task", ycol),
        {t: story_label(t) for t in EMOTIONS},
    )
    st.markdown("**Quick look — average for each story type**")
    st.dataframe(averages_by_story_table(emotion, ycol), use_container_width=True, hide_index=True)

    st.subheader("Does gender change the pattern?")
    st.plotly_chart(fig_box_sex_task(emotion, ycol, "Fillers by story type and gender"), use_container_width=True)
    st.plotly_chart(fig_line_emotion_sex(emotion, ycol), use_container_width=True)

    st.write("**Average fillers per 100 words** (or your chosen scale) for each story and gender:")
    cross = (
        d.pivot_table(index="task", columns="info_sex", values=ycol, aggfunc="mean")
        .reindex(EMOTIONS)
        .round(2)
    )
    cross.columns = [sex_label(c) for c in cross.columns]
    cross.index = [story_label(t) for t in cross.index]
    st.dataframe(cross, use_container_width=True)

    if "L1 group" in d.columns:
        st.subheader("First language (extra context)")
        st.write("There are not many people in every language group, so treat this as background only.")
        st.plotly_chart(fig_bar_l1(emotion, ycol), use_container_width=True)

    pick = st.selectbox(
        "Look at one filler word",
        list(ordered_fillers()),
        format_func=display_phrase,
        key="q2_filler_pick",
    )
    fcol = f"rate {pick}" if use_rate else f"n {pick}"
    st.plotly_chart(
        px.box(
            d,
            x="task",
            y=fcol,
            color="info_sex",
            points="all",
            labels={"task": "Story type", "info_sex": "Gender", fcol: display_phrase(pick)},
            title=f"“{display_phrase(pick)}” across stories and gender",
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Filler study", layout="wide")
    inject_css()

    with st.sidebar:
        st.header("Settings")
        csv_path = st.text_input("Data file", value=str(DEFAULT_CSV))
        use_rate = st.toggle("Show rates per 100 words", value=True)
    try:
        df, count_col = cached(csv_path)
    except FileNotFoundError:
        st.error("Could not find that file.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    if "L1 group" not in df.columns:
        if "info_l1_english" in df.columns:
            df["L1 group"] = df["info_l1_english"].map(l1_group)
        else:
            df["L1 group"] = "Unknown"

    ycol = y_total(use_rate)
    phone = df[df["task"] == PHONE].copy()
    emotion = df[df["task"].isin(EMOTIONS)].copy()
    fillers = list(ordered_fillers())

    st.title("Filler words in your study")
    st.write(
        "This tool **counts filler words** in each recording and shows **simple charts** for your four research questions. "
        "There are no statistical tests — just averages and pictures you can talk through in a presentation."
    )
    st.caption(f"{len(df)} recordings loaded (male and female speakers).")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "1. Phone and gender",
            "2. Emotion stories",
            "3. Phone and gender (categories)",
            "4. Emotion stories (categories)",
            "Transcripts",
            "Download data",
        ]
    )

    with tab1:
        render_question1(phone, use_rate, ycol)

    with tab2:
        render_question2(emotion, use_rate, ycol)

    with tab3:
        render_question3(phone, use_rate)

    with tab4:
        render_question4(emotion, use_rate)

    with tab5:
        st.header("Read transcripts")
        st.write(
            "Highlighted words match our filler list. "
            "**Blue** = placeholders, **orange** = Californese, **green** = feedback."
        )
        task_choice = st.selectbox("Task", ["All tasks", PHONE, *EMOTIONS])
        view = df if task_choice == "All tasks" else df[df["task"] == task_choice].copy()
        speakers = sorted(view["speaker_id"].unique().tolist())
        mode = st.radio("Layout", ("Pick one transcript", "Several in collapsible panels"), horizontal=True)
        chosen = st.multiselect(
            "Which fillers to highlight (empty = all)",
            options=fillers,
            default=fillers,
            format_func=display_phrase,
        )
        hl = set(chosen) if chosen else None

        if mode == "Pick one transcript":
            labels = [
                transcript_label(r.speaker_id, r.task, r.info_sex, r.info_age, r.words)
                for r in view.itertuples(index=False)
            ]
            if not labels:
                st.info("Nothing to show for this filter.")
            else:
                i = st.selectbox("Transcript", range(len(labels)), format_func=lambda j: labels[j])
                row = view.iloc[i]
                st.markdown(f"**{labels[i]}**")
                body = transcript_highlight_html(str(row.text), highlight=hl, for_display=True)
                st.markdown(
                    f'<div style="white-space: pre-wrap; line-height: 1.55;">{body}</div>',
                    unsafe_allow_html=True,
                )
        else:
            cap = st.slider("How many to list", 1, 20, 8)
            pick_sp = st.multiselect("Limit to these speaker IDs (empty = everyone)", speakers, format_func=lambda x: str(int(x)))
            v2 = view if not pick_sp else view[view["speaker_id"].isin(pick_sp)]
            v2 = v2.sort_values(["speaker_id", "task"], kind="stable")
            for _, row in v2.head(cap).iterrows():
                title = transcript_label(
                    row["speaker_id"], row["task"], row["info_sex"], row["info_age"]
                )
                with st.expander(title):
                    body = transcript_highlight_html(str(row["text"]), highlight=hl, for_display=True)
                    st.markdown(
                        f'<div style="white-space: pre-wrap; line-height: 1.55;">{body}</div>',
                        unsafe_allow_html=True,
                    )

    with tab6:
        st.header("Download")
        st.write("Spreadsheet includes word counts, totals, each filler, and the three category columns.")
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="transcripts_with_fillers.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
