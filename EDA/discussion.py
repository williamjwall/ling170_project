"""
Plain-English interpretation below the p-value tables (Streamlit).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from simple_stats import CompareResult, collect_significant, format_p, metric_label


def _means(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    labels: Mapping[str, str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for code, label in labels.items():
        g = df.loc[df[group_col].astype(str).str.strip() == str(code), value_col]
        g = pd.to_numeric(g, errors="coerce").dropna()
        if len(g):
            out[label] = float(g.mean())
    return out


def _higher_label(means: dict[str, float]) -> str | None:
    if len(means) < 2:
        return None
    return max(means, key=means.get)


def render_plain_english(
    *,
    hypothesis: str,
    headline: CompareResult | None,
    all_results: Sequence[CompareResult],
    means: dict[str, float],
    prior_work: str,
    next_steps: str,
    topic: str,
    two_group: bool,
    key: str,
) -> None:
    """Structured “What does this mean?” — ties to green highlights above."""
    sig = collect_significant(list(all_results), headline=headline)

    with st.expander("What does this mean? (plain English)", expanded=False, key=key):
        st.markdown(
            "Use this after the **Interesting findings** and **green table rows** above. "
            "Those show *what* differed; this section helps you say *what it means* for your thesis."
        )

        with st.expander("Words you might not know", expanded=False):
            st.markdown(
                """
| Term | Meaning |
|------|---------|
| **Filler** | Words like *um*, *uh*, *like*, *you know* — pauses, softeners, or holding the floor |
| **Per 100 words** | Fillers ÷ word count × 100 (fair for short vs long recordings) |
| **p-value** | Below **0.05** → groups probably differ more than random luck in *this* sample (not proof everywhere) |
| **Placeholders** | Hesitation: *um*, *uh*, *hmm* |
| **Californese** | Discourse style: *like*, *so*, *basically* |
| **Feedback** | Checking in with the listener: *you know*, *I mean*, *well* |
                """
            )

        st.markdown("#### 1. Your question")
        st.write(hypothesis)

        st.markdown("#### 2. Bottom line")
        _render_bottom_line(
            headline=headline,
            sig=sig,
            means=means,
            topic=topic,
            two_group=two_group,
        )

        st.markdown("#### 3. Tie-in to the highlights above")
        if sig:
            st.markdown("The **green boxes** match these measures:")
            for r in sig[:10]:
                st.markdown(f"- **{metric_label(r.comparison)}** — p = {format_p(r.p_value)}")
            if len(sig) > 10:
                st.caption(f"…and {len(sig) - 10} more in the table.")
        else:
            st.info(
                "Nothing was green above (no p below 0.05). "
                "In your Discussion you can say groups looked **similar** on these tests for this slice."
            )

        st.markdown("#### 4. For your paper’s Discussion (not Results)")
        _render_paper_guidance(
            headline=headline,
            sig=sig,
            means=means,
            two_group=two_group,
        )

        st.markdown("#### 5. Other studies")
        st.write(prior_work)

        st.markdown("#### 6. Limits and what to do next")
        st.markdown(
            """
- One row = **one recording**, not one person’s whole life.
- Counts use **pattern matching** on transcripts (*like* can include grammatical uses unless hybrid tagging is on).
- Many tests at once → treat **overall / category** greens as main; single words are **follow-ups**.
            """
        )
        st.write(next_steps)


def _render_bottom_line(
    *,
    headline: CompareResult | None,
    sig: list[CompareResult],
    means: dict[str, float],
    topic: str,
    two_group: bool,
) -> None:
    if headline and headline.p_value is not None:
        if headline.significant:
            if two_group and len(means) == 2:
                who = _higher_label(means)
                other = [k for k in means if k != who][0] if who else None
                extra = (
                    f" **{who}** averaged more overall ({means[who]:.1f} vs {means[other]:.1f} per 100 words)."
                    if who and other
                    else ""
                )
                st.success(
                    f"**Yes** — for {topic}, the two groups differ on total fillers "
                    f"(p = {format_p(headline.p_value)}).{extra}"
                )
            elif not two_group and means:
                bits = ", ".join(f"{k} {v:.1f}" for k, v in sorted(means.items(), key=lambda x: -x[1]))
                st.success(
                    f"**Yes** — overall filler use differs across {topic} "
                    f"(p = {format_p(headline.p_value)}). Averages: {bits} per 100 words."
                )
            else:
                st.success(
                    f"**Yes** — groups differ on total fillers for {topic} "
                    f"(p = {format_p(headline.p_value)})."
                )
        else:
            if means:
                bits = ", ".join(f"{k} {v:.1f}" for k, v in means.items())
                st.info(
                    f"**No clear difference** on **total** fillers for {topic} "
                    f"(p = {format_p(headline.p_value)}). Averages were still: {bits} per 100 words."
                )
            else:
                st.info(
                    f"**No clear difference** on total fillers for {topic} "
                    f"(p = {format_p(headline.p_value)})."
                )
    elif not sig:
        st.warning("Not enough data to summarize.")
        return

    if sig and headline and not headline.significant:
        st.markdown(
            f"Even though **totals** look similar, **{len(sig)}** specific measure(s) still differed "
            "(see green highlights). That is a **mix** story: same overall filling, different words."
        )
    elif sig and (not headline or headline.significant):
        n_extra = len(sig) - (1 if headline and headline.significant else 0)
        if n_extra > 0:
            st.markdown(
                f"Also check the greens for **specific words or categories** "
                f"({n_extra} more beyond the headline)."
            )


def _render_paper_guidance(
    *,
    headline: CompareResult | None,
    sig: list[CompareResult],
    means: dict[str, float],
    two_group: bool,
) -> None:
    bullets: list[str] = []

    if headline and headline.significant:
        bullets.append("State that your **main comparison** showed a statistically notable gap (p below 0.05).")
        if two_group and len(means) == 2:
            who = _higher_label(means)
            if who:
                bullets.append(f"Say **{who}** had the higher average rate in this sample, with the actual numbers from your table.")
    elif headline and headline.p_value is not None:
        bullets.append(
            "Say your **overall** comparison did **not** reach significance — do not overclaim a big total difference."
        )

    if sig:
        words = ", ".join(f"*{metric_label(r.comparison)}*" for r in sig[:6])
        bullets.append(f"Mention **which fillers** drove the pattern (e.g. {words}).")
        bullets.append("Give **one short transcript quote** that shows the pattern in real speech.")
    else:
        bullets.append("Explain that groups looked **similar** on these measures in the UCLA slice you used.")

    bullets.append("Separate **Results** (numbers, p-values) from **Discussion** (interpretation, limits, links to other studies).")

    for b in bullets:
        st.markdown(f"- {b}")


PRIOR_GENDER = (
    "Some studies report women using more fillers or more *like* / *you know*; others find little gap or the opposite on phone speech. "
    "Compare your green highlights to that literature — agreement where you see the same pattern, tension where you do not."
)

PRIOR_EMOTION = (
    "Stress and planning can increase *um* and *uh*; narrative mood might change *which* words appear even when totals stay flat. "
    "Lab retellings are not identical to real life, but they are still valid evidence if you state the limit."
)

PRIOR_SITUATION = (
    "Reading aloud, monologue to a researcher, and live phone calls load the speaker differently. "
    "Fillers often rise when someone is thinking on the spot or talking with another person."
)

NEXT_GENDER = (
    "Pull 1–2 transcript quotes where the gap shows up. Note age 18–24 and phone-only scope. "
    "If *like* was not green, say gender differences here were **not** mainly about *like*."
)

NEXT_EMOTION = (
    "If only specific words were green, frame emotion as changing **filler type**, not **amount**. "
    "Compare annoyed vs happy in prose using your pairwise table if any row was yellow."
)

NEXT_SITUATION = (
    "Do not compare vowel holds or sentence reading to phone calls in one claim. "
    "Re-run filters for a single task if you want a simpler Discussion paragraph."
)
