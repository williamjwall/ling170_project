"""Plain-language p-value blocks for the EDA Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from simple_stats import CompareResult, compare_many_groups, compare_two_groups, results_to_table


def pvalue_help_expander(*, key: str = "p_help") -> None:
    with st.expander("What does the p-value mean?", expanded=False):
        st.markdown(
            """
Each **dot** on the charts is **one recording**. We compare those scores with:

- **Mann-Whitney U** when there are **two** groups (e.g. male vs female)
- **Kruskal-Wallis** when there are **three or more** (e.g. neutral vs happy vs annoyed)

**How to read p:** if **p < 0.05**, the difference is big enough that it is **unlikely to be only random luck**
in this sample. That is worth mentioning in a paper or presentation. It is **not** proof the groups differ everywhere.

When we test **many filler words at once**, a few small p-values can show up by chance. Lean on the **overall**
and **category** rows first; use single-word rows as hints for what to look at next.
            """
        )


def render_compare_result(result: CompareResult, *, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    if result.p_value is None:
        st.warning(result.detail)
        return
    c1, c2, c3 = st.columns(3)
    sizes = " · ".join(f"{k} (n={v})" for k, v in result.n_by_group.items())
    c1.metric("p-value", format_p_display(result.p_value))
    c2.metric("Test", result.test)
    c3.metric("Who we compared", sizes[:72] + ("…" if len(sizes) > 72 else ""))
    if result.significant:
        st.success(result.verdict())
    else:
        st.info(result.verdict())
    st.caption(result.detail)


def format_p_display(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def render_two_group_block(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    label_a: str,
    label_b: str,
    code_a: str,
    code_b: str,
    metric_name: str,
    title: str | None = None,
) -> CompareResult:
    sub = df[df[group_col].isin([code_a, code_b])]
    result = compare_two_groups(
        sub.loc[sub[group_col] == code_a, value_col],
        sub.loc[sub[group_col] == code_b, value_col],
        label_a=label_a,
        label_b=label_b,
        metric=metric_name,
    )
    render_compare_result(result, title=title or f"Main check: {metric_name}")
    return result


def render_multi_group_block(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    group_order: list[str],
    group_labels: dict[str, str] | None = None,
    metric_name: str,
    title: str | None = None,
) -> CompareResult:
    gl = group_labels or {}
    groups = {}
    for code in group_order:
        g = df.loc[df[group_col] == code, value_col]
        if len(g):
            groups[gl.get(code, code)] = g
    result = compare_many_groups(groups, metric=metric_name)
    render_compare_result(result, title=title or f"Main check: {metric_name}")
    if result.pairwise is not None and len(result.pairwise):
        st.markdown("**Which pairs differ?** (Bonferroni adjusted p in `p_adj`)")
        st.dataframe(result.pairwise, use_container_width=True, hide_index=True)
    return result


def render_metrics_results_table(results: list[CompareResult], *, caption: str = "") -> None:
    if not results:
        return
    st.markdown("**Every measure on this tab**")
    if caption:
        st.caption(caption)
    st.dataframe(results_to_table(results), use_container_width=True, hide_index=True)
