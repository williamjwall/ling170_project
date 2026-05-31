"""Plain-language p-value blocks for the EDA Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from simple_stats import (
    CompareResult,
    collect_significant,
    compare_many_groups,
    compare_two_groups,
    format_p,
    metric_label,
    results_to_table,
)


def pvalue_help_expander(*, key: str = "p_help") -> None:
    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown(
            """
- **Each row** = one person’s recording.
- **Per 100 words** = fillers ÷ word count × 100 (fair for short vs long clips).
- **p-value** — below **0.05** means the groups probably differ **more than random luck** would explain. It is **not** proof for all speakers everywhere.
- **Mann-Whitney U** — compares **two** groups (e.g. female vs male).
- **Kruskal-Wallis** — compares **three or more** groups (e.g. neutral vs happy vs annoyed).
- **Worth mentioning?** — our simple yes/no for p below 0.05.
- We test **many words** at once; treat the **overall / category** rows as the main story. Single-word rows are hints.
            """
        )


def _direction_two_group(means: dict[str, float]) -> str:
    if len(means) != 2:
        return ""
    hi = max(means, key=means.get)
    lo = min(means, key=means.get)
    return f" **{hi}** was higher ({means[hi]:.1f} vs {means[lo]:.1f} fillers per 100 words)."


def _means_snapshot(means: dict[str, float]) -> str:
    if len(means) < 2:
        return ""
    ordered = sorted(means.items(), key=lambda x: -x[1])
    bits = ", ".join(f"{k} {v:.1f}" for k, v in ordered)
    return f" Averages: {bits} (per 100 words)."


def _finding_plain_line(r: CompareResult, means: dict[str, float] | None) -> str:
    name = metric_label(r.comparison)
    groups = r.comparison.split(" · ")[0].strip() if " · " in r.comparison else ""
    p = format_p(r.p_value)
    line = f"**{name}**"
    if groups:
        line += f" ({groups})"
    line += f" — **p = {p}**"
    if means:
        if len(means) == 2:
            line += _direction_two_group(means)
        else:
            line += _means_snapshot(means)
    return line


def _style_results_table(df: pd.DataFrame):
    """Green highlight on rows worth mentioning."""

    def _row_style(row: pd.Series):
        if row.get("Worth mentioning?") == "Yes":
            return ["background-color: #c8f5d0; color: #0a3d18; font-weight: 600"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


def _style_pairwise_table(df: pd.DataFrame):
    def _row_style(row: pd.Series):
        if str(row.get("Significant?", "")).strip() == "Yes":
            return ["background-color: #fff3cd; color: #5c4a00; font-weight: 600"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


def build_metric_means(
    df: pd.DataFrame,
    group_col: str,
    columns: list[str],
    *,
    group_labels: dict[str, str],
    column_labels: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-measure average by group (for highlight bullets)."""
    out: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        label = column_labels.get(col, col)
        gm: dict[str, float] = {}
        for code, lab in group_labels.items():
            g = df.loc[df[group_col].astype(str).str.strip() == str(code), col]
            g = pd.to_numeric(g, errors="coerce").dropna()
            if len(g):
                gm[lab] = float(g.mean())
        if gm:
            out[label] = gm
    return out


def render_significant_findings(
    results: list[CompareResult],
    *,
    headline: CompareResult | None = None,
    means: dict[str, float] | None = None,
    pairwise: pd.DataFrame | None = None,
    metric_means: dict[str, dict[str, float]] | None = None,
) -> None:
    """
    Call out interesting results (p below 0.05) before the full table.

    metric_means: optional per-measure group means, keyed by metric_label(comparison).
    """
    sig = collect_significant(results, headline=headline)
    pw_sig: list[dict[str, object]] = []
    if pairwise is not None and len(pairwise):
        for _, row in pairwise.iterrows():
            if str(row.get("Significant?", "")).strip() == "Yes":
                pw_sig.append(row.to_dict())

    if not sig and not pw_sig:
        st.info(
            "No **highlighted** findings here — nothing in this slice hit **p below 0.05**. "
            "That can still be useful: it means groups look **similar** for these measures."
        )
        return

    st.markdown("### Interesting findings (p below 0.05)")
    st.caption("These are the rows worth talking about in your paper. Green rows in the table below match this list.")

    for r in sig:
        m = means
        if metric_means:
            m = metric_means.get(metric_label(r.comparison), means)
        st.success(_finding_plain_line(r, m))

    if pw_sig:
        st.markdown("**Pairs that differ** (after adjusting for many comparisons):")
        for row in pw_sig[:8]:
            a = row.get("Group A", "?")
            b = row.get("Group B", "?")
            padj = row.get("p_adj", "—")
            st.warning(f"**{a}** vs **{b}** — adjusted p = **{padj}**")


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
        st.markdown("**Which pairs differ?** (`p_adj` = adjusted p so we do not over-count many pairs)")
        st.dataframe(
            _style_pairwise_table(result.pairwise),
            use_container_width=True,
            hide_index=True,
        )
    return result


def render_metrics_results_table(
    results: list[CompareResult],
    *,
    caption: str = "",
    headline: CompareResult | None = None,
    means: dict[str, float] | None = None,
    pairwise: pd.DataFrame | None = None,
    metric_means: dict[str, dict[str, float]] | None = None,
) -> None:
    if not results and not headline:
        return
    render_significant_findings(
        results,
        headline=headline,
        means=means,
        pairwise=pairwise,
        metric_means=metric_means,
    )
    st.markdown("**Full table** (green rows = interesting)")
    if caption:
        st.caption(caption)
    tbl = results_to_table(results)
    if len(tbl):
        try:
            st.dataframe(_style_results_table(tbl), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.caption("—")


def render_discussion_section(title: str, body: str, *, key: str) -> None:
    """Discussion-style interpretation (not a duplicate of Results)."""
    with st.expander(title, expanded=False, key=key):
        st.markdown(body)
