"""
Nonparametric tests on per-recording filler scores (one row = one transcript).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CompareResult:
    test: str
    comparison: str
    p_value: float | None
    n_by_group: dict[str, int]
    detail: str
    pairwise: pd.DataFrame | None = None

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value is None:
            return self.detail
        p = self.p_value
        if p < alpha:
            return (
                f"p = {format_p(p)} — groups probably differ (not just luck). "
                f"We flag anything below p = {alpha:g}."
            )
        return (
            f"p = {format_p(p)} — groups look similar; could be random noise "
            f"(above p = {alpha:g})."
        )


def format_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _clean_values(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return x[np.isfinite(x)]


def compare_two_groups(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    *,
    label_a: str = "Group A",
    label_b: str = "Group B",
    metric: str = "this measure",
) -> CompareResult:
    xa = _clean_values(pd.Series(a))
    xb = _clean_values(pd.Series(b))
    n = {label_a: len(xa), label_b: len(xb)}
    comparison = f"{label_a} vs {label_b} · {metric}"
    if len(xa) < 2 or len(xb) < 2:
        return CompareResult(
            "Mann-Whitney U",
            comparison,
            None,
            n,
            "Need at least 2 recordings in each group.",
        )
    try:
        stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    except ValueError as e:
        return CompareResult("Mann-Whitney U", comparison, None, n, str(e))
    note = "Small sample — treat as a rough hint. " if min(len(xa), len(xb)) < 3 else ""
    detail = f"{note}Mann-Whitney U (two-sided). {CompareResult('', '', float(p), n, '').verdict()}"
    return CompareResult("Mann-Whitney U", comparison, float(p), n, detail)


def compare_many_groups(
    groups: Mapping[str, pd.Series | np.ndarray],
    *,
    metric: str = "this measure",
    pairwise: bool = True,
) -> CompareResult:
    clean = {k: _clean_values(pd.Series(v)) for k, v in groups.items()}
    labels = [k for k in clean if len(clean[k]) > 0]
    n = {k: len(clean[k]) for k in labels}
    comparison = f"{' · '.join(labels)} · {metric}"
    if len(labels) < 2:
        return CompareResult("Kruskal-Wallis", comparison, None, n, "Need at least two groups.")
    try:
        stat, p = stats.kruskal(*[clean[k] for k in labels])
    except ValueError as e:
        return CompareResult("Kruskal-Wallis", comparison, None, n, str(e))
    pw = pairwise_mann_whitney(clean) if pairwise and len(labels) >= 2 else None
    detail = (
        f"Kruskal-Wallis, H ≈ {stat:.2f}. "
        f"{CompareResult('', '', float(p), n, '').verdict()}"
    )
    if pw is not None and len(pw):
        detail += " Pairwise table below."
    return CompareResult("Kruskal-Wallis", comparison, float(p), n, detail, pairwise=pw)


def pairwise_mann_whitney(groups: Mapping[str, np.ndarray]) -> pd.DataFrame:
    labels = list(groups.keys())
    rows: list[dict[str, object]] = []
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            pairs.append((a, b))
    m = max(len(pairs), 1)
    for a, b in pairs:
        xa, xb = groups[a], groups[b]
        try:
            _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        except ValueError:
            p = float("nan")
        padj = min(1.0, float(p) * m) if pd.notna(p) else p
        rows.append(
            {
                "Group A": a,
                "Group B": b,
                "n A": len(xa),
                "n B": len(xb),
                "p": p,
                "p_adj": padj,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sig = ["Yes" if pd.notna(r["p_adj"]) and float(r["p_adj"]) < 0.05 else "No" for r in rows]
    out["Significant?"] = sig
    out["p"] = out["p"].map(lambda x: format_p(x) if pd.notna(x) else "n/a")
    out["p_adj"] = out["p_adj"].map(lambda x: format_p(x) if pd.notna(x) else "n/a")
    return out


def compare_metrics_in_dataframe(
    df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str],
    *,
    group_order: Iterable[str] | None = None,
    metric_labels: Mapping[str, str] | None = None,
    group_labels: Mapping[str, str] | None = None,
) -> list[CompareResult]:
    sub = df.copy()
    sub[group_col] = sub[group_col].astype(str).str.strip()
    if group_order is not None:
        groups = [g for g in group_order if g in set(sub[group_col])]
    else:
        groups = sorted(sub[group_col].dropna().unique(), key=str)
    gl = group_labels or {}
    results: list[CompareResult] = []
    for col in metrics:
        if col not in sub.columns:
            continue
        label = (metric_labels or {}).get(col, col)
        by_group = {
            gl.get(g, g): sub.loc[sub[group_col] == g, col]
            for g in groups
            if (sub[group_col] == g).any()
        }
        if len(by_group) == 2:
            keys = list(by_group.keys())
            results.append(
                compare_two_groups(
                    by_group[keys[0]],
                    by_group[keys[1]],
                    label_a=str(keys[0]),
                    label_b=str(keys[1]),
                    metric=label,
                )
            )
        elif len(by_group) >= 3:
            results.append(compare_many_groups(by_group, metric=label))
    return results


def metric_label(comparison: str) -> str:
    """Short name for a measure (last segment after ·)."""
    if " · " in comparison:
        return comparison.split(" · ")[-1].strip()
    return comparison.strip()


def collect_significant(
    results: Sequence[CompareResult],
    headline: CompareResult | None = None,
) -> list[CompareResult]:
    """Unique significant results, strongest (lowest p) first."""
    seen: set[str] = set()
    out: list[CompareResult] = []
    for r in ([headline] if headline else []) + list(results):
        if not r or not r.significant or r.p_value is None:
            continue
        if r.comparison in seen:
            continue
        seen.add(r.comparison)
        out.append(r)
    out.sort(key=lambda x: x.p_value or 1.0)
    return out


def results_to_table(results: list[CompareResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        ns = ", ".join(f"{k} n={v}" for k, v in r.n_by_group.items())
        rows.append(
            {
                "What we compared": r.comparison,
                "Test": r.test,
                "p": format_p(r.p_value),
                "Worth mentioning?": "Yes" if r.significant else ("No" if r.p_value is not None else "—"),
                "Sample sizes": ns,
            }
        )
    return pd.DataFrame(rows)
