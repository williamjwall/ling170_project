"""
Discussion-style prose from statistical results (for Streamlit, not a substitute for your paper).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from simple_stats import CompareResult, format_p


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


def _sig_results(results: Sequence[CompareResult]) -> list[CompareResult]:
    return [r for r in results if r.significant and r.p_value is not None]


def _metric_short(comparison: str) -> str:
    if " · " in comparison:
        return comparison.split(" · ")[-1].strip()
    return comparison


def discussion_two_groups(
    *,
    topic: str,
    hypothesis: str,
    headline: CompareResult | None,
    all_results: Sequence[CompareResult],
    means: dict[str, float],
    prior_work: str,
    next_steps: str,
) -> str:
    sig = _sig_results(all_results)
    if headline and headline.significant and headline not in sig:
        sig = [headline] + sig

    lines: list[str] = []

    lines.append(f"**What we were testing.** {hypothesis}")

    if headline and headline.p_value is not None:
        if headline.significant:
            who = _higher_label(means)
            direction = ""
            if who and len(means) == 2:
                other = [k for k in means if k != who][0]
                direction = (
                    f" On average, **{who}** had a higher rate "
                    f"({means[who]:.1f} vs {means[other]:.1f} per 100 words in this slice)."
                )
            lines.append(
                f"**Hypothesis and headline result.** The overall comparison for {topic} "
                f"was statistically notable (**p = {format_p(headline.p_value)}**).{direction} "
                "That supports a **partial or full** gender (or group) difference in filler use for this context, "
                "not a blanket claim about all speech everywhere."
            )
        else:
            lines.append(
                f"**Hypothesis and headline result.** For {topic}, **total filler rate did not differ "
                f"enough to clear our p < 0.05 bar** (p = {format_p(headline.p_value)}). "
                "If your thesis predicted a strong overall gap, this slice **does not fully support** that part. "
                "You may still report **pattern-level** differences below."
            )
    elif not all_results:
        lines.append("**Hypothesis and headline result.** Not enough data to test this comparison.")
        return "\n\n".join(lines)

    if sig:
        names = ", ".join(f"*{_metric_short(r.comparison)}*" for r in sig[:8])
        extra = f" (and {len(sig) - 8} more)" if len(sig) > 8 else ""
        lines.append(
            f"**Beyond the headline.** {len(sig)} measure(s) reached p < 0.05, including {names}{extra}. "
            "Interpret these as **which kinds** of fillers drive any gap, not as independent proof each word "
            "matters on its own (we ran many tests)."
        )
    else:
        lines.append(
            "**Beyond the headline.** No individual measure cleared p < 0.05 after the headline test. "
            "Visually different bars may still reflect **small effects or noise** in this sample."
        )

    lines.append(
        f"**How this relates to other work.** {prior_work} "
        "Your numbers sit in that debate: agreement where effects show up, tension where they do not."
    )

    lines.append(
        f"**Implications and next steps.** {next_steps} "
        "For the paper, keep **Results** as the tables and p-values above; use this block to argue **what it means** "
        "for your thesis and what you would test next."
    )

    return "\n\n".join(lines)


def discussion_many_groups(
    *,
    topic: str,
    hypothesis: str,
    headline: CompareResult | None,
    all_results: Sequence[CompareResult],
    means: dict[str, float],
    prior_work: str,
    next_steps: str,
) -> str:
    sig = _sig_results(all_results)
    if headline and headline.significant and headline not in sig:
        sig = [headline] + sig

    lines: list[str] = []
    lines.append(f"**What we were testing.** {hypothesis}")

    if headline and headline.p_value is not None:
        mean_bits = ", ".join(f"**{k}** {v:.1f}" for k, v in means.items())
        if headline.significant:
            lines.append(
                f"**Hypothesis and headline result.** Overall filler rate **did differ** across {topic} "
                f"(p = {format_p(headline.p_value)}). Group means in this slice: {mean_bits} (per 100 words). "
                "That supports a register or task effect on **how much** people fill, at least here."
            )
        else:
            lines.append(
                f"**Hypothesis and headline result.** **Total** filler use did **not** differ significantly "
                f"across {topic} (p = {format_p(headline.p_value)}). Means were still {mean_bits}. "
                "A broad claim that emotion or situation changes **overall** filler density is **weak** in this data. "
                "Your thesis may need to focus on **which** fillers shift, not raw totals."
            )
    elif not all_results:
        lines.append("**Hypothesis and headline result.** Not enough groups to test.")
        return "\n\n".join(lines)

    if sig:
        pattern_sig = [r for r in sig if headline is None or r.comparison != headline.comparison]
        if pattern_sig:
            names = ", ".join(f"*{_metric_short(r.comparison)}*" for r in pattern_sig[:6])
            lines.append(
                f"**Beyond the headline.** Even when totals are flat, specific measures differed (e.g. {names}). "
                "That pattern fits a **mix shift**: same overall filler budget, different words doing the work."
            )
    else:
        lines.append(
            "**Beyond the headline.** No measure cleared p < 0.05. Any chart differences may be descriptive only."
        )

    lines.append(f"**How this relates to other work.** {prior_work}")

    lines.append(f"**Implications and next steps.** {next_steps}")

    return "\n\n".join(lines)


# --- Preset copy for this project (edit in the app or here for your thesis) ---

PRIOR_GENDER = (
    "Past studies often report women using more fillers or more certain types (e.g. *like*, *you know*), "
    "but phone and lab speech can flip or blur that pattern. Lakoff-style stereotypes do not always survive "
    "controlled corpora."
)

PRIOR_EMOTION = (
    "Work on affect and disfluency sometimes predicts **more** hesitation under stress or **more** discourse "
    "markers when narrating. Lab retellings may not match everyday upset or excitement."
)

PRIOR_SITUATION = (
    "Read-aloud, monologue-to-an-experimenter, and real phone calls impose different cognitive loads and "
    "audience design. Filler rates often track **planning pressure** and **interactivity**, not just personality."
)

NEXT_GENDER = (
    "Check whether effects hold with hybrid tagging for *like* / *well* / *so*, compare **category shares**, "
    "and note whether differences are **planning** (*uh*) vs **stance** (*like*). Consider age and L1 as covariates."
)

NEXT_EMOTION = (
    "Pair these rates with **qualitative** transcript examples, test pairwise contrasts with correction, "
    "and ask whether annoyed stories show more **placeholders** vs happy stories show more **feedback** phrases."
)

NEXT_SITUATION = (
    "Separate **phone** from **RA monologue** in the write-up; vowel and read-aloud tasks are not comparable "
    "to dialogue. Future work could model task as a fixed effect with speaker random intercepts."
)
