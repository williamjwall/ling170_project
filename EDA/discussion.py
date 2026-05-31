"""
Simple discussion text from statistical results (for Streamlit).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from simple_stats import CompareResult, format_p

# Shown at the top of every discussion block (short glossary).
TERMS_BOX = """
**Quick terms**
- **Filler** — words like *um*, *uh*, *like*, *you know* that do not add much new meaning; they pause, soften, or hold the floor.
- **Per 100 words** — a fair way to compare a short clip and a long one (like “per minute” instead of raw counts).
- **p-value** — if **p is below 0.05**, the gap between groups is **probably not just luck** in this sample. It is **not** proof everyone everywhere differs.
- **Placeholders** — hesitation sounds (*um*, *uh*). **Californese** — discourse words (*like*, *so*). **Feedback** — listener checks (*you know*, *I mean*).
"""


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


def _sig_word_list(sig: Sequence[CompareResult], limit: int = 5) -> str:
    names = [_metric_short(r.comparison) for r in sig[:limit]]
    if not names:
        return ""
    text = ", ".join(f"*{n}*" for n in names)
    if len(sig) > limit:
        text += f", and {len(sig) - limit} more"
    return text


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

    lines = [TERMS_BOX.strip(), f"**Our question.** {hypothesis}"]

    if headline and headline.p_value is not None:
        if headline.significant:
            who = _higher_label(means)
            if who and len(means) == 2:
                other = [k for k in means if k != who][0]
                lines.append(
                    f"**Main answer.** Yes — for {topic}, the groups differ overall "
                    f"(p = {format_p(headline.p_value)}). "
                    f"**{who}** averaged **{means[who]:.1f}** fillers per 100 words vs "
                    f"**{means[other]:.1f}** for **{other}**."
                )
            else:
                lines.append(
                    f"**Main answer.** Yes — for {topic}, the groups differ overall "
                    f"(p = {format_p(headline.p_value)})."
                )
            lines.append(
                "**What that means.** Your idea that the two groups use fillers differently "
                "gets support here, at least for this data. You can say there is a real-looking gap, "
                "but keep it tied to **this task and this sample** — not all speech everywhere."
            )
        else:
            lines.append(
                f"**Main answer.** No clear overall gap for {topic} "
                f"(p = {format_p(headline.p_value)}, above 0.05). "
                "The two groups look pretty similar on **total** filler use."
            )
            lines.append(
                "**What that means.** If you predicted a big overall difference, this part of the data "
                "does **not** back that up. Check the table above — some **specific** words might still differ."
            )
    elif not all_results:
        lines.append("**Main answer.** Not enough recordings to compare.")
        return "\n\n".join(lines)

    if sig:
        words = _sig_word_list(sig)
        if words:
            lines.append(
                f"**Specific words that differed** (p below 0.05): {words}. "
                "These tell you **what kind** of filler is driving the gap (e.g. *uh* vs *like*)."
            )
    else:
        lines.append(
            "**Specific words.** Nothing else in the table hit p below 0.05. "
            "Charts that look a little different might still be random noise."
        )

    lines.append(f"**Other research.** {prior_work}")
    lines.append(f"**So what / next steps.** {next_steps}")

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

    lines = [TERMS_BOX.strip(), f"**Our question.** {hypothesis}"]

    if headline and headline.p_value is not None:
        mean_bits = ", ".join(f"{k} ({v:.1f}/100 words)" for k, v in means.items())
        if headline.significant:
            lines.append(
                f"**Main answer.** Yes — overall filler use **does** change across {topic} "
                f"(p = {format_p(headline.p_value)}). Averages: {mean_bits}."
            )
            lines.append(
                "**What that means.** The **task or mood** seems to affect **how much** people fill, "
                "not just which words they pick."
            )
        else:
            lines.append(
                f"**Main answer.** No — **total** filler use does **not** clearly differ across {topic} "
                f"(p = {format_p(headline.p_value)}). Averages were still: {mean_bits}."
            )
            lines.append(
                "**What that means.** You probably **cannot** claim people fill more when annoyed vs happy "
                "vs neutral overall. Focus instead on **which** fillers change (see table above)."
            )
    elif not all_results:
        lines.append("**Main answer.** Not enough groups to compare.")
        return "\n\n".join(lines)

    pattern_sig = [
        r for r in sig
        if headline is None or _metric_short(r.comparison) not in _metric_short(headline.comparison)
    ]
    if pattern_sig:
        words = _sig_word_list(pattern_sig)
        lines.append(
            f"**Specific words that differed:** {words}. "
            "Same total fillers, but **different words** — that is a **mix change**, not just more noise."
        )
    elif not (headline and headline.significant):
        lines.append(
            "**Specific words.** Nothing hit p below 0.05. Treat small chart differences as weak evidence."
        )

    lines.append(f"**Other research.** {prior_work}")
    lines.append(f"**So what / next steps.** {next_steps}")

    return "\n\n".join(lines)


PRIOR_GENDER = (
    "Some older studies say women use more fillers or more *like* / *you know*. "
    "Other work finds little difference, or the opposite on phone calls. Your results add one more real-data point."
)

PRIOR_EMOTION = (
    "People sometimes use more *um* when stressed or planning hard. Happy or angry stories might pull different "
    "words. Lab retellings are not the same as real life, but they are still useful."
)

PRIOR_SITUATION = (
    "Reading aloud, talking to a researcher, and calling a friend are different jobs. "
    "Fillers often go up when someone is thinking on the spot or talking back-and-forth."
)

NEXT_GENDER = (
    "In your paper: say **whether** you found a gap, **who** used more, and **which words** (*uh*, *like*, etc.). "
    "Mention limits: one recording per task, college speakers, regex counts. "
    "Next: pull example quotes from transcripts that show the pattern."
)

NEXT_EMOTION = (
    "In your paper: if totals are flat, argue a **mix** story (more *um* here, more *you know* there). "
    "Use transcript examples. Next: compare annoyed vs happy directly with adjusted tests if your class requires it."
)

NEXT_SITUATION = (
    "In your paper: do not lump phone calls with vowel holds or sentence reading. "
    "Compare **similar** tasks only. Next: filter to ages 18–24 or one task at a time for a cleaner story."
)
