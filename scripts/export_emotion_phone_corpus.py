#!/usr/bin/env python3
"""
Emotion (neutral, happy, annoyed) + phonecall, orthographic tier only.

**Default:** slim CSV — columns
  speaker_id, task, text, info_sex, info_age, info_l1_english, info_l1_other,
  info_l2_english_l1, info_l2_english_aoa
where discourse *like* / *well* / *so* are replaced with ``filtered_like``,
``filtered_well``, or ``filtered_so`` (see ``hybrid_filler_spacy.text_with_filtered_fillers``);
grammatical uses of those words stay as the original surface form.

**Wide export:** ``--wide`` keeps all input columns and original ``text``.

  python3 scripts/export_emotion_phone_corpus.py
  python3 scripts/export_emotion_phone_corpus.py --wide --output ucla_box_parsed/emotion_phone_orthographic_with_hybrid.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_filler_spacy import (
    HYBRID_OCCURRENCES_JSON_COL,
    occurrences_from_json,
    text_with_filtered_fillers,
)

EMOTION_AND_PHONE_TASKS = frozenset({"neutral", "happy", "annoyed", "phonecall"})

SIMPLIFIED_COLUMNS = [
    "speaker_id",
    "task",
    "text",
    "info_sex",
    "info_age",
    "info_l1_english",
    "info_l1_other",
    "info_l2_english_l1",
    "info_l2_english_aoa",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export emotion + phone orthographic rows (optionally slim + hybrid-cleaned text)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "ucla_box_parsed" / "ucla_text_state_parsed_with_hybrid.csv",
        help="Full corpus CSV (hybrid_occurrences_json recommended for simplified text).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "ucla_box_parsed" / "emotion_phone_simplified.csv",
        help="Output path.",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Keep all columns and raw text (no hybrid-based removal).",
    )
    p.add_argument(
        "--include-errors",
        action="store_true",
        help="Keep rows whose text starts with [ERROR: (default: drop).",
    )
    p.add_argument(
        "--include-empty-text",
        action="store_true",
        help="Keep rows with blank text after strip (default: drop).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    inp = args.input.expanduser()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].astype(str).replace({"nan": ""})

    if "textgrid_role" not in df.columns:
        raise SystemExit("CSV missing textgrid_role column.")
    mask = df["textgrid_role"].str.strip().str.lower().eq("orthographic")

    if "task" not in df.columns:
        raise SystemExit("CSV missing task column.")
    mask &= df["task"].str.strip().str.lower().isin(EMOTION_AND_PHONE_TASKS)

    txt = df["text"].astype(str) if "text" in df.columns else pd.Series([""] * len(df))
    if not args.include_empty_text:
        mask &= txt.str.strip().ne("")
    if not args.include_errors:
        mask &= ~txt.str.startswith("[ERROR:", na=False)

    out_df = df.loc[mask].copy()

    if args.wide:
        pass
    else:
        missing = [c for c in SIMPLIFIED_COLUMNS if c not in out_df.columns]
        if missing:
            raise SystemExit(f"Input CSV missing columns required for simplified export: {missing}")
        json_col = HYBRID_OCCURRENCES_JSON_COL if HYBRID_OCCURRENCES_JSON_COL in out_df.columns else None

        def clean_row(row: pd.Series) -> str:
            raw = str(row.get("text", ""))
            if not json_col:
                return raw
            occ = occurrences_from_json(str(row.get(json_col, "")))
            return text_with_filtered_fillers(raw, occ)

        work = out_df.copy()
        work["text"] = work.apply(clean_row, axis=1)
        out_df = work[SIMPLIFIED_COLUMNS]

    out: Path = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    mode = "wide" if args.wide else "simplified"
    print(
        f"Wrote {len(out_df)} {mode} rows ({len(df)} in input) to {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
