#!/usr/bin/env python3
"""
Offline spaCy pass: add hybrid columns + JSON token spans to the corpus CSV.

Streamlit does **not** run spaCy; run this once locally after installing spaCy + a model, then point
the app at the output CSV (or replace the input file if you keep a backup).

  pip install 'spacy>=3.7,<4'
  python -m spacy download en_core_web_sm
  python scripts/precompute_hybrid_columns.py --input ucla_box_parsed/ucla_text_state_parsed.csv \\
      --output ucla_box_parsed/ucla_text_state_parsed_with_hybrid.csv

Requires network only for first-time model download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from hybrid_filler_spacy import (
    HYBRID_NUMERIC_COLS,
    HYBRID_OCCURRENCES_JSON_COL,
    analyze_transcript,
    occurrences_to_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Add hybrid / spaCy columns to corpus CSV (offline).")
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "ucla_box_parsed" / "ucla_text_state_parsed.csv",
        help="Input CSV (full file, all textgrid_role rows).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input_stem>_with_hybrid.csv next to input.",
    )
    ap.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy pipeline name (must be installed locally).",
    )
    args = ap.parse_args()
    inp: Path = args.input.expanduser()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out: Path = args.output.expanduser() if args.output else inp.with_name(f"{inp.stem}_with_hybrid{inp.suffix}")

    import spacy

    nlp = spacy.load(args.model)

    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    for c in HYBRID_NUMERIC_COLS:
        df[c] = 0
    df[HYBRID_OCCURRENCES_JSON_COL] = "[]"

    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        role = str(row.get("textgrid_role", "")).strip()
        txt = str(row.get("text", ""))
        if role != "orthographic" or not txt or txt.startswith("[ERROR"):
            continue
        r = analyze_transcript(txt, nlp)
        for c in HYBRID_NUMERIC_COLS:
            df.at[i, c] = int(r.get(c, 0))
        df.at[i, HYBRID_OCCURRENCES_JSON_COL] = occurrences_to_json(r.get("occurrences", []))

        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"Processed {i + 1}/{n} rows…", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
